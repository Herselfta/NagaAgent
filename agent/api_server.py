# agent/api_server.py
# API服务器 - 整合预处理功能
import asyncio
import json
import logging
import re  # 添加re模块导入
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from aiohttp import web
import os

from .preprocessor import get_preprocessor, preprocess_messages
from .plugin_manager import get_plugin_manager, load_plugins

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentAPIServer")

class AgentAPIServer:
    """代理API服务器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.api_key = self.config.get('api_key') or os.getenv('API_Key')
        self.api_url = self.config.get('api_url') or os.getenv('API_URL')
        self.server_key = self.config.get('server_key') or os.getenv('Key')
        self.debug_mode = self.config.get('debug_mode', False) or os.getenv("DEBUG", "False").lower() == "true"
        
        # 初始化组件
        self.preprocessor = get_preprocessor()
        self.plugin_manager = get_plugin_manager()
        
        # 创建web应用
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """设置路由"""
        # 模型列表代理
        self.app.router.add_get('/v1/models', self.handle_models)
        
        # 主对话API
        self.app.router.add_post('/v1/chat/completions', self.handle_chat_completions)
        
        # 插件回调
        self.app.router.add_post('/plugin-callback/{plugin_name}/{task_id}', self.handle_plugin_callback)
    
    async def handle_models(self, request: web.Request) -> web.Response:
        """处理模型列表请求"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/v1/models",
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'User-Agent': request.headers.get('user-agent', ''),
                        'Accept': request.headers.get('accept', 'application/json')
                    }
                ) as response:
                    # 转发状态码和头部
                    headers = dict(response.headers)
                    # 移除hop-by-hop头部
                    for header in ['content-encoding', 'transfer-encoding', 'connection', 'content-length', 'keep-alive']:
                        headers.pop(header, None)
                    
                    return web.Response(
                        body=await response.read(),
                        status=response.status,
                        headers=headers
                    )
        except Exception as e:
            logger.error(f"转发模型列表请求失败: {e}")
            return web.json_response(
                {'error': 'Internal Server Error', 'details': str(e)},
                status=500
            )
    
    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """处理对话完成请求"""
        try:
            # 读取原始请求
            original_body = await request.json()
            
            if self.debug_mode:
                logger.info(f"收到对话请求: {json.dumps(original_body, ensure_ascii=False)[:200]}...")
            
            # 1. 图片处理预处理
            should_process_images = True
            if original_body.get('messages'):
                for msg in original_body['messages']:
                    if msg.get('role') in ['user', 'system']:
                        content = msg.get('content', '')
                        if isinstance(content, str) and '{{ShowBase64}}' in content:
                            should_process_images = False
                            msg['content'] = content.replace('{{ShowBase64}}', '')
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get('type') == 'text':
                                    if '{{ShowBase64}}' in part.get('text', ''):
                                        should_process_images = False
                                        part['text'] = part['text'].replace('{{ShowBase64}}', '')
            
            if should_process_images:
                if self.debug_mode:
                    logger.info("启用图片处理")
                try:
                    original_body['messages'] = await self.plugin_manager.execute_message_preprocessor(
                        "ImageProcessor", original_body['messages']
                    )
                except Exception as e:
                    logger.error(f"图片处理失败: {e}")
            
            # 2. 变量替换和系统提示词注入
            if original_body.get('messages'):
                original_body['messages'] = await preprocess_messages(
                    original_body['messages'], original_body.get('model')
                )
            
            if self.debug_mode:
                logger.info(f"预处理后的请求: {json.dumps(original_body, ensure_ascii=False)[:200]}...")
            
            # 3. 调用LLM API
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'User-Agent': request.headers.get('user-agent', ''),
                    'Accept': 'text/event-stream' if original_body.get('stream') else 'application/json'
                }
                
                async with session.post(
                    f"{self.api_url}/v1/chat/completions",
                    headers=headers,
                    json=original_body
                ) as response:
                    
                    # 检查是否为流式响应
                    is_streaming = original_body.get('stream') and 'text/event-stream' in response.headers.get('content-type', '')
                    
                    if is_streaming:
                        return await self._handle_streaming_response(response, original_body, request)
                    else:
                        return await self._handle_non_streaming_response(response, original_body)
        
        except Exception as e:
            logger.error(f"处理对话请求失败: {e}")
            return web.json_response(
                {'error': 'Internal Server Error', 'details': str(e)},
                status=500
            )
    
    async def _handle_streaming_response(self, response: aiohttp.ClientResponse, original_body: Dict, request: web.Request) -> web.StreamResponse:
        """处理流式响应"""
        stream_response = web.StreamResponse(
            status=response.status,
            headers=dict(response.headers)
        )
        
        # 设置流式响应头部
        stream_response.headers['Content-Type'] = 'text/event-stream'
        stream_response.headers['Cache-Control'] = 'no-cache'
        stream_response.headers['Connection'] = 'keep-alive'
        
        await stream_response.prepare(request)
        
        # 处理工具调用循环
        recursion_depth = 0
        max_recursion = int(os.getenv('MaxhandoffLoopStream', '5'))
        current_messages = original_body.get('messages', [])
        current_ai_content = ''
        
        async for chunk in response.content:
            chunk_str = chunk.decode('utf-8')
            
            # 检查是否包含工具调用
            if '<<<[TOOL_REQUEST]>>>' in chunk_str:
                # 收集完整的AI响应
                current_ai_content += chunk_str
                
                # 解析工具调用
                tool_calls = self._parse_tool_calls(current_ai_content)
                
                if tool_calls and recursion_depth < max_recursion:
                    # 执行工具调用
                    tool_results = await self._execute_tool_calls(tool_calls)
                    
                    # 添加AI响应和工具结果到消息历史
                    current_messages.append({'role': 'assistant', 'content': current_ai_content})
                    current_messages.append({'role': 'user', 'content': tool_results})
                    
                    # 继续调用LLM
                    recursion_depth += 1
                    current_ai_content = ''
                    
                    # 发送换行符给客户端
                    await stream_response.write(b'\n')
                    
                    # 继续处理下一轮响应
                    continue
            
            # 转发原始chunk
            await stream_response.write(chunk)
        
        await stream_response.write_eof()
        return stream_response
    
    async def _handle_non_streaming_response(self, response: aiohttp.ClientResponse, original_body: Dict) -> web.Response:
        """处理非流式响应"""
        response_data = await response.read()
        
        try:
            response_json = json.loads(response_data.decode('utf-8'))
            
            # 处理工具调用循环（非流式）
            recursion_depth = 0
            max_recursion = int(os.getenv('MaxhandoffLoopNonStream', '5'))
            current_messages = original_body.get('messages', [])
            current_ai_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            while recursion_depth < max_recursion:
                # 解析工具调用
                tool_calls = self._parse_tool_calls(current_ai_content)
                
                if not tool_calls:
                    break
                
                # 执行工具调用
                tool_results = await self._execute_tool_calls(tool_calls)
                
                # 添加AI响应和工具结果到消息历史
                current_messages.append({'role': 'assistant', 'content': current_ai_content})
                current_messages.append({'role': 'user', 'content': tool_results})
                
                # 继续调用LLM
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {self.api_key}',
                            'Accept': 'application/json'
                        },
                        json={**original_body, 'messages': current_messages, 'stream': False}
                    ) as next_response:
                        next_response_data = await next_response.read()
                        next_response_json = json.loads(next_response_data.decode('utf-8'))
                        current_ai_content = next_response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                recursion_depth += 1
            
            # 更新最终响应
            response_json['choices'][0]['message']['content'] = current_ai_content
            
            return web.json_response(response_json, status=response.status)
        
        except Exception as e:
            logger.error(f"处理非流式响应失败: {e}")
            return web.Response(body=response_data, status=response.status)
    
    def _parse_tool_calls(self, content: str) -> List[Dict]:
        """解析工具调用，支持MCP和Agent两种类型"""
        tool_calls = []
        
        tool_request_start = "<<<[TOOL_REQUEST]>>>"
        tool_request_end = "<<<[END_TOOL_REQUEST]>>>"
        
        start_index = 0
        while True:
            start_pos = content.find(tool_request_start, start_index)
            if start_pos == -1:
                break
            
            end_pos = content.find(tool_request_end, start_pos)
            if end_pos == -1:
                start_index = start_pos + len(tool_request_start)
                continue
            
            # 提取工具调用内容
            tool_content = content[start_pos + len(tool_request_start):end_pos].strip()
            
            # 先解析所有参数
            tool_args = {}
            param_pattern = r'(\w+)\s*:\s*「始」([\s\S]*?)「末」'
            for match in re.finditer(param_pattern, tool_content):
                key = match.group(1)
                value = match.group(2).strip()
                    tool_args[key] = value
            
            # 判断调用类型
            agent_type = tool_args.get('agentType', 'mcp').lower()
            
            if agent_type == 'agent':
                # Agent类型调用格式
                agent_name = tool_args.get('agent_name')
                prompt = tool_args.get('prompt')
                if agent_name and prompt:
                    tool_calls.append({
                        'name': 'agent_call',
                        'args': {
                            'agentType': 'agent',
                            'agent_name': agent_name,
                            'prompt': prompt
                        }
                    })
            else:
                # MCP类型调用格式（包括默认mcp和旧格式）
                tool_name = tool_args.get('tool_name')
            if tool_name:
                    # 新格式：有service_name
                    if 'service_name' in tool_args:
                        tool_calls.append({
                            'name': tool_name,
                            'args': tool_args
                        })
                    else:
                        # 旧格式：tool_name作为服务名
                        service_name = tool_name
                        tool_args['service_name'] = service_name
                        tool_args['agentType'] = 'mcp'
                tool_calls.append({
                    'name': tool_name,
                    'args': tool_args
                })
            
            start_index = end_pos + len(tool_request_end)
        
        return tool_calls
    
    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> str:
        """执行工具调用"""
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call['name']
                args = tool_call['args']
                agent_type = args.get('agentType', 'mcp').lower()
                
                # 根据agentType分流处理
                if agent_type == 'agent':
                    # Agent类型：交给AgentManager处理
                    try:
                        from mcpserver.agent_manager import get_agent_manager
                        agent_manager = get_agent_manager()
                        
                        agent_name = args.get('agent_name')
                        prompt = args.get('prompt')
                        
                        if not agent_name or not prompt:
                            result = "Agent调用失败: 缺少agent_name或prompt参数"
                        else:
                            # 直接调用Agent
                            result = await agent_manager.call_agent(agent_name, prompt)
                            if result.get("status") == "success":
                                result = result.get("result", "")
                            else:
                                result = f"Agent调用失败: {result.get('error', '未知错误')}"
                                
                    except Exception as e:
                        result = f"Agent调用失败: {str(e)}"
                        
                else:
                    # MCP类型：暂时返回模拟结果
                    result = f"来自工具 \"{tool_name}\" 的结果:\n[工具执行结果]"
                
                results.append(result)
            except Exception as e:
                error_result = f"执行工具 {tool_call['name']} 时发生错误：{str(e)}"
                results.append(error_result)
        
        return "\n\n---\n\n".join(results)
    
    async def handle_plugin_callback(self, request: web.Request) -> web.Response:
        """处理插件回调"""
        plugin_name = request.match_info['plugin_name']
        task_id = request.match_info['task_id']
        
        try:
            callback_data = await request.json()
            
            if self.debug_mode:
                logger.info(f"收到插件回调: {plugin_name}, task_id: {task_id}")
            
            # 这里可以处理回调数据，例如保存到文件、推送到WebSocket等
            
            return web.json_response({
                'status': 'success',
                'message': 'Callback received and processed'
            })
        
        except Exception as e:
            logger.error(f"处理插件回调失败: {e}")
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def start(self, host: str = '127.0.0.1', port: int = 8000):
        """启动服务器"""
        # 加载插件
        await load_plugins()
        
        logger.info(f"启动代理API服务器: {host}:{port}")
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"代理API服务器已启动: http://{host}:{port}")
        
        # 保持运行
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭服务器...")
            await runner.cleanup()

# 便捷函数
async def start_server(host: str = '127.0.0.1', port: int = 8000, config: Dict = None):
    """启动API服务器"""
    server = AgentAPIServer(config)
    await server.start(host, port)

if __name__ == "__main__":
    # 从环境变量获取配置
    config = {
        'api_key': os.getenv('API_Key'),
        'api_url': os.getenv('API_URL'),
        'server_key': os.getenv('Key'),
        'debug_mode': os.getenv("DEBUG", "False").lower() == "true"
    }
    
    # 启动服务器
    asyncio.run(start_server(config=config)) 