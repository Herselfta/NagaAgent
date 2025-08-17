# 标准库导入
import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime
from typing import List, Dict

# 第三方库导入
from openai import AsyncOpenAI
import google.generativeai as genai

# 本地模块导入
from apiserver.tool_call_utils import parse_tool_calls, execute_tool_calls, tool_call_loop
from config import config
from mcpserver.mcp_manager import get_mcp_manager
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
# from thinking import TreeThinkingEngine
from thinking.config import COMPLEX_KEYWORDS

# 配置日志系统
def setup_logging():
    """统一配置日志系统"""
    log_level = getattr(logging, config.system.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    # 设置第三方库日志级别
    for logger_name in ["httpcore.connection", "httpcore.http11", "httpx", "openai._base_client", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger("NagaConversation")

# 全局状态管理
class SystemState:
    """系统状态管理器"""
    _tree_thinking_initialized = False
    _mcp_services_initialized = False
    _voice_enabled_logged = False
    _memory_initialized = False

# GRAG记忆系统导入
def init_memory_manager():
    """初始化GRAG记忆系统"""
    if not config.grag.enabled:
        return None
    
    try:
        from summer_memory.memory_manager import memory_manager
        print("[GRAG] ✅ 夏园记忆系统初始化成功")
        return memory_manager
    except Exception as e:
        logger.error(f"夏园记忆系统加载失败: {e}")
        return None

memory_manager = init_memory_manager()

# 语音系统导入
def init_voice_system():
    """根据配置初始化语音系统"""
    if not config.system.voice_enabled:
        return None
    
    tts_choice = config.tts.TTS_CHOICE.upper()
    logger.info(f"正在初始化语音系统，选择: {tts_choice}")

    if tts_choice == "GPT-SOVITS":
        try:
            from voice.gpt_sovits_tts import GPTSoVITS_TTS
            tts_system = GPTSoVITS_TTS()
            if tts_system.is_enabled:
                logger.info("[TTS] ✅ GPT-SoVITS TTS 初始化成功")
                return tts_system
            else:
                logger.warning("[TTS] ⚠️ GPT-SoVITS TTS 已配置但未启用 (is_enabled: false)")
                return None
        except ImportError:
            logger.error("无法导入 GPTSoVITS_TTS，请确保 voice/gpt_sovits_tts.py 文件存在。")
            return None
        except Exception as e:
            logger.error(f"GPT-SoVITS TTS 初始化失败: {e}")
            return None
    elif tts_choice == "AZURE":
        # 此处可以添加 Azure TTS 的初始化逻辑
        logger.warning("Azure TTS 提供商尚未完全实现。")
        return None
    elif tts_choice == "DISABLE":
        logger.info("TTS 已禁用。")
        return None
    else:
        logger.warning(f"未知的 TTS 选择: {tts_choice}")
        return None

voice_system = init_voice_system()

# 工具函数
def now():
    """获取当前时间戳"""
    return time.strftime('%H:%M:%S:') + str(int(time.time() * 1000) % 10000)

_builtin_print = print
def print(*a, **k):
    """自定义打印函数"""
    return sys.stderr.write('[print] ' + (' '.join(map(str, a))) + '\n')

class NagaConversation: # 对话主类
    def __init__(self):
        self.mcp = get_mcp_manager()
        self.messages = []
        self.dev_mode = False
        self.api_client = None
        self.provider = config.api.provider

        # API client will be initialized on-demand in the process method
        # self._initialize_api_client()

        # 初始化MCP服务系统
        self._init_mcp_services()
        
        # 初始化GRAG记忆系统（只在首次初始化时显示日志）
        self.memory_manager = memory_manager
        if self.memory_manager and not SystemState._memory_initialized:
            logger.info("夏园记忆系统已初始化")
            SystemState._memory_initialized = True
        
        # 初始化语音处理系统
        self.voice = voice_system
        if self.voice and not SystemState._voice_enabled_logged:
            logger.info(f"语音功能已启用，使用 {config.tts.TTS_CHOICE} 提供商。")
            SystemState._voice_enabled_logged = True
        
        # Do not get loop in constructor, it binds to the wrong thread
        # self.loop = asyncio.get_event_loop()

    def _initialize_api_client(self):
        """根据配置初始化API客户端"""
        api_key = config.api.get_api_key()
        base_url = config.api.get_base_url()

        if not api_key or "placeholder" in api_key or "your-gemini-api-key-here" in api_key:
            logger.warning(f"未配置 {self.provider} 的API密钥，API功能可能受限。")
            self.api_client = None
            return

        logger.info(f"正在初始化API客户端，提供商: {self.provider}")

        if self.provider == "gemini":
            try:
                proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
                if proxy_url:
                    genai.configure(api_key=api_key, transport='rest', client_options={"api_endpoint": "generativelanguage.googleapis.com", "proxy": proxy_url})
                    logger.info(f"Google Gemini 使用代理: {proxy_url}")
                else:
                    genai.configure(api_key=api_key)
                
                self.api_client = genai.GenerativeModel(config.api.model)
                logger.info(f"Google Gemini 模型初始化成功: {config.api.model}")
            except Exception as e:
                logger.error(f"Google Gemini 模型初始化失败: {e}", exc_info=True)
                self.api_client = None
        else: # openai, deepseek等兼容OpenAI API的提供商
            if not base_url:
                logger.error(f"{self.provider} 的 base_url 未配置。")
                self.api_client = None
                return
            try:
                self.api_client = AsyncOpenAI(
                    api_key=api_key, 
                    base_url=base_url.rstrip('/') + '/'
                )
                logger.info(f"{self.provider} 客户端初始化成功，模型: {config.api.model}")
            except Exception as e:
                logger.error(f"{self.provider} 客户端初始化失败: {e}", exc_info=True)
                self.api_client = None


    def _init_mcp_services(self):
        """初始化MCP服务系统（只在首次初始化时输出日志，后续静默）"""
        if SystemState._mcp_services_initialized:
            # 静默跳过，不输出任何日志
            return
        try:
            # 自动注册所有MCP服务和handoff
            self.mcp.auto_register_services()
            logger.info("MCP服务系统初始化完成")
            SystemState._mcp_services_initialized = True
            
            # 异步启动NagaPortal自动登录
            self._start_naga_portal_auto_login()
        except Exception as e:
            logger.error(f"MCP服务系统初始化失败: {e}")
    
    def _start_naga_portal_auto_login(self):
        """启动NagaPortal自动登录（异步）"""
        try:
            # 检查是否配置了NagaPortal
            if not config.naga_portal.username or not config.naga_portal.password:
                return  # 静默跳过，不输出日志
            
            # 在新线程中异步执行登录
            def run_auto_login():
                try:
                    import sys
                    import os
                    # 添加项目根目录到Python路径
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    sys.path.insert(0, project_root)
                    
                    from mcpserver.agent_naga_portal.portal_login_manager import auto_login_naga_portal
                    
                    # 创建新的事件循环
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        # 执行自动登录
                        result = loop.run_until_complete(auto_login_naga_portal())
                        
                        if result['success']:
                            # 登录成功，显示状态
                            print("✅ NagaPortal自动登录成功")
                            self._show_naga_portal_status()
                        else:
                            # 登录失败，显示错误
                            error_msg = result.get('message', '未知错误')
                            print(f"❌ NagaPortal自动登录失败: {error_msg}")
                            self._show_naga_portal_status()
                    finally:
                        loop.close()
                        
                except Exception as e:
                    # 登录异常，显示错误
                    print(f"❌ NagaPortal自动登录异常: {e}")
                    self._show_naga_portal_status()
            
            # 启动后台线程
            import threading
            login_thread = threading.Thread(target=run_auto_login, daemon=True)
            login_thread.start()
            
        except Exception as e:
            # 启动异常，显示错误
            print(f"❌ NagaPortal自动登录启动失败: {e}")
            self._show_naga_portal_status()

    def _show_naga_portal_status(self):
        """显示NagaPortal状态（登录完成后调用）"""
        try:
            from mcpserver.agent_naga_portal.portal_login_manager import get_portal_login_manager
            login_manager = get_portal_login_manager()
            status = login_manager.get_status()
            cookies = login_manager.get_cookies()
            
            print(f"🌐 NagaPortal状态:")
            print(f"   地址: {config.naga_portal.portal_url}")
            print(f"   用户: {config.naga_portal.username[:3]}***{config.naga_portal.username[-3:] if len(config.naga_portal.username) > 6 else '***'}")
            
            if cookies:
                print(f"🍪 Cookie信息 ({len(cookies)}个):")
                for name, value in cookies.items():
                    print(f"   {name}: {value}")
            else:
                print(f"🍪 Cookie: 未获取到")
            
            user_id = status.get('user_id')
            if user_id:
                print(f"👤 用户ID: {user_id}")
            else:
                print(f"👤 用户ID: 未获取到")
                
            # 显示登录状态
            if status.get('is_logged_in'):
                print(f"✅ 登录状态: 已登录")
            else:
                print(f"❌ 登录状态: 未登录")
                if status.get('login_error'):
                    print(f"   错误: {status.get('login_error')}")
                    
        except Exception as e:
            print(f"🍪 NagaPortal状态获取失败: {e}")
    
    def save_log(self, u, a):  # 保存对话日志
        if self.dev_mode:
            return  # 开发者模式不写日志
        d = datetime.now().strftime('%Y-%m-%d')
        t = datetime.now().strftime('%H:%M:%S')
        
        # 确保日志目录存在
        log_dir = config.system.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"已创建日志目录: {log_dir}")
        
        # 保存对话日志
        log_file = os.path.join(log_dir, f"{d}.log")
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{t}] 用户: {u}\n")
                f.write(f"[{t}] 娜迦: {a}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            logger.error(f"保存日志失败: {e}")
    
    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        self.messages.append({"role": role, "content": content})
        
        # 限制历史消息数量，避免内存泄漏
        max_messages = 20
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    async def _call_llm(self, messages: List[Dict]) -> Dict:
        """调用LLM API"""
        if not self.api_client:
            return {'content': 'API客户端未初始化，请检查API密钥和配置。', 'status': 'error'}

        try:
            if self.provider == "gemini":
                # 将消息转换为Gemini格式
                # Gemini没有严格的system role，我们将system prompt合并到第一个user message中
                system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
                
                gemini_messages = []
                is_first_user_message = True
                for m in messages:
                    if m['role'] == 'user':
                        if is_first_user_message and system_prompt:
                            gemini_messages.append({'role': 'user', 'parts': [system_prompt + "\n\n" + m['content']]})
                            is_first_user_message = False
                        else:
                            gemini_messages.append({'role': 'user', 'parts': [m['content']]})
                    elif m['role'] == 'assistant':
                        # Gemini API 需要交替的用户和模型角色
                        if gemini_messages and gemini_messages[-1]['role'] == 'model':
                            # 如果上一条也是model，则合并内容
                             gemini_messages[-1]['parts'][0] += "\n" + m['content']
                        else:
                            gemini_messages.append({'role': 'model', 'parts': [m['content']]})

                # 确保第一条消息是 user
                if not gemini_messages or gemini_messages[0]['role'] != 'user':
                     gemini_messages.insert(0, {'role': 'user', 'parts': [system_prompt or "你好"]})


                resp = await self.api_client.generate_content_async(gemini_messages)
                return {
                    'content': resp.text,
                    'status': 'success'
                }
            else: # DeepSeek, OpenAI
                resp = await self.api_client.chat.completions.create(
                    model=config.api.model, 
                    messages=messages, 
                    temperature=config.api.temperature, 
                    max_tokens=config.api.max_tokens, 
                    stream=False
                )
                return {
                    'content': resp.choices[0].message.content,
                    'status': 'success'
                }
        except Exception as e:
            logger.error(f"{self.provider} API调用失败: {e}", exc_info=True)
            # 检查是否是认证失败
            if "API key" in str(e) or "authentication" in str(e).lower():
                 error_message = f"API认证失败，请检查你的 {self.provider} API密钥是否正确。"
            else:
                error_message = f"API调用失败: {str(e)}"
            return {
                'content': error_message,
                'status': 'error'
            }

    # 工具调用循环相关方法
    def handle_llm_response(self, a, mcp):
        # 只保留普通文本流式输出逻辑 #
        async def text_stream():
            for line in a.splitlines():
                yield ("娜迦", line)
        return text_stream()

    def _format_services_for_prompt(self, available_services: dict) -> str:
        """格式化可用服务列表为prompt字符串，MCP服务和Agent服务分开，包含具体调用格式"""
        mcp_services = available_services.get("mcp_services", [])
        agent_services = available_services.get("agent_services", [])
        
        # 获取本地城市信息和当前时间
        local_city = "未知城市"
        current_time = ""
        try:
            # 从WeatherTimeAgent获取本地城市信息
            from mcpserver.agent_weather_time.agent_weather_time import WeatherTimeTool
            weather_tool = WeatherTimeTool()
            local_city = getattr(weather_tool, '_local_city', '未知城市') or '未知城市'
            
            # 获取当前时间
            from datetime import datetime
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"[DEBUG] 获取本地信息失败: {e}")
        
        # 格式化MCP服务列表，包含具体调用格式
        mcp_list = []
        for service in mcp_services:
            name = service.get("name", "")
            description = service.get("description", "")
            display_name = service.get("display_name", name)
            tools = service.get("available_tools", [])
            
            # 展示name+displayName
            if description:
                mcp_list.append(f"- {name}: {description}")
            else:
                mcp_list.append(f"- {name}")
            
            # 为每个工具显示具体调用格式
            if tools:
                for tool in tools:
                    tool_name = tool.get('name', '')
                    tool_desc = tool.get('description', '')
                    tool_example = tool.get('example', '')
                    
                    if tool_name and tool_example:
                        # 解析示例JSON，提取参数
                        try:
                            import json
                            example_data = json.loads(tool_example)
                            params = []
                            for key, value in example_data.items():
                                if key != 'tool_name':
                                    # 特殊处理city参数，注入本地城市信息
                                    if key == 'city' and name == 'WeatherTimeAgent':
                                        params.append(f"{key}: {local_city}")
                                    else:
                                        params.append(f"{key}: {value}")
                            
                            # 构建调用格式
                            format_str = f"  {tool_name}: ｛\n"
                            format_str += f"    \"agentType\": \"mcp\",\n"
                            format_str += f"    \"service_name\": \"{name}\",\n"
                            format_str += f"    \"tool_name\": \"{tool_name}\",\n"
                            for param in params:
                                # 将中文参数名转换为英文
                                param_key, param_value = param.split(': ', 1)
                                if param_key == 'city' and name == 'WeatherTimeAgent':
                                    format_str += f"    \"{param_key}\": \"{local_city}\",\n"
                                else:
                                    format_str += f"    \"{param_key}\": \"{param_value}\",\n"
                            format_str += f"  ｝\n"
                            
                            mcp_list.append(format_str)
                        except:
                            # 如果JSON解析失败，使用简单格式
                            mcp_list.append(f"  {tool_name}: 使用tool_name参数调用")
        
        # 格式化Agent服务列表
        agent_list = []
        
        # 1. 添加handoff服务
        for service in agent_services:
            name = service.get("name", "")
            description = service.get("description", "")
            tool_name = service.get("tool_name", "agent")
            display_name = service.get("display_name", name)
            # 展示name+displayName
            if description:
                agent_list.append(f"- {name}(工具名: {tool_name}): {description}")
            else:
                agent_list.append(f"- {name}(工具名: {tool_name})")
        
        # 2. 直接从AgentManager获取已注册的Agent
        try:
            from mcpserver.agent_manager import get_agent_manager
            agent_manager = get_agent_manager()
            agent_manager_agents = agent_manager.get_available_agents()
            
            for agent in agent_manager_agents:
                name = agent.get("name", "")
                base_name = agent.get("base_name", "")
                description = agent.get("description", "")
                
                # 展示格式：base_name: 描述
                if description:
                    agent_list.append(f"- {base_name}: {description}")
                else:
                    agent_list.append(f"- {base_name}")
                    
        except Exception as e:
            # 如果AgentManager不可用，静默处理
            pass
        
        # 添加本地信息说明
        local_info = f"\n\n【当前环境信息】\n- 本地城市: {local_city}\n- 当前时间: {current_time}\n\n【使用说明】\n- 天气/时间查询时，请使用上述本地城市信息作为city参数\n- 所有时间相关查询都基于当前系统时间"
        
        # 返回格式化的服务列表
        result = {
            "available_mcp_services": "\n".join(mcp_list) + local_info if mcp_list else "无" + local_info,
            "available_agent_services": "\n".join(agent_list) if agent_list else "无"
        }
        
        return result

    async def process(self, u, is_voice_input=False):  # 添加is_voice_input参数
        try:
            # 始终在处理开始时初始化/重新初始化API客户端
            # 这可以确保客户端使用当前工作线程的事件循环
            self._initialize_api_client()

            # 开发者模式优先判断
            if u.strip().lower() == "#devmode":
                self.dev_mode = not self.dev_mode  # 切换模式
                status = "进入" if self.dev_mode else "退出"
                yield ("娜迦", f"已{status}开发者模式")
                return

            # 只在语音输入时显示处理提示
            if is_voice_input:
                print(f"开始处理用户输入：{now()}")  # 语音转文本结束，开始处理
                     
            # 添加handoff提示词
            system_prompt = f"{RECOMMENDED_PROMPT_PREFIX}\n{config.prompts.naga_system_prompt}"
            
            # 获取过滤后的服务列表
            available_services = self.mcp.get_available_services_filtered()
            services_text = self._format_services_for_prompt(available_services)
            
            # 简化的消息拼接逻辑（UI界面使用）
            sysmsg = {"role": "system", "content": system_prompt.format(**services_text)}
            msgs = [sysmsg] if sysmsg else []
            msgs += self.messages[-20:] + [{"role": "user", "content": u}]

            print(f"GTP请求发送：{now()}")  # AI请求前
            
            # 禁用非线性思考判断
            # thinking_task = None
            # if hasattr(self, 'tree_thinking') and self.tree_thinking and getattr(self.tree_thinking, 'is_enabled', False):
            #     # 启动异步思考判断任务
            #     import asyncio
            #     thinking_task = asyncio.create_task(self._async_thinking_judgment(u))
            
            # 普通模式：走工具调用循环（根据配置决定是否流式）
            try:
                # 根据配置决定是否使用流式处理
                is_streaming = config.system.stream_mode
                result = await tool_call_loop(msgs, self.mcp, self._call_llm, is_streaming=is_streaming)
                final_content = result['content']
                recursion_depth = result['recursion_depth']
                
                if recursion_depth > 0:
                    print(f"工具调用循环完成，共执行 {recursion_depth} 轮")
                
                # 根据配置决定输出方式
                if is_streaming:
                    # 流式输出最终结果
                    for line in final_content.splitlines():
                        yield ("娜迦", line)
                else:
                    # 非流式输出完整结果
                    yield ("娜迦", final_content)
                
                # 保存对话历史
                self.messages += [{"role": "user", "content": u}, {"role": "assistant", "content": final_content}]
                self.save_log(u, final_content)
                
                # GRAG记忆存储（开发者模式不写入）
                if self.memory_manager and not self.dev_mode:
                    try:
                        await self.memory_manager.add_conversation_memory(u, final_content)
                    except Exception as e:
                        logger.error(f"GRAG记忆存储失败: {e}")
                
                # 禁用异步思考判断结果检查
                # if thinking_task and not thinking_task.done():
                #     # 等待思考判断完成（最多等待3秒）
                #     try:
                #         await asyncio.wait_for(thinking_task, timeout=3.0)
                #         if thinking_task.result():
                #             yield ("娜迦", "\n💡 这个问题较为复杂，下面我会更详细地解释这个流程...")
                #             # 启动深度思考
                #             try:
                #                 thinking_result = await self.tree_thinking.think_deeply(u)
                #                 if thinking_result and "answer" in thinking_result:
                #                     # 直接使用thinking系统的结果，避免重复处理
                #                     yield ("娜迦", f"\n{thinking_result['answer']}")
                #                     
                #                     # 更新对话历史
                #                     final_thinking_answer = thinking_result['answer']
                #                     self.messages[-1] = {"role": "assistant", "content": final_content + "\n\n" + final_thinking_answer}
                #                     self.save_log(u, final_content + "\n\n" + final_thinking_answer)
                #                     
                #                     # GRAG记忆存储（开发者模式不写入）
                #                     if self.memory_manager and not self.dev_mode:
                #                         try:
                #                             await self.memory_manager.add_conversation_memory(u, final_content + "\n\n" + final_thinking_answer)
                #                         except Exception as e:
                #                             logger.error(f"GRAG记忆存储失败: {e}")
                #             except Exception as e:
                #                 logger.error(f"深度思考处理失败: {e}")
                #                 yield ("娜迦", f"🌳 深度思考系统出错: {str(e)}")
                #     except asyncio.TimeoutError:
                #         # 超时取消任务
                #         thinking_task.cancel()
                #     except Exception as e:
                #         logger.debug(f"思考判断任务异常: {e}")
                
            except Exception as e:
                print(f"工具调用循环失败: {e}")
                yield ("娜迦", f"[MCP异常]: {e}")
                return

            return
        except Exception as e:
            import sys
            import traceback
            traceback.print_exc(file=sys.stderr)
            yield ("娜迦", f"[MCP异常]: {e}")
            return

    async def get_response(self, prompt: str, temperature: float = 0.7) -> str:
        """为树状思考系统等提供API调用接口""" # 统一接口
        try:
            response = await self.async_client.chat.completions.create(
                model=config.api.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=config.api.max_tokens
            )
            return response.choices[0].message.content
        except RuntimeError as e:
            if "handler is closed" in str(e):
                logger.debug(f"忽略连接关闭异常，重新创建客户端: {e}")
                # 重新创建客户端并重试
                self.async_client = AsyncOpenAI(api_key=config.api.api_key, base_url=config.api.base_url.rstrip('/') + '/')
                response = await self.async_client.chat.completions.create(
                    model=config.api.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=config.api.max_tokens
                )
                return response.choices[0].message.content
            else:
                logger.error(f"API调用失败: {e}")
                return f"API调用出错: {str(e)}"
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return f"API调用出错: {str(e)}"

    # async def _async_thinking_judgment(self, question: str) -> bool:
    #     """异步判断问题是否需要深度思考
        
    #     Args:
    #         question: 用户问题
            
    #     Returns:
    #         bool: 是否需要深度思考
    #     """
    #     try:
    #         if not self.tree_thinking:
    #             return False
            
    #         # 使用thinking文件夹中现成的难度判断器
    #         difficulty_assessment = await self.tree_thinking.difficulty_judge.assess_difficulty(question)
    #         difficulty = difficulty_assessment.get("difficulty", 3)
            
    #         # 根据难度判断是否需要深度思考
    #         # 难度4-5（复杂/极难）建议深度思考
    #         should_think_deeply = difficulty >= 4
            
    #         logger.info(f"难度判断：{difficulty}/5，建议深度思考：{should_think_deeply}")
    #         return should_think_deeply
                   
    #     except Exception as e:
    #         logger.debug(f"异步思考判断失败: {e}")
    #         return False

async def process_user_message(s,msg):
    # 当启用语音系统且无文本输入时，启动语音识别
    if config.system.voice_enabled and not msg: 
        # 当s.voice不为None时（即语音系统正常初始化）才进行语音识别
        if s.voice:
            async for text in s.voice.stt_stream():
                if text:
                    msg=text
                    break
            return await s.process(msg, is_voice_input=True)  # 语音输入
    return await s.process(msg, is_voice_input=False)  # 文字输入
