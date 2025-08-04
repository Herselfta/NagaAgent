#!/usr/bin/env python3
"""
NagaAgent API服务器
提供RESTful API接口访问NagaAgent功能
"""

import asyncio
import json
import sys
import traceback
import re
import os
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, AsyncGenerator, Any

# 在导入其他模块前先设置HTTP库日志级别
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import aiohttp

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入独立的工具调用模块
from .tool_call_utils import parse_tool_calls, execute_tool_calls, tool_call_loop
from .message_manager import message_manager  # 导入统一的消息管理器

# 导入配置系统
from config import config  # 使用新的配置系统
from ui.response_utils import extract_message  # 导入消息提取工具
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX  # handoff提示词

# 全局NagaAgent实例 - 延迟导入避免循环依赖
naga_agent = None

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # 移除断开的连接
                self.active_connections.remove(connection)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global naga_agent
    try:
        print("[INFO] 正在初始化NagaAgent...")
        # 延迟导入避免循环依赖
        from conversation_core import NagaConversation
        naga_agent = NagaConversation()  # 第四次初始化：API服务器启动时创建
        print("[SUCCESS] NagaAgent初始化完成")
        yield
    except Exception as e:
        print(f"[ERROR] NagaAgent初始化失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("[INFO] 正在清理资源...")
        if naga_agent and hasattr(naga_agent, 'mcp'):
            try:
                await naga_agent.mcp.cleanup()
            except Exception as e:
                print(f"[WARNING] 清理MCP资源时出错: {e}")

# 创建FastAPI应用
app = FastAPI(
    title="NagaAgent API",
    description="智能对话助手API服务",
    version="3.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    status: str = "success"

class MCPRequest(BaseModel):
    service_name: str
    task: Dict
    session_id: Optional[str] = None

class SystemInfoResponse(BaseModel):
    version: str
    status: str
    available_services: List[str]
    api_key_configured: bool

# WebSocket路由
@app.websocket("/ws/mcplog")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点 - 提供MCP实时通知"""
    await manager.connect(websocket)
    try:
        # 发送连接确认
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_ack",
                "message": "WebSocket连接成功"
            }, ensure_ascii=False),
            websocket
        )
        
        # 保持连接
        while True:
            try:
                # 等待客户端消息（心跳检测）
                data = await websocket.receive_text()
                # 可以处理客户端发送的消息
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "message": "收到心跳"
                    }, ensure_ascii=False),
                    websocket
                )
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
    except Exception as e:
        print(f"WebSocket错误: {e}")
        manager.disconnect(websocket)

# API路由
@app.get("/", response_model=Dict[str, str])
async def root():
    """API根路径"""
    return {
        "name": "NagaAgent API",
        "version": "3.0",
        "status": "running",
        "docs": "/docs",
        "websocket": "/ws/mcplog"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "agent_ready": naga_agent is not None,
        "timestamp": str(asyncio.get_event_loop().time())
    }

@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """获取系统信息"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    return SystemInfoResponse(
        version="3.0",
        status="running",
        available_services=naga_agent.mcp.list_mcps(),
        api_key_configured=bool(config.api.api_key and config.api.api_key != "sk-placeholder-key-not-set")
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """普通对话接口"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")
    
    try:
        # 获取或创建会话ID
        session_id = message_manager.create_session(request.session_id)
        
        # 构建系统提示词
        system_prompt = f"{RECOMMENDED_PROMPT_PREFIX}\n{config.prompts.naga_system_prompt}"
        available_services = naga_agent.mcp.get_available_services_filtered()
        services_text = naga_agent._format_services_for_prompt(available_services)
        system_prompt = system_prompt.format(**services_text)
        
        # 使用消息管理器构建完整的对话消息
        messages = message_manager.build_conversation_messages(
            session_id=session_id,
            system_prompt=system_prompt,
            current_message=request.message
        )
        
        # 定义LLM调用函数
        async def call_llm(messages: List[Dict]) -> Dict:
            """调用LLM API"""
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.api.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.api.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.api.model,
                        "messages": messages,
                        "temperature": config.api.temperature,
                        "max_tokens": config.api.max_tokens,
                        "stream": False
                    }
                ) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=resp.status, detail="LLM API调用失败")
                    
                    data = await resp.json()
                    return {
                        'content': data['choices'][0]['message']['content'],
                        'status': 'success'
                    }
        
        # 处理工具调用循环
        result = await tool_call_loop(messages, naga_agent.mcp, call_llm, is_streaming=False)
        
        # 提取最终响应
        response_text = result['content']
        
        # 保存对话历史到消息管理器
        message_manager.add_message(session_id, "user", request.message)
        message_manager.add_message(session_id, "assistant", response_text)
        
        return ChatResponse(
            response=extract_message(response_text) if response_text else response_text,
            session_id=session_id,
            status="success"
        )
    except Exception as e:
        print(f"对话处理错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")
    
    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            # 获取或创建会话ID
            session_id = message_manager.create_session(request.session_id)
            
            # 发送会话ID信息
            yield f"data: session_id: {session_id}\n\n"
            
            # 构建系统提示词
            system_prompt = f"{RECOMMENDED_PROMPT_PREFIX}\n{config.prompts.naga_system_prompt}"
            available_services = naga_agent.mcp.get_available_services_filtered()
            services_text = naga_agent._format_services_for_prompt(available_services)
            system_prompt = system_prompt.format(**services_text)
            
            # 使用消息管理器构建完整的对话消息
            messages = message_manager.build_conversation_messages(
                session_id=session_id,
                system_prompt=system_prompt,
                current_message=request.message
            )
            
            # 定义LLM调用函数
            async def call_llm(messages: List[Dict]) -> Dict:
                """调用LLM API"""
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{config.api.base_url}/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {config.api.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": config.api.model,
                            "messages": messages,
                            "temperature": config.api.temperature,
                            "max_tokens": config.api.max_tokens,
                            "stream": False
                        }
                    ) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=resp.status, detail="LLM API调用失败")
                        
                        data = await resp.json()
                        return {
                            'content': data['choices'][0]['message']['content'],
                            'status': 'success'
                        }
            
            # 处理工具调用循环
            result = await tool_call_loop(messages, naga_agent.mcp, call_llm, is_streaming=True)
            
            # 流式输出最终结果
            final_content = result['content']
            for line in final_content.splitlines():
                if line.strip():
                    yield f"data: {line}\n\n"
            
            # 保存对话历史到消息管理器
            message_manager.add_message(session_id, "user", request.message)
            message_manager.add_message(session_id, "assistant", final_content)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"流式对话处理错误: {e}")
            traceback.print_exc()
            yield f"data: 错误: {str(e)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/mcp/handoff")
async def mcp_handoff(request: MCPRequest):
    """MCP服务调用接口"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    try:
        # 获取或创建会话ID
        session_id = message_manager.get_or_create_session(request.session_id)
        
        # 直接调用MCP handoff
        result = await naga_agent.mcp.handoff(
            service_name=request.service_name,
            task=request.task
        )
        
        return {
            "status": "success",
            "result": result,
            "session_id": session_id  # 使用生成的会话ID
        }
    except Exception as e:
        print(f"MCP handoff错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"handoff失败: {str(e)}")

@app.get("/mcp/services")
async def get_mcp_services():
    """获取可用的MCP服务列表"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    try:
        # 使用动态服务池查询
        services = naga_agent.mcp.get_available_services()
        statistics = naga_agent.mcp.get_service_statistics()
        
        return {
            "status": "success",
            "services": services,
            "statistics": statistics,
            "count": len(services)
        }
    except Exception as e:
        print(f"获取MCP服务列表错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取服务列表失败: {str(e)}")

@app.get("/mcp/services/{service_name}")
async def get_mcp_service_detail(service_name: str):
    """获取指定MCP服务的详细信息"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    try:
        # 使用动态服务池查询
        service_info = naga_agent.mcp.query_service_by_name(service_name)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"服务 {service_name} 不存在")
        
        return {
            "status": "success",
            "service": service_info
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取MCP服务详情错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取服务详情失败: {str(e)}")

@app.get("/mcp/services/search/{capability}")
async def search_mcp_services(capability: str):
    """根据能力关键词搜索MCP服务"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    try:
        # 使用动态服务池查询
        matching_services = naga_agent.mcp.query_services_by_capability(capability)
        
        return {
            "status": "success",
            "capability": capability,
            "services": matching_services,
            "count": len(matching_services)
        }
    except Exception as e:
        print(f"搜索MCP服务错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索服务失败: {str(e)}")

@app.get("/mcp/services/{service_name}/tools")
async def get_mcp_service_tools(service_name: str):
    """获取指定MCP服务的可用工具列表"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    try:
        # 使用动态服务池查询
        tools = naga_agent.mcp.get_service_tools(service_name)
        
        return {
            "status": "success",
            "service_name": service_name,
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        print(f"获取MCP服务工具列表错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取工具列表失败: {str(e)}")

@app.get("/mcp/statistics")
async def get_mcp_statistics():
    """获取MCP服务统计信息"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    try:
        # 使用动态服务池查询
        statistics = naga_agent.mcp.get_service_statistics()
        
        return {
            "status": "success",
            "statistics": statistics
        }
    except Exception as e:
        print(f"获取MCP统计信息错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.post("/system/devmode")
async def toggle_devmode():
    """切换开发者模式"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    try:
        naga_agent.dev_mode = not naga_agent.dev_mode
        return {
            "status": "success",
            "dev_mode": naga_agent.dev_mode,
            "message": f"开发者模式已{'启用' if naga_agent.dev_mode else '禁用'}"
        }
    except Exception as e:
        print(f"切换开发者模式错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"切换开发者模式失败: {str(e)}")

@app.get("/memory/stats")
async def get_memory_stats():
    """获取记忆统计信息"""
    if not naga_agent:
        raise HTTPException(status_code=503, detail="NagaAgent未初始化")
    
    try:
        if hasattr(naga_agent, 'memory_manager') and naga_agent.memory_manager:
            stats = naga_agent.memory_manager.get_memory_stats()
            return {
                "status": "success",
                "memory_stats": stats
            }
        else:
            return {
                "status": "success",
                "memory_stats": {"enabled": False, "message": "记忆系统未启用"}
            }
    except Exception as e:
        print(f"获取记忆统计错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取记忆统计失败: {str(e)}")

@app.get("/sessions")
async def get_sessions():
    """获取所有会话信息"""
    try:
        # 清理过期会话
        message_manager.cleanup_old_sessions()
        
        # 获取所有会话信息
        sessions_info = message_manager.get_all_sessions_info()
        
        return {
            "status": "success",
            "sessions": sessions_info,
            "total_sessions": len(sessions_info)
        }
    except Exception as e:
        print(f"获取会话信息错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取会话信息失败: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """获取指定会话的详细信息"""
    try:
        session_info = message_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return {
            "status": "success",
            "session_id": session_id,
            "session_info": session_info,
            "messages": message_manager.get_messages(session_id),
            "conversation_rounds": session_info["conversation_rounds"]
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取会话详情错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取会话详情失败: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    try:
        success = message_manager.delete_session(session_id)
        if success:
            return {
                "status": "success",
                "message": f"会话 {session_id} 已删除"
            }
        else:
            raise HTTPException(status_code=404, detail="会话不存在")
    except HTTPException:
        raise
    except Exception as e:
        print(f"删除会话错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

@app.delete("/sessions")
async def clear_all_sessions():
    """清空所有会话"""
    try:
        count = message_manager.clear_all_sessions()
        return {
            "status": "success",
            "message": f"已清空 {count} 个会话"
        }
    except Exception as e:
        print(f"清空会话错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"清空会话失败: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NagaAgent API服务器")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="开启自动重载")
    
    args = parser.parse_args()
    
    print(f"🚀 启动NagaAgent API服务器...")
    print(f"📍 地址: http://{args.host}:{args.port}")
    print(f"📚 文档: http://{args.host}:{args.port}/docs")
    print(f"🔄 自动重载: {'开启' if args.reload else '关闭'}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    ) 