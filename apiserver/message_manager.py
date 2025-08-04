#!/usr/bin/env python3
"""
统一的消息管理模块
支持多会话、多agent的消息存储和拼接
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MessageManager:
    """统一的消息管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        # 从配置文件读取最大历史轮数，默认为10轮
        try:
            from config import config
            self.max_history_rounds = config.api.max_history_rounds
            self.max_messages_per_session = self.max_history_rounds * 2  # 每轮对话包含用户和助手各一条消息
        except ImportError:
            self.max_history_rounds = 10
            self.max_messages_per_session = 20  # 默认20条消息
            logger.warning("无法导入配置，使用默认历史轮数设置")
    
    def generate_session_id(self) -> str:
        """生成唯一的会话ID"""
        return str(uuid.uuid4())
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """创建新会话"""
        if not session_id:
            session_id = self.generate_session_id()
        
        self.sessions[session_id] = {
            "created_at": asyncio.get_event_loop().time(),
            "messages": [],
            "agent_type": "default",  # 可以扩展支持不同agent类型
            "last_activity": asyncio.get_event_loop().time()
        }
        
        logger.info(f"创建新会话: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """向会话添加消息"""
        if session_id not in self.sessions:
            logger.warning(f"会话不存在: {session_id}")
            return False
        
        session = self.sessions[session_id]
        session["messages"].append({"role": role, "content": content})
        session["last_activity"] = asyncio.get_event_loop().time()
        
        # 限制消息数量
        if len(session["messages"]) > self.max_messages_per_session:
            session["messages"] = session["messages"][-self.max_messages_per_session:]
        
        logger.debug(f"会话 {session_id} 添加消息: {role} - {content[:50]}...")
        return True
    
    def get_messages(self, session_id: str) -> List[Dict]:
        """获取会话的所有消息"""
        session = self.sessions.get(session_id)
        return session["messages"] if session else []
    
    def get_recent_messages(self, session_id: str, count: Optional[int] = None) -> List[Dict]:
        """获取会话的最近消息"""
        if count is None:
            count = self.max_messages_per_session
        messages = self.get_messages(session_id)
        return messages[-count:] if messages else []
    
    def build_conversation_messages(self, session_id: str, system_prompt: str, 
                                  current_message: str, include_history: bool = True) -> List[Dict]:
        """构建完整的对话消息列表"""
        messages = []
        
        # 添加系统提示词
        messages.append({"role": "system", "content": system_prompt})
        
        # 添加历史对话
        if include_history:
            recent_messages = self.get_recent_messages(session_id)
            messages.extend(recent_messages)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": current_message})
        
        return messages
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话详细信息"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "message_count": len(session["messages"]),
            "conversation_rounds": len(session["messages"]) // 2,
            "agent_type": session["agent_type"],
            "max_history_rounds": self.max_history_rounds,  # 添加最大历史轮数信息
            "last_message": session["messages"][-1]["content"][:100] + "..." if session["messages"] else "无对话历史"
        }
    
    def get_all_sessions_info(self) -> Dict[str, Dict]:
        """获取所有会话信息"""
        sessions_info = {}
        for session_id, session in self.sessions.items():
            sessions_info[session_id] = self.get_session_info(session_id)
        return sessions_info
    
    def delete_session(self, session_id: str) -> bool:
        """删除指定会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False
    
    def clear_all_sessions(self) -> int:
        """清空所有会话"""
        count = len(self.sessions)
        self.sessions.clear()
        logger.info(f"清空所有会话，共 {count} 个")
        return count
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """清理过期会话"""
        current_time = asyncio.get_event_loop().time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_activity"] > max_age_hours * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
        
        return len(expired_sessions)
    
    def set_agent_type(self, session_id: str, agent_type: str) -> bool:
        """设置会话的agent类型"""
        if session_id in self.sessions:
            self.sessions[session_id]["agent_type"] = agent_type
            return True
        return False
    
    def get_agent_type(self, session_id: str) -> Optional[str]:
        """获取会话的agent类型"""
        session = self.sessions.get(session_id)
        return session["agent_type"] if session else None

# 全局消息管理器实例
message_manager = MessageManager() 