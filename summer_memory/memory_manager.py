import logging
import asyncio
import traceback
from typing import List, Dict, Optional, Tuple
from .quintuple_extractor import extract_quintuples
from .quintuple_graph import store_quintuples, query_graph_by_keywords, get_all_quintuples
from .quintuple_rag_query import query_knowledge, set_context
import config

logger = logging.getLogger(__name__)

class GRAGMemoryManager:
    """GRAG知识图谱记忆管理器"""
    
    def __init__(self):
        self.enabled = config.GRAG_ENABLED
        self.auto_extract = config.GRAG_AUTO_EXTRACT
        self.context_length = config.GRAG_CONTEXT_LENGTH
        self.similarity_threshold = config.GRAG_SIMILARITY_THRESHOLD
        self.recent_context = [] # 最近对话上下文
        self.extraction_cache = set() # 避免重复提取
        
        if not self.enabled:
            logger.info("GRAG记忆系统已禁用")
            return
            
        try:
            # 初始化Neo4j连接
            from .quintuple_graph import graph
            logger.info("GRAG记忆系统初始化成功")
        except Exception as e:
            logger.error(f"GRAG记忆系统初始化失败: {e}")
            self.enabled = False

    async def add_conversation_memory(self, user_input: str, ai_response: str) -> bool:
        """添加对话记忆到知识图谱（同时更新上下文和三元组）"""
        if not self.enabled:
            return False
        try:
            # 拼接本轮内容
            conversation_text = f"用户: {user_input}\n娜迦: {ai_response}"

            # 更新recent_context（限制长度）
            self.recent_context.append(conversation_text)
            if len(self.recent_context) > self.context_length:
                self.recent_context = self.recent_context[-self.context_length:]

            # 提取和存储五元组
            if self.auto_extract:
                # 创建并等待任务完成
                task = asyncio.create_task(self._extract_and_store_quintuples(conversation_text))
                # 添加超时防止永久阻塞
                try:
                    await asyncio.wait_for(task, timeout=20.0)
                except asyncio.TimeoutError:
                    logger.warning("五元组提取任务超时")
            return True
        except Exception as e:
            logger.error(f"添加对话记忆失败: {e}")
            return False

    async def _extract_and_store_quintuples(self, text: str) -> bool:
        try:
            import hashlib
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            if text_hash in self.extraction_cache:
                logger.debug(f"跳过已处理的文本: {text[:50]}...")
                return True

            logger.info(f"开始提取五元组: {text[:100]}...")
            quintuples = await asyncio.to_thread(extract_quintuples, text)

            if not quintuples:
                logger.warning("未提取到五元组")
                return False

            logger.info(f"提取到 {len(quintuples)} 个五元组，准备存储")

            # 存储到Neo4j
            store_success = await asyncio.to_thread(store_quintuples, quintuples)

            if store_success:
                self.extraction_cache.add(text_hash)
                logger.info("五元组存储成功")
                return True
            else:
                logger.error("五元组存储失败")
                return False

        except Exception as e:
            logger.error(f"提取和存储五元组失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def query_memory(self, question: str) -> Optional[str]:
        """查询记忆"""
        if not self.enabled:
            return None
            
        try:
            # 设置查询上下文
            set_context(self.recent_context)
            
            # 异步查询
            result = await asyncio.to_thread(query_knowledge, question)
            
            if result and "未在知识图谱中找到相关信息" not in result:
                logger.info("从记忆中找到相关信息")
                return result
            return None
        except Exception as e:
            logger.error(f"查询记忆失败: {e}")
            return None
    
    async def get_relevant_memories(self, query: str, limit: int = 3) -> List[Tuple[str, str, str, str, str]]:
        """获取相关记忆（五元组格式）"""
        if not self.enabled:
            return []
            
        try:
            # 从Neo4j查询相关五元组
            quintuples = await asyncio.to_thread(query_graph_by_keywords, [query])
            
            # 限制返回数量
            return quintuples[:limit]
        except Exception as e:
            logger.error(f"获取相关记忆失败: {e}")
            return []
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息"""
        if not self.enabled:
            return {"enabled": False}
            
        try:
            all_quintuples = get_all_quintuples()
            return {
                "enabled": True,
                "total_quintuples": len(all_quintuples),
                "context_length": len(self.recent_context),
                "cache_size": len(self.extraction_cache)
            }
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {"enabled": False, "error": str(e)}
    
    async def clear_memory(self) -> bool:
        """清空记忆"""
        if not self.enabled:
            return False
            
        try:
            self.recent_context.clear()
            self.extraction_cache.clear()
            logger.info("记忆已清空")
            return True
        except Exception as e:
            logger.error(f"清空记忆失败: {e}")
            return False

# 全局记忆管理器实例
memory_manager = GRAGMemoryManager() 
