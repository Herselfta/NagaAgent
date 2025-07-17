"""
树状思考核心引擎
协调各个组件，实现完整的树状外置思考系统
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from .thinking_node import ThinkingNode, ThinkingBranch
from .difficulty_judge import DifficultyJudge
from .preference_filter import PreferenceFilter, UserPreference
from .genetic_pruning import GeneticPruning
from .thread_pools import ThreadPoolManager, TaskBatch
from .config import TREE_THINKING_CONFIG

logger = logging.getLogger("TreeThinkingEngine")

# 全局子系统实例，避免重复初始化
_global_subsystems = {
    "difficulty_judge": None,
    "preference_filter": None,
    "genetic_pruning": None,
    "thread_pool": None
}

class TreeThinkingEngine:
    """树状思考核心引擎"""
    
    def __init__(self, api_client=None, memory_manager=None):
        self.api_client = api_client
        self.memory_manager = memory_manager
        self.config = TREE_THINKING_CONFIG
        
        # 初始化或复用子系统（参考handoff的全局变量保护机制）
        global _global_subsystems
        
        # 初始化子系统（只在第一次创建时初始化）
        if _global_subsystems["difficulty_judge"] is None:
            _global_subsystems["difficulty_judge"] = DifficultyJudge(api_client)
            _global_subsystems["preference_filter"] = PreferenceFilter(api_client)
            _global_subsystems["genetic_pruning"] = GeneticPruning(api_client)
            _global_subsystems["thread_pool"] = ThreadPoolManager()
            print("[TreeThinkingEngine] 🌳 树状思考引擎子系统初始化完成")
            print("[TreeThinkingEngine] 🚀 树状思考引擎初始化完成")
        else:
            # 复用时不播报，避免重复日志
            pass
        
        # 使用全局子系统实例
        self.difficulty_judge = _global_subsystems["difficulty_judge"]
        self.preference_filter = _global_subsystems["preference_filter"]
        self.genetic_pruning = _global_subsystems["genetic_pruning"]
        self.thread_pool = _global_subsystems["thread_pool"]
        
        # 运行状态
        self.is_enabled = self.config["enabled"]
        self.current_session = None
        self.thinking_history = []
    
    async def think_deeply(self, question: str, user_preferences: Optional[List[UserPreference]] = None) -> Dict[str, Any]:
        """
        深度思考主入口
        """
        if not self.is_enabled:
            logger.info("树状思考系统未启用，使用基础回答")
            return await self._basic_response(question)
        
        try:
            start_time = time.time()
            session_id = f"thinking_{int(start_time)}"
            self.current_session = session_id
            
            logger.info(f"开始深度思考会话: {session_id}")
            logger.info(f"问题: {question[:100]}...")
            
            # 1. 问题难度评估
            difficulty_assessment = await self.difficulty_judge.assess_difficulty(question)
            logger.info(f"难度评估: {difficulty_assessment['reasoning']}")
            
            # 2. 更新用户偏好
            if user_preferences:
                self.preference_filter.update_preferences(user_preferences)
            
            # 3. 生成多路思考
            thinking_routes = await self._generate_thinking_routes(
                question, difficulty_assessment
            )
            
            # 4. 偏好打分
            if thinking_routes:
                route_scores = await self.preference_filter.score_thinking_nodes(thinking_routes)
                logger.info(f"完成 {len(thinking_routes)} 条思考路线的偏好打分")
            else:
                route_scores = {}
            
            # 5. 遗传算法剪枝
            if len(thinking_routes) > 3:
                optimal_routes = await self.genetic_pruning.evolve_thinking_tree(
                    thinking_routes, target_count=3
                )
                logger.info(f"遗传剪枝后保留 {len(optimal_routes)} 条最优路线")
            else:
                optimal_routes = thinking_routes
            
            # 6. 综合最终答案
            final_answer = await self._synthesize_final_answer(
                question, optimal_routes, difficulty_assessment
            )
            
            # 7. 记录思考过程
            thinking_session = {
                "session_id": session_id,
                "question": question,
                "difficulty_assessment": difficulty_assessment,
                "thinking_routes": len(thinking_routes),
                "optimal_routes": len(optimal_routes),
                "route_scores": route_scores,
                "final_answer": final_answer,
                "processing_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
            self.thinking_history.append(thinking_session)
            
            logger.info(f"深度思考完成，耗时 {thinking_session['processing_time']:.2f}秒")
            
            return {
                "answer": final_answer,
                "thinking_process": {
                    "difficulty": difficulty_assessment,
                    "routes_generated": len(thinking_routes),
                    "routes_selected": len(optimal_routes),
                    "processing_time": thinking_session['processing_time'],
                    "thinking_details": [
                        {
                            "route_id": route.id,
                            "branch_type": route.branch_type,
                            "content": route.content,
                            "score": route.score,
                            "fitness": route.fitness
                        }
                        for route in optimal_routes
                    ]
                },
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"深度思考过程出错: {e}")
            # 降级到基础回答
            return await self._basic_response(question)
        
        finally:
            self.current_session = None
    
    async def _generate_thinking_routes(self, question: str, difficulty_assessment: Dict) -> List[ThinkingNode]:
        """生成多路思考"""
        routes_count = difficulty_assessment["routes"]
        temperatures = self.difficulty_judge.get_temperature_distribution(routes_count)
        branch_types = self.difficulty_judge.get_branch_types(routes_count)
        
        logger.info(f"生成 {routes_count} 条思考路线，温度范围: {min(temperatures)}-{max(temperatures)}")
        
        # 创建任务批次
        task_batch = TaskBatch(self.thread_pool)
        
        # 为每个思考路线创建任务
        for i in range(routes_count):
            temperature = temperatures[i]
            branch_type = branch_types[i]
            
            # 根据分支类型调整提示词
            thinking_prompt = self._create_thinking_prompt(question, branch_type, i+1, routes_count)
            
            # 添加API任务
            task_batch.add_api_task(
                self._generate_single_route,
                thinking_prompt, temperature, branch_type, i
            )
        
        # 并行执行所有思考任务
        try:
            thinking_results, api_results = await task_batch.execute_all()
            
            # 合并所有结果
            all_results = (thinking_results or []) + (api_results or [])
            logger.info(f"原始结果数量: thinking={len(thinking_results or [])}, api={len(api_results or [])}, 总计={len(all_results)}")
            
            # 过滤有效结果
            valid_routes = []
            for i, result in enumerate(all_results):
                logger.info(f"结果 {i}: 类型={type(result)}, 是否为ThinkingNode={isinstance(result, ThinkingNode)}")
                if isinstance(result, ThinkingNode):
                    logger.info(f"  内容长度: {len(result.content)}, 内容预览: {result.content[:50]}...")
                    if result.content and result.content.strip():
                        valid_routes.append(result)
                    else:
                        logger.warning(f"  思考节点内容为空")
                else:
                    logger.warning(f"  结果不是ThinkingNode类型: {result}")
            
            # 建立兄弟关系
            if valid_routes:
                self._establish_sibling_relationships(valid_routes)
            
            logger.info(f"成功生成 {len(valid_routes)}/{routes_count} 条思考路线")
            return valid_routes
            
        except Exception as e:
            logger.error(f"生成思考路线失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_thinking_prompt(self, question: str, branch_type: str, route_num: int, total_routes: int) -> str:
        """创建思考提示词"""
        from .config import BRANCH_TYPES
        
        branch_description = BRANCH_TYPES.get(branch_type, "综合分析型")
        
        # 获取相关记忆（如果有记忆管理器）
        memory_context = ""
        if self.memory_manager:
            try:
                related_memories = self.memory_manager.recall_memory(question, k=3)
                if related_memories:
                    memory_context = f"\n相关记忆参考：\n"
                    for memory in related_memories:
                        memory_context += f"- {memory.get('text', '')[:100]}...\n"
            except Exception as e:
                logger.warning(f"获取相关记忆失败: {e}")
        
        prompt = f"""
作为{branch_description}思考者，请深入分析以下问题（第{route_num}/{total_routes}路思考）：

问题：{question}

{memory_context}

请从{branch_description}的角度进行深度思考，要求：
1. 思考过程要体现{branch_description}的特点
2. 提供详细的分析和推理过程
3. 给出具体、可操作的建议或结论
4. 避免过于简化的回答
5. 长度控制在150-400字之间

请直接输出你的深度思考内容：
"""
        return prompt
    
    async def _generate_single_route(self, prompt: str, temperature: float, 
                                   branch_type: str, route_index: int) -> ThinkingNode:
        """生成单条思考路线"""
        try:
            logger.info(f"开始生成思考路线 {route_index}, 温度: {temperature}, 类型: {branch_type}")
            
            # 调用API生成内容
            content = await self.api_client.get_response(prompt, temperature=temperature)
            
            logger.info(f"路线 {route_index} API调用成功，内容长度: {len(content)}")
            logger.info(f"路线 {route_index} 内容预览: {content[:100]}...")
            
            # 创建思考节点
            node = ThinkingNode(
                content=content.strip(),
                temperature=temperature,
                branch_type=branch_type,
                metadata={
                    "route_index": route_index,
                    "prompt_length": len(prompt),
                    "generated_at": time.time()
                }
            )
            
            node.update_content(content.strip())
            
            logger.info(f"路线 {route_index} 思考节点创建成功")
            return node
            
        except Exception as e:
            logger.error(f"生成思考路线 {route_index} 失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回空节点
            return ThinkingNode(
                content=f"思考路线 {route_index} 生成失败: {str(e)}",
                temperature=temperature,
                branch_type=branch_type
            )
    
    async def _synthesize_final_answer(self, question: str, optimal_routes: List[ThinkingNode], 
                                     difficulty_assessment: Dict) -> str:
        """综合最终答案"""
        if not optimal_routes:
            return "抱歉，无法生成有效的思考方案。"
        
        try:
            # 构建综合提示
            routes_summary = ""
            for i, route in enumerate(optimal_routes, 1):
                routes_summary += f"\n思考路线{i}（{route.branch_type}，评分:{route.score:.1f}）：\n{route.content}\n"
            
            synthesis_prompt = f"""
基于以下多路深度思考的结果，请综合生成一个完整、准确的最终答案：

原问题：{question}

问题难度：{difficulty_assessment['difficulty']}/5
{routes_summary}

请综合以上思考路线的优点，生成一个：
1. 逻辑清晰、结构完整的答案
2. 融合不同角度的思考成果
3. 突出重点，条理分明
4. 具有实用价值和指导意义
5. 长度适中（300-600字）

最终答案：
"""
            
            # 使用中等温度生成综合答案
            final_answer = await self.api_client.get_response(
                synthesis_prompt, 
                temperature=0.7
            )
            
            return final_answer.strip()
            
        except Exception as e:
            logger.error(f"综合最终答案失败: {e}")
            # 降级方案：返回最佳思考路线
            best_route = max(optimal_routes, key=lambda x: x.score if x.score > 0 else x.fitness)
            return f"基于最佳思考路线的回答：\n\n{best_route.content}"
    
    async def _basic_response(self, question: str) -> Dict[str, Any]:
        """基础回答（降级方案）"""
        try:
            if self.api_client:
                basic_answer = await self.api_client.get_response(question, temperature=0.7)
            else:
                basic_answer = "抱歉，无法处理这个问题。"
            
            return {
                "answer": basic_answer,
                "thinking_process": {
                    "mode": "basic",
                    "note": "使用基础回答模式"
                },
                "session_id": None
            }
        except Exception as e:
            return {
                "answer": f"处理问题时出现错误：{str(e)}",
                "thinking_process": {"mode": "error"},
                "session_id": None
            }
    
    def enable_tree_thinking(self, enabled: bool = True):
        """启用/禁用树状思考"""
        self.is_enabled = enabled
        logger.info(f"树状思考系统已{'启用' if enabled else '禁用'}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "enabled": self.is_enabled,
            "current_session": self.current_session,
            "total_sessions": len(self.thinking_history),
            "thread_pool_status": self.thread_pool.get_pool_status(),
            "config": self.config,
            "components": {
                "difficulty_judge": "已初始化",
                "preference_filter": "已初始化", 
                "genetic_pruning": "已初始化",
                "thread_pool": "已初始化"
            }
        }
    
    def get_thinking_history(self, limit: int = 10) -> List[Dict]:
        """获取思考历史"""
        return self.thinking_history[-limit:] if self.thinking_history else []
    
    def clear_thinking_history(self):
        """清空思考历史"""
        self.thinking_history.clear()
        logger.info("思考历史已清空")
    
    def cleanup(self):
        """清理资源"""
        logger.info("正在清理树状思考引擎资源...")
        
        # 清理线程池
        self.thread_pool.cleanup()
        
        # 清理历史记录
        self.thinking_history.clear()
        
        logger.info("树状思考引擎资源清理完成")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def _establish_sibling_relationships(self, nodes: List[ThinkingNode]):
        """建立兄弟关系，标注同代节点"""
        try:
            # 为所有同代节点建立兄弟关系
            node_ids = [node.id for node in nodes]
            
            for i, node in enumerate(nodes):
                # 设置同代索引
                node.metadata["family_tree"]["generation_index"] = i
                
                # 设置兄弟节点ID列表（排除自己）
                sibling_ids = [nid for nid in node_ids if nid != node.id]
                node.sibling_ids = sibling_ids
                node.metadata["family_tree"]["siblings"] = sibling_ids
                
                # 标记分支谱系
                node.metadata["family_tree"]["branch_lineage"] = [node.branch_type]
                
                # 标记为初始代
                node.metadata["thinking_process"]["creation_method"] = "initial_generation"
                
                logger.debug(f"节点 {node.id[:8]} 建立兄弟关系: {len(sibling_ids)} 个兄弟节点")
            
            logger.info(f"成功为 {len(nodes)} 个节点建立兄弟关系")
            
        except Exception as e:
            logger.warning(f"建立兄弟关系失败: {e}") 