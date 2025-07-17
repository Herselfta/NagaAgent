"""
黑白名单偏好打分系统
根据用户偏好对思考方案进行评分筛选
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from .thinking_node import ThinkingNode
from .config import TREE_THINKING_CONFIG

logger = logging.getLogger("PreferenceFilter")

@dataclass
class UserPreference:
    """用户偏好配置"""
    name: str
    description: str
    weight: float = 1.0
    enabled: bool = True
    
    # 偏好类型
    prefer_complex: bool = True      # 偏好复杂思考
    prefer_reasoning: bool = True    # 偏好推理完善
    prefer_memory: bool = True       # 偏好调用记忆
    prefer_innovation: bool = False  # 偏好创新方案
    prefer_practical: bool = True    # 偏好实用价值
    
    # 黑名单关键词
    blacklist_keywords: List[str] = None
    # 白名单关键词
    whitelist_keywords: List[str] = None
    
    def __post_init__(self):
        if self.blacklist_keywords is None:
            self.blacklist_keywords = []
        if self.whitelist_keywords is None:
            self.whitelist_keywords = []

class PreferenceFilter:
    """偏好打分过滤器"""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        self.config = TREE_THINKING_CONFIG
        
        # 默认偏好配置
        self.default_preferences = [
            UserPreference(
                name="深度思考偏好",
                description="偏好非简化版的深度思考方案",
                weight=1.2,
                prefer_complex=True,
                whitelist_keywords=["深入", "详细", "全面", "深度", "多角度"]
            ),
            UserPreference(
                name="推理完善偏好",
                description="偏好推理逻辑完善的方案",
                weight=1.1,
                prefer_reasoning=True,
                whitelist_keywords=["因为", "所以", "推理", "逻辑", "证明", "推导"]
            ),
            UserPreference(
                name="记忆调用偏好",
                description="偏好更多调用记忆模块的方案",
                weight=1.0,
                prefer_memory=True,
                whitelist_keywords=["回忆", "记得", "之前", "历史", "经验", "学习"]
            ),
            UserPreference(
                name="创新思维偏好",
                description="偏好创新性思考方案",
                weight=0.9,
                prefer_innovation=True,
                whitelist_keywords=["创新", "新颖", "独特", "原创", "突破", "创造"]
            ),
            UserPreference(
                name="实用导向偏好",
                description="偏好具有实际应用价值的方案",
                weight=1.0,
                prefer_practical=True,
                whitelist_keywords=["实用", "应用", "实践", "操作", "具体", "可行"]
            )
        ]
        
        self.user_preferences = self.default_preferences.copy()
        print("[TreeThinkingEngine] ⭐ 偏好打分系统初始化完成")
    
    def update_preferences(self, new_preferences: List[UserPreference]):
        """更新用户偏好配置"""
        self.user_preferences = new_preferences
        logger.info(f"更新用户偏好配置: {len(new_preferences)}个偏好项")
    
    async def score_thinking_nodes(self, nodes: List[ThinkingNode]) -> Dict[str, float]:
        """
        对思考节点进行偏好打分
        返回: {node_id: score}
        """
        if not nodes:
            return {}
        
        try:
            # 批量评分
            node_scores = {}
            
            # 基础评分
            for node in nodes:
                base_score = self._calculate_base_score(node)
                node_scores[node.id] = base_score
            
            # AI深度评分（可选）
            if self.api_client and len(nodes) > 1:
                ai_scores = await self._ai_batch_scoring(nodes)
                
                # 合并AI评分
                for node_id, ai_score in ai_scores.items():
                    if node_id in node_scores:
                        # 加权平均
                        node_scores[node_id] = (
                            node_scores[node_id] * 0.7 + ai_score * 0.3
                        )
            
            logger.info(f"完成{len(nodes)}个节点的偏好打分")
            return node_scores
            
        except Exception as e:
            logger.error(f"偏好打分失败: {e}")
            # 返回默认均等分数
            return {node.id: 3.0 for node in nodes}
    
    def _calculate_base_score(self, node: ThinkingNode) -> float:
        """计算节点基础偏好分数"""
        total_score = 0.0
        total_weight = 0.0
        
        content = node.content.lower()
        
        for pref in self.user_preferences:
            if not pref.enabled:
                continue
            
            pref_score = 0.0
            
            # 黑名单检查（减分）
            blacklist_penalty = 0
            for keyword in pref.blacklist_keywords:
                if keyword.lower() in content:
                    blacklist_penalty += 0.5
            
            # 白名单检查（加分）
            whitelist_bonus = 0
            for keyword in pref.whitelist_keywords:
                if keyword.lower() in content:
                    whitelist_bonus += 0.5
            
            # 复杂度偏好
            if pref.prefer_complex:
                complexity = self._assess_content_complexity(content)
                pref_score += complexity * 0.3
            
            # 推理完善偏好
            if pref.prefer_reasoning:
                reasoning_quality = self._assess_reasoning_quality(content)
                pref_score += reasoning_quality * 0.3
            
            # 记忆调用偏好
            if pref.prefer_memory:
                memory_usage = self._assess_memory_usage(content)
                pref_score += memory_usage * 0.2
            
            # 创新性偏好
            if pref.prefer_innovation:
                innovation_level = self._assess_innovation(content)
                pref_score += innovation_level * 0.2
            
            # 实用性偏好
            if pref.prefer_practical:
                practical_value = self._assess_practical_value(content)
                pref_score += practical_value * 0.2
            
            # 应用白名单奖励和黑名单惩罚
            pref_score += whitelist_bonus - blacklist_penalty
            
            # 确保分数在合理范围内
            pref_score = max(0, min(5, pref_score))
            
            # 加权累计
            total_score += pref_score * pref.weight
            total_weight += pref.weight
        
        # 计算最终分数
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 3.0  # 默认中等分数
        
        return round(final_score, 2)
    
    def _assess_content_complexity(self, content: str) -> float:
        """评估内容复杂度"""
        # 长度因子
        length_factor = min(len(content) / 200, 1.0)
        
        # 词汇复杂度
        complex_words = ["分析", "评估", "综合", "推导", "验证", "优化"]
        complexity_count = sum(1 for word in complex_words if word in content)
        complexity_factor = min(complexity_count / 3, 1.0)
        
        # 句式复杂度
        punctuation_count = content.count('，') + content.count('。') + content.count('；')
        structure_factor = min(punctuation_count / 5, 1.0)
        
        return (length_factor + complexity_factor + structure_factor) / 3 * 5
    
    def _assess_reasoning_quality(self, content: str) -> float:
        """评估推理质量"""
        reasoning_indicators = [
            "因为", "所以", "由于", "因此", "导致", "基于", "根据", 
            "推导", "证明", "说明", "表明", "可见", "可以得出"
        ]
        
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in content)
        return min(reasoning_count / 3, 1.0) * 5
    
    def _assess_memory_usage(self, content: str) -> float:
        """评估记忆使用程度"""
        memory_indicators = [
            "记得", "回忆", "之前", "以前", "历史", "经验", 
            "学过", "见过", "遇到", "类似", "相关"
        ]
        
        memory_count = sum(1 for indicator in memory_indicators if indicator in content)
        return min(memory_count / 2, 1.0) * 5
    
    def _assess_innovation(self, content: str) -> float:
        """评估创新程度"""
        innovation_indicators = [
            "创新", "新颖", "独特", "原创", "突破", "创造", 
            "不同", "另辟蹊径", "新思路", "改进", "优化"
        ]
        
        innovation_count = sum(1 for indicator in innovation_indicators if indicator in content)
        return min(innovation_count / 2, 1.0) * 5
    
    def _assess_practical_value(self, content: str) -> float:
        """评估实用价值"""
        practical_indicators = [
            "实用", "应用", "实践", "操作", "具体", "可行", 
            "方法", "步骤", "实施", "执行", "效果", "结果"
        ]
        
        practical_count = sum(1 for indicator in practical_indicators if indicator in content)
        return min(practical_count / 3, 1.0) * 5
    
    async def _ai_batch_scoring(self, nodes: List[ThinkingNode]) -> Dict[str, float]:
        """AI批量评分"""
        if not self.api_client or not nodes:
            return {}
        
        try:
            # 构建评分提示
            nodes_text = ""
            for i, node in enumerate(nodes, 1):
                nodes_text += f"\n方案{i} (ID: {node.id}):\n{node.content}\n"
            
            # 生成偏好描述
            preferences_text = ""
            for pref in self.user_preferences:
                if pref.enabled:
                    preferences_text += f"- {pref.name}: {pref.description} (权重: {pref.weight})\n"
            
            prompt = f"""
请根据用户偏好对以下思考方案进行评分（1-5分）：

用户偏好：
{preferences_text}

思考方案：
{nodes_text}

评分标准：
- 5分：完全符合用户偏好，质量优秀
- 4分：基本符合偏好，质量良好
- 3分：部分符合偏好，质量一般
- 2分：不太符合偏好，质量较差
- 1分：不符合偏好，质量很差

请返回JSON格式评分结果：
{{
    "scores": [
        {{"node_id": "节点ID", "score": 分数, "reason": "评分理由"}},
        ...
    ]
}}
"""
            
            response = await self.api_client.get_response(prompt, temperature=0.3)
            
            # 解析AI评分
            scores = {}
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                for score_data in result.get("scores", []):
                    node_id = score_data.get("node_id")
                    score = float(score_data.get("score", 3))
                    scores[node_id] = score
            
            logger.info(f"AI批量评分完成: {len(scores)}个节点")
            return scores
            
        except Exception as e:
            logger.warning(f"AI批量评分失败: {e}")
            return {}
    
    def get_top_nodes(self, node_scores: Dict[str, float], 
                     nodes: List[ThinkingNode], count: int) -> List[ThinkingNode]:
        """获取评分最高的节点"""
        # 按评分排序
        sorted_items = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 获取对应的节点
        node_dict = {node.id: node for node in nodes}
        top_nodes = []
        
        for node_id, score in sorted_items[:count]:
            if node_id in node_dict:
                node = node_dict[node_id]
                node.score = score  # 更新节点分数
                top_nodes.append(node)
        
        return top_nodes
    
    def get_preference_summary(self) -> Dict:
        """获取偏好配置摘要"""
        summary = {
            "total_preferences": len(self.user_preferences),
            "enabled_preferences": len([p for p in self.user_preferences if p.enabled]),
            "preferences": []
        }
        
        for pref in self.user_preferences:
            summary["preferences"].append({
                "name": pref.name,
                "weight": pref.weight,
                "enabled": pref.enabled,
                "whitelist_count": len(pref.whitelist_keywords),
                "blacklist_count": len(pref.blacklist_keywords)
            })
        
        return summary 