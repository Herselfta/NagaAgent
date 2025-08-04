"""
问题难度判断器
分析问题复杂度并决定思考路线数量
"""

import re
import logging
from typing import Dict, List, Tuple
from .config import TREE_THINKING_CONFIG, COMPLEX_KEYWORDS, BRANCH_TYPES

logger = logging.getLogger("DifficultyJudge")

class DifficultyJudge:
    """问题难度判断器"""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        self.config = TREE_THINKING_CONFIG
        self.complex_keywords = COMPLEX_KEYWORDS
        
        # 难度评估权重
        self.weights = {
            "length": 0.15,           # 文本长度
            "keywords": 0.25,         # 关键词匹配
            "sentence_structure": 0.20, # 句式复杂度
            "question_type": 0.25,    # 问题类型
            "ai_assessment": 0.15     # AI深度评估
        }
        
        print("[TreeThinkingEngine] 🎯 问题难度判断器初始化完成")
    
    async def assess_difficulty(self, question: str) -> Dict:
        """评估问题难度"""
        try:
            # 基础指标计算
            text_metrics = self._analyze_text_metrics(question)
            keyword_metrics = self._analyze_keywords(question)
            structure_metrics = self._analyze_structure(question)
            
            # AI深度评估（优先使用快速模型）
            ai_metrics = await self._ai_deep_assessment(question)
            
            # 综合评分
            final_score = self._calculate_final_score(
                question, text_metrics, keyword_metrics, structure_metrics, ai_metrics
            )
            
            difficulty = min(5, max(1, round(final_score)))
            routes = self.config["difficulty_routes"][difficulty]
            
            # 生成推理说明
            reasoning = self._generate_reasoning(
                difficulty, text_metrics, keyword_metrics, structure_metrics, ai_metrics
            )
            
            logger.info(f"问题难度评估完成: 难度{difficulty}/5, {routes}条思考路线")
            
            return {
                "difficulty": difficulty,
                "routes": routes,
                "reasoning": reasoning,
                "metrics": {
                    "text": text_metrics,
                    "keywords": keyword_metrics,
                    "structure": structure_metrics,
                    "ai_assessment": ai_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"难度评估失败: {e}")
            # 返回默认难度
            return {
                "difficulty": 3,
                "routes": 5,
                "reasoning": f"难度评估失败，使用默认值: {str(e)}",
                "metrics": {}
            }
    
    def _analyze_text_metrics(self, question: str) -> float:
        """分析文本长度复杂度"""
        length = len(question)
        
        if length < 20:
            return 1.0  # 很简单
        elif length < 50:
            return 2.0  # 简单
        elif length < 100:
            return 3.0  # 中等
        elif length < 200:
            return 4.0  # 复杂
        else:
            return 5.0  # 很复杂
    
    def _analyze_keywords(self, question: str) -> float:
        """分析关键词复杂度"""
        detected_keywords = self._extract_keywords(question)
        keyword_count = len(detected_keywords)
        
        # 根据关键词数量评分
        if keyword_count == 0:
            return 1.0
        elif keyword_count <= 2:
            return 2.5
        elif keyword_count <= 4:
            return 3.5
        elif keyword_count <= 6:
            return 4.5
        else:
            return 5.0
    
    def _extract_keywords(self, question: str) -> List[str]:
        """提取问题中的复杂关键词"""
        detected = []
        for keyword in self.complex_keywords:
            if keyword in question:
                detected.append(keyword)
        return detected
    
    def _analyze_structure(self, question: str) -> float:
        """分析句式结构复杂度"""
        # 检查标点符号复杂度
        comma_count = question.count(',') + question.count('，')
        semicolon_count = question.count(';') + question.count('；')
        question_marks = question.count('?') + question.count('？')
        
        # 检查连接词
        connectives = ['然而', '但是', '因此', '所以', '由于', '如果', '虽然', '尽管']
        connective_count = sum(1 for conn in connectives if conn in question)
        
        # 计算复杂度分数
        complexity = (comma_count * 0.5 + semicolon_count * 1.0 + 
                     question_marks * 0.5 + connective_count * 1.0)
        
        if complexity < 1:
            return 1.5
        elif complexity < 3:
            return 2.5
        elif complexity < 5:
            return 3.5
        elif complexity < 8:
            return 4.5
        else:
            return 5.0
    
    def _assess_question_type(self, question: str) -> float:
        """评估问题类型复杂度"""
        # 不同类型问题的复杂度映射
        type_patterns = {
            r'什么|是什么|怎么样': 1.5,      # 基础事实类
            r'如何|怎么做|方法': 3.0,        # 方法指导类
            r'为什么|原因|分析': 3.5,        # 分析解释类
            r'比较|对比|区别': 4.0,          # 比较评估类
            r'设计|优化|改进|方案': 4.5,     # 设计优化类
            r'评估|判断|选择|决策': 4.5,     # 决策判断类
            r'创新|创造|发明': 5.0           # 创新创造类
        }
        
        max_score = 1.0
        for pattern, score in type_patterns.items():
            if re.search(pattern, question):
                max_score = max(max_score, score)
        
        return max_score
    
    async def _ai_deep_assessment(self, question: str) -> Dict:
        """AI深度评估问题复杂度"""
        if not self.api_client:
            return {"score": 0, "reasoning": ""}
        
        prompt = f"""
请评估以下问题的复杂度（1-5分，1最简单，5最复杂）：

问题：{question}

评估维度：
1. 认知负荷：需要多少背景知识
2. 推理深度：需要多少层推理
3. 创新程度：需要多少创新思考
4. 综合性：涉及多少个领域

请返回JSON格式：
{{
    "score": 数字(1-5),
    "reasoning": "评估理由"
}}
"""
        
        try:
            response = await self.api_client.get_response(prompt, temperature=0.3)
            
            # 解析响应
            import json
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                return {
                    "score": float(result.get("score", 3)),
                    "reasoning": result.get("reasoning", "")
                }
            else:
                return {"score": 3, "reasoning": "AI评估格式错误"}
                
        except Exception as e:
            logger.warning(f"AI评估解析失败: {e}")
            return {"score": 3, "reasoning": f"AI评估异常: {str(e)}"}
    
    def _calculate_final_score(self, question: str, text_metrics: float, keyword_metrics: float, 
                             structure_metrics: float, ai_metrics: Dict) -> float:
        """计算综合评分"""
        base_score = (
            text_metrics * self.weights["length"] +
            keyword_metrics * self.weights["keywords"] +
            structure_metrics * self.weights["sentence_structure"] +
            self._assess_question_type(question) * self.weights["question_type"]
        )
        
        final_score = base_score + ai_metrics["score"] * self.weights["ai_assessment"]
        return final_score
    
    def _generate_reasoning(self, difficulty: int, text_metrics: float, 
                          keyword_metrics: float, structure_metrics: float, 
                          ai_metrics: Dict) -> str:
        """生成评估推理过程"""
        difficulty_names = {
            1: "简单", 2: "基础", 3: "中等", 4: "复杂", 5: "极难"
        }
        
        reasoning_parts = [
            f"问题难度评估为：{difficulty_names[difficulty]}（{difficulty}/5）"
        ]
        
        # 各维度分析
        if text_metrics >= 4:
            reasoning_parts.append(f"文本长度较长，增加理解难度")
        if keyword_metrics >= 4:
            reasoning_parts.append(f"包含多个复杂关键词")
        if structure_metrics >= 4:
            reasoning_parts.append(f"句式结构复杂，逻辑关系多层")
        
        if ai_metrics["reasoning"]:
            reasoning_parts.append(f"AI深度分析：{ai_metrics['reasoning']}")
        
        return "；".join(reasoning_parts)
    
    def get_temperature_distribution(self, routes: int) -> List[float]:
        """为不同思考路线分配温度值"""
        temp_config = self.config["temperature_range"]
        min_temp = temp_config["min"]
        max_temp = temp_config["max"]
        
        if routes <= 1:
            return [temp_config["default"]]
        
        # 均匀分布温度值
        temperatures = []
        for i in range(routes):
            # 线性插值
            ratio = i / (routes - 1)
            temp = min_temp + ratio * (max_temp - min_temp)
            temperatures.append(round(temp, 2))
        
        return temperatures
    
    def get_branch_types(self, routes: int) -> List[str]:
        """为不同思考路线分配分支类型"""
        branch_keys = list(BRANCH_TYPES.keys())
        
        # 如果路线数少于分支类型数，循环使用
        types = []
        for i in range(routes):
            types.append(branch_keys[i % len(branch_keys)])
        
        return types 