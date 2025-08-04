"""
树状思考系统配置
"""

# 树状思考系统配置
TREE_THINKING_CONFIG = {
    "enabled": True,
    "max_thinking_routes": 10,
    "min_thinking_routes": 2,
    "max_thinking_depth": 3,
    "thinking_timeout": 60,  # 秒
    "api_timeout": 30,       # 秒
    
    # 线程池配置
    "thinking_pool_size": 8,
    "api_pool_size": 4,
    "max_concurrent_api": 3,
    "min_api_interval": 0.5,
    
    # 遗传算法配置
    "selection_rate": 0.6,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "max_generations": 3,
    
    # 评分权重
    "scoring_weights": {
        "content_depth": 0.3,
        "reasoning_quality": 0.25,
        "memory_integration": 0.2,
        "innovation_level": 0.15,
        "practical_value": 0.1
    },
    
    # 难度映射
    "difficulty_routes": {
        1: 2,  # 简单问题：2条思考路线
        2: 3,  # 基础问题：3条思考路线
        3: 5,  # 中等问题：5条思考路线
        4: 7,  # 复杂问题：7条思考路线
        5: 10  # 极难问题：10条思考路线
    },
    
    # 温度范围配置
    "temperature_range": {
        "min": 0.3,    # 最保守的思考
        "max": 1.2,    # 最创新的思考
        "default": 0.7 # 默认温度
    }
}

# 复杂问题关键词
COMPLEX_KEYWORDS = [
    "分析", "设计", "比较", "评估", "优化", "思考", "创新", "研究", 
    "解决", "策略", "方案", "系统", "架构", "深入", "全面", "复杂",
    "介绍", "详细", "深度", "原理", "机制", "过程", "理论", "算法",
    "探讨", "讨论", "阐述", "说明", "解释", "论述", "综述", "总结",
    "猜想", "证明", "推导", "计算", "建模", "仿真", "预测", "预估"
]

# 分支类型定义
BRANCH_TYPES = {
    "logical": "逻辑分析型",
    "creative": "创新思维型", 
    "analytical": "深度解析型",
    "practical": "实用导向型",
    "philosophical": "哲学思辨型"
} 