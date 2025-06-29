"""
树状外置思考系统
高级推理增强机制，支持多分支并行思考、偏好打分和遗传剪枝
"""

from .tree_thinking import TreeThinkingEngine
from .thinking_node import ThinkingNode, ThinkingBranch
from .difficulty_judge import DifficultyJudge
from .preference_filter import PreferenceFilter, UserPreference
from .genetic_pruning import GeneticPruning
from .thread_pools import ThreadPoolManager

__all__ = [
    'TreeThinkingEngine',
    'ThinkingNode', 
    'ThinkingBranch',
    'DifficultyJudge',
    'PreferenceFilter',
    'UserPreference',
    'GeneticPruning',
    'ThreadPoolManager'
]

__version__ = "1.0.0" 