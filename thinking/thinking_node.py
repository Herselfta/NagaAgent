"""
思考节点数据结构定义
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid

@dataclass
class ThinkingNode:
    """思考节点 - 代表一个思考分支"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    depth: int = 0
    content: str = ""
    temperature: float = 0.7
    score: float = 0.0
    generation: int = 0
    
    # 遗传算法属性
    fitness: float = 0.0
    mutation_rate: float = 0.1
    crossover_points: List[int] = field(default_factory=list)
    
    # 家族关系属性
    children_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    
    # 状态标记
    is_completed: bool = False
    is_selected: bool = False
    timestamp: float = field(default_factory=time.time)
    
    # 扩展属性
    branch_type: str = "logical"
    thinking_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.thinking_path:
            self.thinking_path = [self.id]
        
        # 初始化家族关系元数据
        if "family_tree" not in self.metadata:
            self.metadata["family_tree"] = {
                "generation_index": None,
                "siblings": [],
                "is_crossover_child": False,
                "crossover_parents": [],
                "is_mutation_child": False,
                "mutation_parent": None,
                "branch_lineage": []
            }
        
        # 初始化思考过程元数据
        if "thinking_process" not in self.metadata:
            self.metadata["thinking_process"] = {
                "creation_method": "initial",
                "prompt_used": "",
                "api_temperature": self.temperature,
                "generation_time": time.time(),
                "processing_stats": {}
            }
    
    def get_age_seconds(self) -> float:
        """获取节点年龄（秒）"""
        return time.time() - self.timestamp
    
    def update_content(self, new_content: str):
        """更新内容并标记完成"""
        self.content = new_content
        self.is_completed = True
        self.timestamp = time.time()
    
    def create_child(self, content: str = "", branch_type: str = None) -> 'ThinkingNode':
        """创建子节点"""
        child = ThinkingNode(
            parent_id=self.id,
            depth=self.depth + 1,
            content=content,
            temperature=self.temperature,
            generation=self.generation + 1,
            branch_type=branch_type or self.branch_type,
            thinking_path=self.thinking_path + [str(uuid.uuid4())]
        )
        return child

    def add_child(self, child_id: str):
        """添加子节点"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def add_sibling(self, sibling_id: str):
        """添加兄弟节点"""
        if sibling_id not in self.sibling_ids and sibling_id != self.id:
            self.sibling_ids.append(sibling_id)
    
    def set_family_relationships(self, siblings: List[str], generation_index: int):
        """设置家族关系"""
        self.metadata["family_tree"]["siblings"] = [s for s in siblings if s != self.id]
        self.metadata["family_tree"]["generation_index"] = generation_index
        self.sibling_ids = [s for s in siblings if s != self.id]
    
    def mark_as_crossover_child(self, parent1_id: str, parent2_id: str):
        """标记为交叉子代"""
        self.metadata["family_tree"]["is_crossover_child"] = True
        self.metadata["family_tree"]["crossover_parents"] = [parent1_id, parent2_id]
        self.metadata["thinking_process"]["creation_method"] = "crossover"
    
    def mark_as_mutation_child(self, parent_id: str):
        """标记为变异子代"""
        self.metadata["family_tree"]["is_mutation_child"] = True
        self.metadata["family_tree"]["mutation_parent"] = parent_id
        self.metadata["thinking_process"]["creation_method"] = "mutation"
    
    def get_family_info(self) -> Dict:
        """获取家族信息"""
        return {
            "node_id": self.id,
            "generation": self.generation,
            "generation_index": self.metadata["family_tree"]["generation_index"],
            "siblings_count": len(self.sibling_ids),
            "children_count": len(self.children_ids),
            "is_crossover_child": self.metadata["family_tree"]["is_crossover_child"],
            "is_mutation_child": self.metadata["family_tree"]["is_mutation_child"],
            "creation_method": self.metadata["thinking_process"]["creation_method"],
            "branch_type": self.branch_type
        }

@dataclass
class ThinkingBranch:
    """思考分支 - 包含多个相关节点"""
    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: List[ThinkingNode] = field(default_factory=list)
    total_score: float = 0.0
    avg_temperature: float = 0.7
    branch_type: str = "logical"
    
    # 分支统计
    created_time: float = field(default_factory=time.time)
    completed_nodes: int = 0
    best_score: float = 0.0
    
    def add_node(self, node: ThinkingNode):
        """添加节点到分支"""
        self.nodes.append(node)
        self._update_statistics()
    
    def _update_statistics(self):
        """更新分支统计信息"""
        if not self.nodes:
            return
            
        # 更新完成节点数
        self.completed_nodes = sum(1 for node in self.nodes if node.is_completed)
        
        # 更新总分和最高分
        completed_nodes = [node for node in self.nodes if node.is_completed]
        if completed_nodes:
            self.total_score = sum(node.score for node in completed_nodes)
            self.best_score = max(node.score for node in completed_nodes)
            self.avg_temperature = sum(node.temperature for node in completed_nodes) / len(completed_nodes)
    
    def get_best_node(self) -> Optional[ThinkingNode]:
        """获取分支中评分最高的节点"""
        completed_nodes = [node for node in self.nodes if node.is_completed and node.score > 0]
        if not completed_nodes:
            return None
        return max(completed_nodes, key=lambda x: x.score)
    
    def get_completion_rate(self) -> float:
        """获取完成率"""
        if not self.nodes:
            return 0.0
        return self.completed_nodes / len(self.nodes)
    
    def is_ready_for_scoring(self) -> bool:
        """判断是否准备好进行评分"""
        return self.completed_nodes > 0 and self.get_completion_rate() >= 0.5

@dataclass 
class ThinkingGeneration:
    """思考代数 - 遗传算法中的一代"""
    generation_id: int
    branches: List[ThinkingBranch] = field(default_factory=list)
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity_score: float = 0.0
    created_time: float = field(default_factory=time.time)
    
    def add_branch(self, branch: ThinkingBranch):
        """添加分支到当代"""
        self.branches.append(branch)
        self._update_generation_stats()
    
    def _update_generation_stats(self):
        """更新代数统计"""
        if not self.branches:
            return
            
        # 收集所有已完成节点
        all_nodes = []
        for branch in self.branches:
            all_nodes.extend([node for node in branch.nodes if node.is_completed])
        
        if all_nodes:
            fitness_scores = [node.fitness for node in all_nodes]
            self.best_fitness = max(fitness_scores)
            self.avg_fitness = sum(fitness_scores) / len(fitness_scores)
            
            # 计算多样性分数（温度方差）
            temperatures = [node.temperature for node in all_nodes]
            if len(temperatures) > 1:
                avg_temp = sum(temperatures) / len(temperatures)
                self.diversity_score = sum((t - avg_temp) ** 2 for t in temperatures) / len(temperatures)
    
    def get_top_nodes(self, count: int) -> List[ThinkingNode]:
        """获取当代最优节点"""
        all_nodes = []
        for branch in self.branches:
            all_nodes.extend([node for node in branch.nodes if node.is_completed])
        
        # 按适应度排序
        all_nodes.sort(key=lambda x: x.fitness, reverse=True)
        return all_nodes[:count] 