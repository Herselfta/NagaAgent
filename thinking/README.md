
---
thinking/
├── init.py # 包初始化与核心类导出
├── config.py # 核心参数与分支类型配置
├── difficulty_judge.py # 问题难度自动评估
├── genetic_pruning.py # 遗传算法剪枝与进化
├── preference_filter.py # 用户偏好打分与筛选
├── thinking_node.py # 思考节点与分支数据结构
├── thread_pools.py # 并发线程池管理
├── tree_thinking.py # 核心引擎，协调各子系统

## 🚀 功能简介

- **多分支并行思考**  
  根据问题难度，自动生成多条不同类型的思考路线，支持逻辑、创新、分析、实用、哲学等多种分支。

- **问题难度自动评估**  
  通过文本长度、关键词、句式结构、AI评估等多维度综合判断问题复杂度，动态调整思考路线数量与深度。

- **用户偏好打分系统**  
  支持自定义偏好（如创新、实用、记忆调用等），对每条思考路线进行加权评分，筛选最优答案。

- **遗传算法剪枝**  
  对多条思考路线进行适应度评估、交叉融合、精英保留等进化操作，自动保留最优解。

- **线程池并发调度**  
  思考与API调用分离，提升并发性能，支持批量任务高效执行。

- **可扩展性强**  
  支持自定义分支类型、评分权重、进化策略等，便于二次开发和集成。

---

## 🛠️ 主要类与模块说明

- `TreeThinkingEngine`  
  核心引擎，负责整体流程调度、思考路线生成、评分、剪枝与答案综合。

- `DifficultyJudge`  
  问题难度评估器，自动决定思考路线数量与分支类型。

- `PreferenceFilter`  
  用户偏好打分系统，支持多维度自定义偏好配置。

- `GeneticPruning`  
  遗传算法剪枝器，自动进化并筛选最优思考方案。

- `ThinkingNode`/`ThinkingBranch`  
  思考节点与分支的数据结构，支持家族关系、分数、温度等属性。

- `ThreadPoolManager`  
  双线程池管理，分别调度思考与API调用任务，提升并发效率。

---

## ⚙️ 快速集成示例

```python
from thinking import TreeThinkingEngine

# 初始化引擎（可传入自定义API客户端和记忆管理器）
engine = TreeThinkingEngine(api_client=your_api_client, memory_manager=your_memory_manager)

# 异步调用深度思考
result = await engine.think_deeply("请分析人工智能未来的发展趋势")

print(result["answer"])  # 输出综合后的最终答案
```

---

## 📝 配置说明

- 所有核心参数均在`config.py`中集中管理，包括分支数量、温度范围、评分权重、分支类型等。
- 支持通过`TREE_THINKING_CONFIG`灵活调整系统行为。

---

## 💡 适用场景

- 复杂问题的多角度分析与决策建议
- 需要融合创新、逻辑、实用等多维度思考的场景
- 高级AI助手、科研、教育、咨询等领域

---

## 📢 备注

- 本系统为异步设计，推荐在支持async/await的环境下使用。
- 可与Naga主流程、夏园记忆系统等模块无缝集成，提升整体智能与推理能力。

---

如需详细API文档或二次开发支持，请查阅各模块源码及注释。