# ⚡ NagaAgent 快速响应小模型指南

## 功能概述

NagaAgent 2.3 支持集成快速响应小模型，用于加速简单任务的处理。提供以下核心功能：

- **快速决策**：支持8种专用决策类型，每种都有专门的prompt和格式验证
- **JSON格式化**：将文本内容转换为结构化JSON格式
- **问题难度判断**：使用小模型智能评估问题复杂程度
- **黑白名单打分系统**：根据用户偏好对思考结果进行1-5分评分和剪枝
- **思考完整性判断**：评估单次思考是否完整，并生成下一级问题
- **输出过滤**：自动过滤AI输出中的思考标签，确保结果纯净

当小模型未配置或调用失败时，系统会自动降级到主模型，确保功能正常运行。

## 🚀 快速开始

### 1. 环境配置

#### 方法一：环境变量配置
```bash
export QUICK_MODEL_ENABLED=true
export QUICK_MODEL_API_KEY="your_api_key"
export QUICK_MODEL_BASE_URL="http://localhost:8000/v1"
export QUICK_MODEL_NAME="qwen2.5-1.5b-instruct"
```

#### 方法二：.env文件配置
```env
QUICK_MODEL_ENABLED=true
QUICK_MODEL_API_KEY=your_api_key
QUICK_MODEL_BASE_URL=http://localhost:8000/v1
QUICK_MODEL_NAME=qwen2.5-1.5b-instruct
```

#### 方法三：对话中动态配置
```
#quick config your_api_key http://localhost:8000/v1 qwen2.5-1.5b-instruct
```

### 2. 状态检查
```
#quick status
```

### 3. 功能测试
```
#quick test
```

## 📋 控制命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `#quick` | 显示帮助信息 | `#quick` |
| `#quick status` | 查看模型状态和统计 | `#quick status` |
| `#quick config` | 配置模型参数 | `#quick config sk-xxx http://localhost:8000/v1` |
| `#quick test` | 测试模型功能 | `#quick test` |
| `#quick enable` | 启用快速模型 | `#quick enable` |
| `#quick disable` | 禁用快速模型 | `#quick disable` |

## 🔧 支持的小模型

### 本地部署模型

#### 1. Qwen2.5系列
```bash
# 使用ollama部署
ollama run qwen2.5:1.5b-instruct

# 配置
QUICK_MODEL_BASE_URL=http://localhost:11434/v1
QUICK_MODEL_NAME=qwen2.5:1.5b-instruct
```

#### 2. Llama3.2系列
```bash
# 使用ollama部署
ollama run llama3.2:1b

# 配置
QUICK_MODEL_BASE_URL=http://localhost:11434/v1
QUICK_MODEL_NAME=llama3.2:1b
```

### 云端API

#### 1. 阿里云通义千问
```env
QUICK_MODEL_API_KEY=your_dashscope_api_key
QUICK_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QUICK_MODEL_NAME=qwen-turbo
```

#### 2. 智谱AI GLM
```env
QUICK_MODEL_API_KEY=your_zhipuai_api_key
QUICK_MODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4
QUICK_MODEL_NAME=glm-4-flash
```

#### 3. 字节跳动豆包
```env
QUICK_MODEL_API_KEY=your_doubao_api_key
QUICK_MODEL_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
QUICK_MODEL_NAME=doubao-lite-4k
```

## 🎨 功能详解

### 🔍 输出过滤功能

自动过滤AI输出中的思考标签，确保用户只看到最终结果：

```python
# 配置示例
OUTPUT_FILTER_CONFIG = {
    "filter_think_tags": True,  # 启用过滤
    "filter_patterns": [
        r'<think>.*?</think>',      # 思考标签
        r'<thinking>.*?</thinking>', # 思考标签  
        r'<reflection>.*?</reflection>', # 反思标签
        r'<internal>.*?</internal>'  # 内部思考标签
    ],
    "clean_output": True  # 清理多余空白字符
}
```

**过滤效果：**
- 输入：`<think>思考过程</think>最终答案`
- 输出：`最终答案`

### 🎯 问题难度判断

使用小模型智能评估问题复杂程度，避免依赖简单的句式判断：

```python
difficulty_result = await quick_model.judge_difficulty(
    question="设计一个分布式机器学习系统",
    context="需要处理PB级数据，支持实时训练"
)
# 输出：{"difficulty": "极难", "model_used": "quick"}
```

**难度等级：**
- **简单**：基础概念、简单操作
- **中等**：需要一定技能和经验
- **困难**：复杂推理、高级技术
- **极难**：前沿技术、创新要求

### ⭐ 黑白名单打分系统

根据用户偏好对思考结果进行1-5分评分，实现有效剪枝：

```python
# 用户设置偏好（最多3个）
user_preferences = ["逻辑清晰准确", "实用性强", "创新思维"]

# 对结果列表进行评分
scored_results = await quick_model.score_results(
    results=thinking_results,
    user_preferences=user_preferences
)
```

**评分机制：**
- **5分**：完全符合用户偏好，质量极高
- **4分**：很好符合偏好，质量良好  
- **3分**：基本符合偏好，质量一般
- **2分**：部分符合偏好，质量较差
- **1分**：不符合偏好或质量很差

**剪枝策略：**
- 默认保留3分以上结果
- 相似度>85%的结果中，保留最高分，其他降为1分
- **宽松模式**（`strict_filtering: False`）：保证至少保留2个结果（即使低于阈值）
- **严格模式**（`strict_filtering: True`）：严格按阈值过滤，宁可返回空结果也不保留低分结果

**结构化输出示例：**
```json
{
  "score": 4,
  "original_score": 4,
  "similarity_penalty": 0,
  "content": "思考内容...",
  "score_details": {
    "preference_match": 4.0,
    "content_quality": 3.5,
    "originality": 4.0,
    "feasibility": 4.5
  }
}
```

### 🧠 思考完整性判断

评估单次思考是否完整，为不完整的思考生成下一级问题：

```python
completeness_result = await quick_model.check_thinking_completeness(
    thinking_content="需要优化数据库查询，减少IO操作",
    question="如何提高网站性能？"
)
```

**判断标准：**
- 问题分析是否充分
- 解决方案是否明确
- 逻辑链条是否完整
- 结论是否清晰合理

**输出示例：**
```json
{
  "is_complete": false,
  "next_question": "具体应该优化哪些类型的数据库查询？",
  "model_used": "quick",
  "response_time": 1.234
}
```

### 快速决策功能

快速决策功能现已支持8种专用决策类型，每种类型都有专门优化的prompt和输出格式验证：

#### 1. 二元判断 (binary)
专门用于是/否判断，输出严格限制为"是"或"否"
```python
decision_result = await quick_model.quick_decision(
    "今天天气好吗？",
    context="当前天气：晴天，温度25度",
    decision_type="binary"
)
# 输出：是 / 否
```

#### 2. 分类判断 (category)
用于内容分类，输出简洁的分类名称
```python
decision_result = await quick_model.quick_decision(
    "这是什么类型的问题？",
    context="用户问：如何学习Python？",
    decision_type="category"
)
# 输出：技术问题 / 学习问题 / 编程问题等
```

#### 3. 数值评分 (score)
用于1-10的数值评分，自动验证和标准化
```python
decision_result = await quick_model.quick_decision(
    "这个回答的质量如何？",
    context="用户回答了一个详细的技术问题",
    decision_type="score"
)
# 输出：1-10的数字
```

#### 4. 优先级判断 (priority)
用于任务优先级判断，输出标准化为高/中/低
```python
decision_result = await quick_model.quick_decision(
    "修复系统安全漏洞",
    context="发现了一个可能导致数据泄露的安全漏洞",
    decision_type="priority"
)
# 输出：高 / 中 / 低
```

#### 5. 情感分析 (sentiment)
用于文本情感倾向分析，输出积极/消极/中性
```python
decision_result = await quick_model.quick_decision(
    "分析这条评论的情感",
    context="评论：这个产品真的很棒，用起来很舒服！",
    decision_type="sentiment"
)
# 输出：积极 / 消极 / 中性
```

#### 6. 紧急度判断 (urgency)
用于事件紧急程度判断，输出紧急/普通/不紧急
```python
decision_result = await quick_model.quick_decision(
    "处理客户投诉",
    context="客户反馈产品有轻微瑕疵，但不影响使用",
    decision_type="urgency"
)
# 输出：紧急 / 普通 / 不紧急
```

#### 7. 复杂度评估 (complexity)
用于任务复杂程度评估，输出简单/中等/复杂
```python
decision_result = await quick_model.quick_decision(
    "实现一个简单的计算器",
    context="需要支持加减乘除四种基本运算",
    decision_type="complexity"
)
# 输出：简单 / 中等 / 复杂
```

#### 8. 自定义判断 (custom)
用于其他类型的自定义判断，输出格式灵活
```python
decision_result = await quick_model.quick_decision(
    "这个问题应该怎么解决？",
    context="遇到了一个技术难题",
    decision_type="custom"
)
# 输出：自定义格式的建议
```

### 输出格式验证

每种决策类型都有专门的输出验证和标准化处理：

- **二元判断**：自动识别"是/否"相关表达，标准化为"是"或"否"
- **数值评分**：自动提取数字，限制在1-10范围内
- **分类输出**：验证输出的合理性和一致性
- **标准选项**：自动映射到预定义的标准选项

### JSON格式化功能

将文本转换为结构化数据：

#### 自动格式化
```python
json_result = await quick_model.format_json(
    "用户名：张三，年龄：25岁，职业：程序员",
    format_type="auto"
)
```

#### 结构化格式
```python
json_result = await quick_model.format_json(
    "这是一篇关于AI的文章内容...",
    format_type="structured"
)
# 输出包含title, content, summary, keywords
```

#### 自定义模式
```python
schema = {
    "name": "姓名",
    "age": "年龄",
    "skills": ["技能列表"]
}

json_result = await quick_model.format_json(
    "张三，25岁，会Python和Java",
    schema=schema
)
```

## 🔧 决策类型选择指南

| 场景 | 推荐类型 | 输出格式 | 使用场景 |
|------|----------|----------|----------|
| 是否判断 | binary | 是/否 | 需要确认、验证、布尔判断 |
| 内容分类 | category | 分类名称 | 文档分类、问题归类 |
| 质量评估 | score | 1-10数字 | 评分、打分、质量评估 |
| 任务排序 | priority | 高/中/低 | 任务管理、重要性排序 |
| 情感识别 | sentiment | 积极/消极/中性 | 评论分析、情感监控 |
| 时间安排 | urgency | 紧急/普通/不紧急 | 事件处理、时间安排 |
| 难度评估 | complexity | 简单/中等/复杂 | 项目评估、学习规划 |
| 灵活判断 | custom | 自定义 | 其他特殊需求 |

## 📊 性能监控

系统会自动记录以下统计信息：

- **总调用次数**：快速模型和降级调用的总和
- **快速模型成功率**：快速模型成功调用的比例  
- **快速模型使用率**：使用快速模型的比例
- **节省时间**：相比大模型节省的总时间
- **响应时间**：每次调用的实际响应时间
- **格式验证**：输出格式的正确性统计
- **新功能统计**：难度判断、评分操作、完整性检查、输出过滤次数

查看统计信息：
```
#quick status
```

## ⚙️ 降级机制

### 自动降级条件

1. **配置问题**：API密钥或地址未配置
2. **连接超时**：超过5秒未响应
3. **调用失败**：API返回错误
4. **格式无效**：输出格式不符合预期要求

### 格式验证降级

当快速模型输出格式不正确时：
- **二元判断**：无法识别是/否时标记为"(格式异常)"
- **数值评分**：无法提取数字时使用默认值"5(默认值)"
- **标准选项**：无法匹配标准选项时使用智能映射
- **自动修正**：系统会尝试自动修正常见格式问题

### 降级流程

```
快速模型调用 → 失败检测 → 自动切换大模型 → 返回结果
```

### 重试机制

- 最大重试次数：2次
- 重试间隔：立即重试
- 失败记录：统计失败次数和原因

## 🔒 安全配置

### API密钥安全
- 支持环境变量配置，避免硬编码
- 状态显示时只显示密钥前8位
- 支持动态更新配置

### 超时保护
- 默认超时时间：5秒
- 防止长时间等待
- 自动降级处理

### 输入验证
- JSON格式验证
- 结果有效性检查
- 错误异常处理
- 输出内容过滤

## 🎯 最佳实践

### 1. 决策类型选择
- **明确需求**：根据实际需要选择最合适的决策类型
- **格式一致**：同类任务使用相同的决策类型确保一致性
- **降级兼容**：设计时考虑降级场景的处理

### 2. Prompt设计
- **简洁明确**：问题描述清晰，避免歧义
- **上下文完整**：提供足够的上下文信息
- **类型匹配**：确保问题类型与决策类型匹配

### 3. 输出处理
- **格式验证**：检查返回的decision字段格式
- **异常处理**：处理格式异常和降级情况
- **结果使用**：基于标准化后的结果进行后续逻辑

### 4. 温度控制
- **极低温度**：使用0.05确保输出稳定一致
- **避免随机性**：小模型主要用于确定性任务
- **结果可重现**：相同输入应得到相同输出

### 5. 打分系统使用
- **偏好设置**：选择1-3个明确的用户偏好
- **阈值调整**：根据需要调整分数保留阈值
- **相似性检测**：利用相似性惩罚去重

## 🐛 故障排查

### 常见问题

#### 1. 配置未生效
```bash
# 检查环境变量
echo $QUICK_MODEL_ENABLED
echo $QUICK_MODEL_API_KEY

# 重新配置
#quick config your_api_key http://localhost:8000/v1
```

#### 2. 连接超时
```bash
# 检查网络连接
curl http://localhost:8000/v1/models

# 调整超时时间（代码级别配置）
```

#### 3. JSON格式错误
- 检查小模型输出是否符合JSON格式
- 使用更稳定的模型或调整温度参数
- 依赖大模型降级处理

#### 4. 输出过滤问题
- 检查过滤模式配置
- 验证正则表达式有效性
- 查看过滤统计信息

### 日志检查
```bash
# 查看快速模型相关日志
grep "QuickModelManager" logs/

# 查看错误日志
grep "ERROR.*quick" logs/
```

## 📈 性能对比

| 决策类型 | 大模型 | 小模型 | 提升倍数 | 格式准确率 |
|----------|--------|--------|----------|------------|
| binary | 3-8秒 | 0.5-2秒 | 2-4倍 | 95%+ |
| category | 4-10秒 | 1-3秒 | 3-4倍 | 90%+ |
| score | 3-6秒 | 0.8-2秒 | 2-3倍 | 98%+ |
| priority | 2-5秒 | 0.5-1.5秒 | 3-4倍 | 95%+ |
| sentiment | 3-7秒 | 1-2秒 | 2-3倍 | 92%+ |
| urgency | 2-6秒 | 0.8-2秒 | 2-4倍 | 94%+ |
| complexity | 4-8秒 | 1-2.5秒 | 3-4倍 | 90%+ |
| custom | 3-10秒 | 1-3秒 | 2-4倍 | 85%+ |
| difficulty | 5-12秒 | 1-3秒 | 3-5倍 | 92%+ |
| scoring | 8-20秒 | 2-5秒 | 3-6倍 | 88%+ |
| completeness | 4-10秒 | 1-3秒 | 3-4倍 | 90%+ |

*注：格式准确率指输出格式符合预期的比例*

## 🔄 更新日志

### v2.3
- ✅ 新增输出过滤功能，自动清理思考标签
- ✅ 新增问题难度判断器，使用AI智能评估
- ✅ 新增黑白名单打分系统，支持1-5分评分和智能剪枝
- ✅ 新增思考完整性判断，自动生成下一级问题
- ✅ 温度优化至0.05，确保输出稳定一致
- ✅ 增强统计系统，追踪所有新功能使用情况

### v2.2beta
- ✅ 初始版本发布
- ✅ 支持快速决策和JSON格式化
- ✅ 自动降级机制
- ✅ 性能统计监控
- ✅ 动态配置支持

### 后续计划
- 🔄 更多小模型适配
- 🔄 批量处理优化
- 🔄 缓存机制
- 🔄 负载均衡支持
- 🔄 树状思考集成

---

📝 **更多信息请参考 [NagaAgent 主文档](README.md)** 