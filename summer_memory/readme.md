# 🧠 中文五元组知识图谱构建与可视化系统

本项目通过接入 DeepSeek API 实现中文文本的五元组抽取，并借助 Neo4j 构建实体关系图谱，同时使用 PyVis 实现知识图谱的可视化展示与交互式查询。

---

## 📦 项目结构

```
.
├── main.py                 # 主程序入口，负责流程调度、用户交互
├── quintuple_extractor.py  # 使用 DeepSeek API 进行五元组抽取
├── quintuple_graph.py      # 操作 Neo4j，存储与查询五元组
├── quintuple_visualize.py  # 使用 PyVis 生成 graph.html 知识图谱可视化页面
├── quintuple_rag_query.py  # 使用 DeepSeek 提取关键词并在图谱中检索答案
├── quintuples.json         # 持久化的五元组缓存文件
├── graph.html              # 可视化结果文件，自动生成
└── README.md               # 项目说明文档
```

---

## 🚀 使用说明

### 1. 安装依赖

```bash
pip install py2neo pyvis requests
```

### 2. 安装Neo4j数据库

**选项A：使用Neo4j Desktop（推荐）**
- 下载并安装 [Neo4j Desktop](https://neo4j.com/download/)
- 创建新的数据库项目
- 设置数据库密码（默认用户名：neo4j）
- 启动数据库服务

**选项B：使用Docker**
```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:latest
```

**选项C：直接安装Neo4j Community Edition**
- 从官网下载Neo4j Community Edition
- 按照官方文档安装和配置

### 3. 配置连接信息

在 `config.py` 中修改Neo4j连接配置：
```python
GRAG_NEO4J_URI = "bolt://localhost:7687"  # Neo4j连接地址
GRAG_NEO4J_USER = "neo4j"                 # 用户名
GRAG_NEO4J_PASSWORD = "your_password"     # 你设置的密码
GRAG_NEO4J_DATABASE = "neo4j"             # 数据库名
```

### 4. 验证连接

启动Neo4j Desktop后，可以通过以下方式验证：
- 访问 http://localhost:7474 打开Neo4j Browser
- 使用配置的用户名密码登录
- 执行测试查询：`MATCH (n) RETURN n LIMIT 5`

### 4. 设置 DeepSeek API Key

在 `quintuple_extractor.py` 和 `quintuple_rag_query.py` 中替换：

API_KEY = "sk-xxx"  # 替换为你自己的 DeepSeek API 密钥

> DeepSeek 注册地址：https://platform.deepseek.com/

---

## 🧪 使用方式

### 方法一：运行主程序

python main.py

程序将提示你选择输入方式：
请选择输入方式：
1 - 手动输入文本
2 - 从文件读取文本


- **手动输入**：支持逐段输入中文语句，提取知识。
- **文件读取**：输入包含多条文本的 `.txt` 文件路径，逐行处理。

成功后将自动打开 `graph.html`，展示生成的知识图谱。

---

### 方法二：构建批处理函数

可独立调用：

```python
import requests
from main_tri import batch_add_texts

texts = ["李雷在操场上打篮球。", "韩梅梅喜欢读书。"]
batch_add_texts(texts)
```
---

### 方法三：知识图谱问答

图谱构建完成后支持交互式问答：

```text
请输入查询问题（输入空行退出）：
> 谁喜欢读书？
```

将返回从图谱中提取的实体与关系信息。

---

## 📊 可视化效果

使用 PyVis 生成交互式图谱页面 `graph.html`，节点和关系可拖动、放缩、查看信息。

## ✅ 示例输入输出

**输入文本：**

小红在教室里看书。
小明和小强是好朋友。

**生成五元组：**

[
  ["小红", "人物", "看", "书", "物品"],
  ["小明", "人物", "是", "好朋友", "关系"],
  ["小强", "人物", "是", "好朋友", "关系"]
]

**可视化图谱：**

- 小红(人物) —[看]→ 书(物品)  
- 小明(人物) —[是]→ 好朋友(关系)  
- 小强(人物) —[是]→ 好朋友(关系)  
