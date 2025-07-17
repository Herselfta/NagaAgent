# Agent 预处理系统

这是一个Python版本的预处理系统，参考了server.js的设计，提供了完整的消息预处理、插件管理和API代理功能。

## 功能特性

### 🎯 核心功能
- **消息预处理**: 变量替换、占位符处理、系统提示词注入
- **图片处理**: 自动下载、格式转换、压缩优化
- **插件系统**: 支持消息预处理器和静态插件
- **API代理**: 完整的OpenAI API代理，支持流式和非流式响应
- **工具调用**: 自动解析和执行LLM返回的工具调用

### 🔧 预处理功能
- **Agent占位符**: `{{AgentName}}` - 动态加载Agent配置
- **环境变量**: `{{Tarxxx}}`, `{{Varxxx}}` - 环境变量替换
- **时间日期**: `{{Date}}`, `{{Time}}`, `{{Today}}` - 时间信息
- **静态插件**: 支持各种静态占位符
- **handoff工具**: `{{handoffAllTools}}`, `{{handoffWeatherInfo}}` - 工具描述
- **表情包**: `{{xxx表情包}}` - 表情包列表
- **日记本**: `{{xxx日记本}}` - 角色日记内容
- **异步结果**: `{{handoff_ASYNC_RESULT::plugin::id}}` - 异步任务结果

## 项目结构

```
agent/
├── __init__.py                 # 包初始化
├── preprocessor.py            # 预处理系统
├── plugin_manager.py          # 插件管理器
├── api_server.py              # API服务器
├── image_processor.py         # 图片处理器
├── start_server.py            # 启动脚本
├── README.md                  # 说明文档
└── plugins/                   # 插件目录
    └── ImageProcessor/        # 图片处理器插件
        ├── plugin-manifest.json
        ├── config.env
        └── image_processor.py
```

## 安装依赖

```bash
pip install aiohttp pillow
```

## 配置环境变量

创建 `.env` 文件或设置环境变量：

```bash
# API配置
API_Key=your_api_key_here
API_URL=https://api.deepseek.com/v1
Key=your_server_key_here

# 服务器配置
HOST=127.0.0.1
PORT=8000

# 调试模式
DEBUG=False

# 工具调用循环限制
MaxhandoffLoopStream=5
MaxhandoffLoopNonStream=5
```

## 启动服务器

### 方法1: 直接运行
```bash
python agent/start_server.py
```

### 方法2: 作为模块运行
```bash
python -m agent.start_server
```

### 方法3: 使用环境变量
```bash
HOST=0.0.0.0 PORT=8080 python agent/start_server.py
```

## API使用

### 1. 模型列表
```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer your_server_key_here"
```

### 2. 对话完成
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer your_server_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "你好，今天是{{Date}}"}
    ],
    "stream": false
  }'
```

### 3. 流式对话
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer your_server_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "请帮我分析这张图片：https://example.com/image.jpg"}
    ],
    "stream": true
  }'
```

## 插件开发

### 1. 消息预处理器插件

创建插件目录结构：
```
plugins/MyPlugin/
├── plugin-manifest.json
├── config.env
└── my_plugin.py
```

#### plugin-manifest.json
```json
{
  "name": "MyPlugin",
  "displayName": "我的插件",
  "description": "插件描述",
  "version": "1.0.0",
  "pluginType": "messagePreprocessor",
  "entryPoint": {
    "script": "my_plugin.py",
    "protocol": "direct"
  },
  "communication": {
    "protocol": "direct",
    "timeout": 30000
  },
  "configSchema": {
    "my_setting": "string",
    "DebugMode": "boolean"
  },
  "capabilities": {
    "messagePreprocessing": true
  }
}
```

#### config.env
```bash
my_setting=default_value
DebugMode=false
```

#### my_plugin.py
```python
import logging
from typing import List, Dict

logger = logging.getLogger("MyPlugin")

async def processMessages(messages: List[Dict], config: Dict = None) -> List[Dict]:
    """处理消息"""
    # 你的处理逻辑
    return messages

async def initialize(config: Dict = None):
    """初始化插件"""
    logger.info("我的插件已初始化")
    return True

async def shutdown():
    """关闭插件"""
    logger.info("我的插件已关闭")
    return True
```

### 2. 静态插件

静态插件用于生成占位符值，在系统启动时执行一次。

#### plugin-manifest.json
```json
{
  "name": "StaticPlugin",
  "displayName": "静态插件",
  "pluginType": "static",
  "entryPoint": {
    "command": "python static_script.py"
  },
  "capabilities": {
    "systemPromptPlaceholders": [
      {
        "placeholder": "{{MyPlaceholder}}"
      }
    ]
  }
}
```

## 与现有系统的集成

### 1. 替换conversation_core.py中的预处理

```python
# 在conversation_core.py中添加
from agent import preprocess_messages

# 在process方法中使用
async def process(self, u, is_voice_input=False):
    # ... 其他代码 ...
    
    # 预处理消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": u}
    ]
    processed_messages = await preprocess_messages(messages, model=DEEPSEEK_MODEL)
    
    # 使用预处理后的消息调用LLM
    resp = await self.async_client.chat.completions.create(
        model=DEEPSEEK_MODEL, 
        messages=processed_messages,
        temperature=TEMPERATURE, 
        max_tokens=MAX_TOKENS, 
        stream=True
    )
    
    # ... 其他代码 ...
```

### 2. 添加图片处理

```python
# 在conversation_core.py中添加
from agent.plugin_manager import get_plugin_manager

# 在process方法中使用
async def process(self, u, is_voice_input=False):
    # ... 其他代码 ...
    
    # 图片处理
    plugin_manager = get_plugin_manager()
    messages = await plugin_manager.execute_message_preprocessor(
        "ImageProcessor", messages
    )
    
    # ... 其他代码 ...
```

## 主要优势

### 🚀 相比server.js的优势
1. **Python生态**: 更好的AI/ML库支持
2. **类型安全**: 完整的类型注解
3. **异步处理**: 原生asyncio支持
4. **模块化**: 清晰的模块分离
5. **易于扩展**: 简单的插件开发接口

### 🔄 与现有系统的兼容性
1. **无缝集成**: 可以直接替换现有预处理逻辑
2. **保持接口**: 兼容现有的API调用方式
3. **渐进迁移**: 可以逐步迁移功能
4. **配置兼容**: 支持现有的环境变量配置

## 故障排除

### 常见问题

1. **插件加载失败**
   - 检查插件目录结构
   - 验证plugin-manifest.json格式
   - 查看日志输出

2. **预处理不生效**
   - 确认占位符格式正确
   - 检查环境变量设置
   - 验证插件配置

3. **图片处理失败**
   - 安装PIL库: `pip install pillow`
   - 检查网络连接
   - 验证图片URL格式

### 调试模式

设置环境变量启用调试模式：
```bash
export DEBUG=True
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。 