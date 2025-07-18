# Agent JSON 标准格式

## 概述

Agent JSON 是用于定义和配置 AI Agent 的标准格式，支持动态注册、能力描述和参数配置。

## 标准格式

```json
{
  "name": "AgentName",
  "displayName": "Agent显示名称",
  "version": "1.0.0",
  "description": "Agent功能描述",
  "author": "作者信息",
  "agentType": "agent",
  "modelId": "deepseek-chat",
  "systemPrompt": "系统提示词，支持{{AgentName}}占位符",
  "maxOutputTokens": 8192,
  "temperature": 0.7,
  "modelProvider": "openai",
  "apiBaseUrl": "https://api.deepseek.com/v1",
  "apiKey": "your_api_key",
  "capabilities": {
    "feature1": true,
    "feature2": true
  },
  "supported_actions": [
    {
      "action": "action_name",
      "description": "动作描述",
      "parameters": {
        "param1": {
          "type": "string",
          "required": true,
          "description": "参数描述"
        },
        "param2": {
          "type": "string",
          "required": false,
          "default": "default_value",
          "description": "参数描述"
        }
      },
      "example": {
        "action": "action_name",
        "param1": "value1"
      }
    }
  ],
  "configSchema": {
    "CONFIG_KEY": {
      "type": "string|integer|boolean",
      "default": "default_value",
      "description": "配置描述"
    }
  },
  "dependencies": {
    "package_name": "version_constraint"
  },
  "metadata": {
    "created": "2024-01-01",
    "updated": "2024-01-01",
    "tags": ["tag1", "tag2"],
    "category": "category_name"
  }
}
```

## 字段说明

### 基础信息
- **name**: Agent的唯一标识符（英文）
- **displayName**: Agent的显示名称（中文）
- **version**: Agent版本号
- **description**: Agent功能描述
- **author**: 作者或组织信息
- **agentType**: 固定为 "agent"

### 模型配置
- **modelId**: 使用的模型ID
- **systemPrompt**: 系统提示词，支持{{AgentName}}占位符
- **maxOutputTokens**: 最大输出token数
- **temperature**: 温度参数（0.0-2.0）
- **modelProvider**: 模型提供商（如openai）
- **apiBaseUrl**: API基础URL
- **apiKey**: API密钥

### 能力描述
- **capabilities**: 能力特性描述，使用布尔值表示

### 支持的动作
- **supported_actions**: 支持的动作列表
  - **action**: 动作名称
  - **description**: 动作描述
  - **parameters**: 参数定义
    - **type**: 参数类型
    - **required**: 是否必需
    - **default**: 默认值
    - **description**: 参数描述
  - **example**: 使用示例

### 配置模式
- **configSchema**: 配置项定义
  - **type**: 配置类型
  - **default**: 默认值
  - **description**: 配置描述

### 依赖和元数据
- **dependencies**: 依赖包列表
- **metadata**: 元数据信息
  - **created**: 创建时间
  - **updated**: 更新时间
  - **tags**: 标签列表
  - **category**: 分类

## 示例

### 简单对话Agent
```json
{
  "name": "ChatAgent",
  "displayName": "对话助手",
  "version": "1.0.0",
  "description": "通用对话助手",
  "author": "Naga系统",
  "agentType": "agent",
  "modelId": "deepseek-chat",
  "systemPrompt": "你是一个友好的AI助手，名叫{{AgentName}}。",
  "maxOutputTokens": 4096,
  "temperature": 0.7,
  "modelProvider": "openai",
  "apiBaseUrl": "https://api.deepseek.com/v1",
  "apiKey": "your_api_key",
  "capabilities": {
    "conversation": true,
    "context_aware": true
  },
  "supported_actions": [],
  "configSchema": {},
  "dependencies": {},
  "metadata": {
    "created": "2024-01-01",
    "updated": "2024-01-01",
    "tags": ["chat", "conversation"],
    "category": "general"
  }
}
```

### 工具型Agent
```json
{
  "name": "FileAgent",
  "displayName": "文件操作Agent",
  "version": "1.0.0",
  "description": "文件操作助手",
  "author": "Naga系统",
  "agentType": "agent",
  "modelId": "deepseek-chat",
  "systemPrompt": "你是文件操作专家{{AgentName}}，擅长文件管理。",
  "maxOutputTokens": 2048,
  "temperature": 0.5,
  "modelProvider": "openai",
  "apiBaseUrl": "https://api.deepseek.com/v1",
  "apiKey": "your_api_key",
  "capabilities": {
    "file_operations": true,
    "path_management": true
  },
  "supported_actions": [
    {
      "action": "read",
      "description": "读取文件内容",
      "parameters": {
        "path": {
          "type": "string",
          "required": true,
          "description": "文件路径"
        }
      },
      "example": {"action": "read", "path": "test.txt"}
    },
    {
      "action": "write",
      "description": "写入文件内容",
      "parameters": {
        "path": {
          "type": "string",
          "required": true,
          "description": "文件路径"
        },
        "content": {
          "type": "string",
          "required": true,
          "description": "写入内容"
        }
      },
      "example": {"action": "write", "path": "test.txt", "content": "hello"}
    }
  ],
  "configSchema": {
    "BASE_DIR": {
      "type": "string",
      "default": ".",
      "description": "基础目录"
    }
  },
  "dependencies": {
    "pathlib": ">=1.0.0"
  },
  "metadata": {
    "created": "2024-01-01",
    "updated": "2024-01-01",
    "tags": ["file", "io", "system"],
    "category": "system_tools"
  }
}
```

## 注册机制

Agent JSON 文件通过以下方式自动注册：

1. 放置在 `mcpserver/` 目录下的子目录中
2. 文件名为 `agent-manifest.json`
3. 系统启动时自动扫描并注册
4. 注册到 AgentManager 进行统一管理

## 最佳实践

1. **命名规范**: 使用 PascalCase 命名 Agent
2. **版本管理**: 遵循语义化版本控制
3. **文档完整**: 提供详细的功能描述和示例
4. **参数验证**: 明确定义必需和可选参数
5. **错误处理**: 在系统提示词中包含错误处理指导
6. **安全性**: 避免在配置中暴露敏感信息 