# Comic Downloader MCP Agent 使用说明

## 概述

这是一个简化的漫画下载器MCP代理，基于原项目的核心下载功能，只保留下载功能，默认保存到桌面。

## 快速开始

### 1. 安装依赖

```bash
cd mcpserver/agent_comic_downloader
pip install -r requirements.txt
```

### 2. 快速测试

```bash
python quick_test.py
```

### 3. 启动服务器

```bash
python start_server.py
```

服务器将在 `http://localhost:8080` 启动。

## 使用方法

### 方法1: 直接调用工具函数

```python
from mcpserver.agent_comic_downloader.mcp_tools import call_tool

# 下载漫画
result = call_tool('download_comic', {'album_id': '422866'})
print(result)

# 获取下载状态
status = call_tool('get_download_status', {'album_id': '422866'})
print(status)
```

### 方法2: 使用HTTP API

```bash
# 下载漫画
curl -X POST "http://localhost:8080/download" \
     -H "Content-Type: application/json" \
     -d '{"album_id": "422866"}'

# 获取下载状态
curl "http://localhost:8080/status/422866"

# 取消下载
curl -X POST "http://localhost:8080/cancel" \
     -H "Content-Type: application/json" \
     -d '{"album_id": "422866"}'
```

### 方法3: 使用示例脚本

```bash
python example_usage.py
```

## 配置说明

### 环境变量

可以通过环境变量配置下载器行为：

```bash
# 服务配置
export COMIC_DOWNLOADER_HOST=0.0.0.0
export COMIC_DOWNLOADER_PORT=8080

# 下载配置
export COMIC_DOWNLOADER_PATH=desktop  # 或指定路径
export COMIC_DOWNLOADER_MAX_CONCURRENT=3

# 线程配置
export COMIC_DOWNLOADER_IMAGE_THREADS=10
export COMIC_DOWNLOADER_PHOTO_THREADS=3

# 重试配置
export COMIC_DOWNLOADER_RETRY_TIMES=3

# 日志配置
export COMIC_DOWNLOADER_LOG_LEVEL=INFO

# 客户端配置
export COMIC_DOWNLOADER_CLIENT_TYPE=api  # api 或 html
```

### 默认配置

- 保存位置：桌面
- 文件夹命名：使用漫画名称
- 图片下载线程：10
- 章节下载线程：3
- 客户端类型：API客户端
- 重试次数：3

## API接口

### 1. 下载漫画

**POST** `/download`

```json
{
  "album_id": "漫画ID"
}
```

### 2. 获取下载状态

**GET** `/status/{album_id}`

### 3. 取消下载

**POST** `/cancel`

```json
{
  "album_id": "漫画ID"
}
```

### 4. 获取所有状态

**GET** `/status`

### 5. 健康检查

**GET** `/health`

## 错误处理

如果下载失败，会返回错误信息：

```json
{
  "success": false,
  "album_id": "漫画ID",
  "message": "下载失败: 错误信息",
  "error": "具体错误信息"
}
```

## 注意事项

1. 确保原项目的依赖已正确安装
2. 下载的漫画会保存到桌面
3. 支持异步下载，可以同时下载多个漫画
4. 可以通过API查询下载进度
5. 支持取消正在进行的下载任务

## 故障排除

### 常见问题

1. **导入错误**
   - 确保原项目路径正确
   - 检查Python路径设置

2. **下载失败**
   - 检查网络连接
   - 确认漫画ID正确
   - 查看日志信息

3. **服务器启动失败**
   - 检查端口是否被占用
   - 确认依赖已安装

### 日志查看

```bash
# 设置日志级别
export COMIC_DOWNLOADER_LOG_LEVEL=DEBUG

# 启动服务器查看详细日志
python start_server.py
```

## 开发说明

本项目基于原项目的核心下载功能，移除了以下功能：

- 搜索功能
- 登录功能
- 收藏夹功能
- 插件系统
- 复杂的配置选项

只保留了核心的下载功能，并默认保存到桌面，简化了使用流程。 