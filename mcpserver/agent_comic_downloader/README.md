# Comic Downloader MCP Agent

一个简化的漫画下载器MCP代理，基于原项目的核心下载功能，只保留下载功能，默认保存到桌面。

## 功能特点

- 简化的漫画下载功能
- 默认保存到桌面
- 支持异步下载
- 提供HTTP API接口
- 支持下载状态查询
- 支持取消下载任务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务器

```bash
python start_server.py
```

服务器将在 `http://localhost:8080` 启动。

## API接口

### 1. 下载漫画

**POST** `/download`

请求体：
```json
{
  "album_id": "漫画ID"
}
```

响应：
```json
{
  "success": true,
  "album_id": "漫画ID",
  "message": "下载完成",
  "album_title": "漫画标题",
  "album_author": "作者",
  "download_path": "下载路径"
}
```

### 2. 获取下载状态

**GET** `/status/{album_id}`

响应：
```json
{
  "status": "completed",
  "message": "下载完成",
  "progress": 100,
  "result": {
    "album_title": "漫画标题",
    "download_path": "下载路径"
  }
}
```

### 3. 取消下载

**POST** `/cancel`

请求体：
```json
{
  "album_id": "漫画ID"
}
```

响应：
```json
{
  "success": true,
  "message": "任务已取消"
}
```

### 4. 获取所有下载状态

**GET** `/status`

响应：
```json
{
  "task_id_1": {
    "status": "completed",
    "progress": 100,
    "message": "下载完成"
  },
  "task_id_2": {
    "status": "downloading",
    "progress": 50,
    "message": "正在下载"
  }
}
```

### 5. 健康检查

**GET** `/health`

响应：
```json
{
  "status": "healthy",
  "service": "comic_downloader"
}
```

## 使用示例

### 使用curl下载漫画

```bash
curl -X POST "http://localhost:8080/download" \
     -H "Content-Type: application/json" \
     -d '{"album_id": "422866"}'
```

### 查询下载状态

```bash
curl "http://localhost:8080/status/422866"
```

### 取消下载

```bash
curl -X POST "http://localhost:8080/cancel" \
     -H "Content-Type: application/json" \
     -d '{"album_id": "422866"}'
```

## 配置说明

下载器使用以下默认配置：

- 保存位置：桌面
- 文件夹命名规则：使用漫画名称
- 图片下载线程数：10
- 章节下载线程数：3
- 客户端类型：API客户端
- 重试次数：3

## 注意事项

1. 确保原项目的依赖已正确安装
2. 下载的漫画会保存到桌面
3. 支持异步下载，可以同时下载多个漫画
4. 可以通过API查询下载进度
5. 支持取消正在进行的下载任务

## 错误处理

如果下载失败，API会返回错误信息：

```json
{
  "success": false,
  "album_id": "漫画ID",
  "message": "下载失败: 错误信息",
  "error": "具体错误信息"
}
```

## 开发说明

本项目基于原项目的核心下载功能，移除了以下功能：

- 搜索功能
- 登录功能
- 收藏夹功能
- 插件系统
- 复杂的配置选项

只保留了核心的下载功能，并默认保存到桌面，简化了使用流程。 