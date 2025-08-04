# NagaAgent 语音服务 🗣️

基于Edge-TTS的OpenAI兼容语音合成服务，为NagaAgent提供高质量的文本转语音功能。

## 功能特性

- **OpenAI兼容接口**：`/v1/audio/speech`，请求结构和行为与OpenAI类似
- **支持多种语音**：将OpenAI语音（alloy, echo, fable, onyx, nova, shimmer）映射到`edge-tts`语音
- **多音频格式**：支持多种音频格式（mp3, opus, aac, flac, wav, pcm）
- **可调节语速**：支持0.25x到4.0x的播放速度
- **可选直接指定edge-tts语音**：既可用OpenAI语音映射，也可直接指定任意edge-tts语音
- **HTTP和WebSocket双模式**：支持REST API和实时WebSocket连接
- **统一配置管理**：与NagaAgent主系统配置完全集成
- **后台直接播放**：使用pygame库进行内存中直接播放，无需创建临时文件

![GitHub stars](https://img.shields.io/github/stars/travisvn/openai-edge-tts?style=social)
![GitHub forks](https://img.shields.io/github/forks/travisvn/openai-edge-tts?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/travisvn/openai-edge-tts)
![GitHub top language](https://img.shields.io/github/languages/top/travisvn/openai-edge-tts)
![GitHub last commit](https://img.shields.io/github/last-commit/travisvn/openai-edge-tts?color=red)
[![Discord](https://img.shields.io/badge/Discord-Voice_AI_%26_TTS_Tools-blue?logo=discord&logoColor=white)](https://discord.gg/GkFbBCBqJ6)
[![LinkedIn](https://img.shields.io/badge/Connect_on_LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/travisvannimwegen)

本项目提供了一个本地的、OpenAI兼容的文本转语音（TTS）API，基于 `edge-tts`。它模拟了 OpenAI 的 TTS 接口（`/v1/audio/speech`），让用户可以像使用 OpenAI API 一样，通过多种语音和播放速度将文本转为语音。

`edge-tts` 使用微软 Edge 的在线文本转语音服务，完全免费。

[在 Docker Hub 查看本项目](https://hub.docker.com/r/travisvn/openai-edge-tts)

# 如果觉得有用请点个⭐️

## 功能特性

- **OpenAI兼容接口**：`/v1/audio/speech`，请求结构和行为与OpenAI类似。
- **支持多种语音**：将OpenAI语音（alloy, echo, fable, onyx, nova, shimmer）映射到`edge-tts`语音。
- **多音频格式**：支持多种音频格式（mp3, opus, aac, flac, wav, pcm）。
- **可调节语速**：支持0.25x到4.0x的播放速度。
- **可选直接指定edge-tts语音**：既可用OpenAI语音映射，也可直接指定任意edge-tts语音。

## 快速开始

### 前置条件

- **Python 3.8+**：确保Python环境已安装
- **依赖包**：安装项目依赖 `pip install -r requirements.txt`
- **pygame**：用于后台音频播放（已包含在requirements.txt中）
- **ffmpeg**（可选）：音频格式转换需要，只用mp3可不装

### 配置说明

语音服务配置在 `config.json` 文件的 `tts` 部分：

```json
{
  "tts": {
    "api_key": "your_api_key_here",
    "port": 5050,
    "default_voice": "en-US-AvaNeural",
    "default_format": "mp3",
    "default_speed": 1.0,
    "default_language": "en-US",
    "remove_filter": false,
    "expand_api": true,
    "require_api_key": true
  }
}
```

### 启动方式

#### 方式1：通过NagaAgent主程序自动启动
```bash
python main.py
```
主程序会自动启动语音服务。

#### 方式2：独立启动语音服务
```bash
# 启动HTTP服务器
python voice/start_voice_service.py --mode http

# 启动WebSocket服务器
python voice/start_voice_service.py --mode websocket

# 同时启动两种模式
python voice/start_voice_service.py --mode both

# 检查依赖
python voice/start_voice_service.py --check-deps

# 自定义端口
python voice/start_voice_service.py --port 8080
```

#### 方式3：直接启动服务器
```bash
# HTTP服务器
python voice/server.py

# WebSocket服务器
python voice/websocket_edge_tts.py
```

## 服务状态检查

### 检查服务状态
```bash
python voice/voice_status.py
```

### 测试TTS功能
```bash
curl -X POST http://127.0.0.1:5050/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "input": "Hello, this is a test.",
    "voice": "alloy",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output test_speech.mp3
```

### 音频播放方式

#### pygame后台直接播放
- **内存播放**：使用pygame库直接在内存中播放音频数据，无需创建临时文件
- **并发支持**：支持多个音频片段排队播放，避免重叠
- **智能分句**：自动将长文本分割成合适长度的句子进行播放
- **高效播放**：直接播放内存中的音频数据，性能更优

### 用法

#### 接口：`/v1/audio/speech`

将输入文本转为音频。可用参数：

**必填参数：**

- **input** (string)：要转为音频的文本（最多4096字符）。

**可选参数：**

- **model** (string)："tts-1" 或 "tts-1-hd"（默认：`tts-1`）。
- **voice** (string)：OpenAI兼容语音（alloy, echo, fable, onyx, nova, shimmer）或任意`edge-tts`语音（默认：`en-US-AvaNeural`）。
- **response_format** (string)：音频格式。可选：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`（默认：`mp3`）。
- **speed** (number)：播放速度（0.25~4.0），默认`1.0`。

curl请求示例，保存为mp3：

```bash
curl -X POST http://localhost:5050/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "input": "Hello, I am your AI assistant! Just let me know how I can help bring your ideas to life.",
    "voice": "echo",
    "response_format": "mp3",
    "speed": 1.1
  }' \
  --output speech.mp3
```

或与OpenAI参数一致的写法：

```bash
curl -X POST http://localhost:5050/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "tts-1",
    "input": "Hello, I am your AI assistant! Just let me know how I can help bring your ideas to life.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

其它语言示例：

```bash
curl -X POST http://localhost:5050/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "tts-1",
    "input": "じゃあ、行く。電車の時間、調べておくよ。",
    "voice": "ja-JP-KeitaNeural"
  }' \
  --output speech.mp3
```

### 其它接口

- **POST/GET /v1/models**：获取可用TTS模型列表。
- **POST/GET /v1/voices**：按语言/地区获取`edge-tts`语音。
- **POST/GET /v1/voices/all**：获取所有`edge-tts`语音及支持信息。

### 贡献

欢迎贡献代码！请fork本仓库并提交PR。

### 许可证

本项目采用GNU GPL v3.0协议，仅限个人用途。如需企业或非个人用途，请联系 tts@travisvn.com

___

## 示例用法

> [!TIP]
> 如果访问有问题，将 `localhost` 换成本机IP（如 `192.168.0.1`）
> 
> _当你在其它服务器/电脑或用Open WebUI等工具访问时，可能需要将URL中的`localhost`换为本机IP（如`192.168.0.1`）_

# Open WebUI

打开管理面板，进入 设置 -> Audio

下图为正确配置本项目替代OpenAI接口的截图：

![Open WebUI管理设置音频接口配置截图](https://utfs.io/f/MMMHiQ1TQaBo9GgL4WcUbjSRlqi86sV3TXh47KYBJCkdQ20M)

如果Open WebUI和本项目都用Docker运行，API地址一般为 `http://host.docker.internal:5050/v1`

> [!NOTE]
> 查看[Open WebUI官方文档关于Edge TTS集成](https://docs.openwebui.com/tutorials/text-to-speech/openai-edge-tts-integration)
___

# 语音示例 🎙️
[试听语音样例及全部Edge TTS语音](https://tts.travisvn.com/)
