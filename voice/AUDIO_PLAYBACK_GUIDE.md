# 音频播放功能使用指南 🎵

## 概述

NagaAgent 3.0 的语音集成模块使用 **pygame** 进行后台直接播放，支持并发音频合成和本地文件管理。

## 音频处理流程

### 🎯 完整处理流程
1. **智能分句**：将长文本自动分割成合适长度的句子
2. **并发合成**：同时申请多个TTS API合成音频文件
3. **本地存储**：将音频文件保存到`logs/audio_temp`目录
4. **顺序播放**：pygame按顺序播放这些音频文件
5. **自动清理**：播放完成后自动删除临时文件（可配置保留）

### 🔄 并发处理
- **并发数量**：默认3个并发任务同时申请API
- **信号量控制**：防止过多并发请求影响性能
- **队列管理**：音频文件按生成顺序加入播放队列

## 播放方式

### 🎯 pygame播放方式
- ✅ **文件播放**：从本地文件播放音频，性能稳定
- ✅ **后台播放**：不阻塞主程序
- ✅ **并发合成**：支持多个音频片段并发合成
- ✅ **智能分句**：自动将长文本分割成合适长度的句子
- ✅ **本地管理**：音频文件统一存储在`logs/audio_temp`目录

## 安装依赖

pygame已包含在项目依赖中，安装项目依赖即可：

```bash
pip install -r requirements.txt
```

## 配置说明

在 `config.json` 中配置语音相关参数：

```json
{
  "system": {
    "voice_enabled": true
  },
  "tts": {
    "port": 5050,
    "default_voice": "zh-CN-XiaoxiaoNeural",
    "default_format": "mp3",
    "default_speed": 1.0,
    "remove_filter": false,
    "keep_audio_files": false
  }
}
```

### 配置参数说明
- `voice_enabled`: 是否启用语音功能
- `port`: TTS服务端口
- `default_voice`: 默认语音
- `default_format`: 音频格式（mp3, wav, opus等）
- `default_speed`: 播放速度（0.25-4.0）
- `remove_filter`: 是否移除文本过滤器
- `keep_audio_files`: 是否保留音频文件用于调试

## 使用方法

### 1. 自动播放

语音集成模块会自动处理文本播放：

```python
from voice.voice_integration import get_voice_integration

# 获取语音集成实例
voice_integration = get_voice_integration()

# 播放完整文本
voice_integration.receive_final_text("你好，这是一个测试。")

# 播放文本片段（支持智能分句）
voice_integration.receive_text_chunk("这是一个很长的文本，")
voice_integration.receive_text_chunk("它会被自动分割成多个句子进行播放。")
```

### 2. 测试播放功能

运行测试脚本验证播放功能：

```bash
python voice/test_audio_playback.py
```

## 功能特性

### 🧠 智能分句
- 自动识别句子结束标点（。！？；. ! ? ;）
- 短句合并：长度≤5且不包含引号的句子会被合并
- 防止播放过长的音频片段

### 🔄 并发合成
- 多个音频片段并发申请API
- 信号量控制并发数量（默认3个）
- 按生成顺序加入播放队列

### 📁 文件管理
- 音频文件统一存储在`logs/audio_temp`目录
- 文件命名：`tts_audio_[时间戳]_[索引].[格式]`
- 播放完成后自动清理（可配置保留）

### 🛡️ 错误处理
- 详细的错误日志记录
- 优雅的异常处理
- 失败任务不影响其他任务

## 故障排除

### pygame不可用
如果pygame不可用，语音播放功能将无法使用：

```bash
# 检查pygame状态
python voice/test_audio_playback.py

# 手动安装pygame
pip install pygame
```

### 音频播放失败
1. 检查TTS服务是否正常运行：
   ```bash
   python voice/voice_status.py
   ```

2. 检查音频文件目录：
   ```bash
   # 查看音频文件
   ls logs/audio_temp/
   
   # 检查文件权限
   ls -la logs/audio_temp/
   ```

3. 检查音频设备：
   - Windows：检查系统音量
   - macOS：检查音频输出设备
   - Linux：检查音频驱动

4. 查看日志：
   ```bash
   # 查看详细日志
   tail -f logs/nagaagent.log
   ```

### 调试音频文件
如果音频播放有问题，可以启用调试模式保留音频文件：

```json
{
  "tts": {
    "keep_audio_files": true
  }
}
```

然后手动检查生成的音频文件：
```bash
# 查看音频文件
ls -la logs/audio_temp/

# 使用系统播放器测试
# Windows
start logs/audio_temp/tts_audio_*.mp3

# macOS
afplay logs/audio_temp/tts_audio_*.mp3

# Linux
mpv logs/audio_temp/tts_audio_*.mp3
```

## 性能优化

### 并发数量调整
可以修改`max_concurrent_tasks`参数调整并发数量：

```python
# 在voice_integration.py中
self.max_concurrent_tasks = 5  # 增加并发数
```

### 音频格式选择
- **mp3**：推荐，兼容性好，文件小
- **wav**：无损，文件大
- **opus**：压缩率高，兼容性一般

### 播放速度
- 默认：1.0（正常速度）
- 范围：0.25-4.0
- 建议：0.8-1.2（最佳体验）

## 开发说明

### 扩展播放方式
如需添加新的播放方式，可以修改 `voice_integration.py` 中的 `_play_audio_file` 方法：

```python
async def _play_audio_file(self, file_path: str):
    """播放音频文件"""
    try:
        if self.pygame_available:
            await self._play_audio_file_with_pygame(file_path)
        else:
            logger.error("pygame不可用，无法播放音频")
    except Exception as e:
        logger.error(f"播放音频文件失败: {e}")
```

### 自定义分句规则
可以修改 `_check_and_play_sentences` 方法中的分句逻辑：

```python
# 自定义句子结束标点
SENTENCE_END_PUNCTUATIONS = r"[。？！；\.\?\!\;]"

# 自定义短句长度阈值
self.min_sentence_length = 5
```

### 自定义并发控制
可以修改并发处理逻辑：

```python
# 调整并发数量
self.max_concurrent_tasks = 5

# 调整信号量
semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
```

## 更新日志

### v3.0.2
- ✅ 新增并发音频合成功能
- ✅ 本地文件存储管理
- ✅ 按顺序播放音频文件
- ✅ 自动文件清理机制
- ✅ 调试模式支持

### v3.0.1
- ✅ 新增pygame后台直接播放
- ✅ 智能分句功能
- ✅ 并发播放支持
- ✅ 移除系统播放器依赖
- ✅ 简化播放逻辑

### v3.0.0
- 🔄 系统播放器播放
- 🔄 临时文件创建
- 🔄 基础TTS功能 