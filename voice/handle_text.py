import asyncio
from config import config
from voice.gpt_sovits_tts import GPTSoVITS_TTS
from collections import namedtuple
import re
import emoji
import logging
import tempfile
import wave
import os
import time
import pyaudio
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# 全局变量
_voice_integration = None
current_speech_task = None

class AudioPlayer:
    """
    使用PyAudio的音频播放器，支持异步播放和完成回调。
    """
    CHUNK_SIZE = 1024 * 4  # 4KB
    
    def __init__(self):
        self._pyaudio = pyaudio.PyAudio()
        self._stream = None
        self._is_playing = False
        self._stop_requested = False
        
    def play(self, wav_path: str, on_complete: Optional[Callable] = None) -> bool:
        def playback_thread():
            try:
                with wave.open(wav_path, 'rb') as wf:
                    format = self._pyaudio.get_format_from_width(wf.getsampwidth())
                    channels = wf.getnchannels()
                    rate = wf.getframerate()
                    
                    stream = self._pyaudio.open(format=format, channels=channels, rate=rate, output=True)
                    self._stream = stream
                    self._is_playing = True
                    self._stop_requested = False
                    
                    try:
                        logger.info("开始播放音频")
                        while not self._stop_requested:
                            data = wf.readframes(self.CHUNK_SIZE)
                            if not data:
                                break
                            stream.write(data)
                        logger.info("音频播放完成")
                    finally:
                        try:
                            stream.stop_stream()
                            stream.close()
                        except: pass
                        self._stream = None
                        self._is_playing = False
            except Exception as e:
                logger.error(f"播放线程出错: {e}")
            finally:
                if on_complete:
                    logger.info("调用播放完成回调")
                    on_complete()
                try:
                    os.remove(wav_path)
                    logger.info(f"已删除临时文件: {wav_path}")
                except Exception as e:
                    logger.error(f"删除临时文件失败: {e}")

        try:
            self.stop()
            thread = threading.Thread(target=playback_thread)
            thread.daemon = True
            thread.start()
            return True
        except Exception as e:
            logger.error(f"启动播放失败: {e}")
            if on_complete:
                on_complete()
            return False
    
    def stop(self):
        if self._is_playing:
            self._stop_requested = True
            if self._stream:
                try: self._stream.stop_stream()
                except: pass
            time.sleep(0.1) # Give thread time to stop
    
    def __del__(self):
        self.stop()
        if self._pyaudio:
            try: self._pyaudio.terminate()
            except: pass

def get_voice_integration():
    """获取语音集成实例（单例模式）"""
    global _voice_integration
    if _voice_integration is None:
        if not config.system.voice_enabled:
            logger.info("语音功能未启用")
            return None
        tts_choice = config.tts.TTS_CHOICE.upper()
        logger.info(f"正在初始化语音系统，选择: {tts_choice}")
        if tts_choice == "GPT-SOVITS":
            try:
                _voice_integration = TextToVoice()
                if _voice_integration.tts_system and _voice_integration.tts_system.is_enabled:
                    logger.info("[TTS] ✅ GPT-SoVITS TTS 初始化成功")
                else:
                    logger.warning("[TTS] ⚠️ GPT-SoVITS TTS 已配置但未启用")
                    _voice_integration = None
            except Exception as e:
                logger.error(f"GPT-SoVITS TTS 初始化失败: {e}")
                _voice_integration = None
        else:
            logger.warning(f"未知的 TTS 选择: {tts_choice}")
            _voice_integration = None
    return _voice_integration

class TextToVoice:
    """文本转语音的管理类，包含流式处理和字符缓冲"""
    
    def __init__(self):
        self.tts_system = GPTSoVITS_TTS()
        self.text_buffer = ""
        self.line_breaks = ["。", "？", "！", ".", "?", "!", "\n"]
        self.player = AudioPlayer()

    def _play_audio(self, audio_data: bytes, on_complete: Optional[Callable] = None):
        """修复并播放音频数据"""
        try:
            # 修复WAV头并保存到临时文件
            data_pos = audio_data.find(b'data')
            if data_pos == -1:
                logger.error("在音频数据中未找到 'data' 块")
                if on_complete: on_complete()
                return

            frames_data = audio_data[data_pos + 8:]
            data_size = len(frames_data)
            if data_size == 0:
                logger.error("音频数据帧为空")
                if on_complete: on_complete()
                return

            new_header = (
                b'RIFF' + (data_size + 36).to_bytes(4, 'little') + b'WAVE' +
                b'fmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') +
                (1).to_bytes(2, 'little') + (32000).to_bytes(4, 'little') +
                (64000).to_bytes(4, 'little') + (2).to_bytes(2, 'little') +
                (16).to_bytes(2, 'little') + b'data' + data_size.to_bytes(4, 'little')
            )
            fixed_wav_data = new_header + frames_data
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.write(fixed_wav_data)
                temp_wav_path = temp_wav.name
            
            logger.info(f"已创建临时WAV文件: {temp_wav_path}")
            self.player.play(temp_wav_path, on_complete)

        except Exception as e:
            logger.error(f"音频处理和播放失败: {e}")
            if on_complete:
                on_complete()
    
    async def _process_text_chunk(self, text_chunk: str, event=None):
        """异步处理文本片段"""
        if not text_chunk or not self.tts_system or not self.tts_system.is_enabled:
            if event: event.set(False)
            return
                
        self.text_buffer += text_chunk
            
        while True:
            end_positions = [(self.text_buffer.find(mark), mark) for mark in self.line_breaks if mark in self.text_buffer]
            if not end_positions:
                break
            
            pos, mark = min(end_positions, key=lambda x: x[0])
            sentence = self.text_buffer[:pos + len(mark)].strip()
            
            if sentence:
                audio_data = await self.tts_system.speak(sentence)
                if audio_data:
                    internal_event = asyncio.Event()
                    def on_complete():
                        try:
                            loop = asyncio.get_running_loop()
                            loop.call_soon_threadsafe(internal_event.set)
                        except RuntimeError: # Loop not running
                            asyncio.run(internal_event.set())

                    self._play_audio(audio_data, on_complete)
                    await internal_event.wait()
                    if event: event.set(True)
                elif event:
                    event.set(False)
            
            self.text_buffer = self.text_buffer[pos + len(mark):].lstrip()

    def receive_text_chunk(self, text_chunk: str, event=None):
        """接收文本片段（流式输入）"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_text_chunk(text_chunk, event))
        except RuntimeError:
            asyncio.run(self._process_text_chunk(text_chunk, event))
        except Exception as e:
            logger.error(f"文本片段处理失败: {e}")
            if event: event.set(False)
    
    async def _process_final_text(self, text: str):
        """异步处理最终文本"""
        if not text or not self.tts_system or not self.tts_system.is_enabled:
            return
        remaining_text = self.text_buffer + text
        if remaining_text.strip():
            audio_data = await self.tts_system.speak(remaining_text)
            if audio_data:
                self._play_audio(audio_data)
        self.text_buffer = ""

    def receive_final_text(self, text: str):
        """接收完整的文本（用于非流式或最终文本）"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_final_text(text))
        except RuntimeError:
            asyncio.run(self._process_final_text(text))
        except Exception as e:
            logger.error(f"最终文本处理失败: {e}")
            
async def speak_text(text: str):
    """
    异步处理文本并使用TTS播放。
    """
    global current_speech_task
    
    if current_speech_task and not current_speech_task.done():
        current_speech_task.cancel()
        logger.info("已取消上一个语音播放任务。")

    voice = get_voice_integration()
    if not voice:
        logger.warning("语音集成未初始化，无法播放语音。")
        return

    cleaned_text = prepare_tts_input_with_context(text)
    
    # 创建一个事件来等待播放完成
    playback_finished_event = asyncio.Event()

    def on_playback_finished():
        logger.info("主调用中收到播放完成信号。")
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(playback_finished_event.set)
        except RuntimeError:
            # 如果没有正在运行的循环，这可能意味着它在一个单独的线程中完成
            # 在这种情况下，我们可能需要不同的同步机制，但对于大多数情况，这是可以的
            pass

    # 发送文本进行处理和播放
    voice.receive_text_chunk(cleaned_text, event=playback_finished_event)
    
    # 等待播放完成
    try:
        logger.info("等待语音播放完成...")
        await asyncio.wait_for(playback_finished_event.wait(), timeout=60.0) # 60秒超时
        logger.info("语音播放任务完成。")
    except asyncio.TimeoutError:
        logger.error("语音播放超时。")

def prepare_tts_input_with_context(text: str) -> str:
    """
    清洗Markdown文本并为部分元素添加上下文提示，适用于TTS输入，保留段落分隔。
    """
    text = emoji.replace_emoji(text, replace='')
    def header_replacer(match):
        level = len(match.group(1))
        header_text = match.group(2).strip()
        if level == 1: return f"Title — {header_text}\n"
        elif level == 2: return f"Section — {header_text}\n"
        else: return f"Subsection — {header_text}\n"
    text = re.sub(r"^(#{1,6})\s+(.*)", header_replacer, text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"```([\s\S]+?)```", r"(code block omitted)", text)
    text = re.sub(r"`([^`]+)`", r"code snippet: \1", text)
    text = re.sub(r"(\*\*|__|\*|_)", '', text)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"Image: \1", text)
    text = re.sub(r"</?[^>]+(>|$)", '', text)
    text = re.sub(r"\n{2,}", '\n\n', text)
    text = re.sub(r" {2,}", ' ', text)
    return text.strip()
