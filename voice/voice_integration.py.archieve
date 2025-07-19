#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音集成模块 - 负责接收文本并调用TTS服务播放音频
"""
import asyncio
import json
import logging
import base64
import tempfile
import os
from typing import Optional, List
import aiohttp
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config

logger = logging.getLogger("VoiceIntegration")

class VoiceIntegration:
    """语音集成类 - 负责文本接收和TTS播放"""
    
    def __init__(self):
        self.enabled = config.system.voice_enabled
        self.tts_url = f"http://127.0.0.1:{config.tts.port}/v1/audio/speech"
        self.text_buffer = []  # 文本缓冲区
        self.sentence_endings = ['.', '!', '?', '。', '！', '？', '；', ';']
        self.min_sentence_length = 10  # 最小句子长度
        self.max_buffer_size = 5  # 最大缓冲区大小
        
    def receive_text_chunk(self, text: str):
        """接收文本片段"""
        if not self.enabled:
            return
            
        if text and text.strip():
            self.text_buffer.append(text.strip())
            logger.debug(f"接收文本片段: {text[:50]}...")
            
            # 检查是否有完整句子
            self._check_and_play_sentences()
    
    def receive_final_text(self, final_text: str):
        """接收最终完整文本"""
        if not self.enabled:
            return
            
        if final_text and final_text.strip():
            logger.info(f"接收最终文本: {final_text[:100]}...")
            # 在后台线程播放最终文本
            self._play_text_in_background(final_text)
    
    def _check_and_play_sentences(self):
        """检查并播放完整句子"""
        if len(self.text_buffer) < 2:
            return
            
        # 合并缓冲区文本
        combined_text = ' '.join(self.text_buffer)
        
        # 查找句子结束位置
        sentence_end_pos = -1
        for ending in self.sentence_endings:
            pos = combined_text.rfind(ending)
            if pos > sentence_end_pos:
                sentence_end_pos = pos
        
        # 如果有完整句子且长度足够
        if sentence_end_pos > 0 and sentence_end_pos >= self.min_sentence_length:
            complete_sentence = combined_text[:sentence_end_pos + 1]
            remaining_text = combined_text[sentence_end_pos + 1:].strip()
            
            # 在后台线程播放完整句子
            self._play_text_in_background(complete_sentence)
            
            # 更新缓冲区
            if remaining_text:
                self.text_buffer = [remaining_text]
            else:
                self.text_buffer = []
        
        # 防止缓冲区过大
        if len(self.text_buffer) > self.max_buffer_size:
            # 强制播放缓冲区内容
            forced_text = ' '.join(self.text_buffer)
            self._play_text_in_background(forced_text)
            self.text_buffer = []
    
    async def _play_text(self, text: str):
        """播放文本音频"""
        try:
            # 调用本地TTS API
            audio_data = await self._generate_audio(text)
            if audio_data:
                await self._play_audio(audio_data)
                logger.info(f"成功播放音频: {text[:50]}...")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
    
    async def _generate_audio(self, text: str) -> Optional[bytes]:
        """生成音频数据"""
        try:
            headers = {}
            if config.tts.require_api_key:
                headers["Authorization"] = f"Bearer {config.tts.api_key}"
            
            payload = {
                "input": text,
                "voice": config.tts.default_voice,
                "response_format": config.tts.default_format,
                "speed": config.tts.default_speed
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.tts_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        error_text = await response.text()
                        logger.error(f"TTS API调用失败: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"生成音频异常: {e}")
            return None
    
    async def _play_audio(self, audio_data: bytes):
        """播放音频数据"""
        try:
            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix=f".{config.tts.default_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # 使用系统默认播放器播放
            await self._play_with_pygame(temp_file_path)
            
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"播放音频文件失败: {e}")
    
    async def _play_audio_file(self, file_path: str):
        """播放音频文件"""
        try:
            import platform
            import subprocess
            import asyncio
            
            system = platform.system()
            
            if system == "Windows":
                # Windows使用winsound或windows media player
                try:
                    import winsound
                    os.startfile(file_path)
                except ImportError:
                    subprocess.run(["start", "", file_path], shell=True, check=False)
                except Exception as e:
                    logger.error(f"os.startfile 播放失败: {e}")
            elif system == "Darwin":  # macOS
                subprocess.run(["afplay", file_path], check=False)
            elif system == "Linux":
                # Linux尝试多种播放器
                players = ["aplay", "paplay", "mpg123", "mpv", "vlc", "xdg-open"]
                for player in players:
                    try:
                        result = subprocess.run([player, file_path], 
                                               check=False, 
                                               capture_output=True, 
                                               timeout=10)
                        if result.returncode == 0:
                            break
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        continue
                else:
                    logger.warning("找不到可用的音频播放器")
            else:
                logger.warning(f"不支持的操作系统: {system}")
                
        except Exception as e:
            logger.error(f"系统播放器调用失败: {e}")
            # 尝试使用 pygame 作为备选方案
            try:
                await self._play_with_pygame(file_path)
            except Exception as pygame_error:
                logger.error(f"pygame播放也失败: {pygame_error}")
    
    async def _play_with_pygame(self, file_path: str):
        """使用pygame播放音频（备选方案）"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
        except ImportError:
            logger.warning("pygame未安装，无法作为备选播放器")
        except Exception as e:
            logger.error(f"pygame播放失败: {e}")
    
    def _play_text_in_background(self, text: str):
        """在后台线程中播放文本音频"""
        import threading
        
        def run_in_thread():
            """在线程中运行异步函数"""
            try:
                # 创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._play_text(text))
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"后台播放音频失败: {e}")
        
        # 在后台线程中运行
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

# 全局实例
_voice_integration_instance: Optional[VoiceIntegration] = None

def get_voice_integration() -> VoiceIntegration:
    """获取语音集成实例（单例模式）"""
    global _voice_integration_instance
    if _voice_integration_instance is None:
        _voice_integration_instance = VoiceIntegration()
    return _voice_integration_instance