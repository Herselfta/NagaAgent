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
            # 直接播放最终文本
            asyncio.create_task(self._play_text(final_text))
    
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
            
            # 播放完整句子
            asyncio.create_task(self._play_text(complete_sentence))
            
            # 更新缓冲区
            if remaining_text:
                self.text_buffer = [remaining_text]
            else:
                self.text_buffer = []
        
        # 防止缓冲区过大
        if len(self.text_buffer) > self.max_buffer_size:
            # 强制播放缓冲区内容
            forced_text = ' '.join(self.text_buffer)
            asyncio.create_task(self._play_text(forced_text))
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
            await self._play_audio_file(temp_file_path)
            
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
            system = platform.system()
            
            if system == "Windows":
                import subprocess
                subprocess.Popen(["start", file_path], shell=True)
            elif system == "Darwin":  # macOS
                import subprocess
                subprocess.Popen(["open", file_path])
            elif system == "Linux":
                import subprocess
                subprocess.Popen(["xdg-open", file_path])
            else:
                logger.warning(f"不支持的操作系统: {system}")
                
        except Exception as e:
            logger.error(f"系统播放器调用失败: {e}")

# 全局实例
_voice_integration_instance: Optional[VoiceIntegration] = None

def get_voice_integration() -> VoiceIntegration:
    """获取语音集成实例（单例模式）"""
    global _voice_integration_instance
    if _voice_integration_instance is None:
        _voice_integration_instance = VoiceIntegration()
    return _voice_integration_instance