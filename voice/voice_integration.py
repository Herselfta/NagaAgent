#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音集成模块 - 负责接收文本并调用TTS服务播放音频
支持Edge TTS和Minimax TTS两种服务
重构版本：实现真正的异步处理，分离文本显示和音频播放
"""
import asyncio
import logging
import tempfile
import os
import threading
import time
import hashlib
import re
import io
from typing import Optional, List, Dict
import aiohttp
import sys
from pathlib import Path
from queue import Queue, Empty

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logger = logging.getLogger("VoiceIntegration")

# 断句正则表达式
SENTENCE_END_PUNCTUATIONS = r"[。？！；\.\?\!\;]"

class VoiceIntegration:
    """语音集成模块 - 重构版本：真正的异步处理"""
    
    def __init__(self):
        self.provider = config.tts.provider
        if self.provider == 'gpt_sovits':
            self.tts_url = f"http://127.0.0.1:{config.tts.port}/tts"
        else:
            self.tts_url = f"http://127.0.0.1:{config.tts.port}/v1/audio/speech"
        
        # 音频播放配置
        self.min_sentence_length = 5  # 最小句子长度
        self.max_concurrent_tasks = 3  # 最大并发任务数
        
        # 音频文件存储目录
        self.audio_temp_dir = Path("logs/audio_temp")
        self.audio_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 音频播放队列和状态管理
        self.audio_queue = Queue()  # 使用标准Queue替代asyncio.Queue
        self.playing_lock = threading.Lock()
        self.playing_texts = set()  # 防止重复播放
        self.audio_files_in_use = set()  # 正在使用的音频文件
        
        # 播放状态控制
        self.is_playing = False
        self.current_playback = None
        
        # pygame音频初始化
        self._init_pygame_audio()
        
        # 启动音频播放工作线程
        self.audio_thread = threading.Thread(target=self._audio_player_worker, daemon=True)
        self.audio_thread.start()
        
        # 启动音频文件清理线程
        self.cleanup_thread = threading.Thread(target=self._audio_cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("语音集成模块初始化完成（重构版本）")

    def _init_pygame_audio(self):
        """初始化pygame音频系统"""
        try:
            import pygame
            pygame.init()
            
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                logger.info("pygame音频系统初始化成功")
            except Exception as e:
                logger.warning(f"使用指定参数初始化失败，尝试默认参数: {e}")
                pygame.mixer.init()
                logger.info("pygame音频系统初始化成功（使用默认参数）")
            
            self.pygame_available = True
            logger.info(f"pygame版本: {pygame.version.ver}")
            
        except ImportError:
            logger.error("pygame未安装，语音播放功能不可用")
            self.pygame_available = False
        except Exception as e:
            logger.error(f"pygame音频初始化失败: {e}")
            self.pygame_available = False

    def receive_final_text(self, final_text: str):
        """接收最终完整文本 - 立即处理，不等待音频"""
        if not config.system.voice_enabled:
            return
            
        if final_text and final_text.strip():
            logger.info(f"接收最终文本: {final_text[:100]}")
            # 立即开始音频处理，不阻塞前端显示
            self._process_audio_async(final_text)

    def receive_text_chunk(self, text: str):
        """接收文本片段 - 流式处理"""
        if not config.system.voice_enabled:
            return
            
        if text and text.strip():
            # 流式文本直接处理，不累积
            self._process_audio_async(text.strip())

    def _process_audio_async(self, text: str):
        """异步处理音频，不阻塞主流程"""
        try:
            # 检查是否正在播放相同文本
            text_hash = hashlib.md5(text.encode()).hexdigest()
            with self.playing_lock:
                if text_hash in self.playing_texts:
                    logger.debug(f"跳过重复播放: {text[:30]}...")
                    return
                self.playing_texts.add(text_hash)
            
            # 在后台线程中处理音频，不阻塞主流程
            threading.Thread(
                target=self._generate_and_play_audio,
                args=(text,),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"创建音频处理任务失败: {e}")

    def _generate_and_play_audio(self, text: str):
        """在后台线程中生成并播放音频"""
        try:
            # 文本预处理
            if not getattr(config.tts, 'remove_filter', False):
                from voice.handle_text import prepare_tts_input_with_context
                text = prepare_tts_input_with_context(text)
            
            # 生成音频文件
            audio_file_path = self._generate_audio_file_sync(text)
            if audio_file_path:
                # 加入播放队列
                self.audio_queue.put(audio_file_path)
                logger.info(f"音频文件已加入播放队列: {text[:50]}... -> {audio_file_path}")
            else:
                logger.warning(f"音频文件生成失败: {text[:50]}...")
                
        except Exception as e:
            logger.error(f"音频处理异常: {e}")

    def _generate_audio_file_sync(self, text: str) -> Optional[str]:
        """同步生成音频文件"""
        try:
            # 生成文件名
            timestamp = int(time.time() * 1000)
            filename = f"tts_audio_{timestamp}_{hash(text) % 1000}.{config.tts.default_format}"
            file_path = self.audio_temp_dir / filename
            
            headers = {}
            if config.tts.require_api_key:
                headers["Authorization"] = f"Bearer {config.tts.api_key}"
            
            if self.provider == 'gpt_sovits':
                sovits_config = config.tts.gpt_sovits
                payload = {
                    "text": text,
                    "text_lang": sovits_config.text_lang,
                    "ref_audio_path": sovits_config.ref_audio_path,
                    "prompt_text": sovits_config.prompt_text,
                    "prompt_lang": sovits_config.prompt_lang,
                    "top_k": sovits_config.top_k,
                    "top_p": sovits_config.top_p,
                    "temperature": sovits_config.temperature,
                    "text_split_method": sovits_config.text_split_method,
                    "batch_size": sovits_config.batch_size,
                    "speed_factor": config.tts.default_speed,
                    "media_type": config.tts.default_format,
                    "streaming_mode": True,
                }
            else:
                payload = {
                    "input": text,
                    "voice": config.tts.default_voice,
                    "response_format": config.tts.default_format,
                    "speed": config.tts.default_speed
                }
            
            # 使用requests进行同步调用
            import requests
            response = requests.post(
                self.tts_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                audio_data = response.content
                
                # 保存到本地文件
                with open(file_path, 'wb') as f:
                    f.write(audio_data)
                
                # 标记文件正在使用
                self.audio_files_in_use.add(str(file_path))
                
                logger.debug(f"音频文件已保存: {file_path} ({len(audio_data)} bytes)")
                return str(file_path)
            else:
                logger.error(f"TTS API调用失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"生成音频文件异常: {e}")
            return None

    def _audio_player_worker(self):
        """音频播放工作线程"""
        logger.info("音频播放工作线程启动")
        
        # 在工作线程中重新初始化pygame
        try:
            import pygame
            pygame.init()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            logger.info("音频播放工作线程中pygame初始化成功")
        except Exception as e:
            logger.error(f"音频播放工作线程中pygame初始化失败: {e}")
            return
        
        try:
            while True:
                try:
                    # 从队列获取音频文件路径
                    audio_file_path = self.audio_queue.get(timeout=1)  # 1秒超时
                    
                    if audio_file_path and os.path.exists(audio_file_path):
                        logger.info(f"开始播放音频文件: {audio_file_path}")
                        self._play_audio_file_sync(audio_file_path)
                    else:
                        logger.warning(f"音频文件不存在或为空: {audio_file_path}")
                        
                except Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    logger.error(f"音频播放工作线程错误: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"音频播放工作线程异常: {e}")
        finally:
            try:
                pygame.mixer.quit()
                pygame.quit()
            except:
                pass

    def _play_audio_file_sync(self, file_path: str):
        """同步播放音频文件"""
        try:
            import pygame
            import time
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"音频文件不存在: {file_path}")
                return
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            logger.info(f"音频文件大小: {file_size} 字节")
            
            # 加载并播放音频文件
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            logger.info(f"音频播放完成: {file_path}")
            
            # 播放完成后从使用列表中移除
            self.audio_files_in_use.discard(file_path)
            
        except Exception as e:
            logger.error(f"播放音频文件失败: {e}")

    def _audio_cleanup_worker(self):
        """音频文件清理工作线程"""
        logger.info("音频文件清理工作线程启动")
        
        while True:
            try:
                time.sleep(30)  # 每30秒清理一次
                
                # 获取所有音频文件
                audio_files = list(self.audio_temp_dir.glob(f"*.{config.tts.default_format}"))
                
                # 清理不在使用中的文件
                files_to_clean = []
                for file_path in audio_files:
                    if str(file_path) not in self.audio_files_in_use:
                        files_to_clean.append(file_path)
                
                if files_to_clean:
                    logger.info(f"开始清理 {len(files_to_clean)} 个音频文件")
                    for file_path in files_to_clean:
                        try:
                            file_path.unlink()
                        except Exception as e:
                            logger.warning(f"删除音频文件失败: {file_path} - {e}")
                    
                    logger.info(f"音频文件清理完成，共清理 {len(files_to_clean)} 个文件")
                    
            except Exception as e:
                logger.error(f"音频文件清理异常: {e}")
                time.sleep(5)

def get_voice_integration() -> VoiceIntegration:
    """获取语音集成实例"""
    if not hasattr(get_voice_integration, '_instance'):
        get_voice_integration._instance = VoiceIntegration()
    return get_voice_integration._instance