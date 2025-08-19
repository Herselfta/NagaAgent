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
from typing import Optional, List, Dict, Any
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
        # 统一标准化TTS选择，兼容历史值（如 GPT-SoVITS -> GPT_SoVITS）
        tts_choice_raw = getattr(config.tts, 'TTS_CHOICE', 'DISABLE') or 'DISABLE'
        normalized = tts_choice_raw.replace('-', '_')
        self.tts_choice = normalized

        # 计算TTS服务URL
        if self.tts_choice.upper() == "GPT_SOVITS":
            gpt_cfg = getattr(config.tts, 'GPT_SoVITS', None)
            self.tts_url = getattr(gpt_cfg, 'gpt_sovits_api_url', 'http://127.0.0.1:9880/tts') if gpt_cfg else 'http://127.0.0.1:9880/tts'
        else:
            self.tts_url = f"http://127.0.0.1:{getattr(config.tts, 'port', 9880)}/v1/audio/speech"

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
        self.audio_files_in_queue = set()  # 在播放队列中等待的音频文件

        # 播放状态控制
        self.is_playing = False
        self.current_playback = None
        self._stop_event = threading.Event()

        # pygame音频初始化
        self._init_pygame_audio()

        # 启动音频播放工作线程
        self.audio_thread = threading.Thread(target=self._audio_player_worker, daemon=True)
        self.audio_thread.start()

        # 启动音频文件清理线程
        self.cleanup_thread = threading.Thread(target=self._audio_cleanup_worker, daemon=True)
        self.cleanup_thread.start()

        logger.info("语音集成模块初始化完成（重构版本）")

    def shutdown(self):
        """优雅关闭音频相关线程并清理临时文件。"""
        try:
            self._stop_event.set()
            # 尝试停止pygame播放
            try:
                import pygame
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
                pygame.quit()
            except Exception:
                pass
            # 尽力清理排队中文件
            with self.playing_lock:
                pending = list(self.audio_files_in_queue)
                self.audio_files_in_queue.clear()
            for p in pending:
                try:
                    if os.path.exists(p) and not getattr(config.tts, 'keep_audio_files', False):
                        os.remove(p)
                except Exception:
                    pass
            # 清理不在使用的临时文件
            if not getattr(config.tts, 'keep_audio_files', False):
                for wav in self.audio_temp_dir.glob('*.wav'):
                    try:
                        with self.playing_lock:
                            if str(wav) not in self.audio_files_in_use:
                                wav.unlink()
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"VoiceIntegration 关闭异常: {e}")

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
        if not getattr(config.system, 'voice_enabled', False):
            return
            
        if final_text and final_text.strip():
            logger.info(f"接收最终文本: {final_text[:100]}")
            # 立即开始音频处理，不阻塞前端显示
            self._process_audio_async(final_text)

    def receive_text_chunk(self, text: str):
        """接收文本片段 - 流式处理"""
        if not getattr(config.system, 'voice_enabled', False):
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
                # 加入播放队列并标记为队列中
                with self.playing_lock:
                    self.audio_queue.put(audio_file_path)
                    self.audio_files_in_queue.add(audio_file_path)
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
            filename = f"tts_audio_{timestamp}_{hash(text) % 1000}.wav"  # 始终使用wav格式
            file_path = self.audio_temp_dir / filename
            
            headers = {
                "accept": "audio/wav",
                "Content-Type": "application/json"
            }
            
            # 选择TTS模式
            if self.tts_choice.upper() == "GPT_SOVITS":
                gpt = getattr(config.tts, 'GPT_SoVITS', None)
                # 先检查是否启用
                if not gpt or not getattr(gpt, 'is_enabled', False):
                    logger.error("GPT-SoVITS TTS 未启用")
                    return None
                # 将模型配置字段映射到API期望字段
                payload = {
                    "text": text,
                    "text_lang": getattr(gpt, 'gpt_sovits_text_language', 'zh'),
                    "ref_audio_path": getattr(gpt, 'gpt_sovits_refer_wav_path', ''),
                    "aux_ref_audio_paths": getattr(gpt, 'aux_ref_audio_paths', []),
                    "prompt_text": getattr(gpt, 'gpt_sovits_prompt_text', ''),
                    "prompt_lang": getattr(gpt, 'gpt_sovits_prompt_language', 'zh'),
                    "top_k": getattr(gpt, 'top_k', 5),
                    "top_p": getattr(gpt, 'top_p', 1.0),
                    "temperature": getattr(gpt, 'temperature', 1.0),
                    "text_split_method": getattr(gpt, 'text_split_method', 'cut1'),
                    "batch_size": getattr(gpt, 'batch_size', 10),
                    "speed_factor": getattr(gpt, 'speed_factor', 1.0),
                    "media_type": "wav",
                    "streaming_mode": True,
                    # 附加参数
                    "batch_threshold": 0.75,
                    "split_bucket": True,
                    "fragment_interval": 0.3,
                    "seed": -1,
                    "parallel_infer": True,
                    "repetition_penalty": 1.35,
                    "sample_steps": 32,
                    "super_sampling": False
                }
            else:
                payload = {
                    "input": text,
                    "voice": getattr(config.tts, "default_voice", "zh-CN-XiaoxiaoNeural"),
                    "response_format": "wav",
                    "speed": getattr(config.tts, "default_speed", 1.0)
                }
            
            # 使用requests进行同步调用
            import requests
            response = requests.post(
                self.tts_url,
                json=payload,
                headers=headers,
                timeout=60  # 增加超时时间到60秒
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
        
        # 在工作线程中检查pygame是否已初始化
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.init()
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                logger.info("音频播放工作线程中pygame初始化成功")
        except Exception as e:
            logger.error(f"音频播放工作线程中pygame初始化失败: {e}")
            return
        
        try:
            while not self._stop_event.is_set():
                try:
                    # 从队列获取音频文件路径
                    audio_file_path = self.audio_queue.get(timeout=1)  # 缩短超时，便于退出
                    
                    if audio_file_path and os.path.exists(audio_file_path):
                        logger.info(f"开始播放音频文件: {audio_file_path}")
                        self._play_audio_file_sync(audio_file_path)
                    else:
                        logger.warning(f"音频文件不存在或为空: {audio_file_path}")
                        # 从队列集合中移除不存在的文件
                        with self.playing_lock:
                            self.audio_files_in_queue.discard(audio_file_path)
                        
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
                # 从队列集合中移除不存在的文件
                with self.playing_lock:
                    self.audio_files_in_queue.discard(file_path)
                return
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            logger.info(f"音频文件大小: {file_size} 字节")
            
            # 标记文件正在使用
            with self.playing_lock:
                self.audio_files_in_use.add(file_path)
                self.audio_files_in_queue.discard(file_path)  # 从队列中移除，开始播放
            
            # 停止当前正在播放的音频
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                time.sleep(0.1)  # 给一点时间让音频停止
            
            # 加载并播放音频文件
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # 等待播放完成
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                # 防止无限等待，设置最长播放时间（5分钟）
                if time.time() - start_time > 300:
                    logger.warning(f"音频播放超时，强制停止: {file_path}")
                    pygame.mixer.music.stop()
                    break
            
            logger.info(f"音频播放完成: {file_path}")
            
            # 播放完成后从使用列表中移除
            with self.playing_lock:
                self.audio_files_in_use.discard(file_path)
            
        except Exception as e:
            logger.error(f"播放音频文件失败: {e}")
            # 发生错误时也要清理状态
            with self.playing_lock:
                self.audio_files_in_use.discard(file_path)
                self.audio_files_in_queue.discard(file_path)

    def _audio_cleanup_worker(self):
        """音频文件清理工作线程"""
        logger.info("音频文件清理工作线程启动")
        
        while not self._stop_event.is_set():
            try:
                # 缩短检查间隔，使得退出更快
                for _ in range(30):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                
                # 获取所有音频文件
                audio_files = list(self.audio_temp_dir.glob("*.wav"))  # 只清理wav文件
                
                # 清理既不在使用中也不在队列中的文件
                files_to_clean = []
                files_to_keep = []
                
                with self.playing_lock:
                    for file_path in audio_files:
                        file_path_str = str(file_path)
                        if file_path_str not in self.audio_files_in_use and file_path_str not in self.audio_files_in_queue:
                            files_to_clean.append(file_path)
                        else:
                            files_to_keep.append(file_path)
                
                if files_to_clean:
                    logger.info(f"开始清理 {len(files_to_clean)} 个音频文件，保留 {len(files_to_keep)} 个文件")
                    for file_path in files_to_clean:
                        try:
                            # 再次检查文件状态，防止竞态条件
                            with self.playing_lock:
                                file_path_str = str(file_path)
                                if file_path_str not in self.audio_files_in_use and file_path_str not in self.audio_files_in_queue:
                                    file_path.unlink()
                                    logger.debug(f"已删除音频文件: {file_path}")
                                else:
                                    logger.debug(f"跳过正在使用的音频文件: {file_path}")
                        except Exception as e:
                            logger.warning(f"删除音频文件失败: {file_path} - {e}")
                    
                    logger.info(f"音频文件清理完成，共清理 {len(files_to_clean)} 个文件")
                else:
                    logger.debug(f"本次清理检查完成，无需要清理的文件")
                    
            except Exception as e:
                logger.error(f"音频文件清理异常: {e}")
                time.sleep(5)

    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        with self.playing_lock:
            return {
                "audio_files_in_use": len(self.audio_files_in_use),
                "audio_files_in_queue": len(self.audio_files_in_queue),
                "queue_size": self.audio_queue.qsize(),
                "playing_texts": len(self.playing_texts),
                "is_playing": self.is_playing,
                "temp_files": len(list(self.audio_temp_dir.glob(f"*.{config.tts.default_format}")))
            }

def get_voice_integration() -> VoiceIntegration:
    """获取语音集成实例"""
    if not hasattr(get_voice_integration, '_instance'):
        get_voice_integration._instance = VoiceIntegration()
    return get_voice_integration._instance
