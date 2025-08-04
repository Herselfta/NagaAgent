#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音集成模块 - 负责接收文本并调用TTS服务播放音频
支持Edge TTS和Minimax TTS两种服务
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
from typing import Optional, List
import aiohttp
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logger = logging.getLogger("VoiceIntegration")

# 断句正则表达式 - 从WebSocket移植
SENTENCE_END_PUNCTUATIONS = r"[。？！；\.\?\!\;]"

class VoiceIntegration:
    """语音集成模块 - 支持智能分句和并发合成"""
    
    def __init__(self):
        self.provider = 'edge_tts'  # 默认使用Edge TTS
        self.tts_url = f"http://127.0.0.1:{config.tts.port}/v1/audio/speech"
        
        # 智能分句配置 - 从WebSocket移植
        self.sentence_endings = ['。', '！', '？', '；', '.', '!', '?', ';']
        self.min_sentence_length = 5  # 最小句子长度
        self.max_buffer_size = 50  # 最大缓冲区大小
        self.max_concurrent_tasks = 3  # 最大并发任务数
        
        # 文本缓冲区和播放队列
        self.text_buffer = []
        self.audio_queue = asyncio.Queue()
        self.playing_lock = threading.Lock()
        self.playing_texts = set()  # 防止重复播放
        
        # 音频文件存储目录
        self.audio_temp_dir = Path("logs/audio_temp")
        self.audio_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # pygame音频初始化
        self._init_pygame_audio()
        
        # 启动音频播放工作线程
        self.audio_thread = threading.Thread(target=self._audio_player_worker, daemon=True)
        self.audio_thread.start()
        
        # 启动异步音频文件清理
        self._start_audio_cleanup()
        
        logger.info("语音集成模块初始化完成")

    def _init_pygame_audio(self):
        """初始化pygame音频系统"""
        try:
            import pygame
            # 先初始化pygame
            pygame.init()
            
            # 尝试使用指定参数初始化音频
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                logger.info("pygame音频系统初始化成功（使用指定参数）")
            except Exception as e:
                logger.warning(f"使用指定参数初始化失败，尝试默认参数: {e}")
                # 如果指定参数失败，尝试默认参数
                pygame.mixer.init()
                logger.info("pygame音频系统初始化成功（使用默认参数）")
            
            self.pygame_available = True
            logger.info(f"pygame版本: {pygame.version.ver}")
            
        except ImportError:
            logger.error("pygame未安装，语音播放功能不可用")
            logger.error("请安装pygame: pip install pygame")
            self.pygame_available = False
        except Exception as e:
            logger.error(f"pygame音频初始化失败: {e}")
            self.pygame_available = False

    def receive_final_text(self, final_text: str):
        """接收最终完整文本 - """
        if not config.system.voice_enabled:
            return
            
        if final_text and final_text.strip():
            logger.info(f"接收最终文本: {final_text[:100]}")
            # 不再播放完整文本，避免与流式分句重复
            # self._play_text_in_background(final_text)  # 注释掉这行

    def receive_text_chunk(self, text: str):
        """接收文本片段 - 使用智能分句逻辑"""
        if not config.system.voice_enabled:
            return
            
        if text and text.strip():
            self.text_buffer.append(text.strip())
            logger.debug(f"接收文本片段: {text[:50]}..., 缓冲区大小: {len(self.text_buffer)}")
            
            # 检查是否有完整句子
            self._check_and_play_sentences()

    def _check_and_play_sentences(self):
        """智能分句检查并播放 - 移植WebSocket逻辑"""
        if len(self.text_buffer) < 1:
            return
            
        # 合并缓冲区文本
        combined_text = ' '.join(self.text_buffer)
        logger.debug(f"合并文本: {combined_text}")
        
        # 使用WebSocket的分句逻辑
        raw_sentences = re.split(SENTENCE_END_PUNCTUATIONS, combined_text)
        logger.debug(f"分割为{len(raw_sentences)}个原始句子: {raw_sentences}")
        
        processed_sentences = []
        
        # 处理最后一个不完整句子
        if raw_sentences and not raw_sentences[-1]:
            raw_sentences.pop()
            remaining_buffer = ""
        elif raw_sentences:
            remaining_buffer = raw_sentences.pop()
        else:
            remaining_buffer = ""
        
        # 智能短句合并 - 移植WebSocket逻辑
        temp_sentence_buffer = ""
        for sentence in raw_sentences:
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue
                
            # 检查是否包含引号
            has_quotes = re.search(r'["\'\'""\']', sentence_stripped)
            
            # 短句合并逻辑：长度<=5且不包含引号的句子进行合并
            if len(sentence_stripped) <= 5 and not has_quotes:
                temp_sentence_buffer += sentence_stripped
                logger.debug(f"合并短句: '{sentence_stripped}' -> 缓冲区: '{temp_sentence_buffer}'")
            else:
                if temp_sentence_buffer:
                    final_sentence = temp_sentence_buffer + sentence_stripped
                    processed_sentences.append(final_sentence)
                    logger.debug(f"添加合并句子: '{final_sentence}'")
                    temp_sentence_buffer = ""
                else:
                    processed_sentences.append(sentence_stripped)
                    logger.debug(f"添加完整句子: '{sentence_stripped}'")
        
        # 处理剩余的短句缓冲区
        if temp_sentence_buffer:
            processed_sentences.append(temp_sentence_buffer)
            logger.debug(f"添加剩余短句: '{temp_sentence_buffer}'")
        
        logger.debug(f"处理后的句子: {processed_sentences}")
        
        # 在后台线程中运行异步任务
        if processed_sentences:
            threading.Thread(
                target=self._run_async_in_thread,
                args=(processed_sentences,),
                daemon=True
            ).start()
        
        # 更新缓冲区
        if remaining_buffer:
            self.text_buffer = [remaining_buffer]
            logger.debug(f"更新缓冲区: '{remaining_buffer}'")
        else:
            self.text_buffer = []
            logger.debug("清空缓冲区")
        
        # 防止缓冲区过大
        if len(self.text_buffer) > self.max_buffer_size:
            forced_text = ' '.join(self.text_buffer)
            logger.warning(f"缓冲区过大，强制播放: {forced_text[:50]}...")
            self._play_text_in_background(forced_text)
            self.text_buffer = []

    async def _generate_audio_files_concurrently(self, sentences: List[str]):
        """并发生成音频文件"""
        if not sentences:
            return
            
        logger.info(f"开始并发生成{len(sentences)}个音频文件")
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # 创建并发任务
        tasks = []
        for i, sentence in enumerate(sentences):
            if sentence and len(sentence.strip()) >= self.min_sentence_length:
                task = asyncio.create_task(
                    self._generate_audio_file_with_semaphore(semaphore, sentence, i)
                )
                tasks.append(task)
        
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"所有音频文件生成完成，共{len(tasks)}个文件")

    async def _generate_audio_file_with_semaphore(self, semaphore: asyncio.Semaphore, text: str, index: int):
        """使用信号量控制并发生成音频文件"""
        async with semaphore:
            try:
                audio_file_path = await self._generate_audio_file(text, index)
                if audio_file_path:
                    # 加入播放队列
                    await self.audio_queue.put(audio_file_path)
                    logger.info(f"音频文件已加入播放队列: {text[:50]}... -> {audio_file_path}")
                else:
                    logger.warning(f"音频文件生成失败: {text[:50]}...")
            except Exception as e:
                logger.error(f"生成音频文件异常: {e}")

    async def _generate_audio_file(self, text: str, index: int) -> Optional[str]:
        """生成音频文件并保存到本地"""
        try:
            # 文本预处理
            if not getattr(config.tts, 'remove_filter', False):
                from voice.handle_text import prepare_tts_input_with_context
                text = prepare_tts_input_with_context(text)
            
            # 生成文件名
            timestamp = int(time.time() * 1000)
            filename = f"tts_audio_{timestamp}_{index}.{config.tts.default_format}"
            file_path = self.audio_temp_dir / filename
            
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
                        audio_data = await response.read()
                        
                        # 保存到本地文件
                        with open(file_path, 'wb') as f:
                            f.write(audio_data)
                        
                        logger.debug(f"音频文件已保存: {file_path} ({len(audio_data)} bytes)")
                        return str(file_path)
                    else:
                        error_text = await response.text()
                        logger.error(f"TTS API调用失败: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"生成音频文件异常: {e}")
            return None

    def _run_async_in_thread(self, sentences: List[str]):
        """在后台线程中运行异步任务"""
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 运行异步任务
            loop.run_until_complete(self._generate_audio_files_concurrently(sentences))
        except Exception as e:
            logger.error(f"后台异步任务执行失败: {e}")
        finally:
            # 清理事件循环
            try:
                loop.close()
            except:
                pass

    def _play_text_in_background(self, text: str):
        """在后台线程播放文本"""
        try:
            # 检查是否正在播放相同文本
            text_hash = hashlib.md5(text.encode()).hexdigest()
            with self.playing_lock:
                if text_hash in self.playing_texts:
                    logger.debug(f"跳过重复播放: {text[:30]}...")
                    return
                self.playing_texts.add(text_hash)
            
            # 在后台线程中运行异步任务
            threading.Thread(
                target=self._run_async_in_thread,
                args=([text],),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"创建播放任务失败: {e}")

    def _audio_player_worker(self):
        """音频播放工作线程"""
        logger.info("音频播放工作线程启动，等待音频文件...")
        
        # 在工作线程中重新初始化pygame（参考测试脚本）
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
                    # 从队列获取音频文件路径并播放
                    audio_file_path = self.audio_queue.get_nowait()  # 使用非阻塞方式
                    logger.info(f"从队列获取到音频文件: {audio_file_path}")
                    
                    if audio_file_path:
                        logger.info(f"开始播放音频文件: {audio_file_path}")
                        # 直接调用同步播放方法
                        self._play_audio_file_sync(audio_file_path)
                    else:
                        logger.warning("获取到空的音频文件路径")
                except asyncio.QueueEmpty:
                    # 队列为空，等待一下再继续
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"音频播放工作线程错误: {e}")
                    import traceback
                    traceback.print_exc()
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
        """同步播放音频文件（参考测试脚本）"""
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
            
            # 加载音频文件
            pygame.mixer.music.load(file_path)
            
            # 播放音频
            pygame.mixer.music.play()
            logger.info(f"开始播放: {file_path}")
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            logger.info(f"音频播放完成: {file_path}")
            
            # 播放完成后，等待更长时间确保文件不再被占用
            time.sleep(1.0)
            
        except Exception as e:
            logger.error(f"同步播放失败: {e}")
            import traceback
            traceback.print_exc()




    def _start_audio_cleanup(self):
        """启动异步音频文件清理"""
        def cleanup_worker():
            """音频文件清理工作线程"""
            import time
            import glob
            
            while True:
                try:
                    # 检查是否启用文件保留
                    if getattr(config.tts, 'keep_audio_files', False):
                        time.sleep(60)  # 如果保留文件，每分钟检查一次
                        continue
                    
                    # 清理所有音频文件
                    audio_pattern = str(self.audio_temp_dir / "*.mp3")
                    audio_files = glob.glob(audio_pattern)
                    
                    if audio_files:
                        logger.info(f"开始清理 {len(audio_files)} 个音频文件")
                        
                        for file_path in audio_files:
                            try:
                                if os.path.exists(file_path):
                                    os.unlink(file_path)
                                    logger.debug(f"已删除音频文件: {file_path}")
                            except PermissionError:
                                logger.debug(f"文件被占用，跳过删除: {file_path}")
                            except Exception as e:
                                logger.warning(f"删除音频文件失败: {file_path}, 错误: {e}")
                        
                        logger.info(f"音频文件清理完成，共清理 {len(audio_files)} 个文件")
                    
                    # 每30秒清理一次
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"音频文件清理异常: {e}")
                    time.sleep(60)  # 出错时等待更长时间
        
        # 启动清理线程
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("音频文件清理线程已启动")

# 全局实例
_voice_integration_instance: Optional[VoiceIntegration] = None

def get_voice_integration() -> VoiceIntegration:
    """获取语音集成实例（单例模式）"""
    global _voice_integration_instance
    if _voice_integration_instance is None:
        _voice_integration_instance = VoiceIntegration()
    return _voice_integration_instance