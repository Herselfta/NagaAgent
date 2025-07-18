#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音集成模块 - 负责接收文本并调用TTS服务播放音频
支持Edge TTS和Minimax TTS两种服务
"""
import asyncio
import json
import logging
import requests
import base64
import tempfile
import os
import ssl
import threading
import time
from typing import Optional, List
import aiohttp
import websockets
from pathlib import Path
from io import BytesIO
import asyncio
# 添加项目根目录到路径
import sys
import pyaudio
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("VoiceIntegration")
executor = ThreadPoolExecutor(max_workers=1)


class VoiceIntegration:
    """语音集成类 - 负责文本接收和TTS播放，支持多种TTS服务"""
    
    def __init__(self):
        self.enabled = config.system.voice_enabled
        self.provider = getattr(config.tts, 'provider', 'edge-tts')
        
        # 通用配置
        self.tts_url = f"http://127.0.0.1:{config.tts.port}/v1/audio/speech"
        self._last_text = None  # 用来记录上一次播放的文本
        self._playing_texts = set()  # 正在播放的文本集合（用于防重复）
        self._text_lock = threading.Lock()  # 线程锁
        # self._call_counter = {}  # 调用计数器，用于调试 - 已注释
        self.text_buffer = []  # 文本缓冲区
        self.sentence_endings = ['.', '!', '?', '。', '！', '？', '；', ';']
        self.min_sentence_length = 10  # 最小句子长度
        self.max_buffer_size = 5  # 最大缓冲区大小
        
        # Minimax配置
        self.api_key = getattr(config.tts, 'api_key', '')
        self.group_id = getattr(config.tts, 'group_id', '')
        self.minimax_url = f"https://api.minimaxi.com/v1/t2a_v2?GroupId={self.group_id}"
        self.tts_model = getattr(config.tts, 'tts_model', 'speech-02-hd')
        self.default_voice = getattr(config.tts, 'default_voice', 'male-qn-qingse')
        self.emotion = getattr(config.tts, 'emotion', 'happy')
        self.headers = {
                    'accept': 'application/json, text/plain, */*',
                    'content-type': 'application/json',
                    'authorization': f"Bearer {self.api_key}",
                }

        # 验证Minimax配置
        if self.provider == 'minimax' and (not self.api_key or not self.group_id):
            logger.warning("Minimax配置不完整，切换到Edge TTS")
            self.provider = 'edge-tts'
        
        logger.info(f"语音集成初始化完成，使用提供商: {self.provider}")


    def receive_final_text(self, final_text: str):
        """接收最终完整文本"""
        if not self.enabled:
            return
            
        if final_text and final_text.strip():
            final_text_key = final_text.strip()
            # text_hash = hash(final_text_key)
            
            # 添加调用栈信息 - 调试用，已注释
            # import traceback
            # caller_info = traceback.extract_stack()[-2]
            # logger.info(f"📝 接收最终文本 - 哈希: {text_hash}, 长度: {len(final_text_key)}, 调用者: {caller_info.filename}:{caller_info.lineno}")
            
            # 检查是否已经在播放队列中
            with self._text_lock:
                if final_text_key in self._playing_texts:
                    logger.debug(f"最终文本已在播放队列中，跳过")
                    return
            
            logger.info(f"接收最终文本: {final_text[:100]}...")
            # 清空缓冲区，避免重复播放
            self.text_buffer = []
            # 在后台线程播放最终文本
            self._play_text_in_background(final_text)

    def receive_text_chunk(self, text: str):
        """接收文本片段"""
        if not self.enabled:
            return
            
        if text and text.strip():
            # chunk_hash = hash(text.strip())
            self.text_buffer.append(text.strip())
            logger.debug(f"接收文本片段: {text[:50]}..., 缓冲区大小: {len(self.text_buffer)}")
            
            # 检查是否有完整句子
            self._check_and_play_sentences()
    
    def _check_and_play_sentences(self):
        """检查并播放完整句子"""
        # if len(self.text_buffer) < 1:
        #     return
            
        # 合并缓冲区文本
        combined_text = ' '.join(self.text_buffer)
        
        # 查找句子结束位置
        sentence_end_pos = -1
        for ending in self.sentence_endings:
            pos = combined_text.rfind(ending)
            if pos > sentence_end_pos:
                sentence_end_pos = pos


        # 防止缓冲区过大
        if len(self.text_buffer) > self.max_buffer_size:
            # 强制播放缓冲区内容
            forced_text = ' '.join(self.text_buffer)
            self._play_text_in_background(forced_text)
            self.text_buffer = []

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
        

    
    async def _play_text(self, text: str):
        """播放文本音频"""
        try:
            # 根据配置选择TTS服务
            if self.provider == 'minimax':
                pass
            else:
                # 使用默认的Edge TTS或本地TTS API
                audio_data = await self._generate_audio(text)
            
                if audio_data:
                    await self._play_audio(audio_data)
                    logger.info(f"成功播放音频 ({self.provider}): {text[:50]}...")
                else:
                    logger.warning(f"音频生成失败 ({self.provider}): {text[:50]}...")
                
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
            # 如果Minimax失败，尝试回退到Edge TTS
            if self.provider == 'minimax':
                logger.info("尝试回退到Edge TTS")
                try:
                    audio_data = await self._generate_audio(text)
                    if audio_data:
                        await self._play_audio(audio_data)
                        logger.info(f"回退播放成功: {text[:50]}...")
                except Exception as fallback_error:
                    logger.error(f"回退播放也失败: {fallback_error}")
    
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
            # 尝试使用pydub播放（更好的音频处理）
            if await self._play_with_pyaudio(audio_data):
                return
            
            # 回退到文件播放方式
            with tempfile.NamedTemporaryFile(suffix=f".{config.tts.default_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # 使用pygame播放
            await self._play_with_pygame(temp_file_path)
            
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"播放音频文件失败: {e}")
    

            
        except ImportError:
            logger.debug("pydub未安装，使用备选播放方案")
            return False
        except Exception as e:
            logger.warning(f"pydub播放失败: {e}")
            return False
    
    async def _play_audio_file(self, file_path: str):
        """播放音频文件"""
        try:
            import platform
            import subprocess

            
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
        text_key = text.strip()
        # text_hash = hash(text_key)
        
        # logger.info(f"🎭 准备后台播放 - 哈希: {text_hash}")
        
        # 提前检查避免创建不必要的线程
        with self._text_lock:
            if text_key in self._playing_texts:
                logger.debug(f"文本已在播放队列中，跳过线程创建")
                return
        
        if self.provider == 'minimax':
            def run_in_thread():
                try:
                    # logger.info(f"🚀 启动 Minimax 播放线程 - 哈希: {text_hash}")
                    asyncio.run(self.tts_and_play(text))
                except Exception as e:
                    logger.error(f"后台播放音频失败: {e}")
        else:    
            def run_in_thread():
                """在线程中运行异步函数"""
                try:
                    # logger.info(f"🚀 启动 Edge TTS 播放线程 - 哈希: {text_hash}")
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
        # logger.info(f"🧵 线程已启动 - 哈希: {text_hash}")


    async def tts_and_play(self, text: str):
        # 使用线程锁和集合来防止重复播放
        text_key = text.strip()
        # text_hash = hash(text_key)  # 调试用，已注释
        
        # logger.info(f"🎵 TTS请求 - 文本哈希: {text_hash}, 长度: {len(text_key)}, 内容: '{text_key[:100]}...'")
        
        with self._text_lock:
            if text_key in self._playing_texts:
                logger.debug(f"跳过重复的 TTS 请求")
                return
            self._playing_texts.add(text_key)
            # logger.info(f"✅ 添加到播放队列 - 哈希: {text_hash}, 队列大小: {len(self._playing_texts)}")
        
        try:
            # logger.info(f"🚀 开始 TTS 播放 - 哈希: {text_hash}")
            loop = asyncio.get_running_loop()
            
            # 创建音频生成器（只创建一次）
            # logger.info(f"🏭 创建音频生成器 - 哈希: {text_hash}")
            audio_iter = self._generate_minimax_audio(text)
            
            # 将生成器转换为列表，避免重复迭代问题
            # logger.info(f"📋 收集音频数据 - 哈希: {text_hash}")
            audio_chunks = list(audio_iter)
            #logger.info(f"开始播放音频，总音频块: {len(audio_chunks)}")
            
            # 播放音频
            await loop.run_in_executor(executor, self._audio_play_pyaudio, audio_chunks, text_key[:50])
            logger.info(f"TTS 播放完成: {text_key[:50]}...")
        except Exception as e:
            logger.error(f"TTS 播放失败: {e}")
        finally:
            # 播放完成或失败后，从正在播放集合中移除
            with self._text_lock:
                self._playing_texts.discard(text_key)
                # logger.info(f"🗑️ 从播放队列移除 - 哈希: {text_hash}, 队列大小: {len(self._playing_texts)}")

    def _audio_play_pyaudio(self, audio_chunks, text_preview=None):
        """用 PyAudio 实时播放 PCM hex-chunk"""
        if not audio_chunks:
            logger.warning(f"没有音频数据可播放")
            return
            
        # logger.info(f"🔊 开始音频播放 - 文本: {text_preview}, 音频块数量: {len(audio_chunks)}")
        
        seen = set()
        chunk_count = 0
        duplicate_count = 0
        played_count = 0
        
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,    # 与 16-bit PCM 对应
            channels=1,
            rate=32000,
            output=True
        )
        try:
            for hex_chunk in audio_chunks:
                if not hex_chunk:
                    continue
                    
                chunk_count += 1
                chunk_id = hex_chunk[:32] if len(hex_chunk) > 32 else hex_chunk  # 用前32字符作为chunk标识
                
                if chunk_id in seen:
                    duplicate_count += 1
                    # logger.warning(f"🔄 跳过重复音频块 #{chunk_count}, 重复次数: {duplicate_count}")
                    # 跳过重复的chunk - 防止重复播放
                    continue
                else:
                    seen.add(chunk_id)
                    
                try:
                    pcm_bytes = bytes.fromhex(hex_chunk)
                    stream.write(pcm_bytes)
                    played_count += 1
                    
                    # if played_count % 5 == 0:  # 每5个有效chunk记录一次 - 调试用
                    #     logger.debug(f"📦 播放音频块 #{played_count} (原始#{chunk_count}/{len(audio_chunks)})")
                        
                except ValueError as e:
                    logger.error(f"hex解析失败 chunk #{chunk_count}: {e}")
                except Exception as e:
                    logger.error(f"播放 chunk #{chunk_count} 失败: {e}")
                    
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            
            # logger.info(f"🏁 音频播放结束")
            # logger.info(f"   📊 统计: 总块数: {chunk_count}, 有效播放: {played_count}, 重复跳过: {duplicate_count}")
            
            if duplicate_count > 0:
                logger.debug(f"检测到并跳过了{duplicate_count}个重复音频块")
                # logger.warning(f"⚠️ 检测到Minimax API返回了{duplicate_count}个重复音频块，已自动跳过避免重复播放")



    def build_tts_stream_body(self, text: str) -> str:
        # 流式请求
        return json.dumps({
            "model": self.tts_model,
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": "danya_xuejie",
                "speed": config.tts.default_speed, "vol": 1.0, "pitch": 0,
                "emotion": self.emotion
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": config.tts.default_format,    # 这里改成 pcm
                "channel": 1
            }
        })

    def _generate_minimax_audio(self, text: str):
        """向 TTS 接口发请求，yield PCM 数据的十六进制字符串"""
        # text_hash = hash(text.strip())
        
        # 调用计数检测 - 调试用，已注释
        # with self._text_lock:
        #     call_count = self._call_counter.get(text_hash, 0) + 1
        #     self._call_counter[text_hash] = call_count
        #     if call_count > 1:
        #         logger.error(f"🚨 检测到重复API调用！- 哈希: {text_hash}, 调用次数: {call_count}")
        
        # logger.info(f"🌐 开始请求 Minimax TTS - 哈希: {text_hash}, 调用次数: {call_count}")
        
        # 用于保存调试信息 - 调试用，已注释
        # debug_chunks = []
        # audio_chunks_received = []
        
        try:
            resp = requests.post(self.minimax_url, headers=self.headers,
                                data=self.build_tts_stream_body(text), stream=True)
            
            # logger.info(f"📡 TTS API 响应状态: {resp.status_code} - 哈希: {text_hash}")
            
            if resp.status_code != 200:
                logger.error(f"TTS API 请求失败: {resp.status_code} - {resp.text}")
                return
            
            chunk_count = 0
            audio_chunk_count = 0
            
            for chunk in resp.raw:
                chunk_count += 1
                if not chunk or not chunk.startswith(b'data:'):
                    continue
                    
                try:
                    # 保存原始chunk用于调试 - 已注释
                    # debug_chunks.append(chunk.decode('utf-8', errors='ignore'))
                    
                    payload = json.loads(chunk[5:])
                    data = payload.get("data", {})
                    if "audio" in data:
                        audio_chunk_count += 1
                        audio_data = data["audio"]
                        
                        # 保存音频数据和相关信息用于分析 - 调试用，已注释
                        # audio_info = {
                        #     'index': audio_chunk_count,
                        #     'audio_data': audio_data,
                        #     'audio_length': len(audio_data),
                        #     'chunk_id': audio_data[:32] if len(audio_data) > 32 else audio_data,
                        #     'full_payload': payload
                        # }
                        # audio_chunks_received.append(audio_info)
                        
                        # if audio_chunk_count % 5 == 0:  # 每5个audio chunk记录一次
                        #     logger.debug(f"🎶 收到音频数据 #{audio_chunk_count} - 哈希: {text_hash}, 长度: {len(audio_data)}")
                        yield audio_data  # 这是一个 hex string
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败 - chunk #{chunk_count}: {e}")
                except Exception as e:
                    logger.error(f"处理chunk #{chunk_count}失败: {e}")
                    
            logger.debug(f"TTS流式响应完成 - 总chunks: {chunk_count}, 音频chunks: {audio_chunk_count}")
            
            # 分析音频数据重复情况 - 调试用，已注释
            # self._analyze_audio_chunks(audio_chunks_received, text_hash)
            
            # 保存调试数据到文件 - 调试用，已注释
            # self._save_debug_data(text_hash, debug_chunks, audio_chunks_received)
            
        except Exception as e:
            logger.error(f"TTS请求异常: {e}")
            raise
        # finally:
        #     # 清理调用计数
        #     with self._text_lock:
        #         if text_hash in self._call_counter:
        #             del self._call_counter[text_hash]
    

    
    def switch_provider(self, provider: str):
        """切换TTS服务提供商"""
        if provider not in ['edge-tts', 'minimax']:
            logger.error(f"不支持的TTS提供商: {provider}")
            return False
        
        if provider == 'minimax' and (not self.api_key or not self.group_id):
            logger.error("Minimax配置不完整，无法切换")
            return False
        
        old_provider = self.provider
        self.provider = provider
        config.tts.provider = provider
        
        logger.info(f"TTS提供商已从 {old_provider} 切换到 {provider}")
        return True
    
    def get_provider_info(self) -> dict:
        """获取当前提供商信息"""
        info = {
            "current_provider": self.provider,
            "enabled": self.enabled,
            "available_providers": []
        }
        
        # 检查可用的提供商
        info["available_providers"].append("edge-tts")
        
        if self.api_key and self.group_id:
            info["available_providers"].append("minimax")
        
        # 添加配置信息
        if self.provider == 'minimax':
            info["minimax_config"] = {
                "model": self.tts_model,
                "voice_id": self.default_voice,
                "emotion": self.emotion,
                "api_key_configured": bool(self.api_key),
                "group_id_configured": bool(self.group_id)
            }
        
        return info
    
    def set_minimax_config(self, voice_id: str = None, emotion: str = None, model: str = None):
        """动态设置Minimax配置"""
        changed = []
        
        if voice_id and voice_id != self.default_voice:
            self.default_voice = voice_id
            config.tts.default_voice = voice_id
            changed.append(f"voice_id: {voice_id}")
        
        if emotion and emotion != self.minimax_emotion:
            self.minimax_emotion = emotion
            config.tts.minimax_emotion = emotion
            changed.append(f"emotion: {emotion}")
        
        if model and model != self.tts_model:
            self.tts_model = model
            config.tts.tts_model = model
            changed.append(f"model: {model}")
        
        if changed:
            logger.info(f"Minimax配置已更新: {', '.join(changed)}")
            return True
        return False
    
    async def test_provider(self, provider: str = None) -> bool:
        """测试TTS提供商是否可用"""
        test_provider = provider or self.provider
        test_text = "这是一个TTS服务测试。"
        
        try:
            if test_provider == 'minimax':
                audio_data = await self._generate_minimax_audio(test_text)
            else:
                audio_data = await self._generate_audio(test_text)
            
            success = audio_data is not None and len(audio_data) > 0
            logger.info(f"TTS提供商 {test_provider} 测试{'成功' if success else '失败'}")
            return success
            
        except Exception as e:
            logger.error(f"TTS提供商 {test_provider} 测试异常: {e}")
            return False

    # === 以下是调试方法，正常使用时可以注释掉 ===
    
    def debug_test_duplicate(self, test_text: str = "这是一个测试文本。"):
        """调试测试重复播放问题"""
        logger.info("🧪 开始调试测试...")
        
        # 清理状态
        with self._text_lock:
            self._playing_texts.clear()
            # self._call_counter.clear()
            
        # 模拟接收最终文本
        logger.info("🧪 步骤1: 调用 receive_final_text")
        self.receive_final_text(test_text)
        
        # 等待一段时间
        import time
        time.sleep(3)
        
        logger.info("🧪 调试测试完成")
        return True
    
    def analyze_saved_debug_data(self, text_hash=None):
        """分析已保存的调试数据"""
        try:
            debug_dir = "logs/voice_debug"
            if not os.path.exists(debug_dir):
                logger.warning("调试数据目录不存在")
                return
            
            files = os.listdir(debug_dir)
            debug_files = [f for f in files if f.startswith('tts_debug_')]
            
            if not debug_files:
                logger.warning("没有找到调试数据文件")
                return
            
            if text_hash:
                target_file = f"tts_debug_{text_hash}.json"
                if target_file not in debug_files:
                    logger.warning(f"未找到指定哈希的调试文件: {target_file}")
                    return
                debug_files = [target_file]
            
            logger.info(f"📁 找到 {len(debug_files)} 个调试文件")
            
            for debug_file in debug_files:
                file_path = os.path.join(debug_dir, debug_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    logger.info(f"📄 分析文件: {debug_file}")
                    logger.info(f"   哈希: {data.get('text_hash')}")
                    logger.info(f"   时间戳: {data.get('timestamp')}")
                    logger.info(f"   总原始chunks: {data.get('total_raw_chunks')}")
                    logger.info(f"   总音频chunks: {data.get('total_audio_chunks')}")
                    
                    # 分析重复情况
                    audio_chunks = data.get('audio_chunks_summary', [])
                    chunk_ids = [chunk['chunk_id'] for chunk in audio_chunks]
                    unique_chunks = len(set(chunk_ids))
                    duplicates = len(chunk_ids) - unique_chunks
                    
                    if duplicates > 0:
                        logger.warning(f"   🔄 发现重复: {duplicates} 个重复音频块")
                    else:
                        logger.info(f"   ✅ 无重复音频块")
                        
                except Exception as e:
                    logger.error(f"❌ 分析文件失败 {debug_file}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ 分析调试数据失败: {e}")

    def _analyze_audio_chunks(self, audio_chunks_received, text_hash):
        """分析音频chunk重复情况"""
        logger.info(f"🔍 开始分析音频数据 - 哈希: {text_hash}")
        
        chunk_ids = []
        duplicates = []
        
        for i, chunk_info in enumerate(audio_chunks_received):
            chunk_id = chunk_info['chunk_id']
            if chunk_id in chunk_ids:
                # 找到重复的chunk
                original_index = chunk_ids.index(chunk_id) + 1
                duplicates.append({
                    'original_index': original_index,
                    'duplicate_index': i + 1,
                    'chunk_id': chunk_id,
                    'audio_length': chunk_info['audio_length']
                })
                logger.warning(f"🔄 API返回重复音频块: 原始#{original_index}, 重复#{i + 1}, chunk_id: {chunk_id[:16]}...")
            else:
                chunk_ids.append(chunk_id)
        
        if duplicates:
            logger.error(f"🚨 Minimax API返回了 {len(duplicates)} 个重复音频块！- 哈希: {text_hash}")
            for dup in duplicates:
                logger.error(f"   重复: #{dup['original_index']} -> #{dup['duplicate_index']}, 长度: {dup['audio_length']}")
        else:
            logger.info(f"✅ 音频数据无重复 - 哈希: {text_hash}")
    
    def _save_debug_data(self, text_hash, debug_chunks, audio_chunks_received):
        """保存调试数据到文件"""
        try:
            debug_dir = "logs/voice_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            debug_file = f"{debug_dir}/tts_debug_{text_hash}.json"
            
            debug_data = {
                'text_hash': text_hash,
                'timestamp': str(time.time()),
                'total_raw_chunks': len(debug_chunks),
                'total_audio_chunks': len(audio_chunks_received),
                'raw_chunks': debug_chunks[:30],  # 只保存前30个原始chunk避免文件过大
                'audio_chunks_summary': [
                    {
                        'index': chunk['index'],
                        'audio_length': chunk['audio_length'],
                        'chunk_id': chunk['chunk_id'],
                        'audio_preview': chunk['audio_data'][:5000]  # 只保存前5000字符
                    }
                    for chunk in audio_chunks_received
                ]
            }
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 调试数据已保存: {debug_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存调试数据失败: {e}")

# 全局实例
_voice_integration_instance: Optional[VoiceIntegration] = None

def get_voice_integration() -> VoiceIntegration:
    """获取语音集成实例（单例模式）"""
    global _voice_integration_instance
    if _voice_integration_instance is None:
        _voice_integration_instance = VoiceIntegration()
    return _voice_integration_instance

def switch_tts_provider(provider: str) -> bool:
    """全局切换TTS提供商"""
    voice = get_voice_integration()
    return voice.switch_provider(provider)

def get_tts_provider_info() -> dict:
    """获取TTS提供商信息"""
    voice = get_voice_integration()
    return voice.get_provider_info()

async def test_tts_provider(provider: str = None) -> bool:
    """测试TTS提供商"""
    voice = get_voice_integration()
    return await voice.test_provider(provider)

def set_minimax_voice_config(voice_id: str = None, emotion: str = None, model: str = None) -> bool:
    """设置Minimax语音配置"""
    voice = get_voice_integration()
    return voice.set_minimax_config(voice_id, emotion, model)