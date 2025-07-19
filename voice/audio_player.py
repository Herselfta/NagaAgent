#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频播放器模块 - 提供跨平台音频播放功能
"""
import asyncio
import logging
import platform
import subprocess
import tempfile
import os
from typing import Optional, Union
from pathlib import Path
from asyncio.subprocess import Process

logger = logging.getLogger("AudioPlayer")

class AudioPlayer:
    """跨平台音频播放器"""
    
    def __init__(self):
        self.system = platform.system()
        self._current_process: Optional[Process] = None
        
    async def play_audio_file(self, file_path: Union[str, Path]) -> bool:
        """播放音频文件"""
        try:
            file_path = str(file_path)
            if not os.path.exists(file_path):
                logger.error(f"音频文件不存在: {file_path}")
                return False
            
            # 停止当前播放
            await self.stop()
            
            # 根据系统选择播放方式
            if self.system == "Windows":
                return await self._play_windows(file_path)
            elif self.system == "Darwin":  # macOS
                return await self._play_macos(file_path)
            elif self.system == "Linux":
                return await self._play_linux(file_path)
            else:
                logger.warning(f"不支持的操作系统: {self.system}")
                return False
                
        except Exception as e:
            logger.error(f"播放音频文件失败: {e}")
            return False
    
    async def play_audio_data(self, audio_data: bytes, format: str = "mp3") -> bool:
        """播放音频数据"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # 播放临时文件
            success = await self.play_audio_file(temp_file_path)
            
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            return success
            
        except Exception as e:
            logger.error(f"播放音频数据失败: {e}")
            return False
    
    async def stop(self):
        """停止当前播放"""
        if self._current_process and self._current_process.returncode is None:
            try:
                self._current_process.terminate()
                await asyncio.wait_for(self._current_process.wait(), timeout=2.0)
            except ProcessLookupError:
                # 进程已经结束，这是正常情况，无需处理
                pass
            except asyncio.TimeoutError:
                self._current_process.kill()
                await self._current_process.wait()
            except Exception as e:
                logger.warning(f"停止播放时出错: {e!r}")
            finally:
                self._current_process = None
    
    async def _play_windows(self, file_path: str) -> bool:
        """Windows系统播放"""
        try:
            cmd = f'start "" "{file_path}"'
            self._current_process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            return True
        except Exception as e:
            logger.error(f"Windows播放失败: {e}")
            return False
    
    async def _play_macos(self, file_path: str) -> bool:
        """macOS系统播放"""
        try:
            # 优先使用 afplay，这是一个会等待播放完成的命令行工具
            self._current_process = await asyncio.create_subprocess_exec(
                "afplay", file_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            return True
        except FileNotFoundError:
            logger.error("macOS 播放失败: 'afplay' 命令未找到。这是 macOS 自带的命令，请检查您的系统。")
            return False
        except Exception as e:
            logger.error(f"macOS 使用 'afplay' 播放失败: {e!r}")
            return False
    
    async def _play_linux(self, file_path: str) -> bool:
        """Linux系统播放"""
        try:
            # 尝试多种播放器
            players = ["xdg-open", "aplay", "paplay", "mpg123", "ffplay"]
            
            for player in players:
                try:
                    self._current_process = await asyncio.create_subprocess_exec(
                        player, file_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    return True
                except FileNotFoundError:
                    continue
            
            logger.error("Linux系统未找到可用的音频播放器")
            return False
            
        except Exception as e:
            logger.error(f"Linux播放失败: {e}")
            return False
    
    def is_playing(self) -> bool:
        """检查是否正在播放"""
        if self._current_process:
            return self._current_process.returncode is None
        return False

# 全局实例
_audio_player_instance: Optional[AudioPlayer] = None

def get_audio_player() -> AudioPlayer:
    """获取音频播放器实例（单例模式）"""
    global _audio_player_instance
    if _audio_player_instance is None:
        _audio_player_instance = AudioPlayer()
    return _audio_player_instance