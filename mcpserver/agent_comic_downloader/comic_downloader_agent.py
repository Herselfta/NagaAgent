"""
Comic Downloader MCP Agent
提供漫画下载功能的MCP代理
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    from .comic_downloader import ComicDownloader
except ImportError:
    from comic_downloader import ComicDownloader

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComicDownloaderAgent:
    """漫画下载器MCP代理"""
    
    def __init__(self):
        self.downloader = ComicDownloader()
        self.active_tasks = {}
        
    async def download_comic(self, album_id: str) -> Dict[str, Any]:
        """
        下载漫画
        
        Args:
            album_id: 漫画ID
            
        Returns:
            下载结果
        """
        try:
            logger.info(f"开始下载漫画: {album_id}")
            
            # 启动异步下载
            task_id = self.downloader.download_comic_async(album_id)
            self.active_tasks[album_id] = task_id
            
            # 等待下载完成
            while True:
                status = self.downloader.get_download_status(task_id)
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    break
                await asyncio.sleep(1)
            
            return status
            
        except Exception as e:
            logger.error(f"下载漫画 {album_id} 时发生错误: {e}")
            return {
                'success': False,
                'album_id': album_id,
                'error': str(e),
                'message': f'下载失败: {str(e)}'
            }
    
    async def get_download_status(self, album_id: str) -> Dict[str, Any]:
        """
        获取下载状态
        
        Args:
            album_id: 漫画ID
            
        Returns:
            下载状态
        """
        try:
            if album_id not in self.active_tasks:
                return {
                    'status': 'not_found',
                    'message': f'漫画 {album_id} 没有活跃的下载任务'
                }
            
            task_id = self.active_tasks[album_id]
            return self.downloader.get_download_status(task_id)
            
        except Exception as e:
            logger.error(f"获取下载状态时发生错误: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': f'获取状态失败: {str(e)}'
            }
    
    async def cancel_download(self, album_id: str) -> Dict[str, Any]:
        """
        取消下载
        
        Args:
            album_id: 漫画ID
            
        Returns:
            取消结果
        """
        try:
            if album_id not in self.active_tasks:
                return {
                    'success': False,
                    'message': f'漫画 {album_id} 没有活跃的下载任务'
                }
            
            task_id = self.active_tasks[album_id]
            result = self.downloader.cancel_download(task_id)
            
            if result['success']:
                del self.active_tasks[album_id]
            
            return result
            
        except Exception as e:
            logger.error(f"取消下载时发生错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'取消下载失败: {str(e)}'
            }
    
    async def get_all_status(self) -> Dict[str, Any]:
        """
        获取所有下载任务状态
        
        Returns:
            所有任务状态
        """
        try:
            return self.downloader.get_all_download_status()
        except Exception as e:
            logger.error(f"获取所有状态时发生错误: {e}")
            return {
                'error': str(e),
                'message': f'获取状态失败: {str(e)}'
            }

# 全局代理实例
agent = ComicDownloaderAgent()

# MCP工具函数
async def download_comic_tool(album_id: str) -> Dict[str, Any]:
    """
    MCP工具：下载漫画
    
    Args:
        album_id: 漫画ID
        
    Returns:
        下载结果
    """
    return await agent.download_comic(album_id)

async def get_download_status_tool(album_id: str) -> Dict[str, Any]:
    """
    MCP工具：获取下载状态
    
    Args:
        album_id: 漫画ID
        
    Returns:
        下载状态
    """
    return await agent.get_download_status(album_id)

async def cancel_download_tool(album_id: str) -> Dict[str, Any]:
    """
    MCP工具：取消下载
    
    Args:
        album_id: 漫画ID
        
    Returns:
        取消结果
    """
    return await agent.cancel_download(album_id)

async def get_all_status_tool() -> Dict[str, Any]:
    """
    MCP工具：获取所有下载状态
    
    Returns:
        所有任务状态
    """
    return await agent.get_all_status()

# 工厂函数
def create_comic_downloader_agent(config: Dict[str, Any] = None) -> ComicDownloaderAgent:
    """
    创建ComicDownloaderAgent实例
    
    Args:
        config: 配置字典
        
    Returns:
        ComicDownloaderAgent实例
    """
    try:
        agent_instance = ComicDownloaderAgent()
        logger.info("ComicDownloaderAgent实例创建成功")
        return agent_instance
    except Exception as e:
        logger.error(f"创建ComicDownloaderAgent实例失败: {e}")
        return None

def validate_agent_config(config: Dict[str, Any]) -> bool:
    """
    验证Agent配置
    
    Args:
        config: 配置字典
        
    Returns:
        配置是否有效
    """
    try:
        # 检查必需的配置项
        required_configs = [
            'COMIC_DOWNLOADER_HOST',
            'COMIC_DOWNLOADER_PORT',
            'COMIC_DOWNLOADER_PATH',
            'COMIC_DOWNLOADER_MAX_CONCURRENT'
        ]
        
        for config_key in required_configs:
            if config_key not in config:
                logger.warning(f"缺少配置项: {config_key}")
        
        logger.info("ComicDownloaderAgent配置验证通过")
        return True
        
    except Exception as e:
        logger.error(f"ComicDownloaderAgent配置验证失败: {e}")
        return False

def get_agent_dependencies() -> List[str]:
    """
    获取Agent依赖列表
    
    Returns:
        依赖包列表
    """
    return [
        'fastapi>=0.104.0',
        'uvicorn>=0.24.0',
        'pydantic>=2.0.0',
        'requests>=2.31.0',
        'pathlib2>=2.3.7',
        'commonx>=0.6.38',
        'curl-cffi',
        'pillow',
        'pycryptodome',
        'pyyaml',
        'psutil',
        'zhconv'
    ] 