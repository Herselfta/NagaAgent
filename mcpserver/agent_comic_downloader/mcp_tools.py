"""
MCP工具调用接口
提供与LLM交互的工具函数
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from .comic_downloader_agent import ComicDownloaderAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPTools:
    """MCP工具类"""
    
    def __init__(self):
        self.agent = ComicDownloaderAgent()
    
    def download_comic(self, album_id: str) -> Dict[str, Any]:
        """
        下载漫画工具
        
        Args:
            album_id: 漫画ID
            
        Returns:
            下载结果
        """
        try:
            logger.info(f"MCP工具调用: 下载漫画 {album_id}")
            
            # 启动异步下载
            task_id = self.agent.downloader.download_comic_async(album_id)
            self.agent.active_tasks[album_id] = task_id
            
            # 等待下载完成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self.agent.download_comic(album_id))
            finally:
                loop.close()
            
            return result
            
        except Exception as e:
            logger.error(f"下载漫画时发生错误: {e}")
            return {
                'success': False,
                'album_id': album_id,
                'error': str(e),
                'message': f'下载失败: {str(e)}'
            }
    
    def get_download_status(self, album_id: str) -> Dict[str, Any]:
        """
        获取下载状态工具
        
        Args:
            album_id: 漫画ID
            
        Returns:
            下载状态
        """
        try:
            logger.info(f"MCP工具调用: 获取下载状态 {album_id}")
            
            if album_id not in self.agent.active_tasks:
                return {
                    'status': 'not_found',
                    'message': f'漫画 {album_id} 没有活跃的下载任务'
                }
            
            task_id = self.agent.active_tasks[album_id]
            return self.agent.downloader.get_download_status(task_id)
            
        except Exception as e:
            logger.error(f"获取下载状态时发生错误: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': f'获取状态失败: {str(e)}'
            }
    
    def cancel_download(self, album_id: str) -> Dict[str, Any]:
        """
        取消下载工具
        
        Args:
            album_id: 漫画ID
            
        Returns:
            取消结果
        """
        try:
            logger.info(f"MCP工具调用: 取消下载 {album_id}")
            
            if album_id not in self.agent.active_tasks:
                return {
                    'success': False,
                    'message': f'漫画 {album_id} 没有活跃的下载任务'
                }
            
            task_id = self.agent.active_tasks[album_id]
            result = self.agent.downloader.cancel_download(task_id)
            
            if result['success']:
                del self.agent.active_tasks[album_id]
            
            return result
            
        except Exception as e:
            logger.error(f"取消下载时发生错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'取消下载失败: {str(e)}'
            }
    
    def get_all_status(self) -> Dict[str, Any]:
        """
        获取所有下载状态工具
        
        Returns:
            所有任务状态
        """
        try:
            logger.info("MCP工具调用: 获取所有下载状态")
            return self.agent.downloader.get_all_download_status()
        except Exception as e:
            logger.error(f"获取所有状态时发生错误: {e}")
            return {
                'error': str(e),
                'message': f'获取状态失败: {str(e)}'
            }

# 全局MCP工具实例
mcp_tools = MCPTools()

# 工具函数映射
TOOLS = {
    'download_comic': {
        'name': 'download_comic',
        'description': '下载指定ID的漫画到桌面',
        'parameters': {
            'type': 'object',
            'properties': {
                'album_id': {
                    'type': 'string',
                    'description': '漫画的ID号'
                }
            },
            'required': ['album_id']
        }
    },
    'get_download_status': {
        'name': 'get_download_status',
        'description': '获取指定漫画的下载状态',
        'parameters': {
            'type': 'object',
            'properties': {
                'album_id': {
                    'type': 'string',
                    'description': '漫画的ID号'
                }
            },
            'required': ['album_id']
        }
    },
    'cancel_download': {
        'name': 'cancel_download',
        'description': '取消指定漫画的下载任务',
        'parameters': {
            'type': 'object',
            'properties': {
                'album_id': {
                    'type': 'string',
                    'description': '漫画的ID号'
                }
            },
            'required': ['album_id']
        }
    },
    'get_all_status': {
        'name': 'get_all_status',
        'description': '获取所有下载任务的状态',
        'parameters': {
            'type': 'object',
            'properties': {}
        }
    }
}

def get_tools() -> List[Dict[str, Any]]:
    """
    获取所有可用工具
    
    Returns:
        工具列表
    """
    return list(TOOLS.values())

def call_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用工具
    
    Args:
        tool_name: 工具名称
        parameters: 工具参数
        
    Returns:
        工具执行结果
    """
    try:
        if tool_name == 'download_comic':
            return mcp_tools.download_comic(parameters['album_id'])
        elif tool_name == 'get_download_status':
            return mcp_tools.get_download_status(parameters['album_id'])
        elif tool_name == 'cancel_download':
            return mcp_tools.cancel_download(parameters['album_id'])
        elif tool_name == 'get_all_status':
            return mcp_tools.get_all_status()
        else:
            return {
                'error': f'未知工具: {tool_name}',
                'message': f'工具 {tool_name} 不存在'
            }
    except Exception as e:
        logger.error(f"调用工具 {tool_name} 时发生错误: {e}")
        return {
            'error': str(e),
            'message': f'工具调用失败: {str(e)}'
        } 