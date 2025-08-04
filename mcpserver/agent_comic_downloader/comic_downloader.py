"""
简化的漫画下载器
基于原项目的核心下载功能，只保留下载功能，默认保存到桌面
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import time

try:
    from .jmcomic import download_album, create_option_by_str
    from .jmcomic.jm_config import JmModuleConfig
    from .jmcomic.jm_option import JmOption
    from .jmcomic.jm_downloader import JmDownloader
    from .jmcomic.jm_entity import JmAlbumDetail
    from .jmcomic.jm_exception import JmcomicException
except ImportError:
    try:
        from jmcomic import download_album, create_option_by_str
        from jmcomic.jm_config import JmModuleConfig
        from jmcomic.jm_option import JmOption
        from jmcomic.jm_downloader import JmDownloader
        from jmcomic.jm_entity import JmAlbumDetail
        from jmcomic.jm_exception import JmcomicException
    except ImportError as e:
        logging.error(f"Failed to import jmcomic modules: {e}")
        raise

class ComicDownloader:
    """简化的漫画下载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.download_threads = {}
        self.download_status = {}
        
    def get_desktop_path(self) -> str:
        """获取桌面路径"""
        try:
            # Windows
            desktop = Path.home() / "Desktop"
            if desktop.exists():
                return str(desktop)
            
            # macOS
            desktop = Path.home() / "Desktop"
            if desktop.exists():
                return str(desktop)
                
            # Linux
            desktop = Path.home() / "Desktop"
            if desktop.exists():
                return str(desktop)
                
            # 如果都找不到，使用当前目录
            return str(Path.cwd())
        except Exception as e:
            self.logger.warning(f"Failed to get desktop path: {e}")
            return str(Path.cwd())
    
    def create_simple_option(self, album_id: str) -> JmOption:
        """创建简化的下载配置"""
        desktop_path = self.get_desktop_path()
        
        # 创建简化的配置字典
        option_dict = {
            'version': '2.1',
            'log': True,
            'dir_rule': {
                'rule': 'Bd_Pname',  # 使用本子名称作为文件夹名
                'base_dir': desktop_path,  # 保存到桌面
            },
            'download': {
                'cache': True,
                'image': {
                    'decode': True,
                    'suffix': None,  # 保持原始格式
                },
                'threading': {
                    'image': 10,  # 图片下载线程数
                    'photo': 3,   # 章节下载线程数
                },
            },
            'client': {
                'cache': None,
                'domain': [],
                'postman': {
                    'type': 'curl_cffi',
                    'meta_data': {
                        'impersonate': 'chrome',
                        'headers': None,
                        'proxies': None,
                    }
                },
                'impl': 'api',  # 使用API客户端
                'retry_times': 3,
            },
            'plugins': {
                'valid': 'log',
            },
        }
        
        return JmOption.construct(option_dict)
    
    def download_comic(self, album_id: str, callback=None) -> Dict[str, Any]:
        """
        下载漫画
        
        Args:
            album_id: 漫画ID
            callback: 回调函数，用于更新下载进度
            
        Returns:
            下载结果字典
        """
        try:
            self.logger.info(f"开始下载漫画: {album_id}")
            
            # 创建下载配置
            option = self.create_simple_option(album_id)
            
            # 创建下载器
            downloader = JmDownloader(option)
            
            # 开始下载
            album, downloader_result = download_album(
                album_id, 
                option=option, 
                downloader=JmDownloader,
                check_exception=True
            )
            
            # 获取下载结果
            result = {
                'success': True,
                'album_id': album_id,
                'album_title': album.title if album else None,
                'album_author': album.author if album else None,
                'download_path': option.decide_image_save_dir(album[0]) if album and len(album) > 0 else None,
                'message': f"漫画 {album_id} 下载完成"
            }
            
            self.logger.info(f"漫画 {album_id} 下载成功")
            return result
            
        except JmcomicException as e:
            error_msg = f"下载漫画 {album_id} 失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'album_id': album_id,
                'error': str(e),
                'message': error_msg
            }
        except Exception as e:
            error_msg = f"下载漫画 {album_id} 时发生未知错误: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'album_id': album_id,
                'error': str(e),
                'message': error_msg
            }
    
    def download_comic_async(self, album_id: str) -> str:
        """
        异步下载漫画
        
        Args:
            album_id: 漫画ID
            
        Returns:
            任务ID
        """
        task_id = f"download_{album_id}_{int(time.time())}"
        
        def download_task():
            try:
                self.download_status[task_id] = {
                    'status': 'downloading',
                    'progress': 0,
                    'message': f'开始下载漫画 {album_id}'
                }
                
                result = self.download_comic(album_id)
                
                if result['success']:
                    self.download_status[task_id] = {
                        'status': 'completed',
                        'progress': 100,
                        'message': result['message'],
                        'result': result
                    }
                else:
                    self.download_status[task_id] = {
                        'status': 'failed',
                        'progress': 0,
                        'message': result['message'],
                        'error': result.get('error')
                    }
                    
            except Exception as e:
                self.download_status[task_id] = {
                    'status': 'failed',
                    'progress': 0,
                    'message': f'下载任务异常: {str(e)}',
                    'error': str(e)
                }
        
        # 启动下载线程
        thread = threading.Thread(target=download_task, daemon=True)
        thread.start()
        
        self.download_threads[task_id] = thread
        
        return task_id
    
    def get_download_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取下载状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            下载状态字典
        """
        if task_id not in self.download_status:
            return {
                'status': 'not_found',
                'message': f'任务 {task_id} 不存在'
            }
        
        return self.download_status[task_id]
    
    def get_all_download_status(self) -> Dict[str, Any]:
        """
        获取所有下载任务状态
        
        Returns:
            所有任务状态字典
        """
        return self.download_status
    
    def cancel_download(self, task_id: str) -> Dict[str, Any]:
        """
        取消下载任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            取消结果
        """
        if task_id not in self.download_threads:
            return {
                'success': False,
                'message': f'任务 {task_id} 不存在'
            }
        
        try:
            # 标记任务为取消状态
            if task_id in self.download_status:
                self.download_status[task_id]['status'] = 'cancelled'
                self.download_status[task_id]['message'] = '任务已取消'
            
            return {
                'success': True,
                'message': f'任务 {task_id} 已取消'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'取消任务失败: {str(e)}'
            } 