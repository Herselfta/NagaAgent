"""
Comic Downloader MCP Agent 配置文件
"""

import os
from pathlib import Path

# 基础配置
class Config:
    """基础配置类"""
    
    # 服务配置
    HOST = os.getenv('COMIC_DOWNLOADER_HOST', '0.0.0.0')
    PORT = int(os.getenv('COMIC_DOWNLOADER_PORT', 8080))
    
    # 下载配置
    DEFAULT_DOWNLOAD_PATH = os.getenv('COMIC_DOWNLOADER_PATH', 'desktop')
    MAX_CONCURRENT_DOWNLOADS = int(os.getenv('COMIC_DOWNLOADER_MAX_CONCURRENT', 3))
    
    # 线程配置
    IMAGE_THREADS = int(os.getenv('COMIC_DOWNLOADER_IMAGE_THREADS', 10))
    PHOTO_THREADS = int(os.getenv('COMIC_DOWNLOADER_PHOTO_THREADS', 3))
    
    # 重试配置
    RETRY_TIMES = int(os.getenv('COMIC_DOWNLOADER_RETRY_TIMES', 3))
    
    # 日志配置
    LOG_LEVEL = os.getenv('COMIC_DOWNLOADER_LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('COMIC_DOWNLOADER_LOG_FILE', None)
    
    # 客户端配置
    CLIENT_TYPE = os.getenv('COMIC_DOWNLOADER_CLIENT_TYPE', 'api')  # 'api' or 'html'
    
    @classmethod
    def get_download_path(cls) -> str:
        """获取下载路径"""
        if cls.DEFAULT_DOWNLOAD_PATH.lower() == 'desktop':
            return str(Path.home() / "Desktop")
        else:
            return cls.DEFAULT_DOWNLOAD_PATH
    
    @classmethod
    def get_option_dict(cls) -> dict:
        """获取下载配置字典"""
        return {
            'version': '2.1',
            'log': True,
            'dir_rule': {
                'rule': 'Bd_Pname',  # 使用本子名称作为文件夹名
                'base_dir': cls.get_download_path(),  # 保存到指定路径
            },
            'download': {
                'cache': True,
                'image': {
                    'decode': True,
                    'suffix': None,  # 保持原始格式
                },
                'threading': {
                    'image': cls.IMAGE_THREADS,
                    'photo': cls.PHOTO_THREADS,
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
                'impl': cls.CLIENT_TYPE,
                'retry_times': cls.RETRY_TIMES,
            },
            'plugins': {
                'valid': 'log',
            },
        }

# 开发环境配置
class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# 生产环境配置
class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# 测试环境配置
class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    DEFAULT_DOWNLOAD_PATH = str(Path.cwd() / "test_downloads")

# 配置映射
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}

def get_config(env: str = None) -> Config:
    """
    获取配置
    
    Args:
        env: 环境名称
        
    Returns:
        配置对象
    """
    if env is None:
        env = os.getenv('COMIC_DOWNLOADER_ENV', 'development')
    
    return config_map.get(env, DevelopmentConfig) 