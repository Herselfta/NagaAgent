# app_cache.py # 应用列表缓存与同步预加载
import platform, os
from pathlib import Path

APP_LIST_CACHE = []  # 全局缓存

def preload_apps():
    """同步预加载本机应用列表"""
    global APP_LIST_CACHE
    apps = scan_apps()
    APP_LIST_CACHE.clear()
    APP_LIST_CACHE.extend(apps)

def get_cached_apps():
    """获取缓存的应用列表"""
    return APP_LIST_CACHE

def scan_apps():
    """同步扫描本机应用（支持Windows/Mac/Linux）"""
    apps = []
    system = platform.system()
    if system == "Windows":
        start_menu_dirs = [
            os.path.expandvars(r'%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs'),
            r'C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs'
        ]
        for d in start_menu_dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith('.lnk'):
                        apps.append({"name": f[:-4], "path": os.path.join(root, f)})
    elif system == "Darwin":
        app_dirs = ["/Applications", str(Path.home() / "Applications")]
        for d in app_dirs:
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if f.endswith('.app'):
                    apps.append({"name": f[:-4], "path": os.path.join(d, f)})
    elif system == "Linux":
        desktop_dirs = ["/usr/share/applications", str(Path.home() / ".local/share/applications")]
        for d in desktop_dirs:
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if f.endswith('.desktop'):
                    apps.append({"name": f[:-8], "path": os.path.join(d, f)})
    return apps
