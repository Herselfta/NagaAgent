# agent_app_launcher.py # 应用启动与管理Agent
import os
import platform
import subprocess
import asyncio
import json
from .app_cache import get_cached_apps, preload_apps  # 应用缓存模块

class AppLauncherAgent(object):
    """应用启动与管理Agent，支持打开、列出本机应用"""  # 类注释
    name = "AppLauncher Agent"  # Agent名称

    def __init__(self):
        from .app_cache import preload_apps, get_cached_apps
        preload_apps()  # 初始化时同步预加载
        import sys
        sys.stderr.write(f'✅ AppLauncherAgent初始化完成，预加载应用数: {len(get_cached_apps())}\n')

    def run(self, action, app=None, args=None):
        """
        action: 操作类型（open/list/refresh）
        app: 应用名或路径
        args: 启动参数
        """
        if action == "open":
            return self.open_app(app, args)  # 打开应用
        elif action == "list":
            return {"status": "success", "apps": get_cached_apps()}  # 返回缓存应用列表
        elif action == "refresh":
            asyncio.create_task(preload_apps())  # 异步刷新
            return {"status": "success", "message": "正在刷新应用列表，请稍后再试"}
        else:
            return {"status": "error", "message": f"未知操作: {action}"}  # 错误处理

    def open_app(self, app, args=None):
        """打开指定应用（严格等值匹配）"""  # 右侧注释
        print(f"open_app收到app参数: {app}")
        print(f"缓存应用名: {[item['name'] for item in get_cached_apps()]}" )
        exe_path = None
        app_list = get_cached_apps()
        # 1. 严格等值匹配
        if app:
            for item in app_list:
                if app == item["name"]:
                    exe_path = item["path"]
                    break
        # 2. 支持绝对路径
        if not exe_path and app and os.path.exists(app):
            exe_path = app
        if not exe_path or not os.path.exists(exe_path):
            return {"status": "error", "message": f"未找到应用: {app}"}  # 匹配失败
        try:
            if exe_path.lower().endswith('.lnk'):
                os.startfile(exe_path)  # 用系统方式打开快捷方式
            else:
                cmd = [exe_path]
                if args:
                    if isinstance(args, str):
                        cmd += args.split()
                    elif isinstance(args, list):
                        cmd += args
                subprocess.Popen(cmd, shell=False)
            return {"status": "success", "message": f"已启动: {exe_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def handle_handoff(self, data: dict) -> str:
        """
        MCP标准接口，处理handoff请求
        使用tool_name参数，与LLM生成的工具调用格式匹配
        """
        try:
            # 使用tool_name参数，与LLM生成的工具调用格式匹配
            action = data.get("tool_name")
            if not action:
                return json.dumps({"status": "error", "message": "缺少tool_name参数", "data": {}}, ensure_ascii=False)
                
            app = data.get("app")
            args = data.get("args")
            
            if action == "open":
                if not app:
                    return json.dumps({"status": "error", "message": "open操作需要app参数", "data": {}}, ensure_ascii=False)
                result = self.open_app(app, args)
                return json.dumps(result, ensure_ascii=False)
            elif action == "list":
                result = self.run("list")
                return json.dumps(result, ensure_ascii=False)
            elif action == "refresh":
                result = self.run("refresh")
                return json.dumps(result, ensure_ascii=False)
            else:
                return json.dumps({"status": "error", "message": f"未知操作: {action}", "data": {}}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e), "data": {}}, ensure_ascii=False)

# 工厂函数：动态创建Agent实例
def create_app_launcher_agent():
    """创建AppLauncherAgent实例"""
    return AppLauncherAgent()

# 获取Agent元数据
def get_agent_metadata():
    """获取Agent元数据"""
    import os
    manifest_path = os.path.join(os.path.dirname(__file__), "agent-manifest.json")
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载元数据失败: {e}")
        return None

# 验证配置
def validate_agent_config(config):
    """验证Agent配置"""
    return True

# 获取依赖
def get_agent_dependencies():
    """获取Agent依赖"""
    return []
