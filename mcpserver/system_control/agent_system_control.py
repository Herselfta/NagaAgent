from agents import Agent
from config import config  # 统一变量管理 #
import asyncio
import json
import ctypes
from ctypes import wintypes
import subprocess
import os
try:
    import screen_brightness_control as sbc  # 屏幕亮度调节 #
except ImportError:
    sbc = None
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    import comtypes  # COM组件初始化 #
except ImportError:
    AudioUtilities = None

# Windows API常量定义
EWX_LOGOFF = 0x00000000
EWX_SHUTDOWN = 0x00000001
EWX_REBOOT = 0x00000002
EWX_FORCE = 0x00000004
EWX_POWEROFF = 0x00000008
EWX_FORCEIFHUNG = 0x00000010

# 关机原因常量
SHTDN_REASON_MAJOR_OTHER = 0x00000000
SHTDN_REASON_MAJOR_HARDWARE = 0x00010000
SHTDN_REASON_MAJOR_OPERATINGSYSTEM = 0x00020000
SHTDN_REASON_MAJOR_SOFTWARE = 0x00030000
SHTDN_REASON_MAJOR_APPLICATION = 0x00040000
SHTDN_REASON_MAJOR_SYSTEM = 0x00050000
SHTDN_REASON_MAJOR_POWER = 0x00060000
SHTDN_REASON_MAJOR_LEGACY_API = 0x00070000

SHTDN_REASON_MINOR_OTHER = 0x00000000
SHTDN_REASON_MINOR_MAINTENANCE = 0x00000001
SHTDN_REASON_MINOR_INSTALLATION = 0x00000002
SHTDN_REASON_MINOR_UPGRADE = 0x00000003
SHTDN_REASON_MINOR_RECONFIG = 0x00000004
SHTDN_REASON_MINOR_HUNG = 0x00000005
SHTDN_REASON_MINOR_UNSTABLE = 0x00000006
SHTDN_REASON_MINOR_DISK = 0x00000007
SHTDN_REASON_MINOR_PROCESSOR = 0x00000008
SHTDN_REASON_MINOR_NETWORKCARD = 0x00000009
SHTDN_REASON_MINOR_POWER_SUPPLY = 0x0000000A
SHTDN_REASON_MINOR_CORDUNPLUGGED = 0x0000000B
SHTDN_REASON_MINOR_ENVIRONMENT = 0x0000000C
SHTDN_REASON_MINOR_HARDWARE_DRIVER = 0x0000000D
SHTDN_REASON_MINOR_OTHERDRIVER = 0x0000000E
SHTDN_REASON_MINOR_BLUESCREEN = 0x0000000F
SHTDN_REASON_MINOR_SERVICEPACK = 0x00000010
SHTDN_REASON_MINOR_HOTFIX = 0x00000011
SHTDN_REASON_MINOR_SECURITYFIX = 0x00000012
SHTDN_REASON_MINOR_SECURITY = 0x00000013
SHTDN_REASON_MINOR_NETWORK_CONNECTIVITY = 0x00000014
SHTDN_REASON_MINOR_WMI = 0x00000015
SHTDN_REASON_MINOR_SERVICEPACK_UNINSTALL = 0x00000016
SHTDN_REASON_MINOR_HOTFIX_UNINSTALL = 0x00000017
SHTDN_REASON_MINOR_SECURITYFIX_UNINSTALL = 0x00000018
SHTDN_REASON_MINOR_MMC = 0x00000019
SHTDN_REASON_MINOR_SYSTEMRESTORE = 0x0000001A
SHTDN_REASON_MINOR_TERMSERVICE = 0x0000001B
SHTDN_REASON_MINOR_DC_PROMOTION = 0x0000001C
SHTDN_REASON_MINOR_DC_DEMOTION = 0x0000001D

class SystemControlAgent(Agent):
    """系统控制Agent #"""
    name = "SystemControlAgent"  # Agent名称 #
    instructions = "系统控制：定时关机、定时重启、亮度、音量调节"  # 角色描述 #
    
    def __init__(self):
        super().__init__(
            name=self.name,
            instructions=self.instructions,
            tools=[],
            model=config.api.model
        )
        # 初始化Windows API
        self._init_windows_api()

    def _init_windows_api(self):
        """初始化Windows API"""
        try:
            self.user32 = ctypes.windll.user32
            self.advapi32 = ctypes.windll.advapi32
            self.kernel32 = ctypes.windll.kernel32
            
            # 定义API函数签名
            self._ExitWindowsEx = self.user32.ExitWindowsEx
            self._ExitWindowsEx.argtypes = [wintypes.UINT, wintypes.DWORD]
            self._ExitWindowsEx.restype = wintypes.BOOL
            
            self._InitiateSystemShutdownEx = self.advapi32.InitiateSystemShutdownExW
            self._InitiateSystemShutdownEx.argtypes = [
                wintypes.LPWSTR, wintypes.LPWSTR, wintypes.DWORD,
                wintypes.BOOL, wintypes.BOOL, wintypes.DWORD
            ]
            self._InitiateSystemShutdownEx.restype = wintypes.BOOL
            
            self._ShutdownBlockReasonCreate = self.user32.ShutdownBlockReasonCreate
            self._ShutdownBlockReasonCreate.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
            self._ShutdownBlockReasonCreate.restype = wintypes.BOOL
            
            self._ShutdownBlockReasonDestroy = self.user32.ShutdownBlockReasonDestroy
            self._ShutdownBlockReasonDestroy.argtypes = [wintypes.HWND]
            self._ShutdownBlockReasonDestroy.restype = wintypes.BOOL
            
        except Exception as e:
            print(f"Windows API初始化失败: {e}")

    def _show_windows_shutdown_dialog(self, action_type="shutdown", timeout=30, force=False):
        """使用Windows API显示关机对话框"""
        try:
            if action_type == "shutdown":
                flags = EWX_SHUTDOWN
            elif action_type == "restart":
                flags = EWX_REBOOT
            else:
                return False
                
            if force:
                flags |= EWX_FORCE
                
            result = self._ExitWindowsEx(flags, 0)
            return result != 0
        except Exception as e:
            print(f"ExitWindowsEx调用失败: {e}")
            return False

    def _show_advanced_shutdown_dialog(self, action_type="shutdown", timeout=30, message="", force=False):
        """使用高级Windows API显示关机对话框"""
        try:
            if action_type == "shutdown":
                shutdown_flag = False
            elif action_type == "restart":
                shutdown_flag = True
            else:
                return False
                
            # 转换字符串为宽字符
            message_w = ctypes.create_unicode_buffer(message) if message else None
            
            result = self._InitiateSystemShutdownEx(
                None,                    # 计算机名（None表示本地）
                message_w,               # 关机消息
                timeout,                 # 超时时间（秒）
                force,                   # 强制关闭应用
                shutdown_flag            # 是否重启
            )
            return result != 0
        except Exception as e:
            print(f"InitiateSystemShutdownEx调用失败: {e}")
            return False

    def _show_rundll_shutdown_dialog(self, action_type="shutdown"):
        """使用rundll32.exe调用Windows关机对话框（备用方法）"""
        try:
            if action_type == "shutdown":
                cmd = "rundll32.exe shell32.dll,SHExitWindowsEx 1"
            elif action_type == "restart":
                cmd = "rundll32.exe shell32.dll,SHExitWindowsEx 2"
            else:
                return False
                
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            print(f"rundll32调用失败: {e}")
            return False

    def _block_shutdown(self, reason="正在保存重要数据..."):
        """阻止关机，显示自定义原因"""
        try:
            reason_w = ctypes.create_unicode_buffer(reason)
            result = self._ShutdownBlockReasonCreate(None, reason_w)
            return result != 0
        except Exception as e:
            print(f"ShutdownBlockReasonCreate调用失败: {e}")
            return False

    def _unblock_shutdown(self):
        """取消阻止关机"""
        try:
            result = self._ShutdownBlockReasonDestroy(None)
            return result != 0
        except Exception as e:
            print(f"ShutdownBlockReasonDestroy调用失败: {e}")
            return False

    def _execute_shutdown_command(self, action_type="shutdown", timeout=0):
        """执行命令行关机/重启"""
        try:
            if action_type == "shutdown":
                cmd = f"shutdown /s /t {timeout}"
            elif action_type == "restart":
                cmd = f"shutdown /r /t {timeout}"
            else:
                return False
                
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            print(f"命令行关机失败: {e}")
            return False

    async def handle_handoff(self, data: dict) -> str:
        """处理系统控制请求"""
        action = data.get("tool_name")
        if not action:
            return json.dumps({"status": "error", "message": "缺少tool_name参数", "data": {}}, ensure_ascii=False)
            
        if action == "shutdown":
            time_sec = int(data.get("time", 0))
            force = data.get("force", False)
            message = data.get("message", "")
            
            # 尝试多种Windows API方法
            success = False
            
            # 方法1: 高级关机API
            if message:
                success = self._show_advanced_shutdown_dialog("shutdown", time_sec, message, force)
            
            # 方法2: 基础关机API
            if not success:
                success = self._show_windows_shutdown_dialog("shutdown", time_sec, force)
            
            # 方法3: rundll32方法
            if not success:
                success = self._show_rundll_shutdown_dialog("shutdown")
            
            # 方法4: 命令行方法
            if not success:
                success = self._execute_shutdown_command("shutdown", time_sec)
            
            if success:
                time_str = f"{time_sec}秒后" if time_sec > 0 else "立即"
                return json.dumps({
                    "status": "success", 
                    "message": f"已显示关机对话框，{time_str}关机", 
                    "data": {"timeout": time_sec}
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "error", 
                    "message": "所有关机方法都失败了", 
                    "data": {}
                }, ensure_ascii=False)
                    
        elif action == "restart":
            time_sec = int(data.get("time", 0))
            force = data.get("force", False)
            message = data.get("message", "")
            
            # 尝试多种Windows API方法
            success = False
            
            # 方法1: 高级重启API
            if message:
                success = self._show_advanced_shutdown_dialog("restart", time_sec, message, force)
            
            # 方法2: 基础重启API
            if not success:
                success = self._show_windows_shutdown_dialog("restart", time_sec, force)
            
            # 方法3: rundll32方法
            if not success:
                success = self._show_rundll_shutdown_dialog("restart")
            
            # 方法4: 命令行方法
            if not success:
                success = self._execute_shutdown_command("restart", time_sec)
            
            if success:
                time_str = f"{time_sec}秒后" if time_sec > 0 else "立即"
                return json.dumps({
                    "status": "success", 
                    "message": f"已显示重启对话框，{time_str}重启", 
                    "data": {"timeout": time_sec}
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "error", 
                    "message": "所有重启方法都失败了", 
                    "data": {}
                }, ensure_ascii=False)
                    
        elif action == "set_brightness":
            value = int(data.get("value", 50))
            if sbc:
                try:
                    sbc.set_brightness(value)
                    return json.dumps({
                        "status": "success", 
                        "message": f"亮度已设置为{value}%", 
                        "data": {"brightness": value}
                    }, ensure_ascii=False)
                except Exception as e:
                    return json.dumps({
                        "status": "error", 
                        "message": f"亮度设置失败: {e}", 
                        "data": {}
                    }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "error", 
                    "message": "未安装screen_brightness_control库，无法调节亮度", 
                    "data": {}
                }, ensure_ascii=False)
                
        elif action == "set_volume":
            value = int(data.get("value", 50))
            if AudioUtilities:
                try:
                    comtypes.CoInitialize()  # 初始化COM组件 #
                    devices = AudioUtilities.GetSpeakers()
                    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    volume = cast(interface, POINTER(IAudioEndpointVolume))
                    volume.SetMasterVolumeLevelScalar(value/100, None)
                    return json.dumps({
                        "status": "success", 
                        "message": f"音量已设置为{value}%", 
                        "data": {"volume": value}
                    }, ensure_ascii=False)
                except Exception as e:
                    return json.dumps({
                        "status": "error", 
                        "message": f"音量设置失败: {e}", 
                        "data": {}
                    }, ensure_ascii=False)
                finally:
                    comtypes.CoUninitialize()  # 清理COM组件 #
            else:
                return json.dumps({
                    "status": "error", 
                    "message": "未安装pycaw库，无法调节音量", 
                    "data": {}
                }, ensure_ascii=False)
                
        elif action == "block_shutdown":
            reason = data.get("reason", "正在保存重要数据...")
            if self._block_shutdown(reason):
                return json.dumps({
                    "status": "success", 
                    "message": f"已阻止关机，原因: {reason}", 
                    "data": {"reason": reason}
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "error", 
                    "message": "阻止关机失败", 
                    "data": {}
                }, ensure_ascii=False)
                
        elif action == "unblock_shutdown":
            if self._unblock_shutdown():
                return json.dumps({
                    "status": "success", 
                    "message": "已取消阻止关机", 
                    "data": {}
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "error", 
                    "message": "取消阻止关机失败", 
                    "data": {}
                }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error", 
                "message": f"未知操作: {action}", 
                "data": {}
            }, ensure_ascii=False)

# 工厂函数，用于动态注册系统创建实例
def create_system_control_agent():
    """创建SystemControlAgent实例的工厂函数"""
    return SystemControlAgent() 