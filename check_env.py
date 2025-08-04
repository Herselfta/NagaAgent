#!/usr/bin/env python3
"""
NagaAgent 跨平台环境检查脚本
用法: python check_env.py 或 python3 check_env.py
支持: Windows 10/11, macOS 10.15+
"""

import os
import sys
import platform
import subprocess

def print_status(message, status):
    """打印状态信息"""
    if status:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    return status

def check_command(cmd):
    """检查命令是否可用"""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_python_package(package):
    """检查 Python 包是否已安装"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def check_windows_specific():
    """Windows 特定检查"""
    print("🖥️  Windows 环境检查:")
    
    # 检查 PowerShell
    powershell_ok = check_command("powershell")
    print_status("PowerShell", powershell_ok)
    
    # 检查 Windows 浏览器
    win_browser_paths = [
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        os.path.expanduser(r'~\AppData\Local\Google\Chrome\Application\chrome.exe')
    ]
    win_browser_found = any(os.path.exists(p) for p in win_browser_paths)
    print_status("Chrome 浏览器", win_browser_found)
    
    # 检查 Visual C++ Build Tools
    try:
        result = subprocess.run(['cl'], capture_output=True, text=True)
        handoffp_ok = result.returncode == 0 or "Microsoft" in result.stderr
    except FileNotFoundError:
        handoffp_ok = False
    print_status("Visual C++ Build Tools", handoffp_ok)
    
    return powershell_ok and win_browser_found

def check_macos_specific():
    """macOS 特定检查"""
    print("🍎 macOS 环境检查:")
    
    # 检查 Homebrew
    brew_ok = check_command("brew")
    print_status("Homebrew", brew_ok)
    
    # 检查 macOS 浏览器
    mac_browser_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
    ]
    mac_browser_found = any(os.path.exists(p) for p in mac_browser_paths)
    print_status("Chrome/浏览器", mac_browser_found)
    
    # 检查 PortAudio (语音功能)
    portaudio_found = os.path.exists("/opt/homebrew/lib/libportaudio.dylib") or \
                     os.path.exists("/usr/local/lib/libportaudio.dylib")
    print_status("PortAudio (语音功能)", portaudio_found)
    
    return brew_ok and mac_browser_found

def main():
    print("🔍 NagaAgent 跨平台环境检查")
    print("=" * 50)
    
    all_good = True
    system = platform.system()
    
    # 检查系统信息
    print(f"📱 操作系统: {system} {platform.release()}")
    print(f"🏗️  架构: {platform.machine()}")
    print(f"🐍 Python: {platform.python_version()}")
    print()
    
    # 检查 Python 版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_ok = sys.version_info >= (3, 8)
    all_good &= print_status(f"Python {python_version} (需要 >= 3.8)", python_ok)
    
    # 检查 Git
    git_ok = check_command("git")
    print_status("Git", git_ok)
    all_good &= git_ok
    
    print()
    
    # 系统特定检查
    if system == "Windows":
        system_ok = check_windows_specific()
        all_good &= system_ok
    elif system == "Darwin":  # macOS
        system_ok = check_macos_specific()
        all_good &= system_ok
    elif system == "Linux":
        print("🐧 Linux 环境检查:")
        linux_browser_paths = [
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium'
        ]
        linux_browser_found = any(os.path.exists(p) for p in linux_browser_paths)
        print_status("Chrome/Chromium 浏览器", linux_browser_found)
        all_good &= linux_browser_found
    else:
        print(f"⚠️  未知操作系统: {system}")
    
    print()
    print("📦 Python 依赖检查:")
    
    # 检查关键 Python 包
    packages = {
        "PyQt5": "PyQt5",
        "OpenAI": "openai", 
        "Requests": "requests",
        "NumPy": "numpy",
        "Transformers": "transformers",
        "Playwright": "playwright",
        "python-dotenv": "dotenv"
    }
    
    for display_name, package_name in packages.items():
        pkg_ok = check_python_package(package_name)
        if display_name in ["PyQt5", "OpenAI", "Requests", "NumPy"]:
            all_good &= print_status(f"{display_name}", pkg_ok)
        else:
            print_status(f"{display_name}", pkg_ok)
    
    print()
    print("📁 项目文件检查:")
    
    # 检查项目文件
    files = {
        "配置文件": "config.py",
        "主程序": "main.py", 
        "依赖列表": "pyproject.toml"
    }
    
    # 系统特定文件
    if system == "Windows":
        files.update({
            "Windows 设置脚本": "setup.ps1",
            "Windows 启动脚本": "start.bat"
        })
    elif system == "Darwin":
        files.update({
            "Mac 设置脚本": "setup_mac.sh",
            "Mac 一键部署": "quick_deploy_mac.sh"
        })
    
    for display_name, filename in files.items():
        file_ok = os.path.exists(filename)
        print_status(f"{display_name} ({filename})", file_ok)
    
    # 检查虚拟环境
    venv_ok = os.path.exists(".venv")
    print_status("虚拟环境 (.venv)", venv_ok)
    
    # 检查GRAG记忆系统
    grag_ok = os.path.exists("summer_memory")
    print_status("GRAG记忆系统", grag_ok)
    
    print()
    print("⚙️  配置检查:")
    
    # 检查 API 密钥配置
    try:
        import config
        api_key = getattr(config, 'API_KEY', '')
        api_key_ok = api_key and api_key.strip() and api_key != " "
        print_status("DeepSeek API 密钥已配置", api_key_ok)
    except ImportError:
        print("❌ 无法导入 config.py 配置文件")
        api_key_ok = False
    
    # 检查可选的 .env 文件
    env_file_ok = os.path.exists(".env")
    if env_file_ok:
        print_status("环境配置文件 (.env)", env_file_ok)
    
    print()
    print("=" * 50)
    
    if all_good:
        print("🎉 环境检查通过！")
        if system == "Windows":
            print("   可以运行: .\\start.bat")
        elif system == "Darwin":
            print("   可以运行: ./start_mac.sh")
        else:
            print("   可以运行: python main.py")
    else:
        print("🔧 发现问题，请参考解决方案进行修复")
        print()
        print("📝 常见解决方案:")
        
        if not python_ok:
            if system == "Windows":
                print("   下载安装 Python 3.11: https://www.python.org/downloads/")
            elif system == "Darwin":
                print("   brew install python@3.11")
            else:
                print("   安装 Python 3.11")
        
        if system == "Windows":
            print("   安装 Chrome: https://www.google.com/chrome/")
            print("   安装 Visual C++ Build Tools")
            if not venv_ok:
                print("   运行: .\\setup.ps1")
        elif system == "Darwin":
            print("   brew install --cask google-chrome")
            print("   brew install portaudio")
            if not venv_ok:
                print("   运行: ./setup_mac.sh")
        
        if not api_key_ok:
            print("   配置 API 密钥: 修改 config.py 中的 API_KEY")
            print("   或设置环境变量: export API_KEY=your_key")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main()) 
