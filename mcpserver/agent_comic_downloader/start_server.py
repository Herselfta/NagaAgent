#!/usr/bin/env python3
"""
Comic Downloader MCP Server 启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 添加原项目路径
comic_project = project_root / "Comic_Crawle_master"
if comic_project.exists():
    sys.path.insert(0, str(comic_project / "src"))

from mcpserver.agent_comic_downloader.server import start_server

def main():
    """主函数"""
    print("启动Comic Downloader MCP服务器...")
    print("服务器将在 http://localhost:8080 启动")
    print("按 Ctrl+C 停止服务器")
    
    try:
        start_server(host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器时发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 