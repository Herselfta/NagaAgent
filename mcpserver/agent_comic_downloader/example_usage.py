#!/usr/bin/env python3
"""
Comic Downloader 使用示例
演示如何使用MCP工具下载漫画
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 添加原项目路径
comic_project = project_root / "Comic_Crawle_master"
if comic_project.exists():
    sys.path.insert(0, str(comic_project / "src"))

from mcpserver.agent_comic_downloader.mcp_tools import call_tool, get_tools

def example_download_comic():
    """示例：下载漫画"""
    print("=== 漫画下载示例 ===")
    
    # 漫画ID（这是一个示例ID，实际使用时需要替换为真实的漫画ID）
    album_id = "422866"
    
    print(f"开始下载漫画: {album_id}")
    
    # 调用下载工具
    result = call_tool('download_comic', {'album_id': album_id})
    
    print(f"下载结果: {result}")
    
    if result.get('success'):
        print("✅ 下载成功!")
        print(f"漫画标题: {result.get('album_title')}")
        print(f"作者: {result.get('album_author')}")
        print(f"下载路径: {result.get('download_path')}")
    else:
        print("❌ 下载失败!")
        print(f"错误信息: {result.get('error')}")

def example_get_status():
    """示例：获取下载状态"""
    print("\n=== 获取下载状态示例 ===")
    
    album_id = "422866"
    
    # 获取下载状态
    status = call_tool('get_download_status', {'album_id': album_id})
    
    print(f"下载状态: {status}")

def example_cancel_download():
    """示例：取消下载"""
    print("\n=== 取消下载示例 ===")
    
    album_id = "422866"
    
    # 取消下载
    result = call_tool('cancel_download', {'album_id': album_id})
    
    print(f"取消结果: {result}")

def example_get_all_status():
    """示例：获取所有下载状态"""
    print("\n=== 获取所有下载状态示例 ===")
    
    # 获取所有状态
    all_status = call_tool('get_all_status', {})
    
    print(f"所有下载状态: {all_status}")

def example_list_tools():
    """示例：列出所有可用工具"""
    print("\n=== 可用工具列表 ===")
    
    tools = get_tools()
    
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool['name']}")
        print(f"   描述: {tool['description']}")
        print(f"   参数: {tool['parameters']}")
        print()

def main():
    """主函数"""
    print("Comic Downloader 使用示例")
    print("=" * 50)
    
    # 列出可用工具
    example_list_tools()
    
    # 下载漫画示例
    example_download_comic()
    
    # 获取状态示例
    example_get_status()
    
    # 取消下载示例
    example_cancel_download()
    
    # 获取所有状态示例
    example_get_all_status()
    
    print("\n示例执行完成!")

if __name__ == "__main__":
    main() 