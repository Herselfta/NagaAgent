#!/usr/bin/env python3
"""
快速测试脚本
用于快速测试漫画下载功能
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

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    try:
        # 测试导入
        from mcpserver.agent_comic_downloader.comic_downloader import ComicDownloader
        print("✅ 导入成功")
        
        # 测试创建下载器
        downloader = ComicDownloader()
        print("✅ 下载器创建成功")
        
        # 测试获取桌面路径
        desktop_path = downloader.get_desktop_path()
        print(f"✅ 桌面路径: {desktop_path}")
        
        # 测试创建配置
        option = downloader.create_simple_option("422866")
        print("✅ 配置创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_download_functionality():
    """测试下载功能"""
    print("\n测试下载功能...")
    
    try:
        from mcpserver.agent_comic_downloader.mcp_tools import call_tool
        
        # 测试下载（使用一个简单的测试ID）
        result = call_tool('download_comic', {'album_id': '422866'})
        
        print(f"下载结果: {result}")
        
        if result.get('success'):
            print("✅ 下载功能测试成功!")
            return True
        else:
            print(f"❌ 下载功能测试失败: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ 下载功能测试失败: {e}")
        return False

def test_server_functionality():
    """测试服务器功能"""
    print("\n测试服务器功能...")
    
    try:
        from mcpserver.agent_comic_downloader.server import app
        print("✅ 服务器导入成功")
        
        # 测试服务器启动（不实际启动）
        print("✅ 服务器功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 服务器功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("Comic Downloader 快速测试")
    print("=" * 40)
    
    # 测试基本功能
    basic_ok = test_basic_functionality()
    
    # 测试下载功能
    download_ok = test_download_functionality()
    
    # 测试服务器功能
    server_ok = test_server_functionality()
    
    # 总结
    print("\n" + "=" * 40)
    print("测试总结:")
    print(f"基本功能: {'✅ 通过' if basic_ok else '❌ 失败'}")
    print(f"下载功能: {'✅ 通过' if download_ok else '❌ 失败'}")
    print(f"服务器功能: {'✅ 通过' if server_ok else '❌ 失败'}")
    
    if all([basic_ok, download_ok, server_ok]):
        print("\n🎉 所有测试通过! 系统可以正常使用。")
    else:
        print("\n⚠️  部分测试失败，请检查配置和依赖。")

if __name__ == "__main__":
    main() 