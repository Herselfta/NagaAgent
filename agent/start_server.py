#!/usr/bin/env python3
# agent/start_server.py
# 启动代理API服务器

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.api_server import start_server

async def main():
    """主函数"""
    # 从环境变量获取配置
    config = {
        'api_key': os.getenv('API_Key'),
        'api_url': os.getenv('API_URL'),
        'server_key': os.getenv('Key'),
        'debug_mode': os.getenv("DEBUG", "False").lower() == "true"
    }
    
    # 获取服务器配置
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '8000'))
    
    print(f"启动代理API服务器...")
    print(f"配置信息:")
    print(f"  - API URL: {config['api_url']}")
    print(f"  - 服务器地址: {host}:{port}")
    print(f"  - 调试模式: {config['debug_mode']}")
    print(f"  - API密钥: {'已设置' if config['api_key'] else '未设置'}")
    print()
    
    try:
        await start_server(host=host, port=port, config=config)
    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭服务器...")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 