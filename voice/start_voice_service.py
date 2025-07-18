#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音服务启动脚本
支持HTTP和WebSocket两种模式
"""
import sys
import os
import argparse
import threading
import time
from pathlib import Path
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from handle_text import prepare_tts_input_with_context
from config import config
import ssl

def start_http_server():
    """启动HTTP TTS服务器"""
    try:
        from voice.server import app
        from gevent.pywsgi import WSGIServer
        
        print(f"🚀 启动HTTP TTS服务器...")
        print(f"📍 地址: http://127.0.0.1:{config.tts.port}")
        print(f"🔑 API密钥: {'已启用' if config.tts.require_api_key else '已禁用'}")
        
        http_server = WSGIServer(('0.0.0.0', config.tts.port), app)
        http_server.serve_forever()

    except Exception as e:
        print(f"❌ HTTP服务器启动失败: {e}")
        return False


    # from voice.server import app
    # from gevent.pywsgi import WSGIServer
    
    # print(f"🚀 启动HTTP TTS服务器...")
    # print(f"📍 地址: http://127.0.0.1:{config.tts.port}")
    # print(f"🔑 API密钥: {'已启用' if config.tts.require_api_key else '已禁用'}")
    
    # http_server = WSGIServer(('0.0.0.0', config.tts.port), app)
    # http_server.serve_forever()

# def establish_minimax_connection():
#     """建立Minimax WebSocket连接"""
#     url = "wss://api.minimaxi.com/ws/v1/t2a_v2"
#     headers = {"Authorization": f"Bearer {config.tts.api_key}"}
    
#     ssl_context = ssl.create_default_context()
#     ssl_context.check_hostname = False
#     ssl_context.verify_mode = ssl.CERT_NONE
    
#     try:
#         ws = await websockets.connect(url, additional_headers=headers, ssl=ssl_context)
#         connected = json.loads(await ws.recv())
#         if connected.get("event") == "connected_success":
#             logger.info("Minimax WebSocket连接成功")
#             return ws
#         else:
#             logger.error(f"Minimax连接失败: {connected}")
#             return None
#     except Exception as e:
#         logger.error(f"Minimax WebSocket连接异常: {e}")
#         return None

# def start_websocket_server():
#     """启动WebSocket TTS服务器"""
#     try:
#         import uvicorn
#         from voice.websocket_edge_tts import app
        
#         print(f"🚀 启动WebSocket TTS服务器...")
#         print(f"📍 地址: ws://127.0.0.1:{config.tts.port}")
#         print(f"🔑 API密钥: {'已启用' if config.tts.require_api_key else '已禁用'}")
        
#         uvicorn.run(app, host="0.0.0.0", port=config.tts.port)
#     except Exception as e:
#         print(f"❌ edgeTTS WebSocket服务器启动失败: {e}")
#         return False

def check_dependencies():
    """检查依赖是否安装"""
    missing_deps = []
    
    try:
        import edge_tts
    except ImportError:
        missing_deps.append("edge-tts")
    
    try:
        import emoji
    except ImportError:
        missing_deps.append("emoji")
    
    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")
    
    try:
        import soundfile
    except ImportError:
        missing_deps.append("soundfile")
    
    if missing_deps:
        print(f"❌ 缺少依赖: {', '.join(missing_deps)}")
        print("请运行: pip install " + " ".join(missing_deps))
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="语音服务启动器")
    parser.add_argument("--mode", choices=["http", "websocket", "both"], 
                       default="http", help="启动模式")
    parser.add_argument("--port", type=int, help="自定义端口")
    parser.add_argument("--check-deps", action="store_true", help="检查依赖")
    
    args = parser.parse_args()
    
    if args.check_deps:
        if check_dependencies():
            print("✅ 所有依赖已安装")
        return
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 设置端口
    if args.port:
        config.tts.port = args.port
    
    print("=" * 50)
    print("🎤 NagaAgent 语音服务")
    print("=" * 50)
    print(f"📋 配置信息:")
    print(f"   端口: {config.tts.port}")
    print(f"   默认语音: {config.tts.default_voice}")
    print(f"   默认格式: {config.tts.default_format}")
    print(f"   默认语速: {config.tts.default_speed}")
    print(f"   需要API密钥: {config.tts.require_api_key}")
    print(f"   mode: {args.mode}")
    print("=" * 50)
    
    if args.mode == "http":
        start_http_server()
    elif args.mode == "websocket":
        start_websocket_server()
    elif args.mode == "both":
        # 启动HTTP服务器在后台
        http_thread = threading.Thread(target=start_http_server, daemon=True)
        http_thread.start()
        time.sleep(1)
        
        # 启动WebSocket服务器
        start_websocket_server()

if __name__ == "__main__":
    main() 