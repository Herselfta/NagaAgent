#!/usr/bin/env python3
"""
清理不再需要的依赖包
删除faiss-cpu和sentence-transformers等不再使用的包
"""

import subprocess
import sys

def run_command(cmd):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def uninstall_package(package_name):
    """卸载指定的包"""
    print(f"🗑️  正在卸载 {package_name}...")
    success, stdout, stderr = run_command(f"pip uninstall {package_name} -y")
    if success:
        print(f"✅ 已卸载 {package_name}")
    else:
        print(f"❌ 卸载 {package_name} 失败: {stderr}")
    return success

def main():
    """主清理函数"""
    print("🧹 开始清理不再需要的依赖包...")
    print("=" * 50)
    
    # 需要删除的包列表
    packages_to_remove = [
        "faiss-cpu",           # 向量数据库，已替换为GRAG
        "sentence-transformers", # 句子向量化，已不再使用
        "huggingface-hub",     # 如果只用于下载模型，可以删除
        "tokenizers",          # 如果只用于sentence-transformers，可以删除
        "safetensors",         # 如果只用于模型加载，可以删除
    ]
    
    # 可选删除的包（如果确认不需要）
    optional_packages = [
        "langchain-core",      # LangChain相关，如果不用可以删除
        "langchain-deepseek",  # LangChain相关
        "langchain-google-genai", # LangChain相关
        "langchain-openai",    # LangChain相关
        "langgraph",           # LangGraph相关
        "langgraph-checkpoint", # LangGraph相关
        "langgraph-prebuilt",  # LangGraph相关
        "langgraph-sdk",       # LangGraph相关
        "langsmith",           # LangSmith相关
        "google-ai-generativelanguage", # Google AI相关
        "google-api-core",     # Google API相关
        "google-api-python-client", # Google API相关
        "google-auth",         # Google认证相关
        "google-auth-httplib2", # Google认证相关
        "google-generativeai", # Google Generative AI相关
        "googleapis-common-protos", # Google API相关
        "grpcio",              # gRPC相关
        "grpcio-status",       # gRPC相关
        "protobuf",            # Protocol Buffers
        "proto-plus",          # Protocol Buffers增强
    ]
    
    print("📋 将删除以下不再需要的包:")
    for pkg in packages_to_remove:
        print(f"   - {pkg}")
    
    print("\n❓ 可选删除的包（如果确认不需要）:")
    for pkg in optional_packages:
        print(f"   - {pkg}")
    
    # 询问是否删除可选包
    response = input("\n是否删除可选包？(y/N): ").strip().lower()
    if response in ['y', 'yes']:
        packages_to_remove.extend(optional_packages)
        print("✅ 将删除所有可选包")
    else:
        print("⏭️  跳过可选包")
    
    print("\n" + "=" * 50)
    
    # 开始卸载
    removed_count = 0
    for package in packages_to_remove:
        if uninstall_package(package):
            removed_count += 1
    
    print("\n" + "=" * 50)
    print(f"🎉 清理完成！共卸载了 {removed_count} 个包")
    
    # 建议重新安装依赖
    print("\n💡 建议执行以下操作:")
    print("1. 重新安装依赖: pip install -r requirements.txt")
    print("2. 检查环境: python check_env.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 