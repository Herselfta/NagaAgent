#!/bin/bash
# NagaAgent Mac 一键部署脚本
# 使用方法: ./quick_deploy_mac.sh

set -e

echo "🚀 NagaAgent Mac 一键部署开始..."

# 检查 Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ 未找到 Homebrew，正在安装..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 检查并安装 Python
if ! command -v python3 &> /dev/null; then
    echo "📦 安装 Python 3.11..."
    brew install python@3.11
fi

# 检查并安装 Chrome
if [[ ! -d "/Applications/Google Chrome.app" ]]; then
    echo "🌐 安装 Google Chrome..."
    brew install --cask google-chrome
fi

# 检查并安装 PortAudio (语音功能依赖)
if ! brew list portaudio &> /dev/null; then
    echo "🎤 安装 PortAudio (语音功能)..."
    brew install portaudio
fi

# 执行主要设置脚本
echo "⚙️ 执行环境配置..."
./setup_mac.sh

echo "✅ 部署完成！"
echo ""
echo "🎯 启动应用："
echo "   ./start_mac.sh"
echo ""
echo "📚 查看详细说明："
echo "   open README_MAC.md"