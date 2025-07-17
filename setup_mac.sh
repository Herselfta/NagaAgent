#!/bin/bash
# 一键配置脚本 for NagaAgent 3.0 (Mac 版本)
# 版本号由config.py统一管理

set -e  # 设置错误时停止执行

PYTHON_MIN_VERSION="3.8"  # Python最低版本要求
VENV_PATH=".venv"  # 虚拟环境路径

# 颜色输出函数
print_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

# 检查 Python 版本
print_info "检查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    print_error "未找到 python3 命令，请先安装 Python 3.8+"
    print_info "推荐使用 Homebrew 安装: brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | grep -o '[0-9]\+\.[0-9]\+')
REQUIRED_VERSION="3.8"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    print_error "需要 Python $PYTHON_MIN_VERSION 或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi

print_success "Python 版本检查通过: $PYTHON_VERSION"

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建并激活虚拟环境
if [[ ! -d "$VENV_PATH" ]]; then
    print_info "创建虚拟环境..."
    python3 -m venv "$VENV_PATH"
fi

print_info "激活虚拟环境..."
source "$VENV_PATH/bin/activate"

# 升级 pip
print_info "升级 pip..."
pip install --upgrade pip

# 检查是否为 Apple Silicon Mac
if [[ $(uname -m) == "arm64" ]]; then
    print_info "检测到 Apple Silicon Mac，将安装适配的依赖..."
    # 对于 Apple Silicon，某些包可能需要特殊处理
    export SYSTEM_VERSION_COMPAT=1
fi

# 安装系统依赖 (macOS 特定)
print_info "检查系统依赖..."
if ! command -v portaudio &> /dev/null; then
    print_warning "未检测到 PortAudio，PyAudio 可能安装失败"
    print_info "如需语音功能，请运行: brew install portaudio"
fi

# 安装 Python 依赖
print_info "安装 Python 依赖..."
if [ -f "pyproject.toml" ]; then
    echo "使用 uv 安装依赖..."
    uv sync || pip install -e .
elif [ -f "requirements.txt" ]; then
    echo "使用 requirements.txt 安装依赖..."
    pip install -r requirements.txt
else
    echo "未找到依赖文件，请检查项目配置"
    exit 1
fi

# 兼容旧的 requirements 文件扫描（如果存在）
for req_file in requirements*.txt 2>/dev/null; do
    if [[ -f "$req_file" ]]; then
        print_info "安装 $req_file..."
        pip install -r "$req_file" -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
done

# 特殊处理可能在 Mac 上有问题的包
print_info "检查特殊依赖..."

# PyAudio 在 Mac 上可能需要特殊处理
if ! pip show pyaudio > /dev/null 2>&1; then
    print_warning "PyAudio 安装失败，尝试使用 Homebrew 方式..."
    if command -v brew &> /dev/null; then
        brew install portaudio
        pip install pyaudio
    else
        print_warning "未找到 Homebrew，请手动安装 PortAudio: https://formulae.brew.sh/formula/portaudio"
    fi
fi

# 安装 playwright 浏览器驱动
print_info "安装 playwright 浏览器驱动..."
python -m playwright install chromium

# 验证 playwright 安装
print_info "验证 playwright 安装..."
PLAYWRIGHT_VERSION=$(python -m playwright --version)
if [[ $? -ne 0 ]]; then
    print_error "Playwright 安装验证失败"
    exit 1
fi
print_success "Playwright 版本: $PLAYWRIGHT_VERSION"

# 创建启动脚本
print_info "创建启动脚本..."
cat > start_mac.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
python main.py
EOF

chmod +x start_mac.sh

print_success "环境设置完成！"
print_info "启动方式:"
print_info "  ./start_mac.sh"
print_info ""
print_info "如需安装其他浏览器驱动，请运行:"
print_info "  python -m playwright install firefox   # 安装 Firefox"
print_info "  python -m playwright install webkit    # 安装 WebKit"
print_info ""
print_info "如遇到权限问题，请运行: chmod +x setup_mac.sh start_mac.sh" 