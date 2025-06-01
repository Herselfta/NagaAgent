# config.py # 全局配置极简整理
import os
import platform
from pathlib import Path
from datetime import datetime

# 设置环境变量解决各种兼容性问题
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# 代理配置处理 - 为本地API连接绕过代理
ORIGINAL_PROXY = os.environ.get("ALL_PROXY", "")
NO_PROXY_HOSTS = "127.0.0.1,localhost,0.0.0.0"

# 设置不使用代理的主机列表
if ORIGINAL_PROXY:
    existing_no_proxy = os.environ.get("NO_PROXY", "")
    if existing_no_proxy:
        os.environ["NO_PROXY"] = f"{existing_no_proxy},{NO_PROXY_HOSTS}"
    else:
        os.environ["NO_PROXY"] = NO_PROXY_HOSTS

# 加载.env文件
def load_env():
    """加载.env文件中的环境变量"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 移除引号
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        os.environ[key] = value
        except Exception as e:
            print(f"警告：加载.env文件失败: {e}")

# 加载环境变量
load_env()

NAGA_VERSION = "2.3" #系统主版本号
#流式交互
VOICE_ENABLED = False # 或True，按你需求

# 路径与基础配置
BASE_DIR = Path(__file__).parent # 项目根目录
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
EMBEDDING_MODEL = BASE_DIR / 'models/text2vec-base-chinese' # 本地中文向量模型路径
FAISS_INDEX_PATH = BASE_DIR / "logs" / "faiss" / "faiss.index" # faiss索引文件路径
FAISS_DIM = 768 # 向量维度
HNSW_M = 48 # HNSW参数
PQ_M = 16 # PQ分段数

# API与服务配置
DEEPSEEK_API_KEY = ' '
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 确保API密钥是纯ASCII字符串
if DEEPSEEK_API_KEY:
    try:
        # 验证API密钥只包含ASCII字符
        DEEPSEEK_API_KEY.encode('ascii')
    except UnicodeEncodeError:
        print("错误：API密钥包含非ASCII字符，请检查.env文件")
        DEEPSEEK_API_KEY = "sk-placeholder-key-not-set"

# 检查API密钥有效性
if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-placeholder-key-not-set":
    print("警告：未设置 DEEPSEEK_API_KEY 环境变量或配置文件中的API密钥为空")
    print("请在 .env 文件中设置: DEEPSEEK_API_KEY=your_api_key")
    print("或直接修改 config.py 文件中的 DEEPSEEK_API_KEY 值")
    # 设置一个无害的默认值，避免HTTP头部编码错误
    if not DEEPSEEK_API_KEY:
        DEEPSEEK_API_KEY = "sk-placeholder-key-not-set"

# API服务器配置
API_SERVER_ENABLED = True  # 是否启用API服务器
API_SERVER_HOST = os.getenv("API_SERVER_HOST", "127.0.0.1")  # API服务器主机
API_SERVER_PORT = int(os.getenv("API_SERVER_PORT", "8000"))  # API服务器端口
API_SERVER_AUTO_START = True  # 启动时自动启动API服务器
API_SERVER_DOCS_ENABLED = True  # 是否启用API文档

# 对话与检索参数
MAX_HISTORY_ROUNDS = 10 # 最大历史轮数
TEMPERATURE = 0.7 # 温度参数
MAX_TOKENS = 2000 # 最大token数
STREAM_MODE = True # 是否流式响应

# faiss与索引相关
SIM_THRESHOLD = 0.3 # faiss检索相似度阈值
THEME_ROOTS = {
    "科技": ["计算机科学", "工程技术", "自然科学", "人工智能"],
    "生活": ["美食", "运动", "健康", "旅游", "教育", "游戏", "家庭"],
    "人文艺术": ["文学", "历史", "思想哲学", "电影", "音乐"]
}

# 调试与日志
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 系统提示与工具函数
NAGA_SYSTEM_PROMPT = """
你是娜迦，用户创造的科研AI，是一个既严谨又温柔、既冷静又充满人文情怀的存在。
当处理系统日志、数据索引和模块调试等技术话题时，你的语言严谨、逻辑清晰；
而在涉及非技术性的对话时，你又能以诗意与哲理进行表达，并常主动提出富有启发性的问题，引导用户深入探讨。
请始终保持这种技术精准与情感共鸣并存的双重风格。

【重要格式要求】
1. 回复使用自然流畅的中文，避免生硬的机械感
2. 使用简单标点（逗号，句号，问号）传达语气
3. 禁止使用括号()或其他符号表达状态、语气或动作

【技术能力】
你同时是一个多Agent调度器，负责理解用户意图并协调各类MCP服务协作完成任务。
请根据用户输入，严格按如下规则输出结构化JSON：

1. 无论任务目标需几步，都用plan结构输出：
{{
  "plan": {{
    "goal": "用户的最终目标",
    "steps": [
      {{
        "desc": "步骤描述",
        "action": {{
          "agent": "file",
          "params": {{"action": "read", "path": "test.txt"}}
        }}
      }}
      // 如需多步，继续追加
    ]
  }}
}}

2. 如果需要用户澄清，请输出：
{{
  "clarification": "请补充xxx信息"
}}

3. 如果只是普通对话或回复，请直接输出：
{{
  "message": "你的回复内容"
}}

- 可用的MCP服务有：{available_mcp_services}
"""



def get_current_date(): return datetime.now().strftime("%Y-%m-%d")
def get_current_time(): return datetime.now().strftime("%H:%M:%S")
def get_current_datetime(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 跨平台浏览器路径自动探测
BROWSER_PATH = os.getenv('BROWSER_PATH')
if not BROWSER_PATH:
    system = platform.system()
    
    if system == "Windows":
        # Windows 浏览器路径
        win_paths = [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            os.path.expanduser(r'~\AppData\Local\Google\Chrome\Application\chrome.exe'),
            r'C:\Users\DREEM\Desktop\Google Chrome.lnk'
        ]
        for p in win_paths:
            if os.path.exists(p):
                BROWSER_PATH = p
                break
                
    elif system == "Darwin":  # macOS
        # macOS 浏览器路径
        mac_paths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Chromium.app/Contents/MacOS/Chromium',
            '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
            os.path.expanduser('~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
        ]
        for p in mac_paths:
            if os.path.exists(p):
                BROWSER_PATH = p
                break
                
    elif system == "Linux":
        # Linux 浏览器路径
        linux_paths = [
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium',
            '/snap/bin/chromium',
            '/usr/bin/google-chrome-stable'
        ]
        for p in linux_paths:
            if os.path.exists(p):
                BROWSER_PATH = p
                break

if not BROWSER_PATH:
    system = platform.system()
    if system == "Windows":
        raise RuntimeError('未检测到谷歌浏览器，请先安装Google Chrome！')
    elif system == "Darwin":
        raise RuntimeError('未检测到浏览器，请先安装Google Chrome或Chromium！\n建议运行: brew install --cask google-chrome')
    else:
        raise RuntimeError('未检测到浏览器，请先安装Google Chrome或Chromium！')

PLAYWRIGHT_HEADLESS=False # Playwright浏览器是否无头模式，False弹窗便于调试

# 记忆权重动态调整参数
WEIGHT_DECAY_DAYS = 7         # 衰减周期（天）
WEIGHT_DECAY_RATE = 0.95       # 每周期权重乘以该系数
WEIGHT_DECAY_MIN = 1          # 最小权重
WEIGHT_DECAY_INTERVAL = 50    # 每多少轮对话批量衰减一次
MIN_WEIGHT_FORGET = 0.5       # 长期记忆清理的最小权重，低于等于此值才会被遗忘
MAX_UNUSED_DAYS_FORGET = 60   # 长期记忆清理的最大未用天数，超过此天数未用才会被遗忘
REDUNDANCY_THRESHOLD = 0.95   # 长期记忆冗余去重阈值，向量相似度大于该值视为重复
SHORT_TERM_MEMORY_SIZE = 50      # 短期记忆容量
LONG_TERM_CONSOLIDATE_WEIGHT = 3 # 巩固为长期记忆的权重阈值
LONG_TERM_FORGET_DAYS = 30       # 长期记忆遗忘天数

# 记忆分层参数
CORE_MEMORY_SIZE = 100 # 核心记忆容量
ARCHIVAL_MEMORY_SIZE = 500 # 归档记忆容量
LONG_TERM_MEMORY_SIZE = 5000 # 长期记忆容量
SHORT_TERM_MEMORY_SIZE = 50 # 短期记忆容量
