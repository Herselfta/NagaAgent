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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-placeholder-key-not-set")
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
MAX_HISTORY_ROUNDS = 19
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

# 快速响应小模型配置
# 用于快速决策和JSON格式化的轻量级模型
QUICK_MODEL_ENABLED = os.getenv("QUICK_MODEL_ENABLED", "false").lower() == "true"
QUICK_MODEL_API_KEY = os.getenv("QUICK_MODEL_API_KEY", "")  # 小模型API密钥
QUICK_MODEL_BASE_URL = os.getenv("QUICK_MODEL_BASE_URL", "")  # 小模型API地址
QUICK_MODEL_NAME = os.getenv("QUICK_MODEL_NAME", "qwen2.5-1.5b-instruct")  # 小模型名称

# 小模型参数配置
QUICK_MODEL_CONFIG = {
    "enabled": QUICK_MODEL_ENABLED,
    "api_key": QUICK_MODEL_API_KEY,
    "base_url": QUICK_MODEL_BASE_URL,
    "model_name": QUICK_MODEL_NAME,
    "max_tokens": 512,  # 小模型输出限制
    "temperature": 0.05,  # 极低温度确保稳定一致的输出
    "timeout": 5,  # 快速响应超时时间
    "max_retries": 2,  # 最大重试次数
    
    # 功能配置
    "quick_decision_enabled": True,  # 快速决策功能
    "json_format_enabled": True,    # JSON格式化功能
    "output_filter_enabled": True,  # 输出内容过滤功能
    "difficulty_judgment_enabled": True,  # 问题难度判断功能
    "scoring_system_enabled": True,  # 黑白名单打分系统
    "thinking_completeness_enabled": True,  # 思考完整性判断功能
}

# 输出过滤配置
OUTPUT_FILTER_CONFIG = {
    "filter_think_tags": True,  # 过滤<think></think>标签内容
    "filter_patterns": [
        r'<think>.*?</think>',  # 思考标签
        r'<thinking>.*?</thinking>',  # 思考标签
        r'<reflection>.*?</reflection>',  # 反思标签
        r'<internal>.*?</internal>',  # 内部思考标签
    ],
    "clean_output": True,  # 清理多余空白字符
}

# 问题难度判断配置
DIFFICULTY_JUDGMENT_CONFIG = {
    "enabled": True,
    "use_small_model": True,  # 使用小模型进行难度判断
    "difficulty_levels": ["简单", "中等", "困难", "极难"],
    "factors": [
        "概念复杂度",
        "推理深度", 
        "知识广度",
        "计算复杂度",
        "创新要求"
    ],
    "threshold_simple": 2,    # 简单问题阈值
    "threshold_medium": 4,    # 中等问题阈值
    "threshold_hard": 6,      # 困难问题阈值
}

# 黑白名单打分系统配置
SCORING_SYSTEM_CONFIG = {
    "enabled": True,
    "score_range": [1, 5],  # 评分范围：1-5分
    "score_threshold": 2,   # 结果保留阈值：2分及以下不保留
    "similarity_threshold": 0.85,  # 相似结果识别阈值
    "max_user_preferences": 3,  # 用户最多选择3个偏好
    "default_preferences": [
        "逻辑清晰准确",
        "实用性强", 
        "创新思维"
    ],
    "penalty_for_similar": 1,  # 相似结果的惩罚分数
    "min_results_required": 2,  # 最少保留结果数量（即使低于阈值）
    "strict_filtering": True,  # 严格过滤模式：True时严格按阈值过滤，False时保证最少结果数量
}

# 思考完整性判断配置
THINKING_COMPLETENESS_CONFIG = {
    "enabled": True,
    "use_small_model": True,  # 使用小模型判断思考完整性
    "completeness_criteria": [
        "问题分析充分",
        "解决方案明确",
        "逻辑链条完整",
        "结论清晰合理"
    ],
    "completeness_threshold": 0.8,  # 完整性阈值（0-1）
    "max_thinking_depth": 5,  # 最大思考深度层级
    "next_question_generation": True,  # 生成下一级问题
}

# 快速决策系统提示词
QUICK_DECISION_SYSTEM_PROMPT = """你是一个快速决策助手，专门进行简单判断和分类任务。
请根据用户输入快速给出准确的判断结果，保持简洁明确。
不需要详细解释，只需要给出核心判断结果。
【重要】：只输出最终结果，不要包含思考过程或<think>标签。"""

# JSON格式化系统提示词  
JSON_FORMAT_SYSTEM_PROMPT = """你是一个JSON格式化助手，专门将文本内容转换为结构化JSON格式。
请严格按照要求的JSON格式输出，确保语法正确且结构清晰。
只输出JSON内容，不要包含任何其他文字说明。
【重要】：只输出最终JSON，不要包含思考过程或<think>标签。"""

# 问题难度判断系统提示词
DIFFICULTY_JUDGMENT_SYSTEM_PROMPT = """你是一个问题难度评估专家，专门分析问题的复杂程度。
请根据问题的概念复杂度、推理深度、知识广度、计算复杂度、创新要求等因素进行评估。
只输出难度等级：简单、中等、困难、极难 中的一个。
【重要】：只输出难度等级，不要包含思考过程或解释。"""

# 结果打分系统提示词
RESULT_SCORING_SYSTEM_PROMPT = """你是一个结果评分专家，根据用户偏好和思考质量对结果进行1-5分评分。
评分标准：
- 5分：完全符合用户偏好，质量极高
- 4分：很好符合偏好，质量良好  
- 3分：基本符合偏好，质量一般
- 2分：部分符合偏好，质量较差
- 1分：不符合偏好或质量很差

请根据提供的思考结果和用户偏好进行评分。
【重要】：只输出数字分数，不要包含思考过程或解释。"""

# 思考完整性判断系统提示词
THINKING_COMPLETENESS_SYSTEM_PROMPT = """你是一个思考完整性评估专家，判断当前思考是否已经相对完整。
评估标准：
- 问题分析是否充分
- 解决方案是否明确
- 逻辑链条是否完整
- 结论是否清晰合理

如果思考完整，输出：完整
如果需要进一步思考，输出：不完整
【重要】：只输出"完整"或"不完整"，不要包含思考过程或解释。"""

# 下一级问题生成系统提示词
NEXT_QUESTION_SYSTEM_PROMPT = """你是一个问题设计专家，根据当前不完整的思考结果，设计下一级需要深入思考的核心问题。
要求：
- 问题应该针对当前思考的不足之处
- 问题应该能推进整体思考进程
- 问题应该具体明确，易于思考

请设计一个简洁的核心问题。
【重要】：只输出问题本身，不要包含思考过程或解释。"""
