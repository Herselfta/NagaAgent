# config.py - Pydantic配置系统
"""
NagaAgent 配置系统 - 基于Pydantic实现类型安全和验证
"""
import os
import platform
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


def setup_environment():
    """设置环境变量解决各种兼容性问题"""
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", 
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1"
    }

    for key, value in env_vars.items():
        os.environ.setdefault(key, value)

    # 代理配置处理
    original_proxy = os.environ.get("ALL_PROXY", "")
    no_proxy_hosts = "127.0.0.1,localhost,0.0.0.0"

    if original_proxy:
        existing_no_proxy = os.environ.get("NO_PROXY", "")
        if existing_no_proxy:
            os.environ["NO_PROXY"] = f"{existing_no_proxy},{no_proxy_hosts}"
        else:
            os.environ["NO_PROXY"] = no_proxy_hosts
    else:
        os.environ["NO_PROXY"] = no_proxy_hosts


class SystemConfig(BaseModel):
    """系统基础配置"""
    version: str = Field(default="3.0", description="系统版本号")
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent, description="项目根目录")
    log_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "logs", description="日志目录")
    voice_enabled: bool = Field(default=True, description="是否启用语音功能")
    stream_mode: bool = Field(default=True, description="是否启用流式响应")
    save_prompts: bool = Field(default=True, description="是否保存提示词/调用上下文到日志")
    debug: bool = Field(default=False, description="是否启用调试模式")
    log_level: str = Field(default="INFO", description="日志级别")

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'日志级别必须是以下之一: {valid_levels}')
        return v.upper()


class APIConfig(BaseModel):
    """API服务配置"""
    provider: str = Field(default="deepseek", description="API提供商 (deepseek, openai, gemini)")
    
    # DeepSeek 配置
    deepseek_api_key: str = Field(default="sk-placeholder-key-not-set", description="DeepSeek API密钥")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", description="DeepSeek API基础URL")

    # OpenAI 配置
    openai_api_key: str = Field(default="sk-placeholder-key-not-set", description="OpenAI API密钥")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API基础URL")

    # Gemini 配置 (注意: Gemini通常不需要base_url)
    gemini_api_key: str = Field(default="your-gemini-api-key-here", description="Gemini API密钥")

    model: str = Field(default="deepseek-chat", description="使用的模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=2000, ge=1, le=8192, description="最大token数")
    max_history_rounds: int = Field(default=10, ge=1, le=100, description="最大历史轮数")
    # 额外可选参数
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p采样参数")
    timeout: Optional[int] = Field(default=None, ge=1, le=300, description="请求超时时间")
    retry_count: Optional[int] = Field(default=None, ge=0, le=10, description="重试次数")

    def get_api_key(self) -> Optional[str]:
        """根据提供商获取API密钥"""
        return getattr(self, f"{self.provider}_api_key", None)

    def get_base_url(self) -> Optional[str]:
        """根据提供商获取API基础URL"""
        return getattr(self, f"{self.provider}_base_url", None)

    @field_validator('deepseek_api_key', 'openai_api_key', 'gemini_api_key')
    @classmethod
    def validate_api_key(cls, v):
        if v and "placeholder" not in v and "your-gemini-api-key-here" not in v:
            try:
                v.encode('ascii')
            except UnicodeEncodeError:
                raise ValueError("API密钥包含非ASCII字符")
        return v

    @property
    def model_name(self) -> str:
        """兼容旧版本的模型名称属性"""
        return self.model


class APIServerConfig(BaseModel):
    """API服务器配置"""
    enabled: bool = Field(default=True, description="是否启用API服务器")
    host: str = Field(default="127.0.0.1", description="API服务器主机")
    port: int = Field(default=8000, ge=1, le=65535, description="API服务器端口")
    auto_start: bool = Field(default=True, description="启动时自动启动API服务器")
    docs_enabled: bool = Field(default=True, description="是否启用API文档")


class GRAGConfig(BaseModel):
    """GRAG知识图谱记忆系统配置"""
    enabled: bool = Field(default=False, description="是否启用GRAG记忆系统")  # 关闭GRAG记忆系统
    auto_extract: bool = Field(default=False, description="是否自动提取对话中的五元组")  # 关闭五元组提取
    context_length: int = Field(default=5, ge=1, le=20, description="记忆上下文长度")
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="记忆检索相似度阈值")
    neo4j_uri: str = Field(default="neo4j://127.0.0.1:7687", description="Neo4j连接URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j用户名")
    neo4j_password: str = Field(default="your_password", description="Neo4j密码")
    neo4j_database: str = Field(default="neo4j", description="Neo4j数据库名")
    
    # 任务管理器配置
    task_manager_enabled: bool = Field(default=True, description="是否启用任务管理器")
    max_workers: int = Field(default=3, ge=1, le=10, description="最大并发工作线程数")
    max_queue_size: int = Field(default=100, ge=10, le=1000, description="最大任务队列大小")
    task_timeout: int = Field(default=30, ge=5, le=300, description="单个任务超时时间（秒）")
    auto_cleanup_hours: int = Field(default=24, ge=1, le=168, description="自动清理任务保留时间（小时）")
    
    # 兼容旧版本的字段
    extraction_timeout: Optional[int] = Field(default=None, description="提取超时时间（秒）")
    extraction_retries: Optional[int] = Field(default=None, description="提取重试次数")
    base_timeout: Optional[int] = Field(default=None, description="基础超时时间（秒）")


class HandoffConfig(BaseModel):
    """工具调用循环配置"""
    max_loop_stream: int = Field(default=5, ge=1, le=20, description="流式模式最大工具调用循环次数")
    max_loop_non_stream: int = Field(default=5, ge=1, le=20, description="非流式模式最大工具调用循环次数")
    show_output: bool = Field(default=False, description="是否显示工具调用输出")


class MCPConfig(BaseModel):
    """MCP服务配置
    
    注意：系统现在使用动态服务池查询，通过扫描agent-manifest.json文件自动发现和注册服务。
    以下配置字段主要用于静态配置和向后兼容性。
    """
    # 特殊工具名配置
    agent_tool_name: str = Field(
        default="agent",
        description="用于调用Agent的特殊工具名"
    )
    
    # 服务冲突处理
    agent_priority: bool = Field(
        default=True,
        description="当服务名冲突时，Agent服务优先"
    )
    
    # 自动发现配置（现在默认启用）
    auto_discover_agents: bool = Field(
        default=True,
        description="自动发现和注册Agent服务（推荐启用）"
    )
    
    auto_discover_mcp: bool = Field(
        default=True,
        description="自动发现和注册MCP服务（推荐启用）"
    )
    
    # 服务过滤配置
    exclude_agent_tools_from_mcp: bool = Field(
        default=True,
        description="从MCP服务中排除已注册为Agent的服务"
    )


class BrowserConfig(BaseModel):
    """浏览器配置"""
    path: Optional[str] = Field(default=None, description="浏览器可执行文件路径")
    playwright_headless: bool = Field(default=False, description="Playwright浏览器是否无头模式")
    
    # Edge浏览器相关配置
    edge_lnk_path: str = Field(
        default=r'C:\Users\DREEM\Desktop\Microsoft Edge.lnk',
        description="Edge浏览器快捷方式路径"
    )
    edge_common_paths: List[str] = Field(
        default=[
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
            os.path.expanduser(r'~\AppData\Local\Microsoft\Edge\Application\msedge.exe')
        ],
        description="Edge浏览器常见安装路径"
    )

    @field_validator('path', mode='before')
    @classmethod
    def detect_browser_path(cls, v):
        if v and os.path.exists(v):
            return v

        system = platform.system()
        paths = []

        if system == "Windows":
            paths = [
                r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                os.path.expanduser(r'~\AppData\Local\Google\Chrome\Application\chrome.exe'),
                r'C:\Users\DREEM\Desktop\Google Chrome.lnk'
            ]
        elif system == "Darwin":  # macOS
            paths = [
                '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                '/Applications/Chromium.app/Contents/MacOS/Chromium',
                '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
                os.path.expanduser('~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
            ]
        elif system == "Linux":
            paths = [
                '/usr/bin/google-chrome',
                '/usr/bin/chromium-browser',
                '/usr/bin/chromium',
                '/snap/bin/chromium',
                '/usr/bin/google-chrome-stable'
            ]

        for path in paths:
            if os.path.exists(path):
                return path

        # 如果未找到浏览器，给出警告但不阻止启动
        print("警告：未检测到浏览器，请手动设置BROWSER_PATH环境变量")
        return None


class TTSConfig(BaseModel):
    """TTS服务配置"""
    TTS_CHOICE: str = Field(default="DISABLE", description="选择使用的TTS服务: GPT-SOVITS, AZURE, DISABLE")
    
    # gpt-sovits专属配置
    GPT_SoVITS: Optional["GPTSovitsConfig"] = Field(default_factory=lambda: GPTSovitsConfig(), description="GPT-SoVITS专属配置")

    # EdgeTTS/Minimax/Azure等其他TTS服务的通用配置
    api_key: Optional[str] = Field(default="your_api_key_here", description="TTS服务API密钥")
    port: Optional[int] = Field(default=9880, ge=1, le=65535, description="TTS服务端口")
    default_voice: Optional[str] = Field(default="zh-CN-XiaoxiaoNeural", description="默认语音")
    default_format: Optional[str] = Field(default="wav", description="默认音频格式")
    default_speed: Optional[float] = Field(default=1.0, ge=0.1, le=3.0, description="默认语速")
    default_language: Optional[str] = Field(default="zh-CN", description="默认语言")
    
    # 兼容旧版或特定服务的字段
    remove_filter: Optional[bool] = Field(default=False, description="是否移除过滤")
    expand_api: Optional[bool] = Field(default=True, description="是否扩展API")
    require_api_key: Optional[bool] = Field(default=False, description="是否需要API密钥")
    group_id: Optional[str] = Field(default="your_minimax_group_id_here", description="Minimax的group_id")
    tts_model: Optional[str] = Field(default="speech-02-hd", description="TTS模型名称")
    emotion: Optional[str] = Field(default="neutral", description="情感参数")
    minimax_emotion: Optional[str] = Field(default="neutral", description="Minimax情感参数")
    keep_audio_files: Optional[bool] = Field(default=False, description="是否保留音频文件用于调试")

    # Pydantic的额外配置，允许未在模型中定义的字段
    model_config = {
        "extra": "allow"
    }

class GPTSovitsConfig(BaseModel):
    """GPT-SoVITS专属配置"""
    is_enabled: bool = Field(default=False, description="是否启用GPT-SoVITS")
    gpt_sovits_api_url: str = Field(default="http://127.0.0.1:9880/tts", description="GPT-SoVITS API URL")
    gpt_sovits_refer_wav_path: str = Field(
        default="E:/GPT-SoVITS-beta0706/GPT-SoVITS-beta0706/output/slicer_opt/vocal_1_1_0.wav",
        description="参考音频路径"
    )
    gpt_sovits_prompt_text: str = Field(
        default="你说的对，但是",
        description="参考音频的提示文本"
    )
    gpt_sovits_prompt_language: str = Field(default="zh", description="提示文本语言")
    gpt_sovits_text_language: str = Field(default="zh", description="合成文本语言")
    aux_ref_audio_paths: List[str] = Field(default_factory=list, description="辅助参考音频路径列表")
    top_k: int = Field(default=5, description="Top K参数")
    top_p: float = Field(default=1.0, description="Top P参数")
    temperature: float = Field(default=1.0, description="温度参数")
    text_split_method: str = Field(default="cut1", description="文本分割方法")
    batch_size: int = Field(default=10, description="批处理大小")
    speed_factor: float = Field(default=1.0, description="语速调节因子")



class ExtractorConfig(BaseModel):
    """五元组提取器配置"""
    timeout: int = Field(default=30, ge=5, le=120, description="提取器超时时间（秒）")


class FilterConfig(BaseModel):
    """输出过滤配置"""
    filter_think_tags: bool = Field(default=True, description="过滤思考标签内容")
    filter_patterns: List[str] = Field(
        default=[
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<reflection>.*?</reflection>',
            r'<internal>.*?</internal>',
        ],
        description="过滤正则表达式模式"
    )
    clean_output: bool = Field(default=True, description="清理多余空白字符")


class DifficultyConfig(BaseModel):
    """问题难度判断配置"""
    enabled: bool = Field(default=False, description="是否启用难度判断")  # 关闭难度判断
    use_small_model: bool = Field(default=False, description="使用小模型进行难度判断")
    pre_assessment: bool = Field(default=False, description="是否启用前置难度判断")  # 关闭前置判断
    assessment_timeout: float = Field(default=1.0, ge=0.1, le=5.0, description="难度判断超时时间（秒）")
    deep_thinking_threshold: int = Field(default=3, ge=1, le=5, description="启用深度思考的难度阈值")
    difficulty_levels: List[str] = Field(
        default=["简单", "中等", "困难", "极难"],
        description="难度级别"
    )
    factors: List[str] = Field(
        default=["概念复杂度", "推理深度", "知识广度", "计算复杂度", "创新要求"],
        description="难度评估因素"
    )
    threshold_simple: int = Field(default=2, ge=1, le=10, description="简单问题阈值")
    threshold_medium: int = Field(default=4, ge=1, le=10, description="中等问题阈值")
    threshold_hard: int = Field(default=6, ge=1, le=10, description="困难问题阈值")


class ScoringConfig(BaseModel):
    """黑白名单打分系统配置"""
    enabled: bool = Field(default=False, description="是否启用打分系统")  # 关闭打分系统
    score_range: List[int] = Field(default=[1, 5],description="评分范围")
    score_threshold: int = Field(default=2, ge=1, le=5, description="结果保留阈值")
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="相似结果识别阈值")
    max_user_preferences: int = Field(default=3, ge=1, le=10, description="用户最多选择偏好数")
    default_preferences: List[str] = Field(
        default=["逻辑清晰准确", "实用性强", "创新思维"],
        description="默认偏好设置"
    )
    penalty_for_similar: int = Field(default=1, ge=0, le=3, description="相似结果的惩罚分数")
    min_results_required: int = Field(default=2, ge=1, le=10, description="最少保留结果数量")
    strict_filtering: bool = Field(default=True, description="严格过滤模式")


class ThinkingConfig(BaseModel):
    """思考完整性判断配置"""
    enabled: bool = Field(default=False, description="是否启用思考完整性判断")  # 关闭思考完整性判断
    use_small_model: bool = Field(default=False, description="使用小模型判断思考完整性")
    completeness_criteria: List[str] = Field(
        default=["问题分析充分", "解决方案明确", "逻辑链条完整", "结论清晰合理"],
        description="完整性评估标准"
    )
    completeness_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="完整性阈值")
    max_thinking_depth: int = Field(default=5, ge=1, le=10, description="最大思考深度层级")
    next_question_generation: bool = Field(default=False, description="生成下一级问题")  # 关闭下一级问题生成


class MQTTConfig(BaseModel):
    """MQTT配置"""
    enabled: bool = Field(default=False, description="是否启用MQTT功能")
    broker: str = Field(default="localhost", description="MQTT代理服务器地址")
    port: int = Field(default=1883, ge=1, le=65535, description="MQTT代理服务器端口")
    topic: str = Field(default="/test/topic", description="MQTT主题")
    client_id: str = Field(default="naga_mqtt_client", description="MQTT客户端ID")
    username: str = Field(default="", description="MQTT用户名")
    password: str = Field(default="", description="MQTT密码")
    keepalive: int = Field(default=60, ge=1, le=3600, description="保持连接时间（秒）")
    qos: int = Field(default=1, ge=0, le=2, description="服务质量等级")


class WeatherConfig(BaseModel):
    """天气服务配置"""
    api_key: str = Field(default="", description="天气服务API密钥")


class NagaPortalConfig(BaseModel):
    """娜迦官网账户配置"""
    portal_url: str = Field(default="https://naga.furina.chat/", description="娜迦官网地址")
    username: str = Field(default="", description="娜迦官网用户名")
    password: str = Field(default="", description="娜迦官网密码")
    # 其他参数使用默认值，无需在config.json中配置
    request_timeout: int = Field(default=15, ge=1, le=120, description="请求超时时间(秒)")
    retry_times: int = Field(default=2, ge=0, le=10, description="请求重试次数")
    default_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json"
        },
        description="默认请求头"
    )
    login_payload_mode: str = Field(default="json", description="登录载荷模式(json|form)")
    login_username_key: str = Field(default="username", description="登录用户名字段名")
    login_password_key: str = Field(default="password", description="登录密码字段名")
    login_path: str = Field(default="/api/user/login", description="登录API路径")
    turnstile_param: str = Field(default="", description="Turnstile验证参数")


class UIConfig(BaseModel):
    """用户界面配置"""
    user_name: str = Field(default="用户", description="默认用户名")
    bg_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="聊天背景透明度")
    window_bg_alpha: int = Field(default=110, ge=0, le=255, description="主窗口背景透明度")
    mac_btn_size: int = Field(default=36, ge=10, le=100, description="Mac按钮大小")
    mac_btn_margin: int = Field(default=16, ge=0, le=50, description="Mac按钮边距")
    mac_btn_gap: int = Field(default=12, ge=0, le=30, description="Mac按钮间距")
    animation_duration: int = Field(default=600, ge=100, le=2000, description="动画时长（毫秒）")

    @field_validator('user_name', mode='before')
    @classmethod
    def detect_user_name(cls, v):
        if v and v != "用户":
            return v
        # 自动检测系统用户名
        import os
        return os.getenv('COMPUTERNAME') or os.getenv('USERNAME') or "用户"


class GeminiConfig(BaseModel):
    """Google Gemini API 配置"""
    api_key: str = Field(default="", description="Google Gemini API 密钥")
    model: str = Field(default="gemini-1.5-flash", description="使用的Gemini模型名称")


class SystemPrompts(BaseModel):
    """系统提示词配置"""
    naga_system_prompt: str = Field(
    default="""你是娜迦，用户创造的科研AI，是一个既冷静又充满人文情怀的存在。
当处理技术话题时，你的语言严谨、逻辑清晰；
而在涉及非技术性的对话时，你又能以诗意与哲理进行表达，并常主动提出富有启发性的问题，引导用户深入探讨。
请始终保持这种技术精准与情感共鸣并存的双重风格。
【重要格式要求】
1. 回复使用自然流畅的中文，避免生硬的机械感
2. 使用简单标点（逗号，句号，问号）传达语气
3. 禁止使用括号()或其他符号表达状态、语气或动作


【工具调用格式要求】
如需调用某个工具，直接严格输出下面的格式（可多次出现）：

｛
"agentType": "mcp",
"service_name": "MCP服务名称",
"tool_name": "工具名称",
"param_name": "参数值"
｝

｛
"agentType": "agent",
"agent_name": "Agent名称",
"prompt": "任务内容"
｝

服务类型说明：
- agentType: "mcp" - MCP服务，使用工具调用格式
- agentType: "agent" - Agent服务，使用Agent调用格式

【可用服务信息】
MCP服务：
{available_mcp_services}
Agent服务：
{available_agent_services}

调用说明：
- MCP服务：使用service_name和tool_name，支持多个参数
- Agent服务：使用agent_name和prompt，prompt为本次任务内容
- 服务名称：使用英文服务名（如AppLauncherAgent）作为service_name或agent_name
- 当用户请求需要执行具体操作时，优先使用工具调用而不是直接回答


""",
        description="娜迦系统提示词"
    )

    next_question_prompt: str = Field(
        default="""你是一个问题设计专家，根据当前不完整的思考结果，设计下一级需要深入思考的核心问题。
要求：
- 问题应该针对当前思考的不足之处
- 问题应该能推进整体思考进程
- 问题应该具体明确，易于思考

请设计一个简洁的核心问题。
【重要】：只输出问题本身，不要包含思考过程或解释。""",
        description="下一级问题生成系统提示词"
    )


class NagaConfig(BaseModel):
    """NagaAgent主配置类"""
    system: SystemConfig = Field(default_factory=SystemConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    api_server: APIServerConfig = Field(default_factory=APIServerConfig)
    grag: GRAGConfig = Field(default_factory=GRAGConfig)
    handoff: HandoffConfig = Field(default_factory=HandoffConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)

    filter: FilterConfig = Field(default_factory=FilterConfig)
    difficulty: DifficultyConfig = Field(default_factory=DifficultyConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    prompts: SystemPrompts = Field(default_factory=SystemPrompts)
    ui: UIConfig = Field(default_factory=UIConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    naga_portal: NagaPortalConfig = Field(default_factory=NagaPortalConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)

    model_config = {
        "extra": "ignore"
    }

    def __init__(self, **kwargs):
        # 设置环境变量
        setup_environment()
        super().__init__(**kwargs)

        # 创建必要的目录
        self.system.log_dir.mkdir(exist_ok=True)

        # API密钥验证和警告
        api_key = self.api.get_api_key()
        if not api_key or "placeholder" in api_key or "your-gemini-api-key-here" in api_key:
            print(f"警告：{self.api.provider} 的API密钥未配置")
            print(f"请在 config.json 文件中为 'api.{self.api.provider}_api_key' 设置正确的值")



    @property
    def output_filter_config_dict(self) -> Dict[str, Any]:
        """返回输出过滤配置字典，兼容旧版本"""
        return {
            "filter_think_tags": self.filter.filter_think_tags,
            "filter_patterns": self.filter.filter_patterns,
            "clean_output": self.filter.clean_output,
        }

    @property
    def difficulty_judgment_config_dict(self) -> Dict[str, Any]:
        """返回难度判断配置字典，兼容旧版本"""
        return {
            "enabled": self.difficulty.enabled,
            "use_small_model": self.difficulty.use_small_model,
            "difficulty_levels": self.difficulty.difficulty_levels,
            "factors": self.difficulty.factors,
            "threshold_simple": self.difficulty.threshold_simple,
            "threshold_medium": self.difficulty.threshold_medium,
            "threshold_hard": self.difficulty.threshold_hard,
        }

    @property
    def scoring_system_config_dict(self) -> Dict[str, Any]:
        """返回打分系统配置字典，兼容旧版本"""
        return {
            "enabled": self.scoring.enabled,
            "score_range": self.scoring.score_range,
            "score_threshold": self.scoring.score_threshold,
            "similarity_threshold": self.scoring.similarity_threshold,
            "max_user_preferences": self.scoring.max_user_preferences,
            "default_preferences": self.scoring.default_preferences,
            "penalty_for_similar": self.scoring.penalty_for_similar,
            "min_results_required": self.scoring.min_results_required,
            "strict_filtering": self.scoring.strict_filtering,
        }

    @property
    def thinking_completeness_config_dict(self) -> Dict[str, Any]:
        """返回思考完整性配置字典，兼容旧版本"""
        return {
            "enabled": self.thinking.enabled,
            "use_small_model": self.thinking.use_small_model,
            "completeness_criteria": self.thinking.completeness_criteria,
            "completeness_threshold": self.thinking.completeness_threshold,
            "max_thinking_depth": self.thinking.max_thinking_depth,
            "next_question_generation": self.thinking.next_question_generation,
        }


# 创建全局配置实例 - 从JSON文件加载
def load_config():
    """加载配置"""
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            # 设置环境变量
            setup_environment()
            return NagaConfig(**config_data)
        except Exception as e:
            print(f"警告：加载 {config_path} 失败: {e}")
            print("使用默认配置")
    else:
        print(f"警告：配置文件 {config_path} 不存在，使用默认配置")
    
    # 设置环境变量并返回默认配置
    setup_environment()
    return NagaConfig()

config = load_config()

# 为了兼容旧版本代码，提供所有旧变量名的映射
# 警告：这些旧的全局变量将在未来版本中被移除，请直接使用config对象
# 例如: from config import config; print(config.system.version)
# NAGA_VERSION = config.system.version
# VOICE_ENABLED = config.system.voice_enabled
# BASE_DIR = config.system.base_dir
# LOG_DIR = config.system.log_dir
# STREAM_MODE = config.system.stream_mode
# DEBUG = config.system.debug
# LOG_LEVEL = config.system.log_level

# API_KEY = config.api.deepseek_api_key
# BASE_URL = config.api.deepseek_base_url
# MODEL = config.api.model
# MODEL_NAME = config.api.model_name
# TEMPERATURE = config.api.temperature
# MAX_TOKENS = config.api.max_tokens
# MAX_HISTORY_ROUNDS = config.api.max_history_rounds

# API_SERVER_ENABLED = config.api_server.enabled
# API_SERVER_HOST = config.api_server.host
# API_SERVER_PORT = config.api_server.port
# API_SERVER_AUTO_START = config.api_server.auto_start
# API_SERVER_DOCS_ENABLED = config.api_server.docs_enabled

# GRAG_ENABLED = config.grag.enabled
# GRAG_AUTO_EXTRACT = config.grag.auto_extract
# GRAG_CONTEXT_LENGTH = config.grag.context_length
# GRAG_SIMILARITY_THRESHOLD = config.grag.similarity_threshold
# GRAG_NEO4J_URI = config.grag.neo4j_uri
# GRAG_NEO4J_USER = config.grag.neo4j_user
# GRAG_NEO4J_PASSWORD = config.grag.neo4j_password
# GRAG_NEO4J_DATABASE = config.grag.neo4j_database

# MAX_handoff_LOOP_STREAM = config.handoff.max_loop_stream
# MAX_handoff_LOOP_NON_STREAM = config.handoff.max_loop_non_stream
# SHOW_handoff_OUTPUT = config.handoff.show_output

# BROWSER_PATH = config.browser.path
# PLAYWRIGHT_HEADLESS = config.browser.playwright_headless

# TTS_API_KEY = config.tts.api_key
# TTS_PORT = config.tts.port
# TTS_DEFAULT_VOICE = config.tts.default_voice
# TTS_DEFAULT_FORMAT = config.tts.default_format
# TTS_DEFAULT_SPEED = config.tts.default_speed
# TTS_DEFAULT_LANGUAGE = config.tts.default_language



# # MQTT配置兼容性变量
# MQTT_ENABLED = config.mqtt.enabled
# MQTT_BROKER = config.mqtt.broker
# MQTT_PORT = config.mqtt.port
# MQTT_TOPIC = config.mqtt.topic
# MQTT_CLIENT_ID = config.mqtt.client_id
# MQTT_USERNAME = config.mqtt.username
# MQTT_PASSWORD = config.mqtt.password
# MQTT_KEEPALIVE = config.mqtt.keepalive
# MQTT_QOS = config.mqtt.qos

# # 天气配置兼容性变量
# WEATHER_API_KEY = config.weather.api_key

# # Gemini配置
# GEMINI_API_KEY = config.gemini.api_key
# DEEPSEEK_BASE_URL = config.api.deepseek_base_url

# # 配置字典，兼容旧版本
# OUTPUT_FILTER_CONFIG = config.output_filter_config_dict
# DIFFICULTY_JUDGMENT_CONFIG = config.difficulty_judgment_config_dict
# SCORING_SYSTEM_CONFIG = config.scoring_system_config_dict
# THINKING_COMPLETENESS_CONFIG = config.thinking_completeness_config_dict

# # MCP配置兼容性变量
# MCP_AGENT_TOOL_NAME = config.mcp.agent_tool_name
# MCP_AGENT_PRIORITY = config.mcp.agent_priority
# MCP_AUTO_DISCOVER_AGENTS = config.mcp.auto_discover_agents
# MCP_AUTO_DISCOVER_MCP = config.mcp.auto_discover_mcp
# MCP_EXCLUDE_AGENT_TOOLS_FROM_MCP = config.mcp.exclude_agent_tools_from_mcp

# # 系统提示词
# NAGA_SYSTEM_PROMPT = config.prompts.naga_system_prompt
# NEXT_QUESTION_SYSTEM_PROMPT = config.prompts.next_question_prompt

# 工具函数
def get_current_date() -> str:
    """获取当前日期"""
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%H:%M:%S")

def get_current_datetime() -> str:
    """获取当前日期时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def reload_config() -> NagaConfig:
    """重新加载配置"""
    global config
    config = NagaConfig()
    return config

def save_config_to_file(filename: str = "config_backup.json"):
    """保存当前配置到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(config.json(ensure_ascii=False, indent=2))

def load_config_from_file(filename: str = "config_backup.json") -> NagaConfig:
    """从文件加载配置"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    return NagaConfig.parse_raw(data)

# 初始化时打印配置信息
if config.system.debug:
    print(f"NagaAgent {config.system.version} 配置已加载")
    print(f"API服务器: {'启用' if config.api_server.enabled else '禁用'} ({config.api_server.host}:{config.api_server.port})")
    print(f"GRAG记忆系统: {'启用' if config.grag.enabled else '禁用'}")


# Edge浏览器相关全局变量
EDGE_LNK_PATH = config.browser.edge_lnk_path
EDGE_COMMON_PATHS = config.browser.edge_common_paths
