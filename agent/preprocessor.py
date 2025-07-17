# agent/preprocessor.py
# 预处理系统 - Python版本
import os
import json
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentPreprocessor")

class AgentPreprocessor:
    """代理预处理系统 - 处理消息中的占位符、变量替换等"""
    
    def __init__(self, project_base_path: str = None):
        self.project_base_path = project_base_path or os.getcwd()
        self.agent_dir = Path(self.project_base_path) / "agent"
        self.static_placeholder_values = {}  # 静态占位符值
        self.cached_emoji_lists = {}  # 缓存的表情包列表
        self.debug_mode = os.getenv("DEBUG", "False").lower() == "true"
        self.detectors = []  # 系统提示词转换规则
        self.super_detectors = []  # 全局上下文转换规则
        
        # 确保agent目录存在
        self.agent_dir.mkdir(exist_ok=True)
        
        # 初始化占位符处理器
        self._init_placeholder_processors()
    
    def _init_placeholder_processors(self):
        """初始化各种占位符处理器"""
        self.processors = {
            'agent': self._process_agent_placeholders,
            'env_vars': self._process_env_variables,
            'time_date': self._process_time_date,
            'static_plugins': self._process_static_plugins,
            'handoff_tools': self._process_handoff_tools,
            'emoji': self._process_emoji_placeholders,
            'diary': self._process_diary_placeholders,
            'system_prompts': self._process_system_prompts,
            'async_results': self._process_async_results
        }
        
        # 加载detectors和superDetectors规则
        for key, value in os.environ.items():
            if key.startswith('Detector') and key[8:].isdigit():
                index = key[8:]
                output_key = f'Detector_Output{index}'
                if output_key in os.environ:
                    self.detectors.append({
                        'detector': value,
                        'output': os.environ[output_key]
                    })
            
            if key.startswith('SuperDetector') and key[13:].isdigit():
                index = key[13:]
                output_key = f'SuperDetector_Output{index}'
                if output_key in os.environ:
                    self.super_detectors.append({
                        'detector': value,
                        'output': os.environ[output_key]
                    })
        
        if self.detectors:
            logger.info(f"加载了 {len(self.detectors)} 条系统提示词转换规则")
        if self.super_detectors:
            logger.info(f"加载了 {len(self.super_detectors)} 条全局上下文转换规则")
    
    async def preprocess_messages(self, messages: List[Dict], model: str = None) -> List[Dict]:
        """预处理消息列表"""
        if self.debug_mode:
            logger.info(f"开始预处理 {len(messages)} 条消息")
        
        processed_messages = []
        for msg in messages:
            processed_msg = await self._process_single_message(msg, model)
            processed_messages.append(processed_msg)
        
        if self.debug_mode:
            logger.info("消息预处理完成")
        
        return processed_messages
    
    async def _process_single_message(self, message: Dict, model: str = None) -> Dict:
        """处理单条消息"""
        processed_msg = message.copy()
        
        # 处理字符串内容
        if isinstance(processed_msg.get('content'), str):
            processed_msg['content'] = await self._process_text_content(processed_msg['content'], model)
        
        # 处理数组内容（如多模态消息）
        elif isinstance(processed_msg.get('content'), list):
            processed_content = []
            for part in processed_msg['content']:
                if isinstance(part, dict) and part.get('type') == 'text':
                    processed_part = part.copy()
                    processed_part['text'] = await self._process_text_content(part['text'], model)
                    processed_content.append(processed_part)
                else:
                    processed_content.append(part)
            processed_msg['content'] = processed_content
        
        return processed_msg
    
    async def _process_text_content(self, text: str, model: str = None) -> str:
        """处理文本内容中的所有占位符"""
        if not text:
            return text
        
        processed_text = text
        
        # 依次应用所有处理器
        for processor_name, processor_func in self.processors.items():
            try:
                processed_text = await processor_func(processed_text, model)
            except Exception as e:
                logger.error(f"处理器 {processor_name} 执行失败: {e}")
                continue
        
        return processed_text
    
    async def _process_agent_placeholders(self, text: str, model: str = None) -> str:
        """处理Agent占位符 {{AgentName}}"""
        # 查找所有Agent环境变量
        agent_configs = {}
        for key, value in os.environ.items():
            if key.startswith('Agent') and len(key) > 5:
                agent_name = key[5:]  # 去掉"Agent"前缀
                agent_configs[agent_name] = value
        
        for agent_name, agent_file in agent_configs.items():
            placeholder = f"{{{{{agent_name}}}}}"
            if placeholder in text:
                try:
                    agent_file_path = self.agent_dir / agent_file
                    if agent_file_path.exists():
                        with open(agent_file_path, 'r', encoding='utf-8') as f:
                            agent_content = f.read()
                        # 递归处理agent内容中的占位符
                        agent_content = await self._process_text_content(agent_content, model)
                        text = text.replace(placeholder, agent_content)
                    else:
                        text = text.replace(placeholder, f"[Agent {agent_name} ({agent_file}) not found]")
                except Exception as e:
                    logger.error(f"处理Agent占位符 {agent_name} 失败: {e}")
                    text = text.replace(placeholder, f"[Error processing Agent {agent_name}]")
        
        return text
    
    async def _process_env_variables(self, text: str, model: str = None) -> str:
        """处理环境变量占位符"""
        # 处理Tarxxx变量
        for key, value in os.environ.items():
            if key.startswith('Tar'):
                placeholder = f"{{{{{key}}}}}"
                text = text.replace(placeholder, value or f"未配置{key}")
        
        # 处理Varxxx变量
        for key, value in os.environ.items():
            if key.startswith('Var'):
                placeholder = f"{{{{{key}}}}}"
                text = text.replace(placeholder, value or f"未配置{key}")
        
        # 处理Port变量
        if 'PORT' in os.environ:
            text = text.replace("{{Port}}", os.environ['PORT'])
        
        # 处理Image_Key变量
        image_key = os.getenv('Image_Key')
        if image_key:
            text = text.replace("{{Image_Key}}", image_key)
        
        return text
    
    async def _process_time_date(self, text: str, model: str = None) -> str:
        """处理时间日期占位符"""
        now = datetime.now()
        
        # 日期
        date_str = now.strftime('%Y-%m-%d')
        text = text.replace("{{Date}}", date_str)
        
        # 时间
        time_str = now.strftime('%H:%M:%S')
        text = text.replace("{{Time}}", time_str)
        
        # 星期
        weekday_names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
        weekday = weekday_names[now.weekday()]
        text = text.replace("{{Today}}", weekday)
        
        return text
    
    async def _process_static_plugins(self, text: str, model: str = None) -> str:
        """处理静态插件占位符"""
        for placeholder, value in self.static_placeholder_values.items():
            if placeholder in text:
                text = text.replace(placeholder, value)
        
        return text
    
    async def _process_handoff_tools(self, text: str, model: str = None) -> str:
        """处理handoff工具占位符"""
        # 处理{{handoffAllTools}}占位符
        if "{{handoffAllTools}}" in text:
            handoff_descriptions = []
            # 这里可以从插件管理器获取工具描述
            handoff_descriptions.append("示例工具1: 这是一个示例工具的描述")
            handoff_descriptions.append("示例工具2: 这是另一个示例工具的描述")
            
            all_tools_string = "\n\n---\n\n".join(handoff_descriptions) if handoff_descriptions else "没有可用的handoff工具描述信息"
            text = text.replace("{{handoffAllTools}}", all_tools_string)
        
        # 处理{{handoffWeatherInfo}}占位符
        text = text.replace("{{handoffWeatherInfo}}", "天气信息不可用")
        
        return text
    
    async def _process_emoji_placeholders(self, text: str, model: str = None) -> str:
        """处理表情包占位符"""
        emoji_pattern = r'\{\{(.+?表情包)\}\}'
        
        def replace_emoji(match):
            emoji_name = match.group(1)
            emoji_list = self.cached_emoji_lists.get(emoji_name, f"{emoji_name}列表不可用")
            return emoji_list
        
        text = re.sub(emoji_pattern, replace_emoji, text)
        return text
    
    async def _process_diary_placeholders(self, text: str, model: str = None) -> str:
        """处理日记本占位符"""
        diary_pattern = r'\{\{(.+?)日记本\}\}'
        
        def replace_diary(match):
            character_name = match.group(1)
            # 这里可以从日记系统获取内容
            diary_content = f"[{character_name}日记本内容为空或未从插件获取]"
            return diary_content
        
        text = re.sub(diary_pattern, replace_diary, text)
        return text
    
    async def _process_system_prompts(self, text: str, model: str = None) -> str:
        """处理系统提示词转换规则"""
        # 这里可以从配置文件加载转换规则
        # 示例规则
        for rule in self.detectors:
            if rule['detector'] and rule['output']:
                text = text.replace(rule['detector'], rule['output'])
        
        return text
    
    async def _process_async_results(self, text: str, model: str = None) -> str:
        """处理异步结果占位符"""
        async_result_pattern = r'\{\{handoff_ASYNC_RESULT::([a-zA-Z0-9_.-]+)::([a-zA-Z0-9_-]+)\}\}'
        
        def replace_async_result(match):
            plugin_name = match.group(1)
            request_id = match.group(2)
            # 这里可以从异步结果目录读取结果
            result_text = f"[任务 {plugin_name} (ID: {request_id}) 结果待更新...]"
            return result_text
        
        text = re.sub(async_result_pattern, replace_async_result, text)
        return text
    
    def set_static_placeholder(self, placeholder: str, value: str):
        """设置静态占位符值"""
        self.static_placeholder_values[placeholder] = value
    
    def set_emoji_list(self, emoji_name: str, emoji_list: str):
        """设置表情包列表"""
        self.cached_emoji_lists[emoji_name] = emoji_list
    
    async def process_image_content(self, messages: List[Dict]) -> List[Dict]:
        """处理图片内容"""
        # 检查是否需要图片处理
        should_process_images = True
        
        for msg in messages:
            if msg.get('role') in ['user', 'system']:
                content = msg.get('content', '')
                if isinstance(content, str) and '{{ShowBase64}}' in content:
                    should_process_images = False
                    # 移除ShowBase64占位符
                    msg['content'] = content.replace('{{ShowBase64}}', '')
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            if '{{ShowBase64}}' in part.get('text', ''):
                                should_process_images = False
                                part['text'] = part['text'].replace('{{ShowBase64}}', '')
        
        if should_process_images and self.debug_mode:
            logger.info("图片处理已启用")
            # 这里可以调用图片处理插件
            # messages = await self._call_image_processor(messages)
        
        return messages
    
    async def _call_image_processor(self, messages: List[Dict]) -> List[Dict]:
        """调用图片处理插件"""
        # 这里可以实现图片处理逻辑
        # 例如：将图片转换为base64、调整大小等
        return messages

# 全局预处理器实例
_preprocessor_instance = None

def get_preprocessor() -> AgentPreprocessor:
    """获取全局预处理器实例"""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = AgentPreprocessor()
    return _preprocessor_instance

async def preprocess_messages(messages: List[Dict], model: str = None) -> List[Dict]:
    """便捷函数：预处理消息"""
    preprocessor = get_preprocessor()
    return await preprocessor.preprocess_messages(messages, model) 