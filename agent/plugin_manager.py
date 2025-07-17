# agent/plugin_manager.py
# 插件管理器
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger("AgentPluginManager")

class AgentPluginManager:
    """代理插件管理器"""
    
    def __init__(self, project_base_path: str = None):
        self.project_base_path = project_base_path or os.getcwd()
        self.plugins = {}
        self.message_preprocessors = {}
        self.static_placeholder_values = {}
        self.debug_mode = os.getenv("DEBUG", "False").lower() == "true"
        
        self.plugin_dir = Path(self.project_base_path) / "agent" / "plugins"
        self.plugin_dir.mkdir(exist_ok=True)
    
    async def load_plugins(self):
        """加载所有插件"""
        if self.debug_mode:
            logger.info("开始加载插件...")
        
        self.plugins.clear()
        self.message_preprocessors.clear()
        self.static_placeholder_values.clear()
        
        await self._scan_plugin_directory()
        await self._initialize_plugins()
        
        if self.debug_mode:
            logger.info(f"插件加载完成，共加载 {len(self.plugins)} 个插件")
    
    async def _scan_plugin_directory(self):
        """扫描插件目录"""
        if not self.plugin_dir.exists():
            return
        
        for plugin_folder in self.plugin_dir.iterdir():
            if plugin_folder.is_dir():
                await self._load_plugin_from_folder(plugin_folder)
    
    async def _load_plugin_from_folder(self, plugin_folder: Path):
        """从文件夹加载插件"""
        manifest_path = plugin_folder / "plugin-manifest.json"
        
        if not manifest_path.exists():
            return
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            if not all(key in manifest for key in ['name', 'pluginType', 'entryPoint']):
                return
            
            manifest['basePath'] = str(plugin_folder)
            
            config_path = plugin_folder / "config.env"
            if config_path.exists():
                manifest['pluginSpecificEnvConfig'] = await self._load_env_config(config_path)
            
            self.plugins[manifest['name']] = manifest
            
            if self.debug_mode:
                logger.info(f"加载插件: {manifest.get('displayName', manifest['name'])}")
        
        except Exception as e:
            logger.error(f"加载插件失败 {plugin_folder}: {e}")
    
    async def _load_env_config(self, config_path: Path) -> Dict[str, str]:
        """加载环境配置文件"""
        config = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}")
        
        return config
    
    async def _initialize_plugins(self):
        """初始化插件"""
        for plugin_name, manifest in self.plugins.items():
            try:
                if manifest['pluginType'] == 'messagePreprocessor':
                    await self._initialize_message_preprocessor(plugin_name, manifest)
                elif manifest['pluginType'] == 'static':
                    await self._initialize_static_plugin(plugin_name, manifest)
            except Exception as e:
                logger.error(f"初始化插件失败 {plugin_name}: {e}")
    
    async def _initialize_message_preprocessor(self, plugin_name: str, manifest: Dict):
        """初始化消息预处理器"""
        if 'entryPoint' not in manifest or 'script' not in manifest['entryPoint']:
            return
        
        script_path = Path(manifest['basePath']) / manifest['entryPoint']['script']
        
        if not script_path.exists():
            return
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(plugin_name, script_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            config = self._get_plugin_config(manifest)
            if hasattr(plugin_module, 'initialize'):
                await plugin_module.initialize(config)
            
            self.message_preprocessors[plugin_name] = plugin_module
            
            if self.debug_mode:
                logger.info(f"初始化消息预处理器: {plugin_name}")
        
        except Exception as e:
            logger.error(f"初始化消息预处理器失败 {plugin_name}: {e}")
    
    async def _initialize_static_plugin(self, plugin_name: str, manifest: Dict):
        """初始化静态插件"""
        if 'entryPoint' not in manifest or 'command' not in manifest['entryPoint']:
            return
        
        try:
            config = self._get_plugin_config(manifest)
            result = await self._execute_static_plugin_command(manifest, config)
            
            if 'capabilities' in manifest and 'systemPromptPlaceholders' in manifest['capabilities']:
                for placeholder_info in manifest['capabilities']['systemPromptPlaceholders']:
                    placeholder = placeholder_info['placeholder']
                    self.static_placeholder_values[placeholder] = result
            
            if self.debug_mode:
                logger.info(f"初始化静态插件: {plugin_name}")
        
        except Exception as e:
            logger.error(f"初始化静态插件失败 {plugin_name}: {e}")
    
    async def _execute_static_plugin_command(self, manifest: Dict, config: Dict) -> str:
        """执行静态插件命令"""
        command = manifest['entryPoint']['command']
        base_path = manifest['basePath']
        
        env = os.environ.copy()
        for key, value in config.items():
            if value is not None:
                env[key] = str(value)
        
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=base_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"静态插件执行失败: {stderr.decode()}")
        
        return stdout.decode().strip()
    
    def _get_plugin_config(self, manifest: Dict) -> Dict:
        """获取插件配置"""
        config = {}
        
        if 'configSchema' in manifest:
            for key, expected_type in manifest['configSchema'].items():
                value = os.getenv(key)
                if value is not None:
                    if expected_type == 'integer':
                        try:
                            config[key] = int(value)
                        except ValueError:
                            config[key] = None
                    elif expected_type == 'boolean':
                        config[key] = value.lower() == 'true'
                    else:
                        config[key] = value
        
        if 'pluginSpecificEnvConfig' in manifest:
            for key, value in manifest['pluginSpecificEnvConfig'].items():
                config[key] = value
        
        config['DebugMode'] = self.debug_mode
        
        return config
    
    async def execute_message_preprocessor(self, plugin_name: str, messages: List[Dict]) -> List[Dict]:
        """执行消息预处理器"""
        if plugin_name not in self.message_preprocessors:
            logger.error(f"消息预处理器不存在: {plugin_name}")
            return messages
        
        try:
            plugin_module = self.message_preprocessors[plugin_name]
            if hasattr(plugin_module, 'processMessages'):
                config = self._get_plugin_config(self.plugins[plugin_name])
                return await plugin_module.processMessages(messages, config)
            else:
                logger.error(f"插件缺少processMessages方法: {plugin_name}")
                return messages
        except Exception as e:
            logger.error(f"执行消息预处理器失败 {plugin_name}: {e}")
            return messages
    
    def get_placeholder_value(self, placeholder: str) -> str:
        """获取占位符值"""
        return self.static_placeholder_values.get(placeholder, f"[Placeholder {placeholder} not found]")
    
    def get_plugin(self, plugin_name: str) -> Optional[Dict]:
        """获取插件信息"""
        return self.plugins.get(plugin_name)
    
    def get_resolved_plugin_config_value(self, plugin_name: str, config_key: str) -> Any:
        """获取解析后的插件配置值"""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return None
        
        config = self._get_plugin_config(plugin)
        return config.get(config_key)
    
    async def shutdown_all_plugins(self):
        """关闭所有插件"""
        if self.debug_mode:
            logger.info("开始关闭所有插件...")
        
        for plugin_name, plugin_module in self.message_preprocessors.items():
            try:
                if hasattr(plugin_module, 'shutdown'):
                    await plugin_module.shutdown()
                    if self.debug_mode:
                        logger.info(f"关闭插件: {plugin_name}")
            except Exception as e:
                logger.error(f"关闭插件失败 {plugin_name}: {e}")
        
        if self.debug_mode:
            logger.info("所有插件已关闭")

# 全局插件管理器实例
_plugin_manager_instance = None

def get_plugin_manager() -> AgentPluginManager:
    """获取全局插件管理器实例"""
    global _plugin_manager_instance
    if _plugin_manager_instance is None:
        _plugin_manager_instance = AgentPluginManager()
    return _plugin_manager_instance

async def load_plugins():
    """便捷函数：加载插件"""
    plugin_manager = get_plugin_manager()
    await plugin_manager.load_plugins()

async def execute_message_preprocessor(plugin_name: str, messages: List[Dict]) -> List[Dict]:
    """便捷函数：执行消息预处理器"""
    plugin_manager = get_plugin_manager()
    return await plugin_manager.execute_message_preprocessor(plugin_name, messages) 