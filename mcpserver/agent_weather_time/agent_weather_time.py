# agent_weather_time.py # 天气和时间查询Agent
import json # 导入json模块
import aiohttp # 异步HTTP请求
from agents import Agent, ComputerTool # 导入Agent和工具基类
from config import DEBUG # 导入全局DEBUG配置
import requests # 用于同步获取IP和城市
import re # 用于正则解析
from datetime import datetime, timedelta # 用于日期处理
IPIP_URL = "https://myip.ipip.net/" # 统一配置
from .city_code_map import CITY_CODE_MAP # 导入城市编码表

class WeatherTimeTool:
    """天气和时间工具类"""
    def __init__(self):
        self._ip_info = None # 缓存IP信息
        self._local_ip = None # 本地IP
        self._local_city = None # 本地城市
        self._get_local_ip_and_city() # 初始化时获取本地IP和城市
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._preload_ip_info())
            else:
                self._ip_info = None # 不再异步获取IP
        except Exception:
            self._ip_info = None

    def _get_local_ip_and_city(self):
        """同步获取本地IP和城市"""
        try:
            resp = requests.get(IPIP_URL, timeout=5)
            resp.encoding = 'utf-8'
            html = resp.text
            match = re.search(r"当前 IP：([\d\.]+)\s+来自于：(.+?)\s{2,}", html)
            if match:
                self._local_ip = match.group(1)
                self._local_city = match.group(2)
            else:
                self._local_ip = None
                self._local_city = None
        except Exception as e:
            self._local_ip = None
            self._local_city = None

    async def _preload_ip_info(self):
        pass # 兼容保留，不再异步获取IP

    async def get_weather(self, province, city):
        """调用高德地图天气接口，返回实况天气+未来3天预报"""  # 右侧注释
        # 从config.json获取API key
        import os
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        api_key = config.get('weather_api_key')
        
        url = f'https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={api_key}&extensions=all'
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json(content_type=None)
                # 替换reporttime为系统当前时间 # 右侧注释
                if data.get('lives') and isinstance(data['lives'], list):
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    for live in data['lives']:
                        if isinstance(live, dict):
                            live['reporttime'] = current_time
                return data

    async def handle(self, action=None, ip=None, city=None, query=None, format=None, **kwargs):
        """统一处理入口，支持LLM传入city参数或自动识别本地城市"""  # 右侧注释
        # 优先使用LLM传入的city参数，如果没有则使用本地城市 # 右侧注释
        if city and city.strip():
            # LLM传入了city参数，直接使用
            city_str = city.strip()
            province, city_name = city_str, city_str
            # 尝试解析省市格式
            match = re.match(r"^([\u4e00-\u9fa5]+) ([\u4e00-\u9fa5]+)$", city_str)
            if match:
                province = match.group(1)
                city_name = match.group(2)
            else:
                parts = city_str.split()
                if len(parts) >= 2:
                    province = parts[-2]
                    city_name = parts[-1]
                else:
                    province = city_str
                    city_name = city_str
        else:
            # LLM没有传入city参数，使用本地城市作为默认值
            city_str = getattr(self, '_local_city', '') or ''
            if city_str.startswith('中国'):
                city_str = city_str[2:].strip()
            province, city_name = city_str, city_str
            if city_str:
                match = re.match(r"^([\u4e00-\u9fa5]+) ([\u4e00-\u9fa5]+)$", city_str)
                if match:
                    province = match.group(1)
                    city_name = match.group(2)
                else:
                    parts = city_str.split()
                    if len(parts) >= 2:
                        province = parts[-2]
                        city_name = parts[-1]
                    else:
                        province = city_str
                        city_name = city_str
        
        # 查表获取编码 # 右侧注释
        city_code = CITY_CODE_MAP.get(city_name) or CITY_CODE_MAP.get(province)
        if not city_code:
            return {'status': 'error', 'message': f'未找到城市编码: {city_name}'}
        
        # 今日天气查询，只返回今日天气数据 # 右侧注释
        if action in ['today_weather', 'current_weather', 'today']:
            weather = await self.get_weather(province, city_code)
            
            # 提取今日天气数据
            today_data = None
            if weather.get('forecasts') and isinstance(weather['forecasts'], list):
                for forecast in weather['forecasts']:
                    if forecast.get('casts') and isinstance(forecast['casts'], list):
                        # 获取今天的日期
                        today = datetime.now().strftime('%Y-%m-%d')
                        # 如果API返回的数据中没有今天的日期，则取第一个（通常是今天）
                        target_cast = None
                        for cast in forecast['casts']:
                            if cast.get('date') == today:
                                target_cast = cast
                                break
                        
                        # 如果没找到今天的，取第一个作为今天的数据
                        if not target_cast and forecast['casts']:
                            target_cast = forecast['casts'][0]
                        
                        if target_cast:
                            today_data = {
                                'city': forecast.get('city', city_name),
                                'province': forecast.get('province', province),
                                'reporttime': forecast.get('reporttime', ''),
                                'today_weather': target_cast
                            }
                        break
            
            if today_data:
                return {
                    'status': 'ok',
                    'message': f'今日天气数据 - 查询城市: {city_name}',
                    'data': today_data
                }
            else:
                return {
                    'status': 'error',
                    'message': f'未找到今日天气数据 - 查询城市: {city_name}',
                    'data': {}
                }
        
        # 未来天气查询，返回未来3天预报数据 # 右侧注释
        elif action in ['forecast_weather', 'future_weather', 'forecast', 'weather_forecast']:
            weather = await self.get_weather(province, city_code)
            
            # 提取未来天气数据（排除今天）
            future_data = None
            if weather.get('forecasts') and isinstance(weather['forecasts'], list):
                for forecast in weather['forecasts']:
                    if forecast.get('casts') and isinstance(forecast['casts'], list):
                        # 获取今天的日期
                        today = datetime.now().strftime('%Y-%m-%d')
                        future_casts = []
                        
                        for cast in forecast['casts']:
                            if cast.get('date') != today:  # 排除今天
                                future_casts.append(cast)
                        
                        if future_casts:
                            future_data = {
                                'city': forecast.get('city', city_name),
                                'province': forecast.get('province', province),
                                'reporttime': forecast.get('reporttime', ''),
                                'future_forecast': future_casts
                            }
                        break
            
            if future_data:
                return {
                    'status': 'ok',
                    'message': f'未来天气预报数据 - 查询城市: {city_name}',
                    'data': future_data
                }
            else:
                return {
                    'status': 'error',
                    'message': f'未找到未来天气数据 - 查询城市: {city_name}',
                    'data': {}
                }
        
        # 时间查询
        elif action in ['time', 'get_time', 'current_time']:
            # 时间查询
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return {
                'status': 'ok',
                'message': '当前系统时间',
                'data': {
                    'time': current_time,
                    'city': city_name,
                    'province': province
                }
            }
        else:
            return {'status': 'error', 'message': f'未知操作: {action}'}

class WeatherTimeAgent(Agent):
    """天气和时间Agent"""
    def __init__(self):
        self._tool = WeatherTimeTool() # 预加载
        super().__init__(
            name="WeatherTime Agent", # Agent名称
            instructions="天气和时间智能体", # 角色描述
            tools=[ComputerTool(self._tool)], # 注入工具
            model="weather-time-use-preview" # 使用统一模型
        )
        import sys
        ip_str = getattr(self._tool, '_local_ip', '未获取到IP')  # 直接用本地IP
        city_str = getattr(self._tool, '_local_city', '未知城市') # 获取本地城市
        sys.stderr.write(f'✅ WeatherTimeAgent初始化完成，登陆地址：{city_str}\n')

    async def handle_handoff(self, task: dict) -> str:
        try:
            # 只认tool_name参数
            action = task.get("tool_name")
            if not action:
                return json.dumps({"status": "error", "message": "缺少tool_name参数", "data": {}}, ensure_ascii=False)
            ip = task.get("ip")
            city = task.get("city")
            query = task.get("query")
            format = task.get("format")
            result = await self._tool.handle(action, ip, city, query, format)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e), "data": {}}, ensure_ascii=False)