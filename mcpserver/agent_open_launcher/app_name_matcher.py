# app_name_matcher.py # 智能应用名适配与动态学习
import os
import json
import asyncio
from typing import List, Dict, Optional

# 别名库文件路径
ALIAS_PATH = os.path.join(os.path.dirname(__file__), "app_alias.json")

def load_alias_dict():
    """加载别名库"""
    if os.path.exists(ALIAS_PATH):
        with open(ALIAS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_alias_dict(alias_dict):
    """保存别名库"""
    with open(ALIAS_PATH, "w", encoding="utf-8") as f:
        json.dump(alias_dict, f, ensure_ascii=False, indent=2)

def update_alias(user_input, real_name):
    """动态学习：记录用户说法与真实应用名的映射"""
    alias_dict = load_alias_dict()
    if user_input and real_name and user_input != real_name:
        alias_dict[user_input] = real_name
        save_alias_dict(alias_dict)

async def llm_select_app(user_input: str, app_list: List[Dict]) -> Optional[Dict]:
    """使用LLM智能选择应用"""
    try:
        from openai import AsyncOpenAI
        from config import config
        
        # 创建LLM客户端
        client = AsyncOpenAI(
            api_key=config.api.api_key,
            base_url=config.api.base_url.rstrip('/') + '/'
        )
        
        # 构建应用列表字符串
        app_names = [app['name'] for app in app_list]
        app_list_str = "\n".join([f"- {name}" for name in app_names])
        
        # 构建提示词
        prompt = f"""你是一个应用选择助手。用户想要打开一个应用，请从以下应用列表中选择最合适的一个：

可用应用列表：
{app_list_str}

用户需求：{user_input}

请严格按照以下格式回复，只返回应用名称，不要包含任何其他内容：
应用名称

注意：
1. 只返回一个最匹配的应用名称
2. 如果找不到合适的应用，返回"未找到"
3. 不要添加任何解释或标点符号
4. 应用名称必须完全匹配列表中的名称"""

        # 调用LLM
        response = await client.chat.completions.create(
            model=config.api.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 低温度确保一致性
            max_tokens=50
        )
        
        selected_app_name = response.choices[0].message.content.strip()
        
        # 验证返回的应用是否在列表中
        for app in app_list:
            if app['name'] == selected_app_name:
                return app
        
        return None
        
    except Exception as e:
        print(f"LLM选择应用失败: {e}")
        return None

async def find_best_app_with_llm(user_input: str, app_list: List[Dict]) -> Optional[Dict]:
    """使用LLM智能查找应用"""
    # 1. 先查别名库
    alias_dict = load_alias_dict()
    if user_input in alias_dict:
        real_name = alias_dict[user_input]
        for app in app_list:
            if app['name'] == real_name:
                return app
    
    # 2. 精确匹配
    for app in app_list:
        if user_input.lower() == app['name'].lower():
            return app
    
    # 3. 使用LLM智能选择
    print(f"使用LLM智能选择应用: {user_input}")
    llm_result = await llm_select_app(user_input, app_list)
    if llm_result:
        # 记录LLM的选择到别名库
        update_alias(user_input, llm_result['name'])
        return llm_result
    
    return None
