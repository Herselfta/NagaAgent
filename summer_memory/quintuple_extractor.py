import requests
import json
import logging
import re
import sys
import os

# 添加项目根目录到路径，以便导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import config
API_KEY = config.api.api_key
API_URL = f"{config.api.base_url.rstrip('/')}/chat/completions"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_quintuples(text):
    prompt = f"""
从以下中文文本中抽取五元组（主语-主语类型-谓语-宾语-宾语类型）关系，以 (主体, 主体类型, 动作, 客体, 客体类型) 的格式返回一个 JSON 数组。

类型包括但不限于：人物、地点、组织、物品、概念、时间、事件、活动等。

例如：
输入：小明在公园里踢足球。
输出：[["小明", "人物", "踢", "足球", "物品"], ["小明", "人物", "在", "公园", "地点"]]

请从文本中提取所有可以识别出的五元组：
{text}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": config.api.model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.5
    }

    try:
        response = requests.post(API_URL, headers=headers, json=body, timeout=20)

        print("状态码:", response.status_code)
        print("响应内容:", response.text)

        response.raise_for_status()
        content_json = response.json()

        content = content_json['choices'][0]['message']['content']
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = content.strip()  

        quintuples = json.loads(json_str)
        logger.info(f"提取到的五元组: {quintuples}")
        return [tuple(t) for t in quintuples if len(t) == 5]

    except Exception as e:
        logger.error(f"调用 DeepSeek API 抽取五元组失败: {e}")
        return []