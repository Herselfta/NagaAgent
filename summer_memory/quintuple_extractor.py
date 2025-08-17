import traceback
import json
import logging
import re
import sys
import os
import time
import asyncio
from typing import List, Tuple
from pydantic import BaseModel
import google.generativeai as genai

# 添加项目根目录到路径，以便导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import config
from openai import OpenAI, AsyncOpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key=config.api.get_api_key(),
    base_url=config.api.get_base_url()
)

async_client = AsyncOpenAI(
    api_key=config.api.get_api_key(),
    base_url=config.api.get_base_url()
)

# 初始化Gemini客户端
gemini_client = None
if config.api.provider == "gemini":
    gemini_api_key = config.api.get_api_key()
    if gemini_api_key:
        try:
            # 配置网络代理
            proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
            if proxy_url:
                genai.configure(api_key=gemini_api_key, transport='rest', client_options={"api_endpoint": "generativelanguage.googleapis.com", "proxy": proxy_url})
                logging.info(f"Quintuple Extractor: Google Gemini 使用代理: {proxy_url}")
            else:
                genai.configure(api_key=gemini_api_key)
            
            gemini_client = genai.GenerativeModel(config.api.model)
            logging.info("Quintuple Extractor: Google Gemini model initialized.")
        except Exception as e:
            logging.warning(f"Quintuple Extractor: Google Gemini model initialization failed: {e}", exc_info=True)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# 定义五元组的Pydantic模型
class Quintuple(BaseModel):
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str


class QuintupleResponse(BaseModel):
    quintuples: List[Quintuple]


async def extract_quintuples_async(text):
    """异步版本的五元组提取"""
    
    # 在异步函数内部初始化客户端，以确保使用正确的事件循环
    async_client = AsyncOpenAI(
        api_key=config.api.get_api_key(),
        base_url=config.api.get_base_url()
    )

    gemini_client = None
    if config.api.provider == "gemini":
        gemini_api_key = config.api.get_api_key()
        if gemini_api_key:
            try:
                # 重新配置是安全的，它会设置模块级的变量
                proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
                if proxy_url:
                    genai.configure(api_key=gemini_api_key, transport='rest', client_options={"api_endpoint": "generativelanguage.googleapis.com", "proxy": proxy_url})
                    logging.info(f"Quintuple Extractor: Google Gemini 使用代理: {proxy_url}")
                else:
                    genai.configure(api_key=gemini_api_key)
                
                gemini_client = genai.GenerativeModel(config.api.model)
            except Exception as e:
                logging.warning(f"Quintuple Extractor: Google Gemini model initialization failed: {e}", exc_info=True)

    timeout = getattr(config.grag, "extraction_timeout", 30)

    # 优先尝试使用Gemini
    if gemini_client:
        try:
            logger.info("尝试使用 Gemini 提取五元组")
            
            prompt = f"""
你是一个专业的中文文本信息抽取专家。你的任务是从给定的中文文本中抽取五元组关系。
五元组格式为：(主体, 主体类型, 动作, 客体, 客体类型)。
请以JSON格式返回一个包含 "quintuples" 键的字典，其值为一个五元组列表。
例如:
{{
  "quintuples": [
    {{
      "subject": "小明",
      "subject_type": "人物",
      "predicate": "踢",
      "object": "足球",
      "object_type": "物品"
    }},
    {{
      "subject": "小明",
      "subject_type": "人物",
      "predicate": "在",
      "object": "公园",
      "object_type": "地点"
    }}
  ]
}}

请从以下文本中提取五元组：
{text}

除了JSON对象，不要输出任何其他内容。
"""
            response = await asyncio.wait_for(
                gemini_client.generate_content_async(prompt),
                timeout=timeout
            )
            
            json_text = response.text.strip()
            # 增加对```json ... ```包裹格式的兼容
            match = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1)

            result = json.loads(json_text)
            quintuples_data = result.get("quintuples", [])
            
            quintuples = []
            for q in quintuples_data:
                quintuples.append((
                    q.get("subject"), q.get("subject_type"),
                    q.get("predicate"), q.get("object"), q.get("object_type")
                ))
            
            if quintuples:
                logger.info(f"Gemini 提取成功，提取到 {len(quintuples)} 个五元组")
                return quintuples
            else:
                # Gemini返回了空结果，也算成功，直接返回
                logger.info("Gemini 提取成功，但未发现五元组")
                return []
        except Exception as e:
            logger.warning(f"Gemini 提取失败，降级到备用模型: {e}", exc_info=True)

    # 降级到结构化输出
    logger.info("降级到 DeepSeek 模型进行五元组提取")
    return await _extract_quintuples_async_structured(text, async_client)



async def _extract_quintuples_async_structured(text, async_client):
    """使用结构化输出的异步五元组提取"""
    system_prompt = """
你是一个专业的中文文本信息抽取专家。你的任务是从给定的中文文本中抽取五元组关系。
五元组格式为：(主体, 主体类型, 动作, 客体, 客体类型)。

类型包括但不限于：人物、地点、组织、物品、概念、时间、事件、活动等。

例如：
输入：小明在公园里踢足球。
应该提取出：
- 主体：小明，类型：人物，动作：踢，客体：足球，类型：物品
- 主体：小明，类型：人物，动作：在，客体：公园，类型：地点

请仔细分析文本，提取所有可以识别出的五元组关系。
"""

    # 重试机制配置
    max_retries = 3
    timeout = getattr(config.grag, "extraction_timeout", 30)

    for attempt in range(max_retries + 1):
        logger.info(f"尝试使用结构化输出提取五元组 (第{attempt + 1}次)")

        try:
            # 尝试使用结构化输出
            completion = await asyncio.wait_for(
                async_client.beta.chat.completions.parse(
                    model=config.api.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"请从以下文本中提取五元组：\n\n{text}"}
                    ],
                    response_format=QuintupleResponse,
                    max_tokens=config.api.max_tokens,
                    temperature=0.3,
                ),
                timeout=timeout + (attempt * 5) # 每次重试增加超时
            )

            # 解析结果
            result = completion.choices[0].message.parsed
            quintuples = []
            
            for q in result.quintuples:
                quintuples.append((
                    q.subject, q.subject_type, 
                    q.predicate, q.object, q.object_type
                ))
            
            logger.info(f"结构化输出成功，提取到 {len(quintuples)} 个五元组")
            return quintuples

        except Exception as e:
            logger.warning(f"结构化输出失败: {str(e)}")
            if attempt == max_retries - 1:  # 最后一次尝试，回退到传统方法
                logger.info("回退到传统JSON解析方法")
                return await _extract_quintuples_async_fallback(text, async_client)
            elif attempt < max_retries:
                await asyncio.sleep(1 + attempt)

    return []


async def _extract_quintuples_async_fallback(text, async_client):
    """传统JSON解析的异步五元组提取（回退方案）"""
    prompt = f"""
从以下中文文本中抽取五元组（主语-主语类型-谓语-宾语-宾语类型）关系，以 JSON 数组格式返回。

类型包括但不限于：人物、地点、组织、物品、概念、时间、事件、活动等。

例如：
输入：小明在公园里踢足球。
输出：[["小明", "人物", "踢", "足球", "物品"], ["小明", "人物", "在", "公园", "地点"]]

请从文本中提取所有可以识别出的五元组：
{text}

除了JSON数据，请不要输出任何其他数据，例如：```、```json、以下是我提取的数据：。
"""

    max_retries = 2
    timeout = getattr(config.grag, "extraction_timeout", 30)

    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                async_client.chat.completions.create(
                    model=config.api.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.api.max_tokens,
                    temperature=0.3,
                ),
                timeout=timeout + (attempt * 5) # 每次重试增加超时
            )
            
            content = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                quintuples = json.loads(content)
                logger.info(f"传统方法成功，提取到 {len(quintuples)} 个五元组")
                return [tuple(t) for t in quintuples if len(t) == 5]
            except json.JSONDecodeError:
                logger.error(f"JSON解析失败，原始内容: {content[:200]}")
                # 尝试直接提取数组
                if '[' in content and ']' in content:
                    start = content.index('[')
                    end = content.rindex(']') + 1
                    quintuples = json.loads(content[start:end])
                    return [tuple(t) for t in quintuples if len(t) == 5]
                raise

        except Exception as e:
            logger.error(f"传统方法提取失败: {str(e)}")
            if attempt < max_retries:
                await asyncio.sleep(1 + attempt)

    return []


def extract_quintuples(text):
    """同步版本的五元组提取"""
    # 首先尝试使用结构化输出
    return _extract_quintuples_structured(text)


def _extract_quintuples_structured(text):
    """使用结构化输出的同步五元组提取"""
    system_prompt = """
你是一个专业的中文文本信息抽取专家。你的任务是从给定的中文文本中抽取五元组关系。
五元组格式为：(主体, 主体类型, 动作, 客体, 客体类型)。

类型包括但不限于：人物、地点、组织、物品、概念、时间、事件、活动等。

例如：
输入：小明在公园里踢足球。
应该提取出：
- 主体：小明，类型：人物，动作：踢，客体：足球，类型：物品
- 主体：小明，类型：人物，动作：在，客体：公园，类型：地点

请仔细分析文本，提取所有可以识别出的五元组关系。
"""

    # 重试机制配置
    max_retries = 3
    timeout = getattr(config.grag, "extraction_timeout", 30)

    for attempt in range(max_retries + 1):
        logger.info(f"尝试使用结构化输出提取五元组 (第{attempt + 1}次)")

        try:
            # 尝试使用结构化输出
            completion = client.beta.chat.completions.parse(
                model=config.api.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请从以下文本中提取五元组：\n\n{text}"}
                ],
                response_format=QuintupleResponse,
                max_tokens=config.api.max_tokens,
                temperature=0.3,
                timeout=timeout + (attempt * 5) # 每次重试增加超时
            )

            # 解析结果
            result = completion.choices[0].message.parsed
            quintuples = []
            
            for q in result.quintuples:
                quintuples.append((
                    q.subject, q.subject_type, 
                    q.predicate, q.object, q.object_type
                ))
            
            logger.info(f"结构化输出成功，提取到 {len(quintuples)} 个五元组")
            return quintuples

        except Exception as e:
            logger.warning(f"结构化输出失败: {str(e)}")
            if attempt == max_retries - 1:  # 最后一次尝试，回退到传统方法
                logger.info("回退到传统JSON解析方法")
                return _extract_quintuples_fallback(text)
            elif attempt < max_retries:
                time.sleep(1 + attempt)

    return []


def _extract_quintuples_fallback(text):
    """传统JSON解析的同步五元组提取（回退方案）"""
    prompt = f"""
从以下中文文本中抽取五元组（主语-主语类型-谓语-宾语-宾语类型）关系，以 JSON 数组格式返回。

类型包括但不限于：人物、地点、组织、物品、概念、时间、事件、活动等。

例如：
输入：小明在公园里踢足球。
输出：[["小明", "人物", "踢", "足球", "物品"], ["小明", "人物", "在", "公园", "地点"]]

请从文本中提取所有可以识别出的五元组：
{text}

除了JSON数据，请不要输出任何其他数据，例如：```、```json、以下是我提取的数据：。
"""

    max_retries = 2
    timeout = getattr(config.grag, "extraction_timeout", 30)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.api.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.api.max_tokens,
                temperature=0.5,
                timeout=timeout + (attempt * 5) # 每次重试增加超时
            )

            content = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                quintuples = json.loads(content)
                logger.info(f"传统方法成功，提取到 {len(quintuples)} 个五元组")
                return [tuple(t) for t in quintuples if len(t) == 5]
            except json.JSONDecodeError:
                logger.error(f"JSON解析失败，原始内容: {content[:200]}")
                # 尝试直接提取数组
                if '[' in content and ']' in content:
                    start = content.index('[')
                    end = content.rindex(']') + 1
                    quintuples = json.loads(content[start:end])
                    return [tuple(t) for t in quintuples if len(t) == 5]
                raise

        except Exception as e:
            logger.error(f"传统方法提取失败: {str(e)}")
            if attempt < max_retries:
                time.sleep(1 + attempt)

    return []