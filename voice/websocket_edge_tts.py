import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 加入项目根目录到模块查找路径
import asyncio
import json
import re
# from dotenv import load_dotenv  # 移除，使用主系统配置
import base64
import tempfile
import librosa
import soundfile as sf
from handle_text import prepare_tts_input_with_context
from tts_handler import generate_speech, get_models, get_voices
from utils import getenv_bool, require_api_key, AUDIO_FORMAT_MIME_TYPES
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading
import logging
import uvicorn
import config  # 使用主系统配置


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tts_service.log")
    ]
)
logger = logging.getLogger("tts_service")

# 资源控制
MAX_CONCURRENT_TASKS = 2  # 固定低并发，避免竞争条件
RESULT_QUEUE_LOCK = threading.Lock()  # 结果队列锁
MEMORY_THRESHOLD = 70  # 内存使用率阈值（百分比）
MAX_SENTENCES_PER_BATCH = 3  # 每批处理的最大句子数
EXECUTOR = None  # 全局线程池，确保线程池生命周期管理正确

# 添加超时处理
REQUEST_TIMEOUT = 60  # 请求处理超时（秒）

# 断句正则表达式
SENTENCE_END_PUNCTUATIONS = r"[。？！；\.\?\!\;]"

# load_dotenv()  # 移除，使用主系统配置

API_KEY = config.tts.api_key # 统一配置
PORT = config.tts.port
DEFAULT_VOICE = config.tts.default_voice
DEFAULT_RESPONSE_FORMAT = config.tts.default_format
DEFAULT_SPEED = config.tts.default_speed

REMOVE_FILTER = getenv_bool('REMOVE_FILTER', False)
EXPAND_API = getenv_bool('EXPAND_API', True)

app = FastAPI()

origins = [
    "*",  # 输入自己前端项目的地址
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 音频文件处理函数
# 安全获取音频文件时长，避免使用librosa直接读取整个文件

def get_audio_duration(file_path):
    """安全获取音频文件时长，避免使用librosa直接读取整个文件"""
    try:
        info = sf.info(file_path)  # 用soundfile读取音频信息
        return info.duration
    except Exception as e:
        logger.error(f"无法获取音频时长: {e}")
        try:
            duration = librosa.get_duration(path=file_path)  # 兜底用librosa
            return duration
        except Exception as e2:
            logger.error(f"librosa也无法获取音频时长: {e2}")
            return 1.0  # 返回默认值

@app.websocket("/genVoice")
async def genVoice(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket连接已接受。")
    received_buffer = ""
    seq_counter = 1
    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"收到文本: {data}")
                received_buffer += data  # 文本加入缓冲区
                raw_sentences = re.split(SENTENCE_END_PUNCTUATIONS, received_buffer)  # 按标点分句
                logger.info(f"分割为{len(raw_sentences)}个原始句子")
                processed_sentences = []
                if raw_sentences and not raw_sentences[-1]:
                    raw_sentences.pop()
                    received_buffer = ""
                elif raw_sentences:
                    received_buffer = raw_sentences.pop()
                else:
                    received_buffer = ""
                # 处理短句合并
                temp_sentence_buffer = ""
                for i, sentence in enumerate(raw_sentences):
                    sentence_stripped = sentence.strip()
                    if not sentence_stripped:
                        continue
                    has_quotes = re.search(r'["\'\'""\']', sentence_stripped)  # 特殊处理引号
                    if len(sentence_stripped) <= 5 and not has_quotes:
                        temp_sentence_buffer += sentence_stripped
                    else:
                        if temp_sentence_buffer:
                            processed_sentences.append(temp_sentence_buffer + sentence_stripped)
                            temp_sentence_buffer = ""
                        else:
                            processed_sentences.append(sentence_stripped)
                if temp_sentence_buffer:
                    processed_sentences.append(temp_sentence_buffer)
                # 处理每个完整句子
                for sentence in processed_sentences:
                    if sentence:
                        logger.info(f"处理句子: '{sentence}'")
                        loop = asyncio.get_running_loop()
                        try:
                            result = await loop.run_in_executor(None, text2voice, sentence, seq_counter)  # 线程池TTS
                            await websocket.send_json(result)  # 发送结果
                            seq_counter += 1
                            logger.info(f"已发送音频数据: '{sentence}'")
                        except Exception as e:
                            logger.error(f"生成或发送音频出错: '{sentence}': {e}")
                            try:
                                await websocket.send_text(f"Error: {e}")
                            except:
                                logger.error("无法发送错误信息，客户端可能已断开")
                                break
            except WebSocketDisconnect:
                logger.info("客户端断开连接")
                break
    except WebSocketDisconnect:
        logger.info("WebSocket已断开")
    except Exception as e:
        logger.error(f"WebSocket处理异常: {e}")
    finally:
        logger.info("WebSocket连接已关闭")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.websocket("/")
async def ws_test(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"message": "Hello, WebSocket!"})

def text2voice(text, seq):
    """文本转语音函数，使用更安全的内存管理方式"""
    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)
    voice = DEFAULT_VOICE
    response_format = DEFAULT_RESPONSE_FORMAT
    speed = float(DEFAULT_SPEED)
    output_file_path = generate_speech(text, voice, response_format, speed)
    duration = get_audio_duration(output_file_path)
    with open(output_file_path, "rb") as f:
        audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    result = {
        "seq": seq,
        "text": text,
        "wav_base64": audio_base64,
        "duration": f"{duration:.2f}"
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)