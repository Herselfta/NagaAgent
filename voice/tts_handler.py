import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 加入项目根目录到模块查找路径
import edge_tts
import asyncio
import tempfile
import subprocess
import os
from pathlib import Path
import config # 顶部引入

# 语言默认值（环境变量）
DEFAULT_LANGUAGE = config.tts.default_language # 统一配置

# OpenAI语音名与edge-tts语音名映射
voice_mapping = {
    'alloy': 'en-US-AvaNeural',
    'echo': 'en-US-AndrewNeural',
    'fable': 'en-GB-SoniaNeural',
    'onyx': 'en-US-EricNeural',
    'nova': 'en-US-SteffanNeural',
    'shimmer': 'en-US-EmmaNeural'
}

def is_ffmpeg_installed():
    """检查FFmpeg是否已安装并可用。"""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def _generate_audio(text, voice, response_format, speed):
    """生成TTS音频，并可选转换格式。"""
    # 判断是否为OpenAI兼容语音名，否则直接用edge-tts语音名
    edge_tts_voice = voice_mapping.get(voice, voice)  # 优先用映射，否则原样

    # 先生成mp3格式的临时文件
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    # 转换速度为SSML rate格式
    try:
        speed_rate = speed_to_rate(speed)  # 转换为+X%或-X%
    except Exception as e:
        print(f"速度转换出错: {e}，默认+0%。")
        speed_rate = "+0%"

    # 生成MP3文件
    communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
    asyncio.run(communicator.save(temp_output_file.name))

    # 如果请求mp3格式，直接返回
    if response_format == "mp3":
        return temp_output_file.name

    # 检查FFmpeg
    if not is_ffmpeg_installed():
        print("FFmpeg不可用，返回原始mp3文件。"); return temp_output_file.name

    # 新建转换后输出文件
    converted_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")

    # 构建FFmpeg命令
    ffmpeg_command = [
        "ffmpeg",
        "-i", temp_output_file.name,  # 输入文件
        "-c:a", {
            "aac": "aac",
            "mp3": "libmp3lame",
            "wav": "pcm_s16le",
            "opus": "libopus",
            "flac": "flac"
        }.get(response_format, "aac"),  # 默认aac
        "-b:a", "192k" if response_format != "wav" else None,  # wav不需要码率
        "-f", {
            "aac": "mp4",  # AAC用MP4容器
            "mp3": "mp3",
            "wav": "wav",
            "opus": "ogg",
            "flac": "flac"
        }.get(response_format, response_format),
        "-y",  # 覆盖输出
        converted_output_file.name
    ]

    try:
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg音频转换出错: {e}")

    # 删除原始临时文件
    Path(temp_output_file.name).unlink(missing_ok=True)

    return converted_output_file.name

def generate_speech(text, voice, response_format, speed=1.0):
    return asyncio.run(_generate_audio(text, voice, response_format, speed))

def get_models():
    return [
        {"id": "tts-1", "name": "Text-to-speech v1"},
        {"id": "tts-1-hd", "name": "Text-to-speech v1 HD"}
    ]

def _get_voices(language=None):
    # 列出所有语音，可按语言过滤
    all_voices = asyncio.run(edge_tts.list_voices())
    language = language or DEFAULT_LANGUAGE
    filtered_voices = [
        {"name": v['ShortName'], "gender": v['Gender'], "language": v['Locale']}
        for v in all_voices if language == 'all' or language is None or v['Locale'] == language
    ]
    return filtered_voices

def get_voices(language=None):
    return asyncio.run(_get_voices(language))

def speed_to_rate(speed: float) -> str:
    """
    将倍速值转换为edge-tts的rate格式。
    参数：speed (float): 倍速（如1.5为+50%，0.5为-50%）
    返回：str: 速率字符串（如+50%或-50%）
    """
    if speed < 0 or speed > 2:
        raise ValueError("速度必须在0到2之间（含）")
    percentage_change = (speed - 1) * 100
    return f"{percentage_change:+.0f}%"
