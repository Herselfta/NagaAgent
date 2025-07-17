import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 加入项目根目录到模块查找路径
# server.py
import base64
import librosa
from flask import Flask, request, send_file, jsonify
from gevent.pywsgi import WSGIServer
import os
from flask_cors import CORS

from handle_text import prepare_tts_input_with_context
from tts_handler import generate_speech, get_models, get_voices
from utils import require_api_key, AUDIO_FORMAT_MIME_TYPES
from config import config # 统一配置系统

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

API_KEY = config.tts.api_key
PORT = config.tts.port
DEFAULT_VOICE = config.tts.default_voice
DEFAULT_RESPONSE_FORMAT = config.tts.default_format
DEFAULT_SPEED = config.tts.default_speed

REMOVE_FILTER = config.tts.remove_filter
EXPAND_API = config.tts.expand_api


@app.route('/v1/audio/speech', methods=['POST'])
@app.route('/audio/speech', methods=['POST'])  # 增加别名接口
@require_api_key
# 文本转语音主接口
def text_to_speech():
    data = request.json
    if not data or 'input' not in data:
        return jsonify({"error": "请求体缺少 'input' 字段"}), 400

    text = data.get('input')

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    # model = data.get('model', DEFAULT_MODEL)
    voice = data.get('voice', DEFAULT_VOICE)

    response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
    speed = float(data.get('speed', DEFAULT_SPEED))
    
    mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format, "audio/mpeg")

    # 生成音频文件
    output_file_path = generate_speech(text, voice, response_format, speed)

    # 返回音频文件，设置正确的MIME类型
    return send_file(output_file_path, mimetype=mime_type, as_attachment=True, download_name=f"speech.{response_format}")

@app.route('/v1/models', methods=['GET', 'POST'])
@app.route('/models', methods=['GET', 'POST'])
@require_api_key
# 获取模型列表
def list_models():
    return jsonify({"data": get_models()})

@app.route('/v1/voices', methods=['GET', 'POST'])
@app.route('/voices', methods=['GET', 'POST'])
@require_api_key
# 获取语音列表（可按语言过滤）
def list_voices():
    specific_language = None

    data = request.args if request.method == 'GET' else request.json
    if data and ('language' in data or 'locale' in data):
        specific_language = data.get('language') if 'language' in data else data.get('locale')

    return jsonify({"voices": get_voices(specific_language)})

@app.route('/v1/voices/all', methods=['GET', 'POST'])
@app.route('/voices/all', methods=['GET', 'POST'])
@require_api_key
# 获取所有语音
def list_all_voices():
    return jsonify({"voices": get_voices('all')})

"""
Support for ElevenLabs and Azure AI Speech
    (currently in beta)
"""

# http://localhost:5050/elevenlabs/v1/text-to-speech
# http://localhost:5050/elevenlabs/v1/text-to-speech/en-US-AndrewNeural
@app.route('/elevenlabs/v1/text-to-speech/<voice_id>', methods=['POST'])
@require_api_key
# ElevenLabs风格TTS接口
def elevenlabs_tts(voice_id):
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # 解析JSON请求体
    try:
        payload = request.json
        if not payload or 'text' not in payload:
            return jsonify({"error": "请求体缺少 'text' 字段"}), 400
    except Exception as e:
        return jsonify({"error": f"无效的JSON请求体: {str(e)}"}), 400

    text = payload['text']

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    voice = voice_id  # ElevenLabs用URL中的voice_id

    # 使用edge-tts默认设置
    response_format = 'mp3'
    speed = DEFAULT_SPEED  # 可选自定义

    # 生成语音
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS生成失败: {str(e)}"}), 500

    # 返回生成的音频文件
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

# tts.speech.microsoft.com/cognitiveservices/v1
# https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
# http://localhost:5050/azure/cognitiveservices/v1
@app.route('/azure/cognitiveservices/v1', methods=['POST'])
@require_api_key
# Azure风格TTS接口
def azure_tts():
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # 解析SSML请求体
    try:
        ssml_data = request.data.decode('utf-8')
        if not ssml_data:
            return jsonify({"error": "缺少SSML请求体"}), 400

        # 从SSML中提取文本和voice
        from xml.etree import ElementTree as ET
        root = ET.fromstring(ssml_data)
        text = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').text
        voice = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').get('name')
    except Exception as e:
        return jsonify({"error": f"无效的SSML请求体: {str(e)}"}), 400

    # 使用edge-tts默认设置
    response_format = 'mp3'
    speed = DEFAULT_SPEED

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    # 生成语音
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS生成失败: {str(e)}"}), 500

    # 返回生成的音频文件
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")
@app.route('/', methods=['GET'])
# 测试接口
def azure_tts_test():
    return jsonify({"status": "success"})

@app.route('/genVoice', methods=['POST'])
# 兼容旧版的genVoice接口
def genVoice():
    data = request.json
    if not data or 'genText' not in data:
        return jsonify({"error": "请求体缺少 'input' 字段"}), 400

    text = data.get('genText')

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    # model = data.get('model', DEFAULT_MODEL)
    voice = data.get('voice', DEFAULT_VOICE)

    response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
    speed = float(data.get('speed', DEFAULT_SPEED))

    mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format, "audio/mpeg")

    # 生成音频文件
    output_file_path = generate_speech(text, voice, response_format, speed)

    duration = librosa.get_duration(path=output_file_path)

    with open(output_file_path, "rb") as f:
        audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        f.close()
    # os.remove(output_file_path)
    return {"status": "success", "content": {"audio_base64": audio_base64,"duration":duration,"outputName":output_file_path.split('/')[-1]}}

    # Return the file with the correct MIME type
    # return send_file(output_file_path, mimetype=mime_type, as_attachment=True,
    #                  download_name=f"speech.{response_format}")


print(f" Edge TTS (Free Azure TTS) Replacement for OpenAI's TTS API")  # 启动信息
print(f" ")
print(f" * Serving OpenAI Edge TTS")
print(f" * Server running on http://127.0.0.1:{PORT}")
print(f" * TTS Endpoint: http://127.0.0.1:{PORT}/v1/audio/speech")
print(f" ")

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    http_server.serve_forever()
