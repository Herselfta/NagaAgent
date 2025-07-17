from mcp.server.fastmcp import FastMCP
import sys
import logging
import os
from ctypes import windll
import math
import random
import paho.mqtt.client as mqtt  # MQTT客户端库

# 创建日志记录器，命名为'Calculator'
logger = logging.getLogger('Calculator')

# MQTT配置
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_USERNAME = "admin"
MQTT_PASSWORD = "admin"
MQTT_TOPICS = {
    "led/mcp": "灯",      # 主题对应的中文名称
    "fan/mcp": "风扇"     # 主题对应的中文名称
}

# 初始化MQTT客户端
mqtt_client = mqtt.Client()

# 设置MQTT回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("成功连接到MQTT代理")
        # 订阅内置主题
        for topic in MQTT_TOPICS.keys():
            client.subscribe(topic)
            logger.info(f"已订阅主题: {topic} ({MQTT_TOPICS[topic]})")
    else:
        logger.error(f"连接MQTT代理失败，错误代码: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')  # 明确指定UTF-8解码
        topic_name = MQTT_TOPICS.get(msg.topic, msg.topic)
        logger.info(f"收到来自主题 {msg.topic} ({topic_name}) 的消息: {payload}")
    except UnicodeDecodeError as e:
        logger.error(f"无法解码MQTT消息: {str(e)}")
    except Exception as e:
        logger.error(f"处理MQTT消息时出错: {str(e)}")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# 连接到MQTT代理
try:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    logger.error(f"初始化MQTT客户端失败: {str(e)}")

# 如果是Windows平台，修复控制台的UTF-8编码问题
if sys.platform == 'win32':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        os.system('chcp 65001 > nul')  # 设置控制台代码页为UTF-8
    except Exception as e:
        logger.error(f"设置控制台编码失败: {str(e)}")

# 创建一个FastMCP服务器实例，命名为"Calculator"
mcp = FastMCP("Calculator")

# 使用装饰器@mcp.tool()添加一个计算工具
@mcp.tool()
def calculator(python_expression: str) -> dict:
    """
    数学计算工具，用于计算Python表达式的结果
    参数:
        python_expression: 要计算的Python数学表达式字符串
    返回:
        包含计算结果的字典，格式为{"success": True, "result": 计算结果}
    """
    try:
        result = eval(python_expression)
        logger.info(f"Calculating formula: {python_expression}, result: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"计算错误: {str(e)}")
        return {"success": False, "error": str(e)}

# 添加Windows响铃通知工具
@mcp.tool()
def beep(duration: float = 0.5, frequency: int = 440) -> dict:
    """
    Windows电脑响铃通知工具
    参数:
        duration: 响铃持续时间(秒)，默认0.5秒
        frequency: 响铃频率(Hz)，默认440Hz(A4音高)
    返回:
        包含操作结果的字典，格式为{"success": True, "message": "操作结果"}
    """
    if sys.platform != 'win32':
        return {"success": False, "message": "此功能仅在Windows系统上可用"}
    
    try:
        # 使用Windows API Beep函数
        windll.kernel32.Beep(frequency, int(duration * 1000))
        logger.info(f"Beep sound played - duration: {duration}s, frequency: {frequency}Hz")
        return {"success": True, "message": f"已播放{duration}秒的提示音"}
    except Exception as e:
        logger.error(f"播放提示音失败: {str(e)}")
        return {"success": False, "message": f"播放提示音失败: {str(e)}"}

# 添加MQTT工具
@mcp.tool()
def mqtt_publish(topic: str, message: str, qos: int = 0) -> dict:
    """
    MQTT消息发布工具
    参数:
        topic: 要发布消息的主题，可选主题:
               - led/mcp (灯): 用于控制灯的状态
               - fan/mcp (风扇): 用于控制风扇的状态
        message: 要发布的消息内容(必须是UTF-8可编码的字符串)
        qos: 服务质量等级 (0, 1 或 2)，默认为0
    返回:
        包含操作结果的字典，格式为{"success": True, "message": "操作结果"}
    """
    if topic not in MQTT_TOPICS:
        return {"success": False, "message": f"无效的主题: {topic}，可用主题: {', '.join([f'{k} ({v})' for k, v in MQTT_TOPICS.items()])}"}
    
    try:
        # 确保消息是字符串且可编码为UTF-8
        if not isinstance(message, str):
            message = str(message)
        message.encode('utf-8')  # 测试是否能编码为UTF-8
        
        mqtt_client.publish(topic, message, qos=qos)
        logger.info(f"已发布消息到主题 {topic}: {message}")
        return {"success": True, "message": f"已发布消息到主题 {topic} ({MQTT_TOPICS[topic]})"}
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        logger.error(f"消息编码错误: {str(e)}")
        return {"success": False, "message": f"消息内容必须是有效的UTF-8文本"}
    except Exception as e:
        logger.error(f"发布MQTT消息失败: {str(e)}")
        return {"success": False, "message": f"发布MQTT消息失败: {str(e)}"}

# 如果是直接运行此脚本
if __name__ == "__main__":
    try:
        # 设置环境变量确保UTF-8编码
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # 启动服务器，使用标准输入输出(stdio)作为通信传输方式
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise
    finally:
        # 确保在程序退出时断开MQTT连接
        mqtt_client.loop_stop()
        mqtt_client.disconnect()