# 导入必要的模块
from mcp.server.fastmcp import FastMCP
import sys
import logging
import json
import paho.mqtt.client as mqtt  # MQTT客户端库
import threading  # 用于创建独立的MQTT客户端线程

# 创建日志记录器，命名为'SmartController'
logger = logging.getLogger('SmartController')

# 如果是Windows平台，修复控制台的UTF-8编码问题
if sys.platform == 'win32':
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

# MQTT配置
MQTT_BROKER = "iot-06z00b3k4pfqzty.mqtt.iothub.aliyuncs.com"
MQTT_PORT = 1883
MQTT_TOPIC = "/k28x9wMnNss/E6UtPoMxRw7iDKzQuBVE/user/wechat_3"
MQTT_CLIENT_ID = "k28x9wMnNss.E6UtPoMxRw7iDKzQuBVE|securemode=2,signmethod=hmacsha256,timestamp=1751374746832|"

class SmartLightController:
    def __init__(self):
        """初始化智能控制器"""
        self.mqtt_client = None
        self.connected = False
        self.lock = threading.Lock()
        
        # 初始化MQTT客户端
        self._init_mqtt_client()
        
        # 启动MQTT客户端线程
        self.mqtt_thread = threading.Thread(target=self._run_mqtt_client, daemon=True)
        self.mqtt_thread.start()
    
    def _init_mqtt_client(self):
        """初始化MQTT客户端"""
        self.mqtt_client = mqtt.Client(
            client_id=MQTT_CLIENT_ID,
            clean_session=True
        )
        
        # 设置回调函数
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        # 设置遗嘱消息
        self.mqtt_client.will_set(
            topic=f"{MQTT_TOPIC}/status",
            payload="offline",
            qos=1,
            retain=True
        )
    
    def _run_mqtt_client(self):
        """在独立线程中运行MQTT客户端"""
        try:
            # 连接到MQTT代理
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            
            # 保持连接
            self.mqtt_client.loop_forever(retry_first_connection=True)
        except Exception as e:
            logger.error(f"MQTT客户端线程异常: {str(e)}")
            self.connected = False
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            logger.info("成功连接到MQTT代理")
            self.connected = True
            
            # 订阅主题
            client.subscribe(MQTT_TOPIC, qos=1)
            
            # 发布状态消息
            client.publish(
                topic=f"{MQTT_TOPIC}/status",
                payload="online",
                qos=1,
                retain=True
            )
        else:
            logger.error(f"MQTT连接失败，错误代码: {rc}")
            self.connected = False
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT断开连接回调"""
        logger.warning(f"MQTT连接断开，错误代码: {rc}")
        self.connected = False
        
        # 尝试重新连接
        try:
            client.reconnect()
        except Exception as e:
            logger.error(f"MQTT重连失败: {str(e)}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT消息到达回调"""
        try:
            payload = msg.payload.decode('utf-8')
            logger.info(f"收到MQTT消息: topic={msg.topic}, payload={payload}")
            
            # 这里可以添加对消息的处理逻辑
            # 例如控制台灯的状态
            
        except Exception as e:
            logger.error(f"处理MQTT消息时出错: {str(e)}")
    
    def send_command(self, command: str, value=None):
        """
        发送控制命令到MQTT主题
        
        参数:
            command: 控制命令 (如 "switch", "brightness", "color")
            value: 命令值 (可选)
        """
        with self.lock:
            if not self.connected:
                logger.error("无法发送命令: MQTT未连接")
                return False
            
            try:
                payload = {
                    "command": command,
                    "value": value
                }
                
                # 发布消息
                self.mqtt_client.publish(
                    topic=MQTT_TOPIC,
                    payload=json.dumps(payload),
                    qos=1,
                    retain=False
                )
                
                logger.info(f"已发送命令: {payload}")
                return True
            except Exception as e:
                logger.error(f"发送MQTT命令失败: {str(e)}")
                return False

# 创建全局控制器实例
light_controller = SmartLightController()

# 创建FastMCP服务器实例
mcp = FastMCP("SmartLightController")

@mcp.tool()
def control_light(command: str, value=None) -> dict:
    """
    物联网台灯控制工具
    
    参数:
        command: 控制命令 (switch, brightness, color)
        value: 命令值 (可选)
    
    返回:
        包含操作结果的字典
    
    示例:
        control_light("switch", "on") -> 打开台灯
        control_light("brightness", 80) -> 设置亮度为80%
        control_light("color", "#FF0000") -> 设置颜色为红色
    """
    try:
        # 调用控制器发送命令
        success = light_controller.send_command(command, value)
        
        if success:
            return {
                "success": True,
                "message": f"命令已发送: {command} {value if value else ''}",
                "command": command,
                "value": value
            }
        else:
            return {
                "success": False,
                "message": "无法发送命令: MQTT连接失败",
                "command": command,
                "value": value
            }
    except Exception as e:
        logger.error(f"控制台灯时出错: {str(e)}")
        return {
            "success": False,
            "message": f"控制台灯时出错: {str(e)}",
            "command": command,
            "value": value
        }

@mcp.tool()
def get_light_status() -> dict:
    """
    获取台灯当前状态
    
    返回:
        包含台灯状态的字典
    """
    # 注意: 这是一个模拟实现，实际应用中可能需要从MQTT订阅的消息中获取状态
    return {
        "success": True,
        "status": "online",  # 或 "offline"
        "switch": "on",      # 或 "off"
        "brightness": 80,    # 0-100
        "color": "#FF0000"  # 十六进制颜色代码
    }

if __name__ == "__main__":
    # 启动服务器，使用标准输入输出(stdio)作为通信传输方式
    mcp.run(transport="stdio")