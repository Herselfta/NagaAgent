# MQTT_Tool.py
from mcp.server.fastmcp import FastMCP
import sys
import logging
import paho.mqtt.client as mqtt
import uuid
import threading
import time

logger = logging.getLogger('MQTT_Tool')

# Windows 控制台 UTF-8 编码修复
if sys.platform == 'win32':
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

mcp = FastMCP("MQTT_Tool")

class MQTTConnectionManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.__init__()
        return cls._instance
    
    def __init__(self):
        self.client = mqtt.Client(client_id=f"mcp_{uuid.uuid4().hex[:8]}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False
        self.last_message = None
        self.subscribed_topics = set()
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to broker {client._host}:{client._port}")
            # 重连后自动重新订阅
            for topic in self.subscribed_topics:
                client.subscribe(topic)
        else:
            logger.error(f"Connection failed with code: {rc}")
    
    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
        except UnicodeDecodeError:
            payload = str(msg.payload)
        
        self.last_message = {
            "topic": msg.topic,
            "payload": payload,
            "qos": msg.qos,
            "timestamp": time.time()
        }
        logger.info(f"Message received [{msg.topic}]: {payload}")

@mcp.tool()
def mqtt_connect(
    broker: str = "broker.emqx.io", 
    port: int = 1883, 
    username: str = "", 
    password: str = "",
    keepalive: int = 60
) -> dict:
    """连接到 MQTT 服务器（默认使用公共测试服务器 broker.emqx.io）"""
    manager = MQTTConnectionManager()
    
    if manager.connected:
        return {"success": False, "error": "Already connected to a broker"}
    
    try:
        if username or password:
            manager.client.username_pw_set(username, password)
        
        manager.client.connect(broker, port, keepalive)
        manager.client.loop_start()
        
        # 等待连接确认
        for _ in range(5):
            if manager.connected:
                return {"success": True, "message": f"Connected to {broker}:{port}"}
            time.sleep(0.5)
        
        return {"success": False, "error": "Connection timeout"}
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def mqtt_publish(
    topic: str, 
    payload: str, 
    qos: int = 0, 
    retain: bool = False
) -> dict:
    """向指定主题发布消息"""
    manager = MQTTConnectionManager()
    if not manager.connected:
        return {"success": False, "error": "Not connected to broker"}
    
    result = manager.client.publish(topic, payload, qos, retain)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        logger.info(f"Published to [{topic}]: {payload}")
        return {"success": True, "message": "Message published"}
    return {"success": False, "error": f"Publish failed with code {result.rc}"}

@mcp.tool()
def mqtt_subscribe(topic: str, qos: int = 0) -> dict:
    """订阅指定主题的消息"""
    manager = MQTTConnectionManager()
    if not manager.connected:
        return {"success": False, "error": "Not connected to broker"}
    
    result = manager.client.subscribe(topic, qos)
    if result[0] == mqtt.MQTT_ERR_SUCCESS:
        manager.subscribed_topics.add(topic)
        logger.info(f"Subscribed to [{topic}] with QoS {qos}")
        return {"success": True, "message": "Subscription successful"}
    return {"success": False, "error": f"Subscribe failed with code {result[0]}"}

@mcp.tool()
def get_last_message() -> dict:
    """获取最后接收到的消息"""
    manager = MQTTConnectionManager()
    if not manager.last_message:
        return {"success": False, "error": "No messages received yet"}
    return {"success": True, "message": manager.last_message}

if __name__ == "__main__":
    mcp.run(transport="stdio")
