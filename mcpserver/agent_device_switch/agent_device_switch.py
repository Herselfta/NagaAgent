# agent_device_switch.py # 设备开关控制Agent，仅保留开关控制功能
import json
from agents import Agent

# 导入设备开关工具
try:
    from mqtt_tool.device_switch import device_manager
except Exception as e:
    print(f"导入设备开关工具失败: {e}")

class AgentMqttTool(Agent):
    name = "agent_mqtt_tool"  # Agent名称
    instructions = "设备开关控制MCP Agent，仅支持通过MQTT控制两个设备的开关状态"  # 角色描述
    def __init__(self):
        super().__init__(
            name=self.name,
            instructions=self.instructions,
            tools=[],
            model="device-switch-mcp"
        )
        print(f"✅ agent_mqtt_tool初始化完成")

    async def handle_handoff(self, data: dict) -> str:
        """只处理switch_devices命令"""
        try:
            tool_name = data.get("tool_name")
            if tool_name != "switch_devices":
                return json.dumps({
                    "status": "error",
                    "message": "仅支持switch_devices操作",
                    "data": {}
                }, ensure_ascii=False)
            device1 = data.get("device1")
            device2 = data.get("device2")
            device3 = data.get("device3")
            if device1 is None or device2 is None or device3 is None:
                return json.dumps({
                    "status": "error",
                    "message": "switch_devices操作需要device1、device2和device3参数",
                    "data": {}
                }, ensure_ascii=False)
            # 调用设备开关工具
            success, message = device_manager.switch_devices(device1, device2, device3)
            result = {
                "success": success,
                "message": message,
                "data": {
                    "device1": device1,
                    "device2": device2,
                    "device3": device3
                }
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"设备控制异常: {str(e)}",
                "data": {}
            }, ensure_ascii=False)

def create_device_switch_agent():
    """创建agent_mqtt_tool实例"""
    return AgentMqttTool() 