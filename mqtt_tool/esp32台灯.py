import machine
import network
import time
from umqtt.simple import MQTTClient
import neopixel
import ujson

# WiFi 配置
WIFI_SSID = "waoo2111280"
WIFI_PASSWORD = "waoo2111280"

# MQTT 配置
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "led/mcp"
MQTT_CLIENT_ID = "esp32_led_controller"

# LED 配置
LED_PIN = 5
LED_COUNT = 30  # 根据你的灯带实际数量修改
LED_NP = neopixel.NeoPixel(machine.Pin(LED_PIN), LED_COUNT)

# 状态标志
is_first_run = True  # 标记是否是第一次运行

# 初始化 WiFi
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("正在连接 WiFi...")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        while not wlan.isconnected():
            time.sleep(0.5)
    print("WiFi 连接成功")
    print("IP 地址:", wlan.ifconfig()[0])

# 初始化 MQTT
def connect_mqtt():
    client = MQTTClient(MQTT_CLIENT_ID, MQTT_BROKER, MQTT_PORT)
    client.connect()
    print("已连接到 MQTT 服务器")
    return client

# 设置 LED 颜色和亮度
def set_leds(color, brightness):
    # 解析颜色值 (格式: #RRGGBB)
    if color.startswith("#"):
        color = color[1:]
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    
    # 应用亮度 (0-100 转换为 0-1.0)
    brightness_factor = brightness / 100.0
    r = int(r * brightness_factor)
    g = int(g * brightness_factor)
    b = int(b * brightness_factor)
    
    # 设置所有 LED
    for i in range(LED_COUNT):
        LED_NP[i] = (r, g, b)
    LED_NP.write()

# 处理 MQTT 消息
def mqtt_callback(topic, msg):
    global is_first_run
    
    try:
        data = ujson.loads(msg)
        print("收到命令:", data)
        
        if "command" in data:
            cmd = data["command"].lower()
            value = data.get("value", None)
            
            if cmd == "color" and value:
                set_leds(value, 100)  # 颜色变化时保持当前亮度
                print("设置颜色:", value)
                
            elif cmd == "brightness" and value:
                # 获取当前颜色
                current_color = "{:02x}{:02x}{:02x}".format(LED_NP[0][0], LED_NP[0][1], LED_NP[0][2])
                current_color = "#" + current_color.upper()
                set_leds(current_color, int(value))
                print("设置亮度:", value)
                
            elif cmd == "switch" and value:
                if value.lower() == "on":
                    if is_first_run:
                        # 第一次运行，设置为白色亮度100%
                        set_leds("#FFFFFF", 100)
                        is_first_run = False  # 清除第一次运行标志
                        print("首次运行，设置为白色亮度100%")
                    else:
                        # 获取当前颜色和亮度设置
                        current_color = "{:02x}{:02x}{:02x}".format(LED_NP[0][0], LED_NP[0][1], LED_NP[0][2])
                        current_color = "#" + current_color.upper()
                        # 假设亮度为100（可以根据需要调整）
                        set_leds(current_color, 100)
                    print("开启 LED")
                    
                elif value.lower() == "off":
                    set_leds("#000000", 100)  # 关闭所有 LED
                    print("关闭 LED")
                    
    except Exception as e:
        print("处理消息时出错:", e)

# 主程序
def main():
    global LED_NP, is_first_run
    
    # 初始化硬件
    LED_NP = neopixel.NeoPixel(machine.Pin(LED_PIN), LED_COUNT)
    set_leds("#000000", 100)  # 初始化为关闭状态
    
    # 连接 WiFi
    connect_wifi()
    
    # 连接 MQTT
    client = connect_mqtt()
    client.set_callback(mqtt_callback)
    client.subscribe(MQTT_TOPIC)
    
    print("LED 控制器已启动，等待命令...")
    
    try:
        while True:
            client.check_msg()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("程序终止")
    finally:
        set_leds("#000000", 100)  # 确保关闭 LED
        client.disconnect()
        machine.reset()

if __name__ == "__main__":
    main()