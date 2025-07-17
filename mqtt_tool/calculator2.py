from mcp.server.fastmcp import FastMCP
import sys
import logging
import os
from ctypes import windll

# 创建日志记录器，命名为'Calculator'
logger = logging.getLogger('Calculator')

# 如果是Windows平台，修复控制台的UTF-8编码问题
if sys.platform == 'win32':
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

# 导入数学计算相关的模块
import math
import random

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

# 如果是直接运行此脚本
if __name__ == "__main__":
    try:
        # 启动服务器，使用标准输入输出(stdio)作为通信传输方式
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise