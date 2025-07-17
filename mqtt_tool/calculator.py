# 导入必要的模块
from mcp.server.fastmcp import FastMCP  # 导入FastMCP服务器类
import sys  # 系统相关功能
import logging  # 日志记录功能

# 创建日志记录器，命名为'Calculator'
logger = logging.getLogger('Calculator')

# 如果是Windows平台，修复控制台的UTF-8编码问题
if sys.platform == 'win32':
    # 重新配置标准错误输出的编码为UTF-8
    sys.stderr.reconfigure(encoding='utf-8')
    # 重新配置标准输出的编码为UTF-8
    sys.stdout.reconfigure(encoding='utf-8')

# 导入数学计算相关的模块
import math  # 数学函数
import random  # 随机数生成

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
    
    说明:
        可以使用math和random模块中的函数进行计算
        示例:
            "math.sqrt(16)" -> 4.0
            "random.randint(1, 100)" -> 随机整数
    """
    # 使用eval计算表达式结果
    # 注意: 在实际生产环境中，使用eval可能存在安全风险
    # 这里假设该工具只在受控环境中使用
    result = eval(python_expression)
    
    # 记录日志，包括计算的表达式和结果
    logger.info(f"Calculating formula: {python_expression}, result: {result}")
    
    # 返回包含成功标志和计算结果的字典
    return {"success": True, "result": result}

# 如果是直接运行此脚本(而不是作为模块导入)
if __name__ == "__main__":
    # 启动服务器，使用标准输入输出(stdio)作为通信传输方式
    mcp.run(transport="stdio")