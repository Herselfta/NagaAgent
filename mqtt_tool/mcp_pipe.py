"""
此脚本用于连接到MCP服务器，并通过WebSocket端点进行输入输出管道传输。
版本: 0.1.0

使用方法:

export MCP_ENDPOINT=<mcp_endpoint>  # 设置环境变量
python mcp_pipe.py <mcp_script>      # 运行脚本并指定要执行的MCP脚本

"""

import asyncio
import websockets
import subprocess
import logging
import os
import signal
import sys
import random
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MCP_PIPE')

# 重连设置参数
INITIAL_BACKOFF = 1  # 初始等待时间(秒)
MAX_BACKOFF = 600    # 最大等待时间(秒)
reconnect_attempt = 0  # 重连尝试次数计数器
backoff = INITIAL_BACKOFF  # 当前等待时间

async def connect_with_retry(uri):
    """带重试机制的WebSocket服务器连接函数"""
    global reconnect_attempt, backoff
    while True:  # 无限重试循环
        try:
            if reconnect_attempt > 0:
                # 计算带有随机抖动的等待时间，避免多个客户端同时重连
                wait_time = backoff * (1 + random.random() * 0.1)
                logger.info(f"等待 {wait_time:.2f} 秒后进行第 {reconnect_attempt} 次重连尝试...")
                await asyncio.sleep(wait_time)
                
            # 尝试连接服务器
            await connect_to_server(uri)
        
        except Exception as e:
            reconnect_attempt += 1
            logger.warning(f"连接关闭(尝试次数: {reconnect_attempt}): {e}")            
            # 计算下一次重连的等待时间(指数退避算法)
            backoff = min(backoff * 2, MAX_BACKOFF)

async def connect_to_server(uri):
    """连接到WebSocket服务器并建立与mcp_script的双向通信"""
    global reconnect_attempt, backoff
    try:
        logger.info(f"正在连接WebSocket服务器...")
        async with websockets.connect(uri) as websocket:
            logger.info(f"成功连接到WebSocket服务器")
            
            # 如果连接正常关闭，重置重连计数器
            reconnect_attempt = 0
            backoff = INITIAL_BACKOFF
            
            # 启动mcp_script进程
            process = subprocess.Popen(
                ['python', mcp_script],
                stdin=subprocess.PIPE,   # 标准输入管道
                stdout=subprocess.PIPE,  # 标准输出管道
                stderr=subprocess.PIPE,  # 标准错误管道
                encoding='utf-8',        # 编码格式
                text=True                # 使用文本模式
            )
            logger.info(f"已启动 {mcp_script} 进程")
            
            # 创建三个并发任务:
            # 1. 从WebSocket读取数据并写入进程标准输入
            # 2. 从进程标准输出读取数据并写入WebSocket
            # 3. 从进程标准错误读取数据并输出到终端
            await asyncio.gather(
                pipe_websocket_to_process(websocket, process),
                pipe_process_to_websocket(process, websocket),
                pipe_process_stderr_to_terminal(process)
            )
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket连接关闭: {e}")
        raise  # 重新抛出异常以触发重连
    except Exception as e:
        logger.error(f"连接错误: {e}")
        raise  # 重新抛出异常
    finally:
        # 确保子进程被正确终止
        if 'process' in locals():
            logger.info(f"正在终止 {mcp_script} 进程")
            try:
                process.terminate()  # 尝试优雅终止
                process.wait(timeout=5)  # 等待5秒
            except subprocess.TimeoutExpired:
                process.kill()  # 强制终止
            logger.info(f"{mcp_script} 进程已终止")

async def pipe_websocket_to_process(websocket, process):
    """从WebSocket读取数据并写入进程的标准输入"""
    try:
        while True:
            # 从WebSocket接收消息
            message = await websocket.recv()
            logger.debug(f"<< {message[:120]}...")  # 记录接收的数据(最多120字符)
            
            # 写入进程的标准输入(文本模式)
            if isinstance(message, bytes):
                message = message.decode('utf-8')  # 如果是字节数据则解码
            process.stdin.write(message + '\n')    # 写入数据并添加换行符
            process.stdin.flush()                  # 立即刷新缓冲区
    except Exception as e:
        logger.error(f"WebSocket到进程的管道错误: {e}")
        raise  # 重新抛出异常以触发重连
    finally:
        # 关闭进程的标准输入
        if not process.stdin.closed:
            process.stdin.close()

async def pipe_process_to_websocket(process, websocket):
    """从进程的标准输出读取数据并发送到WebSocket"""
    try:
        while True:
            # 从进程标准输出读取数据(在独立线程中执行)
            data = await asyncio.get_event_loop().run_in_executor(
                None, process.stdout.readline
            )
            
            if not data:  # 如果没有数据，可能进程已结束
                logger.info("进程输出已结束")
                break
                
            # 发送数据到WebSocket
            logger.debug(f">> {data[:120]}...")  # 记录发送的数据(最多120字符)
            # 在文本模式下，数据已经是字符串，无需解码
            await websocket.send(data)
    except Exception as e:
        logger.error(f"进程到WebSocket的管道错误: {e}")
        raise  # 重新抛出异常以触发重连

async def pipe_process_stderr_to_terminal(process):
    """从进程的标准错误读取数据并输出到终端"""
    try:
        while True:
            # 从进程标准错误读取数据(在独立线程中执行)
            data = await asyncio.get_event_loop().run_in_executor(
                None, process.stderr.readline
            )
            
            if not data:  # 如果没有数据，可能进程已结束
                logger.info("进程错误输出已结束")
                break
                
            # 将标准错误数据输出到终端(在文本模式下，数据已经是字符串)
            sys.stderr.write(data)
            sys.stderr.flush()
    except Exception as e:
        logger.error(f"进程标准错误管道错误: {e}")
        raise  # 重新抛出异常以触发重连

def signal_handler(sig, frame):
    """处理中断信号(如Ctrl+C)"""
    logger.info("收到中断信号，正在关闭...")
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 检查是否提供了mcp_script参数
    if len(sys.argv) < 2:
        logger.error("用法: mcp_pipe.py <mcp_script>")
        sys.exit(1)
    
    mcp_script = sys.argv[1]  # 获取要执行的MCP脚本路径
    
    # 从环境变量获取MCP端点URL
    endpoint_url = os.environ.get('MCP_ENDPOINT')
    if not endpoint_url:
        logger.error("请设置MCP_ENDPOINT环境变量")
        sys.exit(1)
    
    # 启动主循环
    try:
        asyncio.run(connect_with_retry(endpoint_url))
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")