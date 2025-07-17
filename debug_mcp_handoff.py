#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Handoff调试脚本
测试工具调用解析和handoff流程
"""

import asyncio
import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversation_core import NagaConversation
from mcpserver.mcp_manager import get_mcp_manager
from mcpserver.agent_manager import get_agent_manager

def test_parse_tool_calls():
    """测试工具调用解析功能"""
    print("=== 测试工具调用解析 ===")
    
    # 创建对话实例
    conversation = NagaConversation()
    
    # 测试Agent格式的工具调用（AppLauncherAgent是agent类型）
    agent_content = """<<<[TOOL_REQUEST]>>>
agentType: 「始」agent「末」
agent_name: 「始」AppLauncherAgent「末」
prompt: 「始」打开网易云音乐「末」
<<<[END_TOOL_REQUEST]>>>"""
    
    print("测试内容 (Agent格式):")
    print(agent_content)
    print()
    
    # 解析工具调用
    tool_calls = conversation._parse_tool_calls(agent_content)
    print("解析结果:")
    for i, tool_call in enumerate(tool_calls):
        print(f"工具调用 {i+1}:")
        print(f"  名称: {tool_call['name']}")
        print(f"  参数: {json.dumps(tool_call['args'], ensure_ascii=False, indent=2)}")
    print()
    
    return tool_calls

async def test_execute_tool_calls():
    """测试工具调用执行功能"""
    print("=== 测试工具调用执行 ===")
    
    # 创建对话实例
    conversation = NagaConversation()
    
    # 测试Agent格式的工具调用
    agent_content = """<<<[TOOL_REQUEST]>>>
agentType: 「始」agent「末」
agent_name: 「始」AppLauncherAgent「末」
prompt: 「始」打开网易云音乐「末」
<<<[END_TOOL_REQUEST]>>>"""
    
    # 解析工具调用
    tool_calls = conversation._parse_tool_calls(agent_content)
    
    if tool_calls:
        print("开始执行工具调用...")
        result = await conversation._execute_tool_calls(tool_calls)
        print("执行结果:")
        print(result)
    else:
        print("没有解析到工具调用")
    print()

async def test_agent_call():
    """测试Agent调用功能"""
    print("=== 测试Agent调用 ===")
    
    # 创建Agent管理器
    agent_manager = get_agent_manager()
    
    # 检查AppLauncherAgent是否已注册
    available_agents = agent_manager.get_available_agents()
    print("已注册的Agent:")
    for agent in available_agents:
        print(f"  - {agent['name']} ({agent['base_name']})")
    print()
    
    # 尝试调用AppLauncherAgent
    try:
        result = await agent_manager.call_agent("AppLauncherAgent", "打开网易云音乐")
        print("Agent调用结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Agent调用失败: {e}")
    print()

async def test_mcp_manager():
    """测试MCP管理器功能"""
    print("=== 测试MCP管理器 ===")
    
    # 创建MCP管理器
    mcp_manager = get_mcp_manager()
    
    # 获取可用服务
    available_services = mcp_manager.get_available_services_filtered()
    print("可用MCP服务:")
    for service in available_services.get("mcp_services", []):
        print(f"  - {service['name']}: {service['description']}")
    print()
    
    print("可用Agent服务:")
    for service in available_services.get("agent_services", []):
        print(f"  - {service['name']}: {service['description']}")
    print()
    
    # 尝试调用AppLauncherAgent（通过AgentManager）
    try:
        from mcpserver.agent_manager import get_agent_manager
        agent_manager = get_agent_manager()
        result = await agent_manager.call_agent("AppLauncherAgent", "打开网易云音乐")
        print("通过AgentManager调用结果:")
        print(result)
    except Exception as e:
        print(f"AgentManager调用失败: {e}")
    print()

async def test_full_handoff_flow():
    """测试完整的handoff流程"""
    print("=== 测试完整Handoff流程 ===")
    
    # 创建对话实例
    conversation = NagaConversation()
    
    # 模拟用户输入
    user_input = "打开网易云音乐"
    
    # 模拟LLM响应（包含Agent格式的工具调用）
    llm_response = """我需要启动网易云音乐应用。

<<<[TOOL_REQUEST]>>>
agentType: 「始」agent「末」
agent_name: 「始」AppLauncherAgent「末」
prompt: 「始」打开网易云音乐「末」
<<<[END_TOOL_REQUEST]>>>"""
    
    print("用户输入:", user_input)
    print("LLM响应:", llm_response)
    print()
    
    # 解析工具调用
    tool_calls = conversation._parse_tool_calls(llm_response)
    print(f"解析到 {len(tool_calls)} 个工具调用")
    
    if tool_calls:
        # 执行工具调用
        print("执行工具调用...")
        result = await conversation._execute_tool_calls(tool_calls)
        print("执行结果:")
        print(result)
    else:
        print("没有解析到工具调用")
    print()

def main():
    """主函数"""
    print("MCP Handoff调试脚本")
    print("=" * 50)
    
    # 测试工具调用解析
    test_parse_tool_calls()
    
    # 运行异步测试
    async def run_async_tests():
        await test_execute_tool_calls()
        await test_agent_call()
        await test_mcp_manager()
        await test_full_handoff_flow()
    
    # 运行异步测试
    asyncio.run(run_async_tests())
    
    print("调试完成")

if __name__ == "__main__":
    main() 