#!/usr/bin/env python3
"""
🌳 树状外置思考系统演示
展示NagaAgent 2.2beta的高级推理能力
"""

import asyncio
import sys
import os

# 确保能够导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversation_core import NagaConversation

async def demo_tree_thinking():
    """演示树状思考系统的完整功能"""
    
    print("🌳" + "=" * 60)
    print("    NagaAgent 2.2beta 树状外置思考系统演示")
    print("=" * 64)
    
    # 初始化对话系统
    print("\n🚀 初始化NagaAgent...")
    conv = NagaConversation()
    
    if not conv.tree_thinking:
        print("❌ 树状思考系统未初始化，无法进行演示")
        return
    
    print("✅ 树状思考系统初始化成功")
    
    # 演示问题列表
    demo_questions = [
        {
            "question": "如何设计一个可扩展的微服务架构？",
            "description": "复杂技术问题，预期触发多路思考"
        },
        {
            "question": "分析人工智能对未来教育的影响",
            "description": "跨领域分析问题，需要深度思考"
        },
        {
            "question": "今天天气如何？",
            "description": "简单问题，不会触发树状思考"
        }
    ]
    
    for i, demo in enumerate(demo_questions, 1):
        print(f"\n📝 演示 {i}: {demo['description']}")
        print(f"问题: {demo['question']}")
        print("-" * 60)
        
        # 收集完整回答
        full_response = []
        
        async for speaker, message in conv.process(demo['question']):
            print(f"{speaker}: {message}")
            full_response.append(f"{speaker}: {message}")
        
        print(f"\n✅ 演示 {i} 完成")
        print("=" * 60)
        
        # 等待一下再进行下一个演示
        await asyncio.sleep(1)
    
    # 显示系统状态
    if conv.tree_thinking:
        status = conv.tree_thinking.get_system_status()
        print(f"\n📊 系统状态总结:")
        print(f"• 树状思考系统: {'启用' if status['enabled'] else '禁用'}")
        print(f"• 总计思考会话: {status['total_sessions']}")
        print(f"• 线程池状态: 正常运行")
    
    print(f"\n🎉 演示完成！")
    print("树状外置思考系统已成功集成到NagaAgent中。")

async def demo_control_commands():
    """演示控制命令功能"""
    print("\n🎛️ 控制命令演示")
    print("-" * 30)
    
    conv = NagaConversation()
    
    commands = [
        "#tree status",
        "#tree off", 
        "#tree on"
    ]
    
    for cmd in commands:
        print(f"\n执行命令: {cmd}")
        async for speaker, message in conv.process(cmd):
            print(f"{speaker}: {message}")

if __name__ == "__main__":
    async def main():
        try:
            await demo_tree_thinking()
            await demo_control_commands()
        except KeyboardInterrupt:
            print("\n⚠️ 演示被用户中断")
        except Exception as e:
            print(f"\n❌ 演示过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main()) 