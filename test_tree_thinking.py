#!/usr/bin/env python3
"""
树状外置思考系统测试脚本
"""

import asyncio
import sys
import time
from thinking import TreeThinkingEngine, UserPreference

# 模拟API客户端
class MockAPIClient:
    async def get_response(self, prompt: str, temperature: float = 0.7) -> str:
        """模拟API响应"""
        print(f"[模拟API调用] 温度: {temperature:.2f}")
        print(f"[提示词长度] {len(prompt)} 字符")
        
        # 根据温度返回不同风格的回答
        if temperature < 0.5:
            return f"基于逻辑分析，这个问题需要从多个角度进行系统性思考。首先我们需要理解问题的本质，然后制定详细的解决方案。通过分析可以得出，最优的解决路径应该是先确定目标，再制定步骤，最后执行验证。"
        elif temperature < 0.8:
            return f"这个问题确实值得深入思考。我认为可以从实用性和创新性两个维度来考虑。实用性方面，我们需要确保方案具有可操作性；创新性方面，我们可以探索一些新的思路和方法。综合这两个方面，我建议采用渐进式的解决方案。"
        else:
            return f"哇，这个问题很有趣！让我从一个全新的角度来思考。如果我们跳出传统思维框架，也许能发现意想不到的解决方案。比如，我们可以尝试逆向思维，或者运用跨领域的知识来解决这个问题。创新往往来自于突破常规的思考方式。"

# 模拟记忆管理器
class MockMemoryManager:
    def search_related_memories(self, query: str, limit: int = 3) -> list:
        """模拟相关记忆搜索"""
        return [
            {"content": "之前我们讨论过类似的问题，当时的解决方案很有效"},
            {"content": "记得上次遇到复杂问题时，分步骤处理效果很好"},
            {"content": "历史经验表明，多角度思考能够得到更全面的解决方案"}
        ][:limit]

async def test_basic_functionality():
    """测试基础功能"""
    print("=" * 60)
    print("🌳 树状外置思考系统 - 基础功能测试")
    print("=" * 60)
    
    # 初始化组件
    api_client = MockAPIClient()
    memory_manager = MockMemoryManager()
    
    # 初始化树状思考引擎
    print("\n1. 初始化树状思考引擎...")
    engine = TreeThinkingEngine(api_client=api_client, memory_manager=memory_manager)
    
    # 检查系统状态
    status = engine.get_system_status()
    print(f"   系统状态: {'启用' if status['enabled'] else '禁用'}")
    print(f"   线程池状态: API池 {status['thread_pool_status']['api_pool']['max_workers']} 线程")
    
    return engine

async def test_difficulty_assessment():
    """测试难度评估"""
    print("\n2. 测试问题难度评估...")
    
    engine = await test_basic_functionality()
    
    test_questions = [
        "今天天气怎么样？",  # 简单
        "请分析人工智能在未来社会发展中的作用和影响。",  # 中等
        "如何设计一个高效、可扩展、安全的分布式系统架构，同时考虑成本优化、性能监控、故障恢复等多个维度？"  # 复杂
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n   问题 {i}: {question[:50]}...")
        assessment = await engine.difficulty_judge.assess_difficulty(question)
        print(f"   难度: {assessment['difficulty']}/5")
        print(f"   思考路线: {assessment['routes']} 条")
        print(f"   评估理由: {assessment['reasoning'][:100]}...")

async def test_preference_system():
    """测试偏好系统"""
    print("\n3. 测试偏好打分系统...")
    
    engine = await test_basic_functionality()
    
    # 创建测试节点
    from thinking import ThinkingNode
    
    test_nodes = [
        ThinkingNode(
            content="这个问题需要深入分析。首先，我们要从逻辑角度考虑，然后结合实际情况进行推理。",
            branch_type="logical"
        ),
        ThinkingNode(
            content="让我们创新性地思考这个问题。也许我们可以尝试一些全新的、突破性的解决方案。",
            branch_type="creative"
        ),
        ThinkingNode(
            content="基于我之前的经验和学习，我记得类似问题的处理方法。历史数据显示这种方法很有效。",
            branch_type="analytical"
        )
    ]
    
    # 进行偏好打分
    scores = await engine.preference_filter.score_thinking_nodes(test_nodes)
    
    print("   节点评分结果:")
    for node in test_nodes:
        score = scores.get(node.id, 0)
        print(f"   - {node.branch_type}: {score:.2f}分")
        print(f"     内容: {node.content[:50]}...")

async def test_tree_thinking():
    """测试完整的树状思考流程"""
    print("\n4. 测试完整树状思考流程...")
    
    engine = await test_basic_functionality()
    
    test_question = "如何在团队中建立有效的沟通机制，提高工作效率？"
    
    print(f"   测试问题: {test_question}")
    print("   开始深度思考...")
    
    start_time = time.time()
    
    # 执行树状思考
    result = await engine.think_deeply(test_question)
    
    end_time = time.time()
    
    print(f"\n   ✅ 思考完成！耗时: {end_time - start_time:.2f}秒")
    print(f"   会话ID: {result.get('session_id', 'N/A')}")
    
    # 显示思考过程
    process_info = result.get('thinking_process', {})
    difficulty = process_info.get('difficulty', {})
    
    print(f"\n   📊 思考过程统计:")
    print(f"   - 问题难度: {difficulty.get('difficulty', 'N/A')}/5")
    print(f"   - 生成路线: {process_info.get('routes_generated', 0)} 条")
    print(f"   - 选择路线: {process_info.get('routes_selected', 0)} 条")
    print(f"   - 处理时间: {process_info.get('processing_time', 0):.2f}秒")
    
    # 显示思考详情
    thinking_details = process_info.get('thinking_details', [])
    if thinking_details:
        print(f"\n   🧠 思考路线详情:")
        for i, detail in enumerate(thinking_details[:3], 1):  # 只显示前3条
            print(f"   路线 {i} ({detail.get('branch_type', 'N/A')}):")
            print(f"   评分: {detail.get('score', 0):.2f}, 适应度: {detail.get('fitness', 0):.3f}")
            print(f"   内容: {detail.get('content', '')[:100]}...")
            print()
    
    # 显示最终答案
    final_answer = result.get('answer', '')
    print(f"   🎯 最终答案:")
    print(f"   {final_answer[:300]}...")
    
    return result

async def test_performance():
    """性能测试"""
    print("\n5. 性能测试...")
    
    engine = await test_basic_functionality()
    
    # 测试多个问题的并发处理
    test_questions = [
        "如何提高学习效率？",
        "项目管理的最佳实践是什么？",
        "如何平衡工作与生活？"
    ]
    
    print(f"   并发处理 {len(test_questions)} 个问题...")
    
    start_time = time.time()
    
    # 并发执行
    tasks = [engine.think_deeply(q) for q in test_questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    successful_results = [r for r in results if not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]
    
    print(f"   ✅ 并发测试完成！")
    print(f"   总耗时: {end_time - start_time:.2f}秒")
    print(f"   成功: {len(successful_results)}/{len(test_questions)}")
    print(f"   失败: {len(failed_results)}")
    
    if failed_results:
        print("   失败原因:")
        for error in failed_results[:3]:  # 只显示前3个错误
            print(f"   - {str(error)[:100]}...")

async def main():
    """主测试函数"""
    print("🚀 启动树状外置思考系统测试")
    print("测试时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 运行所有测试
        await test_basic_functionality()
        await test_difficulty_assessment()
        await test_preference_system()
        await test_tree_thinking()
        await test_performance()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main()) 