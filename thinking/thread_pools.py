"""
双线程池管理系统
分离思考处理和API调用，避免限流和提升并发性能
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List
from .config import TREE_THINKING_CONFIG

logger = logging.getLogger("ThreadPoolManager")

class ThreadPoolManager:
    """双线程池管理器"""
    
    def __init__(self):
        config = TREE_THINKING_CONFIG
        
        # 思考线程池：用于并行处理不同思考分支
        self.thinking_pool = ThreadPoolExecutor(
            max_workers=config["thinking_pool_size"], 
            thread_name_prefix="thinking"
        )
        
        # API调用线程池：避免API限流
        self.api_pool = ThreadPoolExecutor(
            max_workers=config["api_pool_size"],
            thread_name_prefix="api"
        )
        
        # API限流控制
        self.api_semaphore = asyncio.Semaphore(config["max_concurrent_api"])
        self.last_api_call = 0
        self.min_api_interval = config["min_api_interval"]
        
        # 统计信息
        self.stats = {
            "thinking_tasks": 0,
            "api_tasks": 0,
            "thinking_completed": 0,
            "api_completed": 0,
            "api_errors": 0
        }
        
        logger.info(f"线程池初始化完成 - 思考池:{config['thinking_pool_size']} API池:{config['api_pool_size']}")
    
    async def submit_thinking_task(self, func: Callable, *args, **kwargs) -> Any:
        """提交思考任务到思考线程池"""
        self.stats["thinking_tasks"] += 1
        
        try:
            # 检查是否为异步函数
            if asyncio.iscoroutinefunction(func):
                # 异步函数直接执行
                result = await func(*args, **kwargs)
            else:
                # 同步函数使用线程池
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.thinking_pool, func, *args, **kwargs)
            
            self.stats["thinking_completed"] += 1
            return result
        except Exception as e:
            logger.error(f"思考任务执行失败: {e}")
            raise
    
    async def submit_api_task(self, func: Callable, *args, **kwargs) -> Any:
        """提交API任务到API线程池，带限流控制"""
        self.stats["api_tasks"] += 1
        
        async with self.api_semaphore:
            try:
                # 限流控制
                await self._rate_limit()
                
                # 检查是否为异步函数
                if asyncio.iscoroutinefunction(func):
                    # 异步函数直接执行
                    result = await func(*args, **kwargs)
                else:
                    # 同步函数使用线程池
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.api_pool, func, *args, **kwargs)
                
                self.stats["api_completed"] += 1
                return result
                
            except Exception as e:
                self.stats["api_errors"] += 1
                logger.error(f"API任务执行失败: {e}")
                raise
    
    async def _rate_limit(self):
        """API限流控制"""
        now = time.time()
        if now - self.last_api_call < self.min_api_interval:
            wait_time = self.min_api_interval - (now - self.last_api_call)
            await asyncio.sleep(wait_time)
        
        self.last_api_call = time.time()
    
    async def submit_batch_thinking_tasks(self, tasks: List[tuple]) -> List[Any]:
        """批量提交思考任务"""
        if not tasks:
            return []
        
        logger.info(f"批量提交 {len(tasks)} 个思考任务")
        
        # 创建异步任务
        async_tasks = []
        for func, args, kwargs in tasks:
            task = asyncio.create_task(
                self.submit_thinking_task(func, *args, **kwargs)
            )
            async_tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"思考任务 {i} 失败: {result}")
            else:
                valid_results.append(result)
        
        logger.info(f"批量思考任务完成: {len(valid_results)}/{len(tasks)} 成功")
        return valid_results
    
    async def submit_batch_api_tasks(self, tasks: List[tuple]) -> List[Any]:
        """批量提交API任务（带限流）"""
        if not tasks:
            return []
        
        logger.info(f"批量提交 {len(tasks)} 个API任务")
        
        # 创建异步任务
        async_tasks = []
        for func, args, kwargs in tasks:
            task = asyncio.create_task(
                self.submit_api_task(func, *args, **kwargs)
            )
            async_tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"API任务 {i} 失败: {result}")
            else:
                valid_results.append(result)
        
        logger.info(f"批量API任务完成: {len(valid_results)}/{len(tasks)} 成功")
        return valid_results
    
    def get_pool_status(self) -> dict:
        """获取线程池状态"""
        try:
            thinking_pool_status = {
                "max_workers": self.thinking_pool._max_workers,
                "active_threads": len(self.thinking_pool._threads) if hasattr(self.thinking_pool, '_threads') and self.thinking_pool._threads else 0,
                "pending_tasks": self.thinking_pool._work_queue.qsize() if hasattr(self.thinking_pool, '_work_queue') else 0
            }
            
            api_pool_status = {
                "max_workers": self.api_pool._max_workers,
                "active_threads": len(self.api_pool._threads) if hasattr(self.api_pool, '_threads') and self.api_pool._threads else 0,
                "pending_tasks": self.api_pool._work_queue.qsize() if hasattr(self.api_pool, '_work_queue') else 0
            }
            
            semaphore_status = {
                "available": getattr(self.api_semaphore, '_value', 0),
                "waiting": len(getattr(self.api_semaphore, '_waiters', [])) if hasattr(self.api_semaphore, '_waiters') and self.api_semaphore._waiters is not None else 0
            }
            
            return {
                "thinking_pool": thinking_pool_status,
                "api_pool": api_pool_status,
                "api_semaphore": semaphore_status,
                "stats": self.stats.copy()
            }
        except Exception as e:
            logger.warning(f"获取线程池状态失败: {e}")
            return {
                "thinking_pool": {"status": "error"},
                "api_pool": {"status": "error"},
                "api_semaphore": {"status": "error"},
                "stats": self.stats.copy(),
                "error": str(e)
            }
    
    def cleanup(self):
        """清理线程池资源"""
        logger.info("正在清理线程池资源...")
        
        # 优雅关闭
        self.thinking_pool.shutdown(wait=True)
        self.api_pool.shutdown(wait=True)
        
        # 输出最终统计
        logger.info(f"线程池清理完成 - 统计信息: {self.stats}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

class TaskBatch:
    """任务批次管理器"""
    
    def __init__(self, pool_manager: ThreadPoolManager):
        self.pool_manager = pool_manager
        self.thinking_tasks = []
        self.api_tasks = []
    
    def add_thinking_task(self, func: Callable, *args, **kwargs):
        """添加思考任务"""
        self.thinking_tasks.append((func, args, kwargs))
    
    def add_api_task(self, func: Callable, *args, **kwargs):
        """添加API任务"""
        self.api_tasks.append((func, args, kwargs))
    
    async def execute_all(self) -> tuple:
        """执行所有任务"""
        # 并行执行思考任务和API任务
        thinking_results, api_results = await asyncio.gather(
            self.pool_manager.submit_batch_thinking_tasks(self.thinking_tasks),
            self.pool_manager.submit_batch_api_tasks(self.api_tasks),
            return_exceptions=True
        )
        
        return thinking_results, api_results
    
    def clear(self):
        """清空任务"""
        self.thinking_tasks.clear()
        self.api_tasks.clear()
    
    def get_task_count(self) -> dict:
        """获取任务数量"""
        return {
            "thinking": len(self.thinking_tasks),
            "api": len(self.api_tasks),
            "total": len(self.thinking_tasks) + len(self.api_tasks)
        } 