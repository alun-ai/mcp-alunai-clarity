"""
Performance Optimization System for Enhanced MCP Discovery.

This module provides comprehensive performance optimization including:
- Intelligent caching strategies
- Parallel discovery operations
- Response time optimization
- Memory usage optimization
- Batch processing optimization
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    cache_hit: bool
    parallel_execution: bool
    memory_usage_mb: Optional[float] = None
    items_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_name': self.operation_name,
            'duration_ms': self.duration * 1000,
            'cache_hit': self.cache_hit,
            'parallel_execution': self.parallel_execution,
            'memory_usage_mb': self.memory_usage_mb,
            'items_processed': self.items_processed,
            'timestamp': datetime.fromtimestamp(self.end_time).isoformat()
        }


class PerformanceCache:
    """High-performance cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """Initialize cache with size limit and TTL."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL check."""
        if key not in self._cache:
            self._stats['misses'] += 1
            return None
        
        entry = self._cache[key]
        current_time = time.time()
        
        # Check TTL
        if current_time > entry['expires_at']:
            self.delete(key)
            self._stats['misses'] += 1
            return None
        
        # Update access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        self._stats['hits'] += 1
        return entry['value']
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        current_time = time.time()
        
        # Remove if already exists
        if key in self._cache:
            self._access_order.remove(key)
        else:
            # Check size limit
            if len(self._cache) >= self.max_size:
                self._evict_lru()
        
        self._cache[key] = {
            'value': value,
            'created_at': current_time,
            'expires_at': current_time + ttl
        }
        self._access_order.append(key)
        self._stats['size'] = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats['size'] = len(self._cache)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        self._stats['size'] = 0
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            self.delete(lru_key)
            self._stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class ParallelExecutor:
    """Optimized parallel execution manager."""
    
    def __init__(self, max_workers: int = 10, timeout: float = 30.0):
        """Initialize parallel executor."""
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def execute_parallel(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """Execute tasks in parallel with timeout and error handling."""
        if not tasks:
            return []
        
        loop = asyncio.get_event_loop()
        futures = []
        
        for func, args, kwargs in tasks:
            # Wrap synchronous functions for async execution
            if asyncio.iscoroutinefunction(func):
                future = loop.create_task(func(*args, **kwargs))
            else:
                future = loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
            futures.append(future)
        
        results = []
        try:
            # Wait for all tasks with timeout
            completed = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=self.timeout
            )
            
            for result in completed:
                if isinstance(result, Exception):
                    logger.warning(f"Parallel task failed: {result}")
                    results.append(None)
                else:
                    results.append(result)
                    
        except asyncio.TimeoutError:
            logger.warning(f"Parallel execution timed out after {self.timeout}s")
            # Cancel remaining tasks
            for future in futures:
                if not future.done():
                    future.cancel()
            results = [None] * len(tasks)
        
        return results
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class PerformanceOptimizer:
    """Comprehensive performance optimization system."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.cache = PerformanceCache(max_size=1000, default_ttl=300.0)
        self.executor = ParallelExecutor(max_workers=10, timeout=30.0)
        self.metrics: List[PerformanceMetrics] = []
        self.max_metrics = 1000  # Keep last 1000 metrics
        
        # Performance targets
        self.target_response_time = 0.5  # 500ms
        self.max_parallel_tasks = 10
        self.cache_warmup_enabled = True
        
        # Optimization strategies
        self.strategies = {
            'server_discovery': self._optimize_server_discovery,
            'tool_discovery': self._optimize_tool_discovery,
            'slash_commands': self._optimize_slash_commands,
            'resource_monitoring': self._optimize_resource_monitoring,
            'workflow_memory': self._optimize_workflow_memory
        }
        
        # Batch processing configuration
        self.batch_sizes = {
            'servers': 5,
            'tools': 20,
            'commands': 10,
            'patterns': 15
        }
        
        # Weak references to avoid circular references
        self._monitored_objects = weakref.WeakSet()
    
    def performance_monitor(self, operation_name: str):
        """Decorator for monitoring operation performance."""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    cache_key = f"{operation_name}:{hash(str(args) + str(kwargs))}"
                    
                    # Check cache first
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        metric = PerformanceMetrics(
                            operation_name=operation_name,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            cache_hit=True,
                            parallel_execution=False
                        )
                        self._record_metric(metric)
                        return cached_result
                    
                    # Execute async function
                    try:
                        result = await func(*args, **kwargs)
                        
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Cache result if it's serializable and reasonably sized
                        try:
                            if self._should_cache(result, duration):
                                self.cache.put(cache_key, result)
                        except Exception as e:
                            logger.debug(f"Could not cache result for {operation_name}: {e}")
                        
                        # Record performance metric
                        metric = PerformanceMetrics(
                            operation_name=operation_name,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            cache_hit=False,
                            parallel_execution=False
                        )
                        self._record_metric(metric)
                        
                        return result
                        
                    except Exception as e:
                        logger.error(f"Performance monitoring error in {operation_name}: {e}")
                        # Still record the metric for failure analysis
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        metric = PerformanceMetrics(
                            operation_name=operation_name,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            cache_hit=False,
                            parallel_execution=False,
                            error=str(e)
                        )
                        self._record_metric(metric)
                        raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    cache_key = f"{operation_name}:{hash(str(args) + str(kwargs))}"
                    
                    # Check cache first
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        metric = PerformanceMetrics(
                            operation_name=operation_name,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            cache_hit=True,
                            parallel_execution=False
                        )
                        self._record_metric(metric)
                        return cached_result
                    
                    # Execute sync function
                    try:
                        result = func(*args, **kwargs)
                        
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Cache result if it's serializable and reasonably sized
                        try:
                            if self._should_cache(result, duration):
                                self.cache.put(cache_key, result)
                        except Exception as e:
                            logger.debug(f"Could not cache result for {operation_name}: {e}")
                        
                        metric = PerformanceMetrics(
                            operation_name=operation_name,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            cache_hit=False,
                            parallel_execution=False,
                            items_processed=len(result) if isinstance(result, (list, dict)) else 1
                        )
                        self._record_metric(metric)
                        
                        return result
                        
                    except Exception as e:
                        logger.error(f"Performance monitoring error in {operation_name}: {e}")
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        metric = PerformanceMetrics(
                            operation_name=operation_name,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            cache_hit=False,
                            parallel_execution=False,
                            error=str(e)
                        )
                        self._record_metric(metric)
                        raise
                
                return sync_wrapper
        return decorator
    
    async def optimize_discovery_workflow(self, discovery_tasks: Dict[str, Callable]) -> Dict[str, Any]:
        """Optimize complete discovery workflow with parallel execution and caching."""
        start_time = time.time()
        results = {}
        
        # Group tasks by dependencies and parallelizability
        parallel_tasks = []
        sequential_tasks = []
        
        for task_name, task_func in discovery_tasks.items():
            if task_name in ['native_discovery', 'config_discovery', 'env_discovery']:
                # These can run in parallel
                parallel_tasks.append((task_name, task_func))
            else:
                # These need results from parallel tasks
                sequential_tasks.append((task_name, task_func))
        
        # Execute parallel tasks
        if parallel_tasks:
            parallel_results = await self._execute_parallel_tasks(parallel_tasks)
            results.update(parallel_results)
        
        # Execute sequential tasks with results from parallel tasks
        for task_name, task_func in sequential_tasks:
            try:
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(results)
                else:
                    result = task_func(results)
                results[task_name] = result
            except Exception as e:
                logger.warning(f"Sequential task {task_name} failed: {e}")
                results[task_name] = None
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record overall workflow performance
        metric = PerformanceMetrics(
            operation_name="discovery_workflow",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            cache_hit=False,
            parallel_execution=True,
            items_processed=len(results)
        )
        self._record_metric(metric)
        
        # Check if we met performance target
        if duration > self.target_response_time:
            logger.warning(f"Discovery workflow took {duration:.3f}s, target is {self.target_response_time}s")
            await self._trigger_optimization()
        
        return results
    
    async def _execute_parallel_tasks(self, tasks: List[Tuple[str, Callable]]) -> Dict[str, Any]:
        """Execute tasks in parallel with optimized batching."""
        if len(tasks) <= self.max_parallel_tasks:
            # Execute all tasks in parallel
            task_functions = [(task_func, (), {}) for _, task_func in tasks]
            results_list = await self.executor.execute_parallel(task_functions)
            
            return {
                task_name: result 
                for (task_name, _), result in zip(tasks, results_list)
                if result is not None
            }
        else:
            # Batch execution for large number of tasks
            results = {}
            for i in range(0, len(tasks), self.max_parallel_tasks):
                batch = tasks[i:i + self.max_parallel_tasks]
                batch_functions = [(task_func, (), {}) for _, task_func in batch]
                batch_results = await self.executor.execute_parallel(batch_functions)
                
                for (task_name, _), result in zip(batch, batch_results):
                    if result is not None:
                        results[task_name] = result
            
            return results
    
    def _optimize_server_discovery(self) -> Dict[str, Any]:
        """Optimize server discovery performance."""
        return {
            'batch_size': self.batch_sizes['servers'],
            'cache_ttl': 300,  # 5 minutes
            'parallel_config_parsing': True,
            'native_discovery_priority': True
        }
    
    def _optimize_tool_discovery(self) -> Dict[str, Any]:
        """Optimize tool discovery performance."""
        return {
            'batch_size': self.batch_sizes['tools'],
            'cache_ttl': 600,  # 10 minutes
            'concurrent_server_connections': 3,
            'connection_timeout': 5.0,
            'tool_inference_fallback': True
        }
    
    def _optimize_slash_commands(self) -> Dict[str, Any]:
        """Optimize slash command discovery performance."""
        return {
            'batch_size': self.batch_sizes['commands'],
            'cache_ttl': 300,
            'prompt_discovery_timeout': 5.0,
            'categorization_cache': True
        }
    
    def _optimize_resource_monitoring(self) -> Dict[str, Any]:
        """Optimize resource reference monitoring performance."""
        return {
            'pattern_cache_size': 500,
            'opportunity_detection_timeout': 1.0,
            'batch_pattern_learning': True,
            'regex_compilation_cache': True
        }
    
    def _optimize_workflow_memory(self) -> Dict[str, Any]:
        """Optimize workflow memory performance."""
        return {
            'pattern_retrieval_limit': 10,
            'similarity_calculation_timeout': 2.0,
            'memory_query_cache': True,
            'batch_pattern_storage': True
        }
    
    async def _trigger_optimization(self):
        """Trigger optimization strategies when performance targets are missed."""
        logger.info("Triggering performance optimization")
        
        # Analyze recent metrics
        recent_metrics = self.metrics[-100:] if len(self.metrics) >= 100 else self.metrics
        slow_operations = [
            m for m in recent_metrics 
            if m.duration > self.target_response_time and not m.cache_hit
        ]
        
        if slow_operations:
            # Identify bottlenecks
            operation_times = {}
            for metric in slow_operations:
                if metric.operation_name not in operation_times:
                    operation_times[metric.operation_name] = []
                operation_times[metric.operation_name].append(metric.duration)
            
            # Apply targeted optimizations
            for operation, times in operation_times.items():
                avg_time = sum(times) / len(times)
                if avg_time > self.target_response_time * 2:  # Significantly slow
                    await self._apply_optimization_strategy(operation)
    
    async def _apply_optimization_strategy(self, operation_name: str):
        """Apply specific optimization strategy for slow operation."""
        base_operation = operation_name.split('_')[0] if '_' in operation_name else operation_name
        
        if base_operation in self.strategies:
            optimization = self.strategies[base_operation]()
            logger.info(f"Applied optimization for {operation_name}: {optimization}")
            
            # Adjust cache TTL for slow operations
            if 'cache_ttl' in optimization:
                # Increase cache time for slow operations
                self.cache.default_ttl = max(self.cache.default_ttl, optimization['cache_ttl'])
    
    def _should_cache(self, result: Any, duration: float) -> bool:
        """Determine if result should be cached based on size and computation time."""
        try:
            # Cache if operation took significant time
            if duration > 0.1:  # 100ms
                return True
            
            # Cache if result is reasonably sized
            if isinstance(result, (dict, list)):
                # Rough size estimation
                size_estimate = len(json.dumps(result, default=str))
                return size_estimate < 100000  # 100KB
            
            return True
            
        except Exception:
            return False
    
    def _record_metric(self, metric: PerformanceMetrics):
        """Record performance metric."""
        self.metrics.append(metric)
        
        # Keep metrics list manageable
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics // 2:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return {'status': 'no_metrics', 'cache_stats': self.cache.get_stats()}
        
        # Calculate overall statistics
        recent_metrics = self.metrics[-100:] if len(self.metrics) >= 100 else self.metrics
        
        total_duration = sum(m.duration for m in recent_metrics)
        avg_duration = total_duration / len(recent_metrics)
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        cache_hit_rate = cache_hits / len(recent_metrics)
        
        # Operation-specific statistics
        operation_stats = {}
        for metric in recent_metrics:
            op = metric.operation_name
            if op not in operation_stats:
                operation_stats[op] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'cache_hits': 0,
                    'items_processed': 0
                }
            
            stats = operation_stats[op]
            stats['count'] += 1
            stats['total_duration'] += metric.duration
            if metric.cache_hit:
                stats['cache_hits'] += 1
            stats['items_processed'] += metric.items_processed
        
        # Calculate averages and rates
        for op, stats in operation_stats.items():
            stats['avg_duration_ms'] = (stats['total_duration'] / stats['count']) * 1000
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['count']
            stats['avg_items_per_operation'] = stats['items_processed'] / stats['count']
        
        # Performance targets analysis
        target_compliance = {
            'avg_response_time_ms': avg_duration * 1000,
            'target_response_time_ms': self.target_response_time * 1000,
            'meets_target': avg_duration <= self.target_response_time,
            'cache_effectiveness': cache_hit_rate,
            'parallel_execution_rate': sum(1 for m in recent_metrics if m.parallel_execution) / len(recent_metrics)
        }
        
        return {
            'status': 'active',
            'overall_stats': {
                'total_operations': len(recent_metrics),
                'avg_duration_ms': avg_duration * 1000,
                'cache_hit_rate': cache_hit_rate,
                'total_duration_ms': total_duration * 1000
            },
            'operation_stats': operation_stats,
            'cache_stats': self.cache.get_stats(),
            'target_compliance': target_compliance,
            'optimization_recommendations': self._generate_optimization_recommendations(operation_stats, target_compliance)
        }
    
    def _generate_optimization_recommendations(self, operation_stats: Dict[str, Any], target_compliance: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if not target_compliance['meets_target']:
            recommendations.append(f"Average response time ({target_compliance['avg_response_time_ms']:.1f}ms) exceeds target ({target_compliance['target_response_time_ms']:.1f}ms)")
        
        if target_compliance['cache_effectiveness'] < 0.3:
            recommendations.append("Low cache hit rate - consider increasing cache TTL or improving cache key strategies")
        
        if target_compliance['parallel_execution_rate'] < 0.5:
            recommendations.append("Low parallel execution rate - consider more aggressive parallelization")
        
        # Find slowest operations
        slow_operations = [
            (op, stats) for op, stats in operation_stats.items()
            if stats['avg_duration_ms'] > target_compliance['target_response_time_ms']
        ]
        
        if slow_operations:
            slow_operations.sort(key=lambda x: x[1]['avg_duration_ms'], reverse=True)
            top_slow = slow_operations[:3]
            slow_ops_list = [f"{op} ({stats['avg_duration_ms']:.1f}ms)" for op, stats in top_slow]
            recommendations.append(f"Slowest operations: {', '.join(slow_ops_list)}")
        
        return recommendations
    
    async def warmup_cache(self, warmup_functions: Dict[str, Callable]):
        """Warmup cache with common operations."""
        if not self.cache_warmup_enabled:
            return
        
        logger.info("Starting cache warmup")
        start_time = time.time()
        
        warmup_tasks = [(func, (), {}) for func in warmup_functions.values()]
        await self.executor.execute_parallel(warmup_tasks)
        
        duration = time.time() - start_time
        logger.info(f"Cache warmup completed in {duration:.2f}s")
    
    def clear_caches(self):
        """Clear all caches for fresh start."""
        self.cache.clear()
        logger.info("All performance caches cleared")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'cache_size': len(self.cache._cache),
                'metrics_count': len(self.metrics)
            }
        except ImportError:
            return {
                'cache_size': len(self.cache._cache),
                'metrics_count': len(self.metrics)
            }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            try:
                del self.executor
            except Exception:
                pass