"""
Async utilities for performance optimization.

This module provides utilities for async/await optimization patterns including:
- Concurrent execution helpers
- Async context managers
- Async iterators and generators
- Async batching utilities
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, TypeVar, Union, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

from clarity.shared.simple_logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchResult:
    """Result of a batch async operation"""
    results: List[Any]
    errors: List[Exception]
    successful: int
    failed: int
    duration: float
    

class AsyncBatcher:
    """Utility for batching async operations with concurrency control"""
    
    def __init__(self, max_concurrency: int = 10, batch_size: int = 50):
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def execute_batch(
        self, 
        items: List[T], 
        async_func: Callable[[T], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Execute async function on items with concurrency control"""
        start_time = time.time()
        results = []
        errors = []
        
        async def process_item(item: T, index: int) -> Tuple[int, Any, Optional[Exception]]:
            async with self.semaphore:
                try:
                    result = await async_func(item)
                    if progress_callback:
                        progress_callback(index + 1, len(items))
                    return index, result, None
                except Exception as e:
                    logger.warning(f"Batch item {index} failed: {e}")
                    return index, None, e
        
        # Create tasks for all items
        tasks = [process_item(item, i) for i, item in enumerate(items)]
        
        # Execute with proper concurrency control
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                errors.append(task_result)
            else:
                index, result, error = task_result
                if error:
                    errors.append(error)
                else:
                    results.append((index, result))
        
        duration = time.time() - start_time
        
        return BatchResult(
            results=[r[1] for r in sorted(results, key=lambda x: x[0])],
            errors=errors,
            successful=len(results),
            failed=len(errors),
            duration=duration
        )


class AsyncFileProcessor:
    """Async file processing utilities"""
    
    @staticmethod
    async def process_files_concurrently(
        file_paths: List[str],
        process_func: Callable[[str], Any],
        max_concurrency: int = 5
    ) -> Dict[str, Any]:
        """Process files concurrently with async I/O"""
        semaphore = asyncio.Semaphore(max_concurrency)
        results = {}
        
        async def process_file(file_path: str):
            async with semaphore:
                try:
                    # Run file processing in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, process_func, file_path)
                    results[file_path] = result
                except Exception as e:
                    results[file_path] = {"error": str(e)}
        
        tasks = [process_file(path) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results


class AsyncIteratorWrapper:
    """Wrapper to make sync iterators async with proper yielding"""
    
    def __init__(self, sync_iterable, chunk_size: int = 100):
        self.sync_iterable = sync_iterable
        self.chunk_size = chunk_size
    
    async def __aiter__(self):
        count = 0
        for item in self.sync_iterable:
            yield item
            count += 1
            # Yield control periodically to prevent blocking
            if count % self.chunk_size == 0:
                await asyncio.sleep(0)


async def gather_with_limit(
    *tasks: Callable[[], Any],
    limit: int = 10,
    return_exceptions: bool = True
) -> List[Any]:
    """Execute async tasks with concurrency limit"""
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_task(task_func):
        async with semaphore:
            return await task_func()
    
    limited_tasks = [limited_task(task) for task in tasks]
    return await asyncio.gather(*limited_tasks, return_exceptions=return_exceptions)


async def timeout_after(seconds: float, coro):
    """Execute coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds} seconds")
        raise


@asynccontextmanager
async def async_timer(operation_name: str):
    """Async context manager for timing operations"""
    start_time = time.time()
    logger.debug(f"Starting async operation: {operation_name}")
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"Completed async operation: {operation_name} in {duration:.3f}s")


class AsyncCache:
    """Simple async-aware cache with TTL"""
    
    def __init__(self, default_ttl: float = 3600.0):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        async with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.time() < expires_at:
                    return value
                else:
                    del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set cached value with TTL"""
        expires_at = time.time() + (ttl or self._default_ttl)
        async with self._lock:
            self._cache[key] = (value, expires_at)
    
    async def get_or_set(
        self, 
        key: str, 
        factory: Callable[[], Any], 
        ttl: Optional[float] = None
    ) -> Any:
        """Get cached value or compute and cache it"""
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        await self.set(key, value, ttl)
        return value


async def run_with_retries(
    coro_func: Callable[[], Any],
    max_retries: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True
) -> Any:
    """Run async operation with exponential backoff retries"""
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {current_delay}s")
                await asyncio.sleep(current_delay)
                
                if exponential_backoff:
                    current_delay *= 2
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
                raise last_exception


class AsyncMetrics:
    """Collect metrics for async operations"""
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def record(self, operation: str, duration: float):
        """Record operation duration"""
        async with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(duration)
    
    async def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        async with self._lock:
            if operation not in self._metrics:
                return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
            
            durations = self._metrics[operation]
            return {
                "count": len(durations),
                "avg": sum(durations) / len(durations) if durations else 0,
                "min": min(durations),
                "max": max(durations)
            }


# Global metrics instance
async_metrics = AsyncMetrics()


def async_timed(operation_name: str):
    """Decorator to time async operations and record metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                await async_metrics.record(operation_name, duration)
        return wrapper
    return decorator


async def parallel_map(
    async_func: Callable[[T], R],
    items: List[T],
    max_concurrency: int = 10
) -> List[R]:
    """Apply async function to items in parallel with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def limited_func(item: T) -> R:
        async with semaphore:
            return await async_func(item)
    
    tasks = [limited_func(item) for item in items]
    return await asyncio.gather(*tasks)


async def async_chain(*async_funcs: Callable[[Any], Any]):
    """Chain async functions together"""
    async def chained_func(initial_value: Any) -> Any:
        result = initial_value
        for func in async_funcs:
            result = await func(result)
        return result
    
    return chained_func