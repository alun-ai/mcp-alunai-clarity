"""
Performance monitoring decorators and context managers.

Provides easy-to-use performance monitoring for functions and code blocks.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from contextlib import contextmanager, asynccontextmanager

from loguru import logger

from .metrics import get_metrics_collector

F = TypeVar('F', bound=Callable[..., Any])


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self, metrics_collector=None):
        self.metrics_collector = metrics_collector or get_metrics_collector()
    
    def measure(self, operation_name: str = None, tags: Optional[Dict[str, str]] = None):
        """Decorator to measure function performance."""
        def decorator(func: F) -> F:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    success = True
                    
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        raise
                    finally:
                        duration = time.perf_counter() - start_time
                        self.metrics_collector.time_operation(op_name, duration, success)
                        
                        if tags:
                            self.metrics_collector.record_histogram(
                                "operation_duration", duration, {**tags, "operation": op_name}
                            )
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    success = True
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        raise
                    finally:
                        duration = time.perf_counter() - start_time
                        self.metrics_collector.time_operation(op_name, duration, success)
                        
                        if tags:
                            self.metrics_collector.record_histogram(
                                "operation_duration", duration, {**tags, "operation": op_name}
                            )
                
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def timer(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager to time a code block."""
        start_time = time.perf_counter()
        success = True
        
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.metrics_collector.time_operation(operation_name, duration, success)
            
            if tags:
                self.metrics_collector.record_histogram(
                    "operation_duration", duration, {**tags, "operation": operation_name}
                )
    
    @asynccontextmanager
    async def async_timer(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Async context manager to time a code block."""
        start_time = time.perf_counter()
        success = True
        
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.metrics_collector.time_operation(operation_name, duration, success)
            
            if tags:
                self.metrics_collector.record_histogram(
                    "operation_duration", duration, {**tags, "operation": operation_name}
                )


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def measure_performance(operation_name: str = None, tags: Optional[Dict[str, str]] = None):
    """Convenience decorator for performance measurement."""
    return performance_monitor.measure(operation_name, tags)


def monitor_memory_operations(func: F) -> F:
    """Specialized decorator for memory operations."""
    return performance_monitor.measure(tags={"component": "memory"})(func)


def monitor_autocode_operations(func: F) -> F:
    """Specialized decorator for AutoCode operations."""
    return performance_monitor.measure(tags={"component": "autocode"})(func)


def monitor_mcp_operations(func: F) -> F:
    """Specialized decorator for MCP operations."""
    return performance_monitor.measure(tags={"component": "mcp"})(func)


def monitor_database_operations(func: F) -> F:
    """Specialized decorator for database operations."""
    return performance_monitor.measure(tags={"component": "database"})(func)


class SystemMetricsCollector:
    """Collect system-level performance metrics."""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self._last_collection = time.time()
    
    def collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Memory metrics
            memory_info = process.memory_info()
            self.metrics_collector.set_gauge("system_memory_rss_bytes", memory_info.rss)
            self.metrics_collector.set_gauge("system_memory_vms_bytes", memory_info.vms)
            
            # CPU metrics
            cpu_percent = process.cpu_percent()
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
            
            # File descriptors (Unix-like systems)
            try:
                num_fds = process.num_fds()
                self.metrics_collector.set_gauge("system_open_files", num_fds)
            except AttributeError:
                pass  # Not available on Windows
            
            # Thread count
            num_threads = process.num_threads()
            self.metrics_collector.set_gauge("system_thread_count", num_threads)
            
            # Process age
            create_time = process.create_time()
            uptime = time.time() - create_time
            self.metrics_collector.set_gauge("system_uptime_seconds", uptime)
            
        except ImportError:
            # psutil not available, collect basic metrics only
            pass
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def collect_memory_domain_metrics(self, domain_manager) -> None:
        """Collect memory domain specific metrics."""
        try:
            # Get stats from domain manager
            stats = asyncio.create_task(domain_manager.get_memory_stats())
            asyncio.get_event_loop().run_until_complete(stats)
            stats_result = stats.result()
            
            # Memory counts by type
            if "memory_counts" in stats_result:
                for memory_type, count in stats_result["memory_counts"].items():
                    self.metrics_collector.set_gauge(
                        "memory_count", count, {"type": memory_type}
                    )
            
            # Cache metrics
            if "cache_stats" in stats_result:
                cache_stats = stats_result["cache_stats"]
                self.metrics_collector.set_gauge("cache_hit_rate", cache_stats.get("hit_rate", 0))
                self.metrics_collector.set_gauge("cache_size", cache_stats.get("size", 0))
            
        except Exception as e:
            logger.warning(f"Failed to collect memory domain metrics: {e}")
    
    def collect_autocode_metrics(self, autocode_domain) -> None:
        """Collect AutoCode domain specific metrics."""
        try:
            stats = asyncio.create_task(autocode_domain.get_stats())
            asyncio.get_event_loop().run_until_complete(stats)
            stats_result = stats.result()
            
            # Component health
            if "component_health" in stats_result:
                for component, health in stats_result["component_health"].items():
                    self.metrics_collector.set_gauge(
                        "component_healthy", 1 if health.get("healthy", False) else 0,
                        {"component": component}
                    )
            
            # Operation counts
            if "operations" in stats_result:
                for op, op_stats in stats_result["operations"].items():
                    self.metrics_collector.set_gauge(
                        "operation_total", op_stats.get("count", 0),
                        {"operation": op}
                    )
            
        except Exception as e:
            logger.warning(f"Failed to collect AutoCode metrics: {e}")


class PerformanceProfiler:
    """Advanced performance profiling utilities."""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self._profiling_data = {}
    
    def start_profiling(self, profile_name: str) -> None:
        """Start a profiling session."""
        self._profiling_data[profile_name] = {
            "start_time": time.perf_counter(),
            "checkpoints": []
        }
    
    def checkpoint(self, profile_name: str, checkpoint_name: str) -> None:
        """Add a checkpoint to a profiling session."""
        if profile_name not in self._profiling_data:
            logger.warning(f"Profiling session '{profile_name}' not found")
            return
        
        current_time = time.perf_counter()
        start_time = self._profiling_data[profile_name]["start_time"]
        
        checkpoint = {
            "name": checkpoint_name,
            "timestamp": current_time,
            "elapsed": current_time - start_time
        }
        
        self._profiling_data[profile_name]["checkpoints"].append(checkpoint)
    
    def end_profiling(self, profile_name: str) -> Dict[str, Any]:
        """End a profiling session and return results."""
        if profile_name not in self._profiling_data:
            logger.warning(f"Profiling session '{profile_name}' not found")
            return {}
        
        session_data = self._profiling_data[profile_name]
        end_time = time.perf_counter()
        total_duration = end_time - session_data["start_time"]
        
        # Calculate checkpoint durations
        checkpoints = session_data["checkpoints"]
        checkpoint_durations = []
        
        prev_time = session_data["start_time"]
        for checkpoint in checkpoints:
            duration = checkpoint["timestamp"] - prev_time
            checkpoint_durations.append({
                "name": checkpoint["name"],
                "duration": duration,
                "cumulative": checkpoint["elapsed"]
            })
            prev_time = checkpoint["timestamp"]
        
        results = {
            "profile_name": profile_name,
            "total_duration": total_duration,
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoint_durations
        }
        
        # Record metrics
        self.metrics_collector.record_histogram(
            "profile_duration", total_duration, {"profile": profile_name}
        )
        
        # Clean up
        del self._profiling_data[profile_name]
        
        return results
    
    def profile_function_calls(self, func: F, sample_rate: float = 1.0) -> F:
        """Decorator to profile function calls with sampling."""
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError("Sample rate must be between 0.0 and 1.0")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Sample based on rate
            import random
            if random.random() > sample_rate:
                return func(*args, **kwargs)
            
            func_name = f"{func.__module__}.{func.__name__}"
            self.start_profiling(func_name)
            
            try:
                result = func(*args, **kwargs)
                self.checkpoint(func_name, "function_complete")
                return result
            finally:
                profile_results = self.end_profiling(func_name)
                if profile_results:
                    logger.debug(f"Function profiling: {func_name} took {profile_results['total_duration']:.4f}s")
        
        return wrapper


# Global profiler instance
performance_profiler = PerformanceProfiler()