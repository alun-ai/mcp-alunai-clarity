"""
Metrics collection and aggregation system for Alunai Clarity.

Provides lightweight metrics collection with minimal performance overhead.
"""

import threading
import time
from typing import Dict, Any, Optional, List, DefaultDict
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

from loguru import logger


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CounterMetric:
    """Counter metric that only increases."""
    name: str
    value: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class GaugeMetric:
    """Gauge metric that can increase or decrease."""
    name: str
    value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


@dataclass
class HistogramMetric:
    """Histogram metric for tracking distributions."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_value(self, value: float) -> None:
        """Add a value to the histogram."""
        self.values.append(MetricPoint(time.time(), value))
    
    def get_percentiles(self, percentiles: List[float] = None) -> Dict[str, float]:
        """Get percentile values."""
        if not self.values:
            return {}
        
        percentiles = percentiles or [50.0, 90.0, 95.0, 99.0]
        values = [point.value for point in self.values]
        
        return {
            f"p{int(p)}": statistics.quantiles(values, n=100)[int(p)-1] if len(values) >= int(p) else 0.0
            for p in percentiles
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get basic statistics."""
        if not self.values:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "stddev": 0.0}
        
        values = [point.value for point in self.values]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0
        }


class MetricsCollector:
    """Thread-safe metrics collector with minimal overhead."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self._lock = threading.RLock()
        
        # Metric storage
        self._counters: Dict[str, CounterMetric] = {}
        self._gauges: Dict[str, GaugeMetric] = {}
        self._histograms: Dict[str, HistogramMetric] = {}
        
        # Performance tracking
        self._operation_times: DefaultDict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._operation_counts: DefaultDict[str, int] = defaultdict(int)
        self._error_counts: DefaultDict[str, int] = defaultdict(int)
        
        # System metrics
        self._start_time = time.time()
        self._last_collection_time = time.time()
        
        logger.debug("MetricsCollector initialized")
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, tags)
            
            if key not in self._counters:
                self._counters[key] = CounterMetric(name, 0, tags or {})
            
            self._counters[key].value += value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, tags)
            
            if key not in self._gauges:
                self._gauges[key] = GaugeMetric(name, value, tags or {})
            else:
                self._gauges[key].value = value
                self._gauges[key].updated_at = time.time()
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric value."""
        with self._lock:
            key = self._make_key(name, tags)
            
            if key not in self._histograms:
                self._histograms[key] = HistogramMetric(name, deque(maxlen=1000), tags or {})
            
            self._histograms[key].add_value(value)
    
    def time_operation(self, operation_name: str, duration: float, success: bool = True) -> None:
        """Record operation timing and success/failure."""
        with self._lock:
            self._operation_times[operation_name].append(MetricPoint(time.time(), duration))
            self._operation_counts[operation_name] += 1
            
            if not success:
                self._error_counts[operation_name] += 1
            
            # Also record as histogram
            self.record_histogram("operation_duration", duration, {"operation": operation_name})
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self._start_time,
                "counters": {},
                "gauges": {},
                "histograms": {},
                "operations": {}
            }
            
            # Counters
            for key, counter in self._counters.items():
                metrics["counters"][key] = {
                    "name": counter.name,
                    "value": counter.value,
                    "tags": counter.tags,
                    "created_at": counter.created_at
                }
            
            # Gauges
            for key, gauge in self._gauges.items():
                metrics["gauges"][key] = {
                    "name": gauge.name,
                    "value": gauge.value,
                    "tags": gauge.tags,
                    "updated_at": gauge.updated_at
                }
            
            # Histograms
            for key, histogram in self._histograms.items():
                stats = histogram.get_stats()
                percentiles = histogram.get_percentiles()
                
                metrics["histograms"][key] = {
                    "name": histogram.name,
                    "tags": histogram.tags,
                    **stats,
                    **percentiles
                }
            
            # Operation metrics
            for op_name, times in self._operation_times.items():
                if not times:
                    continue
                    
                durations = [point.value for point in times]
                total_count = self._operation_counts[op_name]
                error_count = self._error_counts[op_name]
                
                metrics["operations"][op_name] = {
                    "count": total_count,
                    "error_count": error_count,
                    "success_rate": (total_count - error_count) / total_count if total_count > 0 else 0.0,
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "p95_duration": statistics.quantiles(durations, n=20)[18] if len(durations) >= 19 else max(durations),
                    "throughput": total_count / (time.time() - self._start_time)
                }
            
            return metrics
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get detailed stats for a specific operation."""
        with self._lock:
            if operation_name not in self._operation_times:
                return {"error": f"No data for operation: {operation_name}"}
            
            times = self._operation_times[operation_name]
            if not times:
                return {"error": f"No timing data for operation: {operation_name}"}
            
            durations = [point.value for point in times]
            total_count = self._operation_counts[operation_name]
            error_count = self._error_counts[operation_name]
            
            return {
                "operation": operation_name,
                "count": total_count,
                "error_count": error_count,
                "success_rate": (total_count - error_count) / total_count,
                "duration_stats": {
                    "mean": statistics.mean(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "stddev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                    "p50": statistics.median(durations),
                    "p90": statistics.quantiles(durations, n=10)[8] if len(durations) >= 9 else max(durations),
                    "p95": statistics.quantiles(durations, n=20)[18] if len(durations) >= 19 else max(durations),
                    "p99": statistics.quantiles(durations, n=100)[98] if len(durations) >= 99 else max(durations),
                },
                "throughput": total_count / (time.time() - self._start_time),
                "sample_count": len(durations)
            }
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._operation_times.clear()
            self._operation_counts.clear()
            self._error_counts.clear()
            self._start_time = time.time()
            
            logger.info("Metrics collector reset")
    
    def get_top_operations(self, limit: int = 10, sort_by: str = "count") -> List[Dict[str, Any]]:
        """Get top operations by count, duration, or error rate."""
        with self._lock:
            operations = []
            
            for op_name in self._operation_counts.keys():
                stats = self.get_operation_stats(op_name)
                if "error" not in stats:
                    operations.append(stats)
            
            # Sort operations
            if sort_by == "count":
                operations.sort(key=lambda x: x["count"], reverse=True)
            elif sort_by == "duration":
                operations.sort(key=lambda x: x["duration_stats"]["mean"], reverse=True)
            elif sort_by == "errors":
                operations.sort(key=lambda x: x["error_count"], reverse=True)
            elif sort_by == "error_rate":
                operations.sort(key=lambda x: 1.0 - x["success_rate"], reverse=True)
            
            return operations[:limit]
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric identification."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Counters
            for key, counter in self._counters.items():
                lines.append(f"# TYPE {counter.name} counter")
                tag_str = ""
                if counter.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in counter.tags.items()]
                    tag_str = "{" + ",".join(tag_pairs) + "}"
                lines.append(f"{counter.name}{tag_str} {counter.value}")
            
            # Gauges
            for key, gauge in self._gauges.items():
                lines.append(f"# TYPE {gauge.name} gauge")
                tag_str = ""
                if gauge.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in gauge.tags.items()]
                    tag_str = "{" + ",".join(tag_pairs) + "}"
                lines.append(f"{gauge.name}{tag_str} {gauge.value}")
            
            # System metrics
            lines.append(f"# TYPE alunai_clarity_uptime_seconds gauge")
            lines.append(f"alunai_clarity_uptime_seconds {time.time() - self._start_time}")
            
            return "\n".join(lines) + "\n"


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector."""
    global _metrics_collector
    
    with _collector_lock:
        if _metrics_collector is not None:
            _metrics_collector.reset_metrics()


# Convenience functions
def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter metric."""
    get_metrics_collector().increment_counter(name, value, tags)


def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge metric value."""
    get_metrics_collector().set_gauge(name, value, tags)


def record_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a histogram metric value."""
    get_metrics_collector().record_histogram(name, value, tags)


def time_operation(operation_name: str, duration: float, success: bool = True) -> None:
    """Record operation timing and success/failure."""
    get_metrics_collector().time_operation(operation_name, duration, success)