"""
Enhanced observability utilities for monitoring, tracing, and metrics collection.

This module provides:
- Distributed tracing capabilities
- Metrics collection and aggregation
- Health monitoring
- Performance tracking
- Custom instrumentation
"""

import time
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from enum import Enum

from .utils.logging_utils import get_logger
from .audit_trail import AuditEventType, AuditSeverity, get_audit_manager


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(Enum):
    """Health check statuses"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class TraceSpan:
    """Distributed tracing span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "started"
    error: Optional[str] = None


@dataclass
class Metric:
    """Metric data point"""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TracingManager:
    """Distributed tracing manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self._active_spans: Dict[str, TraceSpan] = {}
        self._completed_spans: deque = deque(maxlen=1000)  # Keep last 1000 spans
        self._sampling_rate = self.config.get('sampling_rate', 1.0)
        self._enabled = self.config.get('enabled', True)
    
    def create_trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> str:
        """Create new trace
        
        Args:
            operation_name: Name of the operation being traced
            tags: Optional tags for the trace
            
        Returns:
            Trace ID
        """
        if not self._enabled or not self._should_sample():
            return ""
        
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._active_spans[span_id] = span
        return span_id
    
    def create_child_span(self, parent_span_id: str, operation_name: str, 
                         tags: Optional[Dict[str, Any]] = None) -> str:
        """Create child span
        
        Args:
            parent_span_id: Parent span ID
            operation_name: Child operation name
            tags: Optional tags
            
        Returns:
            Child span ID
        """
        if not self._enabled or parent_span_id not in self._active_spans:
            return ""
        
        parent_span = self._active_spans[parent_span_id]
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=parent_span.trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._active_spans[span_id] = span
        return span_id
    
    def finish_span(self, span_id: str, status: str = "completed", 
                   error: Optional[str] = None) -> None:
        """Finish a span
        
        Args:
            span_id: Span to finish
            status: Final status
            error: Error message if failed
        """
        if span_id not in self._active_spans:
            return
        
        span = self._active_spans[span_id]
        span.end_time = datetime.now(timezone.utc)
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        span.error = error
        
        # Move to completed spans
        self._completed_spans.append(span)
        del self._active_spans[span_id]
        
        self.logger.debug(f"Finished span: {span.operation_name}", context={
            'span_id': span_id,
            'trace_id': span.trace_id,
            'duration_ms': span.duration_ms,
            'status': status
        })
    
    def add_span_tag(self, span_id: str, key: str, value: Any) -> None:
        """Add tag to span"""
        if span_id in self._active_spans:
            self._active_spans[span_id].tags[key] = value
    
    def add_span_log(self, span_id: str, message: str, 
                    fields: Optional[Dict[str, Any]] = None) -> None:
        """Add log entry to span"""
        if span_id in self._active_spans:
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': message,
                'fields': fields or {}
            }
            self._active_spans[span_id].logs.append(log_entry)
    
    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled"""
        import random
        return random.random() < self._sampling_rate
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, 
                             parent_span_id: Optional[str] = None,
                             tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        if parent_span_id:
            span_id = self.create_child_span(parent_span_id, operation_name, tags)
        else:
            span_id = self.create_trace(operation_name, tags)
        
        if not span_id:
            yield None
            return
        
        try:
            yield span_id
            self.finish_span(span_id, "completed")
        except Exception as e:
            self.finish_span(span_id, "error", str(e))
            raise
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a trace"""
        # Find all spans for this trace
        trace_spans = []
        
        # Check active spans
        for span in self._active_spans.values():
            if span.trace_id == trace_id:
                trace_spans.append(span)
        
        # Check completed spans
        for span in self._completed_spans:
            if span.trace_id == trace_id:
                trace_spans.append(span)
        
        if not trace_spans:
            return None
        
        # Calculate trace metrics
        start_time = min(span.start_time for span in trace_spans)
        end_times = [span.end_time for span in trace_spans if span.end_time]
        end_time = max(end_times) if end_times else None
        
        total_duration = (end_time - start_time).total_seconds() * 1000 if end_time else None
        
        return {
            'trace_id': trace_id,
            'span_count': len(trace_spans),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat() if end_time else None,
            'total_duration_ms': total_duration,
            'spans': [
                {
                    'span_id': span.span_id,
                    'operation_name': span.operation_name,
                    'duration_ms': span.duration_ms,
                    'status': span.status,
                    'tags': span.tags
                }
                for span in trace_spans
            ]
        }


class MetricsCollector:
    """Metrics collection and aggregation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._max_history = self.config.get('max_metric_history', 1000)
        self._enabled = self.config.get('enabled', True)
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        if not self._enabled:
            return
        
        metric_key = self._get_metric_key(name, tags)
        self._counters[metric_key] += value
        
        metric = Metric(
            name=name,
            metric_type=MetricType.COUNTER,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._add_metric(name, metric)
    
    def set_gauge(self, name: str, value: float, 
                 tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        if not self._enabled:
            return
        
        metric_key = self._get_metric_key(name, tags)
        self._gauges[metric_key] = value
        
        metric = Metric(
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._add_metric(name, metric)
    
    def record_histogram(self, name: str, value: float, 
                        tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        if not self._enabled:
            return
        
        metric_key = self._get_metric_key(name, tags)
        self._histograms[metric_key].append(value)
        
        # Keep only recent values
        if len(self._histograms[metric_key]) > self._max_history:
            self._histograms[metric_key] = self._histograms[metric_key][-self._max_history:]
        
        metric = Metric(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        self._add_metric(name, metric)
    
    def record_timer(self, name: str, duration_ms: float, 
                    tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer value"""
        if not self._enabled:
            return
        
        metric_key = self._get_metric_key(name, tags)
        self._timers[metric_key].append(duration_ms)
        
        # Keep only recent values
        if len(self._timers[metric_key]) > self._max_history:
            self._timers[metric_key] = self._timers[metric_key][-self._max_history:]
        
        metric = Metric(
            name=name,
            metric_type=MetricType.TIMER,
            value=duration_ms,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {},
            unit="ms"
        )
        
        self._add_metric(name, metric)
    
    @asynccontextmanager
    async def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_timer(name, duration_ms, tags)
    
    def _get_metric_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Generate metric key including tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}#{tag_str}"
    
    def _add_metric(self, name: str, metric: Metric) -> None:
        """Add metric to history"""
        self._metrics[name].append(metric)
        
        # Keep only recent metrics
        if len(self._metrics[name]) > self._max_history:
            self._metrics[name] = self._metrics[name][-self._max_history:]
    
    def get_counter_value(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value"""
        metric_key = self._get_metric_key(name, tags)
        return self._counters.get(metric_key, 0.0)
    
    def get_gauge_value(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value"""
        metric_key = self._get_metric_key(name, tags)
        return self._gauges.get(metric_key)
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[Dict[str, float]]:
        """Get histogram statistics"""
        metric_key = self._get_metric_key(name, tags)
        values = self._histograms.get(metric_key, [])
        
        if not values:
            return None
        
        values_sorted = sorted(values)
        count = len(values)
        
        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / count,
            'p50': values_sorted[int(count * 0.5)],
            'p95': values_sorted[int(count * 0.95)],
            'p99': values_sorted[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[Dict[str, float]]:
        """Get timer statistics"""
        metric_key = self._get_metric_key(name, tags)
        durations = self._timers.get(metric_key, [])
        
        if not durations:
            return None
        
        durations_sorted = sorted(durations)
        count = len(durations)
        
        return {
            'count': count,
            'min_ms': min(durations),
            'max_ms': max(durations),
            'mean_ms': sum(durations) / count,
            'p50_ms': durations_sorted[int(count * 0.5)],
            'p95_ms': durations_sorted[int(count * 0.95)],
            'p99_ms': durations_sorted[int(count * 0.99)]
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {k: self.get_histogram_stats(k.split('#')[0], None) 
                          for k in self._histograms.keys()},
            'timers': {k: self.get_timer_stats(k.split('#')[0], None) 
                      for k in self._timers.keys()},
            'collection_time': datetime.now(timezone.utc).isoformat()
        }


class HealthMonitor:
    """Health monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self._health_checks: Dict[str, Callable] = {}
        self._health_results: Dict[str, HealthCheck] = {}
        self._enabled = self.config.get('enabled', True)
        self._check_interval = self.config.get('check_interval_seconds', 60)
        self._running = False
    
    def register_health_check(self, name: str, check_func: Callable,
                             timeout_seconds: float = 30.0) -> None:
        """Register a health check function
        
        Args:
            name: Name of the health check
            check_func: Function that performs the check
            timeout_seconds: Timeout for the check
        """
        self._health_checks[name] = {
            'func': check_func,
            'timeout': timeout_seconds
        }
        
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_check(self, name: str) -> HealthCheck:
        """Run a specific health check
        
        Args:
            name: Name of the health check to run
            
        Returns:
            Health check result
        """
        if name not in self._health_checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check not found",
                timestamp=datetime.now(timezone.utc),
                duration_ms=0.0
            )
        
        check_config = self._health_checks[name]
        start_time = time.time()
        
        try:
            # Run with timeout
            result = await asyncio.wait_for(
                check_config['func'](),
                timeout=check_config['timeout']
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                message = result.get('message', 'Check passed')
                metadata = result.get('metadata', {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Check passed" if result else "Check failed"
                metadata = {}
            
            health_check = HealthCheck(
                name=name,
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                metadata=metadata
            )
            
            self._health_results[name] = health_check
            return health_check
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            health_check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check_config['timeout']}s",
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms
            )
            self._health_results[name] = health_check
            return health_check
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            health_check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms
            )
            self._health_results[name] = health_check
            return health_check
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        if not self._enabled:
            return {}
        
        results = {}
        for name in self._health_checks:
            results[name] = await self.run_health_check(name)
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status"""
        if not self._health_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self._health_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif any(status == HealthStatus.UNKNOWN for status in statuses):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        return {
            'overall_status': self.get_overall_health().value,
            'checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'timestamp': result.timestamp.isoformat(),
                    'duration_ms': result.duration_ms
                }
                for name, result in self._health_results.items()
            },
            'last_check': datetime.now(timezone.utc).isoformat()
        }


class ObservabilityManager:
    """Main observability manager that coordinates tracing, metrics, and health monitoring"""
    
    _instance = None
    
    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, '_initialized'):
            return
        
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Initialize subsystems
        self.tracing = TracingManager(self.config.get('tracing', {}))
        self.metrics = MetricsCollector(self.config.get('metrics', {}))
        self.health = HealthMonitor(self.config.get('health', {}))
        
        # Initialize audit integration
        self.audit_manager = get_audit_manager()
        
        self._initialized = True
        self.logger.info("Observability manager initialized")
    
    @asynccontextmanager
    async def observe_operation(self, 
                               operation_name: str,
                               actor: str = "system",
                               tags: Optional[Dict[str, Any]] = None,
                               audit_event_type: Optional[AuditEventType] = None):
        """Comprehensive operation observation with tracing, metrics, and auditing"""
        
        # Start tracing
        async with self.tracing.trace_operation(operation_name, tags=tags) as span_id:
            # Start timing
            async with self.metrics.time_operation(f"operation.{operation_name}"):
                # Start auditing if requested
                if audit_event_type:
                    async with self.audit_manager.audit_operation(
                        audit_event_type, actor, operation_name, operation_name
                    ):
                        yield span_id
                else:
                    yield span_id
                
                # Increment operation counter
                self.metrics.increment_counter(f"operations.{operation_name}.count")


# Global observability manager instance
_observability_instance = None

def get_observability_manager(config: Optional[Dict[str, Any]] = None) -> ObservabilityManager:
    """Get global observability manager instance"""
    global _observability_instance
    if _observability_instance is None:
        _observability_instance = ObservabilityManager(config)
    return _observability_instance


# Convenience functions and decorators
def trace_operation(operation_name: str, tags: Optional[Dict[str, Any]] = None):
    """Trace operation using global manager"""
    manager = get_observability_manager()
    return manager.tracing.trace_operation(operation_name, tags=tags)


def time_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Time operation using global manager"""
    manager = get_observability_manager()
    return manager.metrics.time_operation(operation_name, tags)


def observe_operation(operation_name: str, **kwargs):
    """Observe operation using global manager"""
    manager = get_observability_manager()
    return manager.observe_operation(operation_name, **kwargs)