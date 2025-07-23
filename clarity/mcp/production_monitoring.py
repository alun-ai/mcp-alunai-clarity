"""
Production Performance Monitoring for Enhanced MCP Discovery System.

This module provides comprehensive monitoring, alerting, and analytics
for production deployments of the enhanced MCP discovery system.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import threading
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """Defines alerting thresholds for monitoring metrics."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    evaluation_window_seconds: int
    consecutive_violations: int = 3


@dataclass
class Alert:
    """Represents a monitoring alert."""
    alert_id: str
    severity: str  # 'warning' or 'critical'
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: str
    resolved: bool = False
    resolution_time: Optional[str] = None


class ProductionMonitor:
    """Production monitoring system for enhanced MCP discovery."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize production monitor with configuration."""
        self.config = config or self._default_config()
        self.metrics_history = deque(maxlen=self.config['max_metrics_history'])
        self.alerts = {}
        self.alert_handlers = []
        self.monitoring_active = True
        
        # Performance thresholds
        self.thresholds = [
            AlertThreshold(
                metric_name="avg_response_time",
                warning_threshold=500.0,  # 500ms
                critical_threshold=1000.0,  # 1s
                evaluation_window_seconds=300,  # 5 minutes
                consecutive_violations=3
            ),
            AlertThreshold(
                metric_name="cache_hit_rate",
                warning_threshold=0.3,  # 30%
                critical_threshold=0.1,  # 10%
                evaluation_window_seconds=600,  # 10 minutes
                consecutive_violations=5
            ),
            AlertThreshold(
                metric_name="discovery_failure_rate",
                warning_threshold=0.2,  # 20%
                critical_threshold=0.5,  # 50%
                evaluation_window_seconds=600,  # 10 minutes
                consecutive_violations=3
            ),
            AlertThreshold(
                metric_name="memory_usage_mb",
                warning_threshold=500.0,  # 500MB
                critical_threshold=1000.0,  # 1GB
                evaluation_window_seconds=300,  # 5 minutes
                consecutive_violations=5
            )
        ]
        
        # Start monitoring thread
        self._start_monitoring_thread()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'max_metrics_history': 10000,
            'metrics_collection_interval': 60,  # 1 minute
            'alert_evaluation_interval': 30,    # 30 seconds
            'performance_report_interval': 300,  # 5 minutes
            'metrics_export_enabled': True,
            'metrics_export_path': '/tmp/mcp_discovery_metrics.jsonl',
            'alert_log_enabled': True,
            'alert_log_path': '/tmp/mcp_discovery_alerts.log'
        }
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric measurement."""
        timestamp = time.time()
        
        metric = {
            'timestamp': timestamp,
            'metric_name': metric_name,
            'value': value,
            'metadata': metadata or {},
            'datetime': datetime.fromtimestamp(timestamp).isoformat()
        }
        
        self.metrics_history.append(metric)
        
        # Export to file if enabled
        if self.config['metrics_export_enabled']:
            self._export_metric(metric)
    
    def record_discovery_operation(self, operation_type: str, duration_ms: float, 
                                 success: bool, metadata: Dict[str, Any] = None):
        """Record a discovery operation metric."""
        self.record_metric('discovery_operation', duration_ms, {
            'operation_type': operation_type,
            'success': success,
            'duration_ms': duration_ms,
            **(metadata or {})
        })
        
        # Track success/failure rates
        self.record_metric('discovery_success', 1.0 if success else 0.0, {
            'operation_type': operation_type
        })
    
    def record_cache_operation(self, operation: str, hit: bool = False):
        """Record cache operation metrics."""
        self.record_metric('cache_operation', 1.0, {
            'operation': operation,
            'hit': hit
        })
    
    def record_hook_analysis(self, hook_type: str, duration_ms: float, success: bool):
        """Record hook analysis metrics."""
        self.record_metric('hook_analysis', duration_ms, {
            'hook_type': hook_type,
            'success': success,
            'duration_ms': duration_ms
        })
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        if not self.metrics_history:
            return {}
        
        current_time = time.time()
        recent_window = current_time - 300  # Last 5 minutes
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m['timestamp'] > recent_window
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate aggregate metrics
        response_times = [
            m['value'] for m in recent_metrics 
            if m['metric_name'] == 'discovery_operation'
        ]
        
        cache_operations = [
            m for m in recent_metrics 
            if m['metric_name'] == 'cache_operation'
        ]
        
        discovery_operations = [
            m for m in recent_metrics 
            if m['metric_name'] == 'discovery_success'
        ]
        
        metrics = {
            'timestamp': current_time,
            'collection_period_seconds': 300,
            'total_operations': len(recent_metrics)
        }
        
        # Response time metrics
        if response_times:
            metrics['avg_response_time'] = sum(response_times) / len(response_times)
            metrics['max_response_time'] = max(response_times)
            metrics['min_response_time'] = min(response_times)
        else:
            metrics['avg_response_time'] = 0
        
        # Cache metrics
        if cache_operations:
            cache_hits = sum(1 for op in cache_operations if op['metadata'].get('hit', False))
            metrics['cache_hit_rate'] = cache_hits / len(cache_operations)
            metrics['cache_operations_count'] = len(cache_operations)
        else:
            metrics['cache_hit_rate'] = 0
            metrics['cache_operations_count'] = 0
        
        # Discovery success rate
        if discovery_operations:
            successes = sum(m['value'] for m in discovery_operations)
            metrics['discovery_success_rate'] = successes / len(discovery_operations)
            metrics['discovery_failure_rate'] = 1 - metrics['discovery_success_rate']
        else:
            metrics['discovery_success_rate'] = 0
            metrics['discovery_failure_rate'] = 0
        
        # Memory usage (if available)
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
        except ImportError:
            metrics['memory_usage_mb'] = 0
        
        return metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_metrics = self.get_current_metrics()
        
        # Get historical trends
        current_time = time.time()
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        hourly_metrics = [
            m for m in self.metrics_history 
            if m['timestamp'] > hour_ago
        ]
        
        daily_metrics = [
            m for m in self.metrics_history 
            if m['timestamp'] > day_ago
        ]
        
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'current_metrics': current_metrics,
            'hourly_summary': self._calculate_summary_metrics(hourly_metrics),
            'daily_summary': self._calculate_summary_metrics(daily_metrics),
            'active_alerts': [
                asdict(alert) for alert in self.alerts.values() 
                if not alert.resolved
            ],
            'system_health': self._assess_system_health(current_metrics),
            'recommendations': self._generate_recommendations(current_metrics)
        }
        
        return report
    
    def _calculate_summary_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary metrics for a time period."""
        if not metrics:
            return {}
        
        response_times = [
            m['value'] for m in metrics 
            if m['metric_name'] == 'discovery_operation'
        ]
        
        cache_operations = [
            m for m in metrics 
            if m['metric_name'] == 'cache_operation'
        ]
        
        discovery_operations = [
            m for m in metrics 
            if m['metric_name'] == 'discovery_success'
        ]
        
        summary = {
            'total_operations': len(metrics),
            'time_period_hours': (max(m['timestamp'] for m in metrics) - 
                                min(m['timestamp'] for m in metrics)) / 3600
        }
        
        if response_times:
            summary['avg_response_time'] = sum(response_times) / len(response_times)
            summary['p95_response_time'] = self._percentile(response_times, 95)
            summary['p99_response_time'] = self._percentile(response_times, 99)
        
        if cache_operations:
            cache_hits = sum(1 for op in cache_operations if op['metadata'].get('hit', False))
            summary['cache_hit_rate'] = cache_hits / len(cache_operations)
        
        if discovery_operations:
            successes = sum(m['value'] for m in discovery_operations)
            summary['discovery_success_rate'] = successes / len(discovery_operations)
        
        return summary
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _assess_system_health(self, metrics: Dict[str, Any]) -> str:
        """Assess overall system health based on metrics."""
        if not metrics:
            return 'unknown'
        
        health_score = 0
        total_checks = 0
        
        # Response time check
        if 'avg_response_time' in metrics:
            total_checks += 1
            if metrics['avg_response_time'] < 250:
                health_score += 1
            elif metrics['avg_response_time'] < 500:
                health_score += 0.5
        
        # Cache hit rate check
        if 'cache_hit_rate' in metrics:
            total_checks += 1
            if metrics['cache_hit_rate'] > 0.7:
                health_score += 1
            elif metrics['cache_hit_rate'] > 0.3:
                health_score += 0.5
        
        # Discovery success rate check
        if 'discovery_success_rate' in metrics:
            total_checks += 1
            if metrics['discovery_success_rate'] > 0.95:
                health_score += 1
            elif metrics['discovery_success_rate'] > 0.8:
                health_score += 0.5
        
        # Memory usage check
        if 'memory_usage_mb' in metrics and metrics['memory_usage_mb'] > 0:
            total_checks += 1
            if metrics['memory_usage_mb'] < 200:
                health_score += 1
            elif metrics['memory_usage_mb'] < 500:
                health_score += 0.5
        
        if total_checks == 0:
            return 'unknown'
        
        health_ratio = health_score / total_checks
        
        if health_ratio >= 0.8:
            return 'excellent'
        elif health_ratio >= 0.6:
            return 'good'
        elif health_ratio >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        # Response time recommendations
        if metrics.get('avg_response_time', 0) > 500:
            recommendations.append(
                "Consider optimizing server discovery timeouts or increasing parallel processing"
            )
        
        # Cache recommendations
        if metrics.get('cache_hit_rate', 0) < 0.3:
            recommendations.append(
                "Low cache hit rate - consider increasing cache TTL or reviewing cache key strategies"
            )
        
        # Discovery failure recommendations
        if metrics.get('discovery_failure_rate', 0) > 0.2:
            recommendations.append(
                "High discovery failure rate - verify server configurations and network connectivity"
            )
        
        # Memory recommendations
        if metrics.get('memory_usage_mb', 0) > 500:
            recommendations.append(
                "High memory usage - consider reducing metrics history or cache sizes"
            )
        
        return recommendations
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread."""
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Evaluate alerts
                    self._evaluate_alerts()
                    
                    # Generate periodic performance report
                    if int(time.time()) % self.config['performance_report_interval'] == 0:
                        report = self.get_performance_report()
                        logger.info(f"Performance Report: Health={report['system_health']}")
                    
                    time.sleep(self.config['alert_evaluation_interval'])
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(30)  # Wait before retrying
        
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
    
    def _evaluate_alerts(self):
        """Evaluate alert thresholds and trigger alerts."""
        current_metrics = self.get_current_metrics()
        
        for threshold in self.thresholds:
            if threshold.metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[threshold.metric_name]
            
            # Check if threshold is violated
            warning_violated = (
                (threshold.metric_name in ['cache_hit_rate', 'discovery_success_rate'] and 
                 current_value < threshold.warning_threshold) or
                (threshold.metric_name not in ['cache_hit_rate', 'discovery_success_rate'] and 
                 current_value > threshold.warning_threshold)
            )
            
            critical_violated = (
                (threshold.metric_name in ['cache_hit_rate', 'discovery_success_rate'] and 
                 current_value < threshold.critical_threshold) or
                (threshold.metric_name not in ['cache_hit_rate', 'discovery_success_rate'] and 
                 current_value > threshold.critical_threshold)
            )
            
            # Trigger alerts
            if critical_violated:
                self._trigger_alert(threshold, current_value, 'critical')
            elif warning_violated:
                self._trigger_alert(threshold, current_value, 'warning')
            else:
                # Check if alert should be resolved
                self._resolve_alert(threshold.metric_name)
    
    def _trigger_alert(self, threshold: AlertThreshold, current_value: float, severity: str):
        """Trigger an alert for threshold violation."""
        alert_id = f"{threshold.metric_name}_{severity}"
        
        # Don't create duplicate alerts
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            metric_name=threshold.metric_name,
            current_value=current_value,
            threshold=threshold.critical_threshold if severity == 'critical' else threshold.warning_threshold,
            message=f"{threshold.metric_name} {severity}: {current_value:.2f} exceeds threshold {threshold.warning_threshold:.2f}",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        self.alerts[alert_id] = alert
        
        # Log alert
        if self.config['alert_log_enabled']:
            self._log_alert(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _resolve_alert(self, metric_name: str):
        """Resolve alerts for a metric that's no longer violating thresholds."""
        for alert_id, alert in self.alerts.items():
            if alert.metric_name == metric_name and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now(timezone.utc).isoformat()
                
                if self.config['alert_log_enabled']:
                    self._log_alert_resolution(alert)
    
    def _export_metric(self, metric: Dict[str, Any]):
        """Export metric to file."""
        try:
            metrics_file = Path(self.config['metrics_export_path'])
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metric) + '\\n')
        except Exception as e:
            logger.debug(f"Failed to export metric: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to file."""
        try:
            alert_file = Path(self.config['alert_log_path'])
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(alert_file, 'a') as f:
                f.write(f"{alert.timestamp} [{alert.severity.upper()}] {alert.message}\\n")
        except Exception as e:
            logger.debug(f"Failed to log alert: {e}")
    
    def _log_alert_resolution(self, alert: Alert):
        """Log alert resolution to file."""
        try:
            alert_file = Path(self.config['alert_log_path'])
            with open(alert_file, 'a') as f:
                f.write(f"{alert.resolution_time} [RESOLVED] {alert.message}\\n")
        except Exception as e:
            logger.debug(f"Failed to log alert resolution: {e}")
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring_active = False


# Global production monitor instance
_production_monitor = None


def get_production_monitor(config: Dict[str, Any] = None) -> ProductionMonitor:
    """Get or create global production monitor instance."""
    global _production_monitor
    if _production_monitor is None:
        _production_monitor = ProductionMonitor(config)
    return _production_monitor


def record_metric(metric_name: str, value: float, metadata: Dict[str, Any] = None):
    """Convenience function to record metrics."""
    monitor = get_production_monitor()
    monitor.record_metric(metric_name, value, metadata)


def record_discovery_operation(operation_type: str, duration_ms: float, 
                             success: bool, metadata: Dict[str, Any] = None):
    """Convenience function to record discovery operations."""
    monitor = get_production_monitor()
    monitor.record_discovery_operation(operation_type, duration_ms, success, metadata)


# Example alert handlers
def console_alert_handler(alert: Alert):
    """Example alert handler that prints to console."""
    severity_icon = "🚨" if alert.severity == "critical" else "⚠️"
    print(f"{severity_icon} {alert.severity.upper()}: {alert.message}")


def email_alert_handler(alert: Alert):
    """Example email alert handler (would need SMTP configuration)."""
    # This is a placeholder - implement actual email sending
    logger.warning(f"EMAIL ALERT: {alert.severity.upper()} - {alert.message}")


# Example usage and integration
class MonitoredMCPToolIndexer:
    """Example wrapper that adds monitoring to MCPToolIndexer."""
    
    def __init__(self, indexer, monitor: ProductionMonitor = None):
        self.indexer = indexer
        self.monitor = monitor or get_production_monitor()
        
        # Add console alert handler by default
        self.monitor.add_alert_handler(console_alert_handler)
    
    async def discover_and_index_tools(self):
        """Monitored version of discover_and_index_tools."""
        start_time = time.time()
        success = False
        
        try:
            result = await self.indexer.discover_and_index_tools()
            success = True
            return result
            
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            raise
            
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.monitor.record_discovery_operation(
                'complete_discovery', 
                duration_ms, 
                success,
                {'tools_discovered': len(result) if success else 0}
            )