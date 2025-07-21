"""
Telemetry and observability reporting for Alunai Clarity.

Provides structured logging, metrics export, and health reporting.
"""

import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from .metrics import get_metrics_collector


class TelemetryReporter:
    """Centralized telemetry and observability reporting."""
    
    def __init__(self, export_path: Optional[Path] = None):
        self.metrics_collector = get_metrics_collector()
        self.export_path = export_path or Path("./.claude/alunai-clarity/telemetry")
        self.export_path.mkdir(parents=True, exist_ok=True)
        
        self._last_export = time.time()
        
        logger.debug(f"TelemetryReporter initialized, export path: {self.export_path}")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        metrics = self.metrics_collector.get_metrics()
        current_time = datetime.utcnow()
        
        report = {
            "timestamp": current_time.isoformat(),
            "system_info": {
                "uptime_seconds": metrics.get("uptime_seconds", 0),
                "uptime_human": str(timedelta(seconds=int(metrics.get("uptime_seconds", 0)))),
            },
            "health_status": "healthy",  # Will be updated based on checks
            "component_health": {},
            "performance_summary": {},
            "alerts": []
        }
        
        # Analyze component health
        operation_metrics = metrics.get("operations", {})
        for op_name, op_stats in operation_metrics.items():
            component = self._extract_component_from_operation(op_name)
            
            if component not in report["component_health"]:
                report["component_health"][component] = {
                    "status": "healthy",
                    "operations": 0,
                    "error_rate": 0.0,
                    "avg_response_time": 0.0,
                    "issues": []
                }
            
            comp_health = report["component_health"][component]
            comp_health["operations"] += op_stats.get("count", 0)
            
            # Check error rates
            error_rate = 1.0 - op_stats.get("success_rate", 1.0)
            if error_rate > comp_health["error_rate"]:
                comp_health["error_rate"] = error_rate
            
            # Check response times
            avg_duration = op_stats.get("avg_duration", 0.0)
            if avg_duration > comp_health["avg_response_time"]:
                comp_health["avg_response_time"] = avg_duration
            
            # Health checks
            if error_rate > 0.05:  # >5% error rate
                comp_health["status"] = "degraded"
                comp_health["issues"].append(f"High error rate: {error_rate:.1%}")
                report["alerts"].append({
                    "severity": "warning",
                    "component": component,
                    "message": f"High error rate: {error_rate:.1%}",
                    "metric": "error_rate",
                    "value": error_rate
                })
            
            if avg_duration > 5.0:  # >5s average response time
                comp_health["status"] = "degraded" 
                comp_health["issues"].append(f"Slow response time: {avg_duration:.2f}s")
                report["alerts"].append({
                    "severity": "warning", 
                    "component": component,
                    "message": f"Slow response time: {avg_duration:.2f}s",
                    "metric": "response_time",
                    "value": avg_duration
                })
        
        # Overall system health
        degraded_components = [
            comp for comp, health in report["component_health"].items() 
            if health["status"] != "healthy"
        ]
        
        if degraded_components:
            report["health_status"] = "degraded"
        
        # Performance summary
        if operation_metrics:
            all_operations = list(operation_metrics.values())
            total_ops = sum(op.get("count", 0) for op in all_operations)
            total_errors = sum(op.get("error_count", 0) for op in all_operations)
            avg_response_times = [op.get("avg_duration", 0) for op in all_operations if op.get("avg_duration")]
            
            report["performance_summary"] = {
                "total_operations": total_ops,
                "total_errors": total_errors,
                "overall_error_rate": total_errors / total_ops if total_ops > 0 else 0.0,
                "avg_response_time": sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0.0,
                "operations_per_second": total_ops / metrics.get("uptime_seconds", 1),
                "top_operations": self.metrics_collector.get_top_operations(5, "count")
            }
        
        return report
    
    def export_metrics_json(self, filename: Optional[str] = None) -> Path:
        """Export metrics to JSON file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        export_file = self.export_path / filename
        metrics = self.metrics_collector.get_metrics()
        
        with open(export_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {export_file}")
        return export_file
    
    def export_health_report(self, filename: Optional[str] = None) -> Path:
        """Export health report to JSON file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"health_report_{timestamp}.json"
        
        export_file = self.export_path / filename
        report = self.generate_health_report()
        
        with open(export_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Health report exported to {export_file}")
        return export_file
    
    def export_prometheus_metrics(self, filename: Optional[str] = None) -> Path:
        """Export metrics in Prometheus format."""
        if filename is None:
            filename = "metrics.prom"
        
        export_file = self.export_path / filename
        prometheus_data = self.metrics_collector.export_prometheus()
        
        with open(export_file, 'w') as f:
            f.write(prometheus_data)
        
        logger.info(f"Prometheus metrics exported to {export_file}")
        return export_file
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights and recommendations."""
        metrics = self.metrics_collector.get_metrics()
        operation_metrics = metrics.get("operations", {})
        
        insights = {
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": [],
            "performance_issues": [],
            "optimization_opportunities": []
        }
        
        # Analyze operation performance
        for op_name, op_stats in operation_metrics.items():
            avg_duration = op_stats.get("avg_duration", 0)
            error_rate = 1.0 - op_stats.get("success_rate", 1.0)
            count = op_stats.get("count", 0)
            
            # Slow operations
            if avg_duration > 1.0 and count > 10:
                insights["performance_issues"].append({
                    "type": "slow_operation",
                    "operation": op_name,
                    "avg_duration": avg_duration,
                    "description": f"Operation {op_name} is slow (avg: {avg_duration:.2f}s)"
                })
                
                insights["recommendations"].append({
                    "type": "performance",
                    "priority": "high" if avg_duration > 5.0 else "medium",
                    "operation": op_name,
                    "recommendation": "Consider adding caching or optimizing database queries"
                })
            
            # High error rates
            if error_rate > 0.1 and count > 5:  # >10% error rate
                insights["performance_issues"].append({
                    "type": "high_error_rate",
                    "operation": op_name,
                    "error_rate": error_rate,
                    "description": f"Operation {op_name} has high error rate ({error_rate:.1%})"
                })
                
                insights["recommendations"].append({
                    "type": "reliability", 
                    "priority": "high",
                    "operation": op_name,
                    "recommendation": "Investigate error causes and add better error handling"
                })
            
            # High-frequency operations that could benefit from caching
            throughput = op_stats.get("throughput", 0)
            if throughput > 10 and avg_duration > 0.1:  # >10 ops/sec, >100ms duration
                insights["optimization_opportunities"].append({
                    "type": "caching_opportunity",
                    "operation": op_name,
                    "throughput": throughput,
                    "avg_duration": avg_duration,
                    "description": f"High-frequency operation {op_name} could benefit from caching"
                })
        
        # Analyze cache effectiveness
        cache_metrics = metrics.get("histograms", {})
        for key, hist_data in cache_metrics.items():
            if "cache" in key.lower() and hist_data.get("count", 0) > 0:
                # This would need cache-specific metrics to provide meaningful insights
                pass
        
        return insights
    
    def schedule_periodic_export(self, interval_minutes: int = 60) -> None:
        """Schedule periodic export of metrics and health reports."""
        current_time = time.time()
        interval_seconds = interval_minutes * 60
        
        if current_time - self._last_export >= interval_seconds:
            try:
                self.export_metrics_json()
                self.export_health_report()
                self._last_export = current_time
                
                logger.info(f"Periodic telemetry export completed (interval: {interval_minutes}m)")
            except (OSError, ValueError, AttributeError, RuntimeError) as e:
                logger.error(f"Failed to export periodic telemetry: {e}")
    
    def _extract_component_from_operation(self, operation_name: str) -> str:
        """Extract component name from operation name."""
        # Try to extract component from operation name patterns
        if "memory" in operation_name.lower():
            return "memory"
        elif "autocode" in operation_name.lower():
            return "autocode" 
        elif "mcp" in operation_name.lower():
            return "mcp"
        elif "persistence" in operation_name.lower():
            return "persistence"
        elif "qdrant" in operation_name.lower():
            return "database"
        elif "embedding" in operation_name.lower():
            return "embeddings"
        else:
            # Extract from module path if available
            parts = operation_name.split(".")
            if len(parts) >= 2:
                return parts[1]  # Usually clarity.domain.function
            else:
                return "system"


class HealthChecker:
    """System health monitoring and alerting."""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        self.metrics_collector = get_metrics_collector()
        self.alert_thresholds = alert_thresholds or {
            "max_error_rate": 0.05,      # 5%
            "max_response_time": 5.0,     # 5 seconds
            "min_success_rate": 0.95,     # 95%
            "max_memory_mb": 1024,        # 1GB
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "overall_status": "healthy",
            "checks": {},
            "alerts": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check operation health
        top_operations = self.metrics_collector.get_top_operations(10, "count")
        
        for op_stats in top_operations:
            op_name = op_stats["operation"]
            
            # Error rate check
            error_rate = 1.0 - op_stats["success_rate"]
            if error_rate > self.alert_thresholds["max_error_rate"]:
                health_status["checks"][f"{op_name}_error_rate"] = "failed"
                health_status["alerts"].append({
                    "severity": "critical",
                    "check": "error_rate", 
                    "operation": op_name,
                    "value": error_rate,
                    "threshold": self.alert_thresholds["max_error_rate"],
                    "message": f"High error rate: {error_rate:.1%}"
                })
                health_status["overall_status"] = "degraded"
            else:
                health_status["checks"][f"{op_name}_error_rate"] = "passed"
            
            # Response time check
            avg_duration = op_stats["duration_stats"]["mean"]
            if avg_duration > self.alert_thresholds["max_response_time"]:
                health_status["checks"][f"{op_name}_response_time"] = "failed"
                health_status["alerts"].append({
                    "severity": "warning",
                    "check": "response_time",
                    "operation": op_name,
                    "value": avg_duration,
                    "threshold": self.alert_thresholds["max_response_time"], 
                    "message": f"Slow response time: {avg_duration:.2f}s"
                })
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"
            else:
                health_status["checks"][f"{op_name}_response_time"] = "passed"
        
        return health_status
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts."""
        health_check = self.check_system_health()
        alerts = health_check.get("alerts", [])
        
        return {
            "total_alerts": len(alerts),
            "critical_count": len([a for a in alerts if a["severity"] == "critical"]),
            "warning_count": len([a for a in alerts if a["severity"] == "warning"]),
            "overall_status": health_check["overall_status"],
            "alerts": alerts
        }


# Global telemetry reporter instance
_telemetry_reporter: Optional[TelemetryReporter] = None


def get_telemetry_reporter() -> TelemetryReporter:
    """Get global telemetry reporter instance."""
    global _telemetry_reporter
    
    if _telemetry_reporter is None:
        _telemetry_reporter = TelemetryReporter()
    
    return _telemetry_reporter