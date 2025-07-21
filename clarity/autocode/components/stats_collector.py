"""
Statistics collector component for AutoCode domain.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from clarity.shared.infrastructure import get_cache
from clarity.shared.exceptions import AutoCodeError
from ..interfaces import StatsCollector, AutoCodeComponent


class StatsCollectorImpl(StatsCollector, AutoCodeComponent):
    """Implementation of statistics collection"""
    
    def __init__(self, config: Dict[str, Any], persistence_domain):
        """Initialize stats collector"""
        self.config = config
        self.persistence_domain = persistence_domain
        self.autocode_config = config.get("autocode", {})
        
        # Performance tracking
        self._operation_stats = {}
        self._performance_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "learning_accuracy": 0.0
        }
        
        # Component references for collecting stats
        self._components = {}
        self._initialized = False
        self._stats_cache = get_cache(
            "autocode_stats",
            max_size=100,
            max_memory_mb=10,
            default_ttl=300.0  # 5 minutes TTL
        )
    
    async def initialize(self) -> None:
        """Initialize the stats collector"""
        if self._initialized:
            return
            
        logger.info("Initializing Stats Collector")
        
        try:
            # Load historical stats
            await self._load_historical_stats()
            
            self._initialized = True
            logger.info("Stats Collector initialized successfully")
            
        except (AttributeError, ImportError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize Stats Collector: {e}")
            raise AutoCodeError("Stats collector initialization failed", cause=e)
    
    async def shutdown(self) -> None:
        """Shutdown the stats collector"""
        # Save final stats
        await self._save_performance_metrics()
        
        self._initialized = False
        logger.info("Stats Collector shutdown complete")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": "StatsCollector",
            "initialized": self._initialized,
            "tracked_operations": len(self._operation_stats),
            "performance_metrics": self._performance_metrics,
            "registered_components": list(self._components.keys())
        }
    
    def register_component(self, name: str, component) -> None:
        """Register a component for stats collection"""
        self._components[name] = component
        logger.debug(f"Registered component for stats: {name}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create cache key
            cache_key = "comprehensive_stats"
            
            # Try cache first
            cached_stats = self._stats_cache.get(cache_key)
            if cached_stats is not None:
                return cached_stats
            
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "autocode_domain": {
                    "performance_metrics": self._performance_metrics,
                    "operation_stats": self._get_operation_summary(),
                    "uptime_hours": self._get_uptime_hours()
                },
                "components": {},
                "cache_performance": {},
                "learning_metrics": {},
                "system_health": {}
            }
            
            # Collect stats from registered components
            for name, component in self._components.items():
                try:
                    if hasattr(component, 'get_component_info'):
                        stats["components"][name] = component.get_component_info()
                    
                    # Collect cache stats if component has caching
                    if hasattr(component, 'get_cache_stats'):
                        stats["cache_performance"][name] = component.get_cache_stats()
                    elif hasattr(component, 'pattern_cache'):
                        stats["cache_performance"][name] = {
                            "pattern_cache": component.pattern_cache.get_info()
                        }
                    elif hasattr(component, 'suggestion_cache'):
                        stats["cache_performance"][name] = {
                            "suggestion_cache": component.suggestion_cache.get_info()
                        }
                    
                except (AttributeError, ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to collect stats from component {name}: {e}")
                    stats["components"][name] = {"error": str(e)}
            
            # Collect learning metrics
            stats["learning_metrics"] = await self._collect_learning_metrics()
            
            # Collect system health metrics
            stats["system_health"] = await self._collect_system_health()
            
            # Calculate aggregate metrics
            stats["summary"] = self._calculate_summary_metrics(stats)
            
            # Cache the stats
            self._stats_cache.set(cache_key, stats)
            
            logger.debug("Collected comprehensive AutoCode statistics")
            return stats
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to collect stats: {e}")
            raise AutoCodeError("Statistics collection failed", cause=e)
    
    async def track_operation(self, operation: str, duration: float, success: bool, context: Dict[str, Any] = None) -> None:
        """Track operation performance"""
        try:
            # Update operation stats
            if operation not in self._operation_stats:
                self._operation_stats[operation] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_duration": 0.0,
                    "average_duration": 0.0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0,
                    "last_called": None
                }
            
            op_stats = self._operation_stats[operation]
            op_stats["total_calls"] += 1
            op_stats["total_duration"] += duration
            op_stats["average_duration"] = op_stats["total_duration"] / op_stats["total_calls"]
            op_stats["min_duration"] = min(op_stats["min_duration"], duration)
            op_stats["max_duration"] = max(op_stats["max_duration"], duration)
            op_stats["last_called"] = datetime.utcnow().isoformat()
            
            if success:
                op_stats["successful_calls"] += 1
                self._performance_metrics["successful_operations"] += 1
            else:
                op_stats["failed_calls"] += 1
                self._performance_metrics["failed_operations"] += 1
            
            self._performance_metrics["total_operations"] += 1
            
            # Update average response time
            total_duration = sum(stats["total_duration"] for stats in self._operation_stats.values())
            total_calls = sum(stats["total_calls"] for stats in self._operation_stats.values())
            self._performance_metrics["average_response_time"] = total_duration / total_calls if total_calls > 0 else 0.0
            
            logger.debug(f"Tracked operation: {operation} (duration: {duration:.3f}s, success: {success})")
            
        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to track operation {operation}: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self._initialized:
            await self.initialize()
        
        # Calculate current cache hit rates
        total_hits = 0
        total_requests = 0
        
        for component in self._components.values():
            if hasattr(component, 'pattern_cache'):
                cache_info = component.pattern_cache.get_info()
                if "stats" in cache_info:
                    total_hits += cache_info["stats"]["hits"]
                    total_requests += cache_info["stats"]["total_requests"]
            
            if hasattr(component, 'suggestion_cache'):
                cache_info = component.suggestion_cache.get_info()
                if "stats" in cache_info:
                    total_hits += cache_info["stats"]["hits"]
                    total_requests += cache_info["stats"]["total_requests"]
        
        if total_requests > 0:
            self._performance_metrics["cache_hit_rate"] = total_hits / total_requests
        
        return {
            "performance_metrics": self._performance_metrics,
            "operation_breakdown": self._operation_stats,
            "top_operations": self._get_top_operations(),
            "performance_trends": await self._get_performance_trends()
        }
    
    async def _load_historical_stats(self) -> None:
        """Load historical statistics"""
        try:
            # Load recent performance data
            stats_memories = await self.persistence_domain.retrieve_memories(
                query="autocode performance metrics statistics",
                types=["performance_stats", "autocode_metrics"],
                limit=10
            )
            
            if stats_memories:
                latest_stats = stats_memories[0]
                content = latest_stats.get("content", {})
                if isinstance(content, dict) and "performance_metrics" in content:
                    # Restore some metrics but reset counters
                    old_metrics = content["performance_metrics"]
                    self._performance_metrics["learning_accuracy"] = old_metrics.get("learning_accuracy", 0.0)
            
            logger.debug("Loaded historical statistics")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to load historical stats: {e}")
    
    async def _save_performance_metrics(self) -> None:
        """Save performance metrics"""
        try:
            metrics_data = {
                "performance_metrics": self._performance_metrics,
                "operation_stats": self._operation_stats,
                "timestamp": datetime.utcnow().isoformat(),
                "component_count": len(self._components)
            }
            
            await self.persistence_domain.store_memory(
                memory_type="performance_stats",
                content=metrics_data,
                importance=0.6,
                metadata={"stats_type": "autocode_performance"}
            )
            
            logger.debug("Saved performance metrics")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to save performance metrics: {e}")
    
    def _get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of operation statistics"""
        if not self._operation_stats:
            return {}
        
        total_calls = sum(stats["total_calls"] for stats in self._operation_stats.values())
        successful_calls = sum(stats["successful_calls"] for stats in self._operation_stats.values())
        
        return {
            "total_operations": len(self._operation_stats),
            "total_calls": total_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "most_used_operation": max(self._operation_stats.items(), key=lambda x: x[1]["total_calls"])[0] if self._operation_stats else None,
            "slowest_operation": max(self._operation_stats.items(), key=lambda x: x[1]["average_duration"])[0] if self._operation_stats else None
        }
    
    def _get_uptime_hours(self) -> float:
        """Get component uptime in hours"""
        # This would typically track from component initialization
        # For now, return a placeholder
        return 24.0  # Placeholder: 24 hours
    
    async def _collect_learning_metrics(self) -> Dict[str, Any]:
        """Collect learning-specific metrics"""
        metrics = {
            "command_suggestions_accuracy": 0.0,
            "pattern_detection_coverage": 0.0,
            "session_analysis_quality": 0.0,
            "learning_progression_rate": 0.0
        }
        
        try:
            # Get learning engine stats if available
            if "learning_engine" in self._components:
                learning_component = self._components["learning_engine"]
                if hasattr(learning_component, '_learning_stats'):
                    stats = learning_component._learning_stats
                    total_suggestions = stats["successful_suggestions"] + stats["failed_suggestions"]
                    if total_suggestions > 0:
                        metrics["command_suggestions_accuracy"] = stats["successful_suggestions"] / total_suggestions
            
            # Calculate other metrics based on available data
            # This would be expanded with actual calculations
            
        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to collect learning metrics: {e}")
        
        return metrics
    
    async def _collect_system_health(self) -> Dict[str, Any]:
        """Collect system health metrics"""
        health = {
            "components_healthy": 0,
            "components_total": len(self._components),
            "cache_health": "good",
            "memory_usage": "normal",
            "error_rate": 0.0
        }
        
        try:
            # Check component health
            healthy_count = 0
            for component in self._components.values():
                if hasattr(component, '_initialized') and component._initialized:
                    healthy_count += 1
            
            health["components_healthy"] = healthy_count
            health["component_health_rate"] = healthy_count / len(self._components) if self._components else 1.0
            
            # Calculate error rate
            total_ops = self._performance_metrics["total_operations"]
            failed_ops = self._performance_metrics["failed_operations"]
            health["error_rate"] = failed_ops / total_ops if total_ops > 0 else 0.0
            
        except (ValueError, AttributeError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Failed to collect system health: {e}")
        
        return health
    
    def _calculate_summary_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics from collected stats"""
        summary = {
            "overall_health_score": 0.0,
            "performance_score": 0.0,
            "learning_effectiveness": 0.0,
            "cache_efficiency": 0.0,
            "recommendations": []
        }
        
        try:
            # Calculate overall health score (0-100)
            health = stats.get("system_health", {})
            component_health = health.get("component_health_rate", 0.0)
            error_rate = health.get("error_rate", 1.0)
            
            summary["overall_health_score"] = (component_health * 70) + ((1 - error_rate) * 30)
            
            # Calculate performance score
            avg_response = self._performance_metrics["average_response_time"]
            if avg_response > 0:
                # Score based on response time (lower is better)
                summary["performance_score"] = max(0, 100 - (avg_response * 100))
            else:
                summary["performance_score"] = 100
            
            # Calculate cache efficiency
            cache_hit_rate = self._performance_metrics["cache_hit_rate"]
            summary["cache_efficiency"] = cache_hit_rate * 100
            
            # Generate recommendations
            if summary["overall_health_score"] < 80:
                summary["recommendations"].append("Check component health and resolve any initialization issues")
            
            if summary["performance_score"] < 70:
                summary["recommendations"].append("Optimize slow operations and consider caching improvements")
            
            if summary["cache_efficiency"] < 60:
                summary["recommendations"].append("Review caching strategy and TTL settings")
            
        except (ValueError, AttributeError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Failed to calculate summary metrics: {e}")
        
        return summary
    
    def _get_top_operations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top operations by call count"""
        if not self._operation_stats:
            return []
        
        sorted_ops = sorted(
            self._operation_stats.items(),
            key=lambda x: x[1]["total_calls"],
            reverse=True
        )
        
        return [
            {
                "operation": op_name,
                "total_calls": op_stats["total_calls"],
                "average_duration": op_stats["average_duration"],
                "success_rate": op_stats["successful_calls"] / op_stats["total_calls"] if op_stats["total_calls"] > 0 else 0.0
            }
            for op_name, op_stats in sorted_ops[:limit]
        ]
    
    async def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends over time"""
        # This would analyze historical performance data
        # For now, return placeholder data
        return {
            "response_time_trend": "stable",
            "error_rate_trend": "decreasing",
            "cache_hit_rate_trend": "increasing",
            "trend_period_days": 7
        }