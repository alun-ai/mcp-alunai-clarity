"""
MCP Cache Integration with Unified Cache System.

This module provides adapters and integration points to connect the Enhanced MCP
Discovery System with the unified cache, enabling cross-system optimization
and intelligent caching strategies.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib

from ..core.unified_cache import (
    UnifiedCacheManager, CacheType, get_unified_cache,
    cache_get, cache_put, cache_get_mcp_enhanced
)

logger = logging.getLogger(__name__)


@dataclass
class MCPCacheMetrics:
    """MCP-specific cache metrics."""
    discovery_cache_hits: int
    workflow_cache_hits: int
    server_cache_hits: int
    tool_cache_hits: int
    average_discovery_time: float
    cache_effectiveness_score: float


class MCPCacheAdapter:
    """Adapter to integrate MCP caching with unified cache system."""
    
    def __init__(self, unified_cache: Optional[UnifiedCacheManager] = None):
        """Initialize MCP cache adapter."""
        self.unified_cache = unified_cache or get_unified_cache()
        self.metrics = MCPCacheMetrics(0, 0, 0, 0, 0.0, 0.0)
        self._discovery_times = []
        
        # Cache key prefixes for organization
        self.key_prefixes = {
            'server_discovery': 'mcp_discovery_server',
            'tool_discovery': 'mcp_discovery_tool',
            'workflow_pattern': 'mcp_workflow',
            'resource_pattern': 'mcp_resource',
            'slash_command': 'mcp_slash_cmd',
            'performance_metric': 'mcp_perf'
        }
    
    async def cache_server_discovery(self, server_config_hash: str, servers: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> None:
        """Cache server discovery results."""
        key = f"{self.key_prefixes['server_discovery']}_{server_config_hash}"
        
        await cache_put(
            key=key,
            value=servers,
            cache_type=CacheType.MCP_DISCOVERY,
            ttl=300.0,  # 5 minutes for server discovery
            metadata={
                'discovery_type': 'server',
                'server_count': len(servers),
                'discovery_context': context or {},
                'cached_at': time.time()
            }
        )
        
        logger.debug(f"Cached server discovery: {len(servers)} servers")
    
    async def get_cached_server_discovery(self, server_config_hash: str, 
                                        context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached server discovery results."""
        key = f"{self.key_prefixes['server_discovery']}_{server_config_hash}"
        
        result = await cache_get(
            key=key,
            cache_type=CacheType.MCP_DISCOVERY,
            context=context or {}
        )
        
        if result:
            self.metrics.discovery_cache_hits += 1
            logger.debug(f"Cache hit for server discovery: {server_config_hash}")
        
        return result
    
    async def cache_tool_discovery(self, server_name: str, tools: List[Dict[str, Any]], 
                                 context: Dict[str, Any] = None) -> None:
        """Cache tool discovery results for a specific server."""
        key = f"{self.key_prefixes['tool_discovery']}_{server_name}"
        
        await cache_put(
            key=key,
            value=tools,
            cache_type=CacheType.MCP_DISCOVERY,
            ttl=600.0,  # 10 minutes for tool discovery
            metadata={
                'discovery_type': 'tool',
                'server_name': server_name,
                'tool_count': len(tools),
                'discovery_context': context or {},
                'cached_at': time.time()
            }
        )
        
        logger.debug(f"Cached tool discovery for {server_name}: {len(tools)} tools")
    
    async def get_cached_tool_discovery(self, server_name: str, 
                                      context: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached tool discovery results."""
        key = f"{self.key_prefixes['tool_discovery']}_{server_name}"
        
        result = await cache_get(
            key=key,
            cache_type=CacheType.MCP_DISCOVERY,
            context=context or {}
        )
        
        if result:
            self.metrics.tool_cache_hits += 1
            logger.debug(f"Cache hit for tool discovery: {server_name}")
        
        return result
    
    async def cache_workflow_pattern(self, pattern_data: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> str:
        """Cache MCP workflow pattern with intelligent key generation."""
        # Generate cache key based on pattern characteristics
        key_components = [
            pattern_data.get('trigger_context', ''),
            '_'.join(pattern_data.get('tool_sequence', [])),
            pattern_data.get('project_type', ''),
            pattern_data.get('intent', '')
        ]
        
        key_base = '_'.join(filter(None, key_components))
        key_hash = hashlib.md5(key_base.encode()).hexdigest()[:8]
        key = f"{self.key_prefixes['workflow_pattern']}_{key_hash}"
        
        await cache_put(
            key=key,
            value=pattern_data,
            cache_type=CacheType.MCP_WORKFLOW,
            ttl=1800.0,  # 30 minutes for workflow patterns
            metadata={
                'pattern_type': 'workflow',
                'tool_count': len(pattern_data.get('tool_sequence', [])),
                'effectiveness_score': pattern_data.get('effectiveness_score', 0.0),
                'context': context or {},
                'cached_at': time.time()
            }
        )
        
        self.metrics.workflow_cache_hits += 1
        logger.debug(f"Cached workflow pattern: {key}")
        return key
    
    async def get_similar_workflow_patterns(self, query_context: str, 
                                          context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve similar workflow patterns using cache intelligence."""
        # Create search key for pattern matching
        search_components = [
            query_context,
            context.get('project_type', '') if context else '',
            context.get('thinking_stage', '') if context else ''
        ]
        
        search_key_base = '_'.join(filter(None, search_components))
        search_key_hash = hashlib.md5(search_key_base.encode()).hexdigest()[:8]
        search_key = f"{self.key_prefixes['workflow_pattern']}_{search_key_hash}"
        
        # Try direct cache hit first
        direct_result = await cache_get(
            key=search_key,
            cache_type=CacheType.MCP_WORKFLOW,
            context=context or {}
        )
        
        if direct_result:
            return [direct_result]
        
        # Use cache intelligence to find similar patterns
        # This would be enhanced by the unified cache's correlation finding
        cache_stats = self.unified_cache.get_stats()
        similar_patterns = []
        
        # For now, return empty list - would be enhanced with semantic search
        return similar_patterns
    
    async def cache_resource_pattern(self, reference: str, pattern_data: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> None:
        """Cache resource reference pattern."""
        key = f"{self.key_prefixes['resource_pattern']}_{hashlib.md5(reference.encode()).hexdigest()[:8]}"
        
        await cache_put(
            key=key,
            value=pattern_data,
            cache_type=CacheType.MCP_WORKFLOW,
            ttl=900.0,  # 15 minutes for resource patterns
            metadata={
                'pattern_type': 'resource',
                'reference': reference,
                'success_rate': pattern_data.get('success_rate', 0.0),
                'usage_count': pattern_data.get('usage_count', 0),
                'context': context or {},
                'cached_at': time.time()
            }
        )
        
        logger.debug(f"Cached resource pattern: {reference}")
    
    async def get_cached_resource_pattern(self, reference: str, 
                                        context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached resource pattern."""
        key = f"{self.key_prefixes['resource_pattern']}_{hashlib.md5(reference.encode()).hexdigest()[:8]}"
        
        return await cache_get(
            key=key,
            cache_type=CacheType.MCP_WORKFLOW,
            context=context or {}
        )
    
    async def cache_slash_commands(self, server_name: str, commands: List[Dict[str, Any]], 
                                 context: Dict[str, Any] = None) -> None:
        """Cache discovered slash commands."""
        key = f"{self.key_prefixes['slash_command']}_{server_name}"
        
        await cache_put(
            key=key,
            value=commands,
            cache_type=CacheType.MCP_DISCOVERY,
            ttl=450.0,  # 7.5 minutes for slash commands
            metadata={
                'discovery_type': 'slash_command',
                'server_name': server_name,
                'command_count': len(commands),
                'discovery_context': context or {},
                'cached_at': time.time()
            }
        )
        
        logger.debug(f"Cached slash commands for {server_name}: {len(commands)} commands")
    
    async def get_cached_slash_commands(self, server_name: str, 
                                      context: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached slash commands."""
        key = f"{self.key_prefixes['slash_command']}_{server_name}"
        
        return await cache_get(
            key=key,
            cache_type=CacheType.MCP_DISCOVERY,
            context=context or {}
        )
    
    async def cache_performance_metrics(self, operation: str, metrics: Dict[str, Any]) -> None:
        """Cache performance metrics for analysis."""
        key = f"{self.key_prefixes['performance_metric']}_{operation}_{int(time.time() // 300)}"  # 5-minute buckets
        
        await cache_put(
            key=key,
            value=metrics,
            cache_type=CacheType.MCP_DISCOVERY,
            ttl=3600.0,  # 1 hour for performance metrics
            metadata={
                'metric_type': 'performance',
                'operation': operation,
                'timestamp': time.time(),
                'bucket': int(time.time() // 300)
            }
        )
    
    async def invalidate_server_cache(self, server_name: Optional[str] = None) -> int:
        """Invalidate MCP server-related cache entries."""
        if server_name:
            # Invalidate specific server
            patterns = [
                f"{self.key_prefixes['server_discovery']}_{server_name}",
                f"{self.key_prefixes['tool_discovery']}_{server_name}",
                f"{self.key_prefixes['slash_command']}_{server_name}"
            ]
            
            total_invalidated = 0
            for pattern in patterns:
                total_invalidated += self.unified_cache.invalidate_by_pattern(pattern)
            
            return total_invalidated
        else:
            # Invalidate all MCP discovery cache
            return self.unified_cache.invalidate_by_type(CacheType.MCP_DISCOVERY)
    
    async def invalidate_workflow_cache(self) -> int:
        """Invalidate MCP workflow cache entries."""
        return self.unified_cache.invalidate_by_type(CacheType.MCP_WORKFLOW)
    
    def record_discovery_time(self, operation: str, duration: float):
        """Record discovery operation timing."""
        self._discovery_times.append({
            'operation': operation,
            'duration': duration,
            'timestamp': time.time()
        })
        
        # Keep last 100 discovery times
        if len(self._discovery_times) > 100:
            self._discovery_times = self._discovery_times[-50:]
        
        # Update average
        if self._discovery_times:
            self.metrics.average_discovery_time = sum(
                t['duration'] for t in self._discovery_times
            ) / len(self._discovery_times)
    
    def calculate_cache_effectiveness(self) -> float:
        """Calculate overall cache effectiveness score."""
        cache_stats = self.unified_cache.get_stats(CacheType.MCP_DISCOVERY)
        workflow_stats = self.unified_cache.get_stats(CacheType.MCP_WORKFLOW)
        
        if hasattr(cache_stats, 'hit_rate') and hasattr(workflow_stats, 'hit_rate'):
            discovery_hit_rate = cache_stats.hit_rate
            workflow_hit_rate = workflow_stats.hit_rate
            
            # Weighted effectiveness score
            effectiveness = (discovery_hit_rate * 0.6) + (workflow_hit_rate * 0.4)
            self.metrics.cache_effectiveness_score = effectiveness
            
            return effectiveness
        
        return 0.0
    
    async def get_mcp_cache_analytics(self) -> Dict[str, Any]:
        """Get comprehensive MCP cache analytics."""
        discovery_stats = self.unified_cache.get_stats(CacheType.MCP_DISCOVERY)
        workflow_stats = self.unified_cache.get_stats(CacheType.MCP_WORKFLOW)
        overall_stats = self.unified_cache.get_stats()
        
        analytics = {
            'cache_performance': {
                'discovery_hit_rate': discovery_stats.hit_rate if hasattr(discovery_stats, 'hit_rate') else 0,
                'workflow_hit_rate': workflow_stats.hit_rate if hasattr(workflow_stats, 'hit_rate') else 0,
                'overall_effectiveness': self.calculate_cache_effectiveness(),
                'average_discovery_time': self.metrics.average_discovery_time
            },
            'cache_usage': {
                'discovery_requests': discovery_stats.total_requests if hasattr(discovery_stats, 'total_requests') else 0,
                'workflow_requests': workflow_stats.total_requests if hasattr(workflow_stats, 'total_requests') else 0,
                'discovery_hits': self.metrics.discovery_cache_hits,
                'workflow_hits': self.metrics.workflow_cache_hits
            },
            'cache_optimization': {
                'memory_usage_mb': overall_stats.get('overall', {}).get('total_size_mb', 0),
                'cache_utilization': overall_stats.get('overall', {}).get('cache_utilization', 0),
                'evictions': discovery_stats.evictions + workflow_stats.evictions if hasattr(discovery_stats, 'evictions') else 0
            },
            'recent_performance': {
                'recent_discovery_times': self._discovery_times[-10:],
                'performance_trend': self._calculate_performance_trend()
            }
        }
        
        return analytics
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend based on recent discovery times."""
        if len(self._discovery_times) < 5:
            return 'insufficient_data'
        
        recent_avg = sum(t['duration'] for t in self._discovery_times[-5:]) / 5
        older_avg = sum(t['duration'] for t in self._discovery_times[-10:-5]) / 5
        
        if recent_avg < older_avg * 0.9:
            return 'improving'
        elif recent_avg > older_avg * 1.1:
            return 'degrading'
        else:
            return 'stable'
    
    async def optimize_mcp_cache(self) -> Dict[str, Any]:
        """Run MCP-specific cache optimizations."""
        optimization_report = await self.unified_cache.optimize_cache()
        
        # Add MCP-specific optimizations
        mcp_optimizations = []
        
        # Check if discovery cache is underperforming
        discovery_stats = self.unified_cache.get_stats(CacheType.MCP_DISCOVERY)
        if hasattr(discovery_stats, 'hit_rate') and discovery_stats.hit_rate < 0.3:
            mcp_optimizations.append("Increase MCP discovery cache TTL - low hit rate detected")
        
        # Check workflow pattern effectiveness
        if self.metrics.workflow_cache_hits < 10 and time.time() > 3600:  # After 1 hour
            mcp_optimizations.append("Workflow pattern caching may need tuning - low usage detected")
        
        # Add discovery time optimization
        if self.metrics.average_discovery_time > 1.0:  # > 1 second
            mcp_optimizations.append("Discovery operations are slow - consider increasing cache retention")
        
        optimization_report['mcp_specific_optimizations'] = mcp_optimizations
        
        return optimization_report


# Global MCP cache adapter instance
_mcp_cache_adapter = None


def get_mcp_cache_adapter() -> MCPCacheAdapter:
    """Get or create global MCP cache adapter."""
    global _mcp_cache_adapter
    if _mcp_cache_adapter is None:
        _mcp_cache_adapter = MCPCacheAdapter()
    return _mcp_cache_adapter


# Convenience functions for MCP caching
async def cache_mcp_servers(config_hash: str, servers: Dict[str, Any], context: Dict[str, Any] = None) -> None:
    """Convenience function to cache server discovery."""
    adapter = get_mcp_cache_adapter()
    await adapter.cache_server_discovery(config_hash, servers, context)


async def get_cached_mcp_servers(config_hash: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Convenience function to get cached server discovery."""
    adapter = get_mcp_cache_adapter()
    return await adapter.get_cached_server_discovery(config_hash, context)


async def cache_mcp_tools(server_name: str, tools: List[Dict[str, Any]], context: Dict[str, Any] = None) -> None:
    """Convenience function to cache tool discovery."""
    adapter = get_mcp_cache_adapter()
    await adapter.cache_tool_discovery(server_name, tools, context)


async def get_cached_mcp_tools(server_name: str, context: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
    """Convenience function to get cached tool discovery."""
    adapter = get_mcp_cache_adapter()
    return await adapter.get_cached_tool_discovery(server_name, context)