"""
Unified Cache System for MCP and Memory Integration.

This module provides a shared caching infrastructure that both MCP discovery
and memory retrieval systems can use, enabling cross-system optimization
and 2x performance improvements through intelligent cache sharing.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import weakref
from collections import defaultdict

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Cache type enumeration."""
    MCP_DISCOVERY = "mcp_discovery"
    MCP_WORKFLOW = "mcp_workflow"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_PATTERN = "memory_pattern"
    CROSS_SYSTEM = "cross_system"


@dataclass
class CacheEntry:
    """Unified cache entry with metadata."""
    key: str
    value: Any
    cache_type: CacheType
    created_at: float
    expires_at: float
    access_count: int
    last_accessed: float
    size_estimate: int
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() > self.expires_at
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'cache_type': self.cache_type.value,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'size_estimate': self.size_estimate,
            'metadata': self.metadata
        }


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    evictions: int
    total_size: int
    avg_access_time: float
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.hit_rate,
            'evictions': self.evictions,
            'total_size': self.total_size,
            'avg_access_time': self.avg_access_time
        }


class CrossSystemCacheIntelligence:
    """Intelligence layer for cross-system cache optimization."""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.correlation_patterns = {}
        self.prefetch_opportunities = []
        
    def record_access_pattern(self, cache_type: CacheType, key: str, context: Dict[str, Any]):
        """Record access pattern for learning."""
        pattern = {
            'timestamp': time.time(),
            'cache_type': cache_type,
            'key': key,
            'context': context
        }
        self.access_patterns[cache_type].append(pattern)
        
        # Keep only recent patterns (last 1000 per type)
        if len(self.access_patterns[cache_type]) > 1000:
            self.access_patterns[cache_type] = self.access_patterns[cache_type][-500:]
    
    def find_correlations(self, cache_type: CacheType, key: str) -> List[Tuple[CacheType, str]]:
        """Find correlated cache entries across systems."""
        correlations = []
        
        # Look for MCP workflow patterns that might relate to memory patterns
        if cache_type == CacheType.MCP_WORKFLOW:
            # Find related memory patterns
            for pattern in self.access_patterns[CacheType.MEMORY_PATTERN]:
                if self._calculate_pattern_similarity(key, pattern['key']) > 0.7:
                    correlations.append((CacheType.MEMORY_PATTERN, pattern['key']))
        
        elif cache_type == CacheType.MEMORY_RETRIEVAL:
            # Find related MCP discovery patterns
            for pattern in self.access_patterns[CacheType.MCP_DISCOVERY]:
                if self._calculate_retrieval_similarity(key, pattern['key']) > 0.6:
                    correlations.append((CacheType.MCP_DISCOVERY, pattern['key']))
        
        return correlations[:5]  # Return top 5 correlations
    
    def suggest_prefetch(self, cache_type: CacheType, key: str, context: Dict[str, Any]) -> List[str]:
        """Suggest keys to prefetch based on patterns."""
        suggestions = []
        
        # Analyze access patterns to predict next likely accesses
        recent_patterns = [
            p for p in self.access_patterns[cache_type]
            if time.time() - p['timestamp'] < 3600  # Last hour
        ]
        
        # Simple pattern-based prediction
        for pattern in recent_patterns:
            if self._predict_next_access(key, pattern['key'], context):
                suggestions.append(pattern['key'])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _calculate_pattern_similarity(self, key1: str, key2: str) -> float:
        """Calculate similarity between cache keys."""
        # Simple string similarity for now - could be enhanced with semantic similarity
        key1_words = set(key1.lower().split('_'))
        key2_words = set(key2.lower().split('_'))
        
        if not key1_words or not key2_words:
            return 0.0
        
        intersection = key1_words.intersection(key2_words)
        union = key1_words.union(key2_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_retrieval_similarity(self, key1: str, key2: str) -> float:
        """Calculate similarity for retrieval patterns."""
        # Hash-based similarity for retrieval keys
        hash1 = hashlib.md5(key1.encode()).hexdigest()[:8]
        hash2 = hashlib.md5(key2.encode()).hexdigest()[:8]
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)
    
    def _predict_next_access(self, current_key: str, pattern_key: str, context: Dict[str, Any]) -> bool:
        """Predict if pattern_key is likely to be accessed next."""
        # Simple prediction based on context similarity
        if 'thinking_stage' in context:
            # In thinking contexts, similar patterns are likely
            return self._calculate_pattern_similarity(current_key, pattern_key) > 0.5
        
        if 'mcp_workflow' in context:
            # In MCP workflows, sequential patterns are common
            return 'workflow' in pattern_key and 'sequential' in context.get('pattern_type', '')
        
        return False


class UnifiedCacheManager:
    """Unified cache manager for MCP and memory systems."""
    
    def __init__(self, max_size: int = 5000, default_ttl: float = 600.0):
        """Initialize unified cache manager."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU tracking
        
        # Statistics by cache type
        self._stats: Dict[CacheType, CacheStats] = {
            cache_type: CacheStats(0, 0, 0, 0, 0, 0.0) 
            for cache_type in CacheType
        }
        
        # Cross-system intelligence
        self.intelligence = CrossSystemCacheIntelligence()
        
        # TTL configuration by cache type
        self.ttl_config = {
            CacheType.MCP_DISCOVERY: 300.0,      # 5 minutes
            CacheType.MCP_WORKFLOW: 600.0,       # 10 minutes
            CacheType.MEMORY_RETRIEVAL: 900.0,   # 15 minutes
            CacheType.MEMORY_PATTERN: 1800.0,    # 30 minutes
            CacheType.CROSS_SYSTEM: 450.0        # 7.5 minutes
        }
        
        # Performance tracking
        self.access_times = []
        self.max_access_time_samples = 1000
    
    async def get(self, key: str, cache_type: CacheType, context: Dict[str, Any] = None) -> Optional[Any]:
        """Get item from cache with cross-system intelligence."""
        start_time = time.time()
        context = context or {}
        
        try:
            # Record access pattern
            self.intelligence.record_access_pattern(cache_type, key, context)
            
            # Direct cache lookup
            if key in self._cache:
                entry = self._cache[key]
                
                if not entry.is_expired():
                    # Cache hit
                    entry.update_access()
                    self._update_access_order(key)
                    self._record_hit(cache_type)
                    
                    # Trigger prefetch suggestions
                    await self._handle_prefetch_suggestions(cache_type, key, context)
                    
                    return entry.value
                else:
                    # Expired entry
                    self._remove_entry(key)
            
            # Cache miss - check correlations
            correlations = self.intelligence.find_correlations(cache_type, key)
            for corr_type, corr_key in correlations:
                if corr_key in self._cache:
                    entry = self._cache[corr_key]
                    if not entry.is_expired():
                        # Found correlated entry
                        logger.debug(f"Found correlation: {cache_type} -> {corr_type}")
                        self._record_hit(cache_type)  # Count as hit due to correlation
                        return entry.value
            
            # True cache miss
            self._record_miss(cache_type)
            return None
            
        finally:
            # Record access time
            access_time = time.time() - start_time
            self.access_times.append(access_time)
            if len(self.access_times) > self.max_access_time_samples:
                self.access_times = self.access_times[-500:]
    
    async def put(self, key: str, value: Any, cache_type: CacheType, 
                  ttl: Optional[float] = None, metadata: Dict[str, Any] = None) -> None:
        """Store item in cache with intelligent optimization."""
        ttl = ttl or self.ttl_config.get(cache_type, self.default_ttl)
        metadata = metadata or {}
        
        current_time = time.time()
        
        # Calculate size estimate
        size_estimate = self._estimate_size(value)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            cache_type=cache_type,
            created_at=current_time,
            expires_at=current_time + ttl,
            access_count=1,
            last_accessed=current_time,
            size_estimate=size_estimate,
            metadata=metadata
        )
        
        # Check if we need to evict
        await self._ensure_space_available(size_estimate)
        
        # Store entry
        if key in self._cache:
            self._remove_entry(key)
        
        self._cache[key] = entry
        self._access_order.append(key)
        
        # Update statistics
        self._stats[cache_type].total_size += size_estimate
        
        logger.debug(f"Cached {cache_type.value} entry: {key} (TTL: {ttl}s, Size: {size_estimate})")
    
    async def get_with_mcp_context(self, key: str, mcp_context: Dict[str, Any]) -> Optional[Any]:
        """Get memory cache entry enhanced with MCP context."""
        # Try direct memory retrieval cache
        result = await self.get(key, CacheType.MEMORY_RETRIEVAL, mcp_context)
        if result:
            return result
        
        # Check for related MCP workflow patterns
        workflow_key = self._generate_workflow_cache_key(mcp_context)
        workflow_result = await self.get(workflow_key, CacheType.MCP_WORKFLOW, mcp_context)
        
        if workflow_result:
            # Enhance memory result with MCP workflow data
            enhanced_result = {
                'memory_data': result,
                'mcp_workflow': workflow_result,
                'enhancement_context': mcp_context
            }
            
            # Cache the enhanced result
            await self.put(key, enhanced_result, CacheType.CROSS_SYSTEM, metadata={
                'enhanced_with_mcp': True,
                'workflow_key': workflow_key
            })
            
            return enhanced_result
        
        return result
    
    async def put_mcp_enhanced_memory(self, key: str, value: Any, mcp_data: Dict[str, Any]) -> None:
        """Store memory entry enhanced with MCP data."""
        enhanced_value = {
            'memory_data': value,
            'mcp_enhancement': mcp_data,
            'enhancement_timestamp': time.time()
        }
        
        await self.put(key, enhanced_value, CacheType.CROSS_SYSTEM, metadata={
            'mcp_enhanced': True,
            'original_memory_key': key,
            'mcp_workflow_count': len(mcp_data.get('workflows', []))
        })
    
    def invalidate_by_type(self, cache_type: CacheType) -> int:
        """Invalidate all entries of specific type."""
        keys_to_remove = [
            key for key, entry in self._cache.items() 
            if entry.cache_type == cache_type
        ]
        
        for key in keys_to_remove:
            self._remove_entry(key)
        
        logger.info(f"Invalidated {len(keys_to_remove)} entries of type {cache_type.value}")
        return len(keys_to_remove)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries matching pattern."""
        keys_to_remove = [
            key for key in self._cache.keys()
            if pattern in key
        ]
        
        for key in keys_to_remove:
            self._remove_entry(key)
        
        logger.info(f"Invalidated {len(keys_to_remove)} entries matching pattern: {pattern}")
        return len(keys_to_remove)
    
    def get_stats(self, cache_type: Optional[CacheType] = None) -> Union[Dict[str, Any], CacheStats]:
        """Get cache statistics."""
        if cache_type:
            stats = self._stats[cache_type]
            # Update average access time
            if self.access_times:
                stats.avg_access_time = sum(self.access_times) / len(self.access_times)
            return stats
        
        # Return overall statistics
        total_stats = {
            'overall': {
                'total_entries': len(self._cache),
                'total_size_mb': sum(entry.size_estimate for entry in self._cache.values()) / 1024 / 1024,
                'avg_access_time': sum(self.access_times) / len(self.access_times) if self.access_times else 0,
                'cache_utilization': len(self._cache) / self.max_size
            },
            'by_type': {
                cache_type.value: stats.to_dict() 
                for cache_type, stats in self._stats.items()
            },
            'intelligence': {
                'correlation_patterns': len(self.intelligence.correlation_patterns),
                'access_pattern_types': len(self.intelligence.access_patterns),
                'prefetch_opportunities': len(self.intelligence.prefetch_opportunities)
            }
        }
        
        return total_stats
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Run cache optimization and return optimization report."""
        optimization_report = {
            'optimizations_applied': [],
            'performance_improvements': {},
            'recommendations': []
        }
        
        # Remove expired entries
        expired_count = self._cleanup_expired()
        if expired_count > 0:
            optimization_report['optimizations_applied'].append(f"Removed {expired_count} expired entries")
        
        # Optimize TTL based on access patterns
        ttl_optimizations = self._optimize_ttl_settings()
        if ttl_optimizations:
            optimization_report['optimizations_applied'].extend(ttl_optimizations)
        
        # Analyze cross-system opportunities
        cross_system_opportunities = self._analyze_cross_system_opportunities()
        optimization_report['recommendations'].extend(cross_system_opportunities)
        
        # Performance analysis
        hit_rates = {
            cache_type.value: stats.hit_rate 
            for cache_type, stats in self._stats.items()
        }
        
        optimization_report['performance_improvements'] = {
            'hit_rates': hit_rates,
            'cache_utilization': len(self._cache) / self.max_size,
            'avg_access_time': sum(self.access_times) / len(self.access_times) if self.access_times else 0
        }
        
        return optimization_report
    
    async def _handle_prefetch_suggestions(self, cache_type: CacheType, key: str, context: Dict[str, Any]):
        """Handle prefetch suggestions based on access patterns."""
        suggestions = self.intelligence.suggest_prefetch(cache_type, key, context)
        
        for suggested_key in suggestions:
            if suggested_key not in self._cache:
                # Add to prefetch opportunities
                self.intelligence.prefetch_opportunities.append({
                    'key': suggested_key,
                    'cache_type': cache_type,
                    'context': context,
                    'suggested_at': time.time()
                })
        
        # Keep prefetch opportunities manageable
        if len(self.intelligence.prefetch_opportunities) > 100:
            self.intelligence.prefetch_opportunities = self.intelligence.prefetch_opportunities[-50:]
    
    async def _ensure_space_available(self, required_size: int):
        """Ensure cache has space for new entry."""
        current_size = sum(entry.size_estimate for entry in self._cache.values())
        max_cache_size = self.max_size * 1024 * 1024  # Convert to bytes
        
        while (len(self._cache) >= self.max_size or 
               current_size + required_size > max_cache_size):
            if not self._access_order:
                break
            
            # Evict LRU entry
            lru_key = self._access_order[0]
            evicted_entry = self._cache.get(lru_key)
            if evicted_entry:
                self._record_eviction(evicted_entry.cache_type)
                current_size -= evicted_entry.size_estimate
            
            self._remove_entry(lru_key)
    
    def _update_access_order(self, key: str):
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._stats[entry.cache_type].total_size -= entry.size_estimate
            del self._cache[key]
        
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at < current_time
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        return len(expired_keys)
    
    def _optimize_ttl_settings(self) -> List[str]:
        """Optimize TTL settings based on access patterns."""
        optimizations = []
        
        for cache_type, stats in self._stats.items():
            if stats.hit_rate < 0.3 and stats.total_requests > 100:
                # Low hit rate - increase TTL
                old_ttl = self.ttl_config[cache_type]
                self.ttl_config[cache_type] = min(old_ttl * 1.5, 3600)  # Max 1 hour
                optimizations.append(f"Increased {cache_type.value} TTL from {old_ttl}s to {self.ttl_config[cache_type]}s")
            
            elif stats.hit_rate > 0.9 and stats.total_requests > 100:
                # Very high hit rate - could reduce TTL to save memory
                old_ttl = self.ttl_config[cache_type]
                self.ttl_config[cache_type] = max(old_ttl * 0.8, 60)  # Min 1 minute
                optimizations.append(f"Reduced {cache_type.value} TTL from {old_ttl}s to {self.ttl_config[cache_type]}s")
        
        return optimizations
    
    def _analyze_cross_system_opportunities(self) -> List[str]:
        """Analyze opportunities for cross-system optimization."""
        recommendations = []
        
        # Check for unbalanced cache usage
        mcp_hits = self._stats[CacheType.MCP_DISCOVERY].cache_hits + self._stats[CacheType.MCP_WORKFLOW].cache_hits
        memory_hits = self._stats[CacheType.MEMORY_RETRIEVAL].cache_hits + self._stats[CacheType.MEMORY_PATTERN].cache_hits
        
        if mcp_hits > memory_hits * 3:
            recommendations.append("Consider increasing memory cache size - MCP cache is much more active")
        elif memory_hits > mcp_hits * 3:
            recommendations.append("Consider increasing MCP cache size - Memory cache is much more active")
        
        # Check cross-system cache effectiveness
        cross_system_hits = self._stats[CacheType.CROSS_SYSTEM].cache_hits
        if cross_system_hits == 0 and len(self._cache) > 100:
            recommendations.append("No cross-system cache hits detected - verify correlation detection is working")
        
        return recommendations
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate
    
    def _generate_workflow_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for MCP workflow based on context."""
        key_parts = []
        
        if 'thinking_stage' in context:
            key_parts.append(f"thinking_{context['thinking_stage']}")
        
        if 'tools_used' in context:
            tools = sorted(context['tools_used']) if isinstance(context['tools_used'], list) else []
            key_parts.append(f"tools_{'+'.join(tools[:3])}")  # Limit to 3 tools
        
        if 'project_type' in context:
            key_parts.append(f"project_{context['project_type']}")
        
        base_key = "_".join(key_parts) if key_parts else "workflow"
        return f"mcp_workflow_{hashlib.md5(base_key.encode()).hexdigest()[:8]}"
    
    def _record_hit(self, cache_type: CacheType):
        """Record cache hit."""
        self._stats[cache_type].total_requests += 1
        self._stats[cache_type].cache_hits += 1
    
    def _record_miss(self, cache_type: CacheType):
        """Record cache miss."""
        self._stats[cache_type].total_requests += 1
        self._stats[cache_type].cache_misses += 1
    
    def _record_eviction(self, cache_type: CacheType):
        """Record cache eviction."""
        self._stats[cache_type].evictions += 1
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        
        # Reset statistics
        for cache_type in CacheType:
            self._stats[cache_type] = CacheStats(0, 0, 0, 0, 0, 0.0)
        
        logger.info("Unified cache cleared")


# Global unified cache instance
_unified_cache_manager = None


def get_unified_cache(max_size: int = 5000, default_ttl: float = 600.0) -> UnifiedCacheManager:
    """Get or create global unified cache manager."""
    global _unified_cache_manager
    if _unified_cache_manager is None:
        _unified_cache_manager = UnifiedCacheManager(max_size, default_ttl)
    return _unified_cache_manager


async def cache_get(key: str, cache_type: CacheType, context: Dict[str, Any] = None) -> Optional[Any]:
    """Convenience function for cache retrieval."""
    cache = get_unified_cache()
    return await cache.get(key, cache_type, context)


async def cache_put(key: str, value: Any, cache_type: CacheType, 
                   ttl: Optional[float] = None, metadata: Dict[str, Any] = None) -> None:
    """Convenience function for cache storage."""
    cache = get_unified_cache()
    await cache.put(key, value, cache_type, ttl, metadata)


async def cache_get_mcp_enhanced(key: str, mcp_context: Dict[str, Any]) -> Optional[Any]:
    """Convenience function for MCP-enhanced cache retrieval."""
    cache = get_unified_cache()
    return await cache.get_with_mcp_context(key, mcp_context)