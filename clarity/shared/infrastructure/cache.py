import time
import hashlib
import threading
import asyncio
import pickle
from typing import Any, Dict, Optional, Union, Callable, TypeVar, Generic, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
from abc import ABC, abstractmethod

from clarity.shared.simple_logging import get_logger
from clarity.shared.exceptions import ClarityException

logger = get_logger(__name__)

T = TypeVar('T')


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used  
    TTL = "ttl"  # Time To Live only
    FIFO = "fifo"  # First In, First Out


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at
    
    def idle_time(self) -> float:
        """Get idle time in seconds"""
        return time.time() - self.last_accessed


class CacheStats:
    """Cache performance statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_removals = 0
        self.size_evictions = 0
        self.manual_removals = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_hit(self) -> None:
        with self._lock:
            self.hits += 1
    
    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1
    
    def record_eviction(self, reason: str = "general") -> None:
        with self._lock:
            self.evictions += 1
            if reason == "expired":
                self.expired_removals += 1
            elif reason == "size":
                self.size_evictions += 1
            elif reason == "manual":
                self.manual_removals += 1
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            uptime = time.time() - self.start_time
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total_requests,
                "hit_rate": self.get_hit_rate(),
                "evictions": self.evictions,
                "expired_removals": self.expired_removals,
                "size_evictions": self.size_evictions,
                "manual_removals": self.manual_removals,
                "uptime_seconds": uptime,
                "requests_per_second": total_requests / uptime if uptime > 0 else 0
            }


class InMemoryCache(Generic[T]):
    """High-performance in-memory cache with TTL and size limits"""
    
    def __init__(self,
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 default_ttl: Optional[float] = 3600.0,  # 1 hour
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 cleanup_interval: float = 300.0,  # 5 minutes
                 enable_stats: bool = True):
        """Initialize cache
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds (None for no expiration)
            eviction_policy: Eviction policy when cache is full
            cleanup_interval: Interval for background cleanup
            enable_stats: Whether to collect statistics
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.cleanup_interval = cleanup_interval
        self.enable_stats = enable_stats
        
        # Storage
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats() if enable_stats else None
        
        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Memory tracking
        self._current_memory = 0
        
    async def start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task"""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                if self.stats:
                    self.stats.record_miss()
                return default
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self._current_memory -= entry.size_bytes
                if self.stats:
                    self.stats.record_miss()
                    self.stats.record_eviction("expired")
                return default
            
            # Update access metadata
            entry.touch()
            
            if self.stats:
                self.stats.record_hit()
            
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            # Fallback size estimation
            size_bytes = len(str(value)) * 2  # Rough estimate
        
        with self._lock:
            # Check if we need to make space
            self._make_space(size_bytes)
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory -= old_entry.size_bytes
            
            # Add new entry
            self._cache[key] = entry
            self._current_memory += size_bytes
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._current_memory -= entry.size_bytes
                if self.stats:
                    self.stats.record_eviction("manual")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def size(self) -> int:
        """Get number of entries in cache"""
        return len(self._cache)
    
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB"""
        return self._current_memory / (1024 * 1024)
    
    def _make_space(self, needed_bytes: int) -> None:
        """Make space for new entry"""
        # Check size limit
        while len(self._cache) >= self.max_size and self._cache:
            self._evict_one("size")
        
        # Check memory limit
        while (self._current_memory + needed_bytes > self.max_memory_bytes and 
               self._cache):
            self._evict_one("size")
    
    def _evict_one(self, reason: str = "general") -> None:
        """Evict one entry based on policy"""
        if not self._cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].last_accessed)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k].access_count)
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k].created_at)
        else:  # TTL policy - evict oldest
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k].created_at)
        
        entry = self._cache[oldest_key]
        del self._cache[oldest_key]
        self._current_memory -= entry.size_bytes
        
        if self.stats:
            self.stats.record_eviction(reason)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache[key]
            del self._cache[key]
            self._current_memory -= entry.size_bytes
            if self.stats:
                self.stats.record_eviction("expired")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                with self._lock:
                    self._cleanup_expired()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        with self._lock:
            info = {
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "current_size": len(self._cache),
                "current_memory_mb": self.memory_usage_mb(),
                "default_ttl": self.default_ttl,
                "eviction_policy": self.eviction_policy.value,
                "cleanup_interval": self.cleanup_interval
            }
            
            if self.stats:
                info["stats"] = self.stats.get_stats()
            
            return info


class CacheManager:
    """Global cache manager for different cache types"""
    
    def __init__(self):
        self._caches: Dict[str, InMemoryCache] = {}
        self._lock = threading.Lock()
    
    def get_cache(self, 
                  name: str,
                  max_size: int = 1000,
                  max_memory_mb: int = 100,
                  default_ttl: Optional[float] = 3600.0,
                  eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> InMemoryCache:
        """Get or create a named cache
        
        Args:
            name: Cache name
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds
            eviction_policy: Eviction policy
            
        Returns:
            InMemoryCache instance
        """
        with self._lock:
            if name not in self._caches:
                self._caches[name] = InMemoryCache(
                    max_size=max_size,
                    max_memory_mb=max_memory_mb,
                    default_ttl=default_ttl,
                    eviction_policy=eviction_policy
                )
            return self._caches[name]
    
    def remove_cache(self, name: str) -> bool:
        """Remove a named cache
        
        Args:
            name: Cache name
            
        Returns:
            True if cache existed and was removed
        """
        with self._lock:
            if name in self._caches:
                cache = self._caches[name]
                cache.clear()
                del self._caches[name]
                return True
            return False
    
    def clear_all(self) -> None:
        """Clear all caches"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            self._caches.clear()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        with self._lock:
            return {
                name: cache.get_info()
                for name, cache in self._caches.items()
            }
    
    async def start_all_cleanup_tasks(self) -> None:
        """Start cleanup tasks for all caches"""
        with self._lock:
            for cache in self._caches.values():
                await cache.start_cleanup_task()
    
    async def stop_all_cleanup_tasks(self) -> None:
        """Stop cleanup tasks for all caches"""
        with self._lock:
            for cache in self._caches.values():
                await cache.stop_cleanup_task()


# Global cache manager instance
cache_manager = CacheManager()


def get_cache(name: str, **kwargs) -> InMemoryCache:
    """Convenience function to get a named cache"""
    return cache_manager.get_cache(name, **kwargs)


def cache_key(*args, **kwargs) -> str:
    """Generate a consistent cache key from arguments"""
    # Serialize arguments deterministically
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items()) if kwargs else {}
    }
    
    # Create hash
    key_str = str(key_data)
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]


def cached(cache_name: str = "default",
          ttl: Optional[float] = None,
          key_func: Optional[Callable] = None):
    """Decorator for caching function results
    
    Args:
        cache_name: Name of cache to use
        ttl: Time to live for cached results
        key_func: Custom function to generate cache key
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = get_cache(cache_name)
        
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)
            return result
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._cache = cache  # Access to cache for manual operations
        
        return wrapper
    
    return decorator


async def async_cached(cache_name: str = "default",
                      ttl: Optional[float] = None,
                      key_func: Optional[Callable] = None):
    """Decorator for caching async function results"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = get_cache(cache_name)
        
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._cache = cache
        
        return wrapper
    
    return decorator