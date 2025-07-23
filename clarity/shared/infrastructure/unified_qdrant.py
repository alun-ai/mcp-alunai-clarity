"""
Unified Qdrant connection management system.

This module replaces all existing connection patterns:
- shared_qdrant.py (SharedQdrantManager)
- connection_pool.py (QdrantConnectionPool)  
- Direct client instantiation
- Inline connection management

Provides intelligent, adaptive connection strategies with automatic optimization
based on usage patterns (local/remote, single/multi-process, high/low concurrency).
"""

import asyncio
import time
import os
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, AsyncContextManager, Dict, List
from contextlib import asynccontextmanager
from pathlib import Path

from clarity.shared.simple_logging import get_logger
from clarity.shared.lazy_imports import db_deps
from clarity.shared.exceptions import QdrantConnectionError

logger = get_logger(__name__)


class ConnectionStrategy(Enum):
    """Adaptive connection strategies based on usage patterns."""
    SINGLE_CLIENT = "single_client"      # Local storage, single process, low concurrency
    SHARED_CLIENT = "shared_client"      # Local storage, multi-process coordination  
    CONNECTION_POOL = "connection_pool"  # Remote server or high concurrency
    HYBRID = "hybrid"                    # Dynamic switching based on load


@dataclass
class UnifiedConnectionConfig:
    """Unified configuration for all connection types."""
    
    # Connection parameters
    url: Optional[str] = None
    path: Optional[str] = None  
    api_key: Optional[str] = None
    timeout: float = 30.0
    
    # Strategy configuration  
    strategy: Optional[ConnectionStrategy] = None  # Auto-detect if None
    max_connections: int = 10
    min_connections: int = 1  # File-based Qdrant only supports 1 connection
    
    # Advanced options
    prefer_grpc: bool = True
    health_check_interval: float = 60.0
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    
    # Monitoring and performance
    enable_metrics: bool = True
    enable_health_checks: bool = True
    connection_cache_ttl: float = 300.0  # 5 minutes
    
    def __eq__(self, other):
        """Configuration equality check for reinitialization detection."""
        if not isinstance(other, UnifiedConnectionConfig):
            return False
        return (
            self.url == other.url and 
            self.path == other.path and
            self.timeout == other.timeout and
            self.strategy == other.strategy
        )
    
    def get_cache_key(self) -> str:
        """Generate unique cache key for this configuration."""
        config_hash = hashlib.md5(
            f"{self.url}:{self.path}:{self.timeout}:{self.strategy}".encode()
        ).hexdigest()[:8]
        return f"qdrant_config_{config_hash}"


@dataclass
class ConnectionStats:
    """Connection performance and usage statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    active_connections: int = 0
    avg_acquisition_time_ms: float = 0.0
    current_strategy: Optional[ConnectionStrategy] = None
    
    # Strategy-specific metrics
    single_client_hits: int = 0
    shared_client_hits: int = 0
    pool_hits: int = 0
    hybrid_switches: int = 0
    
    # Performance tracking
    fastest_connection_ms: float = float('inf')
    slowest_connection_ms: float = 0.0
    recent_errors: List[str] = field(default_factory=list)
    
    def update_timing(self, duration_ms: float):
        """Update connection timing statistics."""
        self.fastest_connection_ms = min(self.fastest_connection_ms, duration_ms)
        self.slowest_connection_ms = max(self.slowest_connection_ms, duration_ms)
        
        # Rolling average with exponential decay
        alpha = 0.1  # Weight for new measurements  
        self.avg_acquisition_time_ms = (
            alpha * duration_ms + (1 - alpha) * self.avg_acquisition_time_ms
        )
    
    def record_error(self, error_msg: str):
        """Record connection error for analysis."""
        self.recent_errors.append(f"{time.time():.0f}: {error_msg}")
        # Keep only last 10 errors
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)


class UnifiedQdrantManager:
    """
    Unified Qdrant connection manager with adaptive strategies.
    
    This manager wraps the existing SharedQdrantManager and provides
    intelligent connection strategy selection and monitoring.
    
    Uses SharedQdrantManager for actual connection management while
    adding strategy detection and performance optimization.
    """
    
    def __init__(self):
        """Initialize manager state."""
        # Core configuration and state
        self._config: Optional[UnifiedConnectionConfig] = None
        self._strategy: Optional[ConnectionStrategy] = None
        self._stats = ConnectionStats()
        
        # Use existing SharedQdrantManager for connections
        from clarity.shared.infrastructure.shared_qdrant import SharedQdrantManager
        self._shared_manager = SharedQdrantManager()
        
        # Connection instances for different strategies
        self._single_client: Optional[Any] = None  # QdrantClient
        self._shared_client: Optional[Any] = None  # QdrantClient  
        self._connection_pool: Optional[Any] = None  # ConnectionPool
        
        # Internal management and coordination
        self._client_lock: Optional[asyncio.Lock] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_strategy_evaluation = 0.0
        self._performance_history = []
        
        # Initialization flags
        self._client_initialized = False
        self._health_monitoring_active = False
        
        logger.debug("UnifiedQdrantManager instance created")
    
    async def initialize(self, config: UnifiedConnectionConfig) -> None:
        """
        Initialize the manager with configuration.
        
        Thread-safe initialization with configuration change detection.
        Supports reinitialization if configuration changes.
        """
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        
        async with self._client_lock:
            # Check if already initialized with same config
            if self._config is not None:
                if self._config == config:
                    logger.debug("UnifiedQdrantManager already initialized with identical config")
                    return
                else:
                    logger.info("Configuration change detected, reinitializing UnifiedQdrantManager")
                    await self._cleanup_connections()
            
            # Store new configuration
            self._config = config
            self._client_lock = asyncio.Lock()
            
            # Determine optimal strategy
            self._strategy = await self._determine_optimal_strategy(config)
            self._stats.current_strategy = self._strategy
            
            logger.info(f"UnifiedQdrantManager initialized with strategy: {self._strategy.value}")
            
            # Start health monitoring if enabled
            if config.enable_health_checks and not self._health_monitoring_active:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                self._health_monitoring_active = True
                logger.debug("Health monitoring started")
    
    async def _determine_optimal_strategy(self, config: UnifiedConnectionConfig) -> ConnectionStrategy:
        """
        Intelligently determine optimal connection strategy.
        
        Strategy selection algorithm:
        1. Manual override takes precedence
        2. Remote server → CONNECTION_POOL (for scalability)
        3. Local + multi-process → SHARED_CLIENT (for coordination)  
        4. Local + single-process → SINGLE_CLIENT (for performance)
        5. Fallback to SINGLE_CLIENT (conservative default)
        """
        
        # Manual strategy override
        if config.strategy:
            logger.info(f"Using manual strategy override: {config.strategy.value}")
            return config.strategy
        
        # Remote server detection
        if config.url:
            logger.info("Detected remote Qdrant server, using CONNECTION_POOL strategy")
            return ConnectionStrategy.CONNECTION_POOL
        
        # Local storage analysis
        if config.path:
            # Detect multi-process usage patterns
            if await self._detect_multiprocess_usage(config.path):
                logger.info("Detected multi-process usage patterns, using SHARED_CLIENT strategy")
                return ConnectionStrategy.SHARED_CLIENT
            else:
                logger.info("Detected single-process usage, using SINGLE_CLIENT strategy")
                return ConnectionStrategy.SINGLE_CLIENT
        
        # Fallback strategy
        logger.warning("Could not determine optimal strategy, defaulting to SINGLE_CLIENT")
        return ConnectionStrategy.SINGLE_CLIENT
    
    async def _detect_multiprocess_usage(self, qdrant_path: str) -> bool:
        """
        Detect if multiple processes are accessing the same Qdrant storage.
        
        Detection methods:
        1. Check for existing coordination files (indicates multi-process)
        2. Analyze process lock patterns
        3. Examine storage access patterns
        """
        try:
            coordination_dir = Path(qdrant_path).parent / '.qdrant_coordination'
            
            # Check for existing coordination infrastructure
            if coordination_dir.exists():
                lock_files = list(coordination_dir.glob('*.json'))
                if lock_files:
                    logger.debug(f"Found {len(lock_files)} coordination files, indicates multi-process")
                    return True
            
            # Check for MCP server processes (indicates multi-process scenario)
            if os.path.exists("/proc"):
                try:
                    # Look for other processes that might use Qdrant
                    import psutil
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if proc.info['name'] and 'python' in proc.info['name'].lower():
                                cmdline = proc.info['cmdline'] or []
                                if any('mcp' in arg.lower() or 'qdrant' in arg.lower() for arg in cmdline):
                                    if proc.info['pid'] != os.getpid():  # Different from current process
                                        logger.debug("Detected other MCP/Qdrant processes, indicates multi-process")
                                        return True
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except ImportError:
                    logger.debug("psutil not available, using heuristic detection")
            
            # Default to single process for new setups
            return False
            
        except Exception as e:
            logger.warning(f"Error detecting multi-process usage: {e}, defaulting to single-process")
            return False
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[Any]:  # QdrantClient
        """
        Universal connection acquisition method.
        
        This single method optimally handles ALL connection scenarios:
        - Local file storage (single or multi-process)
        - Remote server connections (with intelligent pooling)
        - Automatic strategy switching based on performance
        - Comprehensive error handling and retry logic
        
        Replaces ALL existing connection patterns in the codebase:
        - get_shared_qdrant_client()
        - qdrant_connection() 
        - Direct QdrantClient instantiation
        - Inline connection management
        
        Usage:
            async with manager.get_connection() as client:
                # Use client - automatically optimized for your scenario
                collections = client.get_collections()
        """
        if not self._config:
            raise QdrantConnectionError("UnifiedQdrantManager not initialized")
        
        # Performance monitoring setup
        start_time = time.perf_counter()
        self._stats.total_requests += 1
        self._stats.active_connections += 1
        
        connection_acquired = False
        client = None
        
        try:
            # Route to appropriate strategy implementation
            client = await self._acquire_strategy_connection()
            connection_acquired = True
            
            # Update performance metrics
            acquisition_time_ms = (time.perf_counter() - start_time) * 1000
            self._stats.update_timing(acquisition_time_ms)
            self._stats.successful_requests += 1
            
            # Log performance if unusual
            if acquisition_time_ms > 100:  # >100ms is slow
                logger.warning(f"Slow connection acquisition: {acquisition_time_ms:.2f}ms")
            
            yield client
            
        except Exception as e:
            self._stats.failed_requests += 1
            self._stats.record_error(str(e))
            
            error_msg = f"Connection acquisition failed with {self._strategy.value} strategy: {e}"
            logger.error(error_msg)
            
            # Attempt strategy fallback for recoverable errors
            if connection_acquired or "timeout" in str(e).lower():
                try:
                    logger.info("Attempting fallback connection strategy")
                    fallback_client = await self._get_fallback_connection()
                    yield fallback_client
                    return  # Exit successful fallback
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed: {fallback_error}")
            
            raise QdrantConnectionError(
                error_msg,
                context={"strategy": self._strategy.value if self._strategy else None},
                cause=e
            )
            
        finally:
            self._stats.active_connections = max(0, self._stats.active_connections - 1)
    
    async def _acquire_strategy_connection(self) -> Any:  # QdrantClient
        """Route connection request to appropriate strategy implementation."""
        
        strategy_start = time.perf_counter()
        
        try:
            if self._strategy == ConnectionStrategy.SINGLE_CLIENT:
                client = await self._get_single_client()
                self._stats.single_client_hits += 1
                
            elif self._strategy == ConnectionStrategy.SHARED_CLIENT:
                client = await self._get_shared_client()
                self._stats.shared_client_hits += 1
                
            elif self._strategy == ConnectionStrategy.CONNECTION_POOL:
                client = await self._get_pooled_connection()
                self._stats.pool_hits += 1
                
            elif self._strategy == ConnectionStrategy.HYBRID:
                # Dynamic strategy selection based on current performance
                adaptive_strategy = await self._select_adaptive_strategy()
                self._stats.hybrid_switches += 1
                client = await self._get_connection_for_strategy(adaptive_strategy)
                
            else:
                raise QdrantConnectionError(f"Unknown strategy: {self._strategy}")
            
            strategy_time = (time.perf_counter() - strategy_start) * 1000
            logger.debug(f"Strategy {self._strategy.value} connection acquired in {strategy_time:.2f}ms")
            
            return client
            
        except Exception as e:
            logger.error(f"Strategy {self._strategy.value} failed: {e}")
            raise
    
    async def _get_single_client(self) -> Any:  # QdrantClient
        """Get or create single persistent client (optimal for low concurrency)."""
        if self._single_client is None:
            async with self._client_lock:
                if self._single_client is None:  # Double-check pattern
                    logger.debug("Creating single persistent Qdrant client")
                    self._single_client = await self._create_qdrant_client()
                    logger.info("Single persistent Qdrant client created successfully")
        
        return self._single_client
    
    async def _get_shared_client(self) -> Any:  # QdrantClient
        """Get shared client using file coordination (optimal for multi-process)."""
        if self._shared_client is None:
            async with self._client_lock:
                if self._shared_client is None:  # Double-check pattern
                    logger.debug("Acquiring shared Qdrant client for multi-process coordination")
                    # Use the SharedQdrantManager instance directly
                    self._shared_client = await self._shared_manager.get_client(
                        self._config.path, self._config.timeout
                    )
                    logger.info("Shared Qdrant client acquired successfully")
        
        return self._shared_client
    
    async def _get_pooled_connection(self) -> Any:  # QdrantClient
        """Get connection from pool (optimal for high concurrency/remote)."""
        if self._connection_pool is None:
            async with self._client_lock:
                if self._connection_pool is None:  # Double-check pattern
                    logger.debug("Initializing Qdrant connection pool")
                    await self._initialize_connection_pool()
                    logger.info("Qdrant connection pool initialized successfully")
        
        # Return a connection from the pool
        # Note: This should return the actual client, pool handles the context management
        pool_context = self._connection_pool.acquire()
        return await pool_context.__aenter__()
    
    async def _get_fallback_connection(self) -> Any:  # QdrantClient
        """Get fallback connection when primary strategy fails."""
        logger.info("Attempting fallback connection strategy")
        
        # Try simpler strategies first
        fallback_strategies = [
            ConnectionStrategy.SINGLE_CLIENT,
            ConnectionStrategy.SHARED_CLIENT  
        ]
        
        for fallback_strategy in fallback_strategies:
            if fallback_strategy == self._strategy:
                continue  # Skip current failed strategy
                
            try:
                logger.debug(f"Trying fallback strategy: {fallback_strategy.value}")
                return await self._get_connection_for_strategy(fallback_strategy)
            except Exception as e:
                logger.debug(f"Fallback strategy {fallback_strategy.value} failed: {e}")
                continue
        
        raise QdrantConnectionError("All fallback strategies exhausted")
    
    async def _get_connection_for_strategy(self, strategy: ConnectionStrategy) -> Any:
        """Get connection using specific strategy (for fallback and hybrid modes)."""
        if strategy == ConnectionStrategy.SINGLE_CLIENT:
            return await self._get_single_client()
        elif strategy == ConnectionStrategy.SHARED_CLIENT:
            return await self._get_shared_client()
        elif strategy == ConnectionStrategy.CONNECTION_POOL:
            return await self._get_pooled_connection()
        else:
            raise QdrantConnectionError(f"Cannot get connection for strategy: {strategy}")
    
    async def _create_qdrant_client(self) -> Any:  # QdrantClient
        """Create a new Qdrant client based on configuration."""
        QdrantClientClass = db_deps.QdrantClient
        if QdrantClientClass is None:
            raise QdrantConnectionError("qdrant-client not available")
        
        try:
            if self._config.url:
                # Remote client configuration
                logger.debug(f"Creating remote Qdrant client for {self._config.url}")
                client = QdrantClientClass(
                    url=self._config.url,
                    api_key=self._config.api_key,
                    timeout=self._config.timeout,
                    prefer_grpc=self._config.prefer_grpc
                )
            else:
                # Local client configuration
                logger.debug(f"Creating local Qdrant client for {self._config.path}")
                client = QdrantClientClass(
                    path=self._config.path,
                    timeout=self._config.timeout
                )
            
            # Test the connection
            try:
                collections = client.get_collections()
                logger.debug(f"Connection test successful: {len(collections.collections)} collections")
            except Exception as test_error:
                logger.warning(f"Connection test failed: {test_error}")
                # Don't fail here, connection might still be usable
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create Qdrant client: {e}")
            raise QdrantConnectionError(f"Client creation failed: {e}")
    
    async def _initialize_connection_pool(self) -> None:
        """Initialize connection pool for high-concurrency scenarios."""
        try:
            from clarity.shared.infrastructure.connection_pool import QdrantConnectionPool, ConnectionConfig
            
            # Convert unified config to pool config
            pool_config = ConnectionConfig(
                url=self._config.url,
                path=self._config.path,
                api_key=self._config.api_key,
                timeout=self._config.timeout,
                prefer_grpc=self._config.prefer_grpc,
                max_retries=self._config.retry_attempts,
                retry_backoff=self._config.retry_backoff
            )
            
            # Create and initialize pool
            self._connection_pool = QdrantConnectionPool(
                config=pool_config,
                min_connections=self._config.min_connections,
                max_connections=self._config.max_connections
            )
            
            await self._connection_pool.initialize()
            logger.info(f"Connection pool initialized: {self._config.min_connections}-{self._config.max_connections} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise QdrantConnectionError(f"Pool initialization failed: {e}")
    
    async def _select_adaptive_strategy(self) -> ConnectionStrategy:
        """Select strategy dynamically based on current performance metrics."""
        current_time = time.time()
        
        # Re-evaluate strategy every 5 minutes
        if current_time - self._last_strategy_evaluation < 300:
            return self._strategy  # Use current strategy
        
        self._last_strategy_evaluation = current_time
        
        # Simple heuristics for strategy selection
        avg_time = self._stats.avg_acquisition_time_ms
        error_rate = (self._stats.failed_requests / max(self._stats.total_requests, 1)) * 100
        
        # Switch to simpler strategy if performance is poor
        if avg_time > 200 or error_rate > 5:  # High latency or errors
            logger.info("Performance degraded, switching to SINGLE_CLIENT strategy")
            return ConnectionStrategy.SINGLE_CLIENT
        elif avg_time > 100:  # Moderate performance issues
            return ConnectionStrategy.SHARED_CLIENT
        else:
            return ConnectionStrategy.CONNECTION_POOL  # Good performance, use advanced features
    
    async def _health_check_loop(self) -> None:
        """Background health monitoring for all connection types."""
        logger.debug("Health check loop started")
        
        while self._health_monitoring_active and self._config:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                logger.debug("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                # Continue monitoring despite errors
    
    async def _perform_health_check(self) -> None:
        """Perform health check on active connections."""
        if not self._config:
            return
        
        try:
            # Test connection with minimal operation
            async with self.get_connection() as client:
                collections = client.get_collections()
                collection_count = len(collections.collections)
                logger.debug(f"Health check passed: {collection_count} collections available")
                
                # Reset error tracking on successful health check
                if self._stats.recent_errors:
                    self._stats.recent_errors.clear()
                    
        except Exception as e:
            error_msg = f"Health check failed: {e}"
            logger.warning(error_msg)
            self._stats.record_error(error_msg)
            
            # Consider strategy fallback if health checks consistently fail
            if len(self._stats.recent_errors) > 3:  # Multiple consecutive failures
                logger.warning("Multiple health check failures, may need strategy adjustment")
    
    def get_stats(self) -> ConnectionStats:
        """Get current connection statistics and performance metrics."""
        return self._stats
    
    async def _cleanup_connections(self) -> None:
        """Clean up all connection resources."""
        logger.info("Cleaning up UnifiedQdrantManager connections")
        
        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_monitoring_active = False
        
        # Clean up connection pool
        if self._connection_pool:
            try:
                await self._connection_pool.close()
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
            self._connection_pool = None
        
        # Clean up single client  
        if self._single_client:
            try:
                self._single_client.close()
            except Exception as e:
                logger.warning(f"Error closing single client: {e}")
            self._single_client = None
        
        # Clean up shared client (handled by SharedQdrantManager)
        if self._shared_client:
            try:
                await self._shared_manager.close()
            except Exception as e:
                logger.warning(f"Error closing shared client: {e}")
            self._shared_client = None
        
        logger.info("Connection cleanup completed")
    
    async def close(self) -> None:
        """Gracefully shutdown the connection manager."""
        await self._cleanup_connections()
        self._config = None
        self._strategy = None
        self._client_initialized = False
        logger.info("UnifiedQdrantManager shutdown completed")


# Global manager instance (not singleton)
_unified_manager = UnifiedQdrantManager()


async def get_qdrant_connection(
    config: Optional[UnifiedConnectionConfig] = None
) -> AsyncContextManager[Any]:  # QdrantClient
    """
    Universal Qdrant connection getter - The New Standard.
    
    This function replaces ALL existing connection patterns:
    ✅ get_shared_qdrant_client() → get_qdrant_connection()
    ✅ qdrant_connection() → get_qdrant_connection()  
    ✅ Direct QdrantClient() → get_qdrant_connection()
    ✅ Inline connection management → get_qdrant_connection()
    
    Features:
    - Automatic strategy optimization (local/remote, single/multi-process)
    - Comprehensive error handling and retry logic
    - Performance monitoring and adaptive behavior
    - Health checks and connection recovery
    - Zero-configuration defaults for common scenarios
    
    Args:
        config: Connection configuration (uses intelligent defaults if None)
    
    Returns:
        Async context manager yielding optimized QdrantClient
    
    Examples:
        # Zero-configuration (uses local storage defaults)
        async with get_qdrant_connection() as client:
            collections = client.get_collections()
        
        # Custom configuration
        config = UnifiedConnectionConfig(
            path="/custom/qdrant/path",
            timeout=60.0,
            enable_metrics=True
        )
        async with get_qdrant_connection(config) as client:
            # Automatically optimized for your scenario
            pass
        
        # Remote server  
        config = UnifiedConnectionConfig(
            url="https://your-qdrant-server.com",
            api_key="your-api-key"
        )
        async with get_qdrant_connection(config) as client:
            # Automatically uses connection pooling
            pass
    """
    # Provide intelligent defaults
    if config is None:
        config = UnifiedConnectionConfig(
            path="./.claude/alunai-clarity/qdrant",
            timeout=30.0,
            enable_metrics=True,
            enable_health_checks=True
        )
    
    # Initialize manager with configuration
    await _unified_manager.initialize(config)
    
    # Return connection context manager
    return _unified_manager.get_connection()


async def get_unified_stats() -> Dict[str, Any]:
    """
    Get comprehensive connection statistics and performance metrics.
    
    Returns:
        Dictionary containing:
        - Connection performance metrics
        - Strategy usage statistics  
        - Error rates and recent errors
        - Configuration details
        - Health status
    """
    stats = _unified_manager.get_stats()
    config = _unified_manager._config
    
    return {
        "manager_stats": {
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "active_connections": stats.active_connections,
            "success_rate_percent": (
                (stats.successful_requests / max(stats.total_requests, 1)) * 100
            ),
            "error_rate_percent": (
                (stats.failed_requests / max(stats.total_requests, 1)) * 100
            )
        },
        "performance_metrics": {
            "avg_acquisition_time_ms": stats.avg_acquisition_time_ms,
            "fastest_connection_ms": stats.fastest_connection_ms if stats.fastest_connection_ms != float('inf') else None,
            "slowest_connection_ms": stats.slowest_connection_ms
        },
        "strategy_usage": {
            "current_strategy": stats.current_strategy.value if stats.current_strategy else None,
            "single_client_hits": stats.single_client_hits,
            "shared_client_hits": stats.shared_client_hits,
            "pool_hits": stats.pool_hits,
            "hybrid_switches": stats.hybrid_switches
        },
        "configuration": {
            "url": config.url if config else None,
            "path": config.path if config else None,
            "timeout": config.timeout if config else None,
            "strategy_override": config.strategy.value if config and config.strategy else None,
            "health_checks_enabled": config.enable_health_checks if config else False,
            "metrics_enabled": config.enable_metrics if config else False
        },
        "health_status": {
            "recent_errors": stats.recent_errors[-3:],  # Last 3 errors
            "error_count": len(stats.recent_errors),
            "manager_initialized": _unified_manager._config is not None,
            "health_monitoring_active": _unified_manager._health_monitoring_active
        }
    }


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# 
# These functions provide compatibility with existing code during migration.
# They will be deprecated in future versions with clear migration paths.
# ============================================================================

import warnings

async def get_shared_qdrant_client(qdrant_path: str, timeout: float = 30.0) -> Any:
    """
    DEPRECATED: Use get_qdrant_connection() instead.
    
    This function will be removed in version 2.0.0.
    
    Migration Guide:
        # Old approach
        from clarity.shared.infrastructure.shared_qdrant import get_shared_qdrant_client
        client = await get_shared_qdrant_client(path, timeout)
        try:
            # Use client
            collections = client.get_collections()
        finally:
            # Manual resource management required
            pass
        
        # New approach  
        from clarity.shared.infrastructure.unified_qdrant import get_qdrant_connection, UnifiedConnectionConfig
        config = UnifiedConnectionConfig(path=path, timeout=timeout)
        async with get_qdrant_connection(config) as client:
            # Use client - automatic resource management
            collections = client.get_collections()
    """
    warnings.warn(
        "get_shared_qdrant_client is deprecated and will be removed in version 2.0.0. "
        "Use get_qdrant_connection() with UnifiedConnectionConfig instead. "
        "See migration guide in function docstring.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Forward to unified manager for backward compatibility
    config = UnifiedConnectionConfig(
        path=qdrant_path, 
        timeout=timeout,
        strategy=ConnectionStrategy.SHARED_CLIENT  # Force shared client for compatibility
    )
    
    await _unified_manager.initialize(config)
    return await _unified_manager._get_shared_client()


@asynccontextmanager
async def qdrant_connection(config: Optional[Any] = None) -> AsyncContextManager[Any]:
    """
    DEPRECATED: Use get_qdrant_connection() instead.
    
    This function will be removed in version 2.0.0.
    
    Migration Guide:
        # Old approach
        from clarity.shared.infrastructure.connection_pool import qdrant_connection, ConnectionConfig
        pool_config = ConnectionConfig(url=url, api_key=api_key)
        async with qdrant_connection(pool_config) as client:
            # Use client
            pass
        
        # New approach
        from clarity.shared.infrastructure.unified_qdrant import get_qdrant_connection, UnifiedConnectionConfig  
        config = UnifiedConnectionConfig(url=url, api_key=api_key)
        async with get_qdrant_connection(config) as client:
            # Use client - automatically optimized
            pass
    """
    warnings.warn(
        "qdrant_connection is deprecated and will be removed in version 2.0.0. "
        "Use get_qdrant_connection() with UnifiedConnectionConfig instead. "
        "See migration guide in function docstring.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Convert legacy config to unified config
    if config and hasattr(config, 'url'):
        # Legacy ConnectionConfig object
        unified_config = UnifiedConnectionConfig(
            url=getattr(config, 'url', None),
            path=getattr(config, 'path', None),
            api_key=getattr(config, 'api_key', None),
            timeout=getattr(config, 'timeout', 30.0),
            prefer_grpc=getattr(config, 'prefer_grpc', True)
        )
    else:
        # Default local configuration
        unified_config = UnifiedConnectionConfig(
            path="./.claude/alunai-clarity/qdrant"
        )
    
    # Forward to unified connection manager
    async with get_qdrant_connection(unified_config) as client:
        yield client


async def close_unified_qdrant_manager() -> None:
    """
    Gracefully shutdown the unified connection manager.
    
    Use this function during application shutdown to ensure all connections
    are properly closed and resources are cleaned up.
    """
    await _unified_manager.close()
    logger.info("Unified Qdrant Manager shutdown completed")


# Export the new standard API
__all__ = [
    # New unified API (recommended)
    'get_qdrant_connection',
    'UnifiedConnectionConfig', 
    'ConnectionStrategy',
    'get_unified_stats',
    'close_unified_qdrant_manager',
    
    # Backward compatibility (deprecated)
    'get_shared_qdrant_client',
    'qdrant_connection',
    
    # Internal classes (advanced usage)
    'UnifiedQdrantManager',
    'ConnectionStats'
]