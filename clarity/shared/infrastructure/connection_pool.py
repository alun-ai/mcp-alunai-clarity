import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncContextManager
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from threading import Lock
import threading

from clarity.shared.exceptions import QdrantConnectionError
from clarity.shared.simple_logging import get_logger
from clarity.shared.lazy_imports import db_deps
from clarity.shared.infrastructure.shared_qdrant import get_shared_qdrant_client

logger = get_logger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for Qdrant connection"""
    url: Optional[str] = None
    path: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 30.0
    prefer_grpc: bool = True
    max_retries: int = 3
    retry_backoff: float = 1.0


@dataclass
class PooledConnection:
    """Wrapper for a pooled Qdrant connection"""
    client: Any  # QdrantClient (lazy loaded)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    in_use: bool = False
    connection_id: str = ""
    usage_count: int = 0
    
    def mark_used(self) -> None:
        """Mark connection as recently used"""
        self.last_used = time.time()
        self.usage_count += 1
    
    def is_expired(self, max_age: float) -> bool:
        """Check if connection has exceeded maximum age"""
        return time.time() - self.created_at > max_age
    
    def is_idle(self, max_idle: float) -> bool:
        """Check if connection has been idle too long"""
        return time.time() - self.last_used > max_idle


class QdrantConnectionPool:
    """High-performance connection pool for Qdrant clients"""
    
    def __init__(self, 
                 config: ConnectionConfig,
                 min_connections: int = 2,
                 max_connections: int = 10,
                 max_connection_age: float = 3600.0,  # 1 hour
                 max_idle_time: float = 300.0,        # 5 minutes
                 health_check_interval: float = 60.0,  # 1 minute
                 acquire_timeout: float = 30.0):
        """Initialize connection pool
        
        Args:
            config: Connection configuration
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
            max_connection_age: Maximum age of a connection before renewal
            max_idle_time: Maximum idle time before connection is closed
            health_check_interval: Interval for health checks
            acquire_timeout: Timeout for acquiring a connection
        """
        self.config = config
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_connection_age = max_connection_age
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.acquire_timeout = acquire_timeout
        
        # Connection tracking
        self._connections: List[PooledConnection] = []
        self._lock = Lock()
        self._connection_semaphore = asyncio.Semaphore(max_connections)
        self._next_connection_id = 0
        
        # Pool state
        self._initialized = False
        self._closed = False
        self._stats = {
            'total_created': 0,
            'total_acquired': 0,
            'total_released': 0,
            'current_active': 0,
            'health_checks': 0,
            'failed_health_checks': 0
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the connection pool"""
        if self._initialized:
            return
            
        logger.info(f"Initializing Qdrant connection pool (min={self.min_connections}, max={self.max_connections})")
        
        try:
            # Create minimum number of connections
            for _ in range(self.min_connections):
                conn = await self._create_connection()
                self._connections.append(conn)
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._initialized = True
            logger.info("Qdrant connection pool initialized successfully")
            
        except (ConnectionError, RuntimeError, OSError, ValueError, ImportError) as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise QdrantConnectionError("Pool initialization failed", cause=e)
    
    async def close(self) -> None:
        """Close the connection pool and all connections"""
        if self._closed:
            return
            
        logger.info("Closing Qdrant connection pool")
        self._closed = True
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        with self._lock:
            for conn in self._connections:
                try:
                    conn.client.close()
                except (ConnectionError, RuntimeError, OSError) as e:
                    logger.warning(f"Error closing connection {conn.connection_id}: {e}")
            
            self._connections.clear()
            
        logger.info("Connection pool closed")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[Any]:  # QdrantClient
        """Acquire a connection from the pool
        
        Yields:
            QdrantClient instance
            
        Raises:
            QdrantConnectionError: If unable to acquire connection
        """
        if self._closed:
            raise QdrantConnectionError("Connection pool is closed")
            
        if not self._initialized:
            await self.initialize()
        
        # Wait for available connection slot
        try:
            await asyncio.wait_for(
                self._connection_semaphore.acquire(),
                timeout=self.acquire_timeout
            )
        except asyncio.TimeoutError:
            raise QdrantConnectionError("Timeout acquiring connection from pool")
        
        connection = None
        try:
            # Get or create connection
            connection = await self._get_connection()
            self._stats['total_acquired'] += 1
            self._stats['current_active'] += 1
            
            # Mark as in use
            with self._lock:
                connection.in_use = True
                connection.mark_used()
            
            logger.debug(f"Acquired connection {connection.connection_id}")
            yield connection.client
            
        except (QdrantConnectionError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Error using connection: {e}")
            # If connection failed, remove it from pool
            if connection:
                await self._remove_connection(connection)
            raise QdrantConnectionError("Connection error", cause=e)
            
        finally:
            # Release connection back to pool
            if connection:
                with self._lock:
                    connection.in_use = False
                    
                self._stats['total_released'] += 1
                self._stats['current_active'] -= 1
                logger.debug(f"Released connection {connection.connection_id}")
            
            self._connection_semaphore.release()
    
    async def _get_connection(self) -> PooledConnection:
        """Get an available connection from the pool"""
        with self._lock:
            # Find available connection
            for conn in self._connections:
                if not conn.in_use and not conn.is_expired(self.max_connection_age):
                    # Health check the connection
                    if await self._health_check(conn):
                        return conn
                    else:
                        # Remove unhealthy connection
                        self._connections.remove(conn)
                        break
            
            # Create new connection if under limit
            if len(self._connections) < self.max_connections:
                return await self._create_connection()
            
            # Wait for connection to become available
            # This should not happen due to semaphore, but safety check
            raise QdrantConnectionError("No connections available in pool")
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection"""
        try:
            # Create Qdrant client with retry logic
            client = await self._create_qdrant_client()
            
            # Generate connection ID
            with self._lock:
                self._next_connection_id += 1
                connection_id = f"qdrant_conn_{self._next_connection_id}"
            
            connection = PooledConnection(
                client=client,
                connection_id=connection_id
            )
            
            self._stats['total_created'] += 1
            logger.debug(f"Created new connection {connection_id}")
            
            return connection
            
        except (ConnectionError, RuntimeError, OSError, ValueError, TimeoutError) as e:
            logger.error(f"Failed to create connection: {e}")
            raise QdrantConnectionError("Connection creation failed", cause=e)
    
    async def _create_qdrant_client(self) -> Any:  # QdrantClient
        """Create a new Qdrant client with retry logic and shared access support"""
        if self.config.url:
            # Remote Qdrant instance - use standard approach
            return await self._create_remote_client()
        else:
            # Local Qdrant instance - use shared client approach for concurrent access
            return await get_shared_qdrant_client(self.config.path, self.config.timeout)
    
    async def _create_remote_client(self) -> Any:
        """Create a remote Qdrant client (standard approach)"""
        QdrantClientClass = db_deps.QdrantClient
        if QdrantClientClass is None:
            raise QdrantConnectionError("qdrant-client not available")
            
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                client = QdrantClientClass(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc
                )
                
                # Test the connection
                client.get_collections()
                return client
                
            except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_backoff * (2 ** attempt)
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All connection attempts failed: {e}")
        
        raise last_error
    
    async def _health_check(self, connection: PooledConnection) -> bool:
        """Perform health check on a connection"""
        try:
            # Simple health check - get collections
            connection.client.get_collections()
            self._stats['health_checks'] += 1
            return True
            
        except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
            logger.warning(f"Health check failed for connection {connection.connection_id}: {e}")
            self._stats['failed_health_checks'] += 1
            return False
    
    async def _remove_connection(self, connection: PooledConnection) -> None:
        """Remove a connection from the pool"""
        with self._lock:
            if connection in self._connections:
                self._connections.remove(connection)
                
        try:
            connection.client.close()
        except (ConnectionError, RuntimeError, OSError) as e:
            logger.warning(f"Error closing removed connection {connection.connection_id}: {e}")
            
        logger.debug(f"Removed connection {connection.connection_id}")
    
    async def _cleanup_loop(self) -> None:
        """Background task for connection cleanup"""
        while not self._closed:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._cleanup_connections()
                
            except asyncio.CancelledError:
                break
            except (ConnectionError, RuntimeError, OSError, AttributeError) as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_connections(self) -> None:
        """Clean up expired and idle connections"""
        current_time = time.time()
        connections_to_remove = []
        
        with self._lock:
            for conn in self._connections:
                if conn.in_use:
                    continue
                    
                # Check if connection should be removed
                should_remove = (
                    conn.is_expired(self.max_connection_age) or
                    (conn.is_idle(self.max_idle_time) and len(self._connections) > self.min_connections)
                )
                
                if should_remove:
                    connections_to_remove.append(conn)
        
        # Remove connections outside the lock
        for conn in connections_to_remove:
            await self._remove_connection(conn)
            
        # Ensure minimum connections
        with self._lock:
            available_connections = len([c for c in self._connections if not c.in_use])
            
        if available_connections < self.min_connections:
            needed = self.min_connections - available_connections
            for _ in range(needed):
                try:
                    new_conn = await self._create_connection()
                    with self._lock:
                        self._connections.append(new_conn)
                except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
                    logger.error(f"Failed to create connection during cleanup: {e}")
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._lock:
            total_connections = len(self._connections)
            active_connections = sum(1 for c in self._connections if c.in_use)
            
        return {
            'total_connections': total_connections,
            'active_connections': active_connections,
            'available_connections': total_connections - active_connections,
            'min_connections': self.min_connections,
            'max_connections': self.max_connections,
            **self._stats
        }


# Global connection pool instance
_connection_pool: Optional[QdrantConnectionPool] = None
_pool_lock = threading.Lock()


async def get_connection_pool(config: Optional[ConnectionConfig] = None) -> QdrantConnectionPool:
    """Get or create the global connection pool
    
    Args:
        config: Optional connection configuration
        
    Returns:
        QdrantConnectionPool instance
    """
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is None:
            if config is None:
                # Use default configuration
                config = ConnectionConfig()
            
            # For file-based Qdrant, we can only have 1 connection
            # For remote Qdrant, we can have multiple connections
            min_conns = 1 if config.path else 2
            _connection_pool = QdrantConnectionPool(config, min_connections=min_conns)
            
    return _connection_pool


async def close_connection_pool() -> None:
    """Close the global connection pool"""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool:
            await _connection_pool.close()
            _connection_pool = None


@asynccontextmanager
async def qdrant_connection(config: Optional[ConnectionConfig] = None) -> AsyncContextManager[Any]:  # QdrantClient
    """Convenience context manager for getting a Qdrant connection
    
    Args:
        config: Optional connection configuration
        
    Yields:
        QdrantClient instance
    """
    pool = await get_connection_pool(config)
    async with pool.acquire() as client:
        yield client