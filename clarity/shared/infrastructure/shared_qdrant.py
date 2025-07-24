"""
Shared Qdrant client for concurrent Claude instances.
"""

import asyncio
import os
import tempfile
import json
import time
from typing import Optional, Any
from pathlib import Path

from clarity.shared.simple_logging import get_logger
from clarity.shared.lazy_imports import db_deps
from clarity.shared.exceptions import QdrantConnectionError

logger = get_logger(__name__)


class SharedQdrantManager:
    """
    Manages a shared Qdrant client across multiple Claude instances.
    
    Uses file-based coordination to ensure only one process creates the client,
    while allowing safe sharing through inter-process communication patterns.
    
    Implements true lazy connection - client is only created on first vector operation.
    Enhanced with connection health monitoring and stale connection detection.
    """
    
    _instance: Optional['SharedQdrantManager'] = None
    _client: Optional[Any] = None  # QdrantClient
    _lock_file_path: Optional[str] = None
    _connection_lock: Optional[Any] = None  # asyncio.Lock
    _last_health_check: float = 0
    _connection_healthy: bool = False
    _creation_time: float = 0
    
    def __new__(cls) -> 'SharedQdrantManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._qdrant_path = None
            self._timeout = None
            self._connection_lock = None
    
    async def get_client(self, qdrant_path: str, timeout: float = 30.0) -> Any:
        """
        Get or create a shared Qdrant client (lazy initialization).
        
        This method now performs true lazy initialization - the actual Qdrant 
        connection is only established when this method is first called, not 
        during domain initialization.
        
        Args:
            qdrant_path: Path to Qdrant storage
            timeout: Client timeout
            
        Returns:
            Shared QdrantClient instance
        """
        # Return existing client if available
        if self._client is not None:
            return self._client
        
        # Store connection parameters for lazy initialization
        self._qdrant_path = qdrant_path
        self._timeout = timeout
        
        # Initialize connection lock if needed
        if self._connection_lock is None:
            import asyncio
            self._connection_lock = asyncio.Lock()
        
        # Thread-safe lazy initialization
        async with self._connection_lock:
            # Double-check pattern: another thread might have initialized while we waited
            if self._client is not None:
                return self._client
            
            logger.info(f"🔄 Lazy initializing Qdrant connection to {qdrant_path}")
            await self._initialize_client(qdrant_path, timeout)
            self._initialized = True
            logger.info(f"✅ Qdrant connection established successfully")
        
        return self._client
    
    async def _initialize_client(self, qdrant_path: str, timeout: float):
        """Initialize the shared client with proper coordination."""
        QdrantClientClass = db_deps.QdrantClient
        if QdrantClientClass is None:
            raise QdrantConnectionError("qdrant-client not available")
        
        # Create coordination directory
        coord_dir = Path(qdrant_path).parent / '.qdrant_coordination'
        coord_dir.mkdir(exist_ok=True)
        
        lock_file = coord_dir / 'qdrant_lock.json'
        self._lock_file_path = str(lock_file)
        
        # Try to acquire coordination lock
        max_wait_time = 30.0  # 30 seconds max wait
        wait_interval = 0.5   # Check every 500ms
        waited = 0.0
        
        while waited < max_wait_time:
            try:
                # Try to create client with timeout
                if await asyncio.wait_for(
                    self._try_create_client(qdrant_path, timeout, lock_file),
                    timeout=min(5.0, max_wait_time - waited)  # Max 5s per attempt
                ):
                    return
                    
                # If failed due to lock, wait and retry
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                
            except asyncio.TimeoutError:
                logger.warning(f"Client creation timeout after {waited:.1f}s, retrying...")
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                
            except Exception as e:
                if "already accessed" in str(e):
                    logger.debug(f"Storage locked, waiting... ({waited:.1f}s)")
                    await asyncio.sleep(wait_interval)
                    waited += wait_interval
                    continue
                else:
                    raise
        
        raise QdrantConnectionError(f"Could not acquire Qdrant client after {max_wait_time}s")
    
    async def _try_create_client(self, qdrant_path: str, timeout: float, lock_file: Path) -> bool:
        """Try to create the client with proper singleton handling and stale lock detection."""
        QdrantClientClass = db_deps.QdrantClient
        
        try:
            # Check for existing lock file and validate if it's stale
            if lock_file.exists():
                if await self._is_lock_stale(lock_file):
                    logger.warning("Detected stale lock file, attempting cleanup")
                    await self._cleanup_stale_lock(lock_file)
                else:
                    # Valid lock exists, check if we can reuse existing client
                    if self._client is not None:
                        try:
                            # Test existing client health
                            await asyncio.wait_for(
                                asyncio.to_thread(self._client.get_collections),
                                timeout=5.0
                            )
                            self._connection_healthy = True
                            self._last_health_check = time.time()
                            return True
                        except Exception as health_error:
                            logger.warning(f"Existing client health check failed: {health_error}")
                            self._client = None
                            self._connection_healthy = False
            
            # For file-based Qdrant storage, only ONE client instance can exist per path
            # Don't try to create multiple instances - just create once and reuse
            if self._client is not None:
                # Client already exists, test it and return
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self._client.get_collections),
                        timeout=5.0
                    )
                    return True
                except Exception:
                    # Client is stale, will recreate below
                    logger.debug("Existing client failed health check, recreating")
                    self._client = None
            
            # Create the client - there should only ever be one
            client = QdrantClientClass(
                path=qdrant_path,
                timeout=timeout
            )
            
            # Test the connection with timeout
            await asyncio.wait_for(
                asyncio.to_thread(client.get_collections),
                timeout=10.0
            )
            
            # Register this client instance with enhanced metadata
            import time
            lock_info = {
                'pid': os.getpid(),
                'created_at': time.time(),
                'path': qdrant_path,
                'timeout': timeout,
                'last_health_check': time.time(),
                'manager_id': id(self)  # Unique manager instance ID
            }
            
            with open(lock_file, 'w') as f:
                json.dump(lock_info, f)
            
            self._client = client
            self._connection_healthy = True
            self._last_health_check = time.time()
            self._creation_time = time.time()
            logger.info(f"Created shared Qdrant client for {qdrant_path} (PID: {os.getpid()})")
            return True
            
        except Exception as e:
            if "already accessed" in str(e):
                # This means another client instance is already using this path
                # For Qdrant file storage, this is not recoverable within the same process
                logger.error(f"Qdrant storage path {qdrant_path} is locked by another client instance")
                logger.error("File-based Qdrant storage only supports one client instance per path")
                raise QdrantConnectionError(f"Qdrant storage already in use: {e}")
            else:
                logger.debug(f"Failed to create client: {e}")
                return False
    
    async def _is_lock_stale(self, lock_file: Path) -> bool:
        """Check if a lock file represents a stale connection."""
        try:
            with open(lock_file, 'r') as f:
                lock_info = json.load(f)
            
            lock_pid = lock_info.get('pid')
            created_at = lock_info.get('created_at', 0)
            last_health_check = lock_info.get('last_health_check', created_at)
            
            current_time = time.time()
            
            # Check if process is still running
            if lock_pid and not self._is_process_running(lock_pid):
                logger.debug(f"Lock file PID {lock_pid} is no longer running")
                return True
            
            # Check if lock is too old (older than 5 minutes without health check)
            if current_time - last_health_check > 300:  # 5 minutes
                logger.debug(f"Lock file is stale (no health check for {current_time - last_health_check:.1f}s)")
                return True
            
            # Check if lock is extremely old (older than 1 hour regardless)
            if current_time - created_at > 3600:  # 1 hour
                logger.debug(f"Lock file is extremely old ({current_time - created_at:.1f}s)")
                return True
            
            return False
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Invalid lock file, considering stale: {e}")
            return True
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is still running."""
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            # Fallback method if psutil not available
            try:
                os.kill(pid, 0)  # Send signal 0 to check if process exists
                return True
            except (OSError, ProcessLookupError):
                return False
    
    async def _cleanup_stale_lock(self, lock_file: Path) -> None:
        """Clean up a stale lock file safely."""
        try:
            # Read lock info for logging
            try:
                with open(lock_file, 'r') as f:
                    lock_info = json.load(f)
                logger.info(f"Cleaning up stale lock from PID {lock_info.get('pid', 'unknown')}")
            except:
                logger.info("Cleaning up corrupted lock file")
            
            # Remove the stale lock file
            lock_file.unlink(missing_ok=True)
            
            # Clear any stale client state
            if self._client:
                try:
                    self._client.close()
                except:
                    pass
                self._client = None
            
            self._connection_healthy = False
            logger.debug("Stale lock cleanup completed")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup stale lock: {e}")
    
    async def close(self):
        """Close the shared client and cleanup."""
        if self._client:
            try:
                self._client.close()
            except:
                pass
            self._client = None
        
        # Cleanup lock file
        if self._lock_file_path and os.path.exists(self._lock_file_path):
            try:
                with open(self._lock_file_path, 'r') as f:
                    lock_info = json.load(f)
                
                # Only remove if we created it
                if lock_info.get('pid') == os.getpid():
                    os.remove(self._lock_file_path)
            except:
                pass
        
        self._initialized = False


# Global instance
_shared_manager = SharedQdrantManager()


async def get_shared_qdrant_client(qdrant_path: str, timeout: float = 30.0) -> Any:
    """
    Get a shared Qdrant client that can be safely used by multiple Claude instances.
    
    Args:
        qdrant_path: Path to Qdrant storage directory
        timeout: Client timeout
        
    Returns:
        QdrantClient instance
    """
    return await _shared_manager.get_client(qdrant_path, timeout)


async def close_shared_qdrant_client():
    """Close the shared Qdrant client."""
    await _shared_manager.close()