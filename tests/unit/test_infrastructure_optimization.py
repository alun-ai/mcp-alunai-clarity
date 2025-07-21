"""
Unit tests for infrastructure optimization components in Alunai Clarity.

Tests the caching infrastructure, connection pooling, and other optimizations
implemented during the codebase optimization phase.
"""

import asyncio
import pytest
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from clarity.shared.infrastructure.cache import InMemoryCache, get_cache
from clarity.shared.infrastructure.connection_pool import QdrantConnectionPool, qdrant_connection
from clarity.shared.exceptions import QdrantConnectionError, MemoryOperationError


@pytest.mark.unit
class TestCacheInfrastructure:
    """Test caching infrastructure components."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations (get, set, delete)."""
        cache = InMemoryCache[str](max_size=100, default_ttl=3600.0)
        
        # Test set and get
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test default value
        assert cache.get("nonexistent_key", "default") == "default"
        
        # Test delete
        assert cache.delete("test_key") is True
        assert cache.get("test_key") is None
        
        # Test delete nonexistent key
        assert cache.delete("nonexistent_key") is False
    
    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        cache = InMemoryCache[str](max_size=100, default_ttl=0.1)  # 100ms TTL
        
        cache.set("ttl_key", "ttl_value")
        assert cache.get("ttl_key") == "ttl_value"
        
        # Wait for TTL expiration
        time.sleep(0.15)
        assert cache.get("ttl_key") is None
    
    def test_cache_custom_ttl(self):
        """Test custom TTL per key."""
        cache = InMemoryCache[str](max_size=100, default_ttl=3600.0)
        
        # Set with short custom TTL
        cache.set("short_ttl", "value", ttl=0.1)
        cache.set("long_ttl", "value", ttl=1.0)
        
        time.sleep(0.15)
        assert cache.get("short_ttl") is None
        assert cache.get("long_ttl") == "value"
    
    def test_cache_size_limits_lru(self):
        """Test LRU eviction when cache reaches size limit."""
        cache = InMemoryCache[str](max_size=3, eviction_policy="lru")
        
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2") 
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add another key - should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Still present
        assert cache.get("key4") == "value4"  # New key
    
    def test_cache_size_limits_lfu(self):
        """Test LFU eviction when cache reaches size limit."""
        cache = InMemoryCache[str](max_size=3, eviction_policy="lfu")
        
        # Fill cache and establish frequency
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 multiple times
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")  # Access once
        
        # Add another key - should evict key3 (least frequently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Most frequent
        assert cache.get("key2") == "value2"  # Still present
        assert cache.get("key3") is None      # Evicted (never accessed)
        assert cache.get("key4") == "value4"  # New key
    
    def test_cache_memory_limits(self):
        """Test memory-based eviction."""
        cache = InMemoryCache[str](max_size=1000, max_memory_mb=0.001)  # 1KB limit
        
        # Add large values that should trigger memory eviction
        large_value = "x" * 1000  # 1KB value
        
        cache.set("key1", large_value)
        cache.set("key2", large_value)  # Should trigger eviction of key1
        
        # Only the most recent key should remain
        assert cache.get("key1") is None
        assert cache.get("key2") == large_value
    
    def test_cache_statistics(self):
        """Test cache performance statistics."""
        cache = InMemoryCache[str](max_size=100)
        
        # Generate cache activity
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert "evictions" in stats
        assert "memory_usage_mb" in stats
    
    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = InMemoryCache[str](max_size=100)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.size == 2
        
        cache.clear()
        
        assert cache.size == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_global_cache_factory(self):
        """Test global cache factory function."""
        cache1 = get_cache("test_cache", max_size=50, default_ttl=1800.0)
        cache2 = get_cache("test_cache")  # Should return same instance
        
        cache1.set("shared_key", "shared_value")
        assert cache2.get("shared_key") == "shared_value"
        
        # Different cache name should return different instance
        cache3 = get_cache("other_cache")
        assert cache3.get("shared_key") is None


@pytest.mark.unit
class TestConnectionPooling:
    """Test Qdrant connection pooling implementation."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = MagicMock()
        client.close = AsyncMock()
        client.get_collections = MagicMock()
        return client
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    def test_connection_pool_creation(self, mock_qdrant_client_class):
        """Test connection pool creation and configuration."""
        from clarity.shared.infrastructure.connection_pool import ConnectionPoolConfig
        
        config = ConnectionPoolConfig(
            min_connections=2,
            max_connections=5,
            connection_timeout=10.0,
            max_retries=3,
            retry_delay=1.0
        )
        
        pool = QdrantConnectionPool(":memory:", config)
        
        assert pool.config.min_connections == 2
        assert pool.config.max_connections == 5
        assert pool.config.connection_timeout == 10.0
        assert len(pool._pool) == 0  # Not initialized yet
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test connection pool initialization."""
        from clarity.shared.infrastructure.connection_pool import ConnectionPoolConfig
        
        mock_qdrant_client_class.return_value = mock_qdrant_client
        
        config = ConnectionPoolConfig(min_connections=2, max_connections=5)
        pool = QdrantConnectionPool(":memory:", config)
        
        await pool.initialize()
        
        assert len(pool._pool) == 2  # Should create min_connections
        assert pool._stats['total_connections'] == 2
        assert mock_qdrant_client_class.call_count == 2
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    @pytest.mark.asyncio
    async def test_connection_acquisition_and_release(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test acquiring and releasing connections."""
        from clarity.shared.infrastructure.connection_pool import ConnectionPoolConfig
        
        mock_qdrant_client_class.return_value = mock_qdrant_client
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = QdrantConnectionPool(":memory:", config)
        
        await pool.initialize()
        
        # Acquire a connection
        async with pool.acquire() as conn:
            assert conn is not None
            assert len(pool._pool) == 0  # Connection removed from pool
        
        # Connection should be returned to pool
        assert len(pool._pool) == 1
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    @pytest.mark.asyncio
    async def test_connection_pool_expansion(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test connection pool expansion when needed."""
        from clarity.shared.infrastructure.connection_pool import ConnectionPoolConfig
        
        mock_qdrant_client_class.return_value = mock_qdrant_client
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = QdrantConnectionPool(":memory:", config)
        
        await pool.initialize()
        
        # Acquire all connections plus one more (should create new connection)
        connections = []
        for _ in range(2):  # More than min_connections
            conn = await pool.acquire().__aenter__()
            connections.append(conn)
        
        assert pool._stats['total_connections'] == 2
        
        # Clean up
        for conn in connections:
            await pool.acquire().__aexit__(None, None, None)
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    @pytest.mark.asyncio
    async def test_connection_pool_health_check(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test connection health checking."""
        from clarity.shared.infrastructure.connection_pool import ConnectionPoolConfig
        
        mock_qdrant_client_class.return_value = mock_qdrant_client
        # Mock a healthy connection
        mock_qdrant_client.get_collections.return_value = MagicMock()
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = QdrantConnectionPool(":memory:", config)
        
        await pool.initialize()
        
        # Get a connection from pool
        connection = pool._pool[0]
        
        # Health check should pass
        is_healthy = await pool._health_check(connection)
        assert is_healthy is True
        
        # Mock unhealthy connection
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")
        
        is_healthy = await pool._health_check(connection)
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_qdrant_connection_context_manager(self):
        """Test the qdrant_connection context manager."""
        with patch('clarity.shared.infrastructure.connection_pool._connection_pool') as mock_pool:
            mock_pool.acquire = AsyncMock()
            mock_connection = MagicMock()
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock()
            
            async with qdrant_connection() as conn:
                assert conn == mock_connection
            
            mock_pool.acquire.assert_called_once()


@pytest.mark.unit
class TestExceptionHandling:
    """Test specific exception handling improvements."""
    
    def test_memory_operation_error_hierarchy(self):
        """Test custom exception hierarchy."""
        from clarity.shared.exceptions import MemoryOperationError, ClarityException
        
        error = MemoryOperationError(
            "Test error",
            error_code="TEST_001",
            context={"operation": "store"},
            cause=ValueError("Original error")
        )
        
        assert isinstance(error, ClarityException)
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.context["operation"] == "store"
        assert isinstance(error.cause, ValueError)
        assert error.timestamp is not None
    
    def test_qdrant_connection_error_context(self):
        """Test Qdrant-specific error handling."""
        from clarity.shared.exceptions import QdrantConnectionError
        
        error = QdrantConnectionError(
            "Connection failed",
            context={"host": "localhost", "port": 6333}
        )
        
        assert error.message == "Connection failed"
        assert error.context["host"] == "localhost"
        assert error.context["port"] == 6333
    
    def test_exception_serialization(self):
        """Test exception serialization for logging."""
        from clarity.shared.exceptions import AutoCodeError
        
        error = AutoCodeError(
            "Pattern detection failed",
            error_code="AUTOCODE_001",
            context={"project_path": "/test/project"},
            cause=FileNotFoundError("File not found")
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "AUTOCODE_001"
        assert error_dict["message"] == "Pattern detection failed"
        assert error_dict["context"]["project_path"] == "/test/project"
        assert error_dict["timestamp"] is not None


@pytest.mark.unit
class TestSharedUtilities:
    """Test shared utility functions."""
    
    def test_json_response_builder(self):
        """Test JSON response builder utility."""
        from clarity.shared.utils.json_responses import build_success_response, build_error_response
        
        # Test success response
        success = build_success_response(
            {"memory_id": "test_123"}, 
            message="Memory stored successfully"
        )
        
        assert success["success"] is True
        assert success["data"]["memory_id"] == "test_123"
        assert success["message"] == "Memory stored successfully"
        
        # Test error response
        error = build_error_response(
            "Memory not found", 
            error_code="NOT_FOUND"
        )
        
        assert error["success"] is False
        assert error["error"]["message"] == "Memory not found"
        assert error["error"]["code"] == "NOT_FOUND"
    
    def test_config_manager_caching(self):
        """Test configuration manager caching."""
        from clarity.shared.utils.config_manager import ConfigManager
        
        # Mock file system operations
        with patch('builtins.open'), patch('json.load') as mock_json_load, patch('os.path.getmtime') as mock_getmtime:
            mock_json_load.return_value = {"test": "config"}
            mock_getmtime.return_value = 1234567890
            
            manager = ConfigManager("/fake/path/config.json")
            
            # First load should read from file
            config1 = manager.get_config()
            assert config1["test"] == "config"
            assert mock_json_load.call_count == 1
            
            # Second load should use cache (same mtime)
            config2 = manager.get_config()
            assert config2 == config1
            assert mock_json_load.call_count == 1  # Not called again
            
            # Simulate file modification
            mock_getmtime.return_value = 1234567891
            config3 = manager.get_config()
            assert mock_json_load.call_count == 2  # Called again due to file change


@pytest.mark.unit
class TestDomainInterfaces:
    """Test domain interface implementations."""
    
    def test_memory_storage_interface_contract(self):
        """Test that domain adapters implement the storage interface correctly."""
        from clarity.domains.interfaces import MemoryStorageInterface
        from clarity.domains.adapters import MemoryStorageAdapter
        
        # Mock persistence domain
        mock_persistence = AsyncMock()
        adapter = MemoryStorageAdapter(mock_persistence)
        
        # Should implement the interface
        assert isinstance(adapter, MemoryStorageInterface)
        
        # Check method signatures exist
        assert hasattr(adapter, 'store_memory')
        assert hasattr(adapter, 'retrieve_memories')
        assert hasattr(adapter, 'get_memory')
        assert hasattr(adapter, 'update_memory')
        assert hasattr(adapter, 'delete_memories')
    
    def test_domain_registry_functionality(self):
        """Test domain registry and dependency injection."""
        from clarity.domains.registry import DomainRegistryImpl
        from clarity.domains.interfaces import MemoryDomainInterface
        
        registry = DomainRegistryImpl()
        
        # Mock domain
        mock_domain = MagicMock(spec=MemoryDomainInterface)
        
        # Test registration
        registry.register_domain("test_domain", mock_domain)
        
        # Test retrieval
        retrieved_domain = registry.get_domain("test_domain")
        assert retrieved_domain == mock_domain
        
        # Test dependency management
        registry.register_dependency("dependent_domain", ["test_domain"])
        
        init_order = registry.get_initialization_order()
        # test_domain should come before dependent_domain
        test_index = init_order.index("test_domain")
        dependent_index = init_order.index("dependent_domain")
        assert test_index < dependent_index