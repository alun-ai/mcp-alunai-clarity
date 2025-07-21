"""
Performance tests for codebase optimizations in Alunai Clarity.

Tests performance improvements from:
- Connection pooling
- Caching infrastructure  
- Modular architecture
- Optimized exception handling
"""

import asyncio
import pytest
import time
import numpy as np
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import concurrent.futures
import statistics

from clarity.shared.infrastructure.cache import InMemoryCache
from clarity.shared.infrastructure.connection_pool import QdrantConnectionPool, ConnectionPoolConfig
from clarity.autocode.domain_refactored import AutoCodeDomainRefactored


@pytest.mark.performance
class TestCachePerformance:
    """Performance tests for caching infrastructure."""
    
    def test_cache_memory_efficiency(self):
        """Test cache memory usage efficiency."""
        cache = InMemoryCache[str](max_size=10000, max_memory_mb=10)
        
        # Fill cache with data
        test_data = "x" * 100  # 100 bytes per item
        start_memory = cache.get_stats()["memory_usage_mb"]
        
        for i in range(1000):
            cache.set(f"key_{i}", test_data)
        
        end_memory = cache.get_stats()["memory_usage_mb"]
        memory_growth = end_memory - start_memory
        
        # Should use approximately 100KB (0.1MB) for 1000 * 100 byte items
        assert memory_growth < 1.0, f"Memory usage too high: {memory_growth}MB"
    
    def test_cache_access_speed(self):
        """Test cache access performance."""
        cache = InMemoryCache[str](max_size=10000)
        
        # Populate cache
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Time cache access
        start_time = time.perf_counter()
        
        for i in range(10000):  # 10K accesses
            cache.get(f"key_{i % 1000}")
        
        elapsed = time.perf_counter() - start_time
        ops_per_second = 10000 / elapsed
        
        # Should achieve >100K ops/second
        assert ops_per_second > 100000, f"Cache too slow: {ops_per_second:.0f} ops/sec"
    
    def test_cache_eviction_performance(self):
        """Test performance during cache eviction."""
        cache = InMemoryCache[str](max_size=100, eviction_policy="lru")
        
        # Fill cache to capacity
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Time eviction operations
        start_time = time.perf_counter()
        
        # Add items that will trigger eviction
        for i in range(100, 1100):  # 1000 evictions
            cache.set(f"key_{i}", f"value_{i}")
        
        elapsed = time.perf_counter() - start_time
        ops_per_second = 1000 / elapsed
        
        # Should maintain >10K ops/second even with eviction
        assert ops_per_second > 10000, f"Eviction too slow: {ops_per_second:.0f} ops/sec"
    
    def test_concurrent_cache_access(self):
        """Test cache performance under concurrent access."""
        cache = InMemoryCache[str](max_size=10000)
        
        # Populate cache
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        
        def cache_worker():
            operations = 1000
            start_time = time.perf_counter()
            
            for i in range(operations):
                cache.get(f"key_{i % 1000}")
            
            elapsed = time.perf_counter() - start_time
            return operations / elapsed
        
        # Test with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cache_worker) for _ in range(4)]
            results = [future.result() for future in futures]
        
        avg_throughput = statistics.mean(results)
        
        # Should maintain good performance under concurrent load
        assert avg_throughput > 50000, f"Concurrent performance too low: {avg_throughput:.0f} ops/sec"


@pytest.mark.performance
class TestConnectionPoolPerformance:
    """Performance tests for connection pooling."""
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    @pytest.mark.asyncio
    async def test_connection_pool_throughput(self, mock_qdrant_client_class):
        """Test connection pool throughput."""
        # Mock client creation
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_qdrant_client_class.return_value = mock_client
        
        config = ConnectionPoolConfig(
            min_connections=5,
            max_connections=20,
            connection_timeout=1.0
        )
        
        pool = QdrantConnectionPool(":memory:", config)
        await pool.initialize()
        
        async def connection_worker():
            operations = 100
            start_time = time.perf_counter()
            
            for _ in range(operations):
                async with pool.acquire() as conn:
                    # Simulate database operation
                    await asyncio.sleep(0.001)  # 1ms operation
            
            elapsed = time.perf_counter() - start_time
            return operations / elapsed
        
        # Test concurrent connection usage
        tasks = [connection_worker() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        avg_throughput = statistics.mean(results)
        
        # Should achieve good concurrent throughput
        assert avg_throughput > 50, f"Connection pool throughput too low: {avg_throughput:.1f} ops/sec"
        
        await pool.shutdown()
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    @pytest.mark.asyncio
    async def test_connection_acquisition_latency(self, mock_qdrant_client_class):
        """Test connection acquisition latency."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_qdrant_client_class.return_value = mock_client
        
        config = ConnectionPoolConfig(
            min_connections=10,
            max_connections=20
        )
        
        pool = QdrantConnectionPool(":memory:", config)
        await pool.initialize()
        
        # Measure acquisition times
        acquisition_times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            async with pool.acquire() as conn:
                elapsed = time.perf_counter() - start_time
                acquisition_times.append(elapsed)
        
        avg_latency = statistics.mean(acquisition_times)
        p95_latency = np.percentile(acquisition_times, 95)
        
        # Should have low acquisition latency
        assert avg_latency < 0.001, f"Average latency too high: {avg_latency*1000:.1f}ms"
        assert p95_latency < 0.005, f"P95 latency too high: {p95_latency*1000:.1f}ms"
        
        await pool.shutdown()
    
    @patch('clarity.shared.infrastructure.connection_pool.QdrantClient')
    @pytest.mark.asyncio
    async def test_connection_pool_scaling(self, mock_qdrant_client_class):
        """Test connection pool scaling under load."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_qdrant_client_class.return_value = mock_client
        
        config = ConnectionPoolConfig(
            min_connections=2,
            max_connections=10
        )
        
        pool = QdrantConnectionPool(":memory:", config)
        await pool.initialize()
        
        initial_connections = pool._stats['total_connections']
        
        async def load_generator():
            async with pool.acquire() as conn:
                await asyncio.sleep(0.1)  # Hold connection briefly
        
        # Generate concurrent load
        tasks = [load_generator() for _ in range(8)]
        await asyncio.gather(*tasks)
        
        final_connections = pool._stats['total_connections']
        
        # Pool should have scaled up
        assert final_connections > initial_connections
        assert final_connections <= config.max_connections
        
        await pool.shutdown()


@pytest.mark.performance
class TestAutoCodePerformance:
    """Performance tests for refactored AutoCode domain."""
    
    @pytest.fixture
    def test_config(self):
        """Performance test configuration."""
        return {
            "autocode": {
                "enabled": True,
                "pattern_detection": {"enabled": True, "max_scan_depth": 2},
                "session_analysis": {"enabled": True},
                "command_learning": {"enabled": True},
                "stats_collection": {"enabled": True}
            }
        }
    
    @pytest.fixture
    def mock_persistence_domain(self):
        """Fast mock persistence domain."""
        domain = AsyncMock()
        domain.store_memory = AsyncMock(return_value="perf_test_123")
        domain.retrieve_memories = AsyncMock(return_value=[])
        domain.generate_embedding = AsyncMock(return_value=[0.1] * 384)
        return domain
    
    @pytest.mark.asyncio
    async def test_autocode_initialization_time(self, test_config, mock_persistence_domain):
        """Test AutoCode domain initialization performance."""
        domain = AutoCodeDomainRefactored(test_config, mock_persistence_domain)
        
        start_time = time.perf_counter()
        await domain.initialize()
        elapsed = time.perf_counter() - start_time
        
        # Should initialize quickly
        assert elapsed < 1.0, f"Initialization too slow: {elapsed:.3f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_operation_throughput(self, test_config, mock_persistence_domain):
        """Test concurrent operation performance."""
        domain = AutoCodeDomainRefactored(test_config, mock_persistence_domain)
        await domain.initialize()
        
        async def operation_worker():
            operations = 50
            start_time = time.perf_counter()
            
            for i in range(operations):
                await domain.process_bash_execution(
                    f"test_command_{i}",
                    "/test/dir",
                    True,
                    "success"
                )
            
            elapsed = time.perf_counter() - start_time
            return operations / elapsed
        
        # Run concurrent operations
        tasks = [operation_worker() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        avg_throughput = statistics.mean(results)
        
        # Should handle concurrent operations efficiently
        assert avg_throughput > 100, f"Throughput too low: {avg_throughput:.1f} ops/sec"
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, test_config, mock_persistence_domain):
        """Test memory usage during high-load operations."""
        import psutil
        import os
        
        domain = AutoCodeDomainRefactored(test_config, mock_persistence_domain)
        await domain.initialize()
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        for i in range(1000):
            await domain.suggest_command(f"test_intent_{i % 10}")
            
            # Occasional garbage collection simulation
            if i % 100 == 0:
                import gc
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Should not grow memory excessively
        assert memory_growth < 50, f"Memory growth too high: {memory_growth:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_stats_collection_overhead(self, test_config, mock_persistence_domain):
        """Test performance overhead of stats collection."""
        domain = AutoCodeDomainRefactored(test_config, mock_persistence_domain)
        await domain.initialize()
        
        # Measure performance with stats collection
        start_time = time.perf_counter()
        
        for _ in range(1000):
            await domain.suggest_command("test_command")
        
        elapsed_with_stats = time.perf_counter() - start_time
        
        # Disable stats collection
        domain.stats_collector = MagicMock()
        domain.stats_collector.track_operation = AsyncMock()
        
        start_time = time.perf_counter()
        
        for _ in range(1000):
            await domain.suggest_command("test_command")
        
        elapsed_without_stats = time.perf_counter() - start_time
        
        # Stats collection overhead should be minimal
        overhead_ratio = elapsed_with_stats / elapsed_without_stats
        assert overhead_ratio < 1.2, f"Stats overhead too high: {overhead_ratio:.2f}x"


@pytest.mark.performance
class TestExceptionHandlingPerformance:
    """Test performance impact of improved exception handling."""
    
    def test_exception_creation_performance(self):
        """Test performance of custom exception creation."""
        from clarity.shared.exceptions import MemoryOperationError
        
        start_time = time.perf_counter()
        
        # Create many exception instances
        exceptions = []
        for i in range(10000):
            exc = MemoryOperationError(
                f"Test error {i}",
                error_code=f"TEST_{i:04d}",
                context={"operation": "test", "index": i}
            )
            exceptions.append(exc)
        
        elapsed = time.perf_counter() - start_time
        creations_per_second = 10000 / elapsed
        
        # Should be able to create exceptions quickly
        assert creations_per_second > 50000, f"Exception creation too slow: {creations_per_second:.0f}/sec"
    
    def test_exception_handling_vs_generic(self):
        """Compare performance of specific vs generic exception handling."""
        from clarity.shared.exceptions import MemoryOperationError, ValidationError
        
        def generic_handler():
            try:
                raise ValueError("Test error")
            except Exception as e:
                return str(e)
        
        def specific_handler():
            try:
                raise ValueError("Test error") 
            except (ValueError, TypeError, KeyError) as e:
                return str(e)
        
        # Time generic handling
        start_time = time.perf_counter()
        for _ in range(10000):
            generic_handler()
        generic_time = time.perf_counter() - start_time
        
        # Time specific handling
        start_time = time.perf_counter()
        for _ in range(10000):
            specific_handler()
        specific_time = time.perf_counter() - start_time
        
        # Specific handling should not be significantly slower
        performance_ratio = specific_time / generic_time
        assert performance_ratio < 1.5, f"Specific exception handling too slow: {performance_ratio:.2f}x"


@pytest.mark.performance
@pytest.mark.slow
class TestIntegratedPerformance:
    """End-to-end performance tests for optimized system."""
    
    @pytest.fixture
    async def optimized_system(self):
        """Set up complete optimized system."""
        from clarity.domains.manager import MemoryDomainManager
        
        config = {
            "server": {"host": "localhost", "port": 8080},
            "alunai-clarity": {
                "max_short_term_items": 1000,
                "max_long_term_items": 5000,
                "short_term_threshold": 0.3
            },
            "qdrant": {
                "path": ":memory:",
                "index_params": {"m": 16, "ef_construct": 100, "full_scan_threshold": 1000}
            },
            "embedding": {
                "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384
            },
            "autocode": {"enabled": True}
        }
        
        # Mock the heavy dependencies
        with patch('clarity.domains.persistence.QdrantClient'), \
             patch('clarity.utils.embeddings.SentenceTransformer') as mock_model:
            
            mock_model.return_value.encode.return_value = np.random.rand(384)
            
            manager = MemoryDomainManager(config)
            manager.persistence_domain.qdrant_client = AsyncMock()
            manager.persistence_domain.embedding_manager.get_embedding = MagicMock(
                return_value=np.random.rand(384)
            )
            
            await manager.initialize()
            yield manager
    
    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, optimized_system):
        """Test performance with mixed memory operations."""
        manager = optimized_system
        
        start_time = time.perf_counter()
        
        # Mixed workload simulation
        tasks = []
        
        # Storage operations (50%)
        for i in range(500):
            task = manager.store_memory(
                memory_type="fact",
                content={"fact": f"Test fact {i}"},
                importance=0.5
            )
            tasks.append(task)
        
        # Retrieval operations (30%) 
        for i in range(300):
            task = manager.retrieve_memories(
                query=f"test query {i % 10}",
                limit=5
            )
            tasks.append(task)
        
        # Update operations (20%)
        for i in range(200):
            task = manager.update_memory(
                memory_id=f"mem_{i % 100}",
                updates={"importance": 0.8}
            )
            tasks.append(task)
        
        # Execute all operations concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.perf_counter() - start_time
        total_ops = 1000
        ops_per_second = total_ops / elapsed
        
        # Should handle mixed workload efficiently
        assert ops_per_second > 100, f"Mixed workload too slow: {ops_per_second:.1f} ops/sec"
    
    @pytest.mark.asyncio
    async def test_system_scalability(self, optimized_system):
        """Test system performance scaling."""
        manager = optimized_system
        
        # Test with increasing load
        load_sizes = [100, 500, 1000, 2000]
        performance_results = []
        
        for load_size in load_sizes:
            start_time = time.perf_counter()
            
            # Generate load
            tasks = []
            for i in range(load_size):
                if i % 3 == 0:
                    task = manager.store_memory(
                        memory_type="conversation",
                        content={"message": f"Message {i}"},
                        importance=0.5
                    )
                else:
                    task = manager.retrieve_memories(f"query {i % 10}")
                
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            elapsed = time.perf_counter() - start_time
            ops_per_second = load_size / elapsed
            performance_results.append(ops_per_second)
        
        # Performance should not degrade significantly with scale
        performance_degradation = performance_results[0] / performance_results[-1]
        assert performance_degradation < 3.0, f"Performance degrades too much: {performance_degradation:.1f}x"
    
    @pytest.mark.asyncio
    async def test_concurrent_client_simulation(self, optimized_system):
        """Simulate multiple concurrent clients."""
        manager = optimized_system
        
        async def client_simulation(client_id: int):
            """Simulate a client session."""
            operations = 100
            start_time = time.perf_counter()
            
            for i in range(operations):
                if i % 4 == 0:
                    await manager.store_memory(
                        memory_type="conversation",
                        content={"client": client_id, "message": f"Message {i}"},
                        importance=0.5
                    )
                else:
                    await manager.retrieve_memories(f"client {client_id} query {i}")
            
            elapsed = time.perf_counter() - start_time
            return operations / elapsed
        
        # Simulate 10 concurrent clients
        tasks = [client_simulation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        avg_client_performance = statistics.mean(results)
        min_client_performance = min(results)
        
        # All clients should maintain reasonable performance
        assert avg_client_performance > 50, f"Average client performance too low: {avg_client_performance:.1f} ops/sec"
        assert min_client_performance > 20, f"Worst client performance too low: {min_client_performance:.1f} ops/sec"