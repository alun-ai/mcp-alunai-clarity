#!/usr/bin/env python3
"""
SQLite Performance Tests

Performance benchmarks for SQLite memory persistence:
- Storage performance (batch and individual)
- Search performance (various dataset sizes)
- Memory usage monitoring
- Concurrent access performance
- Cache effectiveness
- Database optimization validation
"""

import asyncio
import gc
import psutil
import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence


class TestSQLitePerformance:
    """Performance test suite for SQLite memory persistence."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="perf_test_")
        db_path = os.path.join(temp_dir, "perf_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Fast mock embedding model for performance testing."""
        mock_model = Mock()
        
        # Predictable fast embedding generation
        def fast_encode(text):
            # Simple deterministic embedding based on text hash
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
            embedding = [0.0] * 384
            
            # Set a few dimensions based on hash
            for i in range(min(10, len(text) // 10)):
                embedding[i] = (hash_val + i) / 10000.0
            
            return embedding
        
        mock_model.encode.side_effect = fast_encode
        return mock_model
    
    @pytest.fixture
    def sqlite_persistence(self, temp_db_path, mock_embedding_model):
        """Create SQLiteMemoryPersistence instance for testing."""
        return SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
    
    def generate_test_memories(self, count: int, content_length: int = 100) -> List[Dict[str, Any]]:
        """Generate test memories for performance testing."""
        memories = []
        memory_types = ["structured_thinking", "episodic", "procedural", "semantic"]
        tiers = ["short_term", "long_term", "archival", "system"]
        
        for i in range(count):
            content = f"Performance test memory {i} " + "content " * (content_length // 8)
            
            memory = {
                "id": f"perf-{i:06d}",
                "type": memory_types[i % len(memory_types)],
                "content": content[:content_length],
                "importance": 0.1 + (i % 10) * 0.09,  # 0.1 to 0.91
                "tier": tiers[i % len(tiers)],
                "metadata": {
                    "category": f"category_{i % 20}",
                    "batch": i // 100,
                    "index": i,
                    "priority": ["low", "medium", "high"][i % 3]
                }
            }
            memories.append(memory)
        
        return memories
    
    @pytest.mark.asyncio
    async def test_individual_storage_performance(self, sqlite_persistence):
        """Test individual memory storage performance."""
        test_memories = self.generate_test_memories(100, content_length=200)
        
        # Measure individual storage times
        storage_times = []
        
        for memory in test_memories[:20]:  # Test first 20 for detailed timing
            start_time = time.perf_counter()
            memory_id = await sqlite_persistence.store_memory(memory)
            storage_time = time.perf_counter() - start_time
            
            storage_times.append(storage_time * 1000)  # Convert to milliseconds
            assert memory_id is not None
        
        # Performance assertions
        avg_storage_time = sum(storage_times) / len(storage_times)
        max_storage_time = max(storage_times)
        
        print(f"\nIndividual Storage Performance:")
        print(f"  Average: {avg_storage_time:.2f}ms")
        print(f"  Maximum: {max_storage_time:.2f}ms")
        print(f"  Minimum: {min(storage_times):.2f}ms")
        
        # Performance targets
        assert avg_storage_time < 100.0  # Average under 100ms
        assert max_storage_time < 500.0  # No single storage over 500ms
    
    @pytest.mark.asyncio
    async def test_batch_storage_performance(self, sqlite_persistence):
        """Test batch storage performance."""
        batch_sizes = [10, 50, 100, 200]
        
        results = {}
        
        for batch_size in batch_sizes:
            test_memories = self.generate_test_memories(batch_size, content_length=150)
            
            start_time = time.perf_counter()
            
            # Store all memories in batch
            stored_ids = []
            for memory in test_memories:
                memory_id = await sqlite_persistence.store_memory(memory)
                stored_ids.append(memory_id)
            
            total_time = time.perf_counter() - start_time
            throughput = batch_size / total_time  # memories per second
            
            results[batch_size] = {
                "total_time": total_time,
                "throughput": throughput,
                "avg_per_memory": (total_time / batch_size) * 1000  # ms per memory
            }
            
            print(f"\nBatch Storage Performance ({batch_size} memories):")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} memories/sec")
            print(f"  Avg per memory: {results[batch_size]['avg_per_memory']:.2f}ms")
            
            # Verify all memories were stored
            assert len(stored_ids) == batch_size
            assert all(sid is not None for sid in stored_ids)
        
        # Performance assertions
        assert results[100]["throughput"] > 10  # At least 10 memories/sec for 100-memory batch
        assert results[200]["avg_per_memory"] < 200  # Under 200ms average for larger batches
    
    @pytest.mark.asyncio
    async def test_search_performance_scaling(self, sqlite_persistence):
        """Test search performance scaling with dataset size."""
        dataset_sizes = [100, 500, 1000, 2000]
        search_queries = [
            "performance test memory content",
            "database optimization analysis",
            "system architecture design",
            "procedural memory content"
        ]
        
        performance_results = {}
        
        for size in dataset_sizes:
            print(f"\nTesting search performance with {size} memories...")
            
            # Generate and store dataset
            test_memories = self.generate_test_memories(size, content_length=100)
            
            store_start = time.perf_counter()
            for memory in test_memories:
                await sqlite_persistence.store_memory(memory)
            store_time = time.perf_counter() - store_start
            
            print(f"  Dataset preparation: {store_time:.2f}s")
            
            # Test search performance
            search_times = []
            result_counts = []
            
            for query in search_queries:
                search_start = time.perf_counter()
                results = await sqlite_persistence.retrieve_memories(
                    query,
                    limit=10,
                    min_similarity=0.0
                )
                search_time = time.perf_counter() - search_start
                
                search_times.append(search_time * 1000)  # Convert to ms
                result_counts.append(len(results))
            
            # Calculate statistics
            avg_search_time = sum(search_times) / len(search_times)
            max_search_time = max(search_times)
            avg_results = sum(result_counts) / len(result_counts)
            
            performance_results[size] = {
                "avg_search_time": avg_search_time,
                "max_search_time": max_search_time,
                "avg_results": avg_results,
                "store_time": store_time
            }
            
            print(f"  Average search time: {avg_search_time:.2f}ms")
            print(f"  Maximum search time: {max_search_time:.2f}ms") 
            print(f"  Average results: {avg_results:.1f}")
        
        # Performance assertions
        # Search time should not grow too much with dataset size
        small_dataset_time = performance_results[100]["avg_search_time"]
        large_dataset_time = performance_results[2000]["avg_search_time"]
        
        # Search time should scale sub-linearly (better than O(n))
        scaling_factor = large_dataset_time / small_dataset_time
        dataset_scaling = 2000 / 100  # 20x dataset size
        
        print(f"\nScaling Analysis:")
        print(f"  Dataset size ratio: {dataset_scaling}x")
        print(f"  Search time ratio: {scaling_factor:.2f}x")
        print(f"  Scaling efficiency: {dataset_scaling / scaling_factor:.2f}")
        
        assert scaling_factor < dataset_scaling * 0.5  # Should scale better than linear
        assert large_dataset_time < 200  # Should stay under 200ms even for large datasets
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, sqlite_persistence):
        """Test memory usage during operations."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\nMemory Usage Monitoring:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        
        # Store memories and monitor usage
        memory_measurements = [initial_memory]
        
        for batch in range(5):  # 5 batches of 100 memories each
            batch_memories = self.generate_test_memories(100, content_length=200)
            
            for memory in batch_memories:
                await sqlite_persistence.store_memory(memory)
            
            # Force garbage collection and measure
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)
            
            print(f"  After batch {batch + 1}: {current_memory:.2f} MB "
                  f"(+{current_memory - initial_memory:.2f} MB)")
        
        # Test search memory usage
        search_start_memory = process.memory_info().rss / 1024 / 1024
        
        # Perform multiple searches
        for _ in range(20):
            await sqlite_persistence.retrieve_memories(
                "test search memory usage",
                limit=50,
                min_similarity=0.0
            )
        
        search_end_memory = process.memory_info().rss / 1024 / 1024
        print(f"  After 20 searches: {search_end_memory:.2f} MB "
              f"(+{search_end_memory - search_start_memory:.2f} MB)")
        
        # Memory growth should be reasonable
        total_growth = memory_measurements[-1] - initial_memory
        assert total_growth < 100  # Should not grow more than 100MB for 500 memories
        
        # Search should not cause significant memory leaks
        search_growth = search_end_memory - search_start_memory
        assert search_growth < 10  # Should not grow more than 10MB for searches
    
    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, sqlite_persistence):
        """Test concurrent access performance."""
        # Prepare initial dataset
        initial_memories = self.generate_test_memories(200, content_length=100)
        for memory in initial_memories:
            await sqlite_persistence.store_memory(memory)
        
        print(f"\nConcurrent Access Performance:")
        
        # Test concurrent reads
        async def concurrent_search(query_id):
            """Perform concurrent search operations."""
            results = []
            for i in range(10):  # 10 searches per thread
                search_results = await sqlite_persistence.retrieve_memories(
                    f"concurrent test query {query_id} {i}",
                    limit=5,
                    min_similarity=0.0
                )
                results.append(len(search_results))
            return results
        
        # Run concurrent searches
        concurrent_start = time.perf_counter()
        
        tasks = []
        for i in range(5):  # 5 concurrent search tasks
            task = asyncio.create_task(concurrent_search(i))
            tasks.append(task)
        
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = time.perf_counter() - concurrent_start
        
        print(f"  Concurrent searches: {concurrent_time:.3f}s")
        print(f"  Total search operations: {len(concurrent_results) * 10}")
        print(f"  Searches per second: {(len(concurrent_results) * 10) / concurrent_time:.1f}")
        
        # Verify all searches completed successfully
        for task_results in concurrent_results:
            assert len(task_results) == 10
            assert all(isinstance(r, int) and r >= 0 for r in task_results)
        
        # Test concurrent writes
        async def concurrent_store(batch_id):
            """Perform concurrent storage operations."""
            batch_memories = self.generate_test_memories(20, content_length=50)
            stored_ids = []
            
            for i, memory in enumerate(batch_memories):
                memory["id"] = f"concurrent-{batch_id}-{i:03d}"
                memory_id = await sqlite_persistence.store_memory(memory)
                stored_ids.append(memory_id)
            
            return stored_ids
        
        # Run concurrent storage operations
        storage_start = time.perf_counter()
        
        storage_tasks = []
        for i in range(3):  # 3 concurrent storage tasks
            task = asyncio.create_task(concurrent_store(i))
            storage_tasks.append(task)
        
        storage_results = await asyncio.gather(*storage_tasks)
        storage_time = time.perf_counter() - storage_start
        
        total_stored = sum(len(results) for results in storage_results)
        print(f"  Concurrent storage: {storage_time:.3f}s")
        print(f"  Total memories stored: {total_stored}")
        print(f"  Storage rate: {total_stored / storage_time:.1f} memories/sec")
        
        # Verify all storage operations completed successfully
        for task_results in storage_results:
            assert len(task_results) == 20
            assert all(sid is not None for sid in task_results)
        
        # Performance assertions
        assert (len(concurrent_results) * 10) / concurrent_time > 5  # At least 5 searches/sec
        assert total_stored / storage_time > 5  # At least 5 stores/sec
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, sqlite_persistence):
        """Test cache effectiveness for memory operations."""
        # Store test memories
        test_memories = self.generate_test_memories(50, content_length=100)
        stored_ids = []
        
        for memory in test_memories:
            memory_id = await sqlite_persistence.store_memory(memory)
            stored_ids.append(memory_id)
        
        print(f"\nCache Effectiveness Test:")
        
        # Test cache hit performance for get_memory
        test_ids = stored_ids[:10]  # Test with first 10 memories
        
        # First access (cache miss)
        first_access_times = []
        for memory_id in test_ids:
            start_time = time.perf_counter()
            memory = await sqlite_persistence.get_memory(memory_id)
            access_time = time.perf_counter() - start_time
            first_access_times.append(access_time * 1000)  # Convert to ms
            assert memory is not None
        
        avg_first_access = sum(first_access_times) / len(first_access_times)
        
        # Second access (cache hit)
        second_access_times = []
        for memory_id in test_ids:
            start_time = time.perf_counter()
            memory = await sqlite_persistence.get_memory(memory_id)
            access_time = time.perf_counter() - start_time
            second_access_times.append(access_time * 1000)  # Convert to ms
            assert memory is not None
        
        avg_second_access = sum(second_access_times) / len(second_access_times)
        
        print(f"  First access (cache miss): {avg_first_access:.3f}ms average")
        print(f"  Second access (cache hit): {avg_second_access:.3f}ms average")
        print(f"  Cache speedup: {avg_first_access / avg_second_access:.2f}x")
        
        # Cache should provide significant speedup
        assert avg_second_access < avg_first_access * 0.8  # At least 20% faster
        
        # Test embedding cache effectiveness
        test_text = "Cache effectiveness test for embeddings"
        
        # First embedding generation
        start_time = time.perf_counter()
        embedding1 = sqlite_persistence._generate_embedding(test_text)
        first_embedding_time = time.perf_counter() - start_time
        
        # Second embedding generation (should use cache)
        start_time = time.perf_counter()
        embedding2 = sqlite_persistence._generate_embedding(test_text)
        second_embedding_time = time.perf_counter() - start_time
        
        print(f"  First embedding: {first_embedding_time * 1000:.3f}ms")
        print(f"  Second embedding: {second_embedding_time * 1000:.3f}ms")
        print(f"  Embedding cache speedup: {first_embedding_time / max(second_embedding_time, 1e-6):.1f}x")
        
        assert embedding1 == embedding2  # Should be identical
        assert second_embedding_time < first_embedding_time * 0.5  # At least 50% faster
    
    @pytest.mark.asyncio
    async def test_database_optimization_validation(self, sqlite_persistence):
        """Test that database optimizations are working correctly."""
        import sqlite3
        
        print(f"\nDatabase Optimization Validation:")
        
        # Check database configuration
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            # Check WAL mode
            cursor = conn.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            print(f"  Journal mode: {journal_mode}")
            assert journal_mode.upper() == "WAL"
            
            # Check synchronous mode
            cursor = conn.execute("PRAGMA synchronous")
            sync_mode = cursor.fetchone()[0]
            print(f"  Synchronous mode: {sync_mode} (NORMAL)")
            assert sync_mode == 1  # NORMAL
            
            # Check cache size
            cursor = conn.execute("PRAGMA cache_size")
            cache_size = cursor.fetchone()[0]
            print(f"  Cache size: {cache_size} pages")
            assert cache_size == 10000
            
            # Check temp store
            cursor = conn.execute("PRAGMA temp_store")
            temp_store = cursor.fetchone()[0]
            print(f"  Temp store: {temp_store} (MEMORY)")
            assert temp_store == 2  # MEMORY
            
            # Check memory map size
            cursor = conn.execute("PRAGMA mmap_size")
            mmap_size = cursor.fetchone()[0]
            print(f"  Memory map size: {mmap_size // 1024 // 1024} MB")
            assert mmap_size == 268435456  # 256MB
        
        # Test index effectiveness
        # Store memories to populate database
        test_memories = self.generate_test_memories(500, content_length=100)
        for memory in test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Query with EXPLAIN QUERY PLAN to check index usage
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            # Test memory_type index usage
            cursor = conn.execute("""
                EXPLAIN QUERY PLAN 
                SELECT * FROM memories WHERE memory_type = 'structured_thinking'
            """)
            plan = cursor.fetchall()
            plan_text = " ".join(str(row[3]) for row in plan)
            print(f"  Type filter plan: {plan_text}")
            
            # Should use index (contains "idx_memory_type" or "SEARCH TABLE memories USING INDEX")
            assert "idx_memory_type" in plan_text or "USING INDEX" in plan_text
            
            # Test importance index usage
            cursor = conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM memories WHERE importance >= 0.8
            """)
            plan = cursor.fetchall()
            plan_text = " ".join(str(row[3]) for row in plan)
            print(f"  Importance filter plan: {plan_text}")
            
            # Test combined index usage
            cursor = conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM memories WHERE memory_type = 'episodic' AND tier = 'short_term'
            """)
            plan = cursor.fetchall()
            plan_text = " ".join(str(row[3]) for row in plan)
            print(f"  Combined filter plan: {plan_text}")
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, sqlite_persistence):
        """Test performance with larger datasets (stress test)."""
        print(f"\nLarge Dataset Performance Test:")
        
        # Generate and store large dataset
        large_dataset_size = 1000
        large_memories = self.generate_test_memories(large_dataset_size, content_length=300)
        
        # Measure storage time for large dataset
        storage_start = time.perf_counter()
        
        batch_size = 100
        for i in range(0, large_dataset_size, batch_size):
            batch = large_memories[i:i + batch_size]
            for memory in batch:
                await sqlite_persistence.store_memory(memory)
            
            print(f"  Stored {min(i + batch_size, large_dataset_size)}/{large_dataset_size} memories...")
        
        storage_time = time.perf_counter() - storage_start
        storage_throughput = large_dataset_size / storage_time
        
        print(f"  Storage completed: {storage_time:.2f}s")
        print(f"  Storage throughput: {storage_throughput:.1f} memories/sec")
        
        # Test search performance on large dataset
        search_queries = [
            "performance test memory large dataset",
            "content analysis system optimization",
            "database architecture design patterns",
            "procedural memory batch processing"
        ]
        
        search_times = []
        result_counts = []
        
        for query in search_queries:
            search_start = time.perf_counter()
            results = await sqlite_persistence.retrieve_memories(
                query,
                limit=20,
                min_similarity=0.0
            )
            search_time = time.perf_counter() - search_start
            
            search_times.append(search_time * 1000)  # Convert to ms
            result_counts.append(len(results))
            
            print(f"  Search '{query[:30]}...': {search_time * 1000:.2f}ms, {len(results)} results")
        
        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)
        
        print(f"  Average search time: {avg_search_time:.2f}ms")
        print(f"  Maximum search time: {max_search_time:.2f}ms")
        
        # Get database statistics
        stats = await sqlite_persistence.get_memory_stats()
        print(f"  Database size: {stats.get('database_size_bytes', 0) / 1024 / 1024:.2f} MB")
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        
        # Performance assertions for large dataset
        assert storage_throughput > 5  # At least 5 memories/sec for large batches
        assert avg_search_time < 100  # Average search under 100ms
        assert max_search_time < 300  # No single search over 300ms
        assert all(count > 0 for count in result_counts)  # All searches should return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements