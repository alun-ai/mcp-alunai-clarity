#!/usr/bin/env python3
"""
SQLite Concurrent Access Tests

Integration tests for concurrent access to SQLite memory persistence:
- Multi-process safety
- Multi-threaded operations
- WAL mode validation
- Lock handling and deadlock prevention
- Data consistency under concurrent load
- Performance under concurrent access
"""

import asyncio
import multiprocessing
import pytest
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from unittest.mock import Mock
from typing import Any, Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence


class TestSQLiteConcurrentAccess:
    """Test suite for SQLite concurrent access functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="concurrent_test_")
        db_path = os.path.join(temp_dir, "concurrent_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Fast mock embedding model for concurrent testing."""
        mock_model = Mock()
        
        def fast_encode(text):
            # Very fast deterministic embedding
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000
            return [hash_val / 1000.0] * 384
        
        mock_model.encode.side_effect = fast_encode
        return mock_model
    
    @pytest.fixture
    def sqlite_persistence(self, temp_db_path, mock_embedding_model):
        """Create SQLiteMemoryPersistence instance for testing."""
        return SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
    
    def generate_test_memory(self, index: int, thread_id: str = "default") -> Dict[str, Any]:
        """Generate test memory for concurrent operations."""
        return {
            "id": f"concurrent-{thread_id}-{index:04d}",
            "type": ["structured_thinking", "episodic", "procedural", "semantic"][index % 4],
            "content": f"Concurrent access test memory {index} from thread {thread_id}",
            "importance": 0.1 + (index % 10) * 0.09,
            "tier": ["short_term", "long_term", "archival", "system"][index % 4],
            "metadata": {
                "thread_id": thread_id,
                "index": index,
                "test_type": "concurrent_access"
            }
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_writes(self, sqlite_persistence):
        """Test concurrent write operations."""
        print("\nTesting concurrent writes...")
        
        async def concurrent_writer(writer_id: str, count: int) -> List[str]:
            """Write memories concurrently."""
            stored_ids = []
            
            for i in range(count):
                memory = self.generate_test_memory(i, writer_id)
                try:
                    memory_id = await sqlite_persistence.store_memory(memory)
                    stored_ids.append(memory_id)
                except Exception as e:
                    print(f"Write error in {writer_id}: {e}")
            
            return stored_ids
        
        # Run concurrent writers
        start_time = time.perf_counter()
        
        tasks = []
        for writer_id in range(5):  # 5 concurrent writers
            task = asyncio.create_task(concurrent_writer(f"writer_{writer_id}", 20))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        write_time = time.perf_counter() - start_time
        
        # Verify results
        total_written = 0
        total_errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Writer {i} failed: {result}")
                total_errors += 1
            else:
                total_written += len(result)
                print(f"Writer {i}: {len(result)} memories written")
        
        print(f"Total written: {total_written}, Errors: {total_errors}, Time: {write_time:.3f}s")
        print(f"Write rate: {total_written / write_time:.1f} memories/sec")
        
        # Verify data integrity
        stats = await sqlite_persistence.get_memory_stats()
        assert stats["total_memories"] >= total_written * 0.9  # Allow some tolerance
        
        # Performance assertions
        assert total_written > 50  # Should write most memories
        assert total_errors < 2  # Should have minimal errors
        assert write_time < 30  # Should complete within reasonable time
    
    @pytest.mark.asyncio
    async def test_concurrent_reads(self, sqlite_persistence):
        """Test concurrent read operations."""
        print("\nTesting concurrent reads...")
        
        # First, populate database with test data
        test_memories = []
        for i in range(50):
            memory = self.generate_test_memory(i, "setup")
            await sqlite_persistence.store_memory(memory)
            test_memories.append(memory)
        
        async def concurrent_reader(reader_id: str, query_count: int) -> List[int]:
            """Read memories concurrently."""
            read_counts = []
            
            for i in range(query_count):
                try:
                    # Vary queries to test different access patterns
                    if i % 3 == 0:
                        # Search by content
                        results = await sqlite_persistence.retrieve_memories(
                            f"concurrent access test memory {reader_id}",
                            limit=10,
                            min_similarity=0.0
                        )
                    elif i % 3 == 1:
                        # Search by type
                        results = await sqlite_persistence.search_memories(
                            types=["episodic", "structured_thinking"],
                            limit=15
                        )
                    else:
                        # Search with filters
                        results = await sqlite_persistence.search_memories(
                            filters={"tier": "short_term"},
                            limit=20
                        )
                    
                    read_counts.append(len(results))
                    
                except Exception as e:
                    print(f"Read error in {reader_id}: {e}")
                    read_counts.append(0)
            
            return read_counts
        
        # Run concurrent readers
        start_time = time.perf_counter()
        
        tasks = []
        for reader_id in range(8):  # 8 concurrent readers
            task = asyncio.create_task(concurrent_reader(f"reader_{reader_id}", 15))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        read_time = time.perf_counter() - start_time
        
        # Verify results
        total_reads = 0
        total_results = 0
        read_errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Reader {i} failed: {result}")
                read_errors += 1
            else:
                reader_reads = len(result)
                reader_results = sum(result)
                total_reads += reader_reads
                total_results += reader_results
                print(f"Reader {i}: {reader_reads} queries, {reader_results} total results")
        
        print(f"Total queries: {total_reads}, Total results: {total_results}, Errors: {read_errors}")
        print(f"Read rate: {total_reads / read_time:.1f} queries/sec")
        print(f"Average results per query: {total_results / max(total_reads, 1):.1f}")
        
        # Performance assertions
        assert total_reads >= 100  # Should complete most queries
        assert read_errors == 0  # Should have no read errors
        assert total_results > 0  # Should find results
        assert read_time < 15  # Should complete within reasonable time
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write_mix(self, sqlite_persistence):
        """Test mixed concurrent read and write operations."""
        print("\nTesting concurrent read/write mix...")
        
        # Pre-populate with some data
        for i in range(20):
            memory = self.generate_test_memory(i, "initial")
            await sqlite_persistence.store_memory(memory)
        
        async def mixed_worker(worker_id: str, operations: int) -> Dict[str, int]:
            """Perform mixed read/write operations."""
            results = {"reads": 0, "writes": 0, "updates": 0, "errors": 0}
            
            for i in range(operations):
                try:
                    if i % 3 == 0:
                        # Write operation
                        memory = self.generate_test_memory(i, f"mixed_{worker_id}")
                        await sqlite_persistence.store_memory(memory)
                        results["writes"] += 1
                        
                    elif i % 3 == 1:
                        # Read operation
                        search_results = await sqlite_persistence.retrieve_memories(
                            f"test memory {worker_id}",
                            limit=5,
                            min_similarity=0.0
                        )
                        results["reads"] += 1
                        
                    else:
                        # Update operation (if we have stored memories)
                        if results["writes"] > 0:
                            # Try to update a memory we've written
                            search_results = await sqlite_persistence.search_memories(
                                filters={"metadata": f"mixed_{worker_id}"},
                                limit=1
                            )
                            if search_results:
                                memory_id = search_results[0]["id"]
                                updates = {"importance": 0.95, "metadata": {"updated": True}}
                                success = await sqlite_persistence.update_memory(memory_id, updates)
                                if success:
                                    results["updates"] += 1
                        
                except Exception as e:
                    print(f"Mixed operation error in {worker_id}: {e}")
                    results["errors"] += 1
            
            return results
        
        # Run mixed workers
        start_time = time.perf_counter()
        
        tasks = []
        for worker_id in range(6):  # 6 concurrent mixed workers
            task = asyncio.create_task(mixed_worker(f"worker_{worker_id}", 25))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        mixed_time = time.perf_counter() - start_time
        
        # Aggregate results
        total_reads = 0
        total_writes = 0
        total_updates = 0
        total_errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Mixed worker {i} failed: {result}")
                total_errors += 1
            else:
                total_reads += result["reads"]
                total_writes += result["writes"]
                total_updates += result["updates"]
                total_errors += result["errors"]
                print(f"Worker {i}: R:{result['reads']} W:{result['writes']} U:{result['updates']} E:{result['errors']}")
        
        total_operations = total_reads + total_writes + total_updates
        
        print(f"Mixed operations: {total_operations} total, {total_errors} errors, {mixed_time:.3f}s")
        print(f"Operation rate: {total_operations / mixed_time:.1f} ops/sec")
        print(f"Breakdown: {total_reads} reads, {total_writes} writes, {total_updates} updates")
        
        # Verify data consistency
        final_stats = await sqlite_persistence.get_memory_stats()
        print(f"Final database stats: {final_stats['total_memories']} memories")
        
        # Performance assertions
        assert total_operations >= 120  # Should complete most operations
        assert total_errors < 5  # Should have minimal errors
        assert mixed_time < 20  # Should complete within reasonable time
        assert final_stats["total_memories"] >= 20 + total_writes * 0.8  # Account for writes
    
    def test_thread_safety_validation(self, temp_db_path, mock_embedding_model):
        """Test thread safety using Python threading."""
        print("\nTesting thread safety...")
        
        # Shared results storage
        results = {"stored": [], "errors": [], "lock": threading.Lock()}
        
        def thread_worker(thread_id: int, count: int):
            """Worker function for thread safety test."""
            # Each thread creates its own persistence instance
            persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
            
            for i in range(count):
                try:
                    memory = self.generate_test_memory(i, f"thread_{thread_id}")
                    
                    # Use asyncio in thread
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    memory_id = loop.run_until_complete(persistence.store_memory(memory))
                    
                    with results["lock"]:
                        results["stored"].append(memory_id)
                    
                    loop.close()
                    
                except Exception as e:
                    with results["lock"]:
                        results["errors"].append(f"Thread {thread_id}: {e}")
        
        # Run threads
        start_time = time.perf_counter()
        threads = []
        
        for thread_id in range(4):  # 4 threads
            thread = threading.Thread(target=thread_worker, args=(thread_id, 15))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        thread_time = time.perf_counter() - start_time
        
        # Verify results
        stored_count = len(results["stored"])
        error_count = len(results["errors"])
        
        print(f"Thread safety test: {stored_count} stored, {error_count} errors, {thread_time:.3f}s")
        
        if results["errors"]:
            print("Errors:", results["errors"][:3])  # Show first few errors
        
        # Verify data in database
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        loop = asyncio.new_event_loop()
        stats = loop.run_until_complete(persistence.get_memory_stats())
        loop.close()
        
        print(f"Database contains {stats['total_memories']} memories")
        
        # Assertions
        assert stored_count >= 50  # Should store most memories
        assert error_count < 10  # Should have minimal errors
        assert stats["total_memories"] >= stored_count * 0.9  # Account for potential issues
    
    @pytest.mark.asyncio
    async def test_wal_mode_effectiveness(self, sqlite_persistence):
        """Test that WAL mode enables concurrent access effectively."""
        print("\nTesting WAL mode effectiveness...")
        
        import sqlite3
        
        # Verify WAL mode is enabled
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            print(f"Journal mode: {journal_mode}")
            assert journal_mode.upper() == "WAL"
        
        # Test that concurrent readers don't block each other
        # Pre-populate database
        for i in range(30):
            memory = self.generate_test_memory(i, "wal_test")
            await sqlite_persistence.store_memory(memory)
        
        async def wal_reader(reader_id: str, duration: float) -> int:
            """Continuous reader for WAL test."""
            read_count = 0
            start_time = time.perf_counter()
            
            while time.perf_counter() - start_time < duration:
                try:
                    results = await sqlite_persistence.retrieve_memories(
                        f"wal test memory {reader_id}",
                        limit=5,
                        min_similarity=0.0
                    )
                    read_count += 1
                    
                    # Small delay to allow other operations
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    print(f"WAL reader {reader_id} error: {e}")
                    break
            
            return read_count
        
        async def wal_writer(writer_id: str, duration: float) -> int:
            """Continuous writer for WAL test."""
            write_count = 0
            start_time = time.perf_counter()
            index = 0
            
            while time.perf_counter() - start_time < duration:
                try:
                    memory = self.generate_test_memory(index, f"wal_writer_{writer_id}")
                    await sqlite_persistence.store_memory(memory)
                    write_count += 1
                    index += 1
                    
                    # Small delay
                    await asyncio.sleep(0.05)
                    
                except Exception as e:
                    print(f"WAL writer {writer_id} error: {e}")
                    break
            
            return write_count
        
        # Run concurrent readers and writers for 3 seconds
        test_duration = 3.0
        
        tasks = []
        
        # Start multiple readers
        for reader_id in range(4):
            task = asyncio.create_task(wal_reader(f"reader_{reader_id}", test_duration))
            tasks.append(task)
        
        # Start writers
        for writer_id in range(2):
            task = asyncio.create_task(wal_writer(f"writer_{writer_id}", test_duration))
            tasks.append(task)
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        actual_time = time.perf_counter() - start_time
        
        # Analyze results
        reader_results = results[:4]  # First 4 are readers
        writer_results = results[4:]  # Last 2 are writers
        
        total_reads = sum(r for r in reader_results if isinstance(r, int))
        total_writes = sum(w for w in writer_results if isinstance(w, int))
        
        read_rate = total_reads / actual_time
        write_rate = total_writes / actual_time
        
        print(f"WAL test results ({actual_time:.1f}s):")
        print(f"  Total reads: {total_reads} ({read_rate:.1f} reads/sec)")
        print(f"  Total writes: {total_writes} ({write_rate:.1f} writes/sec)")
        print(f"  Reader results: {reader_results}")
        print(f"  Writer results: {writer_results}")
        
        # Verify no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            print(f"Exceptions: {exceptions}")
        
        # WAL mode should allow good concurrent performance
        assert total_reads > 50  # Should get many reads
        assert total_writes > 10  # Should get some writes
        assert len(exceptions) == 0  # Should not have exceptions
        assert read_rate > 15  # Good read throughput
    
    @pytest.mark.asyncio
    async def test_deadlock_prevention(self, sqlite_persistence):
        """Test deadlock prevention mechanisms."""
        print("\nTesting deadlock prevention...")
        
        # Create scenario that could cause deadlocks without proper handling
        async def complex_transaction_worker(worker_id: str, iterations: int) -> Dict[str, int]:
            """Perform complex operations that could cause deadlocks."""
            results = {"success": 0, "errors": 0}
            
            for i in range(iterations):
                try:
                    # Store a memory
                    memory = self.generate_test_memory(i, f"deadlock_{worker_id}")
                    memory_id = await sqlite_persistence.store_memory(memory)
                    
                    # Immediately search for it
                    search_results = await sqlite_persistence.retrieve_memories(
                        f"deadlock {worker_id} {i}",
                        limit=3,
                        min_similarity=0.0
                    )
                    
                    # Update if found
                    if search_results:
                        found_memory = search_results[0]
                        await sqlite_persistence.update_memory(
                            found_memory["id"],
                            {"importance": 0.9, "metadata": {"processed": True}}
                        )
                    
                    # Get stats (involves aggregation queries)
                    stats = await sqlite_persistence.get_memory_stats()
                    
                    results["success"] += 1
                    
                except Exception as e:
                    print(f"Deadlock test worker {worker_id} error: {e}")
                    results["errors"] += 1
                    
                    # Small delay before retrying to prevent tight error loops
                    await asyncio.sleep(0.01)
            
            return results
        
        # Run workers that perform complex interlocking operations
        start_time = time.perf_counter()
        
        tasks = []
        for worker_id in range(6):  # 6 workers doing complex operations
            task = asyncio.create_task(complex_transaction_worker(f"worker_{worker_id}", 20))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        deadlock_time = time.perf_counter() - start_time
        
        # Analyze results
        total_success = 0
        total_errors = 0
        worker_exceptions = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Worker {i} failed with exception: {result}")
                worker_exceptions += 1
            else:
                total_success += result["success"]
                total_errors += result["errors"]
                print(f"Worker {i}: {result['success']} success, {result['errors']} errors")
        
        success_rate = total_success / (total_success + total_errors) if (total_success + total_errors) > 0 else 0
        
        print(f"Deadlock prevention test: {total_success} success, {total_errors} errors, {worker_exceptions} exceptions")
        print(f"Success rate: {success_rate:.2%}, Total time: {deadlock_time:.3f}s")
        
        # Assertions for deadlock prevention
        assert worker_exceptions == 0  # No workers should fail completely
        assert success_rate > 0.8  # At least 80% operations should succeed
        assert deadlock_time < 25  # Should complete within reasonable time
        assert total_success > 80  # Should complete most operations
    
    @pytest.mark.asyncio
    async def test_high_concurrent_load(self, sqlite_persistence):
        """Test behavior under high concurrent load."""
        print("\nTesting high concurrent load...")
        
        async def high_load_worker(worker_id: str, operations: int) -> Dict[str, Any]:
            """Worker for high load test."""
            results = {
                "stores": 0,
                "retrievals": 0,
                "searches": 0,
                "updates": 0,
                "errors": 0,
                "times": []
            }
            
            for i in range(operations):
                op_start = time.perf_counter()
                
                try:
                    operation = i % 4
                    
                    if operation == 0:
                        # Store operation
                        memory = self.generate_test_memory(i, f"load_{worker_id}")
                        await sqlite_persistence.store_memory(memory)
                        results["stores"] += 1
                        
                    elif operation == 1:
                        # Retrieval operation
                        await sqlite_persistence.retrieve_memories(
                            f"load test {worker_id}",
                            limit=8,
                            min_similarity=0.0
                        )
                        results["retrievals"] += 1
                        
                    elif operation == 2:
                        # Search operation
                        await sqlite_persistence.search_memories(
                            types=["episodic", "structured_thinking"],
                            limit=10
                        )
                        results["searches"] += 1
                        
                    else:
                        # Update operation (try to update recent memory)
                        if results["stores"] > 5:
                            search_results = await sqlite_persistence.search_memories(
                                filters={"metadata": f"load_{worker_id}"},
                                limit=1
                            )
                            if search_results:
                                await sqlite_persistence.update_memory(
                                    search_results[0]["id"],
                                    {"importance": 0.85}
                                )
                                results["updates"] += 1
                    
                    op_time = time.perf_counter() - op_start
                    results["times"].append(op_time)
                    
                except Exception as e:
                    results["errors"] += 1
                    print(f"High load worker {worker_id} error: {e}")
            
            return results
        
        # Run high concurrent load
        start_time = time.perf_counter()
        
        tasks = []
        for worker_id in range(12):  # 12 concurrent workers
            task = asyncio.create_task(high_load_worker(f"worker_{worker_id}", 30))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        load_time = time.perf_counter() - start_time
        
        # Aggregate results
        total_operations = 0
        total_errors = 0
        all_times = []
        operation_counts = {"stores": 0, "retrievals": 0, "searches": 0, "updates": 0}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"High load worker {i} failed: {result}")
                total_errors += 100  # Penalize complete failures
            else:
                worker_ops = result["stores"] + result["retrievals"] + result["searches"] + result["updates"]
                total_operations += worker_ops
                total_errors += result["errors"]
                all_times.extend(result["times"])
                
                for op_type in operation_counts:
                    operation_counts[op_type] += result[op_type]
                
                print(f"Worker {i}: {worker_ops} ops, {result['errors']} errors")
        
        # Calculate performance metrics
        avg_op_time = sum(all_times) / len(all_times) if all_times else 0
        max_op_time = max(all_times) if all_times else 0
        operations_per_second = total_operations / load_time
        error_rate = total_errors / (total_operations + total_errors) if total_operations + total_errors > 0 else 1.0
        
        print(f"\nHigh Load Test Results:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total errors: {total_errors}")
        print(f"  Error rate: {error_rate:.2%}")
        print(f"  Total time: {load_time:.2f}s")
        print(f"  Operations/sec: {operations_per_second:.1f}")
        print(f"  Average op time: {avg_op_time * 1000:.2f}ms")
        print(f"  Max op time: {max_op_time * 1000:.2f}ms")
        print(f"  Operation breakdown: {operation_counts}")
        
        # Final database state
        final_stats = await sqlite_persistence.get_memory_stats()
        print(f"  Final database: {final_stats['total_memories']} memories")
        
        # Performance assertions for high load
        assert total_operations > 250  # Should complete most operations
        assert error_rate < 0.1  # Less than 10% error rate
        assert operations_per_second > 15  # Reasonable throughput under load
        assert avg_op_time < 0.2  # Average operation under 200ms
        assert max_op_time < 2.0  # No operation should take more than 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements