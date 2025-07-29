#!/usr/bin/env python3
"""
SQLite Fallback Mechanisms Tests

Integration tests for error handling and fallback mechanisms:
- sqlite-vec extension failure handling
- Database corruption recovery
- Embedding model failure fallbacks
- Network/disk failure scenarios
- Graceful degradation testing
- Recovery and restoration procedures
"""

import asyncio
import os
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence
from clarity.shared.exceptions.base import MemoryOperationError


class TestSQLiteFallbackMechanisms:
    """Test suite for SQLite fallback mechanisms and error handling."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="fallback_test_")
        db_path = os.path.join(temp_dir, "fallback_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1] * 384
        return mock_model
    
    @pytest.fixture
    def sqlite_persistence(self, temp_db_path, mock_embedding_model):
        """Create SQLiteMemoryPersistence instance for testing."""
        return SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
    
    @pytest.mark.asyncio
    async def test_sqlite_vec_extension_fallback(self, temp_db_path, mock_embedding_model):
        """Test fallback when sqlite-vec extension is not available."""
        print("\nTesting sqlite-vec extension fallback...")
        
        # Create persistence instance that will fail to load sqlite-vec
        with patch('sqlite3.Connection.load_extension') as mock_load_ext:
            mock_load_ext.side_effect = sqlite3.Error("Extension not found")
            
            # This should still work, just log a warning
            persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
            
            # Verify database was still created
            assert os.path.exists(temp_db_path)
            
            # Test that basic operations still work without the extension
            test_memory = {
                "id": "fallback-001",
                "type": "structured_thinking",
                "content": "Testing fallback without sqlite-vec extension",
                "importance": 0.8,
                "tier": "short_term",
                "metadata": {"test": "sqlite_vec_fallback"}
            }
            
            # Store memory should work
            memory_id = await persistence.store_memory(test_memory)
            assert memory_id is not None
            
            # Retrieval should work (using fallback similarity calculation)
            results = await persistence.retrieve_memories(
                "fallback without sqlite-vec",
                limit=5,
                min_similarity=0.0
            )
            
            assert len(results) > 0
            assert results[0]["id"] == memory_id
            assert "similarity_score" in results[0]
            
            print(f"  Successfully stored and retrieved memory without sqlite-vec")
            print(f"  Similarity score: {results[0]['similarity_score']:.3f}")
    
    @pytest.mark.asyncio
    async def test_embedding_model_failure_fallback(self, temp_db_path):
        """Test fallback when embedding model fails."""
        print("\nTesting embedding model failure fallback...")
        
        # Create failing embedding model
        failing_model = Mock()
        failing_model.encode.side_effect = Exception("Model loading failed")
        
        # Test that persistence handles embedding failure gracefully
        with pytest.raises(MemoryOperationError):
            persistence = SQLiteMemoryPersistence(temp_db_path, failing_model)
            
            test_memory = {
                "id": "embedding-fail-001",
                "type": "episodic",
                "content": "Testing embedding model failure",
                "importance": 0.7,
                "tier": "short_term"
            }
            
            # This should fail with a clear error message
            await persistence.store_memory(test_memory)
    
    @pytest.mark.asyncio
    async def test_embedding_model_lazy_loading_fallback(self, temp_db_path):
        """Test fallback embedding models during lazy loading."""
        print("\nTesting embedding model lazy loading fallback...")
        
        # Create persistence without embedding model (lazy loading)
        persistence = SQLiteMemoryPersistence(temp_db_path, None)
        
        # Mock the lazy loading to test fallback models
        with patch('clarity.shared.lazy_imports.ml_deps.SentenceTransformer') as mock_transformer:
            # First model fails
            def side_effect(model_name):
                if model_name == "paraphrase-MiniLM-L3-v2":
                    raise Exception("First model failed")
                elif model_name == "all-MiniLM-L6-v2":
                    # Second model succeeds
                    mock_model = Mock()
                    mock_model.encode.return_value = [0.2] * 384
                    return mock_model
                else:
                    raise Exception("Model not found")
            
            mock_transformer.side_effect = side_effect
            
            # Store memory should trigger lazy loading and use fallback model
            test_memory = {
                "id": "lazy-fallback-001",
                "type": "semantic",
                "content": "Testing lazy loading fallback model",
                "importance": 0.6,
                "tier": "archival"
            }
            
            memory_id = await persistence.store_memory(test_memory)
            assert memory_id is not None
            
            # Verify that the fallback model was used
            assert persistence.embedding_model is not None
            print(f"  Successfully used fallback embedding model")
    
    @pytest.mark.asyncio
    async def test_database_corruption_handling(self, temp_db_path, mock_embedding_model):
        """Test handling of database corruption scenarios."""
        print("\nTesting database corruption handling...")
        
        # Create normal persistence and add some data
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        # Store some test data
        for i in range(5):
            memory = {
                "id": f"corruption-test-{i}",
                "type": "episodic",
                "content": f"Test memory {i} before corruption",
                "importance": 0.5,
                "tier": "short_term"
            }
            await persistence.store_memory(memory)
        
        # Verify data exists
        stats = await persistence.get_memory_stats()
        assert stats["total_memories"] >= 5
        
        # Simulate database corruption by writing invalid data
        with open(temp_db_path, 'wb') as f:
            f.write(b"CORRUPTED DATABASE CONTENT")
        
        # Create new persistence instance with corrupted database
        # This should handle the corruption gracefully
        with patch('os.makedirs'), patch('shutil.copy2'):  # Prevent actual file operations
            corrupted_persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
            
            # The database should be recreated
            assert os.path.exists(temp_db_path)
            
            # Should be able to store new data
            recovery_memory = {
                "id": "recovery-001",
                "type": "structured_thinking",
                "content": "Memory stored after corruption recovery",
                "importance": 0.8,
                "tier": "short_term"
            }
            
            memory_id = await corrupted_persistence.store_memory(recovery_memory)
            assert memory_id is not None
            
            print(f"  Successfully recovered from database corruption")
    
    @pytest.mark.asyncio
    async def test_disk_space_handling(self, temp_db_path, mock_embedding_model):
        """Test handling of disk space limitations."""
        print("\nTesting disk space handling...")
        
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        # Store some initial data
        initial_memory = {
            "id": "disk-space-001",
            "type": "procedural",
            "content": "Initial memory before disk space test",
            "importance": 0.7,
            "tier": "long_term"
        }
        
        memory_id = await persistence.store_memory(initial_memory)
        assert memory_id is not None
        
        # Mock disk space error during database operations
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            
            # First call succeeds (connection), second fails (disk full)
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execute.side_effect = [
                mock_cursor,  # First query succeeds
                sqlite3.Error("database or disk is full"),  # Second fails
            ]
            
            # Attempt to store memory with disk full error
            disk_full_memory = {
                "id": "disk-full-001",
                "type": "episodic",
                "content": "Memory during disk full scenario",
                "importance": 0.6,
                "tier": "short_term"
            }
            
            # Should raise MemoryOperationError with clear message
            with pytest.raises(MemoryOperationError) as exc_info:
                await persistence.store_memory(disk_full_memory)
            
            assert "disk" in str(exc_info.value).lower() or "full" in str(exc_info.value).lower()
            print(f"  Properly handled disk full error: {exc_info.value}")
    
    @pytest.mark.asyncio
    async def test_concurrent_access_failure_recovery(self, temp_db_path, mock_embedding_model):
        """Test recovery from concurrent access failures."""
        print("\nTesting concurrent access failure recovery...")
        
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        # Store initial data
        for i in range(10):
            memory = {
                "id": f"concurrent-recovery-{i}",
                "type": "semantic",
                "content": f"Memory {i} for concurrent recovery test",
                "importance": 0.5,
                "tier": "short_term"
            }
            await persistence.store_memory(memory)
        
        # Simulate database lock errors during concurrent operations
        original_execute = None
        call_count = 0
        
        def mock_execute_with_failures(*args, **kwargs):
            nonlocal call_count, original_execute
            call_count += 1
            
            # Fail on some calls to simulate lock contention
            if call_count % 3 == 0:  # Every 3rd call fails
                raise sqlite3.OperationalError("database is locked")
            
            # Call original method for other calls
            return original_execute(*args, **kwargs)
        
        # Apply mock with some failures
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Some operations succeed, some fail
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 4 == 0:  # Every 4th operation fails
                    raise sqlite3.OperationalError("database is locked")
                else:
                    # Return a mock cursor
                    mock_cursor = Mock()
                    mock_cursor.fetchall.return_value = []
                    mock_cursor.fetchone.return_value = None
                    return mock_cursor
            
            mock_conn.execute.side_effect = side_effect
            mock_conn.executescript.side_effect = side_effect
            
            # Attempt multiple operations - some should fail, some succeed
            success_count = 0
            error_count = 0
            
            for i in range(10):
                try:
                    test_memory = {
                        "id": f"lock-test-{i}",
                        "type": "episodic",
                        "content": f"Lock test memory {i}",
                        "importance": 0.6,
                        "tier": "short_term"
                    }
                    
                    await persistence.store_memory(test_memory)
                    success_count += 1
                    
                except MemoryOperationError as e:
                    if "lock" in str(e).lower():
                        error_count += 1
                        print(f"    Expected lock error: {e}")
                    else:
                        raise  # Unexpected error
            
            print(f"  Lock contention test: {success_count} success, {error_count} expected failures")
            
            # Should have some successes and some expected failures
            assert error_count > 0  # Should have detected some lock failures
            assert success_count >= 0  # Some operations might succeed
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, sqlite_persistence):
        """Test handling of invalid or malformed data."""
        print("\nTesting invalid data handling...")
        
        # Test various invalid data scenarios
        invalid_data_cases = [
            {
                "name": "None content",
                "memory": {
                    "id": "invalid-001",
                    "type": "structured_thinking",
                    "content": None,
                    "importance": 0.8,
                    "tier": "short_term"
                }
            },
            {
                "name": "Invalid importance range",
                "memory": {
                    "id": "invalid-002",
                    "type": "episodic",
                    "content": "Valid content",
                    "importance": 2.5,  # > 1.0
                    "tier": "short_term"
                }
            },
            {
                "name": "Invalid tier",
                "memory": {
                    "id": "invalid-003",
                    "type": "semantic",
                    "content": "Valid content",
                    "importance": 0.7,
                    "tier": "invalid_tier"
                }
            },
            {
                "name": "Circular reference in content",
                "memory": {
                    "id": "invalid-004",
                    "type": "procedural",
                    "content": {"self_ref": None},  # Will be set to circular reference
                    "importance": 0.6,
                    "tier": "system"
                }
            }
        ]
        
        # Create circular reference
        invalid_data_cases[3]["memory"]["content"]["self_ref"] = invalid_data_cases[3]["memory"]["content"]
        
        success_count = 0
        handled_errors = 0
        
        for case in invalid_data_cases:
            try:
                print(f"  Testing {case['name']}...")
                memory_id = await sqlite_persistence.store_memory(case["memory"])
                
                if memory_id is not None:
                    success_count += 1
                    print(f"    Handled gracefully, stored as {memory_id}")
                    
                    # Verify the stored memory
                    stored = await sqlite_persistence.get_memory(memory_id)
                    if stored:
                        print(f"    Retrieved successfully with corrected data")
                else:
                    print(f"    Returned None (handled gracefully)")
                    handled_errors += 1
                    
            except Exception as e:
                # Should be specific exception types, not generic exceptions
                if isinstance(e, (MemoryOperationError, ValueError, TypeError)):
                    handled_errors += 1
                    print(f"    Properly rejected: {type(e).__name__}: {e}")
                else:
                    print(f"    Unexpected error: {type(e).__name__}: {e}")
                    raise  # Re-raise unexpected errors
        
        print(f"  Invalid data handling: {success_count} corrected, {handled_errors} rejected")
        
        # Should handle all cases either by correction or rejection
        assert (success_count + handled_errors) == len(invalid_data_cases)
    
    @pytest.mark.asyncio
    async def test_large_data_handling(self, sqlite_persistence):
        """Test handling of extremely large data inputs."""
        print("\nTesting large data handling...")
        
        # Test with very large content
        large_content = "Large content test. " * 10000  # ~200KB content
        
        large_memory = {
            "id": "large-001",
            "type": "structured_thinking",
            "content": large_content,
            "importance": 0.7,
            "tier": "archival",
            "metadata": {"size": "large", "length": len(large_content)}
        }
        
        try:
            memory_id = await sqlite_persistence.store_memory(large_memory)
            assert memory_id is not None
            
            # Verify retrieval
            stored = await sqlite_persistence.get_memory(memory_id)
            assert stored is not None
            assert len(str(stored["content"])) > 100000  # Should preserve large content
            
            print(f"  Successfully handled large content ({len(large_content)} chars)")
            
        except MemoryOperationError as e:
            # Acceptable if system has limits
            print(f"  Large content rejected (acceptable): {e}")
        
        # Test with very large metadata
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}  # Large metadata dict
        
        large_meta_memory = {
            "id": "large-meta-001",
            "type": "episodic",
            "content": "Memory with large metadata",
            "importance": 0.6,
            "tier": "short_term",
            "metadata": large_metadata
        }
        
        try:
            memory_id = await sqlite_persistence.store_memory(large_meta_memory)
            assert memory_id is not None
            
            stored = await sqlite_persistence.get_memory(memory_id)
            assert stored is not None
            assert len(stored["metadata"]) >= 100  # Should preserve metadata
            
            print(f"  Successfully handled large metadata ({len(large_metadata)} keys)")
            
        except MemoryOperationError as e:
            # Acceptable if system has limits
            print(f"  Large metadata rejected (acceptable): {e}")
    
    @pytest.mark.asyncio
    async def test_network_timeout_simulation(self, sqlite_persistence):
        """Test handling of simulated network-like timeout scenarios."""
        print("\nTesting timeout scenario handling...")
        
        # Store some initial data
        initial_memory = {
            "id": "timeout-001",
            "type": "procedural",
            "content": "Memory before timeout simulation",
            "importance": 0.8,
            "tier": "system"
        }
        
        memory_id = await sqlite_persistence.store_memory(initial_memory)
        assert memory_id is not None
        
        # Simulate timeout by mocking slow database operations
        with patch('sqlite3.Connection.execute') as mock_execute:
            def slow_execute(*args, **kwargs):
                import time
                time.sleep(2)  # Simulate slow operation
                raise sqlite3.OperationalError("operation timeout")
            
            mock_execute.side_effect = slow_execute
            
            # Attempt operation that should timeout
            timeout_memory = {
                "id": "timeout-002",
                "type": "episodic",
                "content": "Memory during timeout scenario",
                "importance": 0.7,
                "tier": "short_term"
            }
            
            start_time = time.time()
            
            with pytest.raises(MemoryOperationError) as exc_info:
                await persistence.store_memory(timeout_memory)
            
            elapsed_time = time.time() - start_time
            
            # Should fail reasonably quickly and with clear error
            assert elapsed_time < 5  # Should not hang indefinitely
            assert "timeout" in str(exc_info.value).lower() or "operation" in str(exc_info.value).lower()
            
            print(f"  Timeout handled in {elapsed_time:.2f}s: {exc_info.value}")
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, temp_db_path, mock_embedding_model):
        """Test graceful degradation under various failure conditions."""
        print("\nTesting graceful degradation...")
        
        # Create persistence instance
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        # Store some initial data successfully
        for i in range(5):
            memory = {
                "id": f"degradation-{i}",
                "type": "semantic",
                "content": f"Memory {i} for degradation test",
                "importance": 0.6,
                "tier": "short_term"
            }
            await persistence.store_memory(memory)
        
        # Test degraded performance with embedding failures
        original_encode = mock_embedding_model.encode
        
        def intermittent_encode(text):
            import random
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("Intermittent embedding failure")
            return original_encode(text)
        
        mock_embedding_model.encode.side_effect = intermittent_encode
        
        # Attempt multiple operations under degraded conditions
        success_count = 0
        failure_count = 0
        
        for i in range(20):
            try:
                degraded_memory = {
                    "id": f"degraded-{i}",
                    "type": "episodic",
                    "content": f"Memory {i} under degraded conditions",
                    "importance": 0.5,
                    "tier": "short_term"
                }
                
                memory_id = await persistence.store_memory(degraded_memory)
                if memory_id:
                    success_count += 1
                    
            except MemoryOperationError:
                failure_count += 1
        
        success_rate = success_count / (success_count + failure_count) if (success_count + failure_count) > 0 else 0
        
        print(f"  Degraded operations: {success_count} success, {failure_count} failures")
        print(f"  Success rate under degradation: {success_rate:.2%}")
        
        # Should still maintain some functionality
        assert success_rate >= 0.5  # At least 50% should work
        
        # Verify that successful operations are still accessible
        if success_count > 0:
            results = await persistence.retrieve_memories(
                "degraded conditions",
                limit=10,
                min_similarity=0.0
            )
            assert len(results) > 0
            print(f"  Successfully retrieved {len(results)} memories under degradation")
    
    @pytest.mark.asyncio
    async def test_recovery_after_failure(self, temp_db_path, mock_embedding_model):
        """Test system recovery after various failure scenarios."""
        print("\nTesting recovery after failure...")
        
        # Initial setup
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        # Store initial data
        initial_memories = []
        for i in range(3):
            memory = {
                "id": f"recovery-initial-{i}",
                "type": "structured_thinking",
                "content": f"Initial memory {i} before failure",
                "importance": 0.8,
                "tier": "long_term"
            }
            memory_id = await persistence.store_memory(memory)
            initial_memories.append(memory_id)
        
        # Simulate system failure and recovery
        print("  Simulating system failure...")
        
        # Create new persistence instance (simulating restart after failure)
        recovered_persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        # Verify that data persisted through "failure"
        for memory_id in initial_memories:
            recovered_memory = await recovered_persistence.get_memory(memory_id)
            assert recovered_memory is not None
            assert "before failure" in str(recovered_memory["content"])
        
        print("  Data successfully recovered after restart")
        
        # Test that new operations work after recovery
        post_recovery_memory = {
            "id": "recovery-post-001",
            "type": "episodic",
            "content": "Memory stored after system recovery",
            "importance": 0.7,
            "tier": "short_term"
        }
        
        memory_id = await recovered_persistence.store_memory(post_recovery_memory)
        assert memory_id is not None
        
        # Verify search functionality works after recovery
        results = await recovered_persistence.retrieve_memories(
            "memory after recovery",
            limit=5,
            min_similarity=0.0
        )
        
        assert len(results) > 0
        found_post_recovery = any("after system recovery" in str(r.get("content", "")) for r in results)
        assert found_post_recovery
        
        print("  Full functionality restored after recovery")
        
        # Verify database statistics are correct after recovery
        stats = await recovered_persistence.get_memory_stats()
        assert stats["total_memories"] >= 4  # Initial + post-recovery
        
        print(f"  Database stats after recovery: {stats['total_memories']} memories")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements