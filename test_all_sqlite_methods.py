#!/usr/bin/env python3
"""
Comprehensive test suite for all SQLiteMemoryPersistence methods.

This test validates all functionality after the SQL injection fix to ensure
no regressions were introduced and all methods work correctly.
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

# Test the SQLite persistence directly
import sys
sys.path.insert(0, '/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity')

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence


class ComprehensiveMethodTester:
    """Test all SQLiteMemoryPersistence methods comprehensively."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        self.persistence = SQLiteMemoryPersistence(self.db_path)
        self.test_results = {}
        self.stored_memory_ids = []
    
    async def run_all_tests(self):
        """Run comprehensive tests for all methods."""
        print("🧪 Starting comprehensive SQLite method testing...")
        print(f"📁 Test database: {self.db_path}")
        print()
        
        # Test each method systematically
        test_methods = [
            ("Database Initialization", self.test_database_init),
            ("Memory Storage", self.test_store_memory),
            ("Memory Retrieval by ID", self.test_get_memory),
            ("Memory Search", self.test_search_memories),
            ("Memory Retrieval with Query", self.test_retrieve_memories),
            ("Memory Updates", self.test_update_memory),
            ("Memory Statistics", self.test_memory_stats),
            ("Memory Deletion", self.test_delete_memories),
            ("Edge Cases", self.test_edge_cases),
            ("Performance Validation", self.test_performance),
        ]
        
        for test_name, test_method in test_methods:
            print(f"🔍 Testing: {test_name}")
            try:
                await test_method()
                self.test_results[test_name] = "✅ PASS"
                print(f"   ✅ {test_name} - PASSED")
            except Exception as e:
                self.test_results[test_name] = f"❌ FAIL: {e}"
                print(f"   ❌ {test_name} - FAILED: {e}")
            print()
        
        # Final summary
        await self.print_test_summary()
    
    async def test_database_init(self):
        """Test database initialization and configuration."""
        # Check if database file was created
        assert os.path.exists(self.db_path), "Database file not created"
        
        # Check if tables exist
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "memories" in tables, "memories table not created"
            
            # Check table schema
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            expected_columns = [
                'memory_id', 'memory_type', 'content', 'text_content', 
                'importance', 'tier', 'created_at', 'updated_at', 
                'metadata', 'context', 'access_count', 'last_accessed', 'embedding'
            ]
            for col in expected_columns:
                assert col in columns, f"Column {col} missing from schema"
    
    async def test_store_memory(self):
        """Test memory storage functionality."""
        # Test basic memory storage
        memory_data = {
            "content": "This is a test memory for validation",
            "type": "test_memory",
            "importance": 0.8,
            "metadata": {"test": True, "category": "validation"},
            "context": {"session": "test_session"}
        }
        
        memory_id = await self.persistence.store_memory(memory_data, tier="short_term")
        assert memory_id, "Memory ID not returned"
        self.stored_memory_ids.append(memory_id)
        
        # Test with complex content
        complex_memory = {
            "content": {
                "title": "Complex Test Memory",
                "description": "Testing complex content storage",
                "data": [1, 2, 3, {"nested": "value"}]
            },
            "type": "complex_memory",
            "importance": 0.9,
            "tier": "long_term"
        }
        
        complex_id = await self.persistence.store_memory(complex_memory)
        assert complex_id, "Complex memory ID not returned"
        self.stored_memory_ids.append(complex_id)
        
        # Test with minimal data
        minimal_memory = {"content": "Minimal test"}
        minimal_id = await self.persistence.store_memory(minimal_memory)
        assert minimal_id, "Minimal memory ID not returned"
        self.stored_memory_ids.append(minimal_id)
    
    async def test_get_memory(self):
        """Test memory retrieval by ID."""
        if not self.stored_memory_ids:
            await self.test_store_memory()
        
        # Test retrieving first stored memory
        memory = await self.persistence.get_memory(self.stored_memory_ids[0])
        assert memory is not None, "Memory not found by ID"
        assert memory["id"] == self.stored_memory_ids[0], "Memory ID mismatch"
        assert memory["type"] == "test_memory", "Memory type mismatch"
        assert "content" in memory, "Content missing from retrieved memory"
        
        # Test non-existent memory
        fake_id = str(uuid.uuid4())
        missing_memory = await self.persistence.get_memory(fake_id)
        assert missing_memory is None, "Non-existent memory should return None"
    
    async def test_search_memories(self):
        """Test the search_memories method with various filters."""
        if not self.stored_memory_ids:
            await self.test_store_memory()
        
        # Test basic search without filters
        results = await self.persistence.search_memories()
        assert len(results) > 0, "Search should return stored memories"
        
        # Test with type filter
        results = await self.persistence.search_memories(types=["test_memory"])
        assert len(results) > 0, "Should find test_memory type"
        assert all(r["type"] == "test_memory" for r in results), "All results should be test_memory type"
        
        # Test with valid filters (using VALID_FILTER_COLUMNS)
        valid_filters = {
            "tier": "short_term",
            "memory_type": "test_memory",
            "importance": 0.8
        }
        results = await self.persistence.search_memories(filters=valid_filters)
        # Should not raise an error and should return results
        
        # Test with invalid filter (should be ignored, not fail)
        invalid_filters = {
            "metadata_key": "some_value",  # This was causing the original error
            "invalid_column": "test",
            "tier": "short_term"  # This is valid
        }
        results = await self.persistence.search_memories(filters=invalid_filters)
        # Should not raise an error, invalid filters should be ignored
        
        # Test with embedding search
        test_embedding = [0.1] * 384  # 384-dimensional vector
        results = await self.persistence.search_memories(
            embedding=test_embedding, 
            min_similarity=0.0
        )
        assert isinstance(results, list), "Should return list even with low similarity"
    
    async def test_retrieve_memories(self):
        """Test semantic memory retrieval with query."""
        if not self.stored_memory_ids:
            await self.test_store_memory()
        
        # Test basic query retrieval
        results = await self.persistence.retrieve_memories(
            query="test memory validation",
            limit=5
        )
        assert isinstance(results, list), "Should return list of results"
        
        # Test with filters
        results = await self.persistence.retrieve_memories(
            query="test",
            memory_types=["test_memory"],
            min_similarity=0.1,
            include_metadata=True
        )
        assert isinstance(results, list), "Should return list with metadata"
        
        # Test with additional filters
        filters = {"tier": "short_term"}
        results = await self.persistence.retrieve_memories(
            query="test",
            filters=filters
        )
        # Should not raise an error
    
    async def test_update_memory(self):
        """Test memory update functionality."""
        if not self.stored_memory_ids:
            await self.test_store_memory()
        
        memory_id = self.stored_memory_ids[0]
        
        # Test content update
        updates = {
            "content": "Updated test memory content",
            "importance": 0.95,
            "metadata": {"updated": True, "test": False}
        }
        
        success = await self.persistence.update_memory(memory_id, updates)
        assert success, "Memory update should succeed"
        
        # Verify update
        updated_memory = await self.persistence.get_memory(memory_id)
        assert updated_memory["content"] == "Updated test memory content", "Content not updated"
        assert updated_memory["importance"] == 0.95, "Importance not updated"
        
        # Test updating non-existent memory
        fake_id = str(uuid.uuid4())
        success = await self.persistence.update_memory(fake_id, {"content": "test"})
        assert not success, "Update of non-existent memory should fail"
    
    async def test_memory_stats(self):
        """Test memory statistics functionality."""
        stats = await self.persistence.get_memory_stats()
        
        assert "total_memories" in stats, "Stats missing total_memories"
        assert "memory_types" in stats, "Stats missing memory_types"
        assert "memory_tiers" in stats, "Stats missing memory_tiers"
        assert "database_size_bytes" in stats, "Stats missing database_size_bytes"
        assert "database_path" in stats, "Stats missing database_path"
        
        assert isinstance(stats["total_memories"], int), "total_memories should be int"
        assert stats["total_memories"] > 0, "Should have stored memories"
        assert isinstance(stats["memory_types"], dict), "memory_types should be dict"
        assert isinstance(stats["memory_tiers"], dict), "memory_tiers should be dict"
    
    async def test_delete_memories(self):
        """Test memory deletion functionality."""
        # Store a memory specifically for deletion testing
        delete_memory = {
            "content": "Memory to be deleted",
            "type": "delete_test"
        }
        delete_id = await self.persistence.store_memory(delete_memory)
        
        # Verify it exists
        memory = await self.persistence.get_memory(delete_id)
        assert memory is not None, "Memory should exist before deletion"
        
        # Delete it
        deleted_ids = await self.persistence.delete_memories([delete_id])
        assert delete_id in deleted_ids, "Memory should be in deleted list"
        
        # Verify it's gone
        memory = await self.persistence.get_memory(delete_id)
        assert memory is None, "Memory should not exist after deletion"
        
        # Test deleting non-existent memories
        fake_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        deleted_ids = await self.persistence.delete_memories(fake_ids)
        # Should not raise an error, even if memories don't exist
    
    async def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with empty content
        empty_memory = {"content": ""}
        memory_id = await self.persistence.store_memory(empty_memory)
        assert memory_id, "Should handle empty content"
        
        # Test with very long content
        long_content = "x" * 10000
        long_memory = {"content": long_content}
        memory_id = await self.persistence.store_memory(long_memory)
        assert memory_id, "Should handle long content"
        
        # Test with special characters
        special_memory = {
            "content": "Content with 特殊字符 and émojis 🎉",
            "metadata": {"unicode": "test 🌟"}
        }
        memory_id = await self.persistence.store_memory(special_memory)
        assert memory_id, "Should handle unicode characters"
        
        # Test invalid memory ID formats
        invalid_ids = ["", "not-a-uuid", "123", None]
        for invalid_id in invalid_ids:
            try:
                memory = await self.persistence.get_memory(invalid_id) if invalid_id else None
                # Should either return None or handle gracefully
            except Exception as e:
                # Should not crash with unhandled exceptions
                assert "UUID" in str(e) or "invalid" in str(e).lower(), f"Unexpected error: {e}"
    
    async def test_performance(self):
        """Test basic performance characteristics."""
        import time
        
        # Test bulk storage performance
        start_time = time.perf_counter()
        bulk_ids = []
        
        for i in range(50):  # Store 50 memories
            memory = {
                "content": f"Performance test memory {i}",
                "type": "performance_test",
                "importance": 0.5 + (i % 10) * 0.05
            }
            memory_id = await self.persistence.store_memory(memory)
            bulk_ids.append(memory_id)
        
        storage_time = time.perf_counter() - start_time
        storage_rate = 50 / storage_time
        
        print(f"   📊 Storage performance: {storage_rate:.1f} memories/sec")
        assert storage_rate > 10, f"Storage too slow: {storage_rate:.1f} memories/sec"
        
        # Test bulk retrieval performance
        start_time = time.perf_counter()
        
        for memory_id in bulk_ids[:10]:  # Retrieve 10 memories
            await self.persistence.get_memory(memory_id)
        
        retrieval_time = time.perf_counter() - start_time
        retrieval_rate = 10 / retrieval_time
        
        print(f"   📊 Retrieval performance: {retrieval_rate:.1f} memories/sec")
        assert retrieval_rate > 50, f"Retrieval too slow: {retrieval_rate:.1f} memories/sec"
        
        # Test search performance
        start_time = time.perf_counter()
        
        results = await self.persistence.retrieve_memories(
            query="performance test",
            limit=10
        )
        
        search_time = time.perf_counter() - start_time
        
        print(f"   📊 Search time: {search_time*1000:.1f}ms")
        assert search_time < 1.0, f"Search too slow: {search_time*1000:.1f}ms"
        
        # Clean up bulk test data
        await self.persistence.delete_memories(bulk_ids)
    
    async def print_test_summary(self):
        """Print comprehensive test results summary."""
        print("=" * 60)
        print("🧪 COMPREHENSIVE SQLITE METHOD TEST RESULTS")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result.startswith("✅"))
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            print(f"{result:<50} {test_name}")
        
        print()
        print(f"📊 Overall Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! SQLite persistence is working correctly.")
        else:
            print("⚠️  Some tests failed. Review the failures above.")
        
        # Memory statistics
        stats = await self.persistence.get_memory_stats()
        print()
        print("📈 Final Database Statistics:")
        print(f"   • Total memories: {stats.get('total_memories', 0)}")
        print(f"   • Memory types: {len(stats.get('memory_types', {}))}")
        print(f"   • Memory tiers: {len(stats.get('memory_tiers', {}))}")
        print(f"   • Database size: {stats.get('database_size_bytes', 0)} bytes")
        
        print()
        print(f"🗄️  Test database: {self.db_path}")
        print("   (Will be cleaned up automatically)")


async def main():
    """Run comprehensive method testing."""
    tester = ComprehensiveMethodTester()
    
    try:
        await tester.run_all_tests()
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(tester.temp_dir)
            print(f"🧹 Cleaned up test directory: {tester.temp_dir}")
        except Exception as e:
            print(f"⚠️  Could not clean up test directory: {e}")


if __name__ == "__main__":
    asyncio.run(main())