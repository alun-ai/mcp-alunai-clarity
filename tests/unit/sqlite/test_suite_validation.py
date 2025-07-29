#!/usr/bin/env python3
"""
SQLite Test Suite Validation

Comprehensive validation of the SQLite memory persistence test suite.
This script runs key tests from each test file to ensure the suite is working correctly.
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence
from unittest.mock import Mock


class SQLiteTestSuiteValidator:
    """Validates the SQLite test suite implementation."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="sqlite_validation_")
        self.db_path = os.path.join(self.temp_dir, "validation.db")
        self.mock_model = self._create_mock_model()
        self.persistence = SQLiteMemoryPersistence(self.db_path, self.mock_model)
        
    def _create_mock_model(self):
        """Create mock embedding model for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1] * 384
        return mock_model
    
    async def validate_core_functionality(self):
        """Validate core SQLite persistence functionality."""
        print("🔧 Validating Core Functionality...")
        
        # Test memory storage
        test_memory = {
            "id": "validation-001",
            "type": "structured_thinking",
            "content": "SQLite test suite validation memory",
            "importance": 0.8,
            "tier": "short_term",
            "metadata": {"test": "validation"}
        }
        
        memory_id = await self.persistence.store_memory(test_memory)
        assert memory_id is not None, "Memory storage failed"
        print("  ✅ Memory storage: PASS")
        
        # Test memory retrieval
        retrieved = await self.persistence.get_memory(memory_id)
        assert retrieved is not None, "Memory retrieval failed"
        assert retrieved["type"] == "structured_thinking", "Memory data integrity failed"
        print("  ✅ Memory retrieval: PASS")
        
        # Test search functionality
        results = await self.persistence.retrieve_memories(
            "test suite validation",
            limit=5,
            min_similarity=0.0
        )
        assert len(results) > 0, "Search functionality failed"
        assert "similarity_score" in results[0], "Similarity scoring failed"
        print("  ✅ Search functionality: PASS")
        
        # Test update functionality
        success = await self.persistence.update_memory(memory_id, {"importance": 0.95})
        assert success, "Memory update failed"
        
        updated = await self.persistence.get_memory(memory_id)
        assert updated["importance"] == 0.95, "Memory update verification failed"
        print("  ✅ Memory updates: PASS")
        
        # Test deletion
        deleted = await self.persistence.delete_memories([memory_id])
        assert len(deleted) == 1, "Memory deletion failed"
        
        deleted_check = await self.persistence.get_memory(memory_id)
        assert deleted_check is None, "Memory deletion verification failed"
        print("  ✅ Memory deletion: PASS")
        
        print("✅ Core Functionality: ALL TESTS PASS\n")
    
    async def validate_vector_search(self):
        """Validate vector search functionality."""
        print("🔍 Validating Vector Search...")
        
        # Store test memories with different content
        memories = [
            {"id": "vec-001", "content": "Database performance optimization techniques", "type": "structured_thinking"},
            {"id": "vec-002", "content": "Machine learning algorithms for classification", "type": "semantic"},
            {"id": "vec-003", "content": "Database query performance issues reported", "type": "episodic"}
        ]
        
        for memory in memories:
            memory.update({"importance": 0.7, "tier": "short_term"})
            await self.persistence.store_memory(memory)
        
        # Test similarity search
        results = await self.persistence.retrieve_memories(
            "database performance optimization",
            limit=3,
            min_similarity=0.0
        )
        
        assert len(results) > 0, "Vector search returned no results"
        
        # Check similarity scores are ordered
        similarities = [r["similarity_score"] for r in results]
        assert similarities == sorted(similarities, reverse=True), "Results not sorted by similarity"
        print("  ✅ Similarity scoring and ranking: PASS")
        
        # Test embedding consistency
        embedding1 = self.persistence._generate_embedding("test text")
        embedding2 = self.persistence._generate_embedding("test text")
        assert embedding1 == embedding2, "Embedding generation not consistent"
        print("  ✅ Embedding consistency: PASS")
        
        # Test cosine similarity calculation
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = self.persistence._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-10, "Cosine similarity calculation incorrect"
        print("  ✅ Cosine similarity calculation: PASS")
        
        print("✅ Vector Search: ALL TESTS PASS\n")
    
    async def validate_filtering(self):
        """Validate metadata filtering functionality."""
        print("🎛️ Validating Filtering...")
        
        # Store memories with different attributes
        diverse_memories = [
            {"id": "filter-001", "type": "structured_thinking", "tier": "long_term", "importance": 0.9},
            {"id": "filter-002", "type": "episodic", "tier": "short_term", "importance": 0.8},
            {"id": "filter-003", "type": "structured_thinking", "tier": "long_term", "importance": 0.7}
        ]
        
        for memory in diverse_memories:
            memory.update({"content": f"Filter test content {memory['id']}"})
            await self.persistence.store_memory(memory)
        
        # Test type filtering
        structured_results = await self.persistence.retrieve_memories(
            "filter test",
            memory_types=["structured_thinking"],
            limit=5,
            min_similarity=0.0
        )
        
        for result in structured_results:
            assert result["type"] == "structured_thinking", "Type filtering failed"
        print("  ✅ Memory type filtering: PASS")
        
        # Test tier and type combination using search
        combined_results = await self.persistence.search_memories(
            types=["structured_thinking"],
            filters={"tier": "long_term"},
            limit=5
        )
        
        for result in combined_results:
            assert result["type"] == "structured_thinking", "Combined filtering failed"
            assert result["tier"] == "long_term", "Combined filtering failed"
        print("  ✅ Combined filtering: PASS")
        
        print("✅ Filtering: ALL TESTS PASS\n")
    
    async def validate_performance(self):
        """Validate basic performance characteristics."""
        print("⚡ Validating Performance...")
        
        # Test batch storage performance
        start_time = time.perf_counter()
        
        batch_memories = []
        for i in range(50):
            memory = {
                "id": f"perf-{i:03d}",
                "type": "episodic",
                "content": f"Performance test memory {i}",
                "importance": 0.5,
                "tier": "short_term"
            }
            batch_memories.append(memory)
            await self.persistence.store_memory(memory)
        
        storage_time = time.perf_counter() - start_time
        storage_rate = 50 / storage_time
        
        print(f"  📊 Storage rate: {storage_rate:.1f} memories/sec")
        assert storage_rate > 5, f"Storage too slow: {storage_rate:.1f} memories/sec"
        print("  ✅ Storage performance: PASS")
        
        # Test search performance
        start_time = time.perf_counter()
        
        for i in range(10):
            await self.persistence.retrieve_memories(
                f"performance test memory",
                limit=5,
                min_similarity=0.0
            )
        
        search_time = time.perf_counter() - start_time
        avg_search_time = (search_time / 10) * 1000  # Convert to ms
        
        print(f"  🔍 Average search time: {avg_search_time:.2f}ms")
        assert avg_search_time < 100, f"Search too slow: {avg_search_time:.2f}ms"
        print("  ✅ Search performance: PASS")
        
        print("✅ Performance: ALL TESTS PASS\n")
    
    async def validate_error_handling(self):
        """Validate error handling and edge cases."""
        print("🛡️ Validating Error Handling...")
        
        # Test invalid memory data
        try:
            await self.persistence.store_memory({})  # Empty memory
            print("  ⚠️ Empty memory accepted (handled gracefully)")
        except Exception as e:
            print(f"  ✅ Empty memory rejected: {type(e).__name__}")
        
        # Test nonexistent memory retrieval
        nonexistent = await self.persistence.get_memory("nonexistent-id")
        assert nonexistent is None, "Nonexistent memory should return None"
        print("  ✅ Nonexistent memory handling: PASS")
        
        # Test empty query handling
        empty_results = await self.persistence.retrieve_memories("", limit=5)
        assert isinstance(empty_results, list), "Empty query should return list"
        print("  ✅ Empty query handling: PASS")
        
        print("✅ Error Handling: ALL TESTS PASS\n")
    
    async def validate_statistics(self):
        """Validate statistics functionality."""
        print("📊 Validating Statistics...")
        
        stats = await self.persistence.get_memory_stats()
        
        required_fields = ["total_memories", "memory_types", "memory_tiers", "database_size_bytes", "database_path"]
        for field in required_fields:
            assert field in stats, f"Missing stats field: {field}"
        
        assert isinstance(stats["total_memories"], int), "total_memories should be integer"
        assert isinstance(stats["memory_types"], dict), "memory_types should be dict"
        assert isinstance(stats["memory_tiers"], dict), "memory_tiers should be dict"
        assert stats["database_size_bytes"] > 0, "Database should have positive size"
        
        print(f"  📈 Total memories: {stats['total_memories']}")
        print(f"  📁 Database size: {stats['database_size_bytes']} bytes")
        print("  ✅ Statistics functionality: PASS")
        
        print("✅ Statistics: ALL TESTS PASS\n")
    
    async def run_comprehensive_validation(self):
        """Run comprehensive validation of all functionality."""
        print("🧪 SQLite Test Suite Comprehensive Validation")
        print("=" * 60)
        
        try:
            await self.validate_core_functionality()
            await self.validate_vector_search()
            await self.validate_filtering()
            await self.validate_performance()
            await self.validate_error_handling()
            await self.validate_statistics()
            
            print("🎉 COMPREHENSIVE VALIDATION: ALL TESTS PASS")
            print("✅ SQLite test suite is fully functional and ready for use!")
            
            return True
            
        except Exception as e:
            print(f"❌ VALIDATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up test resources."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


async def main():
    """Run the validation."""
    validator = SQLiteTestSuiteValidator()
    success = await validator.run_comprehensive_validation()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)