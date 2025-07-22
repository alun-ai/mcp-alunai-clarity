"""
Integration tests using the systematic data validation framework.

These tests validate complete MCP-to-Qdrant data lifecycle for all
memory operations, ensuring data integrity at every layer.
"""

import pytest
import asyncio
import time
from typing import Dict, Any

from tests.framework import DataValidationTestSuite


@pytest.mark.integration
class TestMemoryOperationsDataValidation(DataValidationTestSuite):
    """
    Systematic validation of all memory operations.
    
    Each test validates:
    1. MCP function behavior
    2. Qdrant data storage accuracy
    3. Vector embedding correctness
    4. Search indexing validation
    5. Data retrieval consistency
    """
    
    @pytest.mark.asyncio
    async def test_store_simple_text_memory_complete_validation(self):
        """
        Validate storing a simple text memory end-to-end.
        
        Tests: MCP store_memory -> Qdrant storage -> vector generation -> search indexing
        """
        # Setup
        await self.setup_test_environment()
        
        try:
            memory_type = "user_note"
            content = {"note": "This is a test note about machine learning concepts"}
            metadata = {
                "source": "test_suite",
                "category": "technical",
                "tags": ["machine_learning", "test"]
            }
            
            result = await self.validate_complete_memory_lifecycle(
                memory_type=memory_type,
                content=content,
                expected_metadata=metadata,
                test_name="store_simple_text_memory"
            )
        
            # Complete validation - all functionality should now work perfectly
            print(f"\n🧪 Test Results:")
            print(f"   ✅ Memory stored: {result.mcp_result['memory_id'] is not None}")
            print(f"   ✅ Vector generated: {result.validation_details['vector_dimensions']} dimensions")
            print(f"   ✅ Search working: {result.validation_details['found_in_search']}")
            print(f"   ✅ Update working: {result.mcp_result['update_success']}")
            print(f"   ✅ Metadata preserved: {len([e for e in result.errors if 'Metadata' in e])} issues")
            print(f"   📊 Performance: Store={result.performance_metrics['store_time_ms']:.1f}ms, Search={result.performance_metrics['search_time_ms']:.1f}ms")
            
            # Full validation - should pass completely now
            assert result.passed, f"Test failed: {result.errors}"
            assert result.mcp_result["memory_id"] is not None
            assert result.validation_details["vector_dimensions"] == 384
            assert result.validation_details["found_in_search"] is True
            assert result.mcp_result["update_success"] is True
            
            # Performance validation
            assert result.performance_metrics["search_time_ms"] < 5000, "Search operation too slow"
            
        finally:
            # Cleanup
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_metadata_preservation_validation(self):
        """
        Test that the metadata preservation bug fix works correctly.
        
        Validates that custom metadata fields are preserved at the top level.
        """
        await self.setup_test_environment()
        
        try:
            # Test with various metadata types
            memory_type = "test_metadata"
            content = {"note": "Testing metadata preservation"}
            metadata = {
                "source": "bug_fix_test",
                "priority": "high", 
                "tags": ["bug_fix", "metadata", "test"],
                "numeric_field": 42,
                "boolean_field": True,
                "nested_object": {"key": "value", "count": 10}
            }
            
            result = await self.validate_complete_memory_lifecycle(
                memory_type=memory_type,
                content=content,
                expected_metadata=metadata,
                test_name="metadata_preservation_test"
            )
            
            print(f"\n🔧 Metadata Preservation Test:")
            print(f"   ✅ All metadata fields preserved: {result.passed}")
            print(f"   📋 Tested fields: {list(metadata.keys())}")
            if result.errors:
                print(f"   ❌ Failed fields: {result.errors}")
            
            assert result.passed, f"Metadata preservation failed: {result.errors}"
            
        finally:
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_store_structured_data_memory_validation(self):
        """
        Validate storing structured data with complex metadata.
        
        Tests: Complex JSON content -> embedding generation -> metadata preservation
        """
        memory_type = "structured_data"
        content = {
            "title": "API Documentation",
            "endpoint": "/api/v1/users",
            "method": "POST",
            "parameters": {
                "name": "string",
                "email": "string",
                "role": "enum"
            },
            "response": {
                "user_id": "uuid",
                "created_at": "timestamp"
            }
        }
        metadata = {
            "api_version": "v1",
            "documentation_type": "endpoint",
            "last_updated": "2025-01-20",
            "complexity_level": "intermediate"
        }
        
        result = await self.validate_complete_memory_lifecycle(
            memory_type=memory_type,
            content=content,
            expected_metadata=metadata,
            test_name="store_structured_data_memory"
        )
        
        assert result.passed, f"Test failed: {result.errors}"
        
        # Validate complex content preservation
        stored_content = result.validation_details["qdrant_payload"]["content"]
        assert stored_content["endpoint"] == "/api/v1/users"
        assert stored_content["parameters"]["name"] == "string"
        
        # Validate search with structured data
        assert result.validation_details["found_in_search"] is True
    
    @pytest.mark.asyncio
    async def test_store_multilingual_memory_validation(self):
        """
        Validate storing multilingual content.
        
        Tests: Unicode handling -> embedding generation -> search functionality
        """
        memory_type = "multilingual_note"
        content = {
            "english": "Hello, this is a test message",
            "spanish": "Hola, este es un mensaje de prueba", 
            "japanese": "こんにちは、これはテストメッセージです",
            "arabic": "مرحبا، هذه رسالة اختبار",
            "emoji": "🌍🚀✨🧪🎯"
        }
        metadata = {
            "languages": ["en", "es", "ja", "ar"],
            "content_type": "multilingual_test",
            "encoding": "utf-8"
        }
        
        result = await self.validate_complete_memory_lifecycle(
            memory_type=memory_type,
            content=content,
            expected_metadata=metadata,
            test_name="store_multilingual_memory"
        )
        
        assert result.passed, f"Test failed: {result.errors}"
        
        # Validate Unicode preservation
        stored_content = result.validation_details["qdrant_payload"]["content"]
        assert stored_content["japanese"] == "こんにちは、これはテストメッセージです"
        assert stored_content["emoji"] == "🌍🚀✨🧪🎯"
    
    @pytest.mark.asyncio
    async def test_update_memory_data_consistency_validation(self):
        """
        Validate memory updates maintain data consistency.
        
        Tests: Original storage -> update operation -> data consistency -> search accuracy
        """
        # First, store a memory
        memory_type = "updatable_note"
        original_content = {"note": "Original content for testing updates"}
        original_metadata = {"version": 1, "status": "draft"}
        
        result = await self.validate_complete_memory_lifecycle(
            memory_type=memory_type,
            content=original_content,
            expected_metadata=original_metadata,
            test_name="store_for_update_test"
        )
        
        assert result.passed, "Initial storage failed"
        memory_id = result.mcp_result["memory_id"]
        
        # Now test update functionality with validation
        update_data = {
            "note": "Updated content with new information",
            "version": 2,
            "status": "published",
            "last_modified": "2025-01-20T10:30:00Z"
        }
        
        update_success = await self.persistence_domain.update_memory(memory_id, update_data)
        assert update_success, "Memory update failed"
        
        # Validate update in Qdrant
        updated_point = await self.qdrant_inspector.get_raw_point_by_id(memory_id)
        assert updated_point is not None, "Updated memory not found in Qdrant"
        
        # Validate all updates are reflected
        payload = updated_point.payload
        assert payload["note"] == "Updated content with new information"
        assert payload["version"] == 2
        assert payload["status"] == "published"
        assert "updated_at" in payload  # Should have update timestamp
        
        # Validate search still works with updated content
        search_results = await self.persistence_domain.search_memories(
            query="Updated content new information",
            limit=10,
            types=[memory_type]
        )
        
        found_memory = next((r for r in search_results if r.get("memory_id") == memory_id), None)
        assert found_memory is not None, "Updated memory not found in search"
    
    @pytest.mark.asyncio
    async def test_delete_memory_cleanup_validation(self):
        """
        Validate memory deletion completely removes data.
        
        Tests: Storage -> deletion -> cleanup verification -> search removal
        """
        # Store a memory to delete
        memory_type = "deletable_note"
        content = {"note": "This memory will be deleted"}
        metadata = {"temporary": True}
        
        result = await self.validate_complete_memory_lifecycle(
            memory_type=memory_type,
            content=content,
            expected_metadata=metadata,
            test_name="store_for_deletion_test"
        )
        
        assert result.passed, "Initial storage failed"
        memory_id = result.mcp_result["memory_id"]
        
        # Verify memory exists before deletion
        pre_delete_point = await self.qdrant_inspector.get_raw_point_by_id(memory_id)
        assert pre_delete_point is not None, "Memory not found before deletion"
        
        # Delete the memory
        deleted_ids = await self.persistence_domain.delete_memories([memory_id])
        assert memory_id in deleted_ids, "Memory not reported as deleted"
        
        # Validate complete removal from Qdrant
        post_delete_point = await self.qdrant_inspector.get_raw_point_by_id(memory_id)
        assert post_delete_point is None, "Memory still exists in Qdrant after deletion"
        
        # Validate removal from search results
        search_results = await self.persistence_domain.search_memories(
            query="This memory will be deleted",
            limit=10,
            types=[memory_type]
        )
        
        found_deleted = any(r.get("memory_id") == memory_id for r in search_results)
        assert not found_deleted, "Deleted memory still appears in search results"
    
    @pytest.mark.asyncio 
    async def test_search_accuracy_validation(self):
        """
        Validate search accuracy and ranking.
        
        Tests: Multiple memories -> search queries -> result relevance -> ranking accuracy
        """
        # Store multiple memories with known relationships
        memories_data = [
            {
                "type": "tech_doc",
                "content": {"title": "Python Machine Learning", "topic": "artificial intelligence algorithms"},
                "metadata": {"category": "programming", "difficulty": "advanced"}
            },
            {
                "type": "tech_doc", 
                "content": {"title": "JavaScript Basics", "topic": "web development fundamentals"},
                "metadata": {"category": "programming", "difficulty": "beginner"}
            },
            {
                "type": "tech_doc",
                "content": {"title": "AI Neural Networks", "topic": "machine learning deep learning"},
                "metadata": {"category": "ai", "difficulty": "expert"}
            },
            {
                "type": "personal_note",
                "content": {"note": "Remember to buy groceries and call mom"},
                "metadata": {"category": "personal", "priority": "high"}
            }
        ]
        
        stored_memory_ids = set()
        
        # Store all test memories
        for memory_data in memories_data:
            result = await self.validate_complete_memory_lifecycle(
                memory_type=memory_data["type"],
                content=memory_data["content"],
                expected_metadata=memory_data["metadata"],
                test_name=f"search_test_{memory_data['type']}"
            )
            assert result.passed, f"Failed to store {memory_data['type']}"
            stored_memory_ids.add(result.mcp_result["memory_id"])
        
        # Test search accuracy for AI-related query
        ai_search_result = await self.validate_search_accuracy(
            query_text="machine learning artificial intelligence",
            expected_memory_ids=set(),  # We expect the AI-related memories
            memory_types=["tech_doc"],
            test_name="ai_search_accuracy"
        )
        
        # The search should find AI-related memories with high relevance
        ai_results = ai_search_result.mcp_result["search_results"]
        ai_titles = [r.get("content", {}).get("title", "") for r in ai_results]
        
        # Should find both AI-related documents
        assert any("Machine Learning" in title for title in ai_titles), "ML document not found"
        assert any("Neural Networks" in title for title in ai_titles), "Neural networks document not found"
        
        # Should rank AI-specific content higher than general programming
        if len(ai_results) >= 2:
            # First result should be more relevant to AI
            first_result_content = str(ai_results[0].get("content", {}))
            assert any(term in first_result_content.lower() 
                      for term in ["neural", "artificial", "machine learning"]), \
                   "Most relevant result doesn't contain AI terms"
    
    @pytest.mark.asyncio
    async def test_data_consistency_validation(self):
        """
        Validate overall data consistency across the system.
        
        Tests: Index integrity -> vector consistency -> metadata accuracy
        """
        # Store several memories to create a realistic dataset
        test_memories = [
            ("consistency_test", {"data": f"Test memory {i}"}, {"index": i})
            for i in range(5)
        ]
        
        memory_ids = []
        for memory_type, content, metadata in test_memories:
            result = await self.validate_complete_memory_lifecycle(
                memory_type=memory_type,
                content=content,
                expected_metadata=metadata,
                test_name=f"consistency_memory_{metadata['index']}"
            )
            assert result.passed, f"Failed to store consistency test memory {metadata['index']}"
            memory_ids.append(result.mcp_result["memory_id"])
        
        # Perform comprehensive consistency validation
        consistency_result = await self.validate_data_consistency("overall_consistency_check")
        
        assert consistency_result.passed, f"Data consistency check failed: {consistency_result.errors}"
        
        # Validate specific consistency metrics
        stats = consistency_result.validation_details["collection_stats"]
        assert stats["total_points"] >= 5, "Not all test memories were stored"
        assert stats["indexed_vectors"] > 0, "No vectors are indexed"
        
        # Should have no duplicate vectors for our unique test data
        assert consistency_result.validation_details["duplicate_vectors"] == 0, \
               "Unexpected duplicate vectors found"
    
    @pytest.mark.asyncio
    async def test_performance_baseline_validation(self):
        """
        Establish performance baselines for all operations.
        
        Tests: Operation timing -> throughput -> resource usage
        """
        # Test batch storage performance
        batch_size = 10
        start_time = time.perf_counter()
        
        batch_memory_ids = []
        for i in range(batch_size):
            result = await self.validate_complete_memory_lifecycle(
                memory_type="performance_test",
                content={"test_data": f"Performance test memory {i}", "batch_index": i},
                expected_metadata={"batch_id": "perf_test_1", "index": i},
                test_name=f"performance_memory_{i}"
            )
            assert result.passed, f"Batch memory {i} failed"
            batch_memory_ids.append(result.mcp_result["memory_id"])
        
        batch_time = (time.perf_counter() - start_time) * 1000
        avg_time_per_memory = batch_time / batch_size
        
        print(f"\n📊 Performance Baseline Results:")
        print(f"   Batch size: {batch_size} memories")
        print(f"   Total time: {batch_time:.2f}ms")
        print(f"   Average per memory: {avg_time_per_memory:.2f}ms")
        print(f"   Throughput: {(batch_size / batch_time * 1000):.1f} memories/second")
        
        # Performance assertions
        assert avg_time_per_memory < 2000, f"Average time per memory too slow: {avg_time_per_memory}ms"
        assert batch_time < 20000, f"Batch processing too slow: {batch_time}ms"
        
        # Test search performance with larger dataset
        search_start = time.perf_counter()
        search_results = await self.persistence_domain.search_memories(
            query="Performance test memory",
            limit=batch_size,
            types=["performance_test"]
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        print(f"   Search time: {search_time:.2f}ms")
        print(f"   Search results: {len(search_results)} found")
        
        assert search_time < 5000, f"Search too slow: {search_time}ms"
        assert len(search_results) == batch_size, f"Not all memories found in search: {len(search_results)}/{batch_size}"
    
    def test_generate_comprehensive_report(self):
        """Generate and validate test report."""
        report = self.generate_test_report()
        
        assert "DATA VALIDATION TEST REPORT" in report
        assert "Total tests:" in report
        assert "Success rate:" in report
        
        print(f"\n{report}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_memory_operations_validation.py -v -s
    pytest.main([__file__, "-v", "-s"])