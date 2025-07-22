"""
Critical MCP Features Test Suite.

This test suite validates the core MCP memory operations that were previously
failing due to response format validation issues. It serves as a regression
test to ensure these critical features continue working.

Key features tested:
- MCP Retrieve Memory Tool response format validation
- Search Results response schema compliance  
- End-to-end memory operations at scale
- Temporal domain calculations and field naming

Run with: pytest tests/integration/test_mcp_critical_features.py -v
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
from tests.framework.mcp_validation import MCPServerTestSuite
from tests.integration.test_mcp_large_dataset_validation import LargeDatasetGenerator


class TestCriticalMCPFeatures:
    """Test suite for critical MCP memory features."""

    @pytest.mark.asyncio
    async def test_mcp_retrieve_memory_format_validation(self):
        """
        Test that retrieve_memory returns properly formatted responses.
        
        This addresses the critical issue where MCP validation was failing
        due to field naming mismatches and temporal domain errors.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Store test memories with various metadata
            test_memories = [
                {
                    "memory_type": "debug_test",
                    "content": "This is debug content for MCP validation testing",
                    "importance": 0.8,
                    "metadata": {"test_type": "format_validation", "source": "test_suite"}
                },
                {
                    "memory_type": "programming_knowledge", 
                    "content": "Python async/await patterns and best practices",
                    "importance": 0.9,
                    "metadata": {"language": "python", "topic": "async"}
                },
                {
                    "memory_type": "technical_documentation",
                    "content": "MCP protocol response format specifications",
                    "importance": 0.7,
                    "metadata": {"protocol": "mcp", "type": "specification"}
                }
            ]
            
            stored_ids = []
            for i, memory_data in enumerate(test_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"critical_store_{i}"
                )
                
                assert result.passed, f"Failed to store memory {i}: {result.errors}"
                memory_id = result.parsed_response["memory_id"]
                stored_ids.append(memory_id)
            
            # Test retrieval with various parameters
            retrieval_tests = [
                {
                    "query": "debug content testing",
                    "limit": 5,
                    "min_similarity": 0.3,
                    "include_metadata": True
                },
                {
                    "query": "python async programming", 
                    "limit": 3,
                    "min_similarity": 0.4,
                    "include_metadata": False
                },
                {
                    "query": "MCP protocol format",
                    "limit": 2, 
                    "min_similarity": 0.5,
                    "include_metadata": True
                }
            ]
            
            for i, retrieval_args in enumerate(retrieval_tests):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments=retrieval_args,
                    test_name=f"critical_retrieve_{i}"
                )
                
                # Validate response format
                assert result.passed, f"Retrieval test {i} failed: {result.errors}"
                assert result.response_validation["valid_json"], "Response must be valid JSON"
                assert result.response_validation["schema_compliant"], "Response must be schema compliant"
                
                # Validate memory structure
                memories = result.parsed_response.get("memories", [])
                assert isinstance(memories, list), "Memories must be a list"
                
                for j, memory in enumerate(memories):
                    assert isinstance(memory, dict), f"Memory {j} must be a dictionary"
                    
                    # Check critical fields (flexible naming)
                    assert "id" in memory, f"Memory {j} missing 'id' field"
                    assert "content" in memory, f"Memory {j} missing 'content' field"
                    assert ("memory_type" in memory or "type" in memory), f"Memory {j} missing type field"
                    assert ("similarity_score" in memory or "similarity" in memory), f"Memory {j} missing similarity field"
                    
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio 
    async def test_search_functionality_validation(self):
        """
        Test search functionality with various query types.
        
        Validates that search results return proper schema-compliant responses
        for different query patterns and similarity thresholds.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create diverse test dataset
            generator = LargeDatasetGenerator()
            dataset = generator.generate_large_dataset(25)  # Smaller focused dataset
            
            # Store dataset
            stored_ids = []
            for i, memory_data in enumerate(dataset):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"search_store_{i}"
                )
                
                assert result.passed, f"Failed to store memory {i}: {result.errors}"
                stored_ids.append(result.parsed_response["memory_id"])
            
            # Test various search patterns
            search_tests = [
                ("programming knowledge", 0.4),
                ("code review best practices", 0.3), 
                ("technical documentation", 0.5),
                ("meeting notes summary", 0.3),
                ("project architecture", 0.4),
                ("python async patterns", 0.6),
                ("database optimization", 0.5)
            ]
            
            successful_searches = 0
            for query, min_similarity in search_tests:
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments={
                        "query": query,
                        "limit": 3,
                        "min_similarity": min_similarity,
                        "include_metadata": True
                    },
                    test_name=f"search_{query.replace(' ', '_')}"
                )
                
                # Validate search response format
                assert result.passed, f"Search for '{query}' failed: {result.errors}"
                assert result.response_validation["valid_json"], f"Search '{query}' returned invalid JSON"
                assert result.response_validation["schema_compliant"], f"Search '{query}' not schema compliant"
                
                memories = result.parsed_response.get("memories", [])
                if memories:  # Some searches may return no results due to similarity threshold
                    successful_searches += 1
                    
                    # Validate each returned memory
                    for memory in memories:
                        assert isinstance(memory, dict), "Search result must be dictionary"
                        assert "id" in memory, "Search result missing 'id'"
                        assert "content" in memory, "Search result missing 'content'"
                        
                        # Validate similarity score
                        similarity = memory.get("similarity_score") or memory.get("similarity", 0)
                        assert similarity >= min_similarity, f"Similarity {similarity} below threshold {min_similarity}"
            
            # Ensure at least some searches returned results
            assert successful_searches >= 3, f"Only {successful_searches} searches returned results, expected at least 3"
            
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_large_scale_memory_operations(self):
        """
        Test end-to-end memory operations at scale.
        
        Validates that the memory system can handle larger datasets
        while maintaining proper response formatting and performance.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Test with moderately sized dataset
            dataset_size = 100
            generator = LargeDatasetGenerator()
            dataset = generator.generate_large_dataset(dataset_size)
            
            # Track performance
            start_time = time.perf_counter()
            store_times = []
            stored_ids = []
            
            # Store all memories
            for i, memory_data in enumerate(dataset):
                store_start = time.perf_counter()
                
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"scale_store_{i}"
                )
                
                store_time = (time.perf_counter() - store_start) * 1000
                store_times.append(store_time)
                
                assert result.passed, f"Failed to store memory {i}: {result.errors}"
                stored_ids.append(result.parsed_response["memory_id"])
            
            total_time = (time.perf_counter() - start_time) * 1000
            avg_store_time = sum(store_times) / len(store_times)
            
            # Validate performance metrics
            assert len(stored_ids) == dataset_size, f"Expected {dataset_size} memories, got {len(stored_ids)}"
            assert avg_store_time < 100, f"Average store time too slow: {avg_store_time:.1f}ms"
            
            # Test bulk retrieval operations
            retrieval_tests = [
                {"query": "programming", "limit": 10, "min_similarity": 0.3},
                {"query": "documentation", "limit": 5, "min_similarity": 0.4},
                {"query": "project management", "limit": 8, "min_similarity": 0.35},
                {"query": "technical review", "limit": 6, "min_similarity": 0.45}
            ]
            
            total_retrieved = 0
            for i, retrieval_args in enumerate(retrieval_tests):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory", 
                    arguments=retrieval_args,
                    test_name=f"scale_retrieve_{i}"
                )
                
                assert result.passed, f"Bulk retrieval {i} failed: {result.errors}"
                
                memories = result.parsed_response.get("memories", [])
                total_retrieved += len(memories)
                
                # Validate response structure for each memory
                for memory in memories:
                    assert isinstance(memory, dict), "Memory must be dictionary"
                    assert "id" in memory, "Memory missing 'id'"
                    assert "content" in memory, "Memory missing 'content'"
            
            # Validate we can retrieve a reasonable amount of data
            assert total_retrieved >= 10, f"Retrieved too few memories: {total_retrieved}"
            
            # Get collection statistics
            collection_stats = await suite.qdrant_inspector.get_collection_stats()
            assert collection_stats["total_points"] >= dataset_size, "Collection should contain at least stored memories"
            
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_temporal_domain_calculations(self):
        """
        Test temporal domain calculations and field handling.
        
        Specifically validates the fixes for:
        - KeyError: 'retrieval' configuration access
        - ZeroDivisionError in recency score calculation
        - Proper temporal adjustment of memory relevance
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Store memories with different importance levels
            test_memories = [
                {
                    "memory_type": "temporal_test",
                    "content": "High importance memory for temporal testing",
                    "importance": 0.9,
                    "metadata": {"test_type": "high_importance"}
                },
                {
                    "memory_type": "temporal_test",
                    "content": "Medium importance memory for temporal testing", 
                    "importance": 0.5,
                    "metadata": {"test_type": "medium_importance"}
                },
                {
                    "memory_type": "temporal_test",
                    "content": "Low importance memory for temporal testing",
                    "importance": 0.1,
                    "metadata": {"test_type": "low_importance"}
                }
            ]
            
            stored_ids = []
            for i, memory_data in enumerate(test_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"temporal_store_{i}"
                )
                
                assert result.passed, f"Failed to store temporal memory {i}: {result.errors}"
                stored_ids.append(result.parsed_response["memory_id"])
            
            # Retrieve memories to trigger temporal processing
            result = await suite.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": "temporal testing memory",
                    "limit": 5,
                    "min_similarity": 0.2,
                    "include_metadata": True
                },
                test_name="temporal_retrieve"
            )
            
            assert result.passed, f"Temporal retrieval failed: {result.errors}"
            
            memories = result.parsed_response.get("memories", [])
            assert len(memories) >= 2, "Should retrieve at least 2 temporal test memories"
            
            # Validate temporal fields are present and valid
            for memory in memories:
                assert "id" in memory, "Memory missing 'id'"
                assert "content" in memory, "Memory missing 'content'"
                
                # Check for temporal-related fields (may be in various formats)
                has_similarity = "similarity_score" in memory or "similarity" in memory
                assert has_similarity, "Memory missing similarity score"
                
                # Validate similarity score is reasonable
                similarity = memory.get("similarity_score") or memory.get("similarity", 0)
                assert 0 <= similarity <= 1, f"Invalid similarity score: {similarity}"
                
                # If adjusted scores are present, validate they're reasonable
                if "adjusted_score" in memory:
                    assert 0 <= memory["adjusted_score"] <= 1, f"Invalid adjusted score: {memory['adjusted_score']}"
                
                if "recency_score" in memory:
                    assert 0 <= memory["recency_score"] <= 1, f"Invalid recency score: {memory['recency_score']}"
            
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_mcp_error_handling_validation(self):
        """
        Test error handling with proper MCP response formatting.
        
        Validates that error responses follow the correct schema
        and contain appropriate error information.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Test various error conditions
            error_tests = [
                {
                    "tool_name": "retrieve_memory",
                    "arguments": {"query": "", "limit": -1},  # Invalid limit
                    "expected_error_pattern": None  # Any error is fine
                },
                {
                    "tool_name": "update_memory", 
                    "arguments": {"memory_id": "nonexistent_id", "updates": {}},
                    "expected_error_pattern": None
                },
                {
                    "tool_name": "delete_memory",
                    "arguments": {"memory_ids": ["nonexistent_id_1", "nonexistent_id_2"]}, 
                    "expected_error_pattern": None
                }
            ]
            
            for i, error_test in enumerate(error_tests):
                result = await suite.validate_mcp_error_handling(
                    tool_name=error_test["tool_name"],
                    invalid_arguments=error_test["arguments"],
                    expected_error_pattern=error_test["expected_error_pattern"],
                    test_name=f"error_handling_{i}"
                )
                
                # For error handling tests, we expect the tool to either:
                # 1. Return a proper error response (success=False)
                # 2. Handle the error gracefully without crashing
                
                # The key is that it doesn't crash and returns a valid response
                assert result.response_validation.get("valid_json", False), "Error response must be valid JSON"
                
                # If it's a successful error response format, validate it
                if result.parsed_response.get("success") is False:
                    assert "error" in result.parsed_response, "Error response should contain error message"
            
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_critical_tests():
        """Run critical tests directly for debugging."""
        test_suite = TestCriticalMCPFeatures()
        
        print("🧪 Running critical MCP feature tests...")
        
        print("\n1️⃣ Testing MCP retrieve memory format validation...")
        await test_suite.test_mcp_retrieve_memory_format_validation()
        print("✅ Retrieve memory format validation passed")
        
        print("\n2️⃣ Testing search functionality validation...")
        await test_suite.test_search_functionality_validation()
        print("✅ Search functionality validation passed")
        
        print("\n3️⃣ Testing large scale memory operations...")
        await test_suite.test_large_scale_memory_operations()
        print("✅ Large scale operations passed")
        
        print("\n4️⃣ Testing temporal domain calculations...")
        await test_suite.test_temporal_domain_calculations()
        print("✅ Temporal domain calculations passed")
        
        print("\n5️⃣ Testing MCP error handling validation...")
        await test_suite.test_mcp_error_handling_validation()
        print("✅ Error handling validation passed")
        
        print("\n🎉 All critical MCP feature tests passed!")
    
    asyncio.run(run_critical_tests())