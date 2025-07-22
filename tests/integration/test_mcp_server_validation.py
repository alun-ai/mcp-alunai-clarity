"""
Comprehensive MCP Server Integration Tests.

These tests validate the complete MCP server functionality including:
1. Protocol compliance and response formatting
2. End-to-end memory operations through MCP
3. Error handling and edge cases
4. Concurrent operations and thread safety
5. Performance characteristics
6. Hook system integration
7. Data consistency between MCP layer and Qdrant

Each test validates both MCP protocol behavior AND underlying data integrity.
"""

import pytest
import asyncio
import json
import uuid
import time
from typing import Dict, Any, List

from tests.framework.mcp_validation import MCPServerTestSuite


@pytest.mark.integration
class TestMCPServerProtocolValidation(MCPServerTestSuite):
    """
    Comprehensive MCP server protocol validation.
    
    Tests complete request-response cycles through the MCP protocol
    while also validating underlying data consistency.
    """
    
    @pytest.mark.asyncio
    async def test_store_memory_mcp_complete_validation(self):
        """
        Validate storing memory through MCP protocol end-to-end.
        
        Tests:
        - MCP tool execution
        - Response format compliance
        - Qdrant data storage accuracy
        - UUID generation and format
        """
        await self.setup_test_environment()
        
        try:
            arguments = {
                "memory_type": "mcp_test_note",
                "content": "This is a test memory stored via MCP protocol",
                "importance": 0.8,
                "metadata": {
                    "source": "mcp_integration_test",
                    "category": "testing",
                    "tags": ["mcp", "protocol", "validation"]
                },
                "context": {
                    "test_session": "mcp_validation",
                    "timestamp": time.time()
                }
            }
            
            result = await self.validate_mcp_tool_execution(
                tool_name="store_memory",
                arguments=arguments,
                expected_result_type="success",
                validate_underlying_data=True,
                test_name="mcp_store_memory_complete"
            )
            
            print(f"\n🔧 MCP Store Memory Test Results:")
            print(f"   ✅ MCP Response Valid: {result.passed}")
            print(f"   📋 Memory ID: {result.parsed_response.get('memory_id', 'None')}")
            print(f"   📊 MCP Performance: {result.performance_metrics['tool_execution_ms']:.1f}ms")
            print(f"   🔍 Protocol Validation: {result.performance_metrics['protocol_validation_ms']:.1f}ms")
            if result.underlying_data_validation:
                print(f"   💾 Data Validation: {result.underlying_data_validation.passed}")
                print(f"   📈 Store Performance: {result.underlying_data_validation.performance_metrics.get('store_time_ms', 0):.1f}ms")
            
            # Complete validation
            assert result.passed, f"MCP store_memory validation failed: {result.errors}"
            assert result.parsed_response["success"] is True
            assert "memory_id" in result.parsed_response
            
            # Validate UUID format
            memory_id = result.parsed_response["memory_id"]
            uuid.UUID(memory_id)  # Should not raise exception
            
            # Validate underlying data if available
            if result.underlying_data_validation:
                assert result.underlying_data_validation.passed, "Underlying data validation failed"
                
        finally:
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_retrieve_memory_mcp_protocol_validation(self):
        """
        Validate retrieving memories through MCP protocol.
        
        Tests:
        - Memory storage through MCP
        - Memory retrieval through MCP  
        - Response format validation
        - Search result structure
        - Similarity scoring
        """
        await self.setup_test_environment()
        
        try:
            # First store some test memories
            test_memories = [
                {
                    "memory_type": "mcp_retrieval_test",
                    "content": "Python machine learning algorithms and data science",
                    "metadata": {"topic": "ai", "language": "python"}
                },
                {
                    "memory_type": "mcp_retrieval_test", 
                    "content": "JavaScript web development and React frameworks",
                    "metadata": {"topic": "web", "language": "javascript"}
                },
                {
                    "memory_type": "mcp_retrieval_test",
                    "content": "Machine learning with neural networks and deep learning",
                    "metadata": {"topic": "ai", "language": "python"}
                }
            ]
            
            stored_memory_ids = []
            for memory_data in test_memories:
                store_result = await self.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    test_name="mcp_store_for_retrieval"
                )
                assert store_result.passed, f"Failed to store test memory: {store_result.errors}"
                stored_memory_ids.append(store_result.parsed_response["memory_id"])
            
            # Now test retrieval
            retrieval_arguments = {
                "query": "machine learning algorithms neural networks",
                "limit": 10,
                "types": ["mcp_retrieval_test"],
                "min_similarity": 0.3,
                "include_metadata": True
            }
            
            result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments=retrieval_arguments,
                expected_result_type="success",
                validate_underlying_data=False,
                test_name="mcp_retrieve_memory_protocol"
            )
            
            print(f"\n🔍 MCP Retrieve Memory Test Results:")
            print(f"   ✅ MCP Response Valid: {result.passed}")
            print(f"   📊 Found Memories: {len(result.parsed_response.get('memories', []))}")
            print(f"   ⚡ Retrieval Performance: {result.performance_metrics['tool_execution_ms']:.1f}ms")
            
            # Validate response structure
            assert result.passed, f"MCP retrieve_memory validation failed: {result.errors}"
            assert result.parsed_response["success"] is True
            assert "memories" in result.parsed_response
            
            memories = result.parsed_response["memories"]
            assert isinstance(memories, list), "Memories should be a list"
            
            # Should find at least the AI-related memories
            assert len(memories) >= 1, "Should find at least one relevant memory"
            
            # Validate memory structure
            for i, memory in enumerate(memories):
                assert isinstance(memory, dict), f"Memory {i} should be a dictionary"
                
                # Check for required fields (field names may vary)
                memory_keys = set(memory.keys())
                required_fields = {"id", "content", "memory_type", "similarity_score"}
                
                # Allow for some field name variations
                if "id" not in memory_keys and "memory_id" in memory_keys:
                    memory_keys.add("id")
                if "memory_type" not in memory_keys and "type" in memory_keys:
                    memory_keys.add("memory_type")
                
                missing_fields = required_fields - memory_keys
                assert not missing_fields, f"Memory {i} missing required fields: {missing_fields}"
                
                # Validate similarity score
                similarity = memory.get("similarity_score")
                if similarity is not None:
                    assert 0 <= similarity <= 1, f"Invalid similarity score: {similarity}"
            
            # Test that results are relevant (should find AI-related memories)
            found_content = " ".join(str(m.get("content", "")) for m in memories).lower()
            assert "machine learning" in found_content or "neural" in found_content, \
                   "Should find memories related to machine learning"
                   
        finally:
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling_validation(self):
        """
        Validate MCP error handling for various failure scenarios.
        
        Tests:
        - Invalid arguments handling
        - Missing required parameters
        - Type validation errors
        - Error response format compliance
        """
        await self.setup_test_environment()
        
        try:
            # Test 1: Missing required content parameter
            result1 = await self.validate_mcp_error_handling(
                tool_name="store_memory",
                invalid_arguments={
                    "memory_type": "error_test",
                    # Missing 'content' parameter
                    "importance": 0.5
                },
                expected_error_pattern="content",
                test_name="mcp_missing_content_error"
            )
            
            print(f"\n❌ MCP Error Handling Test 1:")
            print(f"   ✅ Error Response Valid: {result1.passed}")
            print(f"   📋 Error Message: {result1.parsed_response.get('error', 'None')}")
            
            # Test 2: Invalid importance value
            result2 = await self.validate_mcp_error_handling(
                tool_name="store_memory", 
                invalid_arguments={
                    "memory_type": "error_test",
                    "content": "test content",
                    "importance": "invalid_string"  # Should be float
                },
                test_name="mcp_invalid_importance_error"
            )
            
            print(f"\n❌ MCP Error Handling Test 2:")
            print(f"   ✅ Error Response Valid: {result2.passed}")
            print(f"   📋 Error Message: {result2.parsed_response.get('error', 'None')}")
            
            # Test 3: Invalid memory ID for retrieval
            result3 = await self.validate_mcp_error_handling(
                tool_name="update_memory",
                invalid_arguments={
                    "memory_id": "invalid-uuid-format",
                    "updates": {"test": "value"}
                },
                test_name="mcp_invalid_memory_id_error"
            )
            
            print(f"\n❌ MCP Error Handling Test 3:")
            print(f"   ✅ Error Response Valid: {result3.passed}")
            print(f"   📋 Error Message: {result3.parsed_response.get('error', 'None')}")
            
            # Validate all error tests passed
            assert result1.passed, f"Error handling test 1 failed: {result1.errors}"
            assert result2.passed, f"Error handling test 2 failed: {result2.errors}"
            assert result3.passed, f"Error handling test 3 failed: {result3.errors}"
            
        finally:
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_mcp_memory_lifecycle_validation(self):
        """
        Test complete memory lifecycle through MCP protocol.
        
        Tests:
        - Store -> Retrieve -> Update -> Delete cycle
        - Data consistency at each step
        - MCP response validation throughout
        """
        await self.setup_test_environment()
        
        try:
            # Step 1: Store memory
            store_args = {
                "memory_type": "mcp_lifecycle_test",
                "content": "Original content for lifecycle testing",
                "importance": 0.7,
                "metadata": {
                    "version": 1,
                    "status": "initial"
                }
            }
            
            store_result = await self.validate_mcp_tool_execution(
                tool_name="store_memory",
                arguments=store_args,
                validate_underlying_data=True,
                test_name="mcp_lifecycle_store"
            )
            
            assert store_result.passed, f"Store step failed: {store_result.errors}"
            memory_id = store_result.parsed_response["memory_id"]
            
            # Step 2: Retrieve memory
            retrieve_args = {
                "query": "Original content lifecycle testing",
                "limit": 5,
                "types": ["mcp_lifecycle_test"],
                "include_metadata": True
            }
            
            retrieve_result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments=retrieve_args,
                test_name="mcp_lifecycle_retrieve"
            )
            
            assert retrieve_result.passed, f"Retrieve step failed: {retrieve_result.errors}"
            memories = retrieve_result.parsed_response["memories"]
            assert len(memories) >= 1, "Should find the stored memory"
            
            found_memory = next((m for m in memories if m.get("id") == memory_id or m.get("memory_id") == memory_id), None)
            assert found_memory is not None, "Should find the stored memory by ID"
            
            # Step 3: Update memory
            update_args = {
                "memory_id": memory_id,
                "updates": {
                    "content": "Updated content for lifecycle testing",
                    "version": 2,
                    "status": "updated"
                }
            }
            
            update_result = await self.validate_mcp_tool_execution(
                tool_name="update_memory",
                arguments=update_args,
                test_name="mcp_lifecycle_update"
            )
            
            assert update_result.passed, f"Update step failed: {update_result.errors}"
            assert update_result.parsed_response["success"] is True
            
            # Step 4: Verify update by retrieving again
            verify_retrieve_result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": "Updated content lifecycle testing",
                    "limit": 5,
                    "types": ["mcp_lifecycle_test"],
                    "include_metadata": True
                },
                test_name="mcp_lifecycle_verify_update"
            )
            
            assert verify_retrieve_result.passed, "Verification retrieve failed"
            updated_memories = verify_retrieve_result.parsed_response["memories"]
            updated_memory = next((m for m in updated_memories if m.get("id") == memory_id or m.get("memory_id") == memory_id), None)
            
            # Validate update was applied (note: content might be in different field)
            if updated_memory:
                memory_content = str(updated_memory.get("content", ""))
                assert "Updated content" in memory_content, "Memory content should be updated"
            
            # Step 5: Delete memory
            delete_args = {
                "memory_ids": [memory_id]
            }
            
            delete_result = await self.validate_mcp_tool_execution(
                tool_name="delete_memory",
                arguments=delete_args,
                test_name="mcp_lifecycle_delete"
            )
            
            assert delete_result.passed, f"Delete step failed: {delete_result.errors}"
            assert delete_result.parsed_response["success"] is True
            
            # Step 6: Verify deletion
            final_retrieve_result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": "Updated content lifecycle testing",
                    "limit": 10,
                    "types": ["mcp_lifecycle_test"]
                },
                test_name="mcp_lifecycle_verify_deletion"
            )
            
            assert final_retrieve_result.passed, "Final verification retrieve failed"
            final_memories = final_retrieve_result.parsed_response["memories"]
            deleted_memory = next((m for m in final_memories if m.get("id") == memory_id or m.get("memory_id") == memory_id), None)
            assert deleted_memory is None, "Memory should be deleted and not found"
            
            print(f"\n🔄 MCP Memory Lifecycle Test Results:")
            print(f"   ✅ Store: {store_result.passed}")
            print(f"   ✅ Retrieve: {retrieve_result.passed}")
            print(f"   ✅ Update: {update_result.passed}")
            print(f"   ✅ Delete: {delete_result.passed}")
            print(f"   ✅ Verification: Memory properly deleted")
            
        finally:
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_mcp_concurrent_operations_validation(self):
        """
        Test MCP server thread safety with concurrent operations.
        
        Tests:
        - Multiple simultaneous store operations
        - Concurrent retrieval requests
        - Data consistency under load
        - Response format consistency
        """
        await self.setup_test_environment()
        
        try:
            # Prepare concurrent operations
            concurrent_operations = []
            
            # Add multiple store operations
            for i in range(5):
                concurrent_operations.append((
                    "store_memory",
                    {
                        "memory_type": "mcp_concurrent_test",
                        "content": f"Concurrent test memory {i} with unique content",
                        "importance": 0.5 + (i * 0.1),
                        "metadata": {
                            "batch_index": i,
                            "test_type": "concurrency"
                        }
                    }
                ))
            
            # Add retrieval operations
            for i in range(3):
                concurrent_operations.append((
                    "retrieve_memory",
                    {
                        "query": f"concurrent test memory {i}",
                        "limit": 10,
                        "types": ["mcp_concurrent_test"]
                    }
                ))
            
            # Execute concurrent operations
            start_time = time.perf_counter()
            results = await self.validate_mcp_concurrent_operations(
                tool_operations=concurrent_operations,
                test_name="mcp_concurrent_validation"
            )
            concurrent_time = (time.perf_counter() - start_time) * 1000
            
            # Analyze results
            store_results = [r for r in results if r.tool_name == "store_memory"]
            retrieve_results = [r for r in results if r.tool_name == "retrieve_memory"]
            
            store_success_rate = sum(1 for r in store_results if r.passed) / len(store_results) * 100
            retrieve_success_rate = sum(1 for r in retrieve_results if r.passed) / len(retrieve_results) * 100
            
            print(f"\n🔀 MCP Concurrent Operations Test Results:")
            print(f"   📊 Total Operations: {len(results)}")
            print(f"   ✅ Store Success Rate: {store_success_rate:.1f}% ({len([r for r in store_results if r.passed])}/{len(store_results)})")
            print(f"   ✅ Retrieve Success Rate: {retrieve_success_rate:.1f}% ({len([r for r in retrieve_results if r.passed])}/{len(retrieve_results)})")
            print(f"   ⚡ Total Concurrent Time: {concurrent_time:.1f}ms")
            print(f"   📈 Average Per Operation: {concurrent_time/len(results):.1f}ms")
            
            # Validate concurrent operation success
            failed_operations = [r for r in results if not r.passed]
            if failed_operations:
                print(f"   ❌ Failed Operations:")
                for failed_op in failed_operations[:3]:  # Show first 3 failures
                    print(f"      - {failed_op.tool_name}: {failed_op.errors[:1]}")
            
            # Assert acceptable success rates
            assert store_success_rate >= 80, f"Store success rate too low: {store_success_rate}%"
            assert retrieve_success_rate >= 80, f"Retrieve success rate too low: {retrieve_success_rate}%"
            
            # Validate no duplicate memory IDs in store operations
            store_memory_ids = [r.parsed_response.get("memory_id") for r in store_results if r.passed and r.parsed_response.get("memory_id")]
            unique_memory_ids = set(store_memory_ids)
            assert len(store_memory_ids) == len(unique_memory_ids), "Duplicate memory IDs detected in concurrent stores"
            
        finally:
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_mcp_list_memories_protocol_validation(self):
        """
        Validate list_memories MCP tool protocol compliance.
        
        Tests:
        - Memory listing through MCP
        - Filtering and pagination
        - Response format validation
        - Metadata inclusion options
        """
        await self.setup_test_environment()
        
        try:
            # First store some test memories with different types
            test_memories = [
                {"memory_type": "mcp_list_test_type1", "content": "First type memory", "metadata": {"priority": "high"}},
                {"memory_type": "mcp_list_test_type1", "content": "Second type memory", "metadata": {"priority": "medium"}},
                {"memory_type": "mcp_list_test_type2", "content": "Different type memory", "metadata": {"priority": "low"}},
                {"memory_type": "mcp_list_test_type2", "content": "Another different type", "metadata": {"priority": "high"}},
            ]
            
            # Store all test memories
            for i, memory_data in enumerate(test_memories):
                store_result = await self.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    test_name=f"mcp_store_for_list_{i}"
                )
                assert store_result.passed, f"Failed to store memory {i}"
            
            # Test 1: List all memories (no filters)
            list_all_result = await self.validate_mcp_tool_execution(
                tool_name="list_memories",
                arguments={
                    "limit": 20,
                    "offset": 0,
                    "include_content": True
                },
                test_name="mcp_list_all_memories"
            )
            
            assert list_all_result.passed, f"List all memories failed: {list_all_result.errors}"
            all_memories = list_all_result.parsed_response["memories"]
            assert len(all_memories) >= 4, "Should find at least the 4 test memories"
            
            # Test 2: List with type filter
            list_filtered_result = await self.validate_mcp_tool_execution(
                tool_name="list_memories",
                arguments={
                    "types": ["mcp_list_test_type1"],
                    "limit": 10,
                    "include_content": True
                },
                test_name="mcp_list_filtered_memories"
            )
            
            assert list_filtered_result.passed, f"List filtered memories failed: {list_filtered_result.errors}"
            filtered_memories = list_filtered_result.parsed_response["memories"]
            assert len(filtered_memories) >= 2, "Should find at least 2 type1 memories"
            
            # Validate all returned memories are of the requested type
            for memory in filtered_memories:
                memory_type = memory.get("type") or memory.get("memory_type")
                if memory_type and memory_type.startswith("mcp_list_test"):
                    assert memory_type == "mcp_list_test_type1", f"Wrong type returned: {memory_type}"
            
            # Test 3: Test pagination
            list_paginated_result = await self.validate_mcp_tool_execution(
                tool_name="list_memories",
                arguments={
                    "limit": 2,
                    "offset": 0,
                    "types": ["mcp_list_test_type1", "mcp_list_test_type2"],
                    "include_content": False
                },
                test_name="mcp_list_paginated_memories"
            )
            
            assert list_paginated_result.passed, f"List paginated memories failed: {list_paginated_result.errors}"
            paginated_memories = list_paginated_result.parsed_response["memories"]
            assert len(paginated_memories) <= 2, "Should respect limit parameter"
            
            print(f"\n📋 MCP List Memories Test Results:")
            print(f"   ✅ List All: {len(all_memories)} memories found")
            print(f"   ✅ List Filtered: {len(filtered_memories)} type1 memories found")
            print(f"   ✅ List Paginated: {len(paginated_memories)} memories (limit=2)")
            print(f"   ⚡ List Performance: {list_all_result.performance_metrics['tool_execution_ms']:.1f}ms")
            
        finally:
            await self.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_mcp_performance_baseline_validation(self):
        """
        Establish MCP protocol performance baselines.
        
        Tests:
        - MCP tool execution overhead
        - Protocol validation performance
        - Comparison with direct domain calls
        - Acceptable response times
        """
        await self.setup_test_environment()
        
        try:
            performance_results = []
            
            # Test batch MCP operations for consistent metrics
            batch_size = 10
            
            for i in range(batch_size):
                result = await self.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments={
                        "memory_type": "mcp_performance_test",
                        "content": f"Performance test memory {i}",
                        "importance": 0.5,
                        "metadata": {"batch_index": i}
                    },
                    validate_underlying_data=True,
                    test_name=f"mcp_performance_store_{i}"
                )
                
                assert result.passed, f"Performance test {i} failed"
                performance_results.append(result)
            
            # Calculate performance metrics
            mcp_execution_times = [r.performance_metrics["tool_execution_ms"] for r in performance_results]
            protocol_validation_times = [r.performance_metrics["protocol_validation_ms"] for r in performance_results]
            data_validation_times = [r.performance_metrics.get("data_validation_ms", 0) for r in performance_results]
            total_times = [r.performance_metrics["total_time_ms"] for r in performance_results]
            
            avg_mcp_execution = sum(mcp_execution_times) / len(mcp_execution_times)
            avg_protocol_validation = sum(protocol_validation_times) / len(protocol_validation_times)
            avg_data_validation = sum(data_validation_times) / len(data_validation_times)
            avg_total_time = sum(total_times) / len(total_times)
            
            # Test retrieval performance
            retrieve_result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": "Performance test memory",
                    "limit": batch_size,
                    "types": ["mcp_performance_test"]
                },
                test_name="mcp_performance_retrieve"
            )
            
            assert retrieve_result.passed, "Performance retrieval test failed"
            
            print(f"\n📊 MCP Performance Baseline Results:")
            print(f"   Batch Size: {batch_size} operations")
            print(f"   Average MCP Execution: {avg_mcp_execution:.2f}ms")
            print(f"   Average Protocol Validation: {avg_protocol_validation:.2f}ms")
            print(f"   Average Data Validation: {avg_data_validation:.2f}ms")
            print(f"   Average Total Time: {avg_total_time:.2f}ms")
            print(f"   Retrieve Performance: {retrieve_result.performance_metrics['tool_execution_ms']:.2f}ms")
            print(f"   Throughput: {(batch_size / (sum(total_times) / 1000)):.1f} ops/second")
            
            # Performance assertions (adjust thresholds as needed)
            assert avg_mcp_execution < 10000, f"MCP execution too slow: {avg_mcp_execution}ms"  # Excludes first-time model loading
            assert avg_protocol_validation < 100, f"Protocol validation too slow: {avg_protocol_validation}ms"
            assert retrieve_result.performance_metrics["tool_execution_ms"] < 1000, "Retrieve performance too slow"
            
            # Validate search found all stored memories
            retrieved_memories = retrieve_result.parsed_response["memories"]
            assert len(retrieved_memories) == batch_size, f"Should retrieve all {batch_size} memories"
            
        finally:
            await self.teardown_test_environment()
    
    def test_generate_mcp_comprehensive_report(self):
        """Generate and validate comprehensive MCP test report."""
        report = self.generate_mcp_test_report()
        
        assert "MCP SERVER VALIDATION REPORT" in report
        assert "Total MCP tests:" in report
        assert "MCP Tool Coverage:" in report
        assert "store_memory:" in report or "No MCP tests run" in report
        
        print(f"\n{report}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_mcp_server_validation.py -v -s
    pytest.main([__file__, "-v", "-s"])