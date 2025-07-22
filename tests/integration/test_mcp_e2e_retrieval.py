"""
End-to-End MCP Retrieval Test Suite.

This test suite provides comprehensive end-to-end testing of the MCP retrieval
system, including performance benchmarks and stress testing to ensure the
system remains stable under various loads.
"""

import pytest
import asyncio
import time
import random
from typing import List, Dict, Any
from tests.framework.mcp_validation import MCPServerTestSuite
from tests.integration.test_mcp_large_dataset_validation import LargeDatasetGenerator


class TestMCPE2ERetrieval:
    """End-to-end tests for MCP retrieval functionality."""

    @pytest.mark.asyncio
    async def test_retrieval_performance_benchmark(self):
        """
        Benchmark retrieval performance across different dataset sizes.
        
        This test ensures that retrieval performance remains acceptable
        as the dataset grows and validates response format consistency.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Test with progressively larger datasets
            dataset_sizes = [25, 50, 100, 200]
            performance_results = []
            
            generator = LargeDatasetGenerator()
            
            for dataset_size in dataset_sizes:
                print(f"\n📊 Testing retrieval performance with {dataset_size} memories...")
                
                # Generate and store dataset
                dataset = generator.generate_large_dataset(dataset_size)
                stored_ids = []
                
                # Store memories (timing not critical here)
                for i, memory_data in enumerate(dataset):
                    result = await suite.validate_mcp_tool_execution(
                        tool_name="store_memory",
                        arguments=memory_data,
                        validate_underlying_data=False,
                        test_name=f"perf_store_{dataset_size}_{i}"
                    )
                    
                    assert result.passed, f"Failed to store memory {i} for dataset size {dataset_size}"
                    stored_ids.append(result.parsed_response["memory_id"])
                
                # Test retrieval performance with various queries
                retrieval_queries = [
                    {"query": "programming knowledge", "limit": 5, "min_similarity": 0.4},
                    {"query": "technical documentation", "limit": 10, "min_similarity": 0.3},
                    {"query": "project management", "limit": 8, "min_similarity": 0.35},
                    {"query": "code review process", "limit": 6, "min_similarity": 0.45},
                    {"query": "database optimization", "limit": 4, "min_similarity": 0.5}
                ]
                
                retrieval_times = []
                total_retrieved = 0
                
                for query_data in retrieval_queries:
                    start_time = time.perf_counter()
                    
                    result = await suite.validate_mcp_tool_execution(
                        tool_name="retrieve_memory",
                        arguments=query_data,
                        test_name=f"perf_retrieve_{dataset_size}_{query_data['query'].replace(' ', '_')}"
                    )
                    
                    retrieval_time = (time.perf_counter() - start_time) * 1000
                    retrieval_times.append(retrieval_time)
                    
                    assert result.passed, f"Retrieval failed for dataset size {dataset_size}: {result.errors}"
                    
                    memories = result.parsed_response.get("memories", [])
                    total_retrieved += len(memories)
                    
                    # Validate response format consistency
                    assert result.response_validation["valid_json"], "Response must be valid JSON"
                    assert result.response_validation["schema_compliant"], "Response must be schema compliant"
                
                # Calculate performance metrics
                avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
                max_retrieval_time = max(retrieval_times)
                
                performance_results.append({
                    "dataset_size": dataset_size,
                    "avg_retrieval_time_ms": avg_retrieval_time,
                    "max_retrieval_time_ms": max_retrieval_time,
                    "total_retrieved": total_retrieved,
                    "queries_tested": len(retrieval_queries)
                })
                
                print(f"   ⚡ Avg retrieval time: {avg_retrieval_time:.1f}ms")
                print(f"   📈 Max retrieval time: {max_retrieval_time:.1f}ms") 
                print(f"   📊 Total memories retrieved: {total_retrieved}")
                
                # Performance assertions
                assert avg_retrieval_time < 1000, f"Average retrieval too slow: {avg_retrieval_time:.1f}ms"
                assert max_retrieval_time < 2000, f"Max retrieval too slow: {max_retrieval_time:.1f}ms"
                
                # Clean up for next iteration
                await suite.teardown_test_environment()
                await suite.setup_test_environment()
            
            # Validate performance doesn't degrade significantly with size
            print(f"\n📈 Performance Summary:")
            for result in performance_results:
                print(f"   Dataset {result['dataset_size']}: {result['avg_retrieval_time_ms']:.1f}ms avg")
            
            # Ensure performance scaling is reasonable (shouldn't increase dramatically)
            if len(performance_results) >= 2:
                first_avg = performance_results[0]["avg_retrieval_time_ms"]
                last_avg = performance_results[-1]["avg_retrieval_time_ms"]
                scaling_factor = last_avg / first_avg
                
                assert scaling_factor < 5.0, f"Performance degradation too high: {scaling_factor:.1f}x slower"
                
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_concurrent_retrieval_operations(self):
        """
        Test concurrent retrieval operations for thread safety.
        
        Validates that multiple simultaneous retrieval operations
        don't interfere with each other and maintain data integrity.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create a diverse dataset for concurrent testing
            generator = LargeDatasetGenerator()
            dataset = generator.generate_large_dataset(75)
            
            # Store dataset
            stored_ids = []
            for i, memory_data in enumerate(dataset):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"concurrent_store_{i}"
                )
                
                assert result.passed, f"Failed to store memory {i}"
                stored_ids.append(result.parsed_response["memory_id"])
            
            # Define concurrent operations
            concurrent_operations = []
            for i in range(10):
                query_terms = [
                    "programming", "documentation", "project", "technical", "code",
                    "review", "meeting", "planning", "architecture", "database"
                ]
                
                operation = ("retrieve_memory", {
                    "query": f"{random.choice(query_terms)} {random.choice(query_terms)}",
                    "limit": random.randint(3, 8),
                    "min_similarity": random.uniform(0.3, 0.6),
                    "include_metadata": random.choice([True, False])
                })
                concurrent_operations.append(operation)
            
            # Execute concurrent operations
            start_time = time.perf_counter()
            results = await suite.validate_mcp_concurrent_operations(
                tool_operations=concurrent_operations,
                test_name="concurrent_retrieval_stress"
            )
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Validate all operations succeeded
            passed_operations = sum(1 for result in results if result.passed)
            assert passed_operations == len(concurrent_operations), f"Only {passed_operations}/{len(concurrent_operations)} concurrent operations passed"
            
            # Validate response formats for all operations
            total_memories_retrieved = 0
            for result in results:
                assert result.response_validation.get("valid_json", False), "All concurrent responses must be valid JSON"
                
                memories = result.parsed_response.get("memories", [])
                total_memories_retrieved += len(memories)
                
                # Validate memory structure
                for memory in memories:
                    assert isinstance(memory, dict), "Memory must be dictionary"
                    assert "id" in memory, "Memory must have 'id'"
                    assert "content" in memory, "Memory must have 'content'"
            
            print(f"\n⚡ Concurrent Operations Summary:")
            print(f"   Operations: {len(concurrent_operations)}")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Avg time per operation: {total_time/len(concurrent_operations):.1f}ms")
            print(f"   Total memories retrieved: {total_memories_retrieved}")
            
            # Performance assertions for concurrent operations
            avg_time_per_op = total_time / len(concurrent_operations)
            assert avg_time_per_op < 500, f"Concurrent operations too slow: {avg_time_per_op:.1f}ms per operation"
            
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_retrieval_accuracy_and_relevance(self):
        """
        Test retrieval accuracy and relevance scoring.
        
        Validates that retrieved memories are relevant to queries
        and that similarity/relevance scoring works correctly.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create targeted test memories with known content
            targeted_memories = [
                {
                    "memory_type": "programming_knowledge",
                    "content": "Python asyncio and coroutines for concurrent programming patterns",
                    "importance": 0.9,
                    "metadata": {"language": "python", "topic": "async"}
                },
                {
                    "memory_type": "programming_knowledge", 
                    "content": "JavaScript promises and async/await syntax for asynchronous operations",
                    "importance": 0.8,
                    "metadata": {"language": "javascript", "topic": "async"}
                },
                {
                    "memory_type": "technical_documentation",
                    "content": "Database connection pooling and optimization techniques for high-performance applications",
                    "importance": 0.85,
                    "metadata": {"topic": "database", "type": "optimization"}
                },
                {
                    "memory_type": "code_review",
                    "content": "Code review best practices including security checks and performance analysis",
                    "importance": 0.75,
                    "metadata": {"process": "review", "focus": "quality"}
                },
                {
                    "memory_type": "project_summary",
                    "content": "Machine learning model deployment using Docker containers and Kubernetes orchestration",
                    "importance": 0.9,
                    "metadata": {"topic": "ml", "deployment": "kubernetes"}
                }
            ]
            
            # Store targeted memories
            stored_ids = []
            for i, memory_data in enumerate(targeted_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"accuracy_store_{i}"
                )
                
                assert result.passed, f"Failed to store targeted memory {i}"
                stored_ids.append(result.parsed_response["memory_id"])
            
            # Test specific queries with expected relevance
            accuracy_tests = [
                {
                    "query": "python asyncio coroutines programming",
                    "expected_content_keywords": ["python", "asyncio", "coroutines"],
                    "min_similarity": 0.5,
                    "limit": 3
                },
                {
                    "query": "database optimization connection pooling",
                    "expected_content_keywords": ["database", "pooling", "optimization"],
                    "min_similarity": 0.4,
                    "limit": 2
                },
                {
                    "query": "code review best practices security",
                    "expected_content_keywords": ["code", "review", "practices"],
                    "min_similarity": 0.3,
                    "limit": 2
                },
                {
                    "query": "machine learning docker kubernetes deployment",
                    "expected_content_keywords": ["machine", "docker", "kubernetes"],
                    "min_similarity": 0.4,
                    "limit": 2
                }
            ]
            
            for i, accuracy_test in enumerate(accuracy_tests):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments={
                        "query": accuracy_test["query"],
                        "limit": accuracy_test["limit"],
                        "min_similarity": accuracy_test["min_similarity"],
                        "include_metadata": True
                    },
                    test_name=f"accuracy_retrieve_{i}"
                )
                
                assert result.passed, f"Accuracy test {i} failed: {result.errors}"
                
                memories = result.parsed_response.get("memories", [])
                assert len(memories) > 0, f"Query '{accuracy_test['query']}' should return at least one memory"
                
                # Validate relevance - at least one result should contain expected keywords
                found_relevant = False
                for memory in memories:
                    content = memory.get("content", "").lower()
                    keywords_found = sum(1 for keyword in accuracy_test["expected_content_keywords"] 
                                       if keyword.lower() in content)
                    
                    if keywords_found >= 2:  # At least 2 keywords should match
                        found_relevant = True
                        break
                
                assert found_relevant, f"No relevant memories found for query '{accuracy_test['query']}'"
                
                # Validate similarity scores are reasonable and properly ordered
                similarities = []
                for memory in memories:
                    similarity = memory.get("similarity_score") or memory.get("similarity", 0)
                    similarities.append(similarity)
                    assert similarity >= accuracy_test["min_similarity"], f"Similarity {similarity} below threshold"
                
                # Should be ordered by relevance (highest first)
                assert similarities == sorted(similarities, reverse=True), "Results should be ordered by similarity"
            
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_retrieval_edge_cases(self):
        """
        Test retrieval behavior with edge cases and boundary conditions.
        
        Validates system stability with unusual inputs and edge cases.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Store some baseline memories
            baseline_memories = [
                {
                    "memory_type": "test_memory",
                    "content": "Normal test content for baseline testing",
                    "importance": 0.5,
                    "metadata": {"type": "baseline"}
                },
                {
                    "memory_type": "test_memory", 
                    "content": "Another normal test memory with different content",
                    "importance": 0.6,
                    "metadata": {"type": "baseline"}
                }
            ]
            
            for i, memory_data in enumerate(baseline_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"edge_store_{i}"
                )
                assert result.passed, f"Failed to store baseline memory {i}"
            
            # Test edge case queries
            edge_case_tests = [
                {
                    "name": "empty_query",
                    "query": "",
                    "limit": 3,
                    "min_similarity": 0.1
                },
                {
                    "name": "very_long_query",
                    "query": " ".join(["test"] * 100),  # Very long query
                    "limit": 2,
                    "min_similarity": 0.3
                },
                {
                    "name": "special_characters",
                    "query": "test!@#$%^&*()_+-=[]{}|;':\",./<>?",
                    "limit": 2,
                    "min_similarity": 0.2
                },
                {
                    "name": "unicode_query",
                    "query": "测试 тест परीक्षण テスト",
                    "limit": 2,
                    "min_similarity": 0.2
                },
                {
                    "name": "high_limit",
                    "query": "test content",
                    "limit": 100,  # Higher than available memories
                    "min_similarity": 0.1
                },
                {
                    "name": "very_high_similarity",
                    "query": "test content",
                    "limit": 5,
                    "min_similarity": 0.99  # Very high similarity threshold
                }
            ]
            
            for edge_test in edge_case_tests:
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments={
                        "query": edge_test["query"],
                        "limit": edge_test["limit"],
                        "min_similarity": edge_test["min_similarity"],
                        "include_metadata": True
                    },
                    test_name=f"edge_case_{edge_test['name']}"
                )
                
                # For edge cases, we mainly want to ensure no crashes and proper response format
                # The actual results may vary, but the system should handle them gracefully
                assert result.passed, f"Edge case '{edge_test['name']}' failed: {result.errors}"
                assert result.response_validation["valid_json"], f"Edge case '{edge_test['name']}' returned invalid JSON"
                
                memories = result.parsed_response.get("memories", [])
                assert isinstance(memories, list), f"Edge case '{edge_test['name']}' should return a list"
                
                # If memories are returned, validate their structure
                for memory in memories:
                    assert isinstance(memory, dict), "Returned memory must be dictionary"
                    assert "id" in memory, "Returned memory must have 'id'"
                    assert "content" in memory, "Returned memory must have 'content'"
            
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_e2e_tests():
        """Run E2E retrieval tests directly."""
        test_suite = TestMCPE2ERetrieval()
        
        print("🧪 Running E2E MCP retrieval tests...")
        
        print("\n1️⃣ Testing retrieval performance benchmark...")
        await test_suite.test_retrieval_performance_benchmark()
        print("✅ Retrieval performance benchmark passed")
        
        print("\n2️⃣ Testing concurrent retrieval operations...")
        await test_suite.test_concurrent_retrieval_operations()
        print("✅ Concurrent retrieval operations passed")
        
        print("\n3️⃣ Testing retrieval accuracy and relevance...")
        await test_suite.test_retrieval_accuracy_and_relevance()
        print("✅ Retrieval accuracy and relevance passed")
        
        print("\n4️⃣ Testing retrieval edge cases...")
        await test_suite.test_retrieval_edge_cases()
        print("✅ Retrieval edge cases passed")
        
        print("\n🎉 All E2E MCP retrieval tests passed!")
    
    asyncio.run(run_e2e_tests())