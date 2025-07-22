"""
MCP Search Functionality Test Suite.

Comprehensive testing of search capabilities including semantic search,
filtering, ranking, and specialized search scenarios. This ensures the
search results maintain proper schema compliance and relevance.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any, Set
from tests.framework.mcp_validation import MCPServerTestSuite


class TestMCPSearchFunctionality:
    """Comprehensive tests for MCP search functionality."""

    @pytest.mark.asyncio
    async def test_semantic_search_quality(self):
        """
        Test semantic search quality and relevance.
        
        Validates that search returns semantically relevant results
        even when exact keyword matches aren't present.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create memories with semantic relationships
            semantic_memories = [
                {
                    "memory_type": "programming_knowledge",
                    "content": "Python functions and methods for data manipulation using pandas library",
                    "importance": 0.8,
                    "metadata": {"language": "python", "domain": "data_science"}
                },
                {
                    "memory_type": "programming_knowledge", 
                    "content": "Data analysis workflows with NumPy arrays and mathematical operations",
                    "importance": 0.85,
                    "metadata": {"language": "python", "domain": "data_science"}
                },
                {
                    "memory_type": "programming_knowledge",
                    "content": "Machine learning model training with scikit-learn algorithms",
                    "importance": 0.9,
                    "metadata": {"domain": "machine_learning", "library": "sklearn"}
                },
                {
                    "memory_type": "technical_documentation",
                    "content": "Database query optimization techniques for PostgreSQL performance tuning",
                    "importance": 0.75,
                    "metadata": {"database": "postgresql", "topic": "optimization"}
                },
                {
                    "memory_type": "project_summary",
                    "content": "Web application development using React components and state management",
                    "importance": 0.8,
                    "metadata": {"framework": "react", "domain": "web_development"}
                },
                {
                    "memory_type": "code_review",
                    "content": "Security vulnerability assessment and penetration testing methodologies",
                    "importance": 0.85,
                    "metadata": {"topic": "security", "type": "assessment"}
                }
            ]
            
            # Store memories
            stored_ids = []
            for i, memory_data in enumerate(semantic_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"semantic_store_{i}"
                )
                
                assert result.passed, f"Failed to store semantic memory {i}"
                stored_ids.append(result.parsed_response["memory_id"])
            
            # Test semantic search queries
            semantic_tests = [
                {
                    "query": "data processing and analysis",  # Should match pandas/numpy content
                    "expected_domains": {"data_science"},
                    "min_results": 2,
                    "limit": 4
                },
                {
                    "query": "artificial intelligence and ML algorithms",  # Should match ML content
                    "expected_domains": {"machine_learning"},
                    "min_results": 1,
                    "limit": 3
                },
                {
                    "query": "database performance and speed optimization",  # Should match DB content
                    "expected_topics": {"optimization"},
                    "min_results": 1,
                    "limit": 3
                },
                {
                    "query": "frontend user interface development",  # Should match React content
                    "expected_frameworks": {"react"},
                    "min_results": 1,
                    "limit": 3
                },
                {
                    "query": "cybersecurity threats and vulnerabilities",  # Should match security content
                    "expected_topics": {"security"},
                    "min_results": 1,
                    "limit": 3
                }
            ]
            
            for i, semantic_test in enumerate(semantic_tests):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments={
                        "query": semantic_test["query"],
                        "limit": semantic_test["limit"],
                        "min_similarity": 0.3,
                        "include_metadata": True
                    },
                    test_name=f"semantic_search_{i}"
                )
                
                assert result.passed, f"Semantic search {i} failed: {result.errors}"
                
                memories = result.parsed_response.get("memories", [])
                assert len(memories) >= semantic_test["min_results"], f"Semantic search '{semantic_test['query']}' returned too few results"
                
                # Validate semantic relevance through metadata
                found_relevant = False
                for memory in memories:
                    metadata = memory.get("metadata", {})
                    
                    # Check if memory matches expected domain/topic
                    if "expected_domains" in semantic_test:
                        if metadata.get("domain") in semantic_test["expected_domains"]:
                            found_relevant = True
                    elif "expected_topics" in semantic_test:
                        if metadata.get("topic") in semantic_test["expected_topics"]:
                            found_relevant = True
                    elif "expected_frameworks" in semantic_test:
                        if metadata.get("framework") in semantic_test["expected_frameworks"]:
                            found_relevant = True
                
                assert found_relevant, f"No semantically relevant results for query '{semantic_test['query']}'"
                
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_search_filtering_and_types(self):
        """
        Test search filtering by memory types and other criteria.
        
        Validates that type filtering works correctly and returns
        only memories of the specified types.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create memories of different types
            diverse_memories = [
                {
                    "memory_type": "programming_knowledge",
                    "content": "Python programming best practices and coding standards",
                    "importance": 0.8,
                    "metadata": {"language": "python"}
                },
                {
                    "memory_type": "programming_knowledge",
                    "content": "JavaScript async programming patterns and promises",
                    "importance": 0.75,
                    "metadata": {"language": "javascript"}
                },
                {
                    "memory_type": "technical_documentation",
                    "content": "API documentation for REST service endpoints",
                    "importance": 0.7,
                    "metadata": {"type": "api_docs"}
                },
                {
                    "memory_type": "technical_documentation", 
                    "content": "System architecture diagrams and deployment guides",
                    "importance": 0.85,
                    "metadata": {"type": "architecture"}
                },
                {
                    "memory_type": "meeting_notes",
                    "content": "Weekly team meeting discussing project milestones",
                    "importance": 0.6,
                    "metadata": {"meeting_type": "weekly"}
                },
                {
                    "memory_type": "meeting_notes",
                    "content": "Architecture review meeting with stakeholders",
                    "importance": 0.8,
                    "metadata": {"meeting_type": "architecture_review"}
                },
                {
                    "memory_type": "code_review",
                    "content": "Code review feedback for new feature implementation",
                    "importance": 0.7,
                    "metadata": {"feature": "new_functionality"}
                },
                {
                    "memory_type": "project_summary",
                    "content": "Project status report with timeline and deliverables",
                    "importance": 0.85,
                    "metadata": {"report_type": "status"}
                }
            ]
            
            # Store diverse memories
            stored_ids = []
            for i, memory_data in enumerate(diverse_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"filtering_store_{i}"
                )
                
                assert result.passed, f"Failed to store memory {i} for filtering test"
                stored_ids.append(result.parsed_response["memory_id"])
            
            # Test type filtering
            type_filter_tests = [
                {
                    "query": "programming patterns and practices",
                    "types": ["programming_knowledge"],
                    "expected_min_results": 2,
                    "expected_type": "programming_knowledge"
                },
                {
                    "query": "documentation and guides",
                    "types": ["technical_documentation"],
                    "expected_min_results": 2,
                    "expected_type": "technical_documentation"
                },
                {
                    "query": "meeting discussions", 
                    "types": ["meeting_notes"],
                    "expected_min_results": 2,
                    "expected_type": "meeting_notes"
                },
                {
                    "query": "project and code",
                    "types": ["code_review", "project_summary"],
                    "expected_min_results": 2,
                    "expected_types": {"code_review", "project_summary"}
                }
            ]
            
            for i, filter_test in enumerate(type_filter_tests):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments={
                        "query": filter_test["query"],
                        "types": filter_test["types"],
                        "limit": 5,
                        "min_similarity": 0.2,
                        "include_metadata": True
                    },
                    test_name=f"type_filtering_{i}"
                )
                
                assert result.passed, f"Type filtering test {i} failed: {result.errors}"
                
                memories = result.parsed_response.get("memories", [])
                assert len(memories) >= filter_test["expected_min_results"], f"Type filtering returned too few results for test {i}"
                
                # Validate all returned memories are of expected types
                for memory in memories:
                    memory_type = memory.get("memory_type") or memory.get("type")
                    
                    if "expected_type" in filter_test:
                        assert memory_type == filter_test["expected_type"], f"Unexpected memory type: {memory_type}"
                    elif "expected_types" in filter_test:
                        assert memory_type in filter_test["expected_types"], f"Unexpected memory type: {memory_type}"
                
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_search_similarity_thresholds(self):
        """
        Test search behavior with different similarity thresholds.
        
        Validates that similarity filtering works correctly and
        returns appropriately relevant results.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create memories with varying relevance to test queries
            threshold_memories = [
                {
                    "memory_type": "test_memory",
                    "content": "Python programming language features and syntax",  # High relevance to Python queries
                    "importance": 0.8,
                    "metadata": {"relevance": "high"}
                },
                {
                    "memory_type": "test_memory",
                    "content": "Programming concepts and software development practices",  # Medium relevance
                    "importance": 0.7,
                    "metadata": {"relevance": "medium"}
                },
                {
                    "memory_type": "test_memory",
                    "content": "Computer science fundamentals and algorithms",  # Lower relevance
                    "importance": 0.6,
                    "metadata": {"relevance": "lower"}
                },
                {
                    "memory_type": "test_memory",
                    "content": "Project management methodologies and team coordination",  # Low relevance
                    "importance": 0.5,
                    "metadata": {"relevance": "low"}
                }
            ]
            
            # Store memories
            for i, memory_data in enumerate(threshold_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"threshold_store_{i}"
                )
                assert result.passed, f"Failed to store threshold memory {i}"
            
            # Test with different similarity thresholds
            query = "Python programming language development"
            thresholds = [0.2, 0.4, 0.6, 0.8]
            previous_count = None
            
            for threshold in thresholds:
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments={
                        "query": query,
                        "limit": 10,
                        "min_similarity": threshold,
                        "include_metadata": True
                    },
                    test_name=f"similarity_threshold_{threshold}"
                )
                
                assert result.passed, f"Similarity threshold test failed for {threshold}"
                
                memories = result.parsed_response.get("memories", [])
                current_count = len(memories)
                
                print(f"   Threshold {threshold}: {current_count} results")
                
                # Validate similarity scores
                for memory in memories:
                    similarity = memory.get("similarity_score") or memory.get("similarity", 0)
                    assert similarity >= threshold, f"Memory similarity {similarity} below threshold {threshold}"
                
                # Higher thresholds should return same or fewer results
                if previous_count is not None:
                    assert current_count <= previous_count, f"Higher threshold {threshold} returned more results than lower threshold"
                
                previous_count = current_count
            
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_search_ranking_and_ordering(self):
        """
        Test search result ranking and ordering by relevance.
        
        Validates that results are properly ordered by similarity/relevance
        and that temporal adjustments work correctly.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create memories with different expected relevance scores
            ranking_memories = [
                {
                    "memory_type": "ranking_test",
                    "content": "Machine learning algorithms and artificial intelligence techniques",  # Should rank high for ML query
                    "importance": 0.9,
                    "metadata": {"expected_rank": "high"}
                },
                {
                    "memory_type": "ranking_test",
                    "content": "Data science methodologies and statistical analysis approaches",  # Medium-high rank
                    "importance": 0.8,
                    "metadata": {"expected_rank": "medium_high"}
                },
                {
                    "memory_type": "ranking_test",
                    "content": "Software development practices and programming techniques",  # Medium rank
                    "importance": 0.7,
                    "metadata": {"expected_rank": "medium"}
                },
                {
                    "memory_type": "ranking_test",
                    "content": "Project management and team collaboration strategies",  # Lower rank
                    "importance": 0.6,
                    "metadata": {"expected_rank": "low"}
                }
            ]
            
            # Store ranking test memories
            for i, memory_data in enumerate(ranking_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"ranking_store_{i}"
                )
                assert result.passed, f"Failed to store ranking memory {i}"
            
            # Test ranking with ML-related query
            result = await suite.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": "machine learning and data science algorithms",
                    "limit": 4,
                    "min_similarity": 0.2,
                    "include_metadata": True
                },
                test_name="ranking_test_query"
            )
            
            assert result.passed, f"Ranking test failed: {result.errors}"
            
            memories = result.parsed_response.get("memories", [])
            assert len(memories) >= 3, "Should return multiple memories for ranking test"
            
            # Validate similarity scores are in descending order
            similarities = []
            for memory in memories:
                similarity = memory.get("similarity_score") or memory.get("similarity", 0)
                similarities.append(similarity)
            
            assert similarities == sorted(similarities, reverse=True), f"Results not properly ordered by similarity: {similarities}"
            
            # Validate highest scoring memory is most relevant
            highest_similarity_memory = memories[0]
            expected_rank = highest_similarity_memory.get("metadata", {}).get("expected_rank", "")
            assert expected_rank in ["high", "medium_high"], f"Highest ranked memory should be most relevant, got: {expected_rank}"
            
        finally:
            await suite.teardown_test_environment()

    @pytest.mark.asyncio
    async def test_search_performance_with_large_results(self):
        """
        Test search performance when requesting large numbers of results.
        
        Validates that performance remains acceptable even when
        retrieving many results at once.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create a larger dataset for performance testing
            performance_memories = []
            base_contents = [
                "Programming and software development",
                "Data analysis and visualization", 
                "Machine learning and AI research",
                "Web development and frameworks",
                "Database design and optimization",
                "System architecture and scaling",
                "DevOps and deployment automation",
                "Security and vulnerability assessment"
            ]
            
            # Generate 40 memories with variations
            for i in range(40):
                base_content = base_contents[i % len(base_contents)]
                content = f"{base_content} - variant {i//len(base_contents) + 1} with specific details and examples"
                
                performance_memories.append({
                    "memory_type": "performance_test",
                    "content": content,
                    "importance": 0.5 + (i % 5) * 0.1,
                    "metadata": {"variant": i, "base_topic": base_content}
                })
            
            # Store performance test memories
            for i, memory_data in enumerate(performance_memories):
                result = await suite.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,
                    test_name=f"perf_store_{i}"
                )
                assert result.passed, f"Failed to store performance memory {i}"
            
            # Test performance with different result sizes
            result_sizes = [5, 10, 20, 30]
            
            for limit in result_sizes:
                start_time = time.perf_counter()
                
                result = await suite.validate_mcp_tool_execution(
                    tool_name="retrieve_memory",
                    arguments={
                        "query": "programming development software",
                        "limit": limit,
                        "min_similarity": 0.2,
                        "include_metadata": True
                    },
                    test_name=f"performance_limit_{limit}"
                )
                
                retrieval_time = (time.perf_counter() - start_time) * 1000
                
                assert result.passed, f"Performance test failed for limit {limit}"
                
                memories = result.parsed_response.get("memories", [])
                actual_count = len(memories)
                
                print(f"   Limit {limit}: {actual_count} results in {retrieval_time:.1f}ms")
                
                # Performance assertions
                assert retrieval_time < 1500, f"Retrieval too slow for {limit} results: {retrieval_time:.1f}ms"
                assert actual_count <= limit, f"Returned more results than requested: {actual_count} > {limit}"
                
                # Validate all results have proper format
                for memory in memories:
                    assert "id" in memory, "Memory missing 'id'"
                    assert "content" in memory, "Memory missing 'content'"
                    similarity = memory.get("similarity_score") or memory.get("similarity", 0)
                    assert similarity >= 0.2, f"Similarity below threshold: {similarity}"
                
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_search_tests():
        """Run search functionality tests directly."""
        test_suite = TestMCPSearchFunctionality()
        
        print("🧪 Running MCP search functionality tests...")
        
        print("\n1️⃣ Testing semantic search quality...")
        await test_suite.test_semantic_search_quality()
        print("✅ Semantic search quality passed")
        
        print("\n2️⃣ Testing search filtering and types...")
        await test_suite.test_search_filtering_and_types()
        print("✅ Search filtering and types passed")
        
        print("\n3️⃣ Testing search similarity thresholds...")
        await test_suite.test_search_similarity_thresholds()
        print("✅ Search similarity thresholds passed")
        
        print("\n4️⃣ Testing search ranking and ordering...")
        await test_suite.test_search_ranking_and_ordering()
        print("✅ Search ranking and ordering passed")
        
        print("\n5️⃣ Testing search performance with large results...")
        await test_suite.test_search_performance_with_large_results()
        print("✅ Search performance with large results passed")
        
        print("\n🎉 All MCP search functionality tests passed!")
    
    asyncio.run(run_search_tests())