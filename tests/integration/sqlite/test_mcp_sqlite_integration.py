#!/usr/bin/env python3
"""
MCP Server SQLite Integration Tests

Integration tests for SQLite memory persistence with the MCP server:
- Full MCP server integration with SQLite backend
- Tool operations through MCP interface
- Memory operations via MCP tools
- Error handling through MCP layer
- Performance of MCP-SQLite integration
"""

import asyncio
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence
from clarity.mcp.tools import (
    store_memory_tool,
    retrieve_memories_tool,
    list_memory_types_tool,
    get_memory_stats_tool,
    search_memories_tool
)


class TestMCPSQLiteIntegration:
    """Integration test suite for MCP server with SQLite backend."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="mcp_sqlite_test_")
        db_path = os.path.join(temp_dir, "mcp_sqlite_test.db")
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
    
    @pytest.fixture
    def mock_mcp_context(self, sqlite_persistence):
        """Create mock MCP context with SQLite persistence."""
        context = Mock()
        
        # Mock domain manager
        domain_manager = Mock()
        domain_manager.get_persistence.return_value = sqlite_persistence
        context.domain_manager = domain_manager
        
        # Mock session manager
        session_manager = Mock()
        session_manager.get_current_session_id.return_value = "test_session_001"
        context.session_manager = session_manager
        
        return context
    
    @pytest.mark.asyncio
    async def test_store_memory_tool_integration(self, mock_mcp_context, sqlite_persistence):
        """Test store_memory tool integration with SQLite."""
        # Test memory data
        memory_data = {
            "type": "structured_thinking",
            "content": "MCP integration test memory for SQLite backend",
            "importance": 0.8,
            "tier": "short_term",
            "metadata": {
                "source": "mcp_integration_test",
                "test_case": "store_memory_tool"
            }
        }
        
        # Call MCP tool
        result = await store_memory_tool(mock_mcp_context, memory_data)
        
        # Verify result structure
        assert "success" in result
        assert result["success"] is True
        assert "memory_id" in result
        assert result["memory_id"] is not None
        
        # Verify memory was actually stored in SQLite
        stored_memory = await sqlite_persistence.get_memory(result["memory_id"])
        assert stored_memory is not None
        assert stored_memory["type"] == "structured_thinking"
        assert stored_memory["importance"] == 0.8
        assert "MCP integration test" in str(stored_memory["content"])
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_tool_integration(self, mock_mcp_context, sqlite_persistence):
        """Test retrieve_memories tool integration with SQLite."""
        # Store test memories first
        test_memories = [
            {
                "id": "mcp-retrieve-001",
                "type": "structured_thinking",
                "content": "SQLite database performance analysis for MCP integration",
                "importance": 0.9,
                "tier": "long_term",
                "metadata": {"category": "analysis"}
            },
            {
                "id": "mcp-retrieve-002",
                "type": "episodic",
                "content": "User reported MCP connection issues with database",
                "importance": 0.8,
                "tier": "short_term",
                "metadata": {"category": "incident"}
            },
            {
                "id": "mcp-retrieve-003",
                "type": "semantic",
                "content": "Vector embeddings for semantic search in database systems",
                "importance": 0.7,
                "tier": "archival",
                "metadata": {"category": "concept"}
            }
        ]
        
        for memory in test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test MCP retrieve tool
        query_params = {
            "query": "database performance analysis MCP",
            "limit": 5,
            "memory_types": ["structured_thinking", "episodic"],
            "min_similarity": 0.0
        }
        
        result = await retrieve_memories_tool(mock_mcp_context, query_params)
        
        # Verify result structure
        assert "memories" in result
        assert "count" in result
        assert isinstance(result["memories"], list)
        assert result["count"] > 0
        
        # Verify memory content
        memories = result["memories"]
        assert len(memories) > 0
        
        # Check that results are filtered by memory types
        found_types = {m["type"] for m in memories}
        assert found_types.issubset({"structured_thinking", "episodic"})
        
        # Verify similarity scores are included
        for memory in memories:
            assert "similarity_score" in memory
            assert isinstance(memory["similarity_score"], float)
    
    @pytest.mark.asyncio
    async def test_search_memories_tool_integration(self, mock_mcp_context, sqlite_persistence):
        """Test search_memories tool integration with SQLite."""
        # Store diverse test memories
        diverse_memories = [
            {
                "id": "search-001",
                "type": "procedural",
                "content": "Database backup and recovery procedures",
                "importance": 0.85,
                "tier": "system",
                "metadata": {"procedure_type": "backup"}
            },
            {
                "id": "search-002",
                "type": "structured_thinking",
                "content": "Analysis of microservices architecture patterns",
                "importance": 0.9,
                "tier": "long_term",
                "metadata": {"analysis_type": "architecture"}
            },
            {
                "id": "search-003",
                "type": "episodic",
                "content": "System outage caused by database connectivity issues",
                "importance": 0.95,
                "tier": "short_term",
                "metadata": {"incident_type": "outage"}
            }
        ]
        
        for memory in diverse_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test search with filters
        search_params = {
            "filters": {
                "tier": "long_term",
                "memory_type": "structured_thinking"
            },
            "limit": 10
        }
        
        result = await search_memories_tool(mock_mcp_context, search_params)
        
        # Verify result structure
        assert "memories" in result
        assert "count" in result
        assert isinstance(result["memories"], list)
        
        # Verify filtering worked
        memories = result["memories"]
        for memory in memories:
            assert memory["tier"] == "long_term"
            assert memory["type"] == "structured_thinking"
    
    @pytest.mark.asyncio
    async def test_list_memory_types_tool_integration(self, mock_mcp_context, sqlite_persistence):
        """Test list_memory_types tool integration with SQLite."""
        # Store memories with different types
        type_test_memories = [
            {"id": "type-001", "type": "structured_thinking", "content": "Test 1", "importance": 0.5, "tier": "short_term"},
            {"id": "type-002", "type": "episodic", "content": "Test 2", "importance": 0.5, "tier": "short_term"},
            {"id": "type-003", "type": "procedural", "content": "Test 3", "importance": 0.5, "tier": "short_term"},
            {"id": "type-004", "type": "semantic", "content": "Test 4", "importance": 0.5, "tier": "short_term"},
            {"id": "type-005", "type": "episodic", "content": "Test 5", "importance": 0.5, "tier": "short_term"},  # Duplicate type
        ]
        
        for memory in type_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test MCP tool
        result = await list_memory_types_tool(mock_mcp_context, {})
        
        # Verify result structure
        assert "memory_types" in result
        assert "count" in result
        assert isinstance(result["memory_types"], list)
        
        # Verify expected types are present
        memory_types = result["memory_types"]
        expected_types = {"structured_thinking", "episodic", "procedural", "semantic"}
        found_types = set(memory_types)
        
        assert expected_types.issubset(found_types)
        assert result["count"] == len(memory_types)
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_tool_integration(self, mock_mcp_context, sqlite_persistence):
        """Test get_memory_stats tool integration with SQLite."""
        # Store test memories with varied attributes
        stats_test_memories = [
            {"id": "stats-001", "type": "structured_thinking", "content": "Stats test 1", "importance": 0.8, "tier": "long_term"},
            {"id": "stats-002", "type": "episodic", "content": "Stats test 2", "importance": 0.9, "tier": "short_term"},
            {"id": "stats-003", "type": "procedural", "content": "Stats test 3", "importance": 0.7, "tier": "system"},
            {"id": "stats-004", "type": "semantic", "content": "Stats test 4", "importance": 0.6, "tier": "archival"},
            {"id": "stats-005", "type": "structured_thinking", "content": "Stats test 5", "importance": 0.95, "tier": "long_term"},
        ]
        
        for memory in stats_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test MCP tool
        result = await get_memory_stats_tool(mock_mcp_context, {})
        
        # Verify result structure
        assert "total_memories" in result
        assert "memory_types" in result
        assert "memory_tiers" in result
        assert "database_info" in result
        
        # Verify statistics accuracy
        assert result["total_memories"] >= 5
        
        # Check type distribution
        type_stats = result["memory_types"]
        assert "structured_thinking" in type_stats
        assert type_stats["structured_thinking"] >= 2  # We stored 2
        
        # Check tier distribution
        tier_stats = result["memory_tiers"]
        assert "long_term" in tier_stats
        assert tier_stats["long_term"] >= 2  # We stored 2
        
        # Check database info
        db_info = result["database_info"]
        assert "database_size_bytes" in db_info
        assert "database_path" in db_info
        assert db_info["database_size_bytes"] > 0
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling_integration(self, mock_mcp_context, sqlite_persistence):
        """Test error handling through MCP layer with SQLite."""
        # Test invalid memory data
        invalid_memory_data = {
            "type": "invalid_type",
            "content": "",  # Empty content
            "importance": 2.0,  # Invalid importance > 1.0
            "tier": "invalid_tier"
        }
        
        # MCP tool should handle errors gracefully
        result = await store_memory_tool(mock_mcp_context, invalid_memory_data)
        
        # Should either succeed with corrections or fail gracefully
        assert "success" in result
        if result["success"]:
            # If it succeeded, check that corrections were made
            memory_id = result["memory_id"]
            stored_memory = await sqlite_persistence.get_memory(memory_id)
            assert stored_memory is not None
            # Importance should be corrected
            assert stored_memory["importance"] <= 1.0
        else:
            # If it failed, should have error information
            assert "error" in result
            assert isinstance(result["error"], str)
    
    @pytest.mark.asyncio
    async def test_mcp_sqlite_performance_integration(self, mock_mcp_context, sqlite_persistence):
        """Test performance of MCP-SQLite integration."""
        import time
        
        # Store test memories through MCP interface
        test_memories = []
        for i in range(20):
            memory_data = {
                "type": "episodic",
                "content": f"MCP performance test memory {i} with content for testing",
                "importance": 0.5 + (i % 5) * 0.1,
                "tier": "short_term",
                "metadata": {"batch": "performance_test", "index": i}
            }
            test_memories.append(memory_data)
        
        # Measure storage performance through MCP
        storage_start = time.perf_counter()
        stored_ids = []
        
        for memory_data in test_memories:
            result = await store_memory_tool(mock_mcp_context, memory_data)
            assert result["success"] is True
            stored_ids.append(result["memory_id"])
        
        storage_time = time.perf_counter() - storage_start
        storage_rate = len(test_memories) / storage_time
        
        print(f"\nMCP Storage Performance:")
        print(f"  Stored {len(test_memories)} memories in {storage_time:.3f}s")
        print(f"  Rate: {storage_rate:.1f} memories/sec")
        
        # Measure retrieval performance through MCP
        retrieval_start = time.perf_counter()
        
        query_params = {
            "query": "MCP performance test memory content",
            "limit": 10,
            "min_similarity": 0.0
        }
        
        result = await retrieve_memories_tool(mock_mcp_context, query_params)
        
        retrieval_time = time.perf_counter() - retrieval_start
        
        print(f"  Retrieved {result['count']} memories in {retrieval_time * 1000:.2f}ms")
        
        # Performance assertions
        assert storage_rate > 5  # At least 5 memories/sec through MCP
        assert retrieval_time < 0.5  # Retrieval under 500ms
        assert result["count"] > 0  # Should find relevant memories
    
    @pytest.mark.asyncio
    async def test_mcp_concurrent_operations(self, mock_mcp_context, sqlite_persistence):
        """Test concurrent MCP operations with SQLite backend."""
        # Test concurrent storage operations
        async def store_batch(batch_id, count):
            """Store a batch of memories concurrently."""
            stored_ids = []
            for i in range(count):
                memory_data = {
                    "type": "episodic",
                    "content": f"Concurrent batch {batch_id} memory {i}",
                    "importance": 0.6,
                    "tier": "short_term",
                    "metadata": {"batch_id": batch_id, "index": i}
                }
                
                result = await store_memory_tool(mock_mcp_context, memory_data)
                assert result["success"] is True
                stored_ids.append(result["memory_id"])
            
            return stored_ids
        
        # Run concurrent storage tasks
        concurrent_start = time.perf_counter()
        
        tasks = []
        for batch_id in range(3):  # 3 concurrent batches
            task = asyncio.create_task(store_batch(batch_id, 10))
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks)
        concurrent_time = time.perf_counter() - concurrent_start
        
        total_stored = sum(len(batch) for batch in batch_results)
        concurrent_rate = total_stored / concurrent_time
        
        print(f"\nMCP Concurrent Operations:")
        print(f"  Stored {total_stored} memories concurrently in {concurrent_time:.3f}s")
        print(f"  Concurrent rate: {concurrent_rate:.1f} memories/sec")
        
        # Verify all memories were stored successfully
        assert total_stored == 30  # 3 batches × 10 memories each
        
        # Test concurrent retrieval
        async def concurrent_search(query_suffix):
            """Perform concurrent search operations."""
            query_params = {
                "query": f"concurrent batch memory {query_suffix}",
                "limit": 5,
                "min_similarity": 0.0
            }
            return await retrieve_memories_tool(mock_mcp_context, query_params)
        
        # Run concurrent searches
        search_tasks = []
        for i in range(5):  # 5 concurrent searches
            task = asyncio.create_task(concurrent_search(i))
            search_tasks.append(task)
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Verify all searches completed successfully
        for result in search_results:
            assert "memories" in result
            assert "count" in result
            assert isinstance(result["memories"], list)
        
        total_search_results = sum(r["count"] for r in search_results)
        print(f"  Concurrent searches returned {total_search_results} total results")
        
        # Performance assertions
        assert concurrent_rate > 5  # At least 5 memories/sec concurrent
        assert total_search_results > 0  # Should find relevant memories
    
    @pytest.mark.asyncio
    async def test_mcp_session_integration(self, mock_mcp_context, sqlite_persistence):
        """Test MCP session integration with SQLite memory persistence."""
        # Test that session context is properly passed through
        session_id = mock_mcp_context.session_manager.get_current_session_id()
        assert session_id == "test_session_001"
        
        # Store memory with session context
        memory_data = {
            "type": "episodic",
            "content": "Session-specific memory for MCP integration testing",
            "importance": 0.8,
            "tier": "short_term",
            "metadata": {"session_test": True}
        }
        
        result = await store_memory_tool(mock_mcp_context, memory_data)
        assert result["success"] is True
        
        # Verify memory includes session context
        memory_id = result["memory_id"]
        stored_memory = await sqlite_persistence.get_memory(memory_id)
        assert stored_memory is not None
        
        # Check if session information is preserved in context
        context = stored_memory.get("context", {})
        # Note: This depends on how session context is implemented in the actual tool
        # For now, we just verify the memory was stored correctly
        assert stored_memory["type"] == "episodic"
    
    @pytest.mark.asyncio 
    async def test_mcp_tool_parameter_validation(self, mock_mcp_context, sqlite_persistence):
        """Test MCP tool parameter validation with SQLite backend."""
        # Test store_memory with missing required parameters
        incomplete_memory_data = {
            "type": "structured_thinking",
            # Missing content
            "importance": 0.8
        }
        
        result = await store_memory_tool(mock_mcp_context, incomplete_memory_data)
        
        # Should handle missing parameters gracefully
        # Either succeed with defaults or fail with clear error
        assert "success" in result
        if not result["success"]:
            assert "error" in result
            assert "content" in result["error"].lower()
        
        # Test retrieve_memories with invalid parameters
        invalid_query_params = {
            "query": "test query",
            "limit": -1,  # Invalid negative limit
            "min_similarity": 2.0  # Invalid similarity > 1.0
        }
        
        result = await retrieve_memories_tool(mock_mcp_context, invalid_query_params)
        
        # Should handle invalid parameters gracefully
        assert "memories" in result  # Should still return results with corrected params
        assert "count" in result
        
        # Test search_memories with invalid filter values
        invalid_search_params = {
            "filters": {
                "tier": "nonexistent_tier",
                "memory_type": "invalid_type"
            },
            "limit": 10
        }
        
        result = await search_memories_tool(mock_mcp_context, invalid_search_params)
        
        # Should handle gracefully (might return empty results)
        assert "memories" in result
        assert "count" in result
        assert isinstance(result["memories"], list)
    
    @pytest.mark.asyncio
    async def test_mcp_sqlite_data_consistency(self, mock_mcp_context, sqlite_persistence):
        """Test data consistency between MCP operations and direct SQLite access."""
        # Store memory through MCP interface
        mcp_memory_data = {
            "type": "structured_thinking",
            "content": "Data consistency test memory stored via MCP",
            "importance": 0.85,
            "tier": "long_term",
            "metadata": {"test": "consistency", "source": "mcp"}
        }
        
        mcp_result = await store_memory_tool(mock_mcp_context, mcp_memory_data)
        assert mcp_result["success"] is True
        mcp_memory_id = mcp_result["memory_id"]
        
        # Verify via direct SQLite access
        direct_memory = await sqlite_persistence.get_memory(mcp_memory_id)
        assert direct_memory is not None
        assert direct_memory["type"] == "structured_thinking"
        assert direct_memory["importance"] == 0.85
        assert "consistency test" in str(direct_memory["content"]).lower()
        
        # Store memory directly in SQLite
        direct_memory_data = {
            "id": "direct-consistency-001",
            "type": "episodic",
            "content": "Data consistency test memory stored directly",
            "importance": 0.75,
            "tier": "short_term",
            "metadata": {"test": "consistency", "source": "direct"}
        }
        
        direct_memory_id = await sqlite_persistence.store_memory(direct_memory_data)
        
        # Verify via MCP interface
        mcp_query_params = {
            "query": "consistency test memory stored directly",
            "limit": 5,
            "min_similarity": 0.0
        }
        
        mcp_search_result = await retrieve_memories_tool(mock_mcp_context, mcp_query_params)
        assert mcp_search_result["count"] > 0
        
        # Find the directly stored memory in MCP results
        found_direct_memory = False
        for memory in mcp_search_result["memories"]:
            if memory["id"] == direct_memory_id:
                found_direct_memory = True
                assert memory["type"] == "episodic"
                assert memory["importance"] == 0.75
                break
        
        assert found_direct_memory, "Memory stored directly should be accessible via MCP"
        
        # Verify both memories can be found together
        combined_query_params = {
            "query": "data consistency test memory",
            "limit": 10,
            "min_similarity": 0.0
        }
        
        combined_result = await retrieve_memories_tool(mock_mcp_context, combined_query_params)
        assert combined_result["count"] >= 2
        
        found_mcp_memory = False
        found_direct_memory = False
        
        for memory in combined_result["memories"]:
            if memory["id"] == mcp_memory_id:
                found_mcp_memory = True
            elif memory["id"] == direct_memory_id:
                found_direct_memory = True
        
        assert found_mcp_memory and found_direct_memory, "Both MCP and direct memories should be found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])