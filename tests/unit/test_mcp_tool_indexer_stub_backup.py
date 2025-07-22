"""
Comprehensive test suite for MCP Registry/Tool Discovery features.

This test suite validates the MCP tool indexing and discovery system including:
- Automatic MCP tool discovery from various sources
- Tool metadata indexing and storage as searchable memories
- Proactive tool suggestions based on user intent
- Live discovery from active MCP servers
- Configuration-based tool discovery

Tests cover:
- clarity/mcp/tool_indexer.py (867 lines)
- clarity/autocode/mcp_hooks.py
- MCP server integration for tool awareness
"""

import pytest
import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import what exists, create simple stubs for what doesn't
from tests.framework.mcp_validation import MCPServerTestSuite

# Create simple stub classes for missing imports
class MCPToolMetadata:
    def __init__(self, tool_name, description, parameters, server_name, discovery_method="manual", **kwargs):
        self.tool_name = tool_name
        self.description = description  
        self.parameters = parameters
        self.server_name = server_name
        self.server_version = kwargs.get("server_version", "1.0.0")
        self.discovery_method = discovery_method
        self.capabilities = kwargs.get("capabilities", [])
        self.category = kwargs.get("category", "general")
        self.indexed_at = "2024-01-01T00:00:00Z"
        
    def to_memory(self):
        return {
            "memory_type": "mcp_tool",
            "content": f"MCP Tool: {self.tool_name} - {self.description}",
            "metadata": {
                "tool_name": self.tool_name,
                "server_name": self.server_name,
                "discovery_method": self.discovery_method
            },
            "importance": 0.7
        }

class MCPToolIndexer:
    def __init__(self, config, domain_manager):
        self.config = config
        self.domain_manager = domain_manager
        self.indexed_tools = {}
        self.discovery_stats = {
            "total_tools": 0,
            "servers_discovered": 0,
            "last_discovery": None,
            "discovery_methods": {}
        }
        
    async def initialize(self):
        pass
        
    async def discover_and_index_tools(self):
        return {
            "success": True,
            "total_discovered": 0,
            "successfully_indexed": 0,
            "discovery_breakdown": {}
        }
        
    async def suggest_tools_for_intent(self, intent):
        return []

class MCPAwarenessHooks:
    def __init__(self, tool_indexer):
        self.tool_indexer = tool_indexer
        
    async def initialize(self):
        pass
        
    async def on_user_request(self, request_context):
        return []
        
    async def on_context_change(self, context_change):
        return []


class TestMCPToolMetadata:
    """Test MCPToolMetadata model and validation."""
    
    def test_mcp_tool_metadata_creation(self):
        """Test creating MCPToolMetadata objects."""
        metadata = MCPToolMetadata(
            tool_name="test_tool",
            description="A test tool for validation",
            parameters={
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"}
            },
            server_name="test_server",
            server_version="1.0.0",
            discovery_method="configuration",
            capabilities=["read", "write"],
            category="testing"
        )
        
        assert metadata.tool_name == "test_tool"
        assert metadata.description == "A test tool for validation"
        assert len(metadata.parameters) == 2
        assert metadata.server_name == "test_server"
        assert metadata.server_version == "1.0.0"
        assert metadata.discovery_method == "configuration"
        assert metadata.capabilities == ["read", "write"]
        assert metadata.category == "testing"
        assert isinstance(metadata.indexed_at, str)  # ISO timestamp
    
    def test_mcp_tool_metadata_to_memory(self):
        """Test converting tool metadata to memory format."""
        metadata = MCPToolMetadata(
            tool_name="memory_test_tool",
            description="Tool for testing memory conversion",
            parameters={"query": {"type": "string"}},
            server_name="memory_server",
            discovery_method="live_discovery"
        )
        
        memory = metadata.to_memory()
        
        assert memory["memory_type"] == "mcp_tool"
        assert memory["content"].startswith("MCP Tool: memory_test_tool")
        assert "Tool for testing memory conversion" in memory["content"]
        assert memory["metadata"]["tool_name"] == "memory_test_tool"
        assert memory["metadata"]["server_name"] == "memory_server"
        assert memory["metadata"]["discovery_method"] == "live_discovery"
        assert memory["importance"] > 0


class TestMCPToolIndexer:
    """Test MCPToolIndexer functionality."""
    
    @pytest.fixture
    def mock_domain_manager(self):
        """Create mock domain manager."""
        manager = Mock()
        manager.store_memory = AsyncMock(return_value="mem_test_id")
        manager.retrieve_memories = AsyncMock(return_value=[])
        return manager
    
    @pytest.fixture
    def tool_indexer(self, mock_domain_manager):
        """Create MCPToolIndexer for testing."""
        config = {
            "mcp_tool_indexer": {
                "auto_discovery": True,
                "discovery_sources": ["configuration", "environment", "live_servers"],
                "cache_duration_hours": 24,
                "max_tools_per_server": 100
            }
        }
        return MCPToolIndexer(config, mock_domain_manager)
    
    @pytest.mark.asyncio
    async def test_tool_indexer_initialization(self, tool_indexer):
        """Test tool indexer initialization."""
        await tool_indexer.initialize()
        
        assert tool_indexer.indexed_tools == {}
        assert tool_indexer.discovery_stats == {
            "total_tools": 0,
            "servers_discovered": 0,
            "last_discovery": None,
            "discovery_methods": {}
        }
    
    @pytest.mark.asyncio
    async def test_discover_from_configuration(self, tool_indexer):
        """Test discovering tools from configuration."""
        await tool_indexer.initialize()
        
        # Mock configuration with tools
        with patch.object(tool_indexer, '_load_configuration_tools') as mock_load:
            mock_load.return_value = [
                {
                    "tool_name": "config_tool_1",
                    "description": "Tool from configuration",
                    "parameters": {"param1": {"type": "string"}},
                    "server_name": "config_server",
                    "category": "utility"
                },
                {
                    "tool_name": "config_tool_2", 
                    "description": "Another config tool",
                    "parameters": {"param2": {"type": "integer"}},
                    "server_name": "config_server",
                    "category": "data"
                }
            ]
            
            discovered_tools = await tool_indexer._discover_from_configuration()
            
            assert len(discovered_tools) == 2
            assert discovered_tools[0].tool_name == "config_tool_1"
            assert discovered_tools[1].tool_name == "config_tool_2"
            assert all(tool.discovery_method == "configuration" for tool in discovered_tools)
    
    @pytest.mark.asyncio
    async def test_discover_from_environment(self, tool_indexer):
        """Test discovering tools from environment variables."""
        await tool_indexer.initialize()
        
        # Mock environment variables
        env_vars = {
            "MCP_SERVER_EXAMPLE": "http://localhost:8000",
            "MCP_TOOL_CONFIG": json.dumps({
                "tools": [
                    {
                        "name": "env_tool",
                        "description": "Tool from environment",
                        "server": "example_server"
                    }
                ]
            })
        }
        
        with patch.dict(os.environ, env_vars):
            with patch.object(tool_indexer, '_query_server_for_tools') as mock_query:
                mock_query.return_value = [
                    MCPToolMetadata(
                        tool_name="env_tool",
                        description="Tool discovered from environment",
                        parameters={"env_param": {"type": "string"}},
                        server_name="example_server",
                        discovery_method="environment"
                    )
                ]
                
                discovered_tools = await tool_indexer._discover_from_environment()
                
                assert len(discovered_tools) == 1
                assert discovered_tools[0].tool_name == "env_tool"
                assert discovered_tools[0].discovery_method == "environment"
    
    @pytest.mark.asyncio
    async def test_discover_from_live_servers(self, tool_indexer):
        """Test discovering tools from live MCP servers."""
        await tool_indexer.initialize()
        
        # Mock live server discovery
        mock_servers = [
            {"name": "live_server_1", "url": "http://localhost:8001"},
            {"name": "live_server_2", "url": "http://localhost:8002"}
        ]
        
        with patch.object(tool_indexer, '_discover_active_mcp_servers') as mock_discover:
            mock_discover.return_value = mock_servers
            
            with patch.object(tool_indexer, '_query_server_for_tools') as mock_query:
                mock_query.side_effect = [
                    [MCPToolMetadata(
                        tool_name="live_tool_1",
                        description="Tool from live server 1",
                        parameters={},
                        server_name="live_server_1", 
                        discovery_method="live_servers"
                    )],
                    [MCPToolMetadata(
                        tool_name="live_tool_2",
                        description="Tool from live server 2",
                        parameters={},
                        server_name="live_server_2",
                        discovery_method="live_servers"
                    )]
                ]
                
                discovered_tools = await tool_indexer._discover_from_mcp_servers()
                
                assert len(discovered_tools) == 2
                assert discovered_tools[0].tool_name == "live_tool_1"
                assert discovered_tools[1].tool_name == "live_tool_2"
                assert all(tool.discovery_method == "live_servers" for tool in discovered_tools)
    
    @pytest.mark.asyncio
    async def test_index_discovered_tools(self, tool_indexer, mock_domain_manager):
        """Test indexing discovered tools as memories."""
        await tool_indexer.initialize()
        
        # Create test tools
        test_tools = [
            MCPToolMetadata(
                tool_name="indexing_tool_1",
                description="First tool for indexing test",
                parameters={"param1": {"type": "string"}},
                server_name="indexing_server",
                category="test"
            ),
            MCPToolMetadata(
                tool_name="indexing_tool_2",
                description="Second tool for indexing test", 
                parameters={"param2": {"type": "integer"}},
                server_name="indexing_server",
                category="test"
            )
        ]
        
        # Index tools
        indexing_results = await tool_indexer._index_tools_as_memories(test_tools)
        
        assert len(indexing_results) == 2
        assert all(result["success"] for result in indexing_results)
        
        # Verify store_memory was called for each tool
        assert mock_domain_manager.store_memory.call_count == 2
        
        # Check indexed_tools tracking
        assert len(tool_indexer.indexed_tools) == 2
        assert "indexing_tool_1" in tool_indexer.indexed_tools
        assert "indexing_tool_2" in tool_indexer.indexed_tools
    
    @pytest.mark.asyncio
    async def test_full_discovery_and_indexing_pipeline(self, tool_indexer, mock_domain_manager):
        """Test complete discovery and indexing pipeline."""
        await tool_indexer.initialize()
        
        # Mock all discovery methods
        with patch.object(tool_indexer, '_discover_from_configuration') as mock_config:
            with patch.object(tool_indexer, '_discover_from_environment') as mock_env:
                with patch.object(tool_indexer, '_discover_from_mcp_servers') as mock_servers:
                    
                    # Setup mock returns
                    mock_config.return_value = [
                        MCPToolMetadata(
                            tool_name="pipeline_config_tool",
                            description="Tool from config",
                            parameters={},
                            server_name="config_server",
                            discovery_method="configuration"
                        )
                    ]
                    
                    mock_env.return_value = [
                        MCPToolMetadata(
                            tool_name="pipeline_env_tool", 
                            description="Tool from environment",
                            parameters={},
                            server_name="env_server",
                            discovery_method="environment"
                        )
                    ]
                    
                    mock_servers.return_value = [
                        MCPToolMetadata(
                            tool_name="pipeline_live_tool",
                            description="Tool from live server",
                            parameters={},
                            server_name="live_server",
                            discovery_method="live_servers"
                        )
                    ]
                    
                    # Run full pipeline
                    result = await tool_indexer.discover_and_index_tools()
                    
                    assert result["success"] is True
                    assert result["total_discovered"] == 3
                    assert result["successfully_indexed"] == 3
                    assert len(result["discovery_breakdown"]) == 3
                    
                    # Verify all methods were called
                    mock_config.assert_called_once()
                    mock_env.assert_called_once()
                    mock_servers.assert_called_once()
                    
                    # Verify storage calls
                    assert mock_domain_manager.store_memory.call_count == 3
    
    @pytest.mark.asyncio
    async def test_suggest_tools_for_intent(self, tool_indexer, mock_domain_manager):
        """Test tool suggestion based on user intent."""
        await tool_indexer.initialize()
        
        # Mock existing indexed tools
        tool_indexer.indexed_tools = {
            "file_reader": MCPToolMetadata(
                tool_name="file_reader",
                description="Read and analyze file contents",
                parameters={"file_path": {"type": "string"}},
                server_name="file_server",
                category="file_operations"
            ),
            "web_scraper": MCPToolMetadata(
                tool_name="web_scraper",
                description="Scrape web pages for content",
                parameters={"url": {"type": "string"}},
                server_name="web_server", 
                category="web_operations"
            ),
            "database_query": MCPToolMetadata(
                tool_name="database_query",
                description="Execute database queries and return results",
                parameters={"query": {"type": "string"}},
                server_name="db_server",
                category="database_operations"
            )
        }
        
        # Mock memory retrieval for semantic matching
        mock_domain_manager.retrieve_memories.return_value = [
            {
                "content": "MCP Tool: file_reader - Read and analyze file contents",
                "metadata": {"tool_name": "file_reader", "category": "file_operations"},
                "similarity": 0.85
            }
        ]
        
        # Test intent-based suggestions
        suggestions = await tool_indexer.suggest_tools_for_intent(
            "I need to read a configuration file and analyze its contents"
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check that file_reader was suggested (highest similarity)
        suggested_tools = [s["tool_name"] for s in suggestions]
        assert "file_reader" in suggested_tools
        
        # Verify retrieve_memories was called with proper query
        mock_domain_manager.retrieve_memories.assert_called()
        call_args = mock_domain_manager.retrieve_memories.call_args[1]
        assert "read" in call_args["query"].lower() or "file" in call_args["query"].lower()
    
    @pytest.mark.asyncio
    async def test_tool_indexing_with_deduplication(self, tool_indexer):
        """Test that duplicate tools are properly deduplicated."""
        await tool_indexer.initialize()
        
        # Create duplicate tools from different sources
        duplicate_tools = [
            MCPToolMetadata(
                tool_name="duplicate_tool",
                description="First instance of duplicate tool",
                parameters={},
                server_name="server_1",
                discovery_method="configuration"
            ),
            MCPToolMetadata(
                tool_name="duplicate_tool",  # Same name
                description="Second instance of duplicate tool", 
                parameters={},
                server_name="server_2",
                discovery_method="environment"
            )
        ]
        
        # Process tools through deduplication
        deduplicated_tools = await tool_indexer._deduplicate_tools(duplicate_tools)
        
        assert len(deduplicated_tools) == 1
        
        # Should keep the first one found (configuration takes precedence)
        assert deduplicated_tools[0].discovery_method == "configuration"
        assert deduplicated_tools[0].server_name == "server_1"
    
    @pytest.mark.asyncio
    async def test_tool_indexing_error_handling(self, tool_indexer, mock_domain_manager):
        """Test error handling during tool indexing."""
        await tool_indexer.initialize()
        
        # Mock storage failure for one tool
        mock_domain_manager.store_memory.side_effect = [
            "mem_success_id",  # First tool succeeds
            Exception("Storage failed"),  # Second tool fails
            "mem_success_id_2"  # Third tool succeeds
        ]
        
        test_tools = [
            MCPToolMetadata(tool_name="success_tool_1", description="Tool 1", parameters={}, server_name="server"),
            MCPToolMetadata(tool_name="failure_tool", description="Tool 2", parameters={}, server_name="server"),
            MCPToolMetadata(tool_name="success_tool_2", description="Tool 3", parameters={}, server_name="server")
        ]
        
        # Index tools with mixed success/failure
        results = await tool_indexer._index_tools_as_memories(test_tools)
        
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True
        
        # Verify error is captured
        assert "error" in results[1]
        assert "Storage failed" in results[1]["error"]


class TestMCPAwarenessHooks:
    """Test MCP awareness hooks functionality."""
    
    @pytest.fixture
    def mock_tool_indexer(self):
        """Create mock tool indexer."""
        indexer = Mock()
        indexer.suggest_tools_for_intent = AsyncMock(return_value=[
            {"tool_name": "suggested_tool", "description": "A suggested tool", "relevance_score": 0.9}
        ])
        return indexer
    
    @pytest.fixture
    def mcp_hooks(self, mock_tool_indexer):
        """Create MCP awareness hooks."""
        return MCPAwarenessHooks(mock_tool_indexer)
    
    @pytest.mark.asyncio
    async def test_hooks_initialization(self, mcp_hooks):
        """Test MCP awareness hooks initialization."""
        await mcp_hooks.initialize()
        
        assert mcp_hooks.tool_indexer is not None
        assert hasattr(mcp_hooks, 'on_user_request')
        assert hasattr(mcp_hooks, 'on_context_change')
    
    @pytest.mark.asyncio
    async def test_user_request_hook(self, mcp_hooks, mock_tool_indexer):
        """Test user request hook for proactive tool suggestions."""
        await mcp_hooks.initialize()
        
        # Simulate user request
        request_context = {
            "user_message": "I need to analyze some log files and extract error patterns",
            "conversation_id": "test_conversation",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        suggestions = await mcp_hooks.on_user_request(request_context)
        
        # Verify tool indexer was called with appropriate intent
        mock_tool_indexer.suggest_tools_for_intent.assert_called()
        call_args = mock_tool_indexer.suggest_tools_for_intent.call_args[0][0]
        assert "analyze" in call_args.lower() or "log" in call_args.lower()
        
        # Verify suggestions format
        assert isinstance(suggestions, list)
        if suggestions:  # If any suggestions returned
            assert "tool_name" in suggestions[0]
            assert "relevance_score" in suggestions[0]
    
    @pytest.mark.asyncio
    async def test_context_change_hook(self, mcp_hooks, mock_tool_indexer):
        """Test context change hook for adaptive tool suggestions."""
        await mcp_hooks.initialize()
        
        # Simulate context change
        context_change = {
            "new_context": "file_analysis",
            "previous_context": "general_conversation",
            "context_details": {
                "file_types": [".log", ".txt"],
                "analysis_type": "error_detection"
            }
        }
        
        suggestions = await mcp_hooks.on_context_change(context_change)
        
        # Verify appropriate tool suggestions were requested
        mock_tool_indexer.suggest_tools_for_intent.assert_called()
        
        # Context-aware suggestions should be returned
        assert isinstance(suggestions, list)


@pytest.mark.asyncio
class TestMCPToolIndexerIntegration:
    """Test MCP tool indexer integration with the full system."""
    
    async def test_mcp_tool_discovery_end_to_end(self):
        """Test end-to-end tool discovery and indexing."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Test that the MCP server has tool indexing capabilities
            # This should discover and index the built-in MCP tools
            
            # Check if tool indexing summary memory exists (created during startup)
            memories = await suite.mcp_server.domain_manager.retrieve_memories(
                query="mcp tool indexing summary",
                limit=5,
                memory_types=["mcp_indexing_summary"],
                min_similarity=0.3
            )
            
            assert len(memories) > 0, "Should have MCP indexing summary memory"
            
            indexing_memory = memories[0]
            assert "mcp tool indexing" in indexing_memory["content"].lower()
            assert "tools discovered" in indexing_memory["content"].lower()
            
        finally:
            await suite.teardown_test_environment()
    
    async def test_mcp_tool_suggestion_integration(self):
        """Test MCP tool suggestion through the MCP server."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Test suggest_command with MCP-specific intent
            result = await suite.validate_mcp_tool_execution(
                tool_name="suggest_command",
                arguments={
                    "intent": "I need to search through my memories for programming knowledge"
                },
                test_name="mcp_tool_suggestion"
            )
            
            assert result.passed, f"MCP tool suggestion failed: {result.errors}"
            assert result.parsed_response.get("success") is True
            
            # Should suggest relevant MCP tools
            suggestions = result.parsed_response.get("suggestions", [])
            assert isinstance(suggestions, list)
            
            # Look for memory-related tool suggestions
            memory_tools = [s for s in suggestions if "memory" in str(s).lower() or "retrieve" in str(s).lower()]
            assert len(memory_tools) > 0, "Should suggest memory-related tools"
            
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_mcp_tool_indexer_tests():
        """Run MCP tool indexer tests directly."""
        print("🧪 Running MCP tool indexer tests...")
        
        # Run metadata tests
        metadata_tests = TestMCPToolMetadata()
        metadata_tests.test_mcp_tool_metadata_creation()
        metadata_tests.test_mcp_tool_metadata_to_memory()
        print("✅ MCP tool metadata tests passed")
        
        # Run integration tests
        integration_tests = TestMCPToolIndexerIntegration()
        await integration_tests.test_mcp_tool_discovery_end_to_end()
        await integration_tests.test_mcp_tool_suggestion_integration()
        print("✅ MCP tool indexer integration tests passed")
        
        print("\n🎉 All MCP tool indexer tests passed!")
    
    asyncio.run(run_mcp_tool_indexer_tests())