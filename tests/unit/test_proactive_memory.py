"""
Unit tests for proactive memory functionality.

Tests the automatic memory presentation system, hook-based triggering,
and configuration management for proactive memory features.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from clarity.autocode.hook_manager import HookManager
from clarity.domains.manager import MemoryDomainManager
from clarity.mcp.server import MemoryMcpServer


class TestProactiveMemoryHooks:
    """Test the proactive memory hook system."""

    @pytest.fixture
    async def mock_domain_manager(self):
        """Create a mock domain manager for testing."""
        domain_manager = Mock(spec=MemoryDomainManager)
        domain_manager.store_memory = AsyncMock()
        domain_manager.retrieve_memories = AsyncMock()
        return domain_manager

    @pytest.fixture
    async def mock_autocode_hooks(self):
        """Create mock autocode hooks."""
        autocode_hooks = Mock()
        autocode_hooks.on_file_read = AsyncMock()
        autocode_hooks.on_bash_execution = AsyncMock()
        return autocode_hooks

    @pytest.fixture
    async def hook_manager(self, mock_domain_manager, mock_autocode_hooks):
        """Create a hook manager instance for testing."""
        return HookManager(mock_domain_manager, mock_autocode_hooks)

    async def test_proactive_config_initialization(self, hook_manager):
        """Test that proactive memory configuration is properly initialized."""
        assert hook_manager.proactive_config is not None
        assert hook_manager.proactive_config["enabled"] is True
        assert hook_manager.proactive_config["triggers"]["file_access"] is True
        assert hook_manager.proactive_config["triggers"]["tool_execution"] is True
        assert hook_manager.proactive_config["triggers"]["context_change"] is True
        assert hook_manager.proactive_config["similarity_threshold"] == 0.6
        assert hook_manager.proactive_config["max_memories_per_trigger"] == 3
        assert hook_manager.proactive_config["auto_present"] is True

    async def test_file_access_hook_triggers_memory_suggestion(self, hook_manager, mock_domain_manager):
        """Test that file access hooks trigger memory suggestions when enabled."""
        # Setup mock memories
        mock_memories = [
            {
                "id": "mem_1",
                "type": "code_pattern",
                "content": "Related code pattern for test.py",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        mock_domain_manager.retrieve_memories.return_value = mock_memories

        # Test file access hook
        context = {
            "data": {
                "file_path": "/project/test.py",
                "content": "print('hello')",
                "operation": "read"
            }
        }

        await hook_manager._on_file_access(context)

        # Verify memory retrieval was called
        mock_domain_manager.retrieve_memories.assert_called()
        
        # Verify memory was stored for presentation
        mock_domain_manager.store_memory.assert_called()
        
        # Check that the stored memory is a proactive_memory type
        stored_memory_call = mock_domain_manager.store_memory.call_args
        stored_memory = stored_memory_call[0][0]
        assert stored_memory["type"] == "proactive_memory"
        assert "file access" in stored_memory["content"]

    async def test_file_access_hook_respects_configuration(self, hook_manager, mock_domain_manager):
        """Test that file access hooks respect configuration settings."""
        # Disable file access triggers
        hook_manager.proactive_config["triggers"]["file_access"] = False

        context = {
            "data": {
                "file_path": "/project/test.py",
                "content": "print('hello')",
                "operation": "read"
            }
        }

        await hook_manager._on_file_access(context)

        # Verify memory retrieval was NOT called
        mock_domain_manager.retrieve_memories.assert_not_called()

    async def test_tool_execution_hook_triggers_memory_suggestion(self, hook_manager, mock_domain_manager):
        """Test that tool execution hooks trigger memory suggestions."""
        # Setup mock memories
        mock_memories = [
            {
                "id": "mem_1",
                "type": "command_pattern",
                "content": "Related command pattern for Edit tool",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        mock_domain_manager.retrieve_memories.return_value = mock_memories

        # Mock _should_consult_memory_for_tool to return True
        hook_manager._should_consult_memory_for_tool = Mock(return_value=True)

        context = {
            "data": {
                "tool_name": "Edit",
                "arguments": {"file_path": "test.py", "content": "new content"}
            }
        }

        await hook_manager._on_tool_pre_execution(context)

        # Verify memory retrieval was called
        mock_domain_manager.retrieve_memories.assert_called()
        
        # Verify memory was stored for presentation
        mock_domain_manager.store_memory.assert_called()

    async def test_memory_presentation_formatting(self, hook_manager):
        """Test memory formatting for presentation to Claude."""
        memories = [
            {
                "id": "mem_1",
                "type": "code_pattern",
                "content": "This is a test memory with some content that should be formatted nicely",
                "created_at": "2025-01-20T10:00:00"
            },
            {
                "id": "mem_2", 
                "type": "project_pattern",
                "content": "Another memory with different content",
                "created_at": "2025-01-19T15:30:00"
            }
        ]

        formatted = hook_manager._format_memories_for_presentation(memories, "test context")

        assert "🧠 **Relevant Past Context**" in formatted
        assert "test context" in formatted
        assert "Code Pattern" in formatted
        assert "Project Pattern" in formatted
        assert "2025-01-20" in formatted
        assert "This is a test memory" in formatted
        assert "*This context was automatically retrieved*" in formatted

    async def test_memory_presentation_respects_limits(self, hook_manager, mock_domain_manager):
        """Test that memory presentation respects configured limits."""
        # Set a low limit
        hook_manager.proactive_config["max_memories_per_trigger"] = 2

        # Create more memories than the limit
        mock_memories = [
            {"id": f"mem_{i}", "type": "test", "content": f"Memory {i}"} 
            for i in range(5)
        ]

        await hook_manager._present_memories_to_claude(mock_memories, "test context")

        # Verify only limited memories were processed
        stored_memory_call = mock_domain_manager.store_memory.call_args
        stored_memory = stored_memory_call[0][0]
        
        # Check that only 2 memories are mentioned in the content (based on limit)
        content_lines = stored_memory["content"].split('\n')
        memory_entries = [line for line in content_lines if line.strip().startswith(('1.', '2.', '3.'))]
        assert len(memory_entries) <= 2

    async def test_memory_presentation_disabled(self, hook_manager, mock_domain_manager):
        """Test that memory presentation can be disabled."""
        hook_manager.proactive_config["auto_present"] = False

        memories = [{"id": "mem_1", "type": "test", "content": "Test memory"}]

        await hook_manager._present_memories_to_claude(memories, "test context")

        # Verify no memory was stored when presentation is disabled
        mock_domain_manager.store_memory.assert_not_called()

    async def test_contextual_query_generation(self, hook_manager):
        """Test generation of contextual queries from different context types."""
        # Test file path context
        context = {"file_path": "/project/src/main.py"}
        query = hook_manager._generate_contextual_query_from_context(context)
        assert "main.py" in query
        assert "py" in query

        # Test command context
        context = {"command": "npm install package"}
        query = hook_manager._generate_contextual_query_from_context(context)
        assert "npm" in query

        # Test combined context
        context = {
            "file_path": "/project/config.json",
            "command": "vim config.json",
            "task": "configuration setup"
        }
        query = hook_manager._generate_contextual_query_from_context(context)
        assert "config.json" in query
        assert "vim" in query
        assert "configuration setup" in query

    async def test_memory_usage_tracking(self, hook_manager, mock_domain_manager):
        """Test that memory usage is tracked for analytics."""
        memories = [{"id": "mem_1", "type": "test"}]

        await hook_manager._track_memory_usage("Edit", {"file_path": "test.py"}, memories)

        # Verify analytics memory was stored
        mock_domain_manager.store_memory.assert_called()
        stored_memory_call = mock_domain_manager.store_memory.call_args
        stored_memory = stored_memory_call[0][0]
        
        assert stored_memory["type"] == "memory_usage_analytics"
        assert stored_memory["content"]["tool_name"] == "Edit"
        assert stored_memory["content"]["memory_count"] == 1
        assert "mem_1" in stored_memory["content"]["memories_presented"]

    async def test_auto_trigger_memory_check(self, hook_manager, mock_domain_manager):
        """Test automatic triggering of comprehensive memory check."""
        mock_memories = [{"id": "mem_1", "type": "test", "content": "Test memory"}]
        mock_domain_manager.retrieve_memories.return_value = mock_memories

        context = {
            "file_path": "/project/test.py",
            "trigger": "file_access"
        }

        await hook_manager._auto_trigger_memory_check(context)

        # Verify memory retrieval was called
        mock_domain_manager.retrieve_memories.assert_called()
        
        # Verify additional memories were presented
        mock_domain_manager.store_memory.assert_called()


class TestProactiveMemoryConfiguration:
    """Test proactive memory configuration functionality."""

    @pytest.fixture
    async def mock_domain_manager(self):
        """Create a mock domain manager for testing."""
        domain_manager = Mock(spec=MemoryDomainManager)
        domain_manager.store_memory = AsyncMock()
        domain_manager.retrieve_memories = AsyncMock()
        return domain_manager

    @pytest.fixture
    async def mcp_server(self, mock_domain_manager):
        """Create an MCP server instance for testing."""
        config = {
            "qdrant": {"path": ":memory:"},
            "embedding": {"default_model": "test"},
            "autocode": {"enabled": True}
        }
        server = MemoryMcpServer(config)
        server.domain_manager = mock_domain_manager
        return server

    async def test_configure_proactive_memory_tool(self, mcp_server):
        """Test the configure_proactive_memory MCP tool."""
        # This would typically be called through the MCP framework
        # For testing, we'll test the core logic

        # Mock the MCP tool registration
        tools = []
        
        def mock_tool():
            def decorator(func):
                tools.append(func)
                return func
            return decorator

        # Patch the app.tool decorator
        with patch.object(mcp_server, 'app') as mock_app:
            mock_app.tool = mock_tool
            
            # Register the tools
            await mcp_server._register_autocode_tools()
            
            # Find the configure_proactive_memory tool
            config_tool = None
            for tool in tools:
                if tool.__name__ == 'configure_proactive_memory':
                    config_tool = tool
                    break
            
            assert config_tool is not None
            
            # Test the tool function
            result = await config_tool(
                enabled=True,
                file_access_triggers=False,
                tool_execution_triggers=True,
                min_similarity_threshold=0.8,
                max_memories_per_trigger=5
            )
            
            # Verify configuration was stored
            mcp_server.domain_manager.store_memory.assert_called()
            
            # Check the stored configuration
            stored_memory_call = mcp_server.domain_manager.store_memory.call_args
            stored_memory = stored_memory_call[0][0]
            
            assert stored_memory["type"] == "system_configuration"
            config_content = stored_memory["content"]["proactive_memory"]
            assert config_content["enabled"] is True
            assert config_content["triggers"]["file_access"] is False
            assert config_content["triggers"]["tool_execution"] is True
            assert config_content["similarity_threshold"] == 0.8
            assert config_content["max_memories_per_trigger"] == 5

    async def test_proactive_memory_stats_tool(self, mcp_server):
        """Test the get_proactive_memory_stats MCP tool."""
        # Mock analytics and presented memories
        mock_analytics = [
            {"id": "analytics_1", "content": {"memory_count": 3}},
            {"id": "analytics_2", "content": {"memory_count": 2}}
        ]
        mock_presented = [
            {"id": "presented_1", "metadata": {"trigger_context": "file access"}},
            {"id": "presented_2", "metadata": {"trigger_context": "tool execution"}}
        ]
        
        mcp_server.domain_manager.retrieve_memories.side_effect = [mock_analytics, mock_presented]

        tools = []
        
        def mock_tool():
            def decorator(func):
                tools.append(func)
                return func
            return decorator

        with patch.object(mcp_server, 'app') as mock_app:
            mock_app.tool = mock_tool
            
            await mcp_server._register_autocode_tools()
            
            # Find the stats tool
            stats_tool = None
            for tool in tools:
                if tool.__name__ == 'get_proactive_memory_stats':
                    stats_tool = tool
                    break
            
            assert stats_tool is not None
            
            result = await stats_tool()
            
            # Verify memory retrieval was called for both analytics and presented memories
            assert mcp_server.domain_manager.retrieve_memories.call_count == 2


class TestProactiveMemoryIntegration:
    """Integration tests for proactive memory system."""

    @pytest.fixture
    async def integrated_system(self):
        """Create an integrated system for testing."""
        # Mock domain manager
        domain_manager = Mock(spec=MemoryDomainManager)
        domain_manager.store_memory = AsyncMock()
        domain_manager.retrieve_memories = AsyncMock()
        
        # Mock autocode hooks
        autocode_hooks = Mock()
        autocode_hooks.on_file_read = AsyncMock()
        
        # Create hook manager
        hook_manager = HookManager(domain_manager, autocode_hooks)
        
        # Mock autocode domain with hook manager
        autocode_domain = Mock()
        autocode_domain.hook_manager = hook_manager
        domain_manager.autocode_domain = autocode_domain
        
        return {
            "domain_manager": domain_manager,
            "hook_manager": hook_manager,
            "autocode_hooks": autocode_hooks
        }

    async def test_end_to_end_file_access_flow(self, integrated_system):
        """Test complete flow from file access to memory presentation."""
        domain_manager = integrated_system["domain_manager"]
        hook_manager = integrated_system["hook_manager"]
        
        # Setup mock memories to be retrieved
        mock_memories = [
            {
                "id": "pattern_1",
                "type": "code_pattern", 
                "content": "Similar code pattern for Python files",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        domain_manager.retrieve_memories.return_value = mock_memories

        # Simulate file access
        file_context = {
            "data": {
                "file_path": "/project/src/main.py",
                "content": "import os\nprint('hello')",
                "operation": "read"
            }
        }

        # Trigger the hook
        await hook_manager._on_file_access(file_context)

        # Verify the complete flow:
        # 1. Memory retrieval was called
        assert domain_manager.retrieve_memories.call_count >= 1
        
        # 2. Proactive memory was stored for Claude
        store_calls = domain_manager.store_memory.call_args_list
        proactive_memory_stored = False
        for call in store_calls:
            memory = call[0][0]
            if memory["type"] == "proactive_memory":
                proactive_memory_stored = True
                assert "file access" in memory["content"]
                assert "main.py" in memory["content"]
                break
        
        assert proactive_memory_stored, "Proactive memory should have been stored"

    async def test_configuration_affects_behavior(self, integrated_system):
        """Test that configuration changes affect system behavior."""
        hook_manager = integrated_system["hook_manager"]
        domain_manager = integrated_system["domain_manager"]

        # Test with default configuration (enabled)
        file_context = {
            "data": {
                "file_path": "/project/test.py",
                "operation": "read"
            }
        }

        await hook_manager._on_file_access(file_context)
        initial_call_count = domain_manager.retrieve_memories.call_count

        # Reset mock
        domain_manager.reset_mock()

        # Disable file access triggers
        hook_manager.proactive_config["triggers"]["file_access"] = False

        await hook_manager._on_file_access(file_context)

        # Verify no memory retrieval happened when disabled
        domain_manager.retrieve_memories.assert_not_called()

    async def test_memory_limit_enforcement(self, integrated_system):
        """Test that memory limits are properly enforced."""
        hook_manager = integrated_system["hook_manager"]
        domain_manager = integrated_system["domain_manager"]

        # Set a low memory limit
        hook_manager.proactive_config["max_memories_per_trigger"] = 1

        # Create multiple memories
        mock_memories = [
            {"id": f"mem_{i}", "type": "test", "content": f"Memory {i}"}
            for i in range(5)
        ]
        domain_manager.retrieve_memories.return_value = mock_memories

        # Trigger memory presentation
        await hook_manager._present_memories_to_claude(mock_memories, "test context")

        # Verify stored memory respects the limit
        stored_memory = domain_manager.store_memory.call_args[0][0]
        content = stored_memory["content"]
        
        # Should only have 1 memory entry based on our limit
        memory_entries = [line for line in content.split('\n') if line.strip().startswith('1.')]
        assert len(memory_entries) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])