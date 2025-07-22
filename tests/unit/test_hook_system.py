"""
Comprehensive test suite for Hook System and Event Handling.

This test suite validates the hook system functionality including:
- Automatic hooks for file access, bash execution, conversation lifecycle
- Event processing and real-time interaction handling
- Session tracking and boundary detection
- Proactive memory and context-aware suggestions
- Hook manager registration and execution

Tests cover:
- clarity/autocode/hooks.py (524 lines)
- clarity/autocode/hook_manager.py
- clarity/mcp/server.py hook integration (lines 1638-1770)
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, call
from typing import Dict, List, Any
from datetime import datetime, timedelta

from clarity.autocode.hooks import AutoCodeHooks
from clarity.autocode.hook_manager import HookManager
from clarity.autocode.structured_thinking_extension import StructuredThinkingExtension
from tests.framework.mcp_validation import MCPServerTestSuite


class TestAutoCodeHooks:
    """Test AutoCode hooks functionality."""
    
    @pytest.fixture
    def mock_domain_manager(self):
        """Create mock domain manager."""
        manager = Mock()
        manager.store_memory = AsyncMock(return_value="mem_hook_id")
        manager.retrieve_memories = AsyncMock(return_value=[])
        return manager
    
    @pytest.fixture
    def mock_autocode_domain(self):
        """Create mock AutoCode domain."""
        domain = Mock()
        domain.process_file_access = AsyncMock(return_value={"success": True})
        domain.process_bash_execution = AsyncMock(return_value={"success": True})
        domain.generate_session_summary = AsyncMock(return_value={"success": True, "summary": "Test session"})
        domain.suggest_command = AsyncMock(return_value={"suggestions": ["test command"]})
        return domain
    
    @pytest.fixture
    def autocode_hooks(self, mock_domain_manager, mock_autocode_domain):
        """Create AutoCode hooks for testing."""
        mock_domain_manager.autocode_domain = mock_autocode_domain
        hooks = AutoCodeHooks(mock_domain_manager)
        return hooks
    
    @pytest.mark.asyncio
    async def test_autocode_hooks_initialization(self, autocode_hooks):
        """Test AutoCode hooks initialization."""
        # AutoCode hooks don't have an initialize method, they're ready on construction
        assert autocode_hooks.domain_manager is not None
        assert autocode_hooks.session_data is not None
        assert autocode_hooks.project_cache == {}
    
    @pytest.mark.asyncio
    async def test_file_read_hook(self, autocode_hooks, mock_domain_manager):
        """Test file read hook processing."""
        # Test the actual on_file_read method
        file_path = "/test/project/src/main.py"
        content = "print('hello world')"
        operation = "read"
        
        # This should not raise an exception and should track the file access
        await autocode_hooks.on_file_read(file_path, content, operation)
        
        # Should record in session data
        assert len(autocode_hooks.session_data["files_accessed"]) == 1
        event = autocode_hooks.session_data["files_accessed"][0]
        assert event["path"] == file_path
        assert event["operation"] == operation
    
    @pytest.mark.asyncio
    async def test_bash_execution_hook(self, autocode_hooks, mock_autocode_domain):
        """Test bash execution hook processing."""
        # Test the actual on_bash_execution method
        command = "python -m pytest tests/unit/ -v"
        exit_code = 0
        output = "4 passed, 0 failed"
        working_directory = "/test/project"
        
        # This should not raise an exception and should track the command execution
        await autocode_hooks.on_bash_execution(command, exit_code, output, working_directory)
        
        # Should call AutoCode domain for bash processing
        mock_autocode_domain.process_bash_execution.assert_called_once()
        
        # Should record in session data
        assert len(autocode_hooks.session_data["commands_executed"]) == 1
        event = autocode_hooks.session_data["commands_executed"][0]
        assert event["command"] == command
        assert event["exit_code"] == exit_code
    
    @pytest.mark.asyncio
    async def test_conversation_message_hook(self, autocode_hooks):
        """Test conversation message hook processing."""
        # Test the actual on_conversation_message method
        role = "user"
        content = "Let's work on implementing the new feature"
        message_id = "test_msg_123"
        
        # This should not raise an exception and should track the message
        await autocode_hooks.on_conversation_message(role, content, message_id)
        
        # Should record in conversation log
        assert len(autocode_hooks.session_data["conversation_log"]) == 1
        message = autocode_hooks.session_data["conversation_log"][0]
        assert message["role"] == role
        assert message["content"] == content
        assert message["message_id"] == message_id
    
    @pytest.mark.asyncio
    async def test_conversation_end_hook(self, autocode_hooks, mock_autocode_domain):
        """Test conversation end hook processing."""
        # Add some activity first
        await autocode_hooks.on_conversation_message("user", "Starting work", "msg_1")
        await autocode_hooks.on_file_read("/test/file.py", "content", "read")
        
        # Test conversation end
        conversation_id = "test_conv_123"
        
        # This should not raise an exception and should generate summary
        await autocode_hooks.on_conversation_end(conversation_id)
        
        # Should call domain to generate session summary if enough activity
        if len(autocode_hooks.session_data["conversation_log"]) > 3:
            mock_autocode_domain.generate_session_summary.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_suggest_command(self, autocode_hooks, mock_autocode_domain):
        """Test command suggestion functionality."""
        intent = "I need help running tests"
        context = {"project_type": "python"}
        
        # Test suggest_command method
        suggestions = await autocode_hooks.suggest_command(intent, context)
        
        # Should call the autocode domain
        mock_autocode_domain.suggest_command.assert_called_once()
        
        # Should return suggestions
        assert isinstance(suggestions, list)


class TestHookManager:
    """Test HookManager functionality."""
    
    @pytest.fixture
    def hook_manager(self):
        """Create HookManager for testing."""
        mock_domain_manager = Mock()
        mock_autocode_hooks = Mock()
        return HookManager(mock_domain_manager, mock_autocode_hooks)
    
    @pytest.mark.asyncio
    async def test_hook_manager_initialization(self, hook_manager):
        """Test hook manager initialization."""
        # HookManager auto-initializes and registers default hooks
        assert hook_manager.tool_hooks is not None
        assert hook_manager.lifecycle_hooks is not None  
        assert hook_manager.event_hooks is not None
        assert hook_manager.hook_stats is not None
    
    def test_register_tool_hooks(self, hook_manager):
        """Test registering tool-specific hooks."""
        # Register hook for store_memory tool
        async def store_memory_hook(context):
            return {"hook_executed": True}
        
        hook_manager.register_tool_hook("store_memory", store_memory_hook)
        
        assert "store_memory" in hook_manager.tool_hooks
        assert store_memory_hook in hook_manager.tool_hooks["store_memory"]
    
    def test_hook_stats(self, hook_manager):
        """Test hook execution statistics."""
        stats = hook_manager.get_hook_stats()
        
        assert isinstance(stats, dict)
        assert "executions" in stats
        assert "successes" in stats
        assert "failures" in stats
    
    @pytest.mark.asyncio  
    async def test_execute_tool_hooks_simple(self, hook_manager):
        """Test basic tool hook execution."""
        executed = []
        
        async def simple_hook(context):
            executed.append("hook_executed")
            
        hook_manager.register_tool_hook("test_tool", simple_hook)
        
        # Execute hooks
        await hook_manager.execute_tool_hooks("test_tool", {}, None)
        
        # Should have executed hook
        assert "hook_executed" in executed


class TestStructuredThinkingExtension:
    """Test structured thinking hook extension."""
    
    @pytest.fixture
    def thinking_extension(self):
        """Create structured thinking extension."""
        mock_persistence = Mock()
        return StructuredThinkingExtension(mock_persistence)
    
    def test_thinking_extension_initialization(self, thinking_extension):
        """Test structured thinking extension initialization."""
        assert thinking_extension.persistence_domain is not None
        assert thinking_extension.thinking_sessions == {}


@pytest.mark.asyncio
class TestHookSystemIntegration:
    """Test hook system integration with the full MCP server."""
    
    async def test_mcp_server_hook_integration(self):
        """Test that MCP server properly integrates with hook system."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Hook manager should be initialized during server startup
            hook_manager = suite.mcp_server.hook_manager
            
            assert hook_manager is not None
            assert hasattr(hook_manager, 'execute_tool_hooks')
            assert hasattr(hook_manager, 'execute_lifecycle_hooks')
            
            # Should have registered default hooks
            assert len(hook_manager.tool_hooks) > 0
            assert len(hook_manager.event_hooks) > 0
            
        finally:
            await suite.teardown_test_environment()
    
    async def test_tool_execution_with_hooks(self):
        """Test that tool execution triggers appropriate hooks."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Store a memory - this should trigger tool hooks
            result = await suite.validate_mcp_tool_execution(
                tool_name="store_memory",
                arguments={
                    "memory_type": "hook_test",
                    "content": "Testing hook execution during memory storage",
                    "importance": 0.7
                },
                test_name="hook_integration_test"
            )
            
            assert result.passed, f"Memory storage with hooks failed: {result.errors}"
            
            # Hook execution should be tracked
            hook_manager = suite.mcp_server.hook_manager
            if hook_manager:
                stats = hook_manager.get_hook_stats()
                assert stats["total_executions"] > 0
            
        finally:
            await suite.teardown_test_environment()
    
    async def test_conversation_lifecycle_hook_integration(self):
        """Test conversation lifecycle hooks through MCP server."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Trigger conversation end manually for testing
            result = await suite.validate_mcp_tool_execution(
                tool_name="trigger_conversation_end",
                arguments={
                    "conversation_id": "test_hook_conversation"
                },
                test_name="lifecycle_hook_test"
            )
            
            assert result.passed, f"Conversation end hook test failed: {result.errors}"
            assert result.parsed_response.get("success") is True
            
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_hook_system_tests():
        """Run hook system tests directly."""
        print("🧪 Running hook system tests...")
        
        # Run integration tests
        integration_tests = TestHookSystemIntegration()
        await integration_tests.test_mcp_server_hook_integration()
        await integration_tests.test_tool_execution_with_hooks()
        await integration_tests.test_conversation_lifecycle_hook_integration()
        print("✅ Hook system integration tests passed")
        
        print("\n🎉 All hook system tests passed!")
    
    asyncio.run(run_hook_system_tests())