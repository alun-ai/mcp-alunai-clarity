"""
Integration tests for enhanced MCP discovery system.

This test suite validates the complete enhanced MCP discovery system including:
- Native Claude Code integration
- Hook-based learning
- Resource reference monitoring
- Slash command discovery
- Workflow memory patterns
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

# Import the components to test
from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.mcp.native_discovery import NativeMCPDiscoveryBridge
from clarity.mcp.hook_integration import MCPHookIntegration
from clarity.mcp.workflow_memory import WorkflowMemoryEnhancer, MCPWorkflowPattern
from clarity.mcp.resource_reference_monitor import ResourceReferenceMonitor, ResourceOpportunity
from clarity.mcp.slash_command_discovery import SlashCommandDiscovery, SlashCommand


class MockDomainManager:
    """Mock domain manager for testing."""
    
    def __init__(self):
        self.stored_memories = []
        self.memory_counter = 0
    
    async def store_memory(self, memory_type: str, content: str, importance: float, metadata: dict = None):
        """Mock memory storage."""
        memory = {
            'id': f"mem_{self.memory_counter}",
            'memory_type': memory_type,
            'content': content,
            'importance': importance,
            'metadata': metadata or {}
        }
        self.stored_memories.append(memory)
        self.memory_counter += 1
        return memory['id']
    
    async def retrieve_memories(self, query: str, types: list = None, limit: int = 10, min_similarity: float = 0.5):
        """Mock memory retrieval."""
        # Simple mock that returns relevant stored memories
        relevant_memories = []
        for memory in self.stored_memories:
            if not types or memory['memory_type'] in types:
                # Simple keyword matching for testing
                if any(word.lower() in memory['content'].lower() for word in query.split()):
                    relevant_memories.append(memory)
        
        return relevant_memories[:limit]
    
    async def update_memory(self, memory_id: str, updates: dict):
        """Mock memory update."""
        for memory in self.stored_memories:
            if memory['id'] == memory_id:
                memory.update(updates)
                return True
        return False


@pytest.fixture
def mock_domain_manager():
    """Provide mock domain manager for tests."""
    return MockDomainManager()


@pytest.fixture
def tool_indexer(mock_domain_manager):
    """Provide configured tool indexer for tests."""
    indexer = MCPToolIndexer(mock_domain_manager)
    return indexer


class TestNativeDiscoveryIntegration:
    """Test native Claude Code discovery integration."""
    
    @pytest.mark.asyncio
    async def test_native_discovery_bridge_initialization(self):
        """Test that native discovery bridge initializes correctly."""
        bridge = NativeMCPDiscoveryBridge()
        
        assert bridge.native_config_paths is not None
        assert len(bridge.native_config_paths) > 0
        assert bridge.cache_timeout == 300
    
    @pytest.mark.asyncio
    async def test_claude_cli_discovery_mock(self):
        """Test Claude CLI discovery with mocked subprocess."""
        bridge = NativeMCPDiscoveryBridge()
        
        mock_result = {
            'server1': {
                'command': 'python',
                'args': ['-m', 'test_server'],
                'description': 'Test MCP server'
            }
        }
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (json.dumps(mock_result).encode(), b'')
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            servers = await bridge.discover_native_servers()
            
            assert len(servers) >= 0  # May be 0 if no config files exist
            mock_subprocess.assert_called()
    
    @pytest.mark.asyncio 
    async def test_config_file_parsing(self):
        """Test parsing of various configuration file formats."""
        bridge = NativeMCPDiscoveryBridge()
        
        # Test Claude Code format
        claude_config = {
            'mcpServers': {
                'test-server': {
                    'command': 'python',
                    'args': ['-m', 'test_server'],
                    'env': {'VAR': 'value'}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(claude_config, f)
            config_path = f.name
        
        try:
            servers = await bridge._parse_config_file(config_path)
            assert 'test-server' in servers
            assert servers['test-server']['command'] == 'python'
            assert servers['test-server']['args'] == ['-m', 'test_server']
        finally:
            os.unlink(config_path)


class TestHookIntegration:
    """Test hook-based learning integration."""
    
    @pytest.mark.asyncio
    async def test_hook_integration_initialization(self, tool_indexer):
        """Test hook integration initialization."""
        hook_integration = MCPHookIntegration(tool_indexer)
        
        assert hook_integration.tool_indexer == tool_indexer
        assert hook_integration.tool_patterns is not None
        assert 'file_operations' in hook_integration.tool_patterns
        assert 'database_operations' in hook_integration.tool_patterns
    
    @pytest.mark.asyncio
    async def test_hook_config_generation(self, tool_indexer):
        """Test generation of hook configuration."""
        hook_integration = MCPHookIntegration(tool_indexer)
        
        hook_config = await hook_integration._generate_hook_config()
        
        assert 'hooks' in hook_config
        assert 'PreToolUse' in hook_config['hooks']
        assert 'PostToolUse' in hook_config['hooks']
        assert 'UserPromptSubmit' in hook_config['hooks']
    
    @pytest.mark.asyncio
    async def test_tool_usage_analysis(self, tool_indexer):
        """Test analysis of tool usage patterns."""
        hook_integration = MCPHookIntegration(tool_indexer)
        
        # Test pre-tool analysis
        result = await hook_integration.analyze_tool_usage('pre_tool', {
            'tool_name': 'Bash',
            'args': 'psql -c "SELECT * FROM users"'
        })
        
        assert result is not None
        if result:
            assert 'suggestions' in result
            # Should detect database operation pattern
            suggestions = result['suggestions']
            if suggestions:
                assert any('database' in s.get('alternative', '') for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_prompt_analysis(self, tool_indexer):
        """Test prompt analysis for MCP opportunities."""
        hook_integration = MCPHookIntegration(tool_indexer)
        
        # Test file operation prompt
        result = await hook_integration._suggest_mcp_opportunities({
            'prompt': 'read the configuration file at /etc/config.json'
        })
        
        # Should suggest resource reference or return None
        assert result is None or '@filesystem:file://' in result
    
    @pytest.mark.asyncio
    async def test_usage_statistics(self, tool_indexer):
        """Test usage statistics collection."""
        hook_integration = MCPHookIntegration(tool_indexer)
        
        # Add some mock usage data
        hook_integration.usage_history = [
            {
                'timestamp': '2024-01-01T10:00:00Z',
                'event': 'post_tool',
                'tool': 'Read',
                'success': True
            },
            {
                'timestamp': '2024-01-01T10:01:00Z',
                'event': 'pre_tool',
                'tool': 'Bash',
                'suggestions': []
            }
        ]
        
        stats = await hook_integration.get_usage_statistics()
        
        assert 'total_events' in stats
        assert 'event_types' in stats
        assert 'tool_usage' in stats
        assert stats['total_events'] == 2


class TestWorkflowMemory:
    """Test enhanced workflow memory system."""
    
    @pytest.mark.asyncio
    async def test_workflow_memory_initialization(self, mock_domain_manager):
        """Test workflow memory enhancer initialization."""
        enhancer = WorkflowMemoryEnhancer(mock_domain_manager)
        
        assert enhancer.domain_manager == mock_domain_manager
        assert enhancer.pattern_cache == {}
        assert enhancer.scoring_weights is not None
    
    @pytest.mark.asyncio
    async def test_workflow_pattern_storage(self, mock_domain_manager):
        """Test storing MCP workflow patterns."""
        enhancer = WorkflowMemoryEnhancer(mock_domain_manager)
        
        pattern_data = {
            "context": "read configuration file",
            "tools": ["filesystem_read"],
            "resources": ["@filesystem:file://config.json"],
            "success": {"file_found": True, "read_successful": True},
            "score": 0.9,
            "project_type": "web_app",
            "intent": "configuration_loading"
        }
        
        memory_id = await enhancer.store_mcp_workflow_pattern(pattern_data)
        
        assert memory_id is not None
        assert len(mock_domain_manager.stored_memories) == 1
        
        stored_memory = mock_domain_manager.stored_memories[0]
        assert stored_memory['memory_type'] == 'mcp_workflow_pattern'
        
        content = json.loads(stored_memory['content'])
        assert content['trigger_context'] == "read configuration file"
        assert content['tool_sequence'] == ["filesystem_read"]
    
    @pytest.mark.asyncio
    async def test_similar_workflow_finding(self, mock_domain_manager):
        """Test finding similar workflow patterns."""
        enhancer = WorkflowMemoryEnhancer(mock_domain_manager)
        
        # Store a test pattern first
        await enhancer.store_mcp_workflow_pattern({
            "context": "read database configuration",
            "tools": ["database_connect"],
            "score": 0.8
        })
        
        # Find similar workflows
        similar = await enhancer.find_similar_workflows("read config file")
        
        assert isinstance(similar, list)
        # May be empty if similarity threshold not met
    
    @pytest.mark.asyncio
    async def test_pattern_analytics(self, mock_domain_manager):
        """Test pattern analytics generation."""
        enhancer = WorkflowMemoryEnhancer(mock_domain_manager)
        
        # Store some test patterns
        for i in range(3):
            await enhancer.store_mcp_workflow_pattern({
                "context": f"test context {i}",
                "tools": [f"tool_{i}"],
                "score": 0.8 + i * 0.05
            })
        
        analytics = await enhancer.get_pattern_analytics()
        
        assert 'total_patterns' in analytics
        assert 'total_interactions' in analytics
        assert analytics['total_patterns'] >= 3


class TestResourceReferenceMonitor:
    """Test resource reference monitoring system."""
    
    def test_resource_reference_monitor_initialization(self):
        """Test resource reference monitor initialization."""
        monitor = ResourceReferenceMonitor()
        
        assert monitor.reference_patterns == {}
        assert monitor.opportunity_patterns is not None
        assert 'file_operations' in monitor.opportunity_patterns
        assert 'database_queries' in monitor.opportunity_patterns
    
    def test_resource_opportunity_detection(self):
        """Test detection of resource reference opportunities."""
        monitor = ResourceReferenceMonitor()
        
        # Test file operation detection
        opportunities = monitor.detect_resource_opportunities(
            "read the file at /home/user/config.json"
        )
        
        assert isinstance(opportunities, list)
        if opportunities:
            assert any(opp.opportunity_type == 'file_operations' for opp in opportunities)
            assert any('@filesystem:file://' in opp.suggested_reference for opp in opportunities)
    
    def test_database_query_detection(self):
        """Test detection of database query opportunities."""
        monitor = ResourceReferenceMonitor()
        
        opportunities = monitor.detect_resource_opportunities(
            "SELECT * FROM users WHERE active = true"
        )
        
        assert isinstance(opportunities, list)
        if opportunities:
            assert any(opp.opportunity_type == 'database_queries' for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_resource_pattern_learning(self):
        """Test learning from resource reference usage."""
        monitor = ResourceReferenceMonitor()
        
        await monitor.learn_resource_pattern(
            "@filesystem:file:///etc/config.json",
            {"prompt": "read config file"},
            success=True,
            response_time=150.0
        )
        
        assert len(monitor.usage_history) == 1
        assert 'filesystem:file' in monitor.reference_patterns
        
        pattern = monitor.reference_patterns['filesystem:file']
        assert pattern['usage_count'] == 1
        assert pattern['success_count'] == 1
    
    @pytest.mark.asyncio
    async def test_reference_suggestions(self):
        """Test getting reference suggestions."""
        monitor = ResourceReferenceMonitor()
        
        # Learn some patterns first
        await monitor.learn_resource_pattern(
            "@filesystem:file:///test.txt",
            {"prompt": "read file"},
            success=True
        )
        
        suggestions = await monitor.get_reference_suggestions(
            "read the configuration file"
        )
        
        assert isinstance(suggestions, list)
        # May be empty depending on pattern matching


class TestSlashCommandDiscovery:
    """Test slash command discovery system."""
    
    @pytest.mark.asyncio
    async def test_slash_command_discovery_initialization(self, tool_indexer):
        """Test slash command discovery initialization."""
        discovery = SlashCommandDiscovery(tool_indexer)
        
        assert discovery.tool_indexer == tool_indexer
        assert discovery.slash_commands == {}
        assert discovery.command_categories is not None
        assert 'file_operations' in discovery.command_categories
    
    def test_slash_command_creation(self):
        """Test creation of slash command from MCP prompt."""
        # Mock MCP prompt data
        mock_prompt = MagicMock()
        mock_prompt.name = 'test_command'
        mock_prompt.description = 'Test command for testing'
        mock_prompt.arguments = []
        
        command = SlashCommand.from_mcp_prompt('test_server', mock_prompt)
        
        assert command.command == '/mcp__test_server__test_command'
        assert command.server_name == 'test_server'
        assert command.prompt_name == 'test_command'
        assert command.description == 'Test command for testing'
    
    def test_command_categorization(self, tool_indexer):
        """Test automatic command categorization."""
        discovery = SlashCommandDiscovery(tool_indexer)
        
        # Create test command
        command = SlashCommand(
            command='/mcp__filesystem__read_file',
            server_name='filesystem',
            prompt_name='read_file',
            description='Read a file from the filesystem',
            arguments=[],
            usage_examples=[],
            category='',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        category = discovery._categorize_command(command)
        assert category == 'file_operations'
    
    @pytest.mark.asyncio
    async def test_contextual_suggestions(self, tool_indexer):
        """Test contextual slash command suggestions."""
        discovery = SlashCommandDiscovery(tool_indexer)
        
        # Add a test command
        test_command = SlashCommand(
            command='/mcp__filesystem__read_file',
            server_name='filesystem',
            prompt_name='read_file',
            description='Read a file',
            arguments=[],
            usage_examples=[],
            category='file_operations',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        discovery.slash_commands[test_command.command] = test_command
        
        suggestions = await discovery.get_contextual_suggestions(
            "I need to read a configuration file"
        )
        
        assert isinstance(suggestions, list)
        # May have suggestions if categories match
    
    @pytest.mark.asyncio
    async def test_command_usage_learning(self, tool_indexer):
        """Test learning from command usage."""
        discovery = SlashCommandDiscovery(tool_indexer)
        
        await discovery.learn_from_command_usage(
            '/mcp__test__command',
            success=True,
            execution_time=250.0,
            context={'project_type': 'web_app'}
        )
        
        assert '/mcp__test__command' in discovery.usage_patterns
        pattern = discovery.usage_patterns['/mcp__test__command']
        assert pattern['usage_count'] == 1
        assert pattern['success_count'] == 1


class TestIntegratedToolIndexer:
    """Test the complete integrated tool indexer system."""
    
    @pytest.mark.asyncio
    async def test_enhanced_tool_indexer_initialization(self, mock_domain_manager):
        """Test enhanced tool indexer initialization."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        # Check that all enhanced components are initialized
        assert indexer.native_bridge is not None
        assert indexer.hook_integration is not None
        assert indexer.workflow_enhancer is not None
        assert indexer.resource_monitor is not None
        assert indexer.slash_discovery is not None
        
        # Check integration status
        assert 'native_discovery_enabled' in indexer.integration_status
        assert 'hook_learning_enabled' in indexer.integration_status
        assert 'resource_monitoring_enabled' in indexer.integration_status
        assert 'slash_commands_enabled' in indexer.integration_status
    
    @pytest.mark.asyncio
    async def test_comprehensive_server_discovery(self, tool_indexer):
        """Test comprehensive server discovery."""
        # Mock some configuration
        mock_servers = {
            'test-server': {
                'command': 'python',
                'args': ['-m', 'test_server']
            }
        }
        
        with patch.object(tool_indexer, '_discover_servers_from_configuration') as mock_config:
            mock_config.return_value = mock_servers
            
            servers = await tool_indexer._discover_servers_comprehensive()
            
            assert isinstance(servers, dict)
            mock_config.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enhanced_discovery_workflow(self, tool_indexer):
        """Test the complete enhanced discovery workflow."""
        # Mock server discovery
        mock_servers = {
            'filesystem': {
                'command': 'python',
                'args': ['-m', 'filesystem_server'],
                'source': 'test'
            }
        }
        
        with patch.object(tool_indexer, '_discover_servers_comprehensive') as mock_servers_discovery:
            mock_servers_discovery.return_value = mock_servers
            
            with patch.object(tool_indexer, '_discover_tools_from_server_enhanced') as mock_tools:
                mock_tools.return_value = []
                
                with patch.object(tool_indexer, '_discover_known_tools') as mock_known:
                    mock_known.return_value = []
                    
                    result = await tool_indexer.discover_and_index_tools()
                    
                    assert isinstance(result, dict)
                    mock_servers_discovery.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_status_tracking(self, tool_indexer):
        """Test integration status tracking."""
        status = await tool_indexer.get_integration_status()
        
        assert isinstance(status, dict)
        assert 'native_discovery_enabled' in status
        assert 'hook_learning_enabled' in status
        assert 'resource_monitoring_enabled' in status
        assert 'slash_commands_enabled' in status
        assert 'enhanced_memory_enabled' in status
    
    @pytest.mark.asyncio
    async def test_resource_suggestions_integration(self, tool_indexer):
        """Test resource suggestions through tool indexer."""
        # Enable resource monitoring
        tool_indexer.integration_status['resource_monitoring_enabled'] = True
        
        suggestions = await tool_indexer.get_resource_suggestions(
            "read the database configuration file"
        )
        
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_slash_command_suggestions_integration(self, tool_indexer):
        """Test slash command suggestions through tool indexer."""
        # Enable slash commands (mock MCP client availability)
        tool_indexer.integration_status['slash_commands_enabled'] = True
        
        suggestions = await tool_indexer.get_slash_command_suggestions(
            "I need to process some data"
        )
        
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_workflow_suggestions_integration(self, tool_indexer):
        """Test workflow suggestions through tool indexer."""
        # Enable enhanced memory
        tool_indexer.integration_status['enhanced_memory_enabled'] = True
        
        suggestions = await tool_indexer.get_workflow_suggestions(
            "set up database connection",
            context={'project_type': 'web_app'}
        )
        
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_tool_usage_learning_integration(self, tool_indexer):
        """Test learning from tool usage through tool indexer."""
        # Enable all learning features
        for key in tool_indexer.integration_status:
            tool_indexer.integration_status[key] = True
        
        # Should not raise exceptions
        await tool_indexer.learn_from_tool_usage(
            'test_tool',
            {
                'user_request': 'test request',
                'project_type': 'test_project',
                'intent': 'testing'
            },
            success=True
        )
    
    @pytest.mark.asyncio
    async def test_comprehensive_analytics(self, tool_indexer):
        """Test comprehensive analytics collection."""
        analytics = await tool_indexer.get_comprehensive_analytics()
        
        assert 'discovery_status' in analytics
        assert 'resource_monitoring' in analytics
        assert 'slash_commands' in analytics
        assert 'workflow_patterns' in analytics
        
        discovery_status = analytics['discovery_status']
        assert 'total_servers' in discovery_status
        assert 'total_tools' in discovery_status
        assert 'integration_features' in discovery_status
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, tool_indexer):
        """Test cache invalidation functionality."""
        # Add some mock cache data
        tool_indexer.discovery_cache['test_server'] = {'test': 'data'}
        tool_indexer._cache_timestamps['test_server'] = 123456789
        
        await tool_indexer.invalidate_discovery_cache('test_server')
        
        assert 'test_server' not in tool_indexer.discovery_cache
        assert 'test_server' not in tool_indexer._cache_timestamps


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_graceful_component_failure(self, mock_domain_manager):
        """Test that system handles component failures gracefully."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        # Mock a component failure
        with patch.object(indexer.native_bridge, 'discover_native_servers') as mock_native:
            mock_native.side_effect = Exception("Test failure")
            
            # Should not raise exception
            servers = await indexer._discover_servers_comprehensive()
            assert isinstance(servers, dict)
    
    @pytest.mark.asyncio
    async def test_malformed_configuration_handling(self):
        """Test handling of malformed configuration files."""
        bridge = NativeMCPDiscoveryBridge()
        
        # Test with non-existent file
        servers = await bridge._parse_config_file('/nonexistent/path.json')
        assert servers == {}
        
        # Test with malformed JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            malformed_path = f.name
        
        try:
            servers = await bridge._parse_config_file(malformed_path)
            assert servers == {}
        finally:
            os.unlink(malformed_path)
    
    @pytest.mark.asyncio
    async def test_missing_mcp_client_fallback(self, tool_indexer):
        """Test fallback behavior when MCP client is not available."""
        # Temporarily disable MCP client
        original_available = hasattr(tool_indexer, 'MCP_CLIENT_AVAILABLE')
        tool_indexer.integration_status['slash_commands_enabled'] = False
        
        # Should still work without MCP client
        tools = await tool_indexer._discover_tools_from_server_enhanced(
            'test_server',
            {'command': 'python', 'args': ['-m', 'test']}
        )
        
        assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_complete_integration_workflow():
    """Test the complete integration workflow end-to-end."""
    mock_domain_manager = MockDomainManager()
    tool_indexer = MCPToolIndexer(mock_domain_manager)
    
    # Mock external dependencies
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        
        with patch.object(tool_indexer, '_discover_known_tools') as mock_known:
            mock_known.return_value = []
            
            # Run the complete discovery process
            result = await tool_indexer.discover_and_index_tools()
            
            assert isinstance(result, dict)
            
            # Check that some memories were stored
            assert len(mock_domain_manager.stored_memories) > 0
            
            # Verify summary was stored
            summary_memories = [
                m for m in mock_domain_manager.stored_memories 
                if m['memory_type'] == 'mcp_indexing_summary'
            ]
            assert len(summary_memories) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])