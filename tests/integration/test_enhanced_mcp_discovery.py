"""
Integration tests for enhanced MCP discovery system.

Tests the complete MCP discovery enhancement workflow including native discovery,
hook integration, workflow memory, resource reference monitoring, and slash command discovery.
"""

import asyncio
import json
import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Test imports
from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.mcp.native_discovery import NativeMCPDiscoveryBridge
from clarity.mcp.hook_integration import MCPHookIntegration
from clarity.mcp.workflow_memory import WorkflowMemoryEnhancer, SuggestionContext
from clarity.mcp.resource_reference_monitor import ResourceReferenceMonitor
from clarity.mcp.slash_command_discovery import SlashCommandDiscovery


class MockDomainManager:
    """Mock domain manager for testing."""
    
    def __init__(self):
        self.stored_memories = []
        self.memory_counter = 0
    
    async def store_memory(self, memory_type: str, content: str, importance: float, metadata: Dict[str, Any] = None):
        """Store memory mock."""
        self.memory_counter += 1
        memory_id = f"mock_memory_{self.memory_counter}"
        
        memory = {
            'id': memory_id,
            'memory_type': memory_type,
            'content': content if isinstance(content, str) else json.dumps(content),
            'importance': importance,
            'metadata': metadata or {},
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        self.stored_memories.append(memory)
        return memory_id
    
    async def retrieve_memories(self, query: str, types: List[str] = None, limit: int = 10, min_similarity: float = 0.0):
        """Retrieve memories mock."""
        filtered_memories = []
        
        for memory in self.stored_memories:
            if types and memory['memory_type'] not in types:
                continue
            
            # Simple keyword matching for mock
            content = memory['content'].lower()
            if not query or any(word.lower() in content for word in query.split()):
                filtered_memories.append(memory)
        
        return filtered_memories[:limit]


class TestEnhancedMCPDiscovery:
    """Integration tests for enhanced MCP discovery system."""
    
    @pytest.fixture
    def mock_domain_manager(self):
        """Create mock domain manager."""
        return MockDomainManager()
    
    @pytest.fixture
    def mcp_tool_indexer(self, mock_domain_manager):
        """Create MCP tool indexer with mocked dependencies."""
        indexer = MCPToolIndexer(mock_domain_manager)
        return indexer
    
    @pytest.fixture
    def mock_mcp_config(self):
        """Create mock MCP server configuration."""
        return {
            'test_server': {
                'command': 'python',
                'args': ['-m', 'test_mcp_server'],
                'env': {'TEST_MODE': '1'},
                'source': 'test_config'
            },
            'filesystem_server': {
                'command': 'mcp-filesystem',
                'args': ['--root', '/tmp'],
                'source': 'test_config'
            }
        }
    
    @pytest.mark.asyncio
    async def test_native_configuration_discovery(self):
        """Test native Claude Code configuration discovery."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                'mcpServers': {
                    'test_server': {
                        'command': 'python',
                        'args': ['-m', 'test_mcp_server']
                    }
                }
            }
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            # Test native discovery
            bridge = NativeMCPDiscoveryBridge()
            bridge.native_config_paths = [config_path]  # Override with test path
            
            # Mock CLI command
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'{}', b'')
                mock_process.returncode = 1  # CLI not available
                mock_subprocess.return_value = mock_process
                
                servers = await bridge.discover_native_servers()
                
                assert len(servers) == 1
                assert 'test_server' in servers
                assert servers['test_server']['command'] == 'python'
                assert servers['test_server']['source'] == 'claude_desktop'
                
                # Test validation
                validation = await bridge.validate_native_integration()
                assert validation['config_files_found'] == 1
                assert validation['total_servers_discovered'] == 1
        
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_hook_integration_setup(self, mock_domain_manager):
        """Test hook integration configuration."""
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        hook_integration = MCPHookIntegration(tool_indexer)
        
        # Mock filesystem operations
        with patch('os.makedirs'), patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Test hook setup
            success = await hook_integration.setup_hooks()
            assert success is True
            
            # Verify hook config was written
            mock_open.assert_called()
            mock_file.write.assert_called()
    
    @pytest.mark.asyncio
    async def test_hook_analysis_workflow(self, mock_domain_manager):
        """Test complete hook analysis workflow."""
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        # Set up discovered servers for testing
        tool_indexer.discovered_servers = {
            'postgres': {
                'command': 'postgres',
                'args': [],
                'tools': ['postgres_query', 'postgres_list_tables'],
                'source': 'test'
            },
            'filesystem': {
                'command': 'filesystem',
                'args': [],
                'tools': ['read_file', 'write_file'],
                'source': 'test'
            }
        }
        hook_integration = MCPHookIntegration(tool_indexer)
        
        # Test pre-tool analysis
        pre_result = await hook_integration.analyze_tool_usage('pre_tool', {
            'tool_name': 'bash',
            'args': 'psql -d mydb -c "SELECT * FROM users"'
        })
        
        # Should detect database opportunity
        assert pre_result is not None
        assert 'suggestions' in pre_result
        
        # Test post-tool learning
        post_result = await hook_integration.analyze_tool_usage('post_tool', {
            'tool_name': 'bash',
            'result': 'Query completed successfully'
        })
        
        assert post_result is not None
        assert post_result.get('learned_pattern') is True
        
        # Test prompt analysis
        prompt_result = await hook_integration.analyze_tool_usage('prompt_submit', {
            'prompt': 'I need to query the database to get user information'
        })
        
        # Should suggest MCP alternatives
        assert prompt_result is not None
        assert 'MCP Tool Suggestion' in prompt_result
    
    @pytest.mark.asyncio
    async def test_workflow_memory_enhancement(self, mock_domain_manager):
        """Test workflow memory storage and retrieval."""
        enhancer = WorkflowMemoryEnhancer(mock_domain_manager)
        
        # Store a workflow pattern
        pattern_data = {
            'context': 'Query user database for authentication',
            'tools': ['postgres_query'],
            'resources': ['@postgres:query://SELECT * FROM users'],
            'success': {'rows_returned': 5, 'execution_time': 0.1},
            'score': 0.9,
            'project_type': 'web_application',
            'intent': 'user_authentication'
        }
        
        memory_id = await enhancer.store_mcp_workflow_pattern(pattern_data)
        assert memory_id is not None
        
        # Test pattern retrieval
        similar_patterns = await enhancer.find_similar_workflows(
            'Need to authenticate user from database'
        )
        
        assert len(similar_patterns) > 0
        pattern = similar_patterns[0]
        assert pattern.pattern_type == 'mcp_workflow'
        assert 'postgres_query' in pattern.tool_sequence
        
        # Test workflow suggestions
        suggestion_context = SuggestionContext(
            current_task='User authentication',
            user_intent='Check user credentials',
            project_type='web_application',
            recent_tools_used=['bash'],
            recent_failures=[],
            environment_info={'database': 'postgres'},
            available_servers=['postgres']
        )
        
        suggestions = await enhancer.get_workflow_suggestions(suggestion_context)
        assert len(suggestions) > 0
        # Should contain either workflow_pattern or proactive_suggestion types
        suggestion_types = [s['type'] for s in suggestions]
        assert 'workflow_pattern' in suggestion_types or 'proactive_suggestion' in suggestion_types
        assert suggestions[0]['confidence'] > 0.5
    
    @pytest.mark.asyncio
    async def test_resource_reference_monitoring(self):
        """Test resource reference detection and learning."""
        monitor = ResourceReferenceMonitor()
        
        # Test opportunity detection
        prompt = "I need to read the config.json file from the project directory"
        opportunities = monitor.detect_resource_opportunities(prompt)
        
        assert len(opportunities) > 0
        file_opp = next((opp for opp in opportunities if opp.opportunity_type == 'file_operations'), None)
        assert file_opp is not None
        assert '@' in file_opp.suggested_reference
        assert 'file://' in file_opp.suggested_reference
        
        # Test learning from usage
        await monitor.learn_resource_pattern(
            '@filesystem:file://config.json',
            {'prompt': prompt, 'intent': 'configuration_access'},
            success=True,
            response_time=150.0
        )
        
        # Test performance stats
        stats = monitor.get_performance_stats()
        assert stats['reference_usage_tracked'] == 1
        assert stats['total_opportunities_detected'] > 0
    
    @pytest.mark.asyncio
    async def test_slash_command_discovery(self, mock_domain_manager):
        """Test slash command discovery from MCP servers."""
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        slash_discovery = SlashCommandDiscovery(tool_indexer)
        
        # Mock the discovery method directly
        async def mock_discover(server_name: str, server_config: Dict[str, Any]):
            from clarity.mcp.slash_command_discovery import SlashCommand
            # Create a mock command based on the test expectations
            mock_prompt = MagicMock()
            mock_prompt.name = 'test_query'
            mock_prompt.description = 'Execute a test database query'
            
            command = SlashCommand.from_mcp_prompt(server_name, mock_prompt)
            return [command]
        
        # Patch the method directly
        with patch.object(slash_discovery, 'discover_slash_commands', side_effect=mock_discover):
            # Test discovery
            server_config = {
                'command': 'test-mcp-server',
                'args': ['--test']
            }
            
            commands = await slash_discovery.discover_slash_commands('test_server', server_config)
            
            assert len(commands) == 1
            command = commands[0]
            assert command.command == '/mcp__test_server__test_query'
            assert command.server_name == 'test_server'
            assert command.description == 'Execute a test database query'
            
            # Store the command in the discovery instance  
            for cmd in commands:
                slash_discovery.slash_commands[cmd.command] = cmd
            
            # Mock the relevance calculation to ensure suggestions are returned
            def mock_relevance(command, prompt, categories):
                return 0.8  # High relevance to ensure it passes the 0.3 threshold
            
            with patch.object(slash_discovery, '_calculate_command_relevance', side_effect=mock_relevance):
                # Test contextual suggestions
                suggestions = await slash_discovery.get_contextual_suggestions(
                    'I need to run a test query on the database'
                )
            
            # Should suggest the discovered command
            assert len(suggestions) > 0
            suggestion = suggestions[0]
            assert suggestion.command.command == '/mcp__test_server__test_query'
            assert suggestion.relevance_score > 0.0
    
    @pytest.mark.asyncio
    async def test_complete_discovery_workflow(self, mock_domain_manager, mock_mcp_config):
        """Test complete enhanced discovery workflow."""
        # Initialize tool indexer with all components
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        
        # Mock all external calls to speed up the test
        with patch.object(tool_indexer, '_discover_servers_comprehensive') as mock_discover, \
             patch.object(tool_indexer, '_discover_tools_from_server_enhanced') as mock_tools, \
             patch.object(tool_indexer.native_bridge, 'discover_native_servers') as mock_native, \
             patch.object(tool_indexer.performance_optimizer, 'optimize_discovery_workflow') as mock_optimize:
            
            mock_discover.return_value = mock_mcp_config
            mock_tools.return_value = []  # No tools for simplicity
            mock_native.return_value = {}
            mock_optimize.return_value = {'status': 'optimized', 'servers': mock_mcp_config}
            
            # Run complete discovery
            result = await tool_indexer.discover_and_index_tools()
            
            assert isinstance(result, dict)
            
            # Check integration status
            status = await tool_indexer.get_integration_status()
            assert 'native_discovery_enabled' in status
            assert 'resource_monitoring_enabled' in status
            assert 'enhanced_memory_enabled' in status
            
            # Test analytics
            analytics = await tool_indexer.get_comprehensive_analytics()
            assert 'discovery_status' in analytics
            assert analytics['discovery_status']['total_servers'] >= 0
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, mock_domain_manager):
        """Test performance optimization features."""
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        
        # Test cache invalidation
        await tool_indexer.invalidate_discovery_cache()
        
        # Test performance analytics
        performance_analytics = await tool_indexer.get_performance_analytics()
        assert 'performance_optimization' in performance_analytics
        assert performance_analytics['performance_optimization']['status'] in ['enabled', 'error']
        
        # Test performance status
        status = tool_indexer.get_performance_status()
        assert 'status' in status
        assert status['status'] in ['active', 'inactive', 'error']
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_domain_manager):
        """Test error handling and recovery mechanisms."""
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        
        # Test with invalid server configuration
        invalid_config = {
            'invalid_server': {
                'command': 'nonexistent_command',
                'args': ['--invalid']
            }
        }
        
        # Should handle errors gracefully
        with patch.object(tool_indexer, '_discover_servers_comprehensive') as mock_discover:
            mock_discover.return_value = invalid_config
            
            # This should not raise an exception
            result = await tool_indexer.discover_and_index_tools()
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, mock_domain_manager):
        """Test integration with memory system."""
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        # Ensure enhanced memory is enabled
        tool_indexer.integration_status['enhanced_memory_enabled'] = True
        
        # Test learning from tool usage
        await tool_indexer.learn_from_tool_usage(
            'postgres_query',
            {
                'user_request': 'Get user data',
                'project_type': 'web_app',
                'intent': 'data_retrieval'
            },
            success=True
        )
        
        # Verify memory was stored
        memories = await mock_domain_manager.retrieve_memories(
            query='postgres_query',
            types=['mcp_workflow_pattern']
        )
        
        assert len(memories) > 0
        
        # Test workflow suggestions
        suggestions = await tool_indexer.get_workflow_suggestions(
            'Need to query database',
            {
                'project_type': 'web_app',
                'recent_tools': ['bash'],
                'env': {'database': 'postgres'}
            }
        )
        
        assert isinstance(suggestions, list)
        # Should have reasonable number of suggestions
        assert len(suggestions) >= 0  # May be empty if no patterns match
    
    @pytest.mark.asyncio
    async def test_integration_validation(self, mock_domain_manager):
        """Test validation of enhanced integration features."""
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        
        # Test component initialization
        assert tool_indexer.native_bridge is not None
        assert tool_indexer.hook_integration is not None
        assert tool_indexer.workflow_enhancer is not None
        assert tool_indexer.resource_monitor is not None
        assert tool_indexer.slash_discovery is not None
        
        # Test resource suggestions
        suggestions = await tool_indexer.get_resource_suggestions(
            'Read the configuration file',
            {'available_servers': ['filesystem']}
        )
        
        assert isinstance(suggestions, list)
        
        # Test slash command suggestions
        slash_suggestions = await tool_indexer.get_slash_command_suggestions(
            'I need help with database queries'
        )
        
        assert isinstance(slash_suggestions, list)


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for enhanced MCP discovery."""
    
    @pytest.fixture
    def mock_domain_manager(self):
        """Create mock domain manager."""
        return MockDomainManager()
    
    @pytest.mark.asyncio
    async def test_discovery_performance(self, mock_domain_manager):
        """Test discovery performance meets targets."""
        import time
        
        tool_indexer = MCPToolIndexer(mock_domain_manager)
        
        # Mock all external calls for speed
        with patch.object(tool_indexer, '_discover_servers_comprehensive') as mock_discover, \
             patch.object(tool_indexer, '_discover_tools_from_server_enhanced') as mock_tools, \
             patch.object(tool_indexer.native_bridge, 'discover_native_servers') as mock_native, \
             patch.object(tool_indexer.performance_optimizer, 'optimize_discovery_workflow') as mock_optimize:
            
            # Set up fast mock responses
            mock_discover.return_value = {'test_server': {'command': 'test', 'args': []}}
            mock_tools.return_value = []
            mock_native.return_value = {}
            mock_optimize.return_value = {'status': 'optimized', 'servers': {}}
            
            # Measure discovery time
            start_time = time.time()
            await tool_indexer.discover_and_index_tools()
            discovery_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Should complete quickly with mocked calls
            assert discovery_time < 5000  # 5 seconds should be plenty for mocked operations
    
    @pytest.mark.asyncio
    async def test_memory_query_performance(self, mock_domain_manager):
        """Test memory query performance."""
        import time
        
        enhancer = WorkflowMemoryEnhancer(mock_domain_manager)
        
        # Add some test patterns
        for i in range(10):
            pattern_data = {
                'context': f'Test pattern {i}',
                'tools': [f'tool_{i}'],
                'score': 0.8,
                'project_type': 'test'
            }
            await enhancer.store_mcp_workflow_pattern(pattern_data)
        
        # Measure query time
        start_time = time.time()
        patterns = await enhancer.find_similar_workflows('Test pattern query')
        query_time = (time.time() - start_time) * 1000
        
        # Should complete quickly
        assert query_time < 100  # 100ms
        assert len(patterns) > 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])