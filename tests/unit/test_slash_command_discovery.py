"""
Unit tests for SlashCommandDiscovery.

Tests the slash command discovery and suggestion system in isolation.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from clarity.mcp.slash_command_discovery import (
    SlashCommandDiscovery,
    SlashCommand,
    SlashCommandSuggestion
)


class MockToolIndexer:
    """Mock tool indexer for testing."""
    
    def __init__(self):
        self.domain_manager = MockDomainManager()
        self.discovered_servers = {
            'test_server': {
                'command': 'python',
                'args': ['-m', 'test_server']
            }
        }
    
    async def get_discovered_servers(self):
        """Get discovered servers."""
        return self.discovered_servers


class MockDomainManager:
    """Mock domain manager for testing."""
    
    def __init__(self):
        self.stored_memories = []
    
    async def store_memory(self, memory_type: str, content: str, importance: float, metadata: Dict[str, Any] = None):
        """Store memory mock."""
        memory_id = f"mock_memory_{len(self.stored_memories)}"
        self.stored_memories.append({
            'id': memory_id,
            'memory_type': memory_type,
            'content': content,
            'importance': importance,
            'metadata': metadata or {}
        })
        return memory_id


class MockPrompt:
    """Mock MCP prompt for testing."""
    
    def __init__(self, name: str, description: str = '', arguments: List[Dict] = None):
        self.name = name
        self.description = description
        self.arguments = arguments or []


class MockPromptsResult:
    """Mock prompts result for testing."""
    
    def __init__(self, prompts: List[MockPrompt]):
        self.prompts = prompts


class TestSlashCommandDiscovery:
    """Unit tests for slash command discovery."""
    
    @pytest.fixture
    def mock_tool_indexer(self):
        """Create mock tool indexer."""
        return MockToolIndexer()
    
    @pytest.fixture
    def discovery(self, mock_tool_indexer):
        """Create slash command discovery with mocked dependencies."""
        return SlashCommandDiscovery(mock_tool_indexer)
    
    @pytest.fixture
    def sample_server_config(self):
        """Sample server configuration."""
        return {
            'command': 'python',
            'args': ['-m', 'test_server'],
            'env': {'TEST_MODE': '1'}
        }
    
    def test_slash_command_creation_from_prompt(self):
        """Test creating SlashCommand from MCP prompt data."""
        # Test with simple prompt
        simple_prompt = MockPrompt(
            name='test_query',
            description='Execute a test query'
        )
        
        command = SlashCommand.from_mcp_prompt('test_server', simple_prompt)
        
        assert command.command == '/mcp__test_server__test_query'
        assert command.server_name == 'test_server'
        assert command.prompt_name == 'test_query'
        assert command.description == 'Execute a test query'
        assert len(command.arguments) == 0
        assert command.category == 'mcp_prompt'
        assert command.confidence == 1.0
        assert command.usage_count == 0
    
    def test_slash_command_with_arguments(self):
        """Test creating SlashCommand with arguments."""
        # Mock argument structure
        arg_data = MagicMock()
        arg_data.name = 'query'
        arg_data.description = 'SQL query to execute'
        arg_data.required = True
        arg_data.type = 'string'
        
        prompt_with_args = MockPrompt(
            name='database_query',
            description='Execute database query',
            arguments=[arg_data]
        )
        
        command = SlashCommand.from_mcp_prompt('postgres', prompt_with_args)
        
        assert len(command.arguments) == 1
        assert command.arguments[0]['name'] == 'query'
        assert command.arguments[0]['description'] == 'SQL query to execute'
        assert command.arguments[0]['required'] is True
        assert command.arguments[0]['type'] == 'string'
    
    def test_slash_command_usage_example_generation(self):
        """Test usage example generation for commands."""
        # Command with no arguments
        simple_command = SlashCommand(
            command='/mcp__test__simple',
            server_name='test',
            prompt_name='simple',
            description='Simple command',
            arguments=[],
            usage_examples=[],
            category='utility',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        assert simple_command.generate_usage_example() == '/mcp__test__simple'
        
        # Command with arguments
        complex_command = SlashCommand(
            command='/mcp__db__query',
            server_name='db',
            prompt_name='query',
            description='Database query',
            arguments=[
                {'name': 'sql', 'type': 'string', 'required': True},
                {'name': 'limit', 'type': 'integer', 'required': False}
            ],
            usage_examples=[],
            category='database',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        example = complex_command.generate_usage_example()
        assert '--sql <string>' in example
        assert '[--limit <integer>]' in example  # Optional argument in brackets
    
    def test_command_categorization(self, discovery):
        """Test command categorization based on content."""
        test_cases = [
            # File operations
            {
                'name': 'read_file',
                'description': 'Read file contents',
                'expected_category': 'file_operations'
            },
            # Database operations
            {
                'name': 'execute_query',
                'description': 'Execute SQL query on database',
                'expected_category': 'database'
            },
            # Web operations
            {
                'name': 'fetch_data',
                'description': 'Fetch data from HTTP API endpoint',
                'expected_category': 'web_requests'
            },
            # Git operations
            {
                'name': 'git_status',
                'description': 'Check git repository status',
                'expected_category': 'git_operations'
            },
            # Default category
            {
                'name': 'unknown_command',
                'description': 'Does something mysterious',
                'expected_category': 'utility'
            }
        ]
        
        for test_case in test_cases:
            command = SlashCommand(
                command=f"/mcp__test__{test_case['name']}",
                server_name='test',
                prompt_name=test_case['name'],
                description=test_case['description'],
                arguments=[],
                usage_examples=[],
                category='',  # Will be set by categorization
                confidence=1.0,
                last_discovered='2024-01-01T00:00:00Z',
                usage_count=0
            )
            
            category = discovery._categorize_command(command)
            assert category == test_case['expected_category']
    
    @pytest.mark.asyncio
    @patch('clarity.mcp.slash_command_discovery.MCP_CLIENT_AVAILABLE', True)
    @patch('clarity.mcp.slash_command_discovery.ClientSession')
    @patch('clarity.mcp.slash_command_discovery.StdioClientTransport')
    async def test_discover_slash_commands_success(self, mock_transport, mock_client_session, discovery, sample_server_config):
        """Test successful discovery of slash commands."""
        # Mock transport
        mock_transport_instance = MagicMock()
        mock_transport.return_value = mock_transport_instance
        
        # Mock MCP client session
        mock_session_instance = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Mock prompts
        test_prompts = [
            MockPrompt('query_data', 'Query database for data'),
            MockPrompt('read_file', 'Read file contents')
        ]
        mock_prompts_result = MockPromptsResult(test_prompts)
        
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.list_prompts = AsyncMock(return_value=mock_prompts_result)
        
        # Discover commands
        commands = await discovery.discover_slash_commands('test_server', sample_server_config)
        
        assert len(commands) == 2
        
        # Check first command
        command1 = commands[0]
        assert command1.command == '/mcp__test_server__query_data'
        assert command1.server_name == 'test_server'
        assert command1.prompt_name == 'query_data'
        assert command1.description == 'Query database for data'
        
        # Check second command
        command2 = commands[1]
        assert command2.command == '/mcp__test_server__read_file'
        assert command2.prompt_name == 'read_file'
        
        # Verify commands were stored in registry
        assert '/mcp__test_server__query_data' in discovery.slash_commands
        assert '/mcp__test_server__read_file' in discovery.slash_commands
    
    @pytest.mark.asyncio
    @patch('clarity.mcp.slash_command_discovery.MCP_CLIENT_AVAILABLE', False)
    async def test_discover_slash_commands_no_client(self, discovery, sample_server_config):
        """Test discovery when MCP client is not available."""
        commands = await discovery.discover_slash_commands('test_server', sample_server_config)
        
        assert len(commands) == 0
    
    @pytest.mark.asyncio
    @patch('clarity.mcp.slash_command_discovery.MCP_CLIENT_AVAILABLE', True)
    @patch('clarity.mcp.slash_command_discovery.ClientSession')
    async def test_discover_slash_commands_timeout(self, mock_client_session, discovery, sample_server_config):
        """Test discovery with connection timeout."""
        mock_session_instance = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Mock timeout
        mock_session_instance.initialize = AsyncMock(side_effect=asyncio.TimeoutError())
        
        commands = await discovery.discover_slash_commands('test_server', sample_server_config)
        
        assert len(commands) == 0
    
    @pytest.mark.asyncio
    @patch('clarity.mcp.slash_command_discovery.MCP_CLIENT_AVAILABLE', True)
    @patch('clarity.mcp.slash_command_discovery.ClientSession')
    async def test_discover_slash_commands_connection_error(self, mock_client_session, discovery, sample_server_config):
        """Test discovery with connection error."""
        mock_session_instance = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Mock connection error
        mock_session_instance.initialize = AsyncMock(side_effect=Exception("Connection failed"))
        
        commands = await discovery.discover_slash_commands('test_server', sample_server_config)
        
        assert len(commands) == 0
    
    @pytest.mark.asyncio
    @patch('clarity.mcp.slash_command_discovery.MCP_CLIENT_AVAILABLE', True)
    async def test_caching_mechanism(self, discovery, sample_server_config):
        """Test command discovery caching."""
        # Mock the cache to be valid initially (empty)
        discovery.discovery_cache = {}
        discovery._cache_timestamps = {}
        
        with patch.object(discovery, '_connect_and_discover') as mock_connect:
            mock_connect.return_value = [
                SlashCommand(
                    command='/mcp__test__cached',
                    server_name='test',
                    prompt_name='cached',
                    description='Cached command',
                    arguments=[],
                    usage_examples=[],
                    category='utility',
                    confidence=1.0,
                    last_discovered='2024-01-01T00:00:00Z',
                    usage_count=0
                )
            ]
            
            # First call - should hit the actual discovery
            commands1 = await discovery.discover_slash_commands('test_server', sample_server_config)
            assert len(commands1) == 1
            assert mock_connect.call_count == 1
            
            # Second call - should use cache
            commands2 = await discovery.discover_slash_commands('test_server', sample_server_config)
            assert len(commands2) == 1
            assert mock_connect.call_count == 1  # No additional calls
            
            # Commands should be identical
            assert commands1[0].command == commands2[0].command
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, discovery):
        """Test cache invalidation functionality."""
        # Add some fake cache entries
        discovery.discovery_cache['test_server:123'] = []
        discovery.discovery_cache['other_server:456'] = []
        discovery._cache_timestamps['test_server:123'] = 1000
        discovery._cache_timestamps['other_server:456'] = 2000
        
        # Invalidate specific server
        discovery.invalidate_cache('test_server')
        
        # test_server cache should be cleared
        assert not any(key.startswith('test_server') for key in discovery.discovery_cache.keys())
        assert not any(key.startswith('test_server') for key in discovery._cache_timestamps.keys())
        
        # other_server cache should remain
        assert 'other_server:456' in discovery.discovery_cache
        assert 'other_server:456' in discovery._cache_timestamps
        
        # Invalidate all
        discovery.invalidate_cache()
        
        assert len(discovery.discovery_cache) == 0
        assert len(discovery._cache_timestamps) == 0
    
    @pytest.mark.asyncio
    async def test_store_slash_commands(self, discovery):
        """Test storing discovered slash commands as memories."""
        commands = [
            SlashCommand(
                command='/mcp__test__query',
                server_name='test',
                prompt_name='query',
                description='Execute query',
                arguments=[{'name': 'sql', 'type': 'string', 'required': True}],
                usage_examples=[],
                category='database',
                confidence=1.0,
                last_discovered='2024-01-01T00:00:00Z',
                usage_count=0
            )
        ]
        
        await discovery.store_slash_commands(commands)
        
        # Verify memory was stored
        stored_memories = discovery.tool_indexer.domain_manager.stored_memories
        assert len(stored_memories) == 1
        
        stored_memory = stored_memories[0]
        assert stored_memory['memory_type'] == 'mcp_slash_command'
        assert stored_memory['importance'] == 0.8
        
        metadata = stored_memory['metadata']
        assert metadata['server'] == 'test'
        assert metadata['prompt_name'] == 'query'
        assert metadata['command_category'] == 'database'
        assert metadata['arguments_count'] == 1
        assert metadata['has_description'] is True
    
    @pytest.mark.asyncio
    async def test_contextual_suggestions(self, discovery):
        """Test contextual slash command suggestions."""
        # Add some commands to the registry
        database_command = SlashCommand(
            command='/mcp__postgres__query',
            server_name='postgres',
            prompt_name='query',
            description='Execute SQL database query',
            arguments=[],
            usage_examples=[],
            category='database',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        file_command = SlashCommand(
            command='/mcp__filesystem__read',
            server_name='filesystem',
            prompt_name='read',
            description='Read file contents from filesystem',
            arguments=[],
            usage_examples=[],
            category='file_operations',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        discovery.slash_commands[database_command.command] = database_command
        discovery.slash_commands[file_command.command] = file_command
        
        # Mock relevance calculation to return different scores for different commands
        def mock_relevance(command, prompt, categories):
            if 'database' in command.category and 'database' in prompt.lower():
                return 0.8
            elif 'file' in command.category and 'file' in prompt.lower():
                return 0.8
            else:
                return 0.2
                
        with patch.object(discovery, '_calculate_command_relevance', side_effect=mock_relevance):
            # Test database-related prompt
            db_suggestions = await discovery.get_contextual_suggestions(
                'I need to query the database to get user information',
                context={'available_servers': ['postgres', 'filesystem']}
            )
            
            assert len(db_suggestions) > 0
        
            # Top suggestion should be database-related
            top_suggestion = db_suggestions[0]
            assert isinstance(top_suggestion, SlashCommandSuggestion)
            assert top_suggestion.command.category == 'database'
            assert top_suggestion.relevance_score > 0.3
            assert 'database' in top_suggestion.reason.lower() or 'query' in top_suggestion.reason.lower()
            
            # Test file-related prompt
            file_suggestions = await discovery.get_contextual_suggestions(
                'I need to read the configuration file',
                context={'available_servers': ['filesystem']}
            )
            
            assert len(file_suggestions) > 0
            file_suggestion = file_suggestions[0]
            assert file_suggestion.command.category == 'file_operations'
    
    def test_prompt_category_analysis(self, discovery):
        """Test analysis of prompt categories."""
        test_cases = [
            {
                'prompt': 'I need to read a file and write some data',
                'expected_categories': ['file_operations']
            },
            {
                'prompt': 'Execute SQL query on the database to get user data',
                'expected_categories': ['database']
            },
            {
                'prompt': 'Make HTTP request to the API endpoint',
                'expected_categories': ['web_requests']
            },
            {
                'prompt': 'Check git repository status and commit changes',
                'expected_categories': ['git_operations']
            },
            {
                'prompt': 'No specific keywords here',
                'expected_categories': []
            }
        ]
        
        for test_case in test_cases:
            categories = discovery._analyze_prompt_categories(test_case['prompt'])
            
            if test_case['expected_categories']:
                for expected_cat in test_case['expected_categories']:
                    assert expected_cat in categories
                    assert categories[expected_cat] > 0
            else:
                assert len(categories) == 0
    
    def test_command_relevance_calculation(self, discovery):
        """Test calculation of command relevance to prompts."""
        database_command = SlashCommand(
            command='/mcp__postgres__query',
            server_name='postgres',
            prompt_name='query',
            description='Execute SQL database query',
            arguments=[],
            usage_examples=[],
            category='database',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        # Test high relevance
        high_relevance = discovery._calculate_command_relevance(
            database_command,
            'I need to execute a SQL query on the database',
            {'database': 0.8}
        )
        assert high_relevance > 0.5
        
        # Test low relevance
        low_relevance = discovery._calculate_command_relevance(
            database_command,
            'What is the weather today?',
            {}
        )
        assert low_relevance < 0.3
    
    @pytest.mark.asyncio
    async def test_context_match_calculation(self, discovery):
        """Test context matching for commands."""
        command = SlashCommand(
            command='/mcp__postgres__query',
            server_name='postgres',
            prompt_name='query',
            description='Execute database query',
            arguments=[],
            usage_examples=[],
            category='database',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        # Good context match
        good_context = {
            'project_type': 'web_application',
            'available_servers': ['postgres', 'filesystem'],
            'recent_tools_used': ['bash']
        }
        good_match = discovery._calculate_context_match(command, good_context)
        assert good_match >= 0.4  # Should have good match due to available server
        
        # Poor context match
        poor_context = {
            'project_type': 'frontend',
            'available_servers': ['filesystem'],
            'recent_tools_used': []
        }
        poor_match = discovery._calculate_context_match(command, poor_context)
        assert poor_match < good_match
    
    def test_argument_suggestion(self, discovery):
        """Test suggestion of command arguments based on context."""
        command = SlashCommand(
            command='/mcp__filesystem__read',
            server_name='filesystem',
            prompt_name='read',
            description='Read file contents',
            arguments=[
                {'name': 'file_path', 'type': 'string', 'required': True},
                {'name': 'limit', 'type': 'integer', 'required': False}
            ],
            usage_examples=[],
            category='file_operations',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        prompt = 'Please read the "config.json" file'
        context = {}
        
        suggested_args = discovery._suggest_command_arguments(command, prompt, context)
        
        # Should suggest the file path from the quoted string
        assert 'file_path' in suggested_args
        assert suggested_args['file_path'] == 'config.json'
    
    @pytest.mark.asyncio
    async def test_learn_from_command_usage(self, discovery):
        """Test learning from actual command usage."""
        command_name = '/mcp__test__query'
        
        # Record successful usage
        await discovery.learn_from_command_usage(
            command_name,
            success=True,
            execution_time=250.0,
            context={'project_type': 'web_app', 'user_intent': 'data_retrieval'}
        )
        
        # Verify pattern was recorded
        assert command_name in discovery.usage_patterns
        
        pattern = discovery.usage_patterns[command_name]
        assert pattern['usage_count'] == 1
        assert pattern['success_count'] == 1
        assert pattern['total_execution_time'] == 250.0
        assert len(pattern['contexts']) == 1
        assert pattern['contexts'][0]['project_type'] == 'web_app'
        assert pattern['contexts'][0]['success'] is True
        
        # Record failed usage
        await discovery.learn_from_command_usage(
            command_name,
            success=False,
            execution_time=5000.0,
            context={'project_type': 'web_app', 'user_intent': 'data_retrieval'}
        )
        
        # Verify updated pattern
        updated_pattern = discovery.usage_patterns[command_name]
        assert updated_pattern['usage_count'] == 2
        assert updated_pattern['success_count'] == 1  # Still 1 success
        assert updated_pattern['total_execution_time'] == 5250.0
    
    @pytest.mark.asyncio
    async def test_command_analytics(self, discovery):
        """Test command analytics generation."""
        # Add some commands and usage data
        command1 = SlashCommand(
            command='/mcp__server1__cmd1',
            server_name='server1',
            prompt_name='cmd1',
            description='Command 1',
            arguments=[],
            usage_examples=[],
            category='database',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=5
        )
        
        command2 = SlashCommand(
            command='/mcp__server2__cmd2',
            server_name='server2',
            prompt_name='cmd2',
            description='Command 2',
            arguments=[],
            usage_examples=[],
            category='file_operations',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=3
        )
        
        discovery.slash_commands[command1.command] = command1
        discovery.slash_commands[command2.command] = command2
        
        # Add usage patterns
        discovery.usage_patterns[command1.command] = {
            'usage_count': 5,
            'success_count': 4,
            'total_execution_time': 1000.0,
            'contexts': []
        }
        
        analytics = await discovery.get_command_analytics()
        
        assert analytics['total_commands_discovered'] == 2
        assert analytics['commands_by_server']['server1'] == 1
        assert analytics['commands_by_server']['server2'] == 1
        assert analytics['commands_by_category']['database'] == 1
        assert analytics['commands_by_category']['file_operations'] == 1
        
        # Check usage statistics
        assert command1.command in analytics['usage_statistics']
        usage_stat = analytics['usage_statistics'][command1.command]
        assert usage_stat['usage_count'] == 5
        assert usage_stat['success_rate'] == 0.8  # 4/5
        assert usage_stat['avg_execution_time'] == 200.0  # 1000/5
        
        # Check top commands
        assert len(analytics['top_commands']) >= 1
        top_command = analytics['top_commands'][0]
        assert top_command['command'] == command1.command
        assert top_command['usage_count'] == 5
    
    @pytest.mark.asyncio
    async def test_command_validation(self, discovery):
        """Test validation of discovered commands."""
        # Add a command to the registry
        command = SlashCommand(
            command='/mcp__test__validate',
            server_name='test',
            prompt_name='validate',
            description='Test validation',
            arguments=[],
            usage_examples=[],
            category='utility',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        discovery.slash_commands[command.command] = command
        
        # Mock tool indexer with server discovery
        with patch.object(discovery.tool_indexer, 'get_discovered_servers') as mock_get_servers:
            mock_get_servers.return_value = {
                'test': {'command': 'test_command', 'args': []}
            }
            
            with patch.object(discovery, '_connect_and_discover') as mock_connect:
                mock_connect.return_value = [command]  # Server is accessible
                
                validation = await discovery.validate_commands()
                
                assert validation['total_commands'] == 1
                assert validation['accessible_commands'] == 1
                assert validation['inaccessible_commands'] == 0
                assert 'test' in validation['server_status']
                assert validation['server_status']['test']['accessible'] is True
    
    def test_slash_command_dataclass_methods(self):
        """Test SlashCommand dataclass methods."""
        command = SlashCommand(
            command='/mcp__test__example',
            server_name='test',
            prompt_name='example',
            description='Example command',
            arguments=[{'name': 'arg1', 'type': 'string', 'required': True}],
            usage_examples=[],
            category='utility',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        # Test to_dict
        command_dict = command.to_dict()
        assert isinstance(command_dict, dict)
        assert command_dict['command'] == '/mcp__test__example'
        assert command_dict['server_name'] == 'test'
        
        # Test from_dict
        recreated_command = SlashCommand.from_dict(command_dict)
        assert recreated_command.command == command.command
        assert recreated_command.server_name == command.server_name
        assert recreated_command.arguments == command.arguments
    
    def test_slash_command_suggestion_dataclass(self):
        """Test SlashCommandSuggestion dataclass."""
        command = SlashCommand(
            command='/mcp__test__example',
            server_name='test',
            prompt_name='example',
            description='Example command',
            arguments=[],
            usage_examples=[],
            category='utility',
            confidence=1.0,
            last_discovered='2024-01-01T00:00:00Z',
            usage_count=0
        )
        
        suggestion = SlashCommandSuggestion(
            command=command,
            relevance_score=0.85,
            reason='Highly relevant for testing',
            context_match=0.9,
            suggested_arguments={'arg1': 'value1'}
        )
        
        assert suggestion.relevance_score == 0.85
        assert suggestion.context_match == 0.9
        assert suggestion.suggested_arguments['arg1'] == 'value1'
        
        # Test to_dict
        suggestion_dict = suggestion.to_dict()
        assert isinstance(suggestion_dict, dict)
        assert suggestion_dict['relevance_score'] == 0.85
        assert isinstance(suggestion_dict['command'], dict)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])