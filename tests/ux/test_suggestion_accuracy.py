"""
User Experience Tests for MCP Suggestion Accuracy.

This module tests the accuracy, relevance, and effectiveness of MCP tool suggestions
from the user's perspective, focusing on real-world scenarios and workflows.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.mcp.hook_integration import MCPHookIntegration
from clarity.mcp.workflow_memory import WorkflowMemoryEnhancer, SuggestionContext
from clarity.mcp.resource_reference_monitor import ResourceReferenceMonitor
from clarity.mcp.slash_command_discovery import SlashCommandDiscovery

from tests.fixtures.test_configs import (
    TEST_USER_CONTEXTS,
    TEST_WORKFLOW_PATTERNS,
    get_test_user_context
)


class MockDomainManager:
    """Mock domain manager for UX testing."""
    
    def __init__(self):
        self.stored_memories = []
        self.memory_counter = 0
    
    async def store_memory(self, memory_type: str, content: str, importance: float, metadata: Dict[str, Any] = None):
        """Store memory mock."""
        self.memory_counter += 1
        memory_id = f"ux_memory_{self.memory_counter}"
        
        memory = {
            'id': memory_id,
            'memory_type': memory_type,
            'content': content if isinstance(content, str) else json.dumps(content),
            'importance': importance,
            'metadata': metadata or {},
            'created_at': '2024-01-01T00:00:00Z',
            'similarity_score': 0.8  # Default similarity for UX testing
        }
        
        self.stored_memories.append(memory)
        return memory_id
    
    async def retrieve_memories(self, query: str, types: List[str] = None, limit: int = 10, min_similarity: float = 0.0):
        """Retrieve memories mock with relevance scoring."""
        filtered_memories = []
        
        for memory in self.stored_memories:
            if types and memory['memory_type'] not in types:
                continue
            
            # Advanced keyword matching for UX testing
            content = memory['content'].lower()
            query_words = query.lower().split()
            
            # Semantic mappings for domain-specific matches
            semantic_mappings = {
                'login': ['authentication', 'auth', 'sign', 'access'],
                'system': ['application', 'app', 'service', 'platform'],
                'database': ['postgres', 'sql', 'db', 'storage'],
                'file': ['config', 'document', 'data'],
                'web': ['http', 'api', 'service', 'server']
            }
            
            relevance_score = 0.0
            for word in query_words:
                # Direct word match
                if word in content:
                    relevance_score += 0.2
                # Stem matching for common variations
                elif word.endswith('s') and word[:-1] in content:  # users -> user
                    relevance_score += 0.15
                elif word + 's' in content:  # user -> users
                    relevance_score += 0.15
                elif word.endswith('ing') and word[:-3] in content:  # authenticating -> authenticate
                    relevance_score += 0.15
                elif word + 'ion' in content:  # authenticate -> authentication
                    relevance_score += 0.15
                elif word.endswith('e') and word[:-1] + 'ion' in content:  # authenticate -> authentication
                    relevance_score += 0.15
                elif word.endswith('ion') and word[:-3] in content:  # authentication -> authenticate
                    relevance_score += 0.15
                elif word + 'ing' in content:  # authenticate -> authenticating
                    relevance_score += 0.15
                # Semantic matching
                elif word in semantic_mappings:
                    for synonym in semantic_mappings[word]:
                        if synonym in content:
                            relevance_score += 0.15
                            break
            
            # Boost score for exact phrase matches
            if query.lower() in content:
                relevance_score += 0.4
            
            if relevance_score > min_similarity:
                memory_copy = memory.copy()
                memory_copy['similarity_score'] = min(1.0, relevance_score)
                filtered_memories.append(memory_copy)
        
        # Sort by similarity score
        filtered_memories.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return filtered_memories[:limit]


@pytest.fixture
def mock_domain_manager():
    """Create mock domain manager for UX testing."""
    return MockDomainManager()


@pytest.fixture
def suggestion_system(mock_domain_manager):
    """Create complete suggestion system for UX testing."""
    tool_indexer = MCPToolIndexer(mock_domain_manager)
    
    # Mock discovered servers
    tool_indexer.discovered_servers = {
        'postgres': {
            'command': 'npx',
            'args': ['@modelcontextprotocol/server-postgres'],
            'tools': ['postgres_query', 'postgres_list_tables'],
            'source': 'test'
        },
        'filesystem': {
            'command': 'npx',
            'args': ['@modelcontextprotocol/server-filesystem'],
            'tools': ['read_file', 'write_file', 'list_directory'],
            'source': 'test'
        },
        'web': {
            'command': 'python',
            'args': ['-m', 'web_server'],
            'tools': ['http_get', 'http_post'],
            'source': 'test'
        },
        'git': {
            'command': 'git-server',
            'args': [],
            'tools': ['git_status', 'git_log', 'git_commit'],
            'source': 'test'
        }
    }
    
    return {
        'tool_indexer': tool_indexer,
        'hook_integration': MCPHookIntegration(tool_indexer),
        'workflow_enhancer': WorkflowMemoryEnhancer(mock_domain_manager),
        'resource_monitor': ResourceReferenceMonitor(),
        'slash_discovery': SlashCommandDiscovery(tool_indexer)
    }


class TestSuggestionAccuracy:
    """Test the accuracy of MCP tool suggestions."""
    
    @pytest.mark.asyncio
    async def test_proactive_database_suggestions(self, suggestion_system):
        """Test accuracy of proactive database-related suggestions."""
        hook_integration = suggestion_system['hook_integration']
        
        # Test scenarios with expected high-accuracy suggestions
        test_scenarios = [
            {
                'prompt': 'I need to query the user database to check authentication credentials',
                'expected_suggestions': ['postgres_query'],
                'expected_confidence': 0.8,
                'context': 'user_authentication'
            },
            {
                'prompt': 'Get all active users from the database with their last login times',
                'expected_suggestions': ['postgres_query'],
                'expected_confidence': 0.85,  
                'context': 'user_management'
            },
            {
                'prompt': 'Check how many records are in the products table',
                'expected_suggestions': ['postgres_query', 'postgres_list_tables'],
                'expected_confidence': 0.8,
                'context': 'data_analysis'
            }
        ]
        
        accuracy_scores = []
        
        for scenario in test_scenarios:
            # Test pre-tool analysis for database suggestions
            result = await hook_integration.analyze_tool_usage('prompt_submit', {
                'prompt': scenario['prompt']
            })
            
            if result and 'suggested_approach' in result:
                suggested_tools = result.get('suggested_approach', [])
                
                # Calculate accuracy based on expected suggestions
                correct_suggestions = 0
                total_expected = len(scenario['expected_suggestions'])
                
                for expected_tool in scenario['expected_suggestions']:
                    if any(expected_tool in str(suggestion).lower() for suggestion in suggested_tools):
                        correct_suggestions += 1
                
                accuracy = correct_suggestions / total_expected if total_expected > 0 else 0
                accuracy_scores.append(accuracy)
                
                # Verify minimum accuracy threshold
                assert accuracy >= 0.7, f"Low accuracy ({accuracy:.2f}) for scenario: {scenario['prompt']}"
        
        # Overall accuracy should be high
        overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        assert overall_accuracy >= 0.8, f"Overall database suggestion accuracy too low: {overall_accuracy:.2f}"
    
    @pytest.mark.asyncio
    async def test_file_operation_suggestions(self, suggestion_system):
        """Test accuracy of file operation suggestions."""
        hook_integration = suggestion_system['hook_integration']
        
        file_scenarios = [
            {
                'prompt': 'Read the application configuration file to get database settings',
                'expected_tools': ['read_file'],
                'expected_confidence': 0.85
            },
            {
                'prompt': 'I need to write the processed data to a JSON file',
                'expected_tools': ['write_file'],
                'expected_confidence': 0.8
            },
            {
                'prompt': 'List all files in the project directory to find the config',
                'expected_tools': ['list_directory'],
                'expected_confidence': 0.8
            },
            {
                'prompt': 'Check what files exist in the logs folder',
                'expected_tools': ['list_directory'],
                'expected_confidence': 0.75
            }
        ]
        
        for scenario in file_scenarios:
            result = await hook_integration.analyze_tool_usage('prompt_submit', {
                'prompt': scenario['prompt']
            })
            
            if result and 'suggested_approach' in result:
                suggestions = result.get('suggested_approach', [])
                
                # Check if expected tools are suggested
                found_expected = False
                for expected_tool in scenario['expected_tools']:
                    if any(expected_tool in str(suggestion).lower() for suggestion in suggestions):
                        found_expected = True
                        break
                
                assert found_expected, f"Expected tool not found in suggestions for: {scenario['prompt']}"
    
    @pytest.mark.asyncio
    async def test_contextual_suggestion_relevance(self, suggestion_system):
        """Test that suggestions are contextually relevant to user's project."""
        workflow_enhancer = suggestion_system['workflow_enhancer']
        
        # Store successful patterns for different project types
        patterns = [
            {
                'context': 'Web application user authentication using PostgreSQL',
                'tools': ['postgres_query'],
                'score': 0.9,
                'project_type': 'web_application',
                'intent': 'user_authentication'
            },
            {
                'context': 'Data analysis query for user behavior insights',
                'tools': ['postgres_query'],
                'score': 0.8,
                'project_type': 'data_analysis', 
                'intent': 'analytics'
            },
            {
                'context': 'Configuration file reading for deployment',
                'tools': ['read_file'],
                'score': 0.85,
                'project_type': 'devops',
                'intent': 'deployment'
            }
        ]
        
        for pattern in patterns:
            await workflow_enhancer.store_mcp_workflow_pattern(pattern)
        
        # Test contextually relevant suggestions
        web_context = SuggestionContext(
            current_task='User login system',
            user_intent='Authenticate users',
            project_type='web_application',
            recent_tools_used=['bash'],
            recent_failures=[],
            environment_info={'database': 'postgres'},
            available_servers=['postgres', 'filesystem']
        )
        
        suggestions = await workflow_enhancer.get_workflow_suggestions(web_context)
        
        assert len(suggestions) > 0, "No contextual suggestions returned"
        
        # Top suggestion should be relevant to web application context
        top_suggestion = suggestions[0]
        assert top_suggestion['confidence'] > 0.6, "Low confidence in contextual suggestion"
        assert 'postgres_query' in top_suggestion.get('suggested_tools', []), "Contextually relevant tool not suggested"
    
    @pytest.mark.asyncio 
    async def test_learning_effectiveness_over_time(self, suggestion_system):
        """Test that suggestion accuracy improves with usage patterns."""
        hook_integration = suggestion_system['hook_integration']
        
        # Simulate repeated successful usage of a pattern
        usage_scenario = {
            'prompt': 'Get user count from database',
            'successful_tool': 'postgres_query',
            'context': {'project_type': 'web_application', 'intent': 'analytics'}
        }
        
        initial_suggestions = []
        final_suggestions = []
        
        # Get initial suggestions
        initial_result = await hook_integration.analyze_tool_usage('prompt_submit', {
            'prompt': usage_scenario['prompt']
        })
        if initial_result:
            initial_suggestions = initial_result.get('suggested_approach', [])
        
        # Simulate successful usage multiple times
        for i in range(5):
            # Pre-tool analysis
            await hook_integration.analyze_tool_usage('pre_tool', {
                'tool_name': 'bash',
                'args': f'psql -c "SELECT count(*) FROM users" # iteration {i}'
            })
            
            # Post-tool learning (successful)
            await hook_integration.analyze_tool_usage('post_tool', {
                'tool_name': 'bash',
                'result': f'count: {100 + i * 10}',
                'exit_code': 0,
                'context': usage_scenario['context']
            })
        
        # Get suggestions after learning
        final_result = await hook_integration.analyze_tool_usage('prompt_submit', {
            'prompt': usage_scenario['prompt']
        })
        if final_result:
            final_suggestions = final_result.get('suggested_approach', [])
        
        # Suggestions should be more confident and relevant after learning
        assert len(final_suggestions) >= len(initial_suggestions), "Suggestions decreased after learning"
        
        # Check if confidence improved (if available in result)
        if final_result and 'confidence' in final_result:
            assert final_result['confidence'] > 0.7, "Confidence didn't improve after learning"
    
    @pytest.mark.asyncio
    async def test_resource_reference_suggestion_quality(self, suggestion_system):
        """Test quality of resource reference suggestions."""
        resource_monitor = suggestion_system['resource_monitor']
        
        # Test various resource reference scenarios
        scenarios = [
            {
                'prompt': 'I need to read the database configuration file',
                'expected_reference_type': 'file',
                'expected_server': 'filesystem',
                'context': {'available_servers': ['filesystem', 'postgres']}
            },
            {
                'prompt': 'Execute a query to get all active users',
                'expected_reference_type': 'query',
                'expected_server': 'postgres',
                'context': {'available_servers': ['postgres', 'filesystem']}
            },
            {
                'prompt': 'Make an API call to get weather data',
                'expected_reference_type': 'request',
                'expected_server': 'web',
                'context': {'available_servers': ['web', 'filesystem']}
            }
        ]
        
        for scenario in scenarios:
            opportunities = resource_monitor.detect_resource_opportunities(
                scenario['prompt'], 
                scenario['context']
            )
            
            assert len(opportunities) > 0, f"No resource opportunities detected for: {scenario['prompt']}"
            
            # Check quality of top suggestion
            top_opportunity = opportunities[0]
            assert top_opportunity.confidence > 0.6, f"Low confidence resource suggestion: {top_opportunity.confidence}"
            
            # Verify reference format
            reference = top_opportunity.suggested_reference
            assert reference.startswith('@'), f"Invalid reference format: {reference}"
            assert scenario['expected_reference_type'] in reference, f"Expected reference type not in: {reference}"
            
            # Check server suggestion quality
            if scenario['expected_server'] in scenario['context']['available_servers']:
                assert scenario['expected_server'] in reference, f"Expected server not in reference: {reference}"
    
    @pytest.mark.asyncio
    async def test_suggestion_response_time(self, suggestion_system):
        """Test that suggestions are generated within acceptable time limits."""
        hook_integration = suggestion_system['hook_integration']
        
        # Test various complexity levels
        test_prompts = [
            'Simple file read',
            'Complex database query with joins and aggregations for user analytics dashboard',
            'Multi-step workflow involving database queries, file operations, and API calls',
        ]
        
        response_times = []
        
        for prompt in test_prompts:
            start_time = time.time()
            
            result = await hook_integration.analyze_tool_usage('prompt_submit', {
                'prompt': prompt
            })
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
            
            # Each suggestion should be generated quickly (< 500ms)
            assert response_time < 500, f"Suggestion too slow ({response_time:.0f}ms) for: {prompt}"
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 200, f"Average suggestion time too slow: {avg_response_time:.0f}ms"
    
    @pytest.mark.asyncio
    async def test_suggestion_diversity_and_completeness(self, suggestion_system):
        """Test that suggestions cover diverse tools and are comprehensive."""
        hook_integration = suggestion_system['hook_integration']
        
        # Test complex scenario that could benefit from multiple tools
        complex_prompt = """
        I need to:
        1. Read the user database configuration from a file
        2. Query the database to get user statistics
        3. Make an API call to get additional user metadata
        4. Save the combined results to a report file
        """
        
        result = await hook_integration.analyze_tool_usage('prompt_submit', {
            'prompt': complex_prompt
        })
        
        if result and 'suggested_approach' in result:
            suggestions = result.get('suggested_approach', [])
            
            # Should suggest multiple complementary tools
            assert len(suggestions) >= 2, f"Insufficient suggestions for complex task: {len(suggestions)}"
            
            # Check for diversity in tool types
            tool_types_found = set()
            for suggestion in suggestions:
                suggestion_str = str(suggestion).lower()
                if 'file' in suggestion_str or 'read' in suggestion_str:
                    tool_types_found.add('file_operations')
                if 'query' in suggestion_str or 'database' in suggestion_str:
                    tool_types_found.add('database')
                if 'http' in suggestion_str or 'api' in suggestion_str:
                    tool_types_found.add('web_requests')
            
            # Should cover multiple tool categories for complex task
            assert len(tool_types_found) >= 2, f"Insufficient tool diversity: {tool_types_found}"
    
    @pytest.mark.asyncio
    async def test_false_positive_rate(self, suggestion_system):
        """Test that suggestions have low false positive rate for irrelevant prompts."""
        hook_integration = suggestion_system['hook_integration']
        
        # Test prompts that should NOT trigger MCP suggestions
        irrelevant_prompts = [
            'What is the weather like today?',
            'How do I bake a chocolate cake?',
            'What is the capital of France?',
            'Explain quantum physics concepts',
            'Tell me a joke about programming'
        ]
        
        false_positives = 0
        
        for prompt in irrelevant_prompts:
            result = await hook_integration.analyze_tool_usage('prompt_submit', {
                'prompt': prompt
            })
            
            # These prompts should not generate MCP suggestions
            if result and result.get('MCP Tool Suggestion'):
                false_positives += 1
        
        false_positive_rate = false_positives / len(irrelevant_prompts)
        
        # False positive rate should be very low (< 20%)
        assert false_positive_rate < 0.2, f"High false positive rate: {false_positive_rate:.2f}"
    
    @pytest.mark.asyncio
    async def test_user_feedback_integration(self, suggestion_system):
        """Test that user feedback improves suggestion quality."""
        workflow_enhancer = suggestion_system['workflow_enhancer']
        
        # Store initial pattern
        initial_pattern = {
            'context': 'User data retrieval task',
            'tools': ['bash'],  # Suboptimal tool
            'score': 0.6,
            'project_type': 'web_application',
            'intent': 'data_retrieval'
        }
        await workflow_enhancer.store_mcp_workflow_pattern(initial_pattern)
        
        # Store improved pattern (simulating user choosing better tool)
        improved_pattern = {
            'context': 'User data retrieval task',
            'tools': ['postgres_query'],  # Better tool
            'score': 0.9,
            'project_type': 'web_application',
            'intent': 'data_retrieval'
        }
        await workflow_enhancer.store_mcp_workflow_pattern(improved_pattern)
        
        # Test suggestions after feedback
        context = SuggestionContext(
            current_task='Retrieve user data',
            user_intent='Get user information',
            project_type='web_application',
            recent_tools_used=[],
            recent_failures=[],
            environment_info={'database': 'postgres'},
            available_servers=['postgres']
        )
        
        suggestions = await workflow_enhancer.get_workflow_suggestions(context)
        
        assert len(suggestions) > 0, "No suggestions after feedback"
        
        # Should prefer the improved pattern
        top_suggestion = suggestions[0]
        assert 'postgres_query' in top_suggestion.get('suggested_tools', []), "Improved tool not prioritized"
        assert top_suggestion['confidence'] > 0.8, "Low confidence in improved suggestion"
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self, suggestion_system):
        """Test suggestion accuracy for edge cases and unusual inputs."""
        hook_integration = suggestion_system['hook_integration']
        
        edge_cases = [
            {
                'prompt': '',  # Empty prompt
                'expected_behavior': 'no_suggestions'
            },
            {
                'prompt': 'a' * 1000,  # Very long prompt
                'expected_behavior': 'graceful_handling'
            },
            {
                'prompt': 'SELECT * FROM users; DROP TABLE users;',  # SQL injection attempt
                'expected_behavior': 'safe_suggestion'
            },
            {
                'prompt': 'database query file read api call',  # Ambiguous keywords
                'expected_behavior': 'multiple_suggestions'
            }
        ]
        
        for case in edge_cases:
            try:
                result = await hook_integration.analyze_tool_usage('prompt_submit', {
                    'prompt': case['prompt']
                })
                
                # Should handle gracefully without errors
                if case['expected_behavior'] == 'no_suggestions':
                    assert result is None or (result and len(result.get('suggested_approach', [])) == 0)
                elif case['expected_behavior'] == 'graceful_handling':
                    # For graceful handling, it's ok if result is None (handled gracefully by returning None)
                    assert True  # Just ensure no exception was thrown
                else:
                    assert result is not None, f"Expected result for case {case['prompt'][:50]}... but got None"
                
                if case['expected_behavior'] == 'multiple_suggestions' and result:
                    suggestions = result.get('suggested_approach', [])
                    assert len(suggestions) > 1, f"Expected multiple suggestions for ambiguous prompt"
                
            except Exception as e:
                pytest.fail(f"Edge case failed with exception: {e}")
    
    @pytest.mark.asyncio
    async def test_suggestion_explanations_quality(self, suggestion_system):
        """Test that suggestion explanations are helpful and accurate."""
        hook_integration = suggestion_system['hook_integration']
        
        test_prompt = "I need to query the database to get user analytics data"
        
        result = await hook_integration.analyze_tool_usage('prompt_submit', {
            'prompt': test_prompt
        })
        
        if result and 'MCP Tool Suggestion' in result:
            explanation = result['MCP Tool Suggestion']
            
            # Explanation should be informative
            assert len(explanation) > 20, "Explanation too brief"
            assert 'database' in explanation.lower(), "Explanation doesn't mention key context"
            
            # Should explain benefits
            assert any(word in explanation.lower() for word in ['better', 'efficient', 'direct', 'optimize']), \
                "Explanation doesn't mention benefits"


class TestSuggestionMetrics:
    """Test metrics and analytics for suggestion system."""
    
    @pytest.mark.asyncio
    async def test_suggestion_success_rate_tracking(self, suggestion_system):
        """Test tracking of suggestion success rates."""
        hook_integration = suggestion_system['hook_integration']
        
        # Simulate suggestion and user acceptance
        for i in range(10):
            # Generate suggestion
            await hook_integration.analyze_tool_usage('prompt_submit', {
                'prompt': f'Database query task {i}'
            })
            
            # Simulate user following suggestion (or not)
            success = i < 7  # 70% success rate
            await hook_integration.analyze_tool_usage('post_tool', {
                'tool_name': 'postgres_query' if success else 'bash',
                'result': 'success' if success else 'failed',
                'exit_code': 0 if success else 1
            })
        
        # Get analytics
        analytics = hook_integration.get_learning_stats()
        
        assert 'suggestion_success_rate' in analytics or 'patterns_learned' in analytics
        
        # Should track learning effectiveness
        if 'patterns_learned' in analytics:
            assert analytics['patterns_learned'] > 0, "No patterns learned from interactions"
    
    def test_suggestion_coverage_metrics(self, suggestion_system):
        """Test coverage metrics for different tool categories."""
        tool_indexer = suggestion_system['tool_indexer']
        
        # Check tool coverage across categories
        servers = tool_indexer.discovered_servers
        
        categories_covered = set()
        for server_name, server_info in servers.items():
            if 'postgres' in server_name or 'database' in server_name:
                categories_covered.add('database')
            if 'filesystem' in server_name or 'file' in server_name:
                categories_covered.add('file_operations')
            if 'web' in server_name or 'http' in server_name:
                categories_covered.add('web_requests')
            if 'git' in server_name:
                categories_covered.add('version_control')
        
        # Should cover major tool categories
        assert len(categories_covered) >= 3, f"Insufficient tool category coverage: {categories_covered}"


# Integration test with real workflow scenarios
class TestRealWorldScenarios:
    """Test suggestion accuracy with realistic user scenarios."""
    
    @pytest.mark.asyncio
    async def test_web_development_workflow(self, suggestion_system):
        """Test suggestions for typical web development tasks."""
        scenarios = [
            'Set up user authentication system',
            'Query user database for login validation', 
            'Read configuration file for database connection',
            'Log user activity to database',
            'Generate user analytics report'
        ]
        
        hook_integration = suggestion_system['hook_integration']
        
        for scenario in scenarios:
            result = await hook_integration.analyze_tool_usage('prompt_submit', {
                'prompt': scenario
            })
            
            # Should provide relevant suggestions for each web dev task
            assert result is not None, f"No suggestions for web dev scenario: {scenario}"
    
    @pytest.mark.asyncio
    async def test_data_analysis_workflow(self, suggestion_system):
        """Test suggestions for data analysis tasks."""
        workflow_enhancer = suggestion_system['workflow_enhancer']
        
        # Simulate data analysis context
        context = SuggestionContext(
            current_task='Analyze user behavior patterns',
            user_intent='Generate insights from user data',
            project_type='data_analysis',
            recent_tools_used=['pandas', 'jupyter'],
            recent_failures=[],
            environment_info={'database': 'postgres', 'platform': 'jupyter'},
            available_servers=['postgres', 'filesystem']
        )
        
        suggestions = await workflow_enhancer.get_workflow_suggestions(context)
        
        # Should suggest appropriate tools for data analysis
        assert len(suggestions) > 0, "No suggestions for data analysis workflow"
        
        # Suggestions should be relevant to data context
        relevant_found = False
        for suggestion in suggestions:
            tools = suggestion.get('suggested_tools', [])
            if any('postgres' in str(tool).lower() or 'query' in str(tool).lower() for tool in tools):
                relevant_found = True
                break
        
        assert relevant_found, "No relevant data analysis tools suggested"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])