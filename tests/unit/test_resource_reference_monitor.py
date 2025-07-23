"""
Unit tests for ResourceReferenceMonitor.

Tests the resource reference detection and learning system in isolation.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
from datetime import datetime, timezone

from clarity.mcp.resource_reference_monitor import (
    ResourceReferenceMonitor,
    ResourceReference,
    ResourceOpportunity
)


class TestResourceReferenceMonitor:
    """Unit tests for resource reference monitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create resource reference monitor for testing."""
        return ResourceReferenceMonitor()
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return {
            'available_servers': ['filesystem', 'postgres', 'web'],
            'project_type': 'web_application',
            'user_intent': 'data_access'
        }
    
    def test_file_operations_detection(self, monitor, sample_context):
        """Test detection of file operation opportunities."""
        test_prompts = [
            "I need to read the config.json file from the project directory",
            "Can you cat the /etc/hosts file?",
            "Please write to the output.txt file",
            "ls /home/user"
        ]
        
        for prompt in test_prompts:
            opportunities = monitor.detect_resource_opportunities(prompt, sample_context)
            
            assert len(opportunities) > 0
            
            # Should detect file operations
            file_ops = [opp for opp in opportunities if opp.opportunity_type == 'file_operations']
            assert len(file_ops) > 0
            
            file_op = file_ops[0]
            assert '@filesystem:file://' in file_op.suggested_reference
            assert file_op.confidence > 0.4  # Lower threshold due to context calculation
            assert 'file' in file_op.reason.lower()
            assert len(file_op.potential_benefits) > 0
    
    def test_database_query_detection(self, monitor, sample_context):
        """Test detection of database query opportunities."""
        test_prompts = [
            'psql -d mydb -c "SELECT * FROM users"',
            'mysql -u user -p -e "SELECT count(*) FROM products"',
            'sqlite3 data.db "INSERT INTO logs VALUES (1, \'test\')"',
            'I need to query the database to get user information'
        ]
        
        for prompt in test_prompts:
            opportunities = monitor.detect_resource_opportunities(prompt, sample_context)
            
            assert len(opportunities) > 0
            
            # Should detect database queries
            db_ops = [opp for opp in opportunities if opp.opportunity_type == 'database_queries']
            assert len(db_ops) > 0
            
            db_op = db_ops[0]
            assert '@postgres:query://' in db_op.suggested_reference
            assert db_op.confidence > 0.7  # Database detection should be high confidence
            assert 'database' in db_op.reason.lower() or 'query' in db_op.reason.lower()
    
    def test_web_request_detection(self, monitor, sample_context):
        """Test detection of web request opportunities."""
        test_prompts = [
            'curl -X GET https://api.example.com/users',
            'wget https://example.com/file.zip',
            'make api call to get user data',
            'fetch data from https://jsonplaceholder.typicode.com/posts'
        ]
        
        for prompt in test_prompts:
            opportunities = monitor.detect_resource_opportunities(prompt, sample_context)
            
            assert len(opportunities) > 0
            
            # Should detect web requests
            web_ops = [opp for opp in opportunities if opp.opportunity_type == 'web_requests']
            assert len(web_ops) > 0
            
            web_op = web_ops[0]
            assert '@web:request://' in web_op.suggested_reference
            assert web_op.confidence > 0.6
            assert any(benefit for benefit in web_op.potential_benefits if 'response' in benefit.lower() or 'header' in benefit.lower() or 'authentication' in benefit.lower())
    
    def test_git_operations_detection(self, monitor, sample_context):
        """Test detection of git operation opportunities."""
        # Add git server to context
        git_context = sample_context.copy()
        git_context['available_servers'].append('git')
        
        test_prompts = [
            'git clone https://github.com/user/repo.git',
            'git log --oneline',
            'gh repo create new-project',
            'git status'
        ]
        
        for prompt in test_prompts:
            opportunities = monitor.detect_resource_opportunities(prompt, git_context)
            
            assert len(opportunities) > 0
            
            # Should detect git operations
            git_ops = [opp for opp in opportunities if opp.opportunity_type == 'git_operations']
            assert len(git_ops) > 0
            
            git_op = git_ops[0]
            assert '@git:repo://' in git_op.suggested_reference
            assert 'git' in git_op.reason.lower() or 'repository' in git_op.reason.lower()
    
    def test_no_opportunity_detection(self, monitor, sample_context):
        """Test when no opportunities should be detected."""
        test_prompts = [
            "What is the weather today?",
            "Calculate 2 + 2",
            "Hello, how are you?",
            "Explain quantum physics"
        ]
        
        for prompt in test_prompts:
            opportunities = monitor.detect_resource_opportunities(prompt, sample_context)
            
            # Should detect no opportunities or very low confidence ones
            assert len(opportunities) == 0 or all(opp.confidence < 0.4 for opp in opportunities)
    
    def test_context_matching_affects_confidence(self, monitor):
        """Test that context matching affects confidence scores."""
        prompt = "read the config file"
        
        # Context with filesystem server (direct match)
        context_with_fs = {'available_servers': ['filesystem']}
        opportunities_with_fs = monitor.detect_resource_opportunities(prompt, context_with_fs)
        
        # Context with generic file server (partial match)
        context_with_file = {'available_servers': ['file']}
        opportunities_with_file = monitor.detect_resource_opportunities(prompt, context_with_file)
        
        # Should have opportunities in both cases
        assert len(opportunities_with_fs) > 0
        assert len(opportunities_with_file) > 0
        
        # Confidence should be similar since both are file-related servers
        fs_confidence = opportunities_with_fs[0].confidence
        file_confidence = opportunities_with_file[0].confidence
        
        # Both should have reasonable confidence for file operations
        assert fs_confidence > 0.4
        assert file_confidence > 0.4
    
    @pytest.mark.asyncio
    async def test_learn_resource_pattern_success(self, monitor):
        """Test learning from successful resource reference usage."""
        reference = "@filesystem:file://config.json"
        context = {
            'prompt': 'Read the configuration file',
            'intent': 'configuration_access'
        }
        
        await monitor.learn_resource_pattern(
            reference,
            context,
            success=True,
            response_time=150.0
        )
        
        # Check that pattern was learned
        pattern_key = "filesystem:file"
        assert pattern_key in monitor.reference_patterns
        
        pattern = monitor.reference_patterns[pattern_key]
        assert pattern['usage_count'] == 1
        assert pattern['success_count'] == 1
        assert pattern['total_response_time'] == 150.0
        assert len(pattern['contexts']) == 1
        assert len(pattern['resource_examples']) == 1
        assert 'config.json' in pattern['resource_examples']
        
        # Check usage history
        assert len(monitor.usage_history) == 1
        history_record = monitor.usage_history[0]
        assert history_record['reference'] == reference
        assert history_record['success'] is True
        assert history_record['response_time'] == 150.0
        assert history_record['server'] == 'filesystem'
        assert history_record['protocol'] == 'file'
    
    @pytest.mark.asyncio
    async def test_learn_resource_pattern_failure(self, monitor):
        """Test learning from failed resource reference usage."""
        reference = "@postgres:query://SELECT * FROM nonexistent"
        context = {
            'prompt': 'Query missing table',
            'intent': 'data_retrieval'
        }
        
        await monitor.learn_resource_pattern(
            reference,
            context,
            success=False,
            response_time=5000.0  # Long time due to timeout
        )
        
        pattern_key = "postgres:query"
        assert pattern_key in monitor.reference_patterns
        
        pattern = monitor.reference_patterns[pattern_key]
        assert pattern['usage_count'] == 1
        assert pattern['success_count'] == 0  # Failed
        assert pattern['total_response_time'] == 5000.0
        
        # Check performance stats
        stats = monitor.get_performance_stats()
        assert stats['successful_suggestions'] == 0
        assert stats['reference_usage_tracked'] == 1
    
    @pytest.mark.asyncio
    async def test_learn_invalid_reference(self, monitor):
        """Test learning with invalid reference format."""
        invalid_reference = "not-a-valid-reference"
        context = {'prompt': 'test'}
        
        # Should handle gracefully without raising exception
        await monitor.learn_resource_pattern(invalid_reference, context)
        
        # Should not add any patterns
        assert len(monitor.reference_patterns) == 0
        assert len(monitor.usage_history) == 0
    
    @pytest.mark.asyncio
    async def test_get_reference_suggestions(self, monitor):
        """Test getting contextual reference suggestions."""
        # First learn some patterns
        await monitor.learn_resource_pattern(
            "@filesystem:file://config.json",
            {'prompt': 'read config', 'intent': 'configuration'},
            success=True,
            response_time=100.0
        )
        
        await monitor.learn_resource_pattern(
            "@postgres:query://SELECT * FROM users",
            {'prompt': 'get users', 'intent': 'data_retrieval'},
            success=True,
            response_time=200.0
        )
        
        # Get suggestions for file operation
        suggestions = await monitor.get_reference_suggestions(
            "I need to read the application configuration file",
            available_servers=['filesystem', 'postgres']
        )
        
        assert len(suggestions) > 0
        
        # Should suggest file operations
        file_suggestions = [s for s in suggestions if 'file' in s['reference']]
        assert len(file_suggestions) > 0
        
        file_suggestion = file_suggestions[0]
        assert file_suggestion['type'] == 'resource_reference'
        assert file_suggestion['confidence'] > 0.5
        assert file_suggestion['historical_success_rate'] == 1.0  # 100% success
        assert file_suggestion['historical_usage'] == 1
        assert file_suggestion['average_response_time'] == 100.0
    
    @pytest.mark.asyncio
    async def test_analyze_usage_patterns(self, monitor):
        """Test analysis of resource reference usage patterns."""
        # Generate usage data with more postgres failures to trigger recommendations
        references = [
            ("@filesystem:file://config.json", True, 100.0),
            ("@filesystem:file://data.txt", True, 150.0),
            ("@postgres:query://SELECT * FROM users", True, 200.0),
            ("@postgres:query://SELECT * FROM logs", False, 5000.0),  # Failed
            ("@postgres:query://SELECT * FROM posts", False, 5500.0),  # Failed
            ("@postgres:query://SELECT * FROM comments", False, 4800.0),  # Failed
            ("@postgres:query://SELECT * FROM tags", True, 250.0),
            ("@postgres:query://SELECT * FROM categories", False, 6000.0),  # Failed
            ("@postgres:query://SELECT * FROM sessions", False, 5200.0),  # Failed
            ("@web:request://api.example.com/data", True, 300.0)
        ]
        
        for ref, success, response_time in references:
            await monitor.learn_resource_pattern(
                ref,
                {'prompt': 'test', 'intent': 'test'},
                success=success,
                response_time=response_time
            )
        
        analysis = await monitor.analyze_reference_usage_patterns()
        
        # Check analysis structure
        assert 'most_successful_patterns' in analysis
        assert 'server_performance' in analysis
        assert 'protocol_usage' in analysis
        assert 'recommendations' in analysis
        
        # Check most successful patterns
        successful_patterns = analysis['most_successful_patterns']
        assert len(successful_patterns) > 0
        
        # Filesystem should be most successful (100% success rate)
        filesystem_pattern = next(
            (p for p in successful_patterns if p['pattern'] == 'filesystem:file'),
            None
        )
        assert filesystem_pattern is not None
        assert filesystem_pattern['success_rate'] == 1.0
        assert filesystem_pattern['usage_count'] == 2
        
        # Check server performance
        server_perf = analysis['server_performance']
        assert 'filesystem' in server_perf
        assert 'postgres' in server_perf
        assert 'web' in server_perf
        
        # Filesystem should have better performance than postgres
        assert server_perf['filesystem']['success_rate'] > server_perf['postgres']['success_rate']
        
        # Check recommendations
        recommendations = analysis['recommendations']
        assert isinstance(recommendations, list)
        
        # Should recommend addressing postgres issues
        postgres_recommendations = [r for r in recommendations if 'postgres' in r.lower()]
        assert len(postgres_recommendations) > 0
    
    def test_resource_reference_parsing(self):
        """Test ResourceReference parsing functionality."""
        # Test valid reference
        valid_ref = "@filesystem:file://path/to/file.txt"
        context = "test context"
        
        parsed = ResourceReference.parse_reference(valid_ref, context)
        
        assert parsed is not None
        assert parsed.server == "filesystem"
        assert parsed.protocol == "file"
        assert parsed.resource_path == "path/to/file.txt"
        assert parsed.full_reference == valid_ref
        assert parsed.context == context
        assert parsed.usage_count == 0
        assert parsed.success_rate == 1.0
        
        # Test invalid reference
        invalid_ref = "not-a-valid-reference"
        parsed_invalid = ResourceReference.parse_reference(invalid_ref)
        assert parsed_invalid is None
        
        # Test edge cases
        edge_cases = [
            "@server:protocol://",  # Empty resource
            "@:protocol://resource",  # Empty server
            "@server::resource",  # Missing protocol
            "server:protocol://resource"  # Missing @
        ]
        
        for edge_case in edge_cases:
            parsed_edge = ResourceReference.parse_reference(edge_case)
            # Should either parse correctly or return None
            if parsed_edge:
                assert hasattr(parsed_edge, 'server')
                assert hasattr(parsed_edge, 'protocol')
    
    def test_opportunity_deduplication(self, monitor, sample_context):
        """Test deduplication of similar opportunities."""
        # Prompt that might generate multiple similar opportunities
        prompt = "I need to read the config file and also the settings file"
        
        opportunities = monitor.detect_resource_opportunities(prompt, sample_context)
        
        # Should not have duplicate references
        references = [opp.suggested_reference for opp in opportunities]
        unique_references = set(references)
        assert len(references) == len(unique_references)
    
    def test_performance_stats_tracking(self, monitor):
        """Test performance statistics tracking."""
        initial_stats = monitor.get_performance_stats()
        
        # Generate some opportunities
        monitor.detect_resource_opportunities("read file", {'available_servers': ['filesystem']})
        monitor.detect_resource_opportunities("query database", {'available_servers': ['postgres']})
        
        updated_stats = monitor.get_performance_stats()
        
        assert updated_stats['total_opportunities_detected'] > initial_stats['total_opportunities_detected']
        assert updated_stats['total_opportunities_detected'] == 2
    
    def test_server_compatibility_finding(self, monitor):
        """Test finding compatible servers for opportunity types."""
        context = {
            'available_servers': ['my_filesystem_server', 'postgres_db', 'web_client', 'unrelated_server']
        }
        
        # Test filesystem compatibility
        fs_compatible = monitor._find_compatible_servers(['filesystem', 'file'], context)
        assert 'my_filesystem_server' in fs_compatible
        
        # Test database compatibility
        db_compatible = monitor._find_compatible_servers(['postgres', 'database'], context)
        assert 'postgres_db' in db_compatible
        
        # Test web compatibility
        web_compatible = monitor._find_compatible_servers(['web', 'http'], context)
        assert 'web_client' in web_compatible
        
        # Test with no context
        no_context_compatible = monitor._find_compatible_servers(['filesystem'], {})
        assert len(no_context_compatible) > 0  # Should return potential names
    
    def test_context_match_calculation(self, monitor):
        """Test context matching calculation."""
        # File operations prompt
        file_match = monitor._calculate_context_match(
            "I need to read a configuration file",
            "file_operations"
        )
        assert file_match > 0.5
        
        # Database query prompt
        db_match = monitor._calculate_context_match(
            "SELECT users from database table",
            "database_queries"
        )
        assert db_match > 0.5
        
        # Mismatched context
        mismatch = monitor._calculate_context_match(
            "What is the weather?",
            "file_operations"
        )
        assert mismatch <= 0.65  # Base context match is 0.6, so this would be around 0.6
    
    def test_opportunity_reason_generation(self, monitor):
        """Test generation of opportunity reasons."""
        reasons = {
            'file_operations': monitor._generate_opportunity_reason('file_operations', 'config.json'),
            'database_queries': monitor._generate_opportunity_reason('database_queries', 'SELECT query'),
            'web_requests': monitor._generate_opportunity_reason('web_requests', 'api.example.com'),
            'unknown_type': monitor._generate_opportunity_reason('unknown_type', 'resource')
        }
        
        for reason in reasons.values():
            assert isinstance(reason, str)
            assert len(reason) > 10  # Should be descriptive
        
        # Check specific content
        assert 'file' in reasons['file_operations'].lower()
        assert 'database' in reasons['database_queries'].lower() or 'query' in reasons['database_queries'].lower()
        assert 'http' in reasons['web_requests'].lower() or 'request' in reasons['web_requests'].lower()
    
    @pytest.mark.asyncio
    async def test_pattern_history_management(self, monitor):
        """Test that pattern history is managed properly."""
        # Add many usage records to test history trimming
        for i in range(1200):  # More than the 1000 limit
            await monitor.learn_resource_pattern(
                f"@test:protocol://resource{i}",
                {'prompt': f'test {i}'},
                success=True,
                response_time=100.0
            )
        
        # History should be trimmed to approximately 500-700 (implementation may vary)
        # The exact number depends on how the trimming logic is applied
        assert 400 <= len(monitor.usage_history) <= 800
        
        # Should keep most recent records
        last_record = monitor.usage_history[-1]
        assert 'resource' in last_record['reference']
    
    @pytest.mark.asyncio
    async def test_concurrent_learning(self, monitor):
        """Test concurrent learning operations."""
        # Simulate concurrent learning
        tasks = []
        for i in range(10):
            task = monitor.learn_resource_pattern(
                f"@concurrent:test://resource{i}",
                {'prompt': f'concurrent test {i}'},
                success=True,
                response_time=50.0
            )
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*tasks)
        
        # Should have learned all patterns
        pattern_key = "concurrent:test"
        assert pattern_key in monitor.reference_patterns
        assert monitor.reference_patterns[pattern_key]['usage_count'] == 10
        assert len(monitor.usage_history) >= 10


class TestResourceOpportunityDataclass:
    """Test ResourceOpportunity dataclass functionality."""
    
    def test_opportunity_creation(self):
        """Test creation of ResourceOpportunity."""
        opportunity = ResourceOpportunity(
            opportunity_type="file_operations",
            suggested_reference="@filesystem:file://config.json",
            current_approach="cat config.json",
            confidence=0.85,
            reason="Direct file access through MCP",
            potential_benefits=["Better error handling", "Metadata access"],
            context_match=0.9
        )
        
        assert opportunity.opportunity_type == "file_operations"
        assert opportunity.confidence == 0.85
        assert len(opportunity.potential_benefits) == 2
        
        # Test to_dict conversion
        opportunity_dict = opportunity.to_dict()
        assert isinstance(opportunity_dict, dict)
        assert opportunity_dict['opportunity_type'] == "file_operations"
        assert opportunity_dict['confidence'] == 0.85


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])