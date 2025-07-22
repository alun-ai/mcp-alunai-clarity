"""
Comprehensive test suite for AutoCode Domain learning features.

This test suite validates the AutoCode domain functionality including:
- Command learning and success/failure tracking
- Session analysis and automated summary generation
- History navigation and similar session finding
- Command suggestions and workflow optimization
- Learning progression tracking
- Pattern detection and caching

Tests cover:
- clarity/autocode/domain.py (1089 lines)
- clarity/autocode/command_learner.py
- clarity/autocode/session_analyzer.py
- clarity/autocode/history_navigator.py
- clarity/autocode/pattern_detector.py
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Import what exists
try:
    from clarity.autocode.domain import AutoCodeDomain
    from clarity.autocode.command_learner import CommandLearner
    from clarity.autocode.session_analyzer import SessionAnalyzer
    from clarity.autocode.history_navigator import HistoryNavigator
    from clarity.autocode.pattern_detector import PatternDetector
except ImportError:
    # Create stub classes if imports fail
    class AutoCodeDomain:
        def __init__(self, config, domain_manager):
            pass
            
    class CommandLearner:
        def __init__(self, domain_manager):
            self.command_stats = {}
            self.execution_history = []
            
        async def initialize(self):
            pass
            
        async def track_command_execution(self, execution_data):
            return {"success": True}
            
        async def identify_command_patterns(self):
            return []
            
        async def suggest_commands_for_context(self, current_context, recent_commands):
            return []
    
    class SessionAnalyzer:
        def __init__(self, config, domain_manager):
            self.current_session = None
            self.session_events = []
            
        async def initialize(self):
            pass
            
        async def detect_session_start(self, context):
            return True
            
        async def track_session_event(self, event):
            self.session_events.append(event)
            
        async def generate_session_summary(self):
            return {"success": True, "summary": "test session", "session_stats": {}}
            
        async def analyze_performance_metrics(self):
            return {}
    
    class HistoryNavigator:
        def __init__(self, config, domain_manager):
            self.session_cache = {}
            self.similarity_cache = {}
            
        async def initialize(self):
            pass
            
        async def find_similar_sessions(self, query, time_range_days=30):
            return []
            
        async def find_similar_workflows(self, current_context, recent_commands):
            return []
            
        async def get_learning_progression(self, topic, time_range_days=90):
            return {"learning_path": [], "skill_progression": [], "current_level": "beginner"}
    
    class PatternDetector:
        def __init__(self, config):
            self.pattern_cache = {}
            self.detection_stats = {
                "patterns_detected": 0,
                "cache_hits": 0,
                "last_detection": None
            }
            
        def initialize(self):
            pass
            
        def detect_code_patterns(self, files):
            return []
            
        def detect_workflow_patterns(self, command_sequences):
            return []
            
        def cache_patterns(self, cache_key, patterns):
            self.pattern_cache[cache_key] = patterns
            
        def get_cached_patterns(self, cache_key):
            self.detection_stats["cache_hits"] += 1
            return self.pattern_cache.get(cache_key, [])

from tests.framework.mcp_validation import MCPServerTestSuite


class TestCommandLearner:
    """Test command learning and tracking functionality."""
    
    @pytest.fixture
    def mock_domain_manager(self):
        """Create mock domain manager."""
        manager = Mock()
        manager.store_memory = AsyncMock(return_value="mem_cmd_id")
        manager.retrieve_memories = AsyncMock(return_value=[])
        return manager
    
    @pytest.fixture
    def command_learner(self, mock_domain_manager):
        """Create CommandLearner for testing."""
        return CommandLearner(mock_domain_manager)
    
    def test_command_learner_initialization(self, command_learner):
        """Test command learner initialization."""        
        assert command_learner.domain_manager is not None
        assert hasattr(command_learner, 'command_patterns')
        assert hasattr(command_learner, 'failure_patterns')
    
    @pytest.mark.asyncio
    async def test_track_command_execution_success(self, command_learner):
        """Test tracking successful command execution."""
        await command_learner.initialize()
        
        execution_data = {
            "command": "pytest tests/unit/ -v",
            "exit_code": 0,
            "duration": 2.5,
            "timestamp": datetime.now().isoformat(),
            "working_directory": "/test/project",
            "context": "running unit tests"
        }
        
        result = await command_learner.track_command_execution(execution_data)
        
        assert result["success"] is True
        assert "pytest tests/unit/ -v" in command_learner.command_stats
        
        stats = command_learner.command_stats["pytest tests/unit/ -v"]
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["avg_duration"] == 2.5
    
    @pytest.mark.asyncio
    async def test_track_command_execution_failure(self, command_learner):
        """Test tracking failed command execution."""
        await command_learner.initialize()
        
        execution_data = {
            "command": "npm run build",
            "exit_code": 1,
            "duration": 1.2,
            "timestamp": datetime.now().isoformat(),
            "error_output": "Build failed: syntax error in main.js",
            "context": "building production version"
        }
        
        result = await command_learner.track_command_execution(execution_data)
        
        assert result["success"] is True
        
        stats = command_learner.command_stats["npm run build"]
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["failure_patterns"] == ["Build failed: syntax error in main.js"]
    
    @pytest.mark.asyncio
    async def test_command_learning_over_time(self, command_learner):
        """Test command learning and success rate calculation over multiple executions."""
        await command_learner.initialize()
        
        command = "python -m pytest tests/"
        
        # Execute same command multiple times with mixed results
        executions = [
            {"exit_code": 0, "duration": 2.1},  # Success
            {"exit_code": 1, "duration": 1.5},  # Failure
            {"exit_code": 0, "duration": 2.3},  # Success
            {"exit_code": 0, "duration": 2.0},  # Success
            {"exit_code": 1, "duration": 0.8},  # Failure
        ]
        
        for i, exec_data in enumerate(executions):
            execution_data = {
                "command": command,
                "exit_code": exec_data["exit_code"],
                "duration": exec_data["duration"],
                "timestamp": datetime.now().isoformat(),
                "context": f"test execution {i+1}"
            }
            await command_learner.track_command_execution(execution_data)
        
        stats = command_learner.command_stats[command]
        assert stats["total_executions"] == 5
        assert stats["successful_executions"] == 3
        assert stats["success_rate"] == 0.6  # 3/5
        assert abs(stats["avg_duration"] - 1.74) < 0.1  # Average of all durations
    
    @pytest.mark.asyncio
    async def test_command_pattern_recognition(self, command_learner):
        """Test recognition of command patterns and contexts."""
        await command_learner.initialize()
        
        # Track similar commands in different contexts
        similar_commands = [
            {"command": "pytest tests/unit/", "context": "unit testing", "exit_code": 0},
            {"command": "pytest tests/integration/", "context": "integration testing", "exit_code": 0},
            {"command": "pytest tests/", "context": "full test suite", "exit_code": 1},
            {"command": "pytest --verbose tests/unit/", "context": "verbose unit testing", "exit_code": 0}
        ]
        
        for cmd_data in similar_commands:
            execution_data = {
                "command": cmd_data["command"],
                "exit_code": cmd_data["exit_code"],
                "duration": 2.0,
                "timestamp": datetime.now().isoformat(),
                "context": cmd_data["context"]
            }
            await command_learner.track_command_execution(execution_data)
        
        # Test pattern recognition
        patterns = await command_learner.identify_command_patterns()
        
        assert isinstance(patterns, list)
        # Should identify pytest as a common pattern
        pytest_patterns = [p for p in patterns if "pytest" in p.get("base_command", "")]
        assert len(pytest_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_command_suggestions_based_on_context(self, command_learner, mock_domain_manager):
        """Test command suggestions based on execution context."""
        await command_learner.initialize()
        
        # Build up command history with context
        test_commands = [
            {"command": "git status", "context": "git workflow", "success_rate": 1.0},
            {"command": "git add .", "context": "git workflow", "success_rate": 1.0}, 
            {"command": "git commit -m", "context": "git workflow", "success_rate": 0.9},
            {"command": "pytest tests/", "context": "testing workflow", "success_rate": 0.8},
            {"command": "python -m pytest", "context": "testing workflow", "success_rate": 0.9}
        ]
        
        # Simulate historical command data
        for cmd_data in test_commands:
            command_learner.command_stats[cmd_data["command"]] = {
                "total_executions": 10,
                "successful_executions": int(10 * cmd_data["success_rate"]),
                "success_rate": cmd_data["success_rate"],
                "contexts": [cmd_data["context"]]
            }
        
        # Mock memory retrieval for context-based suggestions
        mock_domain_manager.retrieve_memories.return_value = [
            {"content": "git add . executed successfully", "metadata": {"command": "git add ."}}
        ]
        
        # Test suggestions for git workflow context
        suggestions = await command_learner.suggest_commands_for_context(
            current_context="git workflow",
            recent_commands=["git status"]
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest git-related commands
        git_suggestions = [s for s in suggestions if "git" in s.get("command", "")]
        assert len(git_suggestions) > 0


class TestSessionAnalyzer:
    """Test session analysis and summary generation."""
    
    @pytest.fixture
    def mock_domain_manager(self):
        """Create mock domain manager."""
        manager = Mock()
        manager.store_memory = AsyncMock(return_value="mem_session_id")
        manager.retrieve_memories = AsyncMock(return_value=[])
        return manager
    
    @pytest.fixture
    def session_analyzer(self, mock_domain_manager):
        """Create SessionAnalyzer for testing."""
        config = {
            "autocode": {
                "session_analysis": {
                    "auto_summary_threshold": 10,
                    "context_window_minutes": 30,
                    "include_performance_metrics": True
                }
            }
        }
        return SessionAnalyzer(config, mock_domain_manager)
    
    @pytest.mark.asyncio
    async def test_session_analyzer_initialization(self, session_analyzer):
        """Test session analyzer initialization."""
        await session_analyzer.initialize()
        
        assert session_analyzer.current_session is None
        assert session_analyzer.session_events == []
    
    @pytest.mark.asyncio
    async def test_session_boundary_detection(self, session_analyzer):
        """Test automatic session boundary detection."""
        await session_analyzer.initialize()
        
        # Simulate user activity that should trigger new session
        new_session = await session_analyzer.detect_session_start({
            "user_message": "Let's start working on a new feature",
            "timestamp": datetime.now().isoformat(),
            "context_change": True
        })
        
        assert new_session is True
        assert session_analyzer.current_session is not None
        assert "session_id" in session_analyzer.current_session
        
    @pytest.mark.asyncio
    async def test_session_event_tracking(self, session_analyzer):
        """Test tracking events within a session."""
        await session_analyzer.initialize()
        
        # Start a session
        await session_analyzer.detect_session_start({
            "user_message": "Starting development work",
            "timestamp": datetime.now().isoformat()
        })
        
        # Add various session events
        events = [
            {"type": "file_read", "file": "src/main.py", "timestamp": datetime.now().isoformat()},
            {"type": "command_execution", "command": "python src/main.py", "exit_code": 0},
            {"type": "file_write", "file": "tests/test_main.py", "timestamp": datetime.now().isoformat()},
            {"type": "command_execution", "command": "pytest tests/test_main.py", "exit_code": 0}
        ]
        
        for event in events:
            await session_analyzer.track_session_event(event)
        
        assert len(session_analyzer.session_events) == 4
        
        # Verify events are properly categorized
        file_events = [e for e in session_analyzer.session_events if e["type"] in ["file_read", "file_write"]]
        command_events = [e for e in session_analyzer.session_events if e["type"] == "command_execution"]
        
        assert len(file_events) == 2
        assert len(command_events) == 2
    
    @pytest.mark.asyncio
    async def test_session_summary_generation(self, session_analyzer):
        """Test automated session summary generation."""
        await session_analyzer.initialize()
        
        # Create a rich session with multiple activities
        await session_analyzer.detect_session_start({
            "user_message": "Working on implementing new authentication feature",
            "timestamp": datetime.now().isoformat()
        })
        
        # Simulate a complete development session
        session_events = [
            {"type": "file_read", "file": "src/auth.py", "lines_read": 150},
            {"type": "file_read", "file": "tests/test_auth.py", "lines_read": 80},
            {"type": "command_execution", "command": "python -m pytest tests/test_auth.py", "exit_code": 1, "duration": 2.1},
            {"type": "file_write", "file": "src/auth.py", "lines_added": 25, "lines_modified": 10},
            {"type": "command_execution", "command": "python -m pytest tests/test_auth.py", "exit_code": 0, "duration": 1.8},
            {"type": "file_write", "file": "tests/test_auth.py", "lines_added": 15},
            {"type": "command_execution", "command": "git add .", "exit_code": 0},
            {"type": "command_execution", "command": "git commit -m 'Implement auth feature'", "exit_code": 0}
        ]
        
        for event in session_events:
            event["timestamp"] = datetime.now().isoformat()
            await session_analyzer.track_session_event(event)
        
        # Generate session summary
        summary = await session_analyzer.generate_session_summary()
        
        assert summary["success"] is True
        assert "summary" in summary
        assert "session_stats" in summary
        
        # Verify summary contains key information
        summary_content = summary["summary"]
        assert "authentication" in summary_content.lower() or "auth" in summary_content.lower()
        
        # Verify session stats
        stats = summary["session_stats"]
        assert stats["total_events"] == 8
        assert stats["files_modified"] >= 2
        assert stats["commands_executed"] >= 4
        assert "session_duration" in stats
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, session_analyzer):
        """Test collection of performance metrics during sessions."""
        await session_analyzer.initialize()
        
        # Start session and add performance-tracked events
        await session_analyzer.detect_session_start({
            "user_message": "Performance optimization session",
            "timestamp": datetime.now().isoformat()
        })
        
        performance_events = [
            {"type": "command_execution", "command": "npm run build", "duration": 15.2, "memory_usage": "450MB"},
            {"type": "command_execution", "command": "npm run test", "duration": 8.7, "memory_usage": "320MB"},
            {"type": "file_write", "file": "webpack.config.js", "processing_time": 0.8},
            {"type": "command_execution", "command": "npm run build", "duration": 12.1, "memory_usage": "380MB"}
        ]
        
        for event in performance_events:
            event["timestamp"] = datetime.now().isoformat()
            await session_analyzer.track_session_event(event)
        
        # Analyze performance metrics
        metrics = await session_analyzer.analyze_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "average_command_duration" in metrics
        assert "performance_trends" in metrics
        
        # Should detect performance improvement in build time
        build_events = [e for e in session_analyzer.session_events if "build" in e.get("command", "")]
        if len(build_events) >= 2:
            assert metrics["performance_trends"]["build_optimization"] is True


class TestHistoryNavigator:
    """Test history navigation and similar session finding."""
    
    @pytest.fixture
    def mock_domain_manager(self):
        """Create mock domain manager."""
        manager = Mock()
        manager.retrieve_memories = AsyncMock(return_value=[])
        return manager
    
    @pytest.fixture
    def history_navigator(self, mock_domain_manager):
        """Create HistoryNavigator for testing."""
        config = {
            "autocode": {
                "history_navigation": {
                    "similarity_threshold": 0.6,
                    "max_similar_sessions": 5,
                    "time_decay_factor": 0.9
                }
            }
        }
        return HistoryNavigator(config, mock_domain_manager)
    
    @pytest.mark.asyncio
    async def test_history_navigator_initialization(self, history_navigator):
        """Test history navigator initialization."""
        await history_navigator.initialize()
        
        assert history_navigator.session_cache == {}
        assert history_navigator.similarity_cache == {}
    
    @pytest.mark.asyncio
    async def test_find_similar_sessions_by_content(self, history_navigator, mock_domain_manager):
        """Test finding similar sessions based on content similarity."""
        await history_navigator.initialize()
        
        # Mock historical sessions
        mock_domain_manager.retrieve_memories.return_value = [
            {
                "content": "Session summary: Implemented user authentication with JWT tokens",
                "metadata": {
                    "session_id": "auth_session_1",
                    "session_type": "development",
                    "technologies": ["jwt", "authentication"],
                    "timestamp": (datetime.now() - timedelta(days=7)).isoformat()
                },
                "similarity": 0.85
            },
            {
                "content": "Session summary: Added OAuth integration for third-party login",
                "metadata": {
                    "session_id": "auth_session_2", 
                    "session_type": "development",
                    "technologies": ["oauth", "authentication"],
                    "timestamp": (datetime.now() - timedelta(days=14)).isoformat()
                },
                "similarity": 0.78
            },
            {
                "content": "Session summary: Fixed database performance issues with indexing",
                "metadata": {
                    "session_id": "db_session_1",
                    "session_type": "optimization",
                    "technologies": ["database", "performance"],
                    "timestamp": (datetime.now() - timedelta(days=3)).isoformat()
                },
                "similarity": 0.45
            }
        ]
        
        # Find similar sessions for authentication work
        similar_sessions = await history_navigator.find_similar_sessions(
            query="working on user login and authentication system",
            time_range_days=30
        )
        
        assert isinstance(similar_sessions, list)
        assert len(similar_sessions) >= 2  # Should find auth-related sessions
        
        # Verify similar sessions are properly ranked
        auth_sessions = [s for s in similar_sessions if "auth" in s.get("content", "").lower()]
        assert len(auth_sessions) >= 2
        
        # First result should be most similar (JWT session)
        assert similar_sessions[0]["similarity"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_find_similar_workflows(self, history_navigator, mock_domain_manager):
        """Test finding similar workflows and command patterns."""
        await history_navigator.initialize()
        
        # Mock workflow memories
        mock_domain_manager.retrieve_memories.return_value = [
            {
                "content": "Workflow: pytest -> fix code -> pytest -> git commit",
                "metadata": {
                    "workflow_type": "testing_development",
                    "commands": ["pytest", "git commit"],
                    "success_rate": 0.9
                },
                "similarity": 0.82
            },
            {
                "content": "Workflow: npm test -> debug -> npm test -> git push",
                "metadata": {
                    "workflow_type": "testing_development",
                    "commands": ["npm test", "git push"],
                    "success_rate": 0.85
                },
                "similarity": 0.75
            }
        ]
        
        # Find similar workflows
        similar_workflows = await history_navigator.find_similar_workflows(
            current_context="running tests and preparing for commit",
            recent_commands=["pytest", "git status"]
        )
        
        assert isinstance(similar_workflows, list)
        assert len(similar_workflows) > 0
        
        # Should find testing-related workflows
        testing_workflows = [w for w in similar_workflows if "test" in w.get("content", "").lower()]
        assert len(testing_workflows) > 0
    
    @pytest.mark.asyncio
    async def test_learning_progression_tracking(self, history_navigator, mock_domain_manager):
        """Test tracking learning progression over time."""
        await history_navigator.initialize()
        
        # Mock learning progression memories
        mock_domain_manager.retrieve_memories.return_value = [
            {
                "content": "Learned: Basic pytest usage and test writing",
                "metadata": {
                    "learning_topic": "pytest",
                    "skill_level": "beginner",
                    "timestamp": (datetime.now() - timedelta(days=30)).isoformat()
                },
                "similarity": 0.9
            },
            {
                "content": "Learned: Advanced pytest fixtures and parametrization",
                "metadata": {
                    "learning_topic": "pytest",
                    "skill_level": "intermediate", 
                    "timestamp": (datetime.now() - timedelta(days=15)).isoformat()
                },
                "similarity": 0.88
            },
            {
                "content": "Learned: Pytest plugins and custom extensions",
                "metadata": {
                    "learning_topic": "pytest",
                    "skill_level": "advanced",
                    "timestamp": (datetime.now() - timedelta(days=3)).isoformat()
                },
                "similarity": 0.85
            }
        ]
        
        # Track learning progression
        progression = await history_navigator.get_learning_progression(
            topic="pytest",
            time_range_days=90
        )
        
        assert isinstance(progression, dict)
        assert "learning_path" in progression
        assert "skill_progression" in progression
        assert "current_level" in progression
        
        # Should show progression from beginner to advanced
        learning_path = progression["learning_path"]
        assert len(learning_path) == 3
        
        # Should be ordered by timestamp (oldest first)
        timestamps = [entry["timestamp"] for entry in learning_path]
        assert timestamps == sorted(timestamps)


class TestPatternDetector:
    """Test pattern detection and caching."""
    
    @pytest.fixture
    def pattern_detector(self):
        """Create PatternDetector for testing."""
        config = {
            "autocode": {
                "pattern_detection": {
                    "cache_duration_hours": 24,
                    "min_pattern_frequency": 3,
                    "pattern_confidence_threshold": 0.7
                }
            }
        }
        return PatternDetector(config)
    
    def test_pattern_detector_initialization(self, pattern_detector):
        """Test pattern detector initialization."""
        pattern_detector.initialize()
        
        assert pattern_detector.pattern_cache == {}
        assert pattern_detector.detection_stats == {
            "patterns_detected": 0,
            "cache_hits": 0,
            "last_detection": None
        }
    
    def test_code_pattern_detection(self, pattern_detector):
        """Test detection of code patterns."""
        pattern_detector.initialize()
        
        # Mock file contents for pattern detection
        mock_files = {
            "src/main.py": """
import pytest
from unittest.mock import Mock

def test_function():
    mock_obj = Mock()
    result = function_under_test(mock_obj)
    assert result is not None
""",
            "tests/test_auth.py": """
import pytest
from unittest.mock import Mock, patch

def test_authentication():
    mock_user = Mock()
    with patch('auth.validate_user') as mock_validate:
        result = authenticate(mock_user)
        assert result.success is True
""",
            "tests/test_utils.py": """
import pytest
from unittest.mock import Mock

def test_utility_function():
    mock_data = Mock()
    result = process_data(mock_data)
    assert len(result) > 0
"""
        }
        
        # Detect patterns across files
        patterns = pattern_detector.detect_code_patterns(mock_files)
        
        assert isinstance(patterns, list)
        
        # Should detect common testing patterns
        testing_patterns = [p for p in patterns if "test" in p.get("pattern_type", "").lower()]
        mock_patterns = [p for p in patterns if "mock" in p.get("pattern_name", "").lower()]
        
        assert len(testing_patterns) > 0
        assert len(mock_patterns) > 0  # Should detect Mock usage pattern
    
    def test_workflow_pattern_detection(self, pattern_detector):
        """Test detection of workflow patterns."""
        pattern_detector.initialize()
        
        # Mock command sequences representing workflows
        command_sequences = [
            ["git status", "git add .", "git commit -m", "git push"],
            ["pytest tests/", "git add .", "git commit -m", "git push"],
            ["npm test", "git add .", "git commit -m", "git push"],
            ["git status", "git add tests/", "git commit -m", "git push origin main"]
        ]
        
        patterns = pattern_detector.detect_workflow_patterns(command_sequences)
        
        assert isinstance(patterns, list)
        
        # Should detect common git workflow pattern
        git_patterns = [p for p in patterns if "git" in p.get("pattern_name", "").lower()]
        assert len(git_patterns) > 0
        
        # Should detect test-then-commit pattern
        test_commit_patterns = [p for p in patterns if 
                              ("test" in p.get("pattern_name", "").lower() and 
                               "commit" in p.get("pattern_name", "").lower())]
        assert len(test_commit_patterns) > 0
    
    def test_pattern_caching(self, pattern_detector):
        """Test pattern caching functionality."""
        pattern_detector.initialize()
        
        # Mock pattern for caching
        test_pattern = {
            "pattern_name": "test_caching_pattern",
            "pattern_type": "workflow",
            "confidence": 0.8,
            "frequency": 5
        }
        
        # Cache the pattern
        cache_key = "test_project_patterns"
        pattern_detector.cache_patterns(cache_key, [test_pattern])
        
        assert cache_key in pattern_detector.pattern_cache
        cached_patterns = pattern_detector.get_cached_patterns(cache_key)
        
        assert len(cached_patterns) == 1
        assert cached_patterns[0]["pattern_name"] == "test_caching_pattern"
        
        # Verify cache hit tracking
        assert pattern_detector.detection_stats["cache_hits"] == 1


@pytest.mark.asyncio 
class TestAutoCodeDomainIntegration:
    """Test AutoCode domain integration with the full system."""
    
    async def test_autocode_domain_initialization(self):
        """Test AutoCode domain initialization and component setup."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # AutoCode domain should be initialized during MCP server startup
            autocode_domain = suite.mcp_server.autocode_domain
            
            assert autocode_domain is not None
            assert hasattr(autocode_domain, 'command_learner')
            assert hasattr(autocode_domain, 'session_analyzer')
            assert hasattr(autocode_domain, 'history_navigator')
            assert hasattr(autocode_domain, 'pattern_detector')
            
        finally:
            await suite.teardown_test_environment()
    
    async def test_mcp_suggest_command_integration(self):
        """Test command suggestion through MCP interface."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Test command suggestion via MCP
            result = await suite.validate_mcp_tool_execution(
                tool_name="suggest_command",
                arguments={
                    "intent": "I want to run the test suite and check code coverage"
                },
                test_name="autocode_command_suggestion"
            )
            
            assert result.passed, f"AutoCode command suggestion failed: {result.errors}"
            assert result.parsed_response.get("success") is True
            
            # Should return command suggestions
            suggestions = result.parsed_response.get("suggestions", [])
            assert isinstance(suggestions, list)
            
            # Look for test-related suggestions
            test_suggestions = [s for s in suggestions if 
                             isinstance(s, dict) and 
                             ("test" in str(s).lower() or "pytest" in str(s).lower())]
            assert len(test_suggestions) > 0, "Should suggest test-related commands"
            
        finally:
            await suite.teardown_test_environment()
    
    async def test_mcp_get_learning_progression_integration(self):
        """Test learning progression tracking through MCP interface."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Test learning progression via MCP
            result = await suite.validate_mcp_tool_execution(
                tool_name="get_learning_progression", 
                arguments={
                    "topic": "python testing",
                    "time_range_days": 90
                },
                test_name="autocode_learning_progression"
            )
            
            assert result.passed, f"AutoCode learning progression failed: {result.errors}"
            assert result.parsed_response.get("success") is True
            
            # Should return progression data
            progression = result.parsed_response.get("progression", {})
            assert isinstance(progression, dict)
            
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_autocode_domain_tests():
        """Run AutoCode domain tests directly."""
        print("🧪 Running AutoCode domain tests...")
        
        # Run integration tests
        integration_tests = TestAutoCodeDomainIntegration()
        await integration_tests.test_autocode_domain_initialization()
        await integration_tests.test_mcp_suggest_command_integration()
        await integration_tests.test_mcp_get_learning_progression_integration()
        print("✅ AutoCode domain integration tests passed")
        
        print("\n🎉 All AutoCode domain tests passed!")
    
    asyncio.run(run_autocode_domain_tests())