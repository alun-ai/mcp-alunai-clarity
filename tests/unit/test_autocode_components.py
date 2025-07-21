"""
Unit tests for refactored AutoCode domain components in Alunai Clarity.

Tests the modular components created during the autocode/domain.py decomposition:
- ProjectPatternManager
- SessionManager  
- LearningEngine
- StatsCollector
"""

import asyncio
import json
import pytest
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import os

from clarity.autocode.interfaces import (
    ProjectPatternManager,
    SessionManager,
    LearningEngine,
    StatsCollector
)
from clarity.autocode.components.project_patterns import ProjectPatternManagerImpl
from clarity.autocode.components.session_manager import SessionManagerImpl
from clarity.autocode.components.learning_engine import LearningEngineImpl
from clarity.autocode.components.stats_collector import StatsCollectorImpl
from clarity.shared.exceptions import AutoCodeError


@pytest.mark.unit
class TestProjectPatternManager:
    """Test ProjectPatternManager component."""
    
    @pytest.fixture
    def mock_persistence_domain(self):
        """Create mock persistence domain."""
        domain = AsyncMock()
        domain.store_memory = AsyncMock(return_value="pattern_123")
        domain.retrieve_memories = AsyncMock(return_value=[])
        return domain
    
    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return {
            "autocode": {
                "pattern_detection": {
                    "enabled": True,
                    "supported_languages": ["python", "javascript"],
                    "max_scan_depth": 3
                }
            }
        }
    
    @pytest.fixture
    async def pattern_manager(self, test_config, mock_persistence_domain):
        """Create pattern manager instance."""
        manager = ProjectPatternManagerImpl(test_config, mock_persistence_domain)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_initialization(self, test_config, mock_persistence_domain):
        """Test pattern manager initialization."""
        manager = ProjectPatternManagerImpl(test_config, mock_persistence_domain)
        
        assert not manager._initialized
        
        await manager.initialize()
        
        assert manager._initialized
        assert manager.pattern_cache is not None
    
    @pytest.mark.asyncio
    async def test_get_project_patterns_with_cache(self, pattern_manager):
        """Test getting project patterns with caching."""
        project_path = "/test/project"
        
        # Mock pattern detection
        with patch.object(pattern_manager, 'detect_project_patterns', 
                         return_value={"framework": "FastAPI", "language": "python"}) as mock_detect:
            
            # First call should detect patterns
            patterns1 = await pattern_manager.get_project_patterns(project_path)
            assert patterns1["framework"] == "FastAPI"
            mock_detect.assert_called_once()
            
            # Second call should use cache
            mock_detect.reset_mock()
            patterns2 = await pattern_manager.get_project_patterns(project_path)
            assert patterns2 == patterns1
            mock_detect.assert_not_called()  # Should use cache
    
    @pytest.mark.asyncio 
    async def test_detect_project_patterns(self, pattern_manager):
        """Test project pattern detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test project structure
            os.makedirs(os.path.join(temp_dir, "app"))
            
            # Create requirements.txt with FastAPI
            with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                f.write("fastapi==0.68.0\nuvicorn==0.15.0\n")
            
            # Create main.py
            with open(os.path.join(temp_dir, "main.py"), "w") as f:
                f.write("from fastapi import FastAPI\napp = FastAPI()\n")
            
            patterns = await pattern_manager.detect_project_patterns(temp_dir)
            
            assert "structure" in patterns
            assert "technology_stack" in patterns
            assert "complexity_metrics" in patterns
    
    @pytest.mark.asyncio
    async def test_analyze_project_structure(self, pattern_manager):
        """Test project structure analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            os.makedirs(os.path.join(temp_dir, "src", "models"))
            os.makedirs(os.path.join(temp_dir, "tests"))
            os.makedirs(os.path.join(temp_dir, "docs"))
            
            # Create files
            test_files = [
                "README.md", "setup.py", "requirements.txt",
                "src/__init__.py", "src/main.py", "src/models/user.py",
                "tests/test_main.py", "docs/api.md"
            ]
            
            for file_path in test_files:
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write("# Test content\n")
            
            structure = await pattern_manager.analyze_project_structure(temp_dir)
            
            assert "directories" in structure
            assert "files" in structure
            assert "total_files" in structure
            assert structure["total_files"] == len(test_files)
            assert "src" in structure["directories"]
            assert "tests" in structure["directories"]
    
    @pytest.mark.asyncio
    async def test_detect_technology_stack(self, pattern_manager):
        """Test technology stack detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create package.json for Node.js project
            package_json = {
                "name": "test-project",
                "dependencies": {
                    "express": "^4.18.0",
                    "mongoose": "^6.0.0"
                },
                "devDependencies": {
                    "jest": "^28.0.0"
                }
            }
            
            with open(os.path.join(temp_dir, "package.json"), "w") as f:
                json.dump(package_json, f)
            
            tech_stack = await pattern_manager.detect_technology_stack(temp_dir)
            
            assert "language" in tech_stack
            assert "framework" in tech_stack
            assert "dependencies" in tech_stack
            assert tech_stack["language"] == "javascript"


@pytest.mark.unit
class TestSessionManager:
    """Test SessionManager component."""
    
    @pytest.fixture
    def mock_persistence_domain(self):
        """Create mock persistence domain."""
        domain = AsyncMock()
        domain.store_memory = AsyncMock(return_value="session_123")
        domain.retrieve_memories = AsyncMock(return_value=[])
        return domain
    
    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return {
            "autocode": {
                "session_analysis": {
                    "enabled": True,
                    "track_architectural_decisions": True,
                    "extract_learning_patterns": True
                }
            }
        }
    
    @pytest.fixture
    async def session_manager(self, test_config, mock_persistence_domain):
        """Create session manager instance."""
        manager = SessionManagerImpl(test_config, mock_persistence_domain)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_process_file_access(self, session_manager):
        """Test file access processing."""
        file_path = "/test/project/main.py"
        access_type = "read"
        project_context = {"project_name": "test_project"}
        
        await session_manager.process_file_access(file_path, access_type, project_context)
        
        # Should track the file access
        current_session = session_manager.current_session_data
        assert len(current_session["file_accesses"]) == 1
        assert current_session["file_accesses"][0]["file_path"] == file_path
        assert current_session["file_accesses"][0]["access_type"] == access_type
    
    @pytest.mark.asyncio
    async def test_generate_session_summary(self, session_manager):
        """Test session summary generation."""
        conversation_log = [
            {"role": "user", "message": "Create a FastAPI application"},
            {"role": "assistant", "message": "I'll help you create a FastAPI application..."},
            {"role": "user", "message": "Add database integration"},
            {"role": "assistant", "message": "I'll add PostgreSQL integration..."}
        ]
        
        summary = await session_manager.generate_session_summary(conversation_log)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key information from the conversation
        assert "FastAPI" in summary or "fastapi" in summary
    
    @pytest.mark.asyncio
    async def test_find_similar_sessions(self, session_manager, mock_persistence_domain):
        """Test finding similar sessions."""
        # Mock similar session data
        mock_memories = [
            {
                "id": "session_1",
                "content": {"summary": "Created FastAPI application with database"},
                "similarity": 0.8,
                "metadata": {"session_id": "prev_session_1"}
            }
        ]
        mock_persistence_domain.retrieve_memories.return_value = mock_memories
        
        query = "Create web API with database"
        similar_sessions = await session_manager.find_similar_sessions(query)
        
        assert len(similar_sessions) == 1
        assert similar_sessions[0]["session_id"] == "prev_session_1"
        assert similar_sessions[0]["similarity"] == 0.8
    
    @pytest.mark.asyncio
    async def test_get_context_for_continuation(self, session_manager, mock_persistence_domain):
        """Test getting context for task continuation."""
        # Mock related memories
        mock_memories = [
            {
                "id": "mem_1",
                "content": {"task": "Setup FastAPI project"},
                "metadata": {"task_type": "setup"}
            }
        ]
        mock_persistence_domain.retrieve_memories.return_value = mock_memories
        
        task = "Add authentication to FastAPI project"
        context = await session_manager.get_context_for_continuation(task)
        
        assert "related_memories" in context
        assert "current_task" in context
        assert context["current_task"] == task
        assert len(context["related_memories"]) == 1


@pytest.mark.unit
class TestLearningEngine:
    """Test LearningEngine component."""
    
    @pytest.fixture
    def mock_persistence_domain(self):
        """Create mock persistence domain."""
        domain = AsyncMock()
        domain.store_memory = AsyncMock(return_value="learning_123")
        domain.retrieve_memories = AsyncMock(return_value=[])
        return domain
    
    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return {
            "autocode": {
                "command_learning": {
                    "enabled": True,
                    "min_confidence_threshold": 0.3,
                    "max_suggestions": 3
                }
            }
        }
    
    @pytest.fixture
    async def learning_engine(self, test_config, mock_persistence_domain):
        """Create learning engine instance."""
        engine = LearningEngineImpl(test_config, mock_persistence_domain)
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_process_bash_execution(self, learning_engine):
        """Test bash execution processing."""
        command = "pytest -v"
        working_dir = "/test/project"
        success = True
        output = "All tests passed"
        
        await learning_engine.process_bash_execution(
            command, working_dir, success, output
        )
        
        # Should update command patterns
        assert len(learning_engine.command_patterns) > 0
        pattern = learning_engine.command_patterns[0]
        assert pattern["command"] == command
        assert pattern["success"] == success
    
    @pytest.mark.asyncio
    async def test_suggest_command(self, learning_engine, mock_persistence_domain):
        """Test command suggestion."""
        # Mock command patterns in memory
        mock_patterns = [
            {
                "id": "cmd_1",
                "content": {
                    "command": "pytest -v",
                    "context": {"project_type": "python"},
                    "success_rate": 0.95
                },
                "similarity": 0.8
            }
        ]
        mock_persistence_domain.retrieve_memories.return_value = mock_patterns
        
        intent = "run tests"
        context = {"project_type": "python"}
        
        suggestions = await learning_engine.suggest_command(intent, context)
        
        assert len(suggestions) > 0
        assert suggestions[0]["command"] == "pytest -v"
        assert suggestions[0]["confidence"] >= 0.3  # Above threshold
    
    @pytest.mark.asyncio
    async def test_suggest_workflow_optimizations(self, learning_engine):
        """Test workflow optimization suggestions."""
        current_workflow = [
            "Create virtual environment",
            "Install dependencies", 
            "Write code",
            "Run tests",
            "Commit changes"
        ]
        
        optimizations = await learning_engine.suggest_workflow_optimizations(current_workflow)
        
        assert isinstance(optimizations, list)
        # Should provide optimization suggestions
        if optimizations:
            assert "optimization" in optimizations[0]
            assert "benefit" in optimizations[0]
    
    @pytest.mark.asyncio
    async def test_get_learning_progression(self, learning_engine, mock_persistence_domain):
        """Test learning progression tracking."""
        # Mock progression data
        mock_memories = [
            {
                "id": "prog_1",
                "content": {"topic": "FastAPI", "skill_level": "beginner"},
                "metadata": {"timestamp": "2023-01-01T00:00:00"}
            }
        ]
        mock_persistence_domain.retrieve_memories.return_value = mock_memories
        
        topic = "FastAPI"
        progression = await learning_engine.get_learning_progression(topic)
        
        assert "topic" in progression
        assert "skill_progression" in progression
        assert progression["topic"] == topic


@pytest.mark.unit
class TestStatsCollector:
    """Test StatsCollector component."""
    
    @pytest.fixture
    def mock_persistence_domain(self):
        """Create mock persistence domain."""
        domain = AsyncMock()
        domain.store_memory = AsyncMock(return_value="stats_123")
        return domain
    
    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return {
            "autocode": {
                "stats_collection": {
                    "enabled": True,
                    "collection_interval": 60
                }
            }
        }
    
    @pytest.fixture
    async def stats_collector(self, test_config, mock_persistence_domain):
        """Create stats collector instance."""
        collector = StatsCollectorImpl(test_config, mock_persistence_domain)
        await collector.initialize()
        return collector
    
    @pytest.mark.asyncio
    async def test_register_component(self, stats_collector):
        """Test component registration."""
        mock_component = MagicMock()
        mock_component.get_stats = AsyncMock(return_value={"operations": 42})
        
        stats_collector.register_component("test_component", mock_component)
        
        assert "test_component" in stats_collector.components
    
    @pytest.mark.asyncio
    async def test_track_operation(self, stats_collector):
        """Test operation tracking."""
        operation_name = "test_operation"
        duration = 0.5
        success = True
        context = {"param": "value"}
        
        await stats_collector.track_operation(operation_name, duration, success, context)
        
        stats = await stats_collector.get_stats()
        assert "operations" in stats
        assert operation_name in stats["operations"]
        
        op_stats = stats["operations"][operation_name]
        assert op_stats["count"] == 1
        assert op_stats["success_count"] == 1
        assert op_stats["total_duration"] == duration
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, stats_collector):
        """Test performance metrics collection."""
        # Track some operations
        await stats_collector.track_operation("fast_op", 0.1, True)
        await stats_collector.track_operation("slow_op", 1.0, True)
        await stats_collector.track_operation("failed_op", 0.5, False)
        
        metrics = await stats_collector.get_performance_metrics()
        
        assert "total_operations" in metrics
        assert "success_rate" in metrics
        assert "average_duration" in metrics
        assert metrics["total_operations"] == 3
        assert metrics["success_rate"] == 2/3  # 2 successes out of 3
    
    @pytest.mark.asyncio
    async def test_component_health_monitoring(self, stats_collector):
        """Test component health monitoring."""
        # Register a healthy component
        healthy_component = MagicMock()
        healthy_component.get_stats = AsyncMock(return_value={"status": "healthy"})
        stats_collector.register_component("healthy_comp", healthy_component)
        
        # Register an unhealthy component
        unhealthy_component = MagicMock()
        unhealthy_component.get_stats = AsyncMock(side_effect=Exception("Component error"))
        stats_collector.register_component("unhealthy_comp", unhealthy_component)
        
        health_stats = await stats_collector.get_component_health()
        
        assert "healthy_comp" in health_stats
        assert "unhealthy_comp" in health_stats
        assert health_stats["healthy_comp"]["healthy"] is True
        assert health_stats["unhealthy_comp"]["healthy"] is False
    
    @pytest.mark.asyncio
    async def test_cache_efficiency_analysis(self, stats_collector):
        """Test cache efficiency analysis."""
        from clarity.shared.infrastructure import get_cache
        
        # Create test cache with some activity
        cache = get_cache("test_cache", max_size=100)
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats_collector.register_cache("test_cache", cache)
        
        cache_stats = await stats_collector.get_cache_efficiency()
        
        assert "test_cache" in cache_stats
        assert "hit_rate" in cache_stats["test_cache"]
        assert "size" in cache_stats["test_cache"]


@pytest.mark.integration
class TestAutoCodeComponentsIntegration:
    """Integration tests for AutoCode components working together."""
    
    @pytest.fixture
    def mock_persistence_domain(self):
        """Create comprehensive mock persistence domain."""
        domain = AsyncMock()
        domain.store_memory = AsyncMock(return_value="integration_test_123")
        domain.retrieve_memories = AsyncMock(return_value=[])
        domain.generate_embedding = AsyncMock(return_value=[0.1] * 384)
        return domain
    
    @pytest.fixture
    def test_config(self):
        """Complete test configuration."""
        return {
            "autocode": {
                "enabled": True,
                "pattern_detection": {"enabled": True},
                "session_analysis": {"enabled": True},
                "command_learning": {"enabled": True},
                "stats_collection": {"enabled": True}
            }
        }
    
    @pytest.mark.asyncio
    async def test_components_initialization_order(self, test_config, mock_persistence_domain):
        """Test that components initialize in correct order."""
        from clarity.autocode.domain_refactored import AutoCodeDomainRefactored
        
        domain = AutoCodeDomainRefactored(test_config, mock_persistence_domain)
        
        # Should initialize without errors
        await domain.initialize()
        
        assert domain._initialized
        assert domain.pattern_manager is not None
        assert domain.session_manager is not None
        assert domain.learning_engine is not None
        assert domain.stats_collector is not None
    
    @pytest.mark.asyncio
    async def test_cross_component_data_flow(self, test_config, mock_persistence_domain):
        """Test data flow between components."""
        from clarity.autocode.domain_refactored import AutoCodeDomainRefactored
        
        domain = AutoCodeDomainRefactored(test_config, mock_persistence_domain)
        await domain.initialize()
        
        # Process a bash execution (should involve multiple components)
        await domain.process_bash_execution(
            command="pytest -v",
            working_directory="/test/project", 
            success=True,
            output="All tests passed"
        )
        
        # Stats collector should have tracked this operation
        stats = await domain.get_stats()
        assert "operations" in stats
        assert "process_bash_execution" in stats["operations"]
    
    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, test_config, mock_persistence_domain):
        """Test that performance is tracked across all operations."""
        from clarity.autocode.domain_refactored import AutoCodeDomainRefactored
        
        domain = AutoCodeDomainRefactored(test_config, mock_persistence_domain)
        await domain.initialize()
        
        # Perform various operations
        await domain.get_project_patterns("/test/project")
        await domain.suggest_command("run tests")
        await domain.generate_session_summary([{"role": "user", "message": "test"}])
        
        # Performance metrics should be collected
        metrics = await domain.get_performance_metrics()
        assert "total_operations" in metrics
        assert metrics["total_operations"] >= 3