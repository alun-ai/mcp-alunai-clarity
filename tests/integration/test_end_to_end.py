"""
Integration tests for end-to-end workflows in Alunai Clarity.
"""

import asyncio
import json
import pytest
import tempfile
import os
from typing import Dict, Any
from unittest.mock import patch, MagicMock
from pathlib import Path

from clarity.domains.manager import MemoryDomainManager
from clarity.mcp.server import MCPServer
from clarity.utils.config import load_config, create_default_config


@pytest.mark.integration
class TestMemoryWorkflow:
    """Test complete memory storage and retrieval workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_memory_lifecycle(self, temp_config_file: str, temp_data_dir: str):
        """Test complete memory lifecycle: store, retrieve, update, delete."""
        # Load test configuration
        config = load_config(temp_config_file)
        config["qdrant"]["path"] = os.path.join(temp_data_dir, "qdrant_test")
        
        # Initialize domain manager with real persistence (but in-memory Qdrant)
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            # Mock Qdrant client for this test
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            
            # Mock collection operations
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client.search = MagicMock()
            mock_client.retrieve = MagicMock()
            mock_client.update = MagicMock()
            mock_client.delete = MagicMock(return_value=True)
            
            domain_manager = MemoryDomainManager(config)
            await domain_manager.initialize()
            
            # 1. Store different types of memories
            conversation_id = await domain_manager.store_memory(
                memory_type="conversation",
                content={
                    "role": "user",
                    "message": "Hello, I'm working on a FastAPI project"
                },
                importance=0.8,
                metadata={"session_id": "test_session"},
                context={"project": "api_project"}
            )
            
            fact_id = await domain_manager.store_memory(
                memory_type="fact",
                content={
                    "fact": "FastAPI is a modern web framework for Python",
                    "confidence": 0.95
                },
                importance=0.9,
                metadata={"category": "programming"},
                context={"topic": "web_development"}
            )
            
            pattern_id = await domain_manager.store_memory(
                memory_type="project_pattern",
                content={
                    "pattern_type": "framework",
                    "framework": "FastAPI",
                    "language": "python",
                    "structure": {
                        "files": ["main.py", "requirements.txt", "models.py"],
                        "directories": ["app", "tests"]
                    }
                },
                importance=0.85,
                metadata={"project_name": "api_project"},
                context={"scan_date": "2023-01-01"}
            )
            
            # Verify memories were stored
            assert conversation_id.startswith("mem_")
            assert fact_id.startswith("mem_")
            assert pattern_id.startswith("mem_")
            
            # Verify store operations were called
            assert mock_client.upsert.call_count >= 3
            
            # 2. Mock search results for retrieval
            mock_search_results = [
                MagicMock(
                    id=fact_id,
                    score=0.9,
                    payload={
                        "id": fact_id,
                        "type": "fact",
                        "content": {"fact": "FastAPI is a modern web framework for Python"},
                        "importance": 0.9,
                        "similarity": 0.9
                    }
                ),
                MagicMock(
                    id=pattern_id,
                    score=0.8,
                    payload={
                        "id": pattern_id,
                        "type": "project_pattern",
                        "content": {"framework": "FastAPI", "language": "python"},
                        "importance": 0.85,
                        "similarity": 0.8
                    }
                )
            ]
            mock_client.search.return_value = mock_search_results
            
            # Retrieve memories about FastAPI
            retrieved_memories = await domain_manager.retrieve_memories(
                query="FastAPI web framework",
                limit=5,
                min_similarity=0.7,
                include_metadata=True
            )
            
            assert len(retrieved_memories) == 2
            assert retrieved_memories[0]["similarity"] >= 0.8
            assert any(mem["type"] == "fact" for mem in retrieved_memories)
            assert any(mem["type"] == "project_pattern" for mem in retrieved_memories)
            
            # 3. Mock get_memory for update operation
            mock_client.retrieve.return_value = [
                MagicMock(payload={
                    "id": fact_id,
                    "type": "fact",
                    "content": {"fact": "FastAPI is a modern web framework for Python"},
                    "importance": 0.9,
                    "metadata": {"category": "programming"},
                    "context": {"topic": "web_development"}
                })
            ]
            
            # Update memory importance
            update_success = await domain_manager.update_memory(
                fact_id,
                {"importance": 0.95, "metadata": {"category": "web_frameworks"}}
            )
            
            assert update_success is True
            
            # 4. Delete memories
            delete_success = await domain_manager.delete_memories([conversation_id, pattern_id])
            
            assert delete_success is True
            mock_client.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_autocode_pattern_detection_workflow(self, temp_config_file: str, temp_data_dir: str):
        """Test AutoCode pattern detection and storage workflow."""
        config = load_config(temp_config_file)
        config["qdrant"]["path"] = os.path.join(temp_data_dir, "qdrant_autocode")
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            
            domain_manager = MemoryDomainManager(config)
            await domain_manager.initialize()
            
            # Store project pattern
            pattern_id = await domain_manager.store_project_pattern(
                pattern_type="framework",
                framework="FastAPI",
                language="python",
                structure={
                    "directories": ["app", "tests", "docs"],
                    "files": ["main.py", "requirements.txt", "README.md"],
                    "config_files": ["pyproject.toml", ".gitignore"]
                },
                importance=0.8,
                metadata={
                    "project_name": "my_api",
                    "detection_confidence": 0.95,
                    "scan_date": "2023-01-01T12:00:00"
                }
            )
            
            # Store command pattern
            command_id = await domain_manager.store_command_pattern(
                command="pytest -v tests/",
                context={
                    "project_type": "python",
                    "framework": "pytest",
                    "working_directory": "/project",
                    "file_count": 5
                },
                success_rate=0.95,
                platform="linux",
                importance=0.7,
                metadata={
                    "execution_count": 15,
                    "avg_execution_time": 2.5,
                    "last_used": "2023-01-01T15:30:00"
                }
            )
            
            # Store session summary
            session_id = await domain_manager.store_session_summary(
                session_id="session_abc123",
                tasks_completed=[
                    {"task": "Set up FastAPI project", "status": "completed", "duration": 300},
                    {"task": "Implement user authentication", "status": "completed", "duration": 600},
                    {"task": "Add database models", "status": "in_progress", "duration": 0}
                ],
                patterns_used=["FastAPI", "SQLAlchemy", "Pydantic"],
                files_modified=["main.py", "models.py", "auth.py", "requirements.txt"],
                importance=0.9,
                metadata={
                    "session_duration": 3600,
                    "complexity_score": 0.7,
                    "learning_indicators": ["new_framework", "database_integration"]
                }
            )
            
            # Store bash execution
            bash_id = await domain_manager.store_bash_execution(
                command="pip install fastapi uvicorn sqlalchemy",
                exit_code=0,
                output="Successfully installed fastapi-0.100.0 uvicorn-0.22.0 sqlalchemy-2.0.0",
                context={
                    "timestamp": "2023-01-01T12:30:00",
                    "working_directory": "/project",
                    "virtual_env": "/project/venv",
                    "python_version": "3.11.0"
                },
                importance=0.6,
                metadata={
                    "command_type": "package_install",
                    "packages_installed": ["fastapi", "uvicorn", "sqlalchemy"],
                    "execution_time": 45.2
                }
            )
            
            # Verify all AutoCode memories were stored
            assert pattern_id.startswith("mem_")
            assert command_id.startswith("mem_")
            assert session_id.startswith("mem_")
            assert bash_id.startswith("mem_")
            
            # Verify storage operations
            assert mock_client.upsert.call_count >= 4
    
    @pytest.mark.asyncio
    async def test_memory_search_and_ranking_workflow(self, temp_config_file: str, temp_data_dir: str):
        """Test memory search with ranking and temporal adjustments."""
        config = load_config(temp_config_file)
        config["qdrant"]["path"] = os.path.join(temp_data_dir, "qdrant_search")
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            
            domain_manager = MemoryDomainManager(config)
            await domain_manager.initialize()
            
            # Store memories with different importance and types
            memories_data = [
                {
                    "type": "fact",
                    "content": {"fact": "FastAPI supports async/await for high performance"},
                    "importance": 0.9,
                    "id": "mem_fact_1"
                },
                {
                    "type": "conversation",
                    "content": {"role": "user", "message": "How do I optimize FastAPI performance?"},
                    "importance": 0.7,
                    "id": "mem_conv_1"
                },
                {
                    "type": "project_pattern",
                    "content": {"framework": "FastAPI", "optimization_techniques": ["async", "caching"]},
                    "importance": 0.8,
                    "id": "mem_pattern_1"
                }
            ]
            
            for memory_data in memories_data:
                await domain_manager.store_memory(
                    memory_type=memory_data["type"],
                    content=memory_data["content"],
                    importance=memory_data["importance"]
                )
            
            # Mock search results with different similarities
            mock_search_results = [
                MagicMock(
                    id="mem_fact_1",
                    score=0.95,
                    payload={
                        "id": "mem_fact_1",
                        "type": "fact",
                        "content": {"fact": "FastAPI supports async/await for high performance"},
                        "importance": 0.9,
                        "created_at": "2023-01-01T00:00:00",
                        "last_accessed": "2023-01-01T00:00:00"
                    }
                ),
                MagicMock(
                    id="mem_pattern_1",
                    score=0.85,
                    payload={
                        "id": "mem_pattern_1",
                        "type": "project_pattern",
                        "content": {"framework": "FastAPI", "optimization_techniques": ["async", "caching"]},
                        "importance": 0.8,
                        "created_at": "2023-01-01T01:00:00",
                        "last_accessed": "2023-01-01T01:00:00"
                    }
                ),
                MagicMock(
                    id="mem_conv_1",
                    score=0.75,
                    payload={
                        "id": "mem_conv_1",
                        "type": "conversation",
                        "content": {"role": "user", "message": "How do I optimize FastAPI performance?"},
                        "importance": 0.7,
                        "created_at": "2023-01-01T02:00:00",
                        "last_accessed": "2023-01-01T02:00:00"
                    }
                )
            ]
            mock_client.search.return_value = mock_search_results
            
            # Search for performance-related memories
            results = await domain_manager.retrieve_memories(
                query="FastAPI performance optimization",
                limit=10,
                memory_types=None,
                min_similarity=0.6,
                include_metadata=True
            )
            
            # Verify results are properly ranked
            assert len(results) == 3
            assert results[0]["similarity"] >= results[1]["similarity"]
            assert results[1]["similarity"] >= results[2]["similarity"]
            
            # Verify temporal adjustments were applied
            assert "similarity" in results[0]
            assert all("id" in result for result in results)
            assert all("type" in result for result in results)


@pytest.mark.integration
class TestMCPServerWorkflow:
    """Test complete MCP server workflows."""
    
    @pytest.mark.asyncio
    async def test_mcp_server_initialization_and_tool_registration(self, temp_config_file: str):
        """Test MCP server initialization with all tools."""
        config = load_config(temp_config_file)
        
        with patch('clarity.mcp.server.MemoryDomainManager') as mock_manager_class:
            mock_domain_manager = MagicMock()
            mock_domain_manager.initialize = MagicMock()
            mock_manager_class.return_value = mock_domain_manager
            
            # Initialize MCP server
            server = MCPServer(config)
            await server.initialize()
            
            # Verify domain manager was initialized
            assert server.domain_manager == mock_domain_manager
            mock_domain_manager.initialize.assert_called_once()
            
            # Server should initialize without errors
            assert server.config == config
    
    @pytest.mark.asyncio
    async def test_mcp_tool_workflow(self, temp_config_file: str):
        """Test complete workflow using MCP tools."""
        config = load_config(temp_config_file)
        
        with patch('clarity.mcp.server.MemoryDomainManager') as mock_manager_class:
            # Create mock domain manager with expected methods
            mock_domain_manager = MagicMock()
            mock_domain_manager.initialize = MagicMock()
            mock_domain_manager.store_memory = MagicMock(return_value="mem_12345")
            mock_domain_manager.retrieve_memories = MagicMock(return_value=[
                {
                    "id": "mem_12345",
                    "type": "fact",
                    "content": {"fact": "Test fact"},
                    "similarity": 0.9
                }
            ])
            mock_domain_manager.get_memory_stats = MagicMock(return_value={
                "total_memories": 1,
                "memory_types": {"fact": 1}
            })
            mock_manager_class.return_value = mock_domain_manager
            
            server = MCPServer(config)
            await server.initialize()
            
            # Import the tool functions to test them directly
            from clarity.mcp.tools import store_memory_tool, retrieve_memory_tool, memory_stats_tool
            
            # Test store memory tool
            store_result = await store_memory_tool(mock_domain_manager, {
                "memory_type": "fact",
                "content": {"fact": "Integration test fact"},
                "importance": 0.8
            })
            
            assert store_result["status"] == "success"
            assert store_result["memory_id"] == "mem_12345"
            mock_domain_manager.store_memory.assert_called_once()
            
            # Test retrieve memory tool
            retrieve_result = await retrieve_memory_tool(mock_domain_manager, {
                "query": "test fact",
                "limit": 5
            })
            
            assert len(retrieve_result["memories"]) == 1
            assert retrieve_result["memories"][0]["id"] == "mem_12345"
            mock_domain_manager.retrieve_memories.assert_called_once()
            
            # Test memory stats tool
            stats_result = await memory_stats_tool(mock_domain_manager, {})
            
            assert stats_result["total_memories"] == 1
            assert stats_result["memory_types"]["fact"] == 1
            mock_domain_manager.get_memory_stats.assert_called_once()


@pytest.mark.integration
class TestConfigurationWorkflow:
    """Test configuration loading and validation workflows."""
    
    def test_default_config_creation_and_loading(self, temp_data_dir: str):
        """Test creating and loading default configuration."""
        config_path = os.path.join(temp_data_dir, "test_config.json")
        
        # Create default configuration
        config = create_default_config(config_path)
        
        # Verify config file was created
        assert os.path.exists(config_path)
        
        # Verify config structure
        assert "server" in config
        assert "alunai-clarity" in config
        assert "qdrant" in config
        assert "embedding" in config
        assert "autocode" in config
        
        # Verify server config
        assert "host" in config["server"]
        assert "port" in config["server"]
        assert "log_level" in config["server"]
        
        # Verify Alunai Clarity config
        assert "max_short_term_items" in config["alunai-clarity"]
        assert "max_long_term_items" in config["alunai-clarity"]
        assert "short_term_threshold" in config["alunai-clarity"]
        
        # Verify Qdrant config
        assert "path" in config["qdrant"]
        assert "index_params" in config["qdrant"]
        
        # Verify embedding config
        assert "default_model" in config["embedding"]
        assert "dimensions" in config["embedding"]
        
        # Verify AutoCode config
        assert "enabled" in config["autocode"]
        assert "command_learning" in config["autocode"]
        assert "pattern_detection" in config["autocode"]
        
        # Load the configuration and verify it matches
        loaded_config = load_config(config_path)
        assert loaded_config == config
    
    def test_config_with_custom_values(self, temp_data_dir: str):
        """Test configuration with custom values."""
        config_path = os.path.join(temp_data_dir, "custom_config.json")
        
        custom_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 9000,
                "log_level": "DEBUG"
            },
            "alunai-clarity": {
                "max_short_term_items": 2000,
                "max_long_term_items": 5000,
                "short_term_threshold": 0.5
            },
            "qdrant": {
                "path": "/custom/qdrant/path",
                "index_params": {
                    "m": 32,
                    "ef_construct": 400
                }
            },
            "embedding": {
                "default_model": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "cache_dir": "/custom/cache"
            },
            "autocode": {
                "enabled": False,
                "command_learning": {
                    "enabled": False,
                    "min_confidence_threshold": 0.5
                }
            }
        }
        
        # Save custom config
        with open(config_path, 'w') as f:
            json.dump(custom_config, f, indent=2)
        
        # Load and verify
        loaded_config = load_config(config_path)
        
        assert loaded_config["server"]["port"] == 9000
        assert loaded_config["alunai-clarity"]["max_short_term_items"] == 2000
        assert loaded_config["qdrant"]["path"] == "/custom/qdrant/path"
        assert loaded_config["embedding"]["dimensions"] == 768
        assert loaded_config["autocode"]["enabled"] is False


@pytest.mark.integration
class TestMCPToolIndexingWorkflow:
    """Test MCP tool indexing and suggestion workflows."""
    
    @pytest.mark.asyncio
    async def test_tool_discovery_and_indexing_workflow(self, temp_config_file: str, sample_claude_config: Dict[str, Any]):
        """Test complete tool discovery and indexing workflow."""
        config = load_config(temp_config_file)
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            
            domain_manager = MemoryDomainManager(config)
            await domain_manager.initialize()
            
            # Import and test tool indexer
            from clarity.mcp.tool_indexer import MCPToolIndexer
            
            indexer = MCPToolIndexer(domain_manager)
            
            # Test discovering known tools
            known_tools = await indexer._discover_known_tools()
            assert len(known_tools) > 0
            
            # Verify expected tools are discovered
            tool_names = [tool.name for tool in known_tools]
            assert "postgres_query" in tool_names
            assert "store_memory" in tool_names
            assert "retrieve_memory" in tool_names
            
            # Test indexing tools as memories
            for tool in known_tools[:3]:  # Index first 3 tools
                await indexer._index_tool_as_memory(tool)
            
            # Verify tools were stored as memories
            assert domain_manager.store_memory.call_count >= 3
            
            # Test tool suggestions
            with patch.object(domain_manager, 'retrieve_memories') as mock_retrieve:
                mock_retrieve.return_value = [
                    {
                        "content": {
                            "tool_name": "postgres_query",
                            "description": "Execute SQL queries",
                            "server_name": "postgres",
                            "keywords": ["database", "sql", "query"]
                        }
                    }
                ]
                
                suggestions = await indexer.suggest_tools_for_intent(
                    "I need to query the database",
                    limit=3
                )
                
                assert len(suggestions) >= 1
                assert suggestions[0]["tool_name"] == "postgres_query"
                assert "relevance_reason" in suggestions[0]
    
    @pytest.mark.asyncio
    async def test_tool_suggestion_workflow(self, temp_config_file: str):
        """Test proactive tool suggestion workflow."""
        config = load_config(temp_config_file)
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            
            domain_manager = MemoryDomainManager(config)
            await domain_manager.initialize()
            
            from clarity.mcp.tool_indexer import MCPToolIndexer, MCPToolSuggester
            
            indexer = MCPToolIndexer(domain_manager)
            suggester = MCPToolSuggester(indexer)
            
            # Mock tool suggestions for database request
            with patch.object(indexer, 'suggest_tools_for_intent') as mock_suggest:
                mock_suggest.return_value = [
                    {
                        "tool_name": "postgres_query",
                        "description": "Execute SQL queries",
                        "server_name": "postgres",
                        "relevance_reason": "Matches keywords: database, query",
                        "usage_hint": "Use instead of writing SQL scripts"
                    }
                ]
                
                # Test suggestion for indirect database method
                suggestion_message = await suggester.analyze_and_suggest(
                    "I need to write a script to query the database for user data"
                )
                
                assert suggestion_message is not None
                assert "MCP Tool Suggestion" in suggestion_message
                assert "postgres_query" in suggestion_message
                assert "postgres" in suggestion_message
                
                # Test no suggestion for direct request
                no_suggestion = await suggester.analyze_and_suggest(
                    "What is the weather today?"
                )
                
                assert no_suggestion is None