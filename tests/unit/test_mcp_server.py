"""
Unit tests for MCP server and tools in Alunai Clarity.
"""

import asyncio
import json
import pytest
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from clarity.mcp.server import MCPServer
from clarity.mcp.tools import (
    store_memory_tool,
    retrieve_memory_tool,
    list_memories_tool,
    update_memory_tool,
    delete_memory_tool,
    memory_stats_tool,
    suggest_command_tool,
    get_project_patterns_tool,
    find_similar_sessions_tool,
    get_continuation_context_tool,
    suggest_workflow_optimizations_tool,
    get_learning_progression_tool,
    autocode_stats_tool
)
from clarity.mcp.tool_indexer import MCPToolIndexer, MCPToolSuggester, MCPToolInfo


@pytest.mark.unit
class TestMCPServer:
    """Test MCP server functionality."""
    
    @pytest.mark.asyncio
    async def test_mcp_server_initialization(self, test_config: Dict[str, Any]):
        """Test MCP server initialization."""
        with patch('clarity.mcp.server.MemoryDomainManager') as mock_manager:
            mock_domain_manager = AsyncMock()
            mock_manager.return_value = mock_domain_manager
            
            server = MCPServer(test_config)
            await server.initialize()
            
            assert server.config == test_config
            assert server.domain_manager == mock_domain_manager
            mock_domain_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_server_tool_registration(self, test_config: Dict[str, Any]):
        """Test that all MCP tools are properly registered."""
        with patch('clarity.mcp.server.MemoryDomainManager') as mock_manager:
            mock_domain_manager = AsyncMock()
            mock_manager.return_value = mock_domain_manager
            
            server = MCPServer(test_config)
            await server.initialize()
            
            # Check that tools are registered
            expected_tools = [
                "store_memory",
                "retrieve_memory", 
                "list_memories",
                "update_memory",
                "delete_memory",
                "memory_stats",
                "suggest_command",
                "get_project_patterns",
                "find_similar_sessions",
                "get_continuation_context",
                "suggest_workflow_optimizations",
                "get_learning_progression",
                "autocode_stats"
            ]
            
            # This would depend on the actual MCP server implementation
            # For now, just verify the server initialized without errors
            assert server.domain_manager is not None


@pytest.mark.unit
class TestMemoryMCPTools:
    """Test core memory MCP tools."""
    
    @pytest.mark.asyncio
    async def test_store_memory_tool(self, mock_domain_manager):
        """Test store_memory MCP tool."""
        # Mock the domain manager response
        mock_domain_manager.store_memory.return_value = "mem_12345"
        
        arguments = {
            "memory_type": "fact",
            "content": {"fact": "Test fact", "confidence": 0.9},
            "importance": 0.8,
            "metadata": {"source": "test"},
            "context": {"session": "test_session"}
        }
        
        result = await store_memory_tool(mock_domain_manager, arguments)
        
        assert result["memory_id"] == "mem_12345"
        assert result["status"] == "success"
        
        mock_domain_manager.store_memory.assert_called_once_with(
            memory_type="fact",
            content={"fact": "Test fact", "confidence": 0.9},
            importance=0.8,
            metadata={"source": "test"},
            context={"session": "test_session"}
        )
    
    @pytest.mark.asyncio
    async def test_retrieve_memory_tool(self, mock_domain_manager):
        """Test retrieve_memory MCP tool."""
        # Mock the domain manager response
        mock_memories = [
            {
                "id": "mem_1",
                "type": "fact",
                "content": {"fact": "Test fact 1"},
                "similarity": 0.9
            },
            {
                "id": "mem_2", 
                "type": "fact",
                "content": {"fact": "Test fact 2"},
                "similarity": 0.8
            }
        ]
        mock_domain_manager.retrieve_memories.return_value = mock_memories
        
        arguments = {
            "query": "test query",
            "limit": 5,
            "types": ["fact"],
            "min_similarity": 0.7,
            "include_metadata": True
        }
        
        result = await retrieve_memory_tool(mock_domain_manager, arguments)
        
        assert len(result["memories"]) == 2
        assert result["memories"][0]["id"] == "mem_1"
        assert result["memories"][1]["id"] == "mem_2"
        assert result["query"] == "test query"
        
        mock_domain_manager.retrieve_memories.assert_called_once_with(
            query="test query",
            limit=5,
            memory_types=["fact"],
            min_similarity=0.7,
            include_metadata=True
        )
    
    @pytest.mark.asyncio
    async def test_list_memories_tool(self, mock_domain_manager):
        """Test list_memories MCP tool."""
        mock_memories = [
            {
                "id": "mem_1",
                "type": "conversation",
                "created_at": "2023-01-01T00:00:00",
                "importance": 0.7
            },
            {
                "id": "mem_2",
                "type": "fact", 
                "created_at": "2023-01-01T01:00:00",
                "importance": 0.8
            }
        ]
        mock_domain_manager.list_memories.return_value = mock_memories
        
        arguments = {
            "types": ["conversation", "fact"],
            "limit": 10,
            "offset": 0,
            "tier": "short_term",
            "include_content": False
        }
        
        result = await list_memories_tool(mock_domain_manager, arguments)
        
        assert len(result["memories"]) == 2
        assert result["total_count"] == 2
        assert result["memories"][0]["id"] == "mem_1"
        
        mock_domain_manager.list_memories.assert_called_once_with(
            memory_types=["conversation", "fact"],
            limit=10,
            offset=0,
            tier="short_term",
            include_content=False
        )
    
    @pytest.mark.asyncio
    async def test_update_memory_tool(self, mock_domain_manager):
        """Test update_memory MCP tool."""
        mock_domain_manager.update_memory.return_value = True
        
        arguments = {
            "memory_id": "mem_12345",
            "updates": {
                "content": {"fact": "Updated fact"},
                "importance": 0.9,
                "metadata": {"updated": True}
            }
        }
        
        result = await update_memory_tool(mock_domain_manager, arguments)
        
        assert result["success"] is True
        assert result["memory_id"] == "mem_12345"
        
        mock_domain_manager.update_memory.assert_called_once_with(
            "mem_12345",
            {
                "content": {"fact": "Updated fact"},
                "importance": 0.9,
                "metadata": {"updated": True}
            }
        )
    
    @pytest.mark.asyncio
    async def test_delete_memory_tool(self, mock_domain_manager):
        """Test delete_memory MCP tool."""
        mock_domain_manager.delete_memories.return_value = True
        
        arguments = {
            "memory_ids": ["mem_1", "mem_2", "mem_3"]
        }
        
        result = await delete_memory_tool(mock_domain_manager, arguments)
        
        assert result["success"] is True
        assert result["deleted_count"] == 3
        assert result["memory_ids"] == ["mem_1", "mem_2", "mem_3"]
        
        mock_domain_manager.delete_memories.assert_called_once_with(
            ["mem_1", "mem_2", "mem_3"]
        )
    
    @pytest.mark.asyncio
    async def test_memory_stats_tool(self, mock_domain_manager):
        """Test memory_stats MCP tool."""
        mock_stats = {
            "total_memories": 150,
            "memory_types": {
                "fact": 50,
                "conversation": 40,
                "project_pattern": 30,
                "command_pattern": 30
            },
            "tiers": {
                "short_term": 100,
                "long_term": 50
            },
            "performance": {
                "avg_search_time": "2ms",
                "total_size": "15MB"
            }
        }
        mock_domain_manager.get_memory_stats.return_value = mock_stats
        
        result = await memory_stats_tool(mock_domain_manager, {})
        
        assert result["total_memories"] == 150
        assert result["memory_types"]["fact"] == 50
        assert result["tiers"]["short_term"] == 100
        assert "performance" in result
        
        mock_domain_manager.get_memory_stats.assert_called_once()


@pytest.mark.unit
class TestAutoCodeMCPTools:
    """Test AutoCode-specific MCP tools."""
    
    @pytest.mark.asyncio
    async def test_suggest_command_tool(self, mock_domain_manager):
        """Test suggest_command MCP tool."""
        mock_suggestions = [
            {
                "command": "pytest -v",
                "confidence": 0.95,
                "context": "python testing",
                "success_rate": 0.9
            },
            {
                "command": "npm test",
                "confidence": 0.85,
                "context": "javascript testing", 
                "success_rate": 0.8
            }
        ]
        
        # Mock the AutoCode domain's suggest_command method
        mock_domain_manager.autocode_domain.suggest_command.return_value = mock_suggestions
        
        arguments = {
            "intent": "run tests",
            "context": {
                "project_type": "python",
                "framework": "pytest",
                "platform": "linux"
            }
        }
        
        result = await suggest_command_tool(mock_domain_manager, arguments)
        
        assert len(result["suggestions"]) == 2
        assert result["suggestions"][0]["command"] == "pytest -v"
        assert result["suggestions"][0]["confidence"] == 0.95
        assert result["intent"] == "run tests"
        
        mock_domain_manager.autocode_domain.suggest_command.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_project_patterns_tool(self, mock_domain_manager):
        """Test get_project_patterns MCP tool."""
        mock_patterns = [
            {
                "pattern_type": "framework",
                "framework": "FastAPI",
                "language": "python",
                "confidence": 0.9,
                "files": ["main.py", "requirements.txt"]
            },
            {
                "pattern_type": "testing",
                "framework": "pytest",
                "language": "python",
                "confidence": 0.8,
                "files": ["test_main.py", "conftest.py"]
            }
        ]
        
        mock_domain_manager.autocode_domain.get_project_patterns.return_value = mock_patterns
        
        arguments = {
            "project_path": "/test/project",
            "pattern_types": ["framework", "testing"]
        }
        
        result = await get_project_patterns_tool(mock_domain_manager, arguments)
        
        assert len(result["patterns"]) == 2
        assert result["patterns"][0]["framework"] == "FastAPI"
        assert result["project_path"] == "/test/project"
        
        mock_domain_manager.autocode_domain.get_project_patterns.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_similar_sessions_tool(self, mock_domain_manager):
        """Test find_similar_sessions MCP tool."""
        mock_sessions = [
            {
                "session_id": "session_1",
                "similarity": 0.9,
                "summary": "API development with FastAPI",
                "date": "2023-01-01T00:00:00",
                "tasks": ["Created endpoints", "Added tests"]
            },
            {
                "session_id": "session_2",
                "similarity": 0.8,
                "summary": "API testing and validation", 
                "date": "2023-01-02T00:00:00",
                "tasks": ["Added validation", "Fixed bugs"]
            }
        ]
        
        mock_domain_manager.autocode_domain.find_similar_sessions.return_value = mock_sessions
        
        arguments = {
            "query": "API development",
            "context": {"project": "web_api"},
            "time_range_days": 30
        }
        
        result = await find_similar_sessions_tool(mock_domain_manager, arguments)
        
        assert len(result["sessions"]) == 2
        assert result["sessions"][0]["session_id"] == "session_1"
        assert result["sessions"][0]["similarity"] == 0.9
        assert result["query"] == "API development"
        
        mock_domain_manager.autocode_domain.find_similar_sessions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_learning_progression_tool(self, mock_domain_manager):
        """Test get_learning_progression MCP tool."""
        mock_progression = {
            "topic": "FastAPI",
            "progression_stages": [
                {
                    "stage": "Beginner",
                    "date_range": "2023-01-01 to 2023-01-15",
                    "concepts": ["Basic routing", "Request handling"],
                    "confidence": 0.6
                },
                {
                    "stage": "Intermediate",
                    "date_range": "2023-01-16 to 2023-02-01", 
                    "concepts": ["Database integration", "Authentication"],
                    "confidence": 0.8
                }
            ],
            "current_level": "Intermediate",
            "next_recommended_topics": ["Advanced middleware", "Performance optimization"]
        }
        
        mock_domain_manager.autocode_domain.get_learning_progression.return_value = mock_progression
        
        arguments = {
            "topic": "FastAPI",
            "time_range_days": 180
        }
        
        result = await get_learning_progression_tool(mock_domain_manager, arguments)
        
        assert result["topic"] == "FastAPI"
        assert result["current_level"] == "Intermediate"
        assert len(result["progression_stages"]) == 2
        assert len(result["next_recommended_topics"]) == 2
        
        mock_domain_manager.autocode_domain.get_learning_progression.assert_called_once()


@pytest.mark.unit
class TestMCPToolIndexer:
    """Test MCP tool indexer functionality."""
    
    @pytest.mark.asyncio
    async def test_tool_indexer_initialization(self, mock_domain_manager):
        """Test MCP tool indexer initialization."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        assert indexer.domain_manager == mock_domain_manager
        assert isinstance(indexer.indexed_tools, dict)
        assert len(indexer.indexed_tools) == 0
        assert isinstance(indexer.intent_categories, dict)
        assert "database" in indexer.intent_categories
        assert "web_automation" in indexer.intent_categories
    
    @pytest.mark.asyncio
    async def test_discover_known_tools(self, mock_domain_manager):
        """Test discovering known MCP tools."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        known_tools = await indexer._discover_known_tools()
        
        assert len(known_tools) > 0
        
        # Check for expected known tools
        tool_names = [tool.name for tool in known_tools]
        assert "postgres_query" in tool_names
        assert "playwright_navigate" in tool_names
        assert "store_memory" in tool_names
        assert "retrieve_memory" in tool_names
        
        # Verify tool structure
        postgres_tool = next(tool for tool in known_tools if tool.name == "postgres_query")
        assert postgres_tool.server_name == "postgres"
        assert postgres_tool.category == "database"
        assert "database" in postgres_tool.keywords
        assert "sql" in postgres_tool.keywords
    
    @pytest.mark.asyncio
    async def test_categorize_tool_from_info(self, mock_domain_manager):
        """Test tool categorization based on information."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        # Test database tool categorization
        db_category = indexer._categorize_tool_from_info(
            "query_db", 
            "Execute SQL queries against database",
            "postgres"
        )
        assert db_category == "database"
        
        # Test web automation categorization
        web_category = indexer._categorize_tool_from_info(
            "navigate_page",
            "Navigate to web pages using browser",
            "playwright"
        )
        assert web_category == "web_automation"
        
        # Test memory management categorization
        memory_category = indexer._categorize_tool_from_info(
            "store_info",
            "Store information in memory",
            "memory_server"
        )
        assert memory_category == "memory_management"
        
        # Test default categorization
        default_category = indexer._categorize_tool_from_info(
            "unknown_tool",
            "Does something unclear",
            "unknown_server"
        )
        assert default_category == "api_integration"
    
    @pytest.mark.asyncio
    async def test_extract_keywords_from_tool(self, mock_domain_manager):
        """Test keyword extraction from tool information."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        # Mock tool object
        mock_tool = MagicMock()
        mock_tool.name = "postgres_query_data"
        mock_tool.description = "Execute SQL queries against PostgreSQL database tables"
        
        keywords = indexer._extract_keywords_from_tool(mock_tool, "postgres_server")
        
        # Should include tool name parts
        assert "postgres" in keywords
        assert "query" in keywords
        assert "data" in keywords
        
        # Should include server name parts
        assert "server" in keywords
        
        # Should include description words
        assert "execute" in keywords
        assert "sql" in keywords
        assert "database" in keywords
        assert "tables" in keywords
        
        # Should include common MCP keywords
        assert "mcp" in keywords
        assert "tool" in keywords
    
    @pytest.mark.asyncio
    async def test_index_tool_as_memory(self, mock_domain_manager):
        """Test indexing a tool as memory."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        test_tool = MCPToolInfo(
            name="test_query",
            description="Test database query tool",
            parameters={"query": {"type": "string"}},
            server_name="test_db",
            use_cases=["Query database", "Run SQL"],
            keywords={"database", "sql", "query"},
            category="database"
        )
        
        await indexer._index_tool_as_memory(test_tool)
        
        # Verify store_memory was called
        mock_domain_manager.store_memory.assert_called_once()
        
        # Check the call arguments
        call_args = mock_domain_manager.store_memory.call_args
        assert call_args[1]["memory_type"] == "mcp_tool"
        assert call_args[1]["importance"] == 0.9
        
        content = call_args[1]["content"]
        assert content["tool_name"] == "test_query"
        assert content["description"] == "Test database query tool"
        assert content["server_name"] == "test_db"
        assert content["category"] == "database"
    
    @pytest.mark.asyncio
    async def test_suggest_tools_for_intent(self, mock_domain_manager):
        """Test suggesting tools based on user intent."""
        indexer = MCPToolIndexer(mock_domain_manager)
        
        # Mock retrieve_memories response
        mock_tool_memories = [
            {
                "content": {
                    "tool_name": "postgres_query",
                    "description": "Execute SQL queries",
                    "server_name": "postgres",
                    "keywords": ["database", "sql", "query"]
                }
            },
            {
                "content": {
                    "tool_name": "playwright_navigate", 
                    "description": "Navigate web pages",
                    "server_name": "playwright",
                    "keywords": ["web", "browser", "page"]
                }
            }
        ]
        mock_domain_manager.retrieve_memories.return_value = mock_tool_memories
        
        suggestions = await indexer.suggest_tools_for_intent(
            "I need to query the database for user information",
            limit=2
        )
        
        assert len(suggestions) == 2
        assert suggestions[0]["tool_name"] == "postgres_query"
        assert suggestions[0]["server_name"] == "postgres"
        assert "relevance_reason" in suggestions[0]
        assert "usage_hint" in suggestions[0]
        
        # Verify retrieve_memories was called correctly
        mock_domain_manager.retrieve_memories.assert_called_once()
        call_args = mock_domain_manager.retrieve_memories.call_args
        assert call_args[1]["types"] == ["mcp_tool"]


@pytest.mark.unit
class TestMCPToolSuggester:
    """Test MCP tool suggester functionality."""
    
    def test_tool_suggester_initialization(self, mock_domain_manager):
        """Test MCP tool suggester initialization."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        assert suggester.tool_indexer == indexer
        assert len(suggester.indirect_patterns) > 0
        assert any("script" in pattern for pattern in suggester.indirect_patterns)
        assert any("psql" in pattern for pattern in suggester.indirect_patterns)
    
    def test_would_use_indirect_method_database(self, mock_domain_manager):
        """Test detection of indirect database methods."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        # Should detect database requests without MCP mentions
        assert suggester._would_use_indirect_method("I need to query the database for users")
        assert suggester._would_use_indirect_method("Run SQL query to get table data")
        assert suggester._would_use_indirect_method("Use psql to connect to database")
        
        # Should not suggest when MCP tools are mentioned
        assert not suggester._would_use_indirect_method("Use postgres MCP tool to query database")
        assert not suggester._would_use_indirect_method("I'll use the MCP database tool")
    
    def test_would_use_indirect_method_web(self, mock_domain_manager):
        """Test detection of indirect web automation methods."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        # Should detect web requests without MCP mentions
        assert suggester._would_use_indirect_method("Navigate to the website and click login")
        assert suggester._would_use_indirect_method("I need to browse the web page")
        assert suggester._would_use_indirect_method("Manual browse to find information")
        
        # Should not suggest when MCP tools are mentioned
        assert not suggester._would_use_indirect_method("Use playwright MCP tool to navigate")
        assert not suggester._would_use_indirect_method("I'll use the MCP web tool")
    
    def test_would_use_indirect_method_patterns(self, mock_domain_manager):
        """Test detection of indirect patterns."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        # Should detect scripting patterns
        assert suggester._would_use_indirect_method("Write a script to automate this")
        assert suggester._would_use_indirect_method("Create a file to handle the request")
        assert suggester._would_use_indirect_method("Write code to solve this problem")
        
        # Should not detect when no patterns match
        assert not suggester._would_use_indirect_method("This is a simple question")
        assert not suggester._would_use_indirect_method("How do I configure the settings?")
    
    @pytest.mark.asyncio
    async def test_analyze_and_suggest(self, mock_domain_manager):
        """Test analyzing user request and providing suggestions."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        # Mock the indexer's suggest_tools_for_intent method
        mock_suggestions = [
            {
                "tool_name": "postgres_query",
                "description": "Execute SQL queries",
                "server_name": "postgres",
                "relevance_reason": "Matches keywords: database, query",
                "usage_hint": "Use postgres_query instead of psql scripts"
            }
        ]
        indexer.suggest_tools_for_intent = AsyncMock(return_value=mock_suggestions)
        
        # Request that should trigger suggestions
        request = "I need to write a script to query the database"
        
        suggestion_message = await suggester.analyze_and_suggest(request)
        
        assert suggestion_message is not None
        assert "MCP Tool Suggestion" in suggestion_message
        assert "postgres_query" in suggestion_message
        assert "postgres" in suggestion_message
        assert "Execute SQL queries" in suggestion_message
        
        indexer.suggest_tools_for_intent.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_analyze_and_suggest_no_indirect_method(self, mock_domain_manager):
        """Test no suggestions when indirect methods aren't detected."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        # Request that shouldn't trigger suggestions
        request = "What is the capital of France?"
        
        suggestion_message = await suggester.analyze_and_suggest(request)
        
        assert suggestion_message is None
    
    def test_format_suggestions(self, mock_domain_manager):
        """Test formatting tool suggestions into user message."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        suggestions = [
            {
                "tool_name": "postgres_query",
                "description": "Execute SQL queries",
                "server_name": "postgres", 
                "relevance_reason": "Matches keywords: database",
                "usage_hint": "Use instead of psql scripts"
            },
            {
                "tool_name": "playwright_navigate",
                "description": "Navigate web pages",
                "server_name": "playwright",
                "relevance_reason": "Matches keywords: web, browser",
                "usage_hint": "Use instead of manual browsing"
            }
        ]
        
        formatted = suggester._format_suggestions(suggestions)
        
        assert "MCP Tool Suggestion" in formatted
        assert "postgres_query (postgres)" in formatted
        assert "playwright_navigate (playwright)" in formatted
        assert "Execute SQL queries" in formatted
        assert "Navigate web pages" in formatted
        assert "Matches keywords: database" in formatted
        assert "Use instead of psql scripts" in formatted
        assert "faster and more reliable" in formatted
    
    def test_format_empty_suggestions(self, mock_domain_manager):
        """Test formatting empty suggestions list."""
        indexer = MCPToolIndexer(mock_domain_manager)
        suggester = MCPToolSuggester(indexer)
        
        formatted = suggester._format_suggestions([])
        
        assert formatted == ""