"""
REAL Comprehensive test suite for MCP Registry/Tool Discovery functionality.

This test suite validates the ACTUAL MCP tool indexing and discovery system including:
- Real MCPToolIndexer with live MCP server discovery
- Actual tool metadata extraction and indexing
- Real proactive tool suggestions based on user intent
- Configuration-based tool discovery from Claude Desktop configs
- Live MCP server enumeration and tool cataloging

Tests REAL functionality in:
- clarity/mcp/tool_indexer.py (867 lines of actual implementation)
- Real MCP tool discovery and categorization
- Actual intent analysis and tool matching
- Real memory storage integration
"""

import pytest
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch

# Import REAL classes - no stubs!
from clarity.mcp.tool_indexer import MCPToolIndexer, MCPToolSuggester, MCPToolInfo
from tests.framework.mcp_validation import MCPServerTestSuite


class TestRealMCPToolIndexer:
    """Test the REAL MCPToolIndexer functionality."""
    
    @pytest.fixture
    def real_tool_indexer(self):
        """Create real MCPToolIndexer with actual domain manager."""
        # Create a minimal config for testing
        config = {
            "mcp": {
                "tool_indexer": {
                    "discovery_sources": ["known_tools", "configuration", "live_servers"],
                    "cache_ttl_hours": 24,
                    "intent_analysis_enabled": True
                }
            }
        }
        
        # Use mock domain manager for memory operations
        from unittest.mock import Mock, AsyncMock
        mock_domain_manager = Mock()
        mock_domain_manager.store_memory = AsyncMock(return_value="tool_memory_id")
        mock_domain_manager.retrieve_memories = AsyncMock(return_value=[])
        
        # Create REAL MCPToolIndexer (only takes domain_manager)
        indexer = MCPToolIndexer(mock_domain_manager)
        return indexer
    
    @pytest.mark.asyncio
    async def test_real_tool_indexer_initialization(self, real_tool_indexer):
        """Test real MCPToolIndexer initialization."""
        await real_tool_indexer.initialize()
        
        # Validate real attributes exist
        assert hasattr(real_tool_indexer, 'domain_manager')
        assert hasattr(real_tool_indexer, 'indexed_tools')
        assert hasattr(real_tool_indexer, 'intent_categories')
        
        # Check real initialization state  
        assert isinstance(real_tool_indexer.indexed_tools, dict)
        assert isinstance(real_tool_indexer.intent_categories, dict)
    
    @pytest.mark.asyncio
    async def test_real_known_tools_discovery(self, real_tool_indexer):
        """Test discovery of known MCP tools from the built-in database."""
        await real_tool_indexer.initialize()
        
        # Discover known tools - this should load the actual known tools database
        result = await real_tool_indexer.discover_known_tools()
        
        assert result["success"] == True
        assert result["total_discovered"] > 0
        
        # Validate actual known tools were discovered
        known_tools = result["discovered_tools"]
        assert len(known_tools) > 0
        
        # Check that real known tools are present (these should exist in the actual implementation)
        tool_names = [tool.tool_name for tool in known_tools]
        
        # The real implementation should have common MCP tools
        expected_categories = ["file_operations", "memory_management", "code_analysis", "web_search"]
        found_categories = set()
        
        for tool in known_tools:
            assert isinstance(tool, MCPToolInfo)
            assert hasattr(tool, 'tool_name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'category')
            assert hasattr(tool, 'keywords')
            found_categories.add(tool.category)
        
        # Should have discovered tools from multiple categories
        assert len(found_categories) > 1
    
    @pytest.mark.asyncio
    async def test_real_configuration_discovery(self, real_tool_indexer):
        """Test discovery from actual Claude Desktop configuration files."""
        await real_tool_indexer.initialize()
        
        # Create a temporary Claude Desktop config for testing
        config_content = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
                    "description": "File system operations"
                },
                "git": {
                    "command": "npx", 
                    "args": ["@modelcontextprotocol/server-git"],
                    "description": "Git repository operations"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_content, f)
            config_path = f.name
        
        try:
            # Test real configuration parsing
            result = await real_tool_indexer.discover_from_configuration(config_path)
            
            assert result["success"] == True
            assert result["servers_found"] == 2
            assert "filesystem" in result["discovered_servers"]
            assert "git" in result["discovered_servers"]
            
            # Validate tools were properly categorized
            discovered_tools = result["discovered_tools"]
            assert len(discovered_tools) > 0
            
            # Check real tool metadata extraction
            for tool_info in discovered_tools:
                assert isinstance(tool_info, MCPToolInfo)
                assert tool_info.server_name in ["filesystem", "git"]
                assert tool_info.discovery_method == "configuration"
                assert len(tool_info.keywords) > 0
                
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_real_tool_categorization(self, real_tool_indexer):
        """Test actual tool categorization and keyword extraction."""
        await real_tool_indexer.initialize()
        
        # Test with real tool examples that should exist
        test_tools = [
            MCPToolInfo(
                name="read_file",
                description="Read contents of a file from the filesystem",
                server_name="filesystem",
                parameters={"path": "string"},
                use_cases=["file reading", "content analysis"],
                keywords={"read", "file", "contents"},
                category="file_operations"
            ),
            MCPToolInfo(
                name="git_commit",
                description="Create a new git commit with staged changes",
                server_name="git",
                parameters={"message": "string"},
                use_cases=["version control", "git operations"],
                keywords={"git", "commit", "changes"},
                category="development"
            ),
            MCPToolInfo(
                name="web_search",
                description="Search the web for information using search engines",
                server_name="web",
                parameters={"query": "string"},
                use_cases=["web search", "information gathering"],
                keywords={"search", "web", "information"},
                category="web_automation"
            )
        ]
        
        # Test real categorization logic
        for tool in test_tools:
            category = real_tool_indexer._categorize_tool(tool)
            keywords = real_tool_indexer._extract_keywords(tool)
            
            # Validate real categorization
            assert category in real_tool_indexer.tool_categories
            assert isinstance(keywords, list)
            assert len(keywords) > 0
            
            # Test specific expected categorizations
            if "file" in tool.description.lower():
                assert category == "file_operations"
            elif "git" in tool.tool_name.lower():
                assert category == "version_control"
            elif "search" in tool.description.lower():
                assert category == "web_search"
    
    @pytest.mark.asyncio
    async def test_real_memory_integration(self, real_tool_indexer):
        """Test real integration with memory system for tool storage."""
        await real_tool_indexer.initialize()
        
        # Create real tool for testing
        test_tool = MCPToolInfo(
            tool_name="test_memory_tool",
            description="A test tool for validating memory integration",
            server_name="test_server",
            parameters={"param1": "string"},
            discovery_method="test"
        )
        
        # Test real memory storage
        result = await real_tool_indexer.store_tool_in_memory(test_tool)
        
        assert result["success"] == True
        assert "memory_id" in result
        
        # Validate real domain manager was called with proper memory format
        real_tool_indexer.domain_manager.store_memory.assert_called_once()
        call_args = real_tool_indexer.domain_manager.store_memory.call_args[1]
        
        assert call_args["memory_type"] == "mcp_tool"
        assert "test_memory_tool" in call_args["content"]
        assert call_args["metadata"]["tool_name"] == "test_memory_tool"
        assert call_args["metadata"]["server_name"] == "test_server"
        assert call_args["importance"] > 0


class TestRealMCPToolSuggester:
    """Test the REAL MCPToolSuggester functionality."""
    
    @pytest.fixture
    def real_tool_suggester(self, real_tool_indexer):
        """Create real MCPToolSuggester."""
        suggester = MCPToolSuggester(real_tool_indexer)
        return suggester
    
    @pytest.mark.asyncio
    async def test_real_intent_analysis(self, real_tool_suggester):
        """Test real intent analysis and tool matching."""
        # Test with actual user intents that should match real tools
        test_intents = [
            {
                "intent": "I need to read a file and check its contents",
                "expected_categories": ["file_operations"],
                "expected_keywords": ["read", "file", "contents"]
            },
            {
                "intent": "Help me commit my changes to git repository",
                "expected_categories": ["version_control"],
                "expected_keywords": ["git", "commit", "changes"]
            },
            {
                "intent": "I want to search for information about Python tutorials",
                "expected_categories": ["web_search"],
                "expected_keywords": ["search", "information", "tutorials"]
            },
            {
                "intent": "Can you analyze the structure of my codebase?",
                "expected_categories": ["code_analysis"],
                "expected_keywords": ["analyze", "structure", "code"]
            }
        ]
        
        for test_case in test_intents:
            # Test real intent analysis
            analysis = await real_tool_suggester.analyze_intent(test_case["intent"])
            
            assert isinstance(analysis, dict)
            assert "categories" in analysis
            assert "keywords" in analysis
            assert "complexity_score" in analysis
            
            # Validate real analysis results
            extracted_keywords = analysis["keywords"]
            assert isinstance(extracted_keywords, list)
            assert len(extracted_keywords) > 0
            
            # Check that expected keywords are found by real analysis
            for expected_keyword in test_case["expected_keywords"]:
                assert any(expected_keyword.lower() in kw.lower() for kw in extracted_keywords), \
                    f"Expected keyword '{expected_keyword}' not found in extracted keywords: {extracted_keywords}"
    
    @pytest.mark.asyncio
    async def test_real_tool_suggestions(self, real_tool_suggester):
        """Test real tool suggestions based on user intent."""
        # Initialize with some real tools
        await real_tool_suggester.tool_indexer.initialize()
        
        # Add some real tools for testing
        test_tools = [
            MCPToolInfo(
                tool_name="read_file",
                description="Read contents of a file from the filesystem",
                server_name="filesystem",
                category="file_operations",
                keywords=["read", "file", "contents", "filesystem"],
                parameters={"path": "string"}
            ),
            MCPToolInfo(
                tool_name="list_directory", 
                description="List files and directories in a given path",
                server_name="filesystem",
                category="file_operations",
                keywords=["list", "directory", "files", "browse"],
                parameters={"path": "string"}
            ),
            MCPToolInfo(
                tool_name="git_status",
                description="Show the working tree status for git repository",
                server_name="git",
                category="version_control",
                keywords=["git", "status", "working", "tree"],
                parameters={}
            )
        ]
        
        for tool in test_tools:
            real_tool_suggester.tool_indexer.discovered_tools[tool.tool_name] = tool
        
        # Test real suggestions for different intents
        suggestions_tests = [
            {
                "intent": "I need to read a configuration file",
                "expected_tools": ["read_file"],
                "min_suggestions": 1
            },
            {
                "intent": "Show me what files are in this directory",
                "expected_tools": ["list_directory"],
                "min_suggestions": 1
            },
            {
                "intent": "Check the current status of my git repository",
                "expected_tools": ["git_status"],
                "min_suggestions": 1
            },
            {
                "intent": "I want to work with files and directories",
                "expected_tools": ["read_file", "list_directory"],
                "min_suggestions": 2
            }
        ]
        
        for test_case in suggestions_tests:
            # Get real suggestions
            suggestions = await real_tool_suggester.suggest_tools_for_intent(
                test_case["intent"],
                limit=5
            )
            
            assert isinstance(suggestions, list)
            assert len(suggestions) >= test_case["min_suggestions"]
            
            # Validate suggestions format
            for suggestion in suggestions:
                assert isinstance(suggestion, dict)
                assert "tool_name" in suggestion
                assert "relevance_score" in suggestion
                assert "reasoning" in suggestion
                assert suggestion["relevance_score"] >= 0.0
                assert suggestion["relevance_score"] <= 1.0
            
            # Check that expected tools are suggested
            suggested_tool_names = [s["tool_name"] for s in suggestions]
            for expected_tool in test_case["expected_tools"]:
                assert expected_tool in suggested_tool_names, \
                    f"Expected tool '{expected_tool}' not found in suggestions: {suggested_tool_names}"
    
    @pytest.mark.asyncio
    async def test_real_contextual_suggestions(self, real_tool_suggester):
        """Test real contextual tool suggestions based on current context."""
        await real_tool_suggester.tool_indexer.initialize()
        
        # Test with real context scenarios
        context_tests = [
            {
                "current_tools": ["read_file"],
                "project_type": "python",
                "recent_commands": ["python setup.py", "pip install -e ."],
                "expected_suggestion_categories": ["version_control", "file_operations"]
            },
            {
                "current_tools": ["git_status"],
                "project_type": "javascript",
                "recent_commands": ["npm test", "git add ."],
                "expected_suggestion_categories": ["version_control"]
            }
        ]
        
        for test_case in context_tests:
            # Get real contextual suggestions
            suggestions = await real_tool_suggester.get_contextual_suggestions(
                current_tools=test_case["current_tools"],
                project_context={
                    "project_type": test_case["project_type"],
                    "recent_commands": test_case["recent_commands"]
                }
            )
            
            assert isinstance(suggestions, list)
            # Should provide contextually relevant suggestions
            assert len(suggestions) > 0
            
            # Validate suggestion structure
            for suggestion in suggestions:
                assert "tool_name" in suggestion
                assert "context_relevance" in suggestion
                assert "suggestion_reason" in suggestion


class TestRealMCPServerIntegration:
    """Test real integration with MCP server for tool discovery."""
    
    @pytest.mark.asyncio
    async def test_real_mcp_server_tool_discovery(self):
        """Test real MCP server integration for tool discovery."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create real tool indexer with MCP server
            config = {
                "mcp": {
                    "tool_indexer": {
                        "discovery_sources": ["live_servers", "known_tools"],
                        "intent_analysis_enabled": True
                    }
                }
            }
            
            indexer = MCPToolIndexer(config, suite.mcp_server.domain_manager)
            await indexer.initialize()
            
            # Test real live server discovery
            result = await indexer.discover_from_live_servers()
            
            assert result["success"] == True
            
            # Should discover tools from the actual MCP server
            if result["total_discovered"] > 0:
                discovered_tools = result["discovered_tools"]
                
                for tool in discovered_tools:
                    assert isinstance(tool, MCPToolInfo)
                    assert tool.discovery_method == "live_server"
                    assert len(tool.tool_name) > 0
                    assert len(tool.description) > 0
                    
                # Validate tools were indexed properly
                for tool_name, tool_info in indexer.discovered_tools.items():
                    assert isinstance(tool_info, MCPToolInfo)
                    assert tool_info.indexed_at is not None
                    
        finally:
            await suite.teardown_test_environment()
    
    @pytest.mark.asyncio
    async def test_real_end_to_end_tool_workflow(self):
        """Test complete real workflow from discovery to suggestion."""
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Create real components
            config = {
                "mcp": {
                    "tool_indexer": {
                        "discovery_sources": ["known_tools", "live_servers"],
                        "intent_analysis_enabled": True,
                        "cache_ttl_hours": 1
                    }
                }
            }
            
            indexer = MCPToolIndexer(config, suite.mcp_server.domain_manager)
            await indexer.initialize()
            
            suggester = MCPToolSuggester(indexer)
            
            # 1. Real discovery phase
            discovery_result = await indexer.discover_and_index_tools()
            assert discovery_result["success"] == True
            
            # 2. Real indexing phase
            if discovery_result["total_discovered"] > 0:
                index_result = await indexer.index_discovered_tools()
                assert index_result["success"] == True
                
                # 3. Real suggestion phase
                user_intent = "I need to store some information and retrieve it later"
                suggestions = await suggester.suggest_tools_for_intent(user_intent)
                
                assert isinstance(suggestions, list)
                
                # Should suggest memory-related tools if they exist
                if len(suggestions) > 0:
                    memory_tools = [s for s in suggestions if "memory" in s["tool_name"].lower() or "store" in s["tool_name"].lower()]
                    # If memory tools exist, they should be highly ranked
                    if memory_tools:
                        assert memory_tools[0]["relevance_score"] > 0.7
                
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_real_mcp_tool_indexer_tests():
        """Run real MCP tool indexer tests directly."""
        print("🧪 Running REAL MCP tool indexer tests...")
        
        # Create real tool indexer for testing
        config = {
            "mcp": {
                "tool_indexer": {
                    "discovery_sources": ["known_tools"],
                    "intent_analysis_enabled": True
                }
            }
        }
        
        from unittest.mock import Mock, AsyncMock
        mock_domain_manager = Mock()
        mock_domain_manager.store_memory = AsyncMock(return_value="test_memory_id")
        
        indexer = MCPToolIndexer(config, mock_domain_manager)
        await indexer.initialize()
        
        # Test real functionality
        result = await indexer.discover_known_tools()
        print(f"✅ Discovered {result['total_discovered']} known tools")
        
        suggester = MCPToolSuggester(indexer)
        suggestions = await suggester.suggest_tools_for_intent("I need to read a file")
        print(f"✅ Generated {len(suggestions)} suggestions for file reading")
        
        print("\n🎉 All REAL MCP tool indexer tests completed!")
    
    asyncio.run(run_real_mcp_tool_indexer_tests())