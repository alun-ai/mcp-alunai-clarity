"""
Mock MCP servers and fixtures for testing.

This module provides mock implementations of MCP servers for testing purposes.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from unittest.mock import AsyncMock, MagicMock


@dataclass
class MockTool:
    """Mock MCP tool for testing."""
    name: str
    description: str
    inputSchema: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MockPrompt:
    """Mock MCP prompt for testing."""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MockResource:
    """Mock MCP resource for testing."""
    uri: str
    name: str
    description: str
    mimeType: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MockListToolsResult:
    """Mock list_tools result."""
    
    def __init__(self, tools: List[MockTool]):
        self.tools = tools


class MockListPromptsResult:
    """Mock list_prompts result."""
    
    def __init__(self, prompts: List[MockPrompt]):
        self.prompts = prompts


class MockListResourcesResult:
    """Mock list_resources result."""
    
    def __init__(self, resources: List[MockResource]):
        self.resources = resources


class MockCallToolResult:
    """Mock call_tool result."""
    
    def __init__(self, content: List[Dict[str, Any]], isError: bool = False):
        self.content = content
        self.isError = isError


class MockMCPServer:
    """Mock MCP server for testing."""
    
    def __init__(self, name: str, tools: List[str], prompts: List[str] = None, resources: List[str] = None):
        self.name = name
        self.tools = tools or []
        self.prompts = prompts or []
        self.resources = resources or []
        self.capabilities = {
            'tools': {'listChanged': True} if tools else None,
            'prompts': {'listChanged': True} if prompts else None,
            'resources': {'listChanged': True} if resources else None
        }
        
        # Remove None capabilities
        self.capabilities = {k: v for k, v in self.capabilities.items() if v is not None}
    
    async def handle_list_tools(self) -> MockListToolsResult:
        """Mock list_tools response."""
        mock_tools = []
        for tool_name in self.tools:
            tool = MockTool(
                name=tool_name,
                description=f"Mock {tool_name} tool",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": f"Query for {tool_name}"}
                    },
                    "required": ["query"]
                }
            )
            mock_tools.append(tool)
        
        return MockListToolsResult(mock_tools)
    
    async def handle_list_prompts(self) -> MockListPromptsResult:
        """Mock list_prompts response."""
        mock_prompts = []
        for prompt_name in self.prompts:
            prompt = MockPrompt(
                name=prompt_name,
                description=f"Mock {prompt_name} prompt",
                arguments=[{
                    "name": "input",
                    "description": f"Input for {prompt_name}",
                    "required": True
                }]
            )
            mock_prompts.append(prompt)
        
        return MockListPromptsResult(mock_prompts)
    
    async def handle_list_resources(self) -> MockListResourcesResult:
        """Mock list_resources response."""
        mock_resources = []
        for resource_name in self.resources:
            resource = MockResource(
                uri=f"file://{resource_name}",
                name=resource_name,
                description=f"Mock resource {resource_name}",
                mimeType="text/plain"
            )
            mock_resources.append(resource)
        
        return MockListResourcesResult(mock_resources)
    
    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> MockCallToolResult:
        """Mock call_tool response."""
        if name not in self.tools:
            return MockCallToolResult(
                content=[{"type": "text", "text": f"Tool {name} not found"}],
                isError=True
            )
        
        # Simulate different tool responses
        if 'query' in name.lower() or 'database' in name.lower():
            return MockCallToolResult(
                content=[{
                    "type": "text", 
                    "text": json.dumps({
                        "result": f"Mock query result for {arguments.get('query', 'test')}",
                        "rows": 5,
                        "execution_time": 0.1
                    })
                }]
            )
        
        elif 'file' in name.lower() or 'read' in name.lower():
            return MockCallToolResult(
                content=[{
                    "type": "text",
                    "text": f"Mock file content for {arguments.get('path', 'test.txt')}"
                }]
            )
        
        elif 'web' in name.lower() or 'http' in name.lower():
            return MockCallToolResult(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "status": 200,
                        "data": {"message": f"Mock HTTP response for {arguments.get('url', 'test')}"}
                    })
                }]
            )
        
        else:
            return MockCallToolResult(
                content=[{
                    "type": "text",
                    "text": f"Mock response from {name} with args: {json.dumps(arguments)}"
                }]
            )
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "name": self.name,
            "version": "1.0.0",
            "capabilities": self.capabilities,
            "tools_count": len(self.tools),
            "prompts_count": len(self.prompts),
            "resources_count": len(self.resources)
        }


class MockClientSession:
    """Mock MCP client session for testing."""
    
    def __init__(self, server: MockMCPServer):
        self.server = server
        self.initialized = False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def initialize(self):
        """Mock session initialization."""
        await asyncio.sleep(0.01)  # Simulate initialization delay
        self.initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.server.capabilities,
            "serverInfo": self.server.get_server_info()
        }
    
    async def list_tools(self) -> MockListToolsResult:
        """Mock list_tools call."""
        if not self.initialized:
            raise RuntimeError("Session not initialized")
        return await self.server.handle_list_tools()
    
    async def list_prompts(self) -> MockListPromptsResult:
        """Mock list_prompts call."""
        if not self.initialized:
            raise RuntimeError("Session not initialized")
        return await self.server.handle_list_prompts()
    
    async def list_resources(self) -> MockListResourcesResult:
        """Mock list_resources call."""
        if not self.initialized:
            raise RuntimeError("Session not initialized")
        return await self.server.handle_list_resources()
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MockCallToolResult:
        """Mock call_tool call."""
        if not self.initialized:
            raise RuntimeError("Session not initialized")
        return await self.server.handle_call_tool(name, arguments)


class MockTransport:
    """Mock MCP transport for testing."""
    
    def __init__(self, server: MockMCPServer):
        self.server = server


# Pre-configured mock servers for common testing scenarios
def create_postgres_mock_server() -> MockMCPServer:
    """Create mock PostgreSQL MCP server."""
    return MockMCPServer(
        name="postgres",
        tools=["postgres_query", "postgres_list_tables", "postgres_describe_table"],
        prompts=["query_builder", "schema_analyzer"],
        resources=["schema.sql", "sample_data.sql"]
    )


def create_filesystem_mock_server() -> MockMCPServer:
    """Create mock filesystem MCP server."""
    return MockMCPServer(
        name="filesystem",
        tools=["read_file", "write_file", "list_directory", "create_directory"],
        prompts=["file_search", "directory_summary"],
        resources=["config.json", "README.md", "data/"]
    )


def create_web_mock_server() -> MockMCPServer:
    """Create mock web/HTTP MCP server."""
    return MockMCPServer(
        name="web",
        tools=["http_get", "http_post", "http_put", "http_delete"],
        prompts=["api_explorer", "request_builder"],
        resources=["api_spec.json"]
    )


def create_git_mock_server() -> MockMCPServer:
    """Create mock Git MCP server."""
    return MockMCPServer(
        name="git",
        tools=["git_status", "git_log", "git_diff", "git_commit"],
        prompts=["commit_message", "branch_summary"],
        resources=["git_history", "branches"]
    )


def create_mock_server_fleet() -> Dict[str, MockMCPServer]:
    """Create a fleet of mock servers for comprehensive testing."""
    return {
        "postgres": create_postgres_mock_server(),
        "filesystem": create_filesystem_mock_server(),
        "web": create_web_mock_server(),
        "git": create_git_mock_server(),
        "custom": MockMCPServer(
            name="custom",
            tools=["custom_tool_1", "custom_tool_2"],
            prompts=["custom_prompt"],
            resources=["custom_resource.txt"]
        )
    }


# Test server configurations
TEST_SERVER_CONFIGS = {
    "postgres": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-postgres"],
        "env": {"DATABASE_URL": "postgresql://localhost/test"}
    },
    "filesystem": {
        "command": "npx", 
        "args": ["@modelcontextprotocol/server-filesystem"],
        "env": {"ALLOWED_DIRS": "/tmp"}
    },
    "web": {
        "command": "python",
        "args": ["-m", "web_mcp_server"],
        "env": {"API_KEY": "test_key"}
    },
    "git": {
        "command": "git-mcp-server",
        "args": ["--repo", "/test/repo"],
        "env": {}
    }
}


# Mock session factory
async def create_mock_session(server_name: str, config: Dict[str, Any] = None) -> MockClientSession:
    """Create a mock client session for a server."""
    mock_servers = create_mock_server_fleet()
    
    if server_name in mock_servers:
        server = mock_servers[server_name]
    else:
        # Create a basic server if not in fleet
        server = MockMCPServer(
            name=server_name,
            tools=[f"{server_name}_tool"],
            prompts=[f"{server_name}_prompt"]
        )
    
    session = MockClientSession(server)
    await session.initialize()
    return session


# Error simulation helpers
class FailingMockServer(MockMCPServer):
    """Mock server that simulates various failure conditions."""
    
    def __init__(self, name: str, failure_type: str = "timeout"):
        super().__init__(name, ["failing_tool"])
        self.failure_type = failure_type
    
    async def handle_list_tools(self):
        """Simulate failures in tool listing."""
        if self.failure_type == "timeout":
            await asyncio.sleep(10)  # Simulate timeout
        elif self.failure_type == "error":
            raise Exception("Server error")
        elif self.failure_type == "empty":
            return MockListToolsResult([])
        
        return await super().handle_list_tools()
    
    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]):
        """Simulate failures in tool execution."""
        if self.failure_type == "tool_error":
            return MockCallToolResult(
                content=[{"type": "text", "text": "Tool execution failed"}],
                isError=True
            )
        
        return await super().handle_call_tool(name, arguments)


def create_failing_server(failure_type: str = "timeout") -> FailingMockServer:
    """Create a server that simulates failure conditions."""
    return FailingMockServer("failing_server", failure_type)


# Performance testing helpers
class SlowMockServer(MockMCPServer):
    """Mock server with configurable response delays."""
    
    def __init__(self, name: str, delay_seconds: float = 1.0):
        super().__init__(name, ["slow_tool"])
        self.delay_seconds = delay_seconds
    
    async def handle_list_tools(self):
        """Add delay to tool listing."""
        await asyncio.sleep(self.delay_seconds)
        return await super().handle_list_tools()
    
    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]):
        """Add delay to tool execution."""
        await asyncio.sleep(self.delay_seconds)
        return await super().handle_call_tool(name, arguments)


def create_slow_server(delay_seconds: float = 1.0) -> SlowMockServer:
    """Create a server with artificial delays for performance testing."""
    return SlowMockServer("slow_server", delay_seconds)


# Utilities for test setup
def setup_mock_environment() -> Dict[str, Any]:
    """Set up a complete mock testing environment."""
    return {
        "servers": create_mock_server_fleet(),
        "configs": TEST_SERVER_CONFIGS,
        "failing_server": create_failing_server(),
        "slow_server": create_slow_server(),
        "session_factory": create_mock_session
    }


# Mock context managers for testing
class MockMCPContext:
    """Context manager for mock MCP testing environment."""
    
    def __init__(self):
        self.environment = None
    
    def __enter__(self):
        self.environment = setup_mock_environment()
        return self.environment
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass
    
    async def __aenter__(self):
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)


# Export main classes and functions
__all__ = [
    'MockMCPServer',
    'MockClientSession', 
    'MockTool',
    'MockPrompt',
    'MockResource',
    'create_postgres_mock_server',
    'create_filesystem_mock_server',
    'create_web_mock_server',
    'create_git_mock_server',
    'create_mock_server_fleet',
    'create_mock_session',
    'create_failing_server',
    'create_slow_server',
    'setup_mock_environment',
    'MockMCPContext',
    'TEST_SERVER_CONFIGS'
]