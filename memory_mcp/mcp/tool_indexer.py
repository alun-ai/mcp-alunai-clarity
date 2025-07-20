"""
MCP Tool Discovery and Indexing System.

This module automatically discovers available MCP tools and stores them as memories
for proactive suggestion and usage by Claude.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from loguru import logger

try:
    import mcp
    from mcp.server.models import Tool
    MCP_AVAILABLE = True
    
    # Try to import client components for live discovery
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioClientTransport
        MCP_CLIENT_AVAILABLE = True
    except ImportError:
        MCP_CLIENT_AVAILABLE = False
        logger.debug("MCP client not available, live discovery will be limited")
        
except ImportError:
    MCP_AVAILABLE = False
    MCP_CLIENT_AVAILABLE = False
    logger.warning("MCP not available, tool indexing will be limited")


@dataclass
class MCPToolInfo:
    """Information about an discovered MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str
    use_cases: List[str]
    keywords: Set[str]
    category: str


class MCPToolIndexer:
    """
    Discovers and indexes available MCP tools for proactive usage.
    
    This class scans available MCP servers and tools, extracts their capabilities,
    and stores them as searchable memories for Claude to reference.
    """
    
    def __init__(self, domain_manager):
        """
        Initialize the MCP tool indexer.
        
        Args:
            domain_manager: Memory domain manager for storing tool information
        """
        self.domain_manager = domain_manager
        self.indexed_tools: Dict[str, MCPToolInfo] = {}
        
        # Intent keywords for categorizing tools
        self.intent_categories = {
            "database": [
                "database", "sql", "query", "table", "postgres", "mysql", 
                "sqlite", "db", "schema", "insert", "update", "delete", "select"
            ],
            "web_automation": [
                "browser", "web", "page", "click", "navigate", "playwright", 
                "selenium", "automation", "scrape", "crawl", "javascript"
            ],
            "file_operations": [
                "file", "read", "write", "edit", "create", "delete", "path", 
                "directory", "folder", "copy", "move", "upload", "download"
            ],
            "memory_management": [
                "memory", "remember", "recall", "store", "retrieve", "search",
                "knowledge", "context", "history", "save", "forget"
            ],
            "api_integration": [
                "api", "rest", "http", "request", "response", "endpoint",
                "integration", "webhook", "service", "client"
            ],
            "development": [
                "code", "build", "deploy", "test", "debug", "git", "repository",
                "docker", "container", "ci", "cd", "pipeline"
            ],
            "analytics": [
                "analytics", "metrics", "data", "chart", "graph", "report",
                "dashboard", "statistics", "analysis", "visualization"
            ],
            "communication": [
                "email", "message", "notification", "slack", "discord", "teams",
                "chat", "send", "notify", "alert"
            ]
        }
    
    async def discover_and_index_tools(self) -> Dict[str, MCPToolInfo]:
        """
        Discover all available MCP tools and index them as memories.
        
        Returns:
            Dictionary of discovered tools by name
        """
        logger.info("Starting MCP tool discovery and indexing...")
        
        # Discover tools from various sources
        tools = await self._discover_tools()
        
        # Index each tool as a memory
        indexed_count = 0
        for tool in tools:
            try:
                await self._index_tool_as_memory(tool)
                self.indexed_tools[tool.name] = tool
                indexed_count += 1
            except Exception as e:
                logger.error(f"Failed to index tool {tool.name}: {e}")
        
        # Store summary information
        await self._store_indexing_summary(indexed_count, len(tools))
        
        logger.info(f"Successfully indexed {indexed_count}/{len(tools)} MCP tools")
        return self.indexed_tools
    
    async def _discover_tools(self) -> List[MCPToolInfo]:
        """
        Discover available MCP tools from various sources.
        
        Returns:
            List of discovered MCP tool information
        """
        tools = []
        
        # Method 1: Discover from MCP server instances (if available)
        if MCP_AVAILABLE:
            try:
                mcp_tools = await self._discover_from_mcp_servers()
                tools.extend(mcp_tools)
            except Exception as e:
                logger.warning(f"Could not discover from MCP servers: {e}")
        
        # Method 2: Discover from environment/configuration
        try:
            config_tools = await self._discover_from_configuration()
            tools.extend(config_tools)
        except Exception as e:
            logger.warning(f"Could not discover from configuration: {e}")
        
        # Method 3: Discover common/known MCP tools
        known_tools = await self._discover_known_tools()
        tools.extend(known_tools)
        
        return tools
    
    async def _discover_from_mcp_servers(self) -> List[MCPToolInfo]:
        """Discover tools from active MCP server instances."""
        tools = []
        
        logger.debug("Discovering tools from active MCP servers...")
        
        # Try to discover from environment variables or runtime context
        # This could be expanded to integrate with Claude's MCP server registry
        
        # Check for common MCP server environment variables
        mcp_env_servers = self._get_servers_from_environment()
        
        for server_name, server_config in mcp_env_servers.items():
            try:
                server_tools = await self._discover_tools_from_server_config(server_name, server_config)
                tools.extend(server_tools)
                logger.debug(f"Discovered {len(server_tools)} tools from env server {server_name}")
            except Exception as e:
                logger.warning(f"Failed to discover tools from env server {server_name}: {e}")
        
        return tools
    
    def _get_servers_from_environment(self) -> Dict[str, Dict[str, Any]]:
        """Extract MCP server configurations from environment variables."""
        servers = {}
        
        # Look for MCP server environment variables
        # Common patterns: MCP_SERVER_NAME_COMMAND, MCP_SERVER_NAME_ARGS, etc.
        env_pattern = re.compile(r'^MCP_(.+)_COMMAND$')
        
        for env_var, command in os.environ.items():
            match = env_pattern.match(env_var)
            if match:
                server_name = match.group(1).lower().replace('_', '-')
                
                # Get associated args and env vars
                args_var = f"MCP_{match.group(1)}_ARGS"
                env_var_pattern = f"MCP_{match.group(1)}_ENV_"
                
                args = []
                if args_var in os.environ:
                    # Parse args (assume space-separated for now)
                    args = os.environ[args_var].split()
                
                # Collect environment variables
                server_env = {}
                for env_key, env_val in os.environ.items():
                    if env_key.startswith(env_var_pattern):
                        clean_key = env_key[len(env_var_pattern):]
                        server_env[clean_key] = env_val
                
                servers[server_name] = {
                    "command": command,
                    "args": args,
                    "env": server_env,
                    "type": "stdio"
                }
        
        return servers
    
    async def _discover_from_configuration(self) -> List[MCPToolInfo]:
        """Discover tools from MCP configuration files."""
        tools = []
        
        # Check common MCP configuration locations
        config_paths = [
            "~/.config/claude-desktop/config.json",
            "~/.config/mcp/servers.json", 
            "./.claude/config.json",
            "./.mcp/servers.json",
            "./mcp-servers.json"
        ]
        
        logger.debug("Discovering tools from configuration...")
        
        for config_path in config_paths:
            expanded_path = os.path.expanduser(config_path)
            if os.path.exists(expanded_path):
                try:
                    config_tools = await self._parse_mcp_config_file(expanded_path)
                    tools.extend(config_tools)
                    logger.debug(f"Found {len(config_tools)} tools in {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to parse MCP config {config_path}: {e}")
        
        return tools
    
    async def _discover_known_tools(self) -> List[MCPToolInfo]:
        """Discover commonly available MCP tools."""
        known_tools = [
            MCPToolInfo(
                name="postgres_query",
                description="Execute SQL queries against PostgreSQL databases directly",
                parameters={
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"}
                },
                server_name="postgres",
                use_cases=[
                    "Run SQL queries without writing scripts",
                    "Query database tables directly",
                    "Execute database operations",
                    "Retrieve data from PostgreSQL"
                ],
                keywords={
                    "database", "sql", "postgres", "query", "table", "select", 
                    "insert", "update", "delete", "data"
                },
                category="database"
            ),
            MCPToolInfo(
                name="playwright_navigate",
                description="Navigate web pages and interact with web elements using Playwright",
                parameters={
                    "url": {"type": "string", "description": "URL to navigate to"},
                    "action": {"type": "string", "description": "Action to perform"}
                },
                server_name="playwright",
                use_cases=[
                    "Automate web browser interactions",
                    "Navigate to web pages",
                    "Click buttons and fill forms",
                    "Extract data from websites"
                ],
                keywords={
                    "web", "browser", "navigate", "click", "automation", "page", 
                    "element", "scrape", "playwright"
                },
                category="web_automation"
            ),
            MCPToolInfo(
                name="store_memory",
                description="Store information in persistent memory with vector search",
                parameters={
                    "memory_type": {"type": "string", "description": "Type of memory to store"},
                    "content": {"type": "object", "description": "Memory content"},
                    "importance": {"type": "number", "description": "Importance score"}
                },
                server_name="alunai-memory",
                use_cases=[
                    "Remember important information across sessions",
                    "Store user preferences and settings",
                    "Save project context and decisions",
                    "Build knowledge base over time"
                ],
                keywords={
                    "memory", "remember", "store", "save", "persistent", "knowledge", 
                    "context", "information"
                },
                category="memory_management"
            ),
            MCPToolInfo(
                name="retrieve_memory",
                description="Search and retrieve relevant memories using vector similarity",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results"},
                    "types": {"type": "array", "description": "Memory types to search"}
                },
                server_name="alunai-memory",
                use_cases=[
                    "Find relevant past conversations",
                    "Recall user preferences",
                    "Search project history",
                    "Get contextual information"
                ],
                keywords={
                    "memory", "search", "retrieve", "recall", "find", "query", 
                    "context", "history"
                },
                category="memory_management"
            ),
            MCPToolInfo(
                name="suggest_memory_queries",
                description="Get smart memory search suggestions based on current context",
                parameters={
                    "current_context": {"type": "object", "description": "Current context"},
                    "task_description": {"type": "string", "description": "Current task"}
                },
                server_name="alunai-memory",
                use_cases=[
                    "Get proactive memory suggestions",
                    "Find relevant context automatically",
                    "Discover related information",
                    "Enhance decision making with past knowledge"
                ],
                keywords={
                    "memory", "suggest", "proactive", "context", "relevant", 
                    "automatic", "smart"
                },
                category="memory_management"
            )
        ]
        
        logger.debug(f"Discovered {len(known_tools)} known MCP tools")
        return known_tools
    
    async def _parse_mcp_config_file(self, config_path: str) -> List[MCPToolInfo]:
        """Parse MCP configuration file and extract server information."""
        tools = []
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Handle Claude Desktop config format
            mcp_servers = config.get("mcpServers", {})
            if not mcp_servers:
                # Handle other possible formats
                mcp_servers = config.get("servers", {})
            
            for server_name, server_config in mcp_servers.items():
                try:
                    # Try to discover tools from this server configuration
                    server_tools = await self._discover_tools_from_server_config(server_name, server_config)
                    tools.extend(server_tools)
                    
                except Exception as e:
                    logger.warning(f"Failed to discover tools from server {server_name}: {e}")
            
            logger.debug(f"Parsed {len(tools)} tools from config file {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to parse config file {config_path}: {e}")
        
        return tools
    
    async def _discover_tools_from_server_config(self, server_name: str, server_config: Dict[str, Any]) -> List[MCPToolInfo]:
        """Discover tools from a specific server configuration."""
        tools = []
        
        # First, try live discovery if possible
        if MCP_CLIENT_AVAILABLE:
            try:
                live_tools = await self._discover_tools_live(server_name, server_config)
                if live_tools:
                    tools.extend(live_tools)
                    logger.debug(f"Discovered {len(live_tools)} tools live from {server_name}")
                    return tools
            except Exception as e:
                logger.debug(f"Live discovery failed for {server_name}, falling back to inference: {e}")
        
        # Fall back to inferring tools from server configuration
        inferred_tools = await self._infer_tools_from_config(server_name, server_config)
        tools.extend(inferred_tools)
        
        return tools
    
    async def _discover_tools_live(self, server_name: str, server_config: Dict[str, Any]) -> List[MCPToolInfo]:
        """Attempt to connect to MCP server and discover tools live."""
        if not MCP_CLIENT_AVAILABLE:
            logger.debug("MCP client not available for live discovery")
            return []
        
        try:
            # Extract connection parameters
            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", {})
            
            if not command:
                logger.debug(f"No command specified for server {server_name}")
                return []
            
            # Create transport
            transport = StdioClientTransport(
                command=command,
                args=args,
                env={**os.environ, **env}  # Merge with current environment
            )
            
            # Connect to server with timeout (use asyncio.wait_for for compatibility)
            try:
                async def connect_and_discover():
                    async with ClientSession(transport) as session:
                        # Initialize the connection
                        await session.initialize()
                        
                        # List available tools
                        tools_result = await session.list_tools()
                        
                        # Convert MCP tools to our format
                        discovered_tools = []
                        for tool in tools_result.tools:
                            tool_info = MCPToolInfo(
                                name=tool.name,
                                description=tool.description or f"Tool from {server_name}",
                                parameters=self._extract_parameters_from_schema(tool.inputSchema),
                                server_name=server_name,
                                use_cases=self._generate_use_cases_from_tool(tool, server_name),
                                keywords=self._extract_keywords_from_tool(tool, server_name),
                                category=self._categorize_tool_from_info(tool.name, tool.description, server_name)
                            )
                            discovered_tools.append(tool_info)
                        
                        return discovered_tools
                
                # 5 second timeout for connection and discovery
                discovered_tools = await asyncio.wait_for(connect_and_discover(), timeout=5.0)
                logger.info(f"Successfully discovered {len(discovered_tools)} tools from live server {server_name}")
                return discovered_tools
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout connecting to server {server_name}")
        
        except Exception as e:
            logger.debug(f"Live discovery failed for {server_name}: {e}")
        
        return []
    
    async def _infer_tools_from_config(self, server_name: str, server_config: Dict[str, Any]) -> List[MCPToolInfo]:
        """Infer likely tools from server configuration when live discovery fails."""
        tools = []
        
        # Use server name and configuration to infer likely tools
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        
        # Common patterns for inferring tools
        inference_patterns = {
            "postgres": {
                "tools": ["query", "execute", "schema"],
                "category": "database",
                "keywords": {"sql", "database", "postgres", "query"}
            },
            "playwright": {
                "tools": ["navigate", "click", "type", "screenshot"],
                "category": "web_automation", 
                "keywords": {"browser", "web", "page", "automation"}
            },
            "filesystem": {
                "tools": ["read", "write", "list", "search"],
                "category": "file_operations",
                "keywords": {"file", "directory", "path", "read", "write"}
            },
            "memory": {
                "tools": ["store", "retrieve", "search", "delete"],
                "category": "memory_management",
                "keywords": {"memory", "store", "retrieve", "remember"}
            }
        }
        
        # Try to match server name to known patterns
        for pattern_name, pattern_info in inference_patterns.items():
            if pattern_name.lower() in server_name.lower() or pattern_name.lower() in command.lower():
                for tool_name in pattern_info["tools"]:
                    inferred_tool = MCPToolInfo(
                        name=f"{server_name}_{tool_name}",
                        description=f"Inferred {tool_name} tool from {server_name} server",
                        parameters={"input": {"type": "string", "description": "Tool input"}},
                        server_name=server_name,
                        use_cases=[f"Use {server_name}_{tool_name} for {pattern_info['category']} operations"],
                        keywords=pattern_info["keywords"],
                        category=pattern_info["category"]
                    )
                    tools.append(inferred_tool)
                break
        
        # If no patterns matched, create a generic tool
        if not tools:
            generic_tool = MCPToolInfo(
                name=f"{server_name}_tool",
                description=f"Tool from {server_name} MCP server",
                parameters={"input": {"type": "string", "description": "Tool input"}},
                server_name=server_name,
                use_cases=[f"Use {server_name} for specialized operations"],
                keywords={server_name.lower(), "mcp", "tool"},
                category="api_integration"
            )
            tools.append(generic_tool)
        
        logger.debug(f"Inferred {len(tools)} tools from {server_name} configuration")
        return tools
    
    def _extract_parameters_from_schema(self, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from MCP tool input schema."""
        if not input_schema:
            return {"input": {"type": "string", "description": "Tool input"}}
        
        properties = input_schema.get("properties", {})
        if not properties:
            return {"input": {"type": "string", "description": "Tool input"}}
        
        return properties
    
    def _generate_use_cases_from_tool(self, tool, server_name: str) -> List[str]:
        """Generate use cases from MCP tool information."""
        use_cases = []
        
        # Base use case from description
        if tool.description:
            use_cases.append(f"Use {tool.name} to {tool.description.lower()}")
        
        # Add server-specific context
        use_cases.append(f"Use {tool.name} from {server_name} server for specialized operations")
        
        # Add category-specific use cases based on tool name
        if any(keyword in tool.name.lower() for keyword in ["query", "select", "sql"]):
            use_cases.append(f"Use {tool.name} instead of writing SQL scripts")
        elif any(keyword in tool.name.lower() for keyword in ["navigate", "click", "browser"]):
            use_cases.append(f"Use {tool.name} instead of manual web browsing")
        elif any(keyword in tool.name.lower() for keyword in ["store", "save", "remember"]):
            use_cases.append(f"Use {tool.name} to persist information across sessions")
        
        return use_cases[:3]  # Limit to 3 use cases
    
    def _extract_keywords_from_tool(self, tool, server_name: str) -> Set[str]:
        """Extract keywords from MCP tool for search."""
        keywords = set()
        
        # Add tool name parts
        keywords.update(tool.name.lower().split("_"))
        keywords.update(tool.name.lower().split("-"))
        
        # Add server name parts
        keywords.update(server_name.lower().split("_"))
        keywords.update(server_name.lower().split("-"))
        
        # Add description keywords
        if tool.description:
            # Extract meaningful words from description
            words = re.findall(r'\b\w+\b', tool.description.lower())
            keywords.update(word for word in words if len(word) > 2)
        
        # Add common MCP keywords
        keywords.update({"mcp", "tool", "server"})
        
        return keywords
    
    def _categorize_tool_from_info(self, tool_name: str, description: str, server_name: str) -> str:
        """Categorize a tool based on available information."""
        tool_text = f"{tool_name} {description or ''} {server_name}".lower()
        
        # Check each category's keywords
        for category, keywords in self.intent_categories.items():
            if any(keyword in tool_text for keyword in keywords):
                return category
        
        # Default category
        return "api_integration"
    
    async def _index_tool_as_memory(self, tool: MCPToolInfo) -> None:
        """
        Store a tool as a searchable memory.
        
        Args:
            tool: MCP tool information to store
        """
        memory_content = {
            "tool_name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "server_name": tool.server_name,
            "use_cases": tool.use_cases,
            "keywords": list(tool.keywords),
            "category": tool.category,
            "usage_examples": self._generate_usage_examples(tool),
            "when_to_use": self._generate_when_to_use(tool)
        }
        
        await self.domain_manager.store_memory(
            memory_type="mcp_tool",
            content=memory_content,
            importance=0.9,  # High importance for tool discovery
            metadata={
                "category": "mcp_tools",
                "tool_category": tool.category,
                "server_name": tool.server_name,
                "auto_indexed": True,
                "indexed_at": datetime.now().isoformat(),
                "tool_keywords": list(tool.keywords)
            },
            context={
                "purpose": "mcp_tool_discovery",
                "tool_name": tool.name,
                "auto_generated": True
            }
        )
    
    def _generate_usage_examples(self, tool: MCPToolInfo) -> List[str]:
        """Generate usage examples for a tool."""
        examples = []
        
        if tool.category == "database":
            examples = [
                f"Use {tool.name} to query user data instead of writing SQL scripts",
                f"Run {tool.name} for database operations rather than psql commands",
                f"Execute {tool.name} to retrieve data directly from the database"
            ]
        elif tool.category == "web_automation":
            examples = [
                f"Use {tool.name} to interact with web pages instead of manual browsing",
                f"Run {tool.name} for web automation rather than writing browser scripts",
                f"Execute {tool.name} to extract data from websites automatically"
            ]
        elif tool.category == "memory_management":
            examples = [
                f"Use {tool.name} to persist information instead of asking users to repeat",
                f"Run {tool.name} to find relevant context before providing answers",
                f"Execute {tool.name} to build knowledge over time"
            ]
        else:
            examples = [f"Use {tool.name} when you need to {tool.description.lower()}"]
        
        return examples
    
    def _generate_when_to_use(self, tool: MCPToolInfo) -> str:
        """Generate guidance on when to use this tool."""
        category_guidance = {
            "database": "When you need to query databases, use this MCP tool instead of writing SQL scripts or using command-line tools",
            "web_automation": "When you need to interact with web pages, use this MCP tool instead of manual browsing or writing automation scripts",
            "memory_management": "When you need to remember or recall information, use this MCP tool to persist knowledge across sessions",
            "file_operations": "When you need to work with files, use this MCP tool instead of writing file manipulation scripts",
            "api_integration": "When you need to make API calls, use this MCP tool instead of writing HTTP request code"
        }
        
        return category_guidance.get(tool.category, f"Use this tool when you need to {tool.description.lower()}")
    
    async def _store_indexing_summary(self, indexed_count: int, total_discovered: int) -> None:
        """Store a summary of the indexing process."""
        summary_content = {
            "indexing_summary": True,
            "indexed_tools_count": indexed_count,
            "total_discovered_count": total_discovered,
            "success_rate": (indexed_count / total_discovered * 100) if total_discovered > 0 else 0,
            "indexed_at": datetime.now().isoformat(),
            "tool_categories": list(set(tool.category for tool in self.indexed_tools.values())),
            "available_servers": list(set(tool.server_name for tool in self.indexed_tools.values()))
        }
        
        await self.domain_manager.store_memory(
            memory_type="mcp_indexing_summary",
            content=summary_content,
            importance=0.8,
            metadata={
                "category": "system_info",
                "auto_generated": True,
                "indexing_session": datetime.now().isoformat()
            },
            context={
                "purpose": "mcp_tool_indexing_summary"
            }
        )
    
    async def suggest_tools_for_intent(self, user_request: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Suggest MCP tools based on user intent.
        
        Args:
            user_request: User's request or description of what they want to do
            limit: Maximum number of tools to suggest
            
        Returns:
            List of suggested tools with relevance scores
        """
        # Extract keywords from user request
        request_keywords = self._extract_keywords_from_request(user_request)
        
        # Search for relevant tools
        query = f"MCP tool {' '.join(request_keywords)}"
        tool_memories = await self.domain_manager.retrieve_memories(
            query=query,
            types=["mcp_tool"],
            limit=limit * 2,  # Get more to filter and rank
            min_similarity=0.3
        )
        
        # Rank and filter suggestions
        suggestions = []
        for memory in tool_memories[:limit]:
            tool_info = memory['content']
            suggestion = {
                "tool_name": tool_info['tool_name'],
                "description": tool_info['description'],
                "server_name": tool_info['server_name'],
                "relevance_reason": self._explain_relevance(tool_info, request_keywords),
                "usage_hint": self._get_usage_hint(tool_info, user_request)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _extract_keywords_from_request(self, request: str) -> List[str]:
        """Extract relevant keywords from user request."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', request.lower())
        
        # Filter out common stop words
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'can', 'could', 'should', 'would', 'need', 'want'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _explain_relevance(self, tool_info: Dict[str, Any], request_keywords: List[str]) -> str:
        """Explain why this tool is relevant to the user's request."""
        tool_keywords = set(tool_info.get('keywords', []))
        matching_keywords = set(request_keywords) & tool_keywords
        
        if matching_keywords:
            return f"Matches keywords: {', '.join(matching_keywords)}"
        else:
            return f"Relevant for {tool_info.get('category', 'general')} tasks"
    
    def _get_usage_hint(self, tool_info: Dict[str, Any], user_request: str) -> str:
        """Get a usage hint for this tool based on the user's request."""
        usage_examples = tool_info.get('usage_examples', [])
        if usage_examples:
            return usage_examples[0]  # Return the first usage example
        else:
            return f"Use {tool_info['tool_name']} to {tool_info['description'].lower()}"


class MCPToolSuggester:
    """
    Proactive MCP tool suggestion system.
    
    This class analyzes user requests and proactively suggests relevant MCP tools
    before Claude tries alternative approaches.
    """
    
    def __init__(self, tool_indexer: MCPToolIndexer):
        """
        Initialize the MCP tool suggester.
        
        Args:
            tool_indexer: MCP tool indexer instance
        """
        self.tool_indexer = tool_indexer
        
        # Patterns that indicate indirect approaches
        self.indirect_patterns = [
            r"write.*script",
            r"create.*file.*to",
            r"use.*psql",
            r"use.*mysql",
            r"manual.*browse",
            r"copy.*paste",
            r"write.*code.*to",
            r"install.*and.*run"
        ]
    
    async def analyze_and_suggest(self, user_request: str) -> Optional[str]:
        """
        Analyze user request and suggest MCP tools if appropriate.
        
        Args:
            user_request: User's request
            
        Returns:
            Suggestion message or None if no suggestions
        """
        # Check if user is likely to use indirect methods
        if self._would_use_indirect_method(user_request):
            suggestions = await self.tool_indexer.suggest_tools_for_intent(user_request)
            
            if suggestions:
                return self._format_suggestions(suggestions)
        
        return None
    
    def _would_use_indirect_method(self, request: str) -> bool:
        """Check if the request would likely lead to indirect methods."""
        request_lower = request.lower()
        
        # Check for indirect patterns
        for pattern in self.indirect_patterns:
            if re.search(pattern, request_lower):
                return True
        
        # Check for database-related requests that might use psql
        if any(word in request_lower for word in ['database', 'sql', 'query', 'table']):
            if not any(word in request_lower for word in ['mcp', 'tool']):
                return True
        
        # Check for web-related requests that might use manual methods
        if any(word in request_lower for word in ['website', 'web page', 'browser', 'navigate']):
            if not any(word in request_lower for word in ['mcp', 'tool', 'playwright']):
                return True
        
        return False
    
    def _format_suggestions(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format tool suggestions into a user-friendly message."""
        if not suggestions:
            return ""
        
        message = "💡 **MCP Tool Suggestion**: Instead of writing scripts or using indirect methods, consider these MCP tools:\n\n"
        
        for suggestion in suggestions:
            message += f"**{suggestion['tool_name']}** ({suggestion['server_name']})\n"
            message += f"- {suggestion['description']}\n"
            message += f"- {suggestion['usage_hint']}\n"
            message += f"- Why relevant: {suggestion['relevance_reason']}\n\n"
        
        message += "Using MCP tools directly is often faster and more reliable than alternative approaches!"
        
        return message