"""
Enhanced MCP Tool Discovery and Indexing System.

This module provides comprehensive MCP tool discovery with native Claude Code integration,
hook-based learning, resource reference monitoring, and slash command discovery.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from loguru import logger

# Import enhanced discovery components
from .native_discovery import NativeMCPDiscoveryBridge
from .hook_integration import MCPHookIntegration
from .workflow_memory import WorkflowMemoryEnhancer
from .resource_reference_monitor import ResourceReferenceMonitor
from .slash_command_discovery import SlashCommandDiscovery
from .performance_optimization import PerformanceOptimizer
from .cache_integration import (
    MCPCacheAdapter, get_mcp_cache_adapter,
    cache_mcp_servers, get_cached_mcp_servers,
    cache_mcp_tools, get_cached_mcp_tools
)
from ..core.unified_cache import CacheType

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
    Enhanced MCP Tool Discovery and Indexing System.
    
    This class provides comprehensive MCP tool discovery with:
    - Native Claude Code integration
    - Hook-based learning from real-time usage
    - Resource reference monitoring and suggestions
    - Slash command discovery and management
    - Advanced workflow memory for pattern learning
    """
    
    def __init__(self, domain_manager):
        """
        Initialize the enhanced MCP tool indexer.
        
        Args:
            domain_manager: Memory domain manager for storing tool information
        """
        self.domain_manager = domain_manager
        self.indexed_tools: Dict[str, MCPToolInfo] = {}
        self.discovered_servers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize enhanced discovery components
        self.native_bridge = NativeMCPDiscoveryBridge()
        self.hook_integration = MCPHookIntegration(self)
        self.workflow_enhancer = WorkflowMemoryEnhancer(domain_manager)
        self.resource_monitor = ResourceReferenceMonitor()
        self.cache_adapter = get_mcp_cache_adapter()
        self.slash_discovery = SlashCommandDiscovery(self)
        
        # Initialize performance optimization
        self.performance_optimizer = PerformanceOptimizer()
        
        # Legacy cache support (now delegated to performance optimizer)
        self.discovery_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self._cache_timestamps = {}
        
        # Enhanced integration status
        self.integration_status = {
            'native_discovery_enabled': False,
            'hook_learning_enabled': False,
            'resource_monitoring_enabled': False,
            'slash_commands_enabled': False,
            'enhanced_memory_enabled': False
        }
        
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
        Enhanced discovery of all available MCP tools with performance optimization.
        
        Returns:
            Dictionary of discovered tools by name
        """
        logger.info("Starting enhanced MCP tool discovery and indexing...")
        
        # Use performance monitoring for the entire discovery process
        @self.performance_optimizer.performance_monitor("complete_discovery")
        async def _discovery_workflow():
            # Phase 1: Initialize enhanced components
            await self._initialize_enhanced_components()
            
            # Phase 2: Optimized parallel server and tool discovery
            discovery_tasks = {
                'servers_discovery': self._discover_servers_comprehensive,
                'cache_warmup': lambda: self._warmup_discovery_cache()
            }
            
            # Use performance optimizer for parallel execution
            workflow_results = await self.performance_optimizer.optimize_discovery_workflow({
                'native_discovery': lambda: self.native_bridge.discover_native_servers(),
                'config_discovery': lambda: self._discover_servers_from_configuration(),
                'env_discovery': lambda: self._get_servers_from_environment()
            })
            
            # Combine server discovery results
            servers = {}
            for result in workflow_results.values():
                if isinstance(result, dict):
                    servers.update(result)
            
            self.discovered_servers = servers
            
            # Phase 3: Optimized tool discovery with caching
            tools = await self._discover_tools_enhanced_with_optimization(servers)
            
            # Phase 4: Parallel slash command discovery
            await self._discover_and_store_slash_commands(servers)
            
            # Phase 5: Batch tool indexing for performance
            indexed_count = await self._batch_index_tools(tools)
            
            # Phase 6: Store enhanced summary with performance metrics
            await self._store_enhanced_summary_with_metrics(indexed_count, len(tools), servers)
            
            return self.indexed_tools
        
        # Execute optimized workflow
        result = await _discovery_workflow()
        
        # Generate performance report
        performance_report = self.performance_optimizer.get_performance_report()
        if performance_report['status'] == 'active':
            avg_time_ms = performance_report['overall_stats']['avg_duration_ms']
            target_ms = 500  # 500ms target
            
            if avg_time_ms <= target_ms:
                logger.info(f"✅ Performance target met: {avg_time_ms:.1f}ms ≤ {target_ms}ms")
            else:
                logger.warning(f"⚠️ Performance target missed: {avg_time_ms:.1f}ms > {target_ms}ms")
        
        logger.info(f"Successfully indexed {len(result)} MCP tools from {len(self.discovered_servers)} servers")
        return result
    
    async def _initialize_enhanced_components(self):
        """Initialize all enhanced discovery components."""
        try:
            # Set up hook integration if possible
            if await self.hook_integration.setup_hooks():
                self.integration_status['hook_learning_enabled'] = True
                logger.info("Hook-based learning enabled")
            
            # Validate native integration
            validation = await self.native_bridge.validate_native_integration()
            if validation['native_discovery_available']:
                self.integration_status['native_discovery_enabled'] = True
                logger.info("Native Claude Code discovery enabled")
            
            # Enable other components
            self.integration_status['resource_monitoring_enabled'] = True
            self.integration_status['slash_commands_enabled'] = MCP_CLIENT_AVAILABLE
            self.integration_status['enhanced_memory_enabled'] = True
            
            logger.info(f"Enhanced components initialized: {sum(self.integration_status.values())}/5 enabled")
            
        except Exception as e:
            logger.warning(f"Enhanced component initialization failed: {e}")
    
    async def _discover_servers_comprehensive(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive server discovery from all available sources with unified caching."""
        import hashlib
        import json
        
        # Generate cache key based on configuration
        config_data = {
            'native_enabled': self.integration_status['native_discovery_enabled'],
            'timestamp': int(time.time() // 300)  # 5-minute cache buckets
        }
        config_hash = hashlib.md5(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:8]
        
        # Check unified cache first
        cached_servers = await get_cached_mcp_servers(config_hash, {
            'discovery_type': 'comprehensive',
            'native_enabled': self.integration_status['native_discovery_enabled']
        })
        
        if cached_servers:
            logger.debug(f"Using cached server discovery: {len(cached_servers)} servers")
            self.discovered_servers = cached_servers
            return cached_servers
        
        # Perform discovery
        start_time = time.time()
        servers = {}
        
        # Method 1: Native Claude Code discovery (highest priority)
        if self.integration_status['native_discovery_enabled']:
            try:
                native_servers = await self.native_bridge.discover_native_servers()
                servers.update(native_servers)
                logger.info(f"Found {len(native_servers)} servers via native discovery")
            except Exception as e:
                logger.warning(f"Native server discovery failed: {e}")
        
        # Method 2: Configuration-based discovery
        try:
            config_servers = await self._discover_servers_from_configuration()
            # Merge with priority to native discovery
            for name, config in config_servers.items():
                if name not in servers:
                    servers[name] = config
            logger.debug(f"Found {len(config_servers)} additional servers from configuration")
        except Exception as e:
            logger.warning(f"Configuration server discovery failed: {e}")
        
        # Method 3: Environment-based discovery
        try:
            env_servers = self._get_servers_from_environment()
            for name, config in env_servers.items():
                if name not in servers:
                    servers[name] = config
            logger.debug(f"Found {len(env_servers)} additional servers from environment")
        except Exception as e:
            logger.warning(f"Environment server discovery failed: {e}")
        
        # Record performance metrics
        discovery_time = time.time() - start_time
        self.cache_adapter.record_discovery_time('comprehensive_server_discovery', discovery_time)
        
        # Cache discovered servers
        await cache_mcp_servers(config_hash, servers, {
            'discovery_method': 'comprehensive',
            'server_count': len(servers),
            'discovery_time': discovery_time
        })
        
        self.discovered_servers = servers
        return servers
    
    async def _discover_tools_enhanced(self, servers: Dict[str, Dict[str, Any]]) -> List[MCPToolInfo]:
        """Enhanced tool discovery with comprehensive analysis."""
        tools = []
        
        # Discover tools from each server
        for server_name, server_config in servers.items():
            try:
                server_tools = await self._discover_tools_from_server_enhanced(server_name, server_config)
                tools.extend(server_tools)
                logger.debug(f"Discovered {len(server_tools)} tools from {server_name}")
            except Exception as e:
                logger.warning(f"Failed to discover tools from {server_name}: {e}")
        
        # Add known/common tools as fallback
        try:
            known_tools = await self._discover_known_tools()
            tools.extend(known_tools)
            logger.debug(f"Added {len(known_tools)} known tools")
        except Exception as e:
            logger.warning(f"Failed to add known tools: {e}")
        
        return tools
    
    async def _discover_and_store_slash_commands(self, servers: Dict[str, Dict[str, Any]]):
        """Discover and store slash commands from MCP servers."""
        if not self.integration_status['slash_commands_enabled']:
            logger.debug("Slash command discovery disabled")
            return
        
        total_commands = 0
        for server_name, server_config in servers.items():
            try:
                commands = await self.slash_discovery.discover_slash_commands(server_name, server_config)
                if commands:
                    await self.slash_discovery.store_slash_commands(commands)
                    total_commands += len(commands)
                    logger.debug(f"Discovered {len(commands)} slash commands from {server_name}")
            except Exception as e:
                logger.debug(f"Slash command discovery failed for {server_name}: {e}")
        
        if total_commands > 0:
            logger.info(f"Discovered and stored {total_commands} slash commands")
    
    async def _discover_servers_from_configuration(self) -> Dict[str, Dict[str, Any]]:
        """Enhanced configuration discovery with multiple sources."""
        servers = {}
        
        # Legacy environment discovery
        servers.update(self._get_servers_from_environment())
        
        # Configuration files discovery
        config_files = [
            "~/.config/claude-code/config.json",
            "~/.claude/mcp-servers.json", 
            "./.claude-code/config.json",
            "./.mcp.json",
            "./.mcp.json-dev"
        ]
        
        for config_path in config_files:
            if not config_path:
                continue
            
            try:
                expanded_path = os.path.expanduser(config_path)
                if os.path.exists(expanded_path):
                    with open(expanded_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Handle different config formats
                    if 'mcpServers' in config_data:
                        servers.update(config_data['mcpServers'])
                    elif 'servers' in config_data:
                        servers.update(config_data['servers'])
                    elif 'command' in config_data or 'module' in config_data:
                        # Single server config
                        server_name = os.path.basename(config_path).replace('.json', '').replace('.mcp', '')
                        servers[server_name] = config_data
                    
                    logger.debug(f"Loaded configuration from {config_path}")
                    
            except Exception as e:
                logger.debug(f"Could not load configuration from {config_path}: {e}")
        
        return servers
    
    async def _discover_tools_from_server_enhanced(self, server_name: str, server_config: Dict[str, Any]) -> List[MCPToolInfo]:
        """Enhanced tool discovery from individual server with comprehensive analysis and caching."""
        # Check unified cache first
        cached_tools = await get_cached_mcp_tools(server_name, {
            'server_config': server_config,
            'mcp_available': MCP_CLIENT_AVAILABLE
        })
        
        if cached_tools:
            logger.debug(f"Using cached tool discovery for {server_name}: {len(cached_tools)} tools")
            # Convert cached tools back to MCPToolInfo objects
            return [MCPToolInfo(**tool) if isinstance(tool, dict) else tool for tool in cached_tools]
        
        # Perform discovery
        start_time = time.time()
        tools = []
        
        try:
            if MCP_CLIENT_AVAILABLE:
                # Use MCP client for live discovery
                tools = await self._discover_tools_from_server_config(server_name, server_config)
            else:
                # Fallback to configuration-based tool inference
                tools = await self._infer_tools_from_server_config(server_name, server_config)
                
        except Exception as e:
            logger.debug(f"Enhanced tool discovery failed for {server_name}: {e}")
            # Fallback to basic tool inference
            tools = await self._infer_tools_from_server_config(server_name, server_config)
        
        # Record performance and cache results
        discovery_time = time.time() - start_time
        self.cache_adapter.record_discovery_time(f'tool_discovery_{server_name}', discovery_time)
        
        # Cache the discovered tools
        await cache_mcp_tools(server_name, [tool.to_dict() if hasattr(tool, 'to_dict') else tool.__dict__ for tool in tools], {
            'discovery_method': 'live' if MCP_CLIENT_AVAILABLE else 'inferred',
            'tool_count': len(tools),
            'discovery_time': discovery_time
        })
        
        return tools
    
    async def _infer_tools_from_server_config(self, server_name: str, server_config: Dict[str, Any]) -> List[MCPToolInfo]:
        """Infer likely tools from server configuration when live discovery isn't possible."""
        tools = []
        
        # Analyze server name and configuration for tool patterns
        server_patterns = {
            'filesystem': ['read_file', 'write_file', 'list_directory', 'create_directory'],
            'database': ['query_database', 'execute_sql', 'get_schema'],
            'web': ['fetch_url', 'post_request', 'get_request'],
            'git': ['git_status', 'git_commit', 'git_push', 'git_pull'],
            'api': ['call_api', 'get_endpoint', 'post_endpoint'],
        }
        
        server_name_lower = server_name.lower()
        for pattern, tool_names in server_patterns.items():
            if pattern in server_name_lower:
                for tool_name in tool_names:
                    tool_info = MCPToolInfo(
                        name=f"{server_name}_{tool_name}",
                        description=f"Inferred {tool_name} capability from {server_name} server",
                        parameters={},
                        server_name=server_name,
                        use_cases=[f"{pattern}_operations"],
                        keywords={pattern, tool_name, server_name.lower()},
                        category=pattern
                    )
                    tools.append(tool_info)
                break
        
        return tools
    
    async def _index_tool_as_memory_enhanced(self, tool: MCPToolInfo):
        """Enhanced tool indexing with comprehensive metadata and analytics."""
        try:
            # Enhanced memory content with additional context
            memory_content = {
                "tool_name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "server_name": tool.server_name,
                "use_cases": tool.use_cases,
                "keywords": list(tool.keywords),
                "category": tool.category,
                "discovery_method": "enhanced_discovery",
                "indexed_at": datetime.now().isoformat(),
                "integration_features": {
                    "native_discovery": self.integration_status['native_discovery_enabled'],
                    "hook_learning": self.integration_status['hook_learning_enabled'],
                    "resource_monitoring": self.integration_status['resource_monitoring_enabled']
                }
            }
            
            # Enhanced metadata for better retrieval
            metadata = {
                "category": "mcp_tools",
                "tool_category": tool.category,
                "server": tool.server_name,
                "keywords": list(tool.keywords),
                "use_cases": tool.use_cases,
                "discovery_enhanced": True,
                "parameters_count": len(tool.parameters),
                "complexity": "simple" if len(tool.parameters) <= 3 else "complex"
            }
            
            # Store as memory
            await self.domain_manager.store_memory(
                memory_type="mcp_tool",
                content=json.dumps(memory_content),
                importance=0.9,  # High importance for MCP tools
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to index tool {tool.name} as enhanced memory: {e}")
            raise
    
    async def _store_enhanced_summary(self, indexed_count: int, total_tools: int, servers: Dict[str, Dict[str, Any]]):
        """Store comprehensive summary with enhanced analytics."""
        try:
            summary = {
                "indexing_completed_at": datetime.now().isoformat(),
                "tools_indexed": indexed_count,
                "tools_discovered": total_tools,
                "servers_discovered": len(servers),
                "integration_status": self.integration_status,
                "discovery_sources": {
                    "native_claude_code": self.integration_status['native_discovery_enabled'],
                    "configuration_files": True,
                    "environment_variables": True,
                    "mcp_client": MCP_CLIENT_AVAILABLE
                },
                "enhanced_features": {
                    "slash_commands": self.integration_status['slash_commands_enabled'],
                    "resource_monitoring": self.integration_status['resource_monitoring_enabled'],
                    "workflow_memory": self.integration_status['enhanced_memory_enabled'],
                    "hook_learning": self.integration_status['hook_learning_enabled']
                },
                "server_summary": [
                    {
                        "name": name,
                        "type": config.get("command", config.get("module", "unknown")),
                        "source": config.get("source", "configuration")
                    }
                    for name, config in servers.items()
                ]
            }
            
            await self.domain_manager.store_memory(
                memory_type="mcp_indexing_summary",
                content=json.dumps(summary),
                importance=0.8,
                metadata={
                    "category": "system_status",
                    "enhanced_discovery": True,
                    "servers_count": len(servers),
                    "tools_count": indexed_count
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store enhanced summary: {e}")
    
    # Public interface methods for enhanced functionality
    
    async def get_discovered_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get all discovered MCP servers."""
        return self.discovered_servers.copy()
    
    async def get_integration_status(self) -> Dict[str, bool]:
        """Get status of enhanced integration features."""
        return self.integration_status.copy()
    
    async def get_resource_suggestions(self, prompt: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get resource reference suggestions for a prompt."""
        if not self.integration_status['resource_monitoring_enabled']:
            return []
        
        try:
            available_servers = list(self.discovered_servers.keys())
            return await self.resource_monitor.get_reference_suggestions(prompt, available_servers)
        except Exception as e:
            logger.error(f"Failed to get resource suggestions: {e}")
            return []
    
    async def get_slash_command_suggestions(self, prompt: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get contextual slash command suggestions."""
        if not self.integration_status['slash_commands_enabled']:
            return []
        
        try:
            suggestions = await self.slash_discovery.get_contextual_suggestions(prompt, context)
            return [suggestion.to_dict() for suggestion in suggestions]
        except Exception as e:
            logger.error(f"Failed to get slash command suggestions: {e}")
            return []
    
    async def get_workflow_suggestions(self, current_task: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get workflow pattern suggestions based on learned patterns."""
        if not self.integration_status['enhanced_memory_enabled']:
            return []
        
        try:
            from .workflow_memory import SuggestionContext
            
            suggestion_context = SuggestionContext(
                current_task=current_task,
                user_intent=context.get('intent', ''),
                project_type=context.get('project_type'),
                recent_tools_used=context.get('recent_tools', []),
                recent_failures=context.get('failures', []),
                environment_info=context.get('env', {}),
                available_servers=list(self.discovered_servers.keys())
            )
            
            return await self.workflow_enhancer.get_workflow_suggestions(suggestion_context)
        except Exception as e:
            logger.error(f"Failed to get workflow suggestions: {e}")
            return []
    
    async def learn_from_tool_usage(self, tool_name: str, context: Dict[str, Any], success: bool = True):
        """Learn from successful/failed tool usage for future suggestions."""
        try:
            # Update workflow patterns
            if self.integration_status['enhanced_memory_enabled']:
                pattern_data = {
                    "context": context.get("user_request", ""),
                    "tools": [tool_name],
                    "success": success,
                    "score": 0.9 if success else 0.3,
                    "project_type": context.get("project_type", "unknown"),
                    "intent": context.get("intent", "")
                }
                await self.workflow_enhancer.store_mcp_workflow_pattern(pattern_data)
            
            # Learn resource reference patterns
            if self.integration_status['resource_monitoring_enabled']:
                resource_ref = context.get('resource_reference')
                if resource_ref:
                    await self.resource_monitor.learn_resource_pattern(
                        resource_ref, context, success, context.get('response_time', 0.0)
                    )
            
            # Learn slash command usage
            if self.integration_status['slash_commands_enabled'] and tool_name.startswith('/mcp__'):
                await self.slash_discovery.learn_from_command_usage(
                    tool_name, success, context.get('execution_time', 0.0), context
                )
                
        except Exception as e:
            logger.error(f"Failed to learn from tool usage: {e}")
    
    async def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about MCP discovery and usage."""
        analytics = {
            "discovery_status": {
                "total_servers": len(self.discovered_servers),
                "total_tools": len(self.indexed_tools),
                "integration_features": self.integration_status
            },
            "resource_monitoring": {},
            "slash_commands": {},
            "workflow_patterns": {}
        }
        
        try:
            if self.integration_status['resource_monitoring_enabled']:
                analytics["resource_monitoring"] = self.resource_monitor.get_performance_stats()
        except Exception as e:
            logger.debug(f"Resource monitoring analytics failed: {e}")
        
        try:
            if self.integration_status['slash_commands_enabled']:
                analytics["slash_commands"] = await self.slash_discovery.get_command_analytics()
        except Exception as e:
            logger.debug(f"Slash command analytics failed: {e}")
        
        try:
            if self.integration_status['enhanced_memory_enabled']:
                analytics["workflow_patterns"] = await self.workflow_enhancer.get_pattern_analytics()
        except Exception as e:
            logger.debug(f"Workflow pattern analytics failed: {e}")
        
        return analytics
    
    async def invalidate_discovery_cache(self, server_name: str = None):
        """Invalidate discovery caches to force fresh discovery."""
        try:
            if server_name:
                # Invalidate specific server
                if server_name in self.discovery_cache:
                    del self.discovery_cache[server_name]
                if server_name in self._cache_timestamps:
                    del self._cache_timestamps[server_name]
                
                # Invalidate native bridge cache
                self.native_bridge.invalidate_cache()
                
                # Invalidate slash discovery cache
                if self.integration_status['slash_commands_enabled']:
                    self.slash_discovery.invalidate_cache(server_name)
            else:
                # Invalidate all caches
                self.discovery_cache.clear()
                self._cache_timestamps.clear()
                self.native_bridge.invalidate_cache()
                if self.integration_status['slash_commands_enabled']:
                    self.slash_discovery.invalidate_cache()
            
            logger.info(f"Discovery cache invalidated for {server_name or 'all servers'}")
        except Exception as e:
            logger.error(f"Failed to invalidate discovery cache: {e}")
    
    # Performance-optimized helper methods
    
    async def _warmup_discovery_cache(self):
        """Warm up caches for better performance."""
        try:
            warmup_functions = {
                'native_servers': lambda: asyncio.create_task(self.native_bridge.discover_native_servers()),
                'config_parsing': lambda: self._discover_servers_from_configuration(),
            }
            
            await self.performance_optimizer.warmup_cache(warmup_functions)
        except Exception as e:
            logger.debug(f"Cache warmup failed: {e}")
    
    async def _discover_tools_enhanced_with_optimization(self, servers: Dict[str, Dict[str, Any]]) -> List[MCPToolInfo]:
        """Optimized tool discovery with parallel processing and caching."""
        tools = []
        
        # Batch servers for parallel processing
        batch_size = self.performance_optimizer.batch_sizes.get('servers', 5)
        server_batches = [
            dict(list(servers.items())[i:i + batch_size])
            for i in range(0, len(servers), batch_size)
        ]
        
        # Process batches in parallel
        for batch in server_batches:
            batch_tasks = [
                (self._discover_tools_from_server_enhanced, (name, config), {})
                for name, config in batch.items()
            ]
            
            batch_results = await self.performance_optimizer.executor.execute_parallel(batch_tasks)
            
            for result in batch_results:
                if result:
                    tools.extend(result)
        
        # Add known tools as fallback
        try:
            known_tools = await self._discover_known_tools()
            tools.extend(known_tools)
        except Exception as e:
            logger.debug(f"Known tools discovery failed: {e}")
        
        return tools
    
    async def _batch_index_tools(self, tools: List[MCPToolInfo]) -> int:
        """Batch index tools for better performance."""
        indexed_count = 0
        batch_size = self.performance_optimizer.batch_sizes.get('tools', 20)
        
        # Process tools in batches to avoid overwhelming the memory system
        for i in range(0, len(tools), batch_size):
            batch = tools[i:i + batch_size]
            batch_tasks = []
            
            for tool in batch:
                batch_tasks.append((self._index_tool_as_memory_enhanced, (tool,), {}))
            
            try:
                # Execute batch in parallel
                results = await self.performance_optimizer.executor.execute_parallel(batch_tasks)
                
                # Count successful indexing
                for j, result in enumerate(results):
                    if result is not None:  # Success
                        tool = batch[j]
                        self.indexed_tools[tool.name] = tool
                        indexed_count += 1
                    else:  # Failed
                        tool = batch[j]
                        logger.warning(f"Failed to index tool {tool.name}")
                        
            except Exception as e:
                logger.error(f"Batch indexing failed: {e}")
                # Fallback to sequential processing for this batch
                for tool in batch:
                    try:
                        await self._index_tool_as_memory_enhanced(tool)
                        self.indexed_tools[tool.name] = tool
                        indexed_count += 1
                    except Exception as tool_error:
                        logger.error(f"Failed to index tool {tool.name}: {tool_error}")
        
        return indexed_count
    
    async def _store_enhanced_summary_with_metrics(self, indexed_count: int, total_tools: int, servers: Dict[str, Dict[str, Any]]):
        """Store enhanced summary with performance metrics."""
        try:
            # Get performance report
            performance_report = self.performance_optimizer.get_performance_report()
            memory_usage = self.performance_optimizer.get_memory_usage()
            
            summary = {
                "indexing_completed_at": datetime.now().isoformat(),
                "tools_indexed": indexed_count,
                "tools_discovered": total_tools,
                "servers_discovered": len(servers),
                "integration_status": self.integration_status,
                "performance_metrics": {
                    "avg_response_time_ms": performance_report.get('overall_stats', {}).get('avg_duration_ms', 0),
                    "cache_hit_rate": performance_report.get('overall_stats', {}).get('cache_hit_rate', 0),
                    "target_compliance": performance_report.get('target_compliance', {}),
                    "memory_usage": memory_usage
                },
                "optimization_applied": performance_report.get('optimization_recommendations', []),
                "discovery_sources": {
                    "native_claude_code": self.integration_status['native_discovery_enabled'],
                    "configuration_files": True,
                    "environment_variables": True,
                    "mcp_client": MCP_CLIENT_AVAILABLE
                },
                "enhanced_features": {
                    "slash_commands": self.integration_status['slash_commands_enabled'],
                    "resource_monitoring": self.integration_status['resource_monitoring_enabled'],
                    "workflow_memory": self.integration_status['enhanced_memory_enabled'],
                    "hook_learning": self.integration_status['hook_learning_enabled'],
                    "performance_optimization": True
                },
                "server_summary": [
                    {
                        "name": name,
                        "type": config.get("command", config.get("module", "unknown")),
                        "source": config.get("source", "configuration")
                    }
                    for name, config in servers.items()
                ]
            }
            
            await self.domain_manager.store_memory(
                memory_type="mcp_indexing_summary_enhanced",
                content=json.dumps(summary),
                importance=0.9,  # High importance
                metadata={
                    "category": "system_status",
                    "enhanced_discovery": True,
                    "performance_optimized": True,
                    "servers_count": len(servers),
                    "tools_count": indexed_count,
                    "performance_target_met": performance_report.get('target_compliance', {}).get('meets_target', False)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store enhanced summary with metrics: {e}")
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics."""
        try:
            base_analytics = await self.get_comprehensive_analytics()
            performance_report = self.performance_optimizer.get_performance_report()
            memory_usage = self.performance_optimizer.get_memory_usage()
            
            # Combine all analytics
            performance_analytics = {
                **base_analytics,
                "performance_optimization": {
                    "status": "enabled",
                    "cache_stats": performance_report.get('cache_stats', {}),
                    "response_times": performance_report.get('overall_stats', {}),
                    "target_compliance": performance_report.get('target_compliance', {}),
                    "memory_usage": memory_usage,
                    "optimization_recommendations": performance_report.get('optimization_recommendations', []),
                    "operation_performance": performance_report.get('operation_stats', {})
                }
            }
            
            return performance_analytics
            
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {"error": str(e), "performance_optimization": {"status": "error"}}
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status and metrics."""
        try:
            performance_report = self.performance_optimizer.get_performance_report()
            
            if performance_report['status'] == 'no_metrics':
                return {
                    "status": "inactive",
                    "message": "No performance metrics collected yet",
                    "cache_stats": performance_report.get('cache_stats', {})
                }
            
            target_compliance = performance_report.get('target_compliance', {})
            overall_stats = performance_report.get('overall_stats', {})
            
            return {
                "status": "active",
                "performance_target_met": target_compliance.get('meets_target', False),
                "avg_response_time_ms": overall_stats.get('avg_duration_ms', 0),
                "target_response_time_ms": 500,
                "cache_hit_rate": overall_stats.get('cache_hit_rate', 0),
                "total_operations": overall_stats.get('total_operations', 0),
                "optimization_recommendations": performance_report.get('optimization_recommendations', []),
                "cache_stats": performance_report.get('cache_stats', {}),
                "memory_usage": self.performance_optimizer.get_memory_usage()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _discover_tools(self) -> List[MCPToolInfo]:
        """
        Discover available MCP tools from various sources.
        
        Returns:
            List of discovered MCP tool information
        """
        tools = []
        
        # Method 1: Discover from Claude Code native configuration (highest priority)
        try:
            native_tools = await self._discover_from_claude_code_native()
            tools.extend(native_tools)
            logger.debug(f"Discovered {len(native_tools)} tools from Claude Code native")
        except Exception as e:
            logger.warning(f"Could not discover from Claude Code native: {e}")
        
        # Method 2: Discover from MCP server instances (if available)
        if MCP_AVAILABLE:
            try:
                mcp_tools = await self._discover_from_mcp_servers()
                tools.extend(mcp_tools)
            except Exception as e:
                logger.warning(f"Could not discover from MCP servers: {e}")
        
        # Method 3: Discover from environment/configuration
        try:
            config_tools = await self._discover_from_configuration()
            tools.extend(config_tools)
        except Exception as e:
            logger.warning(f"Could not discover from configuration: {e}")
        
        # Method 4: Discover common/known MCP tools
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
    
    async def _discover_from_claude_code_native(self) -> List[MCPToolInfo]:
        """
        Discover tools from Claude Code's native MCP configuration.
        
        This method integrates with Claude Code's native MCP features:
        - Parse `claude mcp list` output
        - Read Claude Code native MCP configuration files
        - Use Claude Code's OAuth tokens for authentication
        
        Returns:
            List of tools discovered from Claude Code native sources
        """
        tools = []
        
        logger.debug("Discovering tools from Claude Code native configuration...")
        
        # Method 1: Parse claude mcp list command output
        try:
            claude_list_tools = await self._parse_claude_mcp_list()
            tools.extend(claude_list_tools)
            logger.debug(f"Found {len(claude_list_tools)} tools from claude mcp list")
        except Exception as e:
            logger.debug(f"Could not parse claude mcp list: {e}")
        
        # Method 2: Parse Claude Code native configuration files
        native_config_paths = [
            "~/.config/claude-code/mcp_servers.json",
            "~/.claude-code/mcp_config.json",
            "./.claude/mcp_config.json",
            os.environ.get("CLAUDE_CODE_MCP_CONFIG")
        ]
        
        for config_path in native_config_paths:
            if not config_path:
                continue
            
            expanded_path = os.path.expanduser(config_path)
            if os.path.exists(expanded_path):
                try:
                    native_config_tools = await self._parse_claude_native_config(expanded_path)
                    tools.extend(native_config_tools)
                    logger.debug(f"Found {len(native_config_tools)} tools in {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to parse Claude Code native config {config_path}: {e}")
        
        # Method 3: Discover from active Claude Code MCP registry
        try:
            registry_tools = await self._discover_from_claude_registry()
            tools.extend(registry_tools)
            logger.debug(f"Found {len(registry_tools)} tools from Claude registry")
        except Exception as e:
            logger.debug(f"Could not access Claude registry: {e}")
        
        return tools
    
    async def _parse_claude_mcp_list(self) -> List[MCPToolInfo]:
        """Parse output from `claude mcp list` command."""
        tools = []
        
        try:
            # Execute claude mcp list command if available
            import subprocess
            result = subprocess.run(
                ["claude", "mcp", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse the JSON output
                try:
                    mcp_servers = json.loads(result.stdout)
                    for server_name, server_config in mcp_servers.items():
                        server_tools = await self._extract_tools_from_claude_server(
                            server_name, server_config
                        )
                        tools.extend(server_tools)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse claude mcp list JSON output: {e}")
            else:
                logger.debug(f"claude mcp list failed: {result.stderr}")
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"Could not execute claude mcp list: {e}")
        
        return tools
    
    async def _parse_claude_native_config(self, config_path: str) -> List[MCPToolInfo]:
        """Parse Claude Code native MCP configuration file."""
        tools = []
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Handle different Claude Code config formats
            mcp_servers = config_data.get('mcp_servers', config_data.get('servers', {}))
            
            for server_name, server_config in mcp_servers.items():
                try:
                    server_tools = await self._extract_tools_from_claude_server(
                        server_name, server_config
                    )
                    tools.extend(server_tools)
                except Exception as e:
                    logger.warning(f"Failed to extract tools from server {server_name}: {e}")
        
        except Exception as e:
            logger.warning(f"Failed to parse Claude native config {config_path}: {e}")
        
        return tools
    
    async def _discover_from_claude_registry(self) -> List[MCPToolInfo]:
        """Discover tools from Claude Code's active MCP registry."""
        tools = []
        
        # This would integrate with Claude Code's internal MCP registry
        # For now, we'll implement a placeholder that could be extended
        # when Claude Code provides API access to the registry
        
        try:
            # Check if we can access Claude Code's runtime registry
            # This might involve reading from shared memory, files, or IPC
            registry_paths = [
                "/tmp/claude-code-mcp-registry.json",
                "~/.cache/claude-code/mcp-active-servers.json",
                os.environ.get("CLAUDE_CODE_REGISTRY_PATH")
            ]
            
            for registry_path in registry_paths:
                if not registry_path:
                    continue
                    
                expanded_path = os.path.expanduser(registry_path)
                if os.path.exists(expanded_path):
                    try:
                        with open(expanded_path, 'r') as f:
                            registry_data = json.load(f)
                        
                        active_servers = registry_data.get('active_servers', {})
                        for server_name, server_info in active_servers.items():
                            registry_tools = await self._extract_tools_from_registry_info(
                                server_name, server_info
                            )
                            tools.extend(registry_tools)
                            
                    except Exception as e:
                        logger.debug(f"Could not read registry from {registry_path}: {e}")
        
        except Exception as e:
            logger.debug(f"Could not access Claude Code registry: {e}")
        
        return tools
    
    async def _extract_tools_from_claude_server(
        self, 
        server_name: str, 
        server_config: Dict[str, Any]
    ) -> List[MCPToolInfo]:
        """Extract tool information from Claude Code server configuration."""
        tools = []
        
        try:
            # Handle Claude Code specific server config format
            command = server_config.get('command', server_config.get('cmd'))
            args = server_config.get('args', server_config.get('arguments', []))
            env = server_config.get('env', server_config.get('environment', {}))
            
            if command:
                # Try to connect and discover tools directly
                if MCP_CLIENT_AVAILABLE:
                    try:
                        discovered_tools = await self._discover_tools_from_server_config(
                            server_name, {
                                "command": command,
                                "args": args,
                                "env": env,
                                "type": "stdio"
                            }
                        )
                        tools.extend(discovered_tools)
                    except Exception as e:
                        logger.debug(f"Could not live-discover from {server_name}: {e}")
                        
                        # Fall back to inference if live discovery fails
                        inferred_tools = await self._infer_tools_from_server_name(server_name)
                        tools.extend(inferred_tools)
                
                else:
                    # Fall back to inference when MCP client not available
                    inferred_tools = await self._infer_tools_from_server_name(server_name)
                    tools.extend(inferred_tools)
        
        except Exception as e:
            logger.warning(f"Failed to extract tools from Claude server {server_name}: {e}")
        
        return tools
    
    async def _extract_tools_from_registry_info(
        self, 
        server_name: str, 
        server_info: Dict[str, Any]
    ) -> List[MCPToolInfo]:
        """Extract tool information from Claude Code registry info."""
        tools = []
        
        try:
            # Handle registry-specific format
            available_tools = server_info.get('tools', server_info.get('available_tools', []))
            
            for tool_info in available_tools:
                if isinstance(tool_info, dict):
                    tool = MCPToolInfo(
                        name=tool_info.get('name', ''),
                        description=tool_info.get('description', ''),
                        parameters=tool_info.get('parameters', {}),
                        server_name=server_name,
                        use_cases=tool_info.get('use_cases', []),
                        keywords=set(tool_info.get('keywords', [])),
                        category=self._categorize_tool(tool_info.get('name', ''), tool_info.get('description', ''))
                    )
                    tools.append(tool)
                
        except Exception as e:
            logger.warning(f"Failed to extract tools from registry info for {server_name}: {e}")
        
        return tools
    
    async def _infer_tools_from_server_name(self, server_name: str) -> List[MCPToolInfo]:
        """
        Infer likely tools from server name when live discovery fails.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            List of inferred tool information
        """
        tools = []
        
        # Common server name patterns and their typical tools
        server_patterns = {
            "postgres": [
                MCPToolInfo(
                    name=f"{server_name}_query",
                    description="Execute SQL queries against PostgreSQL database",
                    parameters={"query": {"type": "string", "description": "SQL query to execute"}},
                    server_name=server_name,
                    use_cases=["Run SQL queries", "Database operations", "Data retrieval"],
                    keywords={"database", "sql", "postgres", "query"},
                    category="database"
                )
            ],
            "playwright": [
                MCPToolInfo(
                    name=f"{server_name}_navigate",
                    description="Navigate web pages and interact with elements",
                    parameters={"url": {"type": "string", "description": "URL to navigate to"}},
                    server_name=server_name,
                    use_cases=["Web automation", "Browser interaction", "Web scraping"],
                    keywords={"web", "browser", "automation", "playwright"},
                    category="web_automation"
                )
            ],
            "memory": [
                MCPToolInfo(
                    name=f"{server_name}_store",
                    description="Store information in persistent memory",
                    parameters={"content": {"type": "string", "description": "Content to store"}},
                    server_name=server_name,
                    use_cases=["Remember information", "Store context", "Build knowledge base"],
                    keywords={"memory", "store", "remember", "persistent"},
                    category="memory_management"
                )
            ],
            "filesystem": [
                MCPToolInfo(
                    name=f"{server_name}_read",
                    description="Read files from the filesystem",
                    parameters={"path": {"type": "string", "description": "File path to read"}},
                    server_name=server_name,
                    use_cases=["Read files", "File operations", "Data access"],
                    keywords={"file", "read", "filesystem", "io"},
                    category="file_operations"
                )
            ]
        }
        
        # Check if server name matches any known patterns
        for pattern, pattern_tools in server_patterns.items():
            if pattern.lower() in server_name.lower():
                tools.extend(pattern_tools)
                logger.debug(f"Inferred {len(pattern_tools)} tools for {server_name} based on pattern {pattern}")
        
        # If no patterns matched, create a generic tool
        if not tools:
            generic_tool = MCPToolInfo(
                name=f"{server_name}_tool",
                description=f"Tool from {server_name} MCP server",
                parameters={"action": {"type": "string", "description": "Action to perform"}},
                server_name=server_name,
                use_cases=[f"Use {server_name} capabilities", "Server-specific operations"],
                keywords={server_name.lower(), "tool", "mcp"},
                category="api_integration"
            )
            tools.append(generic_tool)
        
        return tools
    
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
                server_name="alunai-clarity",
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
                server_name="alunai-clarity",
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
                server_name="alunai-clarity",
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