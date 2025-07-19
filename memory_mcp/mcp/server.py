"""
MCP server implementation for the memory system.
"""

import json
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from memory_mcp.mcp.tools import MemoryToolDefinitions
from memory_mcp.domains.manager import MemoryDomainManager
from memory_mcp.autocode.server import AutoCodeServerExtension
from memory_mcp.autocode.hooks import AutoCodeHooks
from memory_mcp.autocode.hook_manager import HookManager, HookRegistry


class MemoryMcpServer:
    """
    MCP server implementation for the memory system.
    
    This class sets up an MCP server that exposes memory-related tools
    and handles MCP protocol communication with Claude Desktop.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Memory MCP Server.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.domain_manager = MemoryDomainManager(config)
        self.app = FastMCP("mcp-persistent-memory-server")
        self.tool_definitions = MemoryToolDefinitions(self.domain_manager)
        
        # Initialize AutoCode extensions
        self.autocode_hooks = None
        self.autocode_server = None
        self.hook_manager = None
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register memory-related tools with the MCP server."""
        
        @self.app.tool()
        async def store_memory(
            memory_type: str,
            content: str,
            importance: float = 0.5,
            metadata: Optional[Dict[str, Any]] = None,
            context: Optional[Dict[str, Any]] = None
        ) -> str:
            """Store new information in memory."""
            try:
                memory_id = await self.domain_manager.store_memory(
                    memory_type=memory_type,
                    content=content,
                    importance=importance,
                    metadata=metadata or {},
                    context=context or {}
                )
                
                return json.dumps({
                    "success": True,
                    "memory_id": memory_id
                })
            except Exception as e:
                logger.error(f"Error in store_memory: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def retrieve_memory(
            query: str,
            limit: int = 5,
            types: Optional[List[str]] = None,
            min_similarity: float = 0.6,
            include_metadata: bool = False
        ) -> str:
            """Retrieve relevant memories based on query."""
            try:
                memories = await self.domain_manager.retrieve_memories(
                    query=query,
                    limit=limit,
                    memory_types=types,
                    min_similarity=min_similarity,
                    include_metadata=include_metadata
                )
                
                return json.dumps({
                    "success": True,
                    "memories": memories
                })
            except Exception as e:
                logger.error(f"Error in retrieve_memory: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def list_memories(
            types: Optional[List[str]] = None,
            limit: int = 20,
            offset: int = 0,
            tier: Optional[str] = None,
            include_content: bool = False
        ) -> str:
            """List available memories with filtering options."""
            try:
                memories = await self.domain_manager.list_memories(
                    memory_types=types,
                    limit=limit,
                    offset=offset,
                    tier=tier,
                    include_content=include_content
                )
                
                return json.dumps({
                    "success": True,
                    "memories": memories
                })
            except Exception as e:
                logger.error(f"Error in list_memories: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def update_memory(
            memory_id: str,
            updates: Dict[str, Any]
        ) -> str:
            """Update existing memory entries."""
            try:
                success = await self.domain_manager.update_memory(
                    memory_id=memory_id,
                    updates=updates
                )
                
                return json.dumps({
                    "success": success
                })
            except Exception as e:
                logger.error(f"Error in update_memory: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def delete_memory(memory_ids: List[str]) -> str:
            """Remove specific memories."""
            try:
                success = await self.domain_manager.delete_memories(
                    memory_ids=memory_ids
                )
                
                return json.dumps({
                    "success": success
                })
            except Exception as e:
                logger.error(f"Error in delete_memory: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def memory_stats() -> str:
            """Get statistics about the memory store."""
            try:
                stats = await self.domain_manager.get_memory_stats()
                
                return json.dumps({
                    "success": True,
                    "stats": stats
                })
            except Exception as e:
                logger.error(f"Error in memory_stats: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        # Register AutoCode tools if enabled
        if self.config.get("autocode", {}).get("enabled", True):
            self._register_autocode_tools()
    
    def _register_autocode_tools(self) -> None:
        """Register AutoCode intelligence tools with the MCP server."""
        logger.info("Registering AutoCode tools")
        
        @self.app.tool()
        async def suggest_command(
            intent: str,
            context: Optional[Dict[str, Any]] = None
        ) -> str:
            """Get intelligent command suggestions based on intent and context."""
            try:
                if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                    return json.dumps({
                        "success": False,
                        "error": "AutoCode domain not available"
                    })
                
                suggestions = await self.domain_manager.autocode_domain.suggest_command(
                    intent=intent,
                    context=context or {}
                )
                
                return json.dumps({
                    "success": True,
                    "intent": intent,
                    "suggestions": suggestions,
                    "context": context,
                    "total_suggestions": len(suggestions)
                })
            except Exception as e:
                logger.error(f"Error in suggest_command: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def get_project_patterns(
            project_path: str,
            pattern_types: Optional[List[str]] = None
        ) -> str:
            """Get detected patterns for a project."""
            try:
                if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                    return json.dumps({
                        "success": False,
                        "error": "AutoCode domain not available"
                    })
                
                patterns = await self.domain_manager.autocode_domain.get_project_patterns(project_path)
                
                # Filter by pattern types if specified
                if pattern_types:
                    filtered_patterns = {
                        k: v for k, v in patterns.items() 
                        if k in pattern_types
                    }
                    patterns = filtered_patterns
                
                return json.dumps({
                    "success": True,
                    "project_path": project_path,
                    "patterns": patterns,
                    "pattern_types_requested": pattern_types,
                    "total_patterns": len(patterns)
                })
            except Exception as e:
                logger.error(f"Error in get_project_patterns: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def find_similar_sessions(
            query: str,
            context: Optional[Dict[str, Any]] = None,
            time_range_days: Optional[int] = None
        ) -> str:
            """Find sessions similar to current context."""
            try:
                if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                    return json.dumps({
                        "success": False,
                        "error": "AutoCode domain not available"
                    })
                
                sessions = await self.domain_manager.autocode_domain.find_similar_sessions(
                    query=query,
                    context=context,
                    time_range_days=time_range_days
                )
                
                return json.dumps({
                    "success": True,
                    "query": query,
                    "context": context,
                    "sessions": sessions,
                    "total_found": len(sessions),
                    "time_range_days": time_range_days
                })
            except Exception as e:
                logger.error(f"Error in find_similar_sessions: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def get_continuation_context(
            current_task: str,
            project_context: Optional[Dict[str, Any]] = None
        ) -> str:
            """Get relevant context for continuing work on a task."""
            try:
                if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                    return json.dumps({
                        "success": False,
                        "error": "AutoCode domain not available"
                    })
                
                context = await self.domain_manager.autocode_domain.get_context_for_continuation(
                    current_task=current_task,
                    project_context=project_context
                )
                
                return json.dumps({
                    "success": True,
                    "current_task": current_task,
                    "project_context": project_context,
                    "continuation_context": context
                })
            except Exception as e:
                logger.error(f"Error in get_continuation_context: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def suggest_workflow_optimizations(
            current_workflow: List[str],
            session_context: Optional[Dict[str, Any]] = None
        ) -> str:
            """Suggest workflow optimizations based on historical data."""
            try:
                if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                    return json.dumps({
                        "success": False,
                        "error": "AutoCode domain not available"
                    })
                
                optimizations = await self.domain_manager.autocode_domain.suggest_workflow_optimizations(
                    current_workflow=current_workflow,
                    session_context=session_context
                )
                
                return json.dumps({
                    "success": True,
                    "current_workflow": current_workflow,
                    "session_context": session_context,
                    "optimizations": optimizations,
                    "total_suggestions": len(optimizations)
                })
            except Exception as e:
                logger.error(f"Error in suggest_workflow_optimizations: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def get_learning_progression(
            topic: str,
            time_range_days: int = 180
        ) -> str:
            """Track learning progression on a specific topic."""
            try:
                if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                    return json.dumps({
                        "success": False,
                        "error": "AutoCode domain not available"
                    })
                
                progression = await self.domain_manager.autocode_domain.get_learning_progression(
                    topic=topic,
                    time_range_days=time_range_days
                )
                
                return json.dumps({
                    "success": True,
                    "topic": topic,
                    "time_range_days": time_range_days,
                    "progression": progression
                })
            except Exception as e:
                logger.error(f"Error in get_learning_progression: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def autocode_stats() -> str:
            """Get AutoCode domain statistics."""
            try:
                if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                    return json.dumps({
                        "success": False,
                        "error": "AutoCode domain not available"
                    })
                
                stats = await self.domain_manager.autocode_domain.get_stats()
                
                return json.dumps({
                    "success": True,
                    "stats": stats
                })
            except Exception as e:
                logger.error(f"Error in autocode_stats: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        logger.info("AutoCode tools registered successfully")
    
    async def _trigger_tool_hooks(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any], 
        result: Any = None,
        execution_time: float = None
    ) -> None:
        """Helper method to trigger tool hooks."""
        try:
            if self.hook_manager:
                await self.hook_manager.execute_tool_hooks(
                    tool_name, arguments, result, execution_time
                )
        except Exception as e:
            logger.error(f"Error triggering hooks for {tool_name}: {e}")
    
    async def start(self) -> None:
        """Start the MCP server."""
        # Initialize the memory domain manager
        await self.domain_manager.initialize()
        
        # Initialize AutoCode hooks if enabled
        if self.config.get("autocode", {}).get("enabled", True):
            self.autocode_hooks = AutoCodeHooks(self.domain_manager)
            self.autocode_server = AutoCodeServerExtension(self.domain_manager, self.autocode_hooks)
            
            # Initialize hook manager
            self.hook_manager = HookManager(self.domain_manager, self.autocode_hooks)
            HookRegistry.register_manager(self.hook_manager)
            
            logger.info("AutoCode hooks, server extensions, and hook manager initialized")
        
        logger.info("Starting Memory MCP Server using stdio transport")
        
        # Start the server using FastMCP's run method
        self.app.run()
