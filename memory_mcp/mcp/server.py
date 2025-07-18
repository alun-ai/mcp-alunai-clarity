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
    
    async def start(self) -> None:
        """Start the MCP server."""
        # Initialize the memory domain manager
        await self.domain_manager.initialize()
        
        logger.info("Starting Memory MCP Server using stdio transport")
        
        # Start the server using FastMCP's run method
        self.app.run()
