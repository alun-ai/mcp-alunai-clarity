"""
Integration utilities for seamless AutoCode domain integration.

This module provides utilities and decorators for integrating AutoCode
functionality seamlessly with existing MCP operations.
"""

import time
import functools
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from loguru import logger

from .hook_manager import HookRegistry


def with_autocode_hooks(tool_name: str):
    """
    Decorator to automatically trigger AutoCode hooks for MCP tools.
    
    Args:
        tool_name: Name of the MCP tool
    
    Returns:
        Decorated function that triggers hooks
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Extract arguments for hook context
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                arguments = dict(bound_args.arguments)
                
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Trigger hooks
                await HookRegistry.trigger_tool_hooks(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    execution_time=execution_time
                )
                
                return result
                
            except Exception as e:
                # Calculate execution time even for failures
                execution_time = time.time() - start_time
                
                # Trigger hooks with error information
                arguments = {"error": str(e)}
                await HookRegistry.trigger_tool_hooks(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    execution_time=execution_time
                )
                
                raise
        
        return wrapper
    return decorator


class ToolRouter:
    """
    Routes tool calls to appropriate handlers with AutoCode integration.
    """
    
    def __init__(self, domain_manager):
        """
        Initialize the tool router.
        
        Args:
            domain_manager: Reference to the memory domain manager
        """
        self.domain_manager = domain_manager
        self.tool_handlers = {}
        self.middleware = []
        
        # Register default tool routes
        self._register_default_routes()
    
    def _register_default_routes(self) -> None:
        """Register default tool routing."""
        # Memory tools
        self.register_tool("store_memory", self._route_store_memory)
        self.register_tool("retrieve_memory", self._route_retrieve_memory)
        self.register_tool("list_memories", self._route_list_memories)
        self.register_tool("update_memory", self._route_update_memory)
        self.register_tool("delete_memory", self._route_delete_memory)
        self.register_tool("memory_stats", self._route_memory_stats)
        
        # AutoCode tools
        self.register_tool("suggest_command", self._route_suggest_command)
        self.register_tool("get_project_patterns", self._route_get_project_patterns)
        self.register_tool("find_similar_sessions", self._route_find_similar_sessions)
        self.register_tool("get_continuation_context", self._route_get_continuation_context)
        self.register_tool("suggest_workflow_optimizations", self._route_suggest_workflow_optimizations)
        self.register_tool("get_learning_progression", self._route_get_learning_progression)
        self.register_tool("autocode_stats", self._route_autocode_stats)
    
    def register_tool(self, tool_name: str, handler: Callable) -> None:
        """
        Register a tool handler.
        
        Args:
            tool_name: Name of the tool
            handler: Handler function
        """
        self.tool_handlers[tool_name] = handler
        logger.debug(f"Registered tool handler: {tool_name}")
    
    def add_middleware(self, middleware_func: Callable) -> None:
        """
        Add middleware function to be executed before tool handlers.
        
        Args:
            middleware_func: Middleware function
        """
        self.middleware.append(middleware_func)
    
    async def route_tool_call(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route a tool call to the appropriate handler.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            # Execute middleware
            for middleware in self.middleware:
                arguments = await self._execute_middleware(middleware, tool_name, arguments)
            
            # Get handler
            if tool_name not in self.tool_handlers:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
            
            handler = self.tool_handlers[tool_name]
            
            # Execute handler
            start_time = time.time()
            result = await handler(arguments)
            execution_time = time.time() - start_time
            
            # Trigger hooks
            await HookRegistry.trigger_tool_hooks(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing tool call {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_middleware(
        self, 
        middleware: Callable, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute middleware and return potentially modified arguments."""
        try:
            return await middleware(tool_name, arguments)
        except Exception as e:
            logger.error(f"Error in middleware: {e}")
            return arguments
    
    # Default route handlers
    async def _route_store_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route store_memory tool call."""
        try:
            memory_id = await self.domain_manager.store_memory(
                memory_type=arguments.get("memory_type"),
                content=arguments.get("content"),
                importance=arguments.get("importance", 0.5),
                metadata=arguments.get("metadata", {}),
                context=arguments.get("context", {})
            )
            
            return {
                "success": True,
                "memory_id": memory_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_retrieve_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route retrieve_memory tool call."""
        try:
            memories = await self.domain_manager.retrieve_memories(
                query=arguments.get("query"),
                limit=arguments.get("limit", 5),
                memory_types=arguments.get("types"),
                min_similarity=arguments.get("min_similarity", 0.6),
                include_metadata=arguments.get("include_metadata", False)
            )
            
            return {
                "success": True,
                "memories": memories
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_list_memories(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route list_memories tool call."""
        try:
            memories = await self.domain_manager.list_memories(
                memory_types=arguments.get("types"),
                limit=arguments.get("limit", 20),
                offset=arguments.get("offset", 0),
                tier=arguments.get("tier"),
                include_content=arguments.get("include_content", False)
            )
            
            return {
                "success": True,
                "memories": memories
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_update_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route update_memory tool call."""
        try:
            success = await self.domain_manager.update_memory(
                memory_id=arguments.get("memory_id"),
                updates=arguments.get("updates")
            )
            
            return {
                "success": success
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_delete_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route delete_memory tool call."""
        try:
            success = await self.domain_manager.delete_memories(
                memory_ids=arguments.get("memory_ids")
            )
            
            return {
                "success": success
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_memory_stats(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route memory_stats tool call."""
        try:
            stats = await self.domain_manager.get_memory_stats()
            
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # AutoCode route handlers
    async def _route_suggest_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route suggest_command tool call."""
        try:
            if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                return {
                    "success": False,
                    "error": "AutoCode domain not available"
                }
            
            suggestions = await self.domain_manager.autocode_domain.suggest_command(
                intent=arguments.get("intent"),
                context=arguments.get("context", {})
            )
            
            return {
                "success": True,
                "intent": arguments.get("intent"),
                "suggestions": suggestions,
                "context": arguments.get("context"),
                "total_suggestions": len(suggestions)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_get_project_patterns(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route get_project_patterns tool call."""
        try:
            if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                return {
                    "success": False,
                    "error": "AutoCode domain not available"
                }
            
            patterns = await self.domain_manager.autocode_domain.get_project_patterns(
                arguments.get("project_path")
            )
            
            # Filter by pattern types if specified
            pattern_types = arguments.get("pattern_types")
            if pattern_types:
                filtered_patterns = {
                    k: v for k, v in patterns.items() 
                    if k in pattern_types
                }
                patterns = filtered_patterns
            
            return {
                "success": True,
                "project_path": arguments.get("project_path"),
                "patterns": patterns,
                "pattern_types_requested": pattern_types,
                "total_patterns": len(patterns)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_find_similar_sessions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route find_similar_sessions tool call."""
        try:
            if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                return {
                    "success": False,
                    "error": "AutoCode domain not available"
                }
            
            sessions = await self.domain_manager.autocode_domain.find_similar_sessions(
                query=arguments.get("query"),
                context=arguments.get("context"),
                time_range_days=arguments.get("time_range_days")
            )
            
            return {
                "success": True,
                "query": arguments.get("query"),
                "context": arguments.get("context"),
                "sessions": sessions,
                "total_found": len(sessions),
                "time_range_days": arguments.get("time_range_days")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_get_continuation_context(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route get_continuation_context tool call."""
        try:
            if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                return {
                    "success": False,
                    "error": "AutoCode domain not available"
                }
            
            context = await self.domain_manager.autocode_domain.get_context_for_continuation(
                current_task=arguments.get("current_task"),
                project_context=arguments.get("project_context")
            )
            
            return {
                "success": True,
                "current_task": arguments.get("current_task"),
                "project_context": arguments.get("project_context"),
                "continuation_context": context
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_suggest_workflow_optimizations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route suggest_workflow_optimizations tool call."""
        try:
            if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                return {
                    "success": False,
                    "error": "AutoCode domain not available"
                }
            
            optimizations = await self.domain_manager.autocode_domain.suggest_workflow_optimizations(
                current_workflow=arguments.get("current_workflow"),
                session_context=arguments.get("session_context")
            )
            
            return {
                "success": True,
                "current_workflow": arguments.get("current_workflow"),
                "session_context": arguments.get("session_context"),
                "optimizations": optimizations,
                "total_suggestions": len(optimizations)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_get_learning_progression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route get_learning_progression tool call."""
        try:
            if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                return {
                    "success": False,
                    "error": "AutoCode domain not available"
                }
            
            progression = await self.domain_manager.autocode_domain.get_learning_progression(
                topic=arguments.get("topic"),
                time_range_days=arguments.get("time_range_days", 180)
            )
            
            return {
                "success": True,
                "topic": arguments.get("topic"),
                "time_range_days": arguments.get("time_range_days", 180),
                "progression": progression
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _route_autocode_stats(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route autocode_stats tool call."""
        try:
            if not hasattr(self.domain_manager, 'autocode_domain') or not self.domain_manager.autocode_domain:
                return {
                    "success": False,
                    "error": "AutoCode domain not available"
                }
            
            stats = await self.domain_manager.autocode_domain.get_stats()
            
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class AutoCodeMiddleware:
    """
    Middleware for AutoCode-specific processing.
    """
    
    def __init__(self, domain_manager):
        """
        Initialize AutoCode middleware.
        
        Args:
            domain_manager: Reference to the memory domain manager
        """
        self.domain_manager = domain_manager
    
    async def context_enrichment_middleware(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Middleware to enrich context for AutoCode tools.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            
        Returns:
            Enhanced arguments with additional context
        """
        try:
            # Add automatic context enrichment for relevant tools
            autocode_tools = [
                "suggest_command", "get_project_patterns", "find_similar_sessions",
                "get_continuation_context", "suggest_workflow_optimizations", 
                "get_learning_progression"
            ]
            
            if tool_name in autocode_tools:
                # Add timestamp
                if "context" not in arguments:
                    arguments["context"] = {}
                
                arguments["context"]["request_timestamp"] = datetime.utcnow().isoformat()
                arguments["context"]["tool_name"] = tool_name
                
                # Add platform information
                import platform
                arguments["context"]["platform"] = platform.system().lower()
                
                logger.debug(f"AutoCode middleware: Enhanced context for {tool_name}")
            
            return arguments
            
        except Exception as e:
            logger.error(f"Error in context enrichment middleware: {e}")
            return arguments
    
    async def logging_middleware(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Middleware for enhanced logging of AutoCode operations.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            
        Returns:
            Unchanged arguments (logging only)
        """
        try:
            # Log AutoCode tool usage for analytics
            autocode_tools = [
                "suggest_command", "get_project_patterns", "find_similar_sessions",
                "get_continuation_context", "suggest_workflow_optimizations", 
                "get_learning_progression", "autocode_stats"
            ]
            
            if tool_name in autocode_tools:
                logger.info(f"AutoCode tool called: {tool_name}")
                
                # Log key parameters without sensitive data
                safe_args = {}
                for key, value in arguments.items():
                    if key in ["intent", "query", "topic", "project_path"]:
                        safe_args[key] = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                
                logger.debug(f"AutoCode tool parameters: {safe_args}")
            
            return arguments
            
        except Exception as e:
            logger.error(f"Error in logging middleware: {e}")
            return arguments