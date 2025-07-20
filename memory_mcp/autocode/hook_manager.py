"""
Hook Manager for AutoCode domain.

This module provides automatic hook registration and management for
seamless integration with MCP operations and Claude workflows.
"""

import inspect
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from loguru import logger


class HookManager:
    """
    Manages automatic hook registration and execution for AutoCode domain.
    
    This class provides a centralized system for registering and executing
    hooks that trigger during various MCP operations and Claude interactions.
    """
    
    def __init__(self, domain_manager, autocode_hooks):
        """
        Initialize the Hook Manager.
        
        Args:
            domain_manager: Reference to the memory domain manager
            autocode_hooks: AutoCode hooks instance
        """
        self.domain_manager = domain_manager
        self.autocode_hooks = autocode_hooks
        
        # Hook registries
        self.tool_hooks = {}  # Tool execution hooks
        self.lifecycle_hooks = {}  # Session lifecycle hooks
        self.event_hooks = {}  # Event-based hooks
        
        # Hook execution statistics
        self.hook_stats = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "last_execution": None
        }
        
        # Auto-register default hooks
        self._register_default_hooks()
    
    def _register_default_hooks(self) -> None:
        """Register default AutoCode hooks automatically."""
        try:
            # Tool execution hooks
            self.register_tool_hook("store_memory", self._on_memory_store)
            self.register_tool_hook("retrieve_memory", self._on_memory_retrieve)
            self.register_tool_hook("suggest_command", self._on_command_suggest)
            self.register_tool_hook("get_project_patterns", self._on_pattern_request)
            
            # Proactive memory consultation hooks
            self.register_tool_hook("suggest_memory_queries", self._on_memory_query_suggest)
            self.register_tool_hook("check_relevant_memories", self._on_relevant_memory_check)
            
            # Lifecycle hooks
            self.register_lifecycle_hook("session_start", self._on_session_start)
            self.register_lifecycle_hook("session_end", self._on_session_end)
            self.register_lifecycle_hook("conversation_message", self._on_conversation_message)
            
            # Event hooks for automatic triggering
            self.register_event_hook("file_read", self._on_file_access)
            self.register_event_hook("bash_execution", self._on_bash_execution)
            self.register_event_hook("project_detection", self._on_project_detection)
            
            # Proactive memory event hooks
            self.register_event_hook("tool_pre_execution", self._on_tool_pre_execution)
            self.register_event_hook("context_change", self._on_context_change)
            
            logger.info("Default AutoCode hooks registered successfully")
            
        except Exception as e:
            logger.error(f"Error registering default hooks: {e}")
    
    def register_tool_hook(self, tool_name: str, hook_func: Callable) -> None:
        """
        Register a hook for a specific tool execution.
        
        Args:
            tool_name: Name of the MCP tool
            hook_func: Function to call when tool is executed
        """
        if tool_name not in self.tool_hooks:
            self.tool_hooks[tool_name] = []
        
        self.tool_hooks[tool_name].append(hook_func)
        logger.debug(f"Registered tool hook for {tool_name}")
    
    def register_lifecycle_hook(self, event: str, hook_func: Callable) -> None:
        """
        Register a hook for session lifecycle events.
        
        Args:
            event: Lifecycle event name
            hook_func: Function to call when event occurs
        """
        if event not in self.lifecycle_hooks:
            self.lifecycle_hooks[event] = []
        
        self.lifecycle_hooks[event].append(hook_func)
        logger.debug(f"Registered lifecycle hook for {event}")
    
    def register_event_hook(self, event: str, hook_func: Callable) -> None:
        """
        Register a hook for general events.
        
        Args:
            event: Event name
            hook_func: Function to call when event occurs
        """
        if event not in self.event_hooks:
            self.event_hooks[event] = []
        
        self.event_hooks[event].append(hook_func)
        logger.debug(f"Registered event hook for {event}")
    
    async def execute_tool_hooks(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        result: Any = None,
        execution_time: float = None
    ) -> None:
        """
        Execute hooks for a specific tool.
        
        Args:
            tool_name: Name of the executed tool
            arguments: Tool arguments
            result: Tool execution result
            execution_time: Time taken to execute tool
        """
        if tool_name not in self.tool_hooks:
            return
        
        hook_context = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "execution_time": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for hook_func in self.tool_hooks[tool_name]:
            try:
                await self._execute_hook(hook_func, hook_context)
            except Exception as e:
                logger.error(f"Error executing tool hook {hook_func.__name__} for {tool_name}: {e}")
    
    async def execute_lifecycle_hooks(
        self, 
        event: str, 
        context: Dict[str, Any] = None
    ) -> None:
        """
        Execute hooks for a lifecycle event.
        
        Args:
            event: Lifecycle event name
            context: Event context data
        """
        if event not in self.lifecycle_hooks:
            return
        
        hook_context = {
            "event": event,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for hook_func in self.lifecycle_hooks[event]:
            try:
                await self._execute_hook(hook_func, hook_context)
            except Exception as e:
                logger.error(f"Error executing lifecycle hook {hook_func.__name__} for {event}: {e}")
    
    async def execute_event_hooks(
        self, 
        event: str, 
        event_data: Dict[str, Any] = None
    ) -> None:
        """
        Execute hooks for a general event.
        
        Args:
            event: Event name
            event_data: Event-specific data
        """
        if event not in self.event_hooks:
            return
        
        hook_context = {
            "event": event,
            "data": event_data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for hook_func in self.event_hooks[event]:
            try:
                await self._execute_hook(hook_func, hook_context)
            except Exception as e:
                logger.error(f"Error executing event hook {hook_func.__name__} for {event}: {e}")
    
    async def _execute_hook(self, hook_func: Callable, context: Dict[str, Any]) -> None:
        """Execute a single hook function with proper error handling."""
        try:
            self.hook_stats["executions"] += 1
            self.hook_stats["last_execution"] = datetime.utcnow().isoformat()
            
            # Check if hook function is async
            if inspect.iscoroutinefunction(hook_func):
                await hook_func(context)
            else:
                hook_func(context)
            
            self.hook_stats["successes"] += 1
            
        except Exception as e:
            self.hook_stats["failures"] += 1
            logger.error(f"Hook execution failed: {hook_func.__name__}: {e}")
            raise
    
    # Default hook implementations
    async def _on_memory_store(self, context: Dict[str, Any]) -> None:
        """Hook for memory storage operations."""
        try:
            arguments = context.get("arguments", {})
            memory_type = arguments.get("memory_type", "")
            content = arguments.get("content", "")
            
            # Check if this is code-related content
            if memory_type in ["code", "project_pattern", "session_summary"]:
                logger.debug(f"AutoCode: Processing {memory_type} memory storage")
                
                # Extract additional context for AutoCode processing
                if memory_type == "code" and content:
                    # This could trigger pattern detection
                    await self.autocode_hooks.on_file_read(
                        file_path=arguments.get("metadata", {}).get("file_path", "unknown"),
                        content=str(content),
                        operation="store"
                    )
                    
        except Exception as e:
            logger.error(f"Error in memory store hook: {e}")
    
    async def _on_memory_retrieve(self, context: Dict[str, Any]) -> None:
        """Hook for memory retrieval operations."""
        try:
            arguments = context.get("arguments", {})
            query = arguments.get("query", "")
            result = context.get("result", {})
            
            # Log retrieval patterns for learning
            if isinstance(result, dict) and result.get("success") and query:
                # Track what types of queries are being made
                logger.debug(f"AutoCode: Memory retrieval query: {query[:100]}...")
                
        except Exception as e:
            logger.error(f"Error in memory retrieve hook: {e}")
    
    async def _on_command_suggest(self, context: Dict[str, Any]) -> None:
        """Hook for command suggestion operations."""
        try:
            arguments = context.get("arguments", {})
            intent = arguments.get("intent", "")
            suggestions = context.get("result", {}).get("suggestions", [])
            
            # Log command suggestion patterns
            if intent and suggestions:
                logger.debug(f"AutoCode: Command suggestions for intent '{intent}': {len(suggestions)} suggestions")
                
        except Exception as e:
            logger.error(f"Error in command suggest hook: {e}")
    
    async def _on_pattern_request(self, context: Dict[str, Any]) -> None:
        """Hook for project pattern requests."""
        try:
            arguments = context.get("arguments", {})
            project_path = arguments.get("project_path", "")
            patterns = context.get("result", {}).get("patterns", {})
            
            # Log pattern usage
            if project_path and patterns:
                logger.debug(f"AutoCode: Pattern request for {project_path}: {len(patterns)} pattern types")
                
        except Exception as e:
            logger.error(f"Error in pattern request hook: {e}")
    
    async def _on_session_start(self, context: Dict[str, Any]) -> None:
        """Hook for session start events."""
        try:
            session_context = context.get("context", {})
            
            # Initialize session tracking
            if self.autocode_hooks:
                # Reset session data
                self.autocode_hooks.session_data = {
                    "files_accessed": [],
                    "commands_executed": [],
                    "start_time": datetime.utcnow(),
                    "conversation_log": []
                }
                
            logger.info("AutoCode: Session started, tracking initialized")
            
        except Exception as e:
            logger.error(f"Error in session start hook: {e}")
    
    async def _on_session_end(self, context: Dict[str, Any]) -> None:
        """Hook for session end events."""
        try:
            session_context = context.get("context", {})
            conversation_id = session_context.get("conversation_id")
            
            # Generate session summary
            if self.autocode_hooks:
                await self.autocode_hooks.on_conversation_end(conversation_id)
                
            logger.info("AutoCode: Session ended, summary generated")
            
        except Exception as e:
            logger.error(f"Error in session end hook: {e}")
    
    async def _on_conversation_message(self, context: Dict[str, Any]) -> None:
        """Hook for conversation messages."""
        try:
            message_data = context.get("data", {})
            role = message_data.get("role", "")
            content = message_data.get("content", "")
            message_id = message_data.get("message_id")
            
            # Track conversation messages
            if self.autocode_hooks and content:
                await self.autocode_hooks.on_conversation_message(role, content, message_id)
                
        except Exception as e:
            logger.error(f"Error in conversation message hook: {e}")
    
    async def _on_file_access(self, context: Dict[str, Any]) -> None:
        """Hook for file access events."""
        try:
            file_data = context.get("data", {})
            file_path = file_data.get("file_path", "")
            content = file_data.get("content", "")
            operation = file_data.get("operation", "read")
            
            # Process file access
            if self.autocode_hooks and file_path:
                await self.autocode_hooks.on_file_read(file_path, content, operation)
            
            # Proactive memory consultation for file access
            if file_path and operation == "read":
                await self._suggest_file_related_memories(file_path)
                
        except Exception as e:
            logger.error(f"Error in file access hook: {e}")
    
    async def _on_bash_execution(self, context: Dict[str, Any]) -> None:
        """Hook for bash execution events."""
        try:
            bash_data = context.get("data", {})
            command = bash_data.get("command", "")
            exit_code = bash_data.get("exit_code", 0)
            output = bash_data.get("output", "")
            working_directory = bash_data.get("working_directory")
            
            # Process bash execution
            if self.autocode_hooks and command:
                await self.autocode_hooks.on_bash_execution(
                    command, exit_code, output, working_directory
                )
                
        except Exception as e:
            logger.error(f"Error in bash execution hook: {e}")
    
    async def _on_project_detection(self, context: Dict[str, Any]) -> None:
        """Hook for project detection events."""
        try:
            project_data = context.get("data", {})
            project_root = project_data.get("project_root", "")
            
            # Process project detection
            if self.autocode_hooks and project_root:
                await self.autocode_hooks.on_project_detection(project_root)
                
        except Exception as e:
            logger.error(f"Error in project detection hook: {e}")
    
    def get_hook_stats(self) -> Dict[str, Any]:
        """Get hook execution statistics."""
        return {
            **self.hook_stats,
            "registered_hooks": {
                "tool_hooks": len(self.tool_hooks),
                "lifecycle_hooks": len(self.lifecycle_hooks),
                "event_hooks": len(self.event_hooks)
            }
        }
    
    def list_registered_hooks(self) -> Dict[str, List[str]]:
        """List all registered hooks by category."""
        return {
            "tool_hooks": list(self.tool_hooks.keys()),
            "lifecycle_hooks": list(self.lifecycle_hooks.keys()),
            "event_hooks": list(self.event_hooks.keys())
        }
    
    # Proactive memory consultation hook implementations
    async def _on_memory_query_suggest(self, context: Dict[str, Any]) -> None:
        """Hook for memory query suggestion operations."""
        try:
            arguments = context.get("arguments", {})
            current_context = arguments.get("current_context", {})
            suggestions = context.get("result", {}).get("suggestions", [])
            
            # Log query suggestion patterns for learning
            if current_context and suggestions:
                logger.debug(f"AutoCode: Memory query suggestions generated: {len(suggestions)} suggestions")
                
        except Exception as e:
            logger.error(f"Error in memory query suggest hook: {e}")
    
    async def _on_relevant_memory_check(self, context: Dict[str, Any]) -> None:
        """Hook for relevant memory check operations."""
        try:
            arguments = context.get("arguments", {})
            context_data = arguments.get("context", {})
            memories = context.get("result", {}).get("memories", [])
            
            # Log relevant memory checks for pattern learning
            if context_data and memories:
                logger.debug(f"AutoCode: Relevant memories found: {len(memories)} memories")
                
        except Exception as e:
            logger.error(f"Error in relevant memory check hook: {e}")
    
    async def _on_tool_pre_execution(self, context: Dict[str, Any]) -> None:
        """Hook for pre-tool execution memory consultation."""
        try:
            tool_data = context.get("data", {})
            tool_name = tool_data.get("tool_name", "")
            arguments = tool_data.get("arguments", {})
            
            # Suggest relevant memories before tool execution
            if tool_name and self._should_consult_memory_for_tool(tool_name):
                await self._suggest_contextual_memories(tool_name, arguments)
                
        except Exception as e:
            logger.error(f"Error in tool pre-execution hook: {e}")
    
    async def _on_context_change(self, context: Dict[str, Any]) -> None:
        """Hook for context change events that might require memory consultation."""
        try:
            change_data = context.get("data", {})
            change_type = change_data.get("type", "")
            new_context = change_data.get("context", {})
            
            # Trigger memory consultation for significant context changes
            if change_type in ["project_switch", "directory_change", "task_switch"]:
                await self._suggest_context_memories(change_type, new_context)
                
        except Exception as e:
            logger.error(f"Error in context change hook: {e}")
    
    # Helper methods for proactive memory consultation
    async def _suggest_file_related_memories(self, file_path: str) -> None:
        """Suggest memories related to the accessed file."""
        try:
            if not self.domain_manager:
                return
            
            # Extract keywords from file path for memory search
            keywords = self._extract_file_keywords(file_path)
            if not keywords:
                return
            
            # Search for file-related memories
            query = f"file {file_path} {' '.join(keywords)}"
            memories = await self.domain_manager.retrieve_memories(
                query=query,
                limit=3,
                memory_types=["code", "project_pattern", "session_summary"],
                min_similarity=0.7
            )
            
            if memories:
                logger.info(f"AutoCode: Found {len(memories)} file-related memories for {file_path}")
                # Could trigger a proactive memory presentation here
                
        except Exception as e:
            logger.error(f"Error suggesting file-related memories: {e}")
    
    async def _suggest_contextual_memories(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Suggest memories based on tool context."""
        try:
            if not self.domain_manager:
                return
            
            # Generate context-aware query based on tool and arguments
            query = self._generate_contextual_query(tool_name, arguments)
            if not query:
                return
            
            # Search for contextually relevant memories
            memories = await self.domain_manager.retrieve_memories(
                query=query,
                limit=5,
                min_similarity=0.6
            )
            
            if memories:
                logger.info(f"AutoCode: Found {len(memories)} contextual memories for {tool_name}")
                
        except Exception as e:
            logger.error(f"Error suggesting contextual memories: {e}")
    
    async def _suggest_context_memories(self, change_type: str, context: Dict[str, Any]) -> None:
        """Suggest memories for context changes."""
        try:
            if not self.domain_manager:
                return
            
            # Generate query based on context change
            query_parts = [change_type]
            if "project_path" in context:
                query_parts.append(f"project {context['project_path']}")
            if "directory" in context:
                query_parts.append(f"directory {context['directory']}")
            if "task" in context:
                query_parts.append(f"task {context['task']}")
            
            query = " ".join(query_parts)
            
            # Search for context-change memories
            memories = await self.domain_manager.retrieve_memories(
                query=query,
                limit=5,
                memory_types=["session_summary", "project_pattern", "reflection"],
                min_similarity=0.6
            )
            
            if memories:
                logger.info(f"AutoCode: Found {len(memories)} context-change memories for {change_type}")
                
        except Exception as e:
            logger.error(f"Error suggesting context memories: {e}")
    
    def _extract_file_keywords(self, file_path: str) -> List[str]:
        """Extract meaningful keywords from file path."""
        import os
        from pathlib import Path
        
        path = Path(file_path)
        keywords = []
        
        # Add filename without extension
        if path.stem:
            keywords.append(path.stem)
        
        # Add file extension
        if path.suffix:
            keywords.append(path.suffix[1:])  # Remove dot
        
        # Add parent directory names (last 2 levels)
        parent_parts = path.parent.parts[-2:] if len(path.parent.parts) >= 2 else path.parent.parts
        keywords.extend(parent_parts)
        
        return [k for k in keywords if k and len(k) > 1]
    
    def _generate_contextual_query(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate context-aware query for memory search."""
        query_parts = [tool_name]
        
        # Add relevant argument values as query terms
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) < 50:
                # Add short string values
                query_parts.append(value)
            elif key in ["project_path", "file_path", "command", "intent"]:
                # Add specific important keys
                query_parts.append(str(value))
        
        return " ".join(query_parts[:5])  # Limit query length
    
    def _should_consult_memory_for_tool(self, tool_name: str) -> bool:
        """Determine if memory consultation is beneficial for this tool."""
        consultation_tools = {
            "suggest_command",
            "get_project_patterns", 
            "find_similar_sessions",
            "get_continuation_context",
            "suggest_workflow_optimizations"
        }
        return tool_name in consultation_tools


class HookRegistry:
    """
    Global registry for hook management across the application.
    
    This class provides a singleton pattern for accessing the hook manager
    from anywhere in the application.
    """
    
    _instance = None
    _hook_manager = None
    
    @classmethod
    def get_instance(cls) -> Optional['HookManager']:
        """Get the global hook manager instance."""
        return cls._hook_manager
    
    @classmethod
    def register_manager(cls, hook_manager: HookManager) -> None:
        """Register the hook manager globally."""
        cls._hook_manager = hook_manager
        logger.info("Hook manager registered globally")
    
    @classmethod
    async def trigger_tool_hooks(
        cls, 
        tool_name: str, 
        arguments: Dict[str, Any],
        result: Any = None,
        execution_time: float = None
    ) -> None:
        """Global function to trigger tool hooks."""
        if cls._hook_manager:
            await cls._hook_manager.execute_tool_hooks(
                tool_name, arguments, result, execution_time
            )
    
    @classmethod
    async def trigger_lifecycle_hooks(
        cls, 
        event: str, 
        context: Dict[str, Any] = None
    ) -> None:
        """Global function to trigger lifecycle hooks."""
        if cls._hook_manager:
            await cls._hook_manager.execute_lifecycle_hooks(event, context)
    
    @classmethod
    async def trigger_event_hooks(
        cls, 
        event: str, 
        event_data: Dict[str, Any] = None
    ) -> None:
        """Global function to trigger event hooks."""
        if cls._hook_manager:
            await cls._hook_manager.execute_event_hooks(event, event_data)


# Convenience functions for triggering hooks
async def trigger_file_read_hook(file_path: str, content: str = "", operation: str = "read") -> None:
    """Trigger file read hook."""
    await HookRegistry.trigger_event_hooks("file_read", {
        "file_path": file_path,
        "content": content,
        "operation": operation
    })

async def trigger_bash_execution_hook(
    command: str, 
    exit_code: int, 
    output: str = "",
    working_directory: str = None
) -> None:
    """Trigger bash execution hook."""
    await HookRegistry.trigger_event_hooks("bash_execution", {
        "command": command,
        "exit_code": exit_code,
        "output": output,
        "working_directory": working_directory
    })

async def trigger_conversation_message_hook(
    role: str, 
    content: str, 
    message_id: str = None
) -> None:
    """Trigger conversation message hook."""
    await HookRegistry.trigger_event_hooks("conversation_message", {
        "role": role,
        "content": content,
        "message_id": message_id
    })

async def trigger_project_detection_hook(project_root: str) -> None:
    """Trigger project detection hook."""
    await HookRegistry.trigger_event_hooks("project_detection", {
        "project_root": project_root
    })

async def trigger_tool_pre_execution_hook(tool_name: str, arguments: Dict[str, Any]) -> None:
    """Trigger tool pre-execution hook for proactive memory consultation."""
    await HookRegistry.trigger_event_hooks("tool_pre_execution", {
        "tool_name": tool_name,
        "arguments": arguments
    })

async def trigger_context_change_hook(change_type: str, context: Dict[str, Any]) -> None:
    """Trigger context change hook for proactive memory consultation."""
    await HookRegistry.trigger_event_hooks("context_change", {
        "type": change_type,
        "context": context
    })