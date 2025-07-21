"""
Hook Manager for AutoCode domain.

This module provides automatic hook registration and management for
seamless integration with MCP operations and Claude workflows.
"""

import inspect
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from loguru import logger

from .mcp_hooks import MCPAwarenessHooks


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
        
        # Recursion guards to prevent infinite loops
        self._memory_check_in_progress = False
        
        # Initialize MCP awareness hooks
        self.mcp_awareness_hooks = MCPAwarenessHooks(domain_manager)
        
        # Initialize structured thinking extension
        self.structured_thinking_extension = None
        if domain_manager and hasattr(domain_manager, 'persistence_domain'):
            try:
                from .structured_thinking_extension import StructuredThinkingExtension
                self.structured_thinking_extension = StructuredThinkingExtension(domain_manager.persistence_domain)
                logger.info("Structured thinking extension initialized")
            except ImportError as e:
                logger.warning(f"Could not initialize structured thinking extension: {e}")
        
        # Default proactive memory configuration
        self.proactive_config = {
            "enabled": True,
            "triggers": {
                "file_access": True,
                "tool_execution": True,
                "context_change": True,
                "structured_thinking": True  # New trigger for structured thinking
            },
            "similarity_threshold": 0.6,
            "max_memories_per_trigger": 3,
            "auto_present": True
        }
        
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
        
        # Structured thinking session tracking
        self.active_thinking_sessions = {}
        
        # Auto-register default hooks
        self._register_default_hooks()
    
    async def initialize(self) -> None:
        """Initialize the hook manager and all subsystems."""
        try:
            # Initialize MCP awareness hooks
            await self.mcp_awareness_hooks.initialize()
            logger.info("Hook manager initialized successfully")
        except (ImportError, OSError, ConfigurationError, AttributeError) as e:
            logger.error(f"Error initializing hook manager: {e}")
    
    async def get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt with MCP tool awareness."""
        try:
            return await self.mcp_awareness_hooks.get_enhanced_system_prompt()
        except (AttributeError, KeyError, ValueError, OSError) as e:
            logger.error(f"Error getting enhanced system prompt: {e}")
            return ""
    
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
            
            # Structured thinking hooks
            self.register_tool_hook("process_structured_thought", self._on_structured_thought_process)
            self.register_tool_hook("generate_thinking_summary", self._on_thinking_summary_generate)
            self.register_tool_hook("continue_thinking_process", self._on_thinking_process_continue)
            
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
            
            # MCP awareness hooks
            self.register_event_hook("user_request", self._on_mcp_user_request)
            self.register_event_hook("tool_about_to_execute", self._on_mcp_tool_about_to_execute)
            self.register_event_hook("mcp_context_change", self._on_mcp_context_change)
            self.register_event_hook("error_occurred", self._on_mcp_error_occurred)
            
            logger.info("Default AutoCode hooks registered successfully")
            
        except (AttributeError, ImportError, RuntimeError) as e:
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
            except (AttributeError, RuntimeError, ValueError, KeyError) as e:
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
            except (AttributeError, RuntimeError, ValueError, ImportError) as e:
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
            except (AttributeError, RuntimeError, ValueError, KeyError) as e:
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
            
        except (AttributeError, RuntimeError, ValueError, TypeError, ImportError) as e:
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
                    
        except (MemoryOperationError, AttributeError, KeyError, ValueError) as e:
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
        """Hook for command suggestion operations with intelligent structured thinking trigger."""
        try:
            arguments = context.get("arguments", {})
            intent = arguments.get("intent", "")
            suggestions = context.get("result", {}).get("suggestions", [])
            use_structured_thinking = arguments.get("use_structured_thinking", False)
            
            # Log command suggestion patterns
            if intent and suggestions:
                logger.debug(f"AutoCode: Command suggestions for intent '{intent}': {len(suggestions)} suggestions")
            
            # Intelligent structured thinking trigger
            if intent and self._should_trigger_structured_thinking(intent, context):
                logger.info(f"AutoCode: Triggering structured thinking for complex intent: {intent[:50]}...")
                await self._auto_trigger_structured_thinking(intent, context)
                
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
        """Hook for conversation messages with intelligent structured thinking detection."""
        try:
            message_data = context.get("data", {})
            role = message_data.get("role", "")
            content = message_data.get("content", "")
            message_id = message_data.get("message_id")
            
            # Track conversation messages
            if self.autocode_hooks and content:
                await self.autocode_hooks.on_conversation_message(role, content, message_id)
            
            # Intelligent structured thinking trigger for user messages
            if role == "user" and content and self._should_trigger_structured_thinking(content, context):
                logger.info(f"AutoCode: Detected complex user message triggering structured thinking")
                await self._auto_trigger_structured_thinking(content, {
                    "arguments": {"intent": content},
                    "trigger": "conversation_message",
                    "message_id": message_id
                })
                
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
            if (file_path and operation == "read" and 
                self.proactive_config.get("enabled", True) and 
                self.proactive_config.get("triggers", {}).get("file_access", True)):
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
            if (tool_name and self._should_consult_memory_for_tool(tool_name) and
                self.proactive_config.get("enabled", True) and 
                self.proactive_config.get("triggers", {}).get("tool_execution", True)):
                await self._suggest_contextual_memories(tool_name, arguments)
                
                # For certain tools, also auto-trigger comprehensive memory check
                if tool_name in ["Edit", "Write", "Bash", "Read"]:
                    await self._auto_trigger_memory_check({
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "trigger": "pre_tool_execution"
                    })
                
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
                # Automatically present relevant memories to Claude
                await self._present_memories_to_claude(memories, f"file access: {file_path}")
                
                # Also trigger check_relevant_memories for comprehensive context
                await self._auto_trigger_memory_check({
                    "file_path": file_path,
                    "trigger": "file_access",
                    "auto_context": True
                })
                
        except Exception as e:
            logger.error(f"Error suggesting file-related memories: {e}")
    
    async def _suggest_contextual_memories(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Suggest memories based on tool context."""
        # Prevent recursive memory suggestions during memory operations
        if self._memory_check_in_progress:
            logger.debug("AutoCode: Skipping contextual memory suggestions (recursion prevention)")
            return
            
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
                # Automatically present contextual memories to Claude
                await self._present_memories_to_claude(memories, f"tool context: {tool_name}")
                
                # Store context for learning
                await self._track_memory_usage(tool_name, arguments, memories)
                
        except Exception as e:
            logger.error(f"Error suggesting contextual memories: {e}")
    
    async def _present_memories_to_claude(self, memories: List[Dict[str, Any]], context: str) -> None:
        """
        Present relevant memories to Claude through the MCP system.
        
        This creates a natural integration where Claude receives memory context
        automatically during workflow, making past knowledge immediately available.
        """
        try:
            if not memories:
                return
            
            # Check if proactive memory presentation is enabled
            if not self.proactive_config.get("auto_present", True):
                logger.debug(f"Proactive memory presentation disabled - skipping {len(memories)} memories")
                return
            
            # Limit memories based on configuration
            max_memories = self.proactive_config.get("max_memories_per_trigger", 3)
            limited_memories = memories[:max_memories]
            
            # Format memories for presentation
            memory_summary = self._format_memories_for_presentation(limited_memories, context)
            
            # Store as a special "proactive_memory" type for immediate reference
            proactive_memory = {
                "id": f"proactive_{int(time.time())}",
                "type": "proactive_memory", 
                "content": memory_summary,
                "importance": 0.9,  # High importance for proactive suggestions
                "metadata": {
                    "trigger_context": context,
                    "memory_count": len(memories),
                    "auto_presented": True,
                    "presentation_time": datetime.utcnow().isoformat()
                }
            }
            
            # Store in short-term memory for immediate Claude access
            await self.domain_manager.store_memory(proactive_memory, "short_term")
            
            # Log for visibility 
            logger.info(f"AutoCode: Automatically presented {len(memories)} memories to Claude for {context}")
            
        except Exception as e:
            logger.error(f"Error presenting memories to Claude: {e}")
    
    def _format_memories_for_presentation(self, memories: List[Dict[str, Any]], context: str) -> str:
        """Format memories in a natural way for Claude to understand and use."""
        formatted_lines = [f"🧠 **Relevant Past Context** (triggered by {context}):\n"]
        
        for i, memory in enumerate(memories[:3], 1):  # Limit to top 3 for readability
            content = memory.get("content", "")
            memory_type = memory.get("type", "memory")
            created_at = memory.get("created_at", "")
            
            # Truncate long content for summary
            if len(content) > 200:
                content = content[:200] + "..."
            
            formatted_lines.append(f"{i}. **{memory_type.replace('_', ' ').title()}** {created_at[:10] if created_at else ''}")
            formatted_lines.append(f"   {content}")
            formatted_lines.append("")
        
        formatted_lines.append("*This context was automatically retrieved based on your current activity.*")
        
        return "\n".join(formatted_lines)
    
    async def _auto_trigger_memory_check(self, context: Dict[str, Any]) -> None:
        """
        Automatically trigger the check_relevant_memories MCP tool.
        
        This provides a seamless way to invoke comprehensive memory checking
        without Claude having to explicitly call the tool.
        """
        # Prevent recursive memory checks
        if self._memory_check_in_progress:
            logger.debug("AutoCode: Skipping auto-trigger memory check (recursion prevention)")
            return
            
        try:
            self._memory_check_in_progress = True
            
            # Use the MCP server's check_relevant_memories functionality
            if hasattr(self.domain_manager, 'mcp_server') and self.domain_manager.mcp_server:
                # This would ideally call the check_relevant_memories tool directly
                # For now, we'll use the domain manager's retrieve_memories method
                query = self._generate_contextual_query_from_context(context)
                if query:
                    memories = await self.domain_manager.retrieve_memories(
                        query=query,
                        limit=5,
                        min_similarity=0.6
                    )
                    
                    if memories:
                        logger.info(f"AutoCode: Auto-triggered memory check found {len(memories)} additional memories")
                        await self._present_memories_to_claude(memories, f"auto-check: {context.get('trigger', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error in auto-trigger memory check: {e}")
        finally:
            self._memory_check_in_progress = False
    
    def _generate_contextual_query_from_context(self, context: Dict[str, Any]) -> str:
        """Generate a search query from the provided context."""
        query_parts = []
        
        if "file_path" in context:
            from pathlib import Path
            path = Path(context["file_path"])
            query_parts.extend([path.name, path.suffix.replace(".", "")])
        
        if "command" in context:
            command = context["command"].split()[0]  # Base command
            query_parts.append(command)
        
        if "task" in context:
            query_parts.append(context["task"])
        
        if "project_path" in context:
            from pathlib import Path
            project_name = Path(context["project_path"]).name
            query_parts.append(project_name)
        
        return " ".join(query_parts) if query_parts else ""
    
    async def _track_memory_usage(self, tool_name: str, arguments: Dict[str, Any], memories: List[Dict[str, Any]]) -> None:
        """Track which memories were presented for learning and optimization."""
        try:
            # Store usage analytics for future improvement
            usage_data = {
                "id": f"usage_{int(time.time())}",
                "type": "memory_usage_analytics",
                "content": {
                    "tool_name": tool_name,
                    "context_arguments": arguments,
                    "memories_presented": [m.get("id") for m in memories],
                    "memory_count": len(memories),
                    "presentation_timestamp": datetime.utcnow().isoformat()
                },
                "importance": 0.3,  # Lower importance for analytics
                "metadata": {
                    "analytics_type": "proactive_memory_usage",
                    "auto_generated": True
                }
            }
            
            # Store for learning pattern analysis
            await self.domain_manager.store_memory(usage_data, "long_term")
            
        except Exception as e:
            logger.error(f"Error tracking memory usage: {e}")
    
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
    
    def _should_trigger_structured_thinking(self, intent: str, context: Dict[str, Any]) -> bool:
        """Intelligent detection of when to trigger structured thinking."""
        if not intent:
            return False
        
        # Complexity indicators
        complexity_score = 0
        intent_lower = intent.lower()
        
        # Length-based complexity (longer requests often need more thinking)
        if len(intent) > 100:
            complexity_score += 1
        if len(intent) > 200:
            complexity_score += 1
            
        # Keyword-based complexity detection
        high_complexity_keywords = [
            "implement", "design", "architecture", "solution", "approach", 
            "strategy", "optimize", "refactor", "migrate", "integrate",
            "system", "complex", "multiple", "several", "various",
            "performance", "scalability", "security", "architecture",
            "framework", "pattern", "best practice", "enterprise"
        ]
        
        medium_complexity_keywords = [
            "create", "build", "develop", "add", "improve", "enhance",
            "fix", "debug", "troubleshoot", "configure", "setup",
            "test", "validate", "analyze", "review", "compare"
        ]
        
        question_indicators = [
            "how should", "what approach", "best way", "which method",
            "how to implement", "what pattern", "how do i", "what's the best"
        ]
        
        # Count complexity indicators
        for keyword in high_complexity_keywords:
            if keyword in intent_lower:
                complexity_score += 2
                
        for keyword in medium_complexity_keywords:
            if keyword in intent_lower:
                complexity_score += 1
                
        for indicator in question_indicators:
            if indicator in intent_lower:
                complexity_score += 1
        
        # Context-based complexity
        arguments = context.get("arguments", {})
        if arguments.get("use_structured_thinking", False):
            complexity_score += 3  # Explicit request
            
        # Project context complexity
        project_context = arguments.get("context", {})
        if project_context:
            if project_context.get("project_type") in ["enterprise", "complex", "multi-service"]:
                complexity_score += 2
            if len(project_context) > 5:  # Rich context suggests complexity
                complexity_score += 1
        
        # Multiple sentence/question complexity
        sentences = len([s for s in intent.split('.') if s.strip()])
        questions = len([s for s in intent.split('?') if s.strip()])
        if sentences > 2 or questions > 1:
            complexity_score += 1
            
        # Decision threshold
        return complexity_score >= 4
    
    async def _auto_trigger_structured_thinking(self, intent: str, context: Dict[str, Any]) -> None:
        """Automatically trigger structured thinking process for complex problems."""
        try:
            # Create thinking session
            from datetime import datetime, timezone
            session_id = f"auto_thinking_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"AutoCode: Starting structured thinking session {session_id} for: {intent[:100]}...")
            
            # Stage 1: Problem Definition (automatic)
            await self._process_auto_thinking_stage(
                session_id=session_id,
                stage="problem_definition", 
                content=f"Auto-detected complex problem: {intent}",
                thought_number=1,
                total_expected=3,  # Start with 3 stages, expand if needed
                tags=self._extract_tags_from_intent(intent),
                context=context
            )
            
            # Stage 2: Research (automatic memory lookup)
            research_content = await self._auto_research_stage(intent, context)
            await self._process_auto_thinking_stage(
                session_id=session_id,
                stage="research",
                content=research_content,
                thought_number=2,
                total_expected=3,
                tags=["research", "automatic", "memory_search"],
                context=context
            )
            
            # Stage 3: Analysis suggestion (guide user to continue)
            analysis_prompt = await self._generate_analysis_prompt(intent, context)
            await self._process_auto_thinking_stage(
                session_id=session_id,
                stage="analysis", 
                content=analysis_prompt,
                thought_number=3,
                total_expected=5,  # Expand to full 5-stage process
                tags=["analysis", "user_guided", "next_step"],
                context=context
            )
            
            # Store session summary for user awareness
            await self._store_thinking_session_summary(session_id, intent, context)
            
        except Exception as e:
            logger.error(f"Error in auto-trigger structured thinking: {e}")
    
    async def _process_auto_thinking_stage(
        self, 
        session_id: str, 
        stage: str, 
        content: str, 
        thought_number: int,
        total_expected: int,
        tags: List[str],
        context: Dict[str, Any]
    ) -> None:
        """Process a single structured thinking stage automatically."""
        try:
            # This would call the MCP process_structured_thought tool
            if hasattr(self.domain_manager, 'mcp_server') and self.domain_manager.mcp_server:
                # For now, store as structured thinking memory
                thinking_memory = {
                    "id": f"thinking_{session_id}_{thought_number}",
                    "type": "structured_thinking",
                    "content": content,
                    "importance": 0.8,
                    "metadata": {
                        "session_id": session_id,
                        "stage": stage,
                        "thought_number": thought_number,
                        "total_expected": total_expected,
                        "tags": tags,
                        "auto_generated": True,
                        "trigger_context": context.get("arguments", {}).get("intent", "")[:100]
                    }
                }
                
                await self.domain_manager.store_memory(thinking_memory, "short_term")
                logger.debug(f"AutoCode: Stored auto-thinking stage {stage} for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error processing auto-thinking stage {stage}: {e}")
    
    async def _auto_research_stage(self, intent: str, context: Dict[str, Any]) -> str:
        """Automatically conduct research stage by searching memory."""
        try:
            if not self.domain_manager:
                return "Research stage: Memory system not available for automatic research."
            
            # Search for relevant memories
            research_memories = await self.domain_manager.retrieve_memories(
                query=intent,
                memory_types=["project_pattern", "session_summary", "solution_synthesis", "architectural_decision"],
                limit=5,
                min_similarity=0.6
            )
            
            if research_memories:
                memory_insights = []
                for memory in research_memories[:3]:
                    mem_type = memory.get("type", "memory")
                    content_preview = memory.get("content", "")[:100] + "..." if len(memory.get("content", "")) > 100 else memory.get("content", "")
                    memory_insights.append(f"- {mem_type}: {content_preview}")
                
                research_content = f"Automatic research found {len(research_memories)} relevant memories:\n" + "\n".join(memory_insights)
                research_content += f"\n\nSuggested next step: Analyze how these patterns apply to current problem."
            else:
                research_content = f"Automatic research: No directly relevant memories found for '{intent[:50]}...'. This appears to be a novel problem requiring fresh analysis."
                
            return research_content
            
        except Exception as e:
            logger.error(f"Error in auto-research stage: {e}")
            return f"Research stage encountered error: {str(e)}"
    
    async def _generate_analysis_prompt(self, intent: str, context: Dict[str, Any]) -> str:
        """Generate analysis stage prompt to guide user."""
        components = self._identify_problem_components(intent)
        
        analysis_prompt = f"Analysis stage ready. Based on the problem '{intent[:100]}...', consider these components:\n\n"
        
        for i, component in enumerate(components, 1):
            analysis_prompt += f"{i}. {component}\n"
        
        analysis_prompt += "\nNext steps: Break down each component, identify dependencies, and assess complexity."
        analysis_prompt += "\nContinue with: process_structured_thought(stage='analysis', content='Your analysis...', ...)"
        
        return analysis_prompt
    
    def _identify_problem_components(self, intent: str) -> List[str]:
        """Identify key components of the problem for analysis."""
        components = []
        intent_lower = intent.lower()
        
        # Technical components
        if any(word in intent_lower for word in ["api", "endpoint", "service"]):
            components.append("API/Service integration")
        if any(word in intent_lower for word in ["database", "data", "storage"]):
            components.append("Data persistence and management")
        if any(word in intent_lower for word in ["ui", "interface", "frontend"]):
            components.append("User interface design")
        if any(word in intent_lower for word in ["performance", "scale", "optimization"]):
            components.append("Performance and scalability")
        if any(word in intent_lower for word in ["security", "auth", "authentication"]):
            components.append("Security and authentication")
        if any(word in intent_lower for word in ["test", "testing", "quality"]):
            components.append("Testing and quality assurance")
        
        # Process components
        if any(word in intent_lower for word in ["deploy", "deployment", "production"]):
            components.append("Deployment and operations")
        if any(word in intent_lower for word in ["monitor", "logging", "observability"]):
            components.append("Monitoring and observability")
        
        # Default components for general problems
        if not components:
            components = [
                "Core functionality requirements",
                "Technical implementation approach", 
                "Integration and dependencies",
                "Error handling and edge cases"
            ]
        
        return components
    
    def _extract_tags_from_intent(self, intent: str) -> List[str]:
        """Extract relevant tags from user intent for structured thinking."""
        tags = ["auto_triggered", "complex_problem"]
        intent_lower = intent.lower()
        
        # Technology tags
        tech_keywords = {
            "python": "python", "javascript": "javascript", "typescript": "typescript",
            "react": "react", "vue": "vue", "angular": "angular",
            "api": "api", "rest": "rest", "graphql": "graphql",
            "database": "database", "sql": "sql", "nosql": "nosql",
            "docker": "docker", "kubernetes": "kubernetes",
            "aws": "aws", "azure": "azure", "gcp": "gcp"
        }
        
        for keyword, tag in tech_keywords.items():
            if keyword in intent_lower:
                tags.append(tag)
        
        # Problem type tags
        problem_types = {
            "performance": "performance", "security": "security",
            "scalability": "scalability", "architecture": "architecture",
            "refactor": "refactoring", "migrate": "migration",
            "integrate": "integration", "optimize": "optimization"
        }
        
        for keyword, tag in problem_types.items():
            if keyword in intent_lower:
                tags.append(tag)
        
        return list(set(tags))
    
    async def _store_thinking_session_summary(self, session_id: str, intent: str, context: Dict[str, Any]) -> None:
        """Store a summary of the auto-triggered thinking session."""
        try:
            summary_content = f"Structured thinking session auto-triggered for: {intent}\n"
            summary_content += f"Session ID: {session_id}\n"
            summary_content += f"Status: Problem definition and research completed automatically\n"
            summary_content += f"Next: Continue with analysis stage using process_structured_thought tool"
            
            session_memory = {
                "id": f"session_summary_{session_id}",
                "type": "thinking_session_summary",
                "content": summary_content,
                "importance": 0.9,
                "metadata": {
                    "session_id": session_id,
                    "auto_triggered": True,
                    "original_intent": intent,
                    "stages_completed": 3,
                    "status": "user_continuation_needed"
                }
            }
            
            await self.domain_manager.store_memory(session_memory, "short_term")
            logger.info(f"AutoCode: Stored thinking session summary for {session_id}")
            
        except Exception as e:
            logger.error(f"Error storing thinking session summary: {e}")
    
    # MCP awareness hook implementations
    async def _on_mcp_user_request(self, context: Dict[str, Any]) -> None:
        """Hook for MCP-aware user request processing with structured thinking integration."""
        try:
            request_data = context.get("data", {})
            user_request = request_data.get("request", "")
            request_context = request_data.get("context", {})
            
            # MCP tool awareness
            if user_request and self.mcp_awareness_hooks:
                suggestion = await self.mcp_awareness_hooks.on_user_request(user_request, request_context)
                if suggestion:
                    logger.info(f"MCP tool suggestion: {suggestion[:100]}...")
                    # Could store the suggestion in memory or present it to the user
            
            # Intelligent structured thinking for complex user requests
            if user_request and self._should_trigger_structured_thinking(user_request, context):
                logger.info(f"AutoCode: Complex user request triggering structured thinking: {user_request[:50]}...")
                await self._auto_trigger_structured_thinking(user_request, {
                    "arguments": {"intent": user_request, "context": request_context},
                    "trigger": "mcp_user_request"
                })
                    
        except Exception as e:
            logger.error(f"Error in MCP user request hook: {e}")
    
    async def _on_mcp_tool_about_to_execute(self, context: Dict[str, Any]) -> None:
        """Hook for MCP-aware tool execution interception."""
        try:
            tool_data = context.get("data", {})
            tool_name = tool_data.get("tool_name", "")
            tool_context = tool_data.get("context", {})
            
            if tool_name and self.mcp_awareness_hooks:
                suggestion = await self.mcp_awareness_hooks.on_tool_about_to_execute(tool_name, tool_context)
                if suggestion:
                    logger.info(f"MCP alternative suggestion for {tool_name}: {suggestion[:100]}...")
                    
        except Exception as e:
            logger.error(f"Error in MCP tool about to execute hook: {e}")
    
    async def _on_mcp_context_change(self, context: Dict[str, Any]) -> None:
        """Hook for MCP-aware context change processing."""
        try:
            change_data = context.get("data", {})
            new_context = change_data.get("context", {})
            
            if new_context and self.mcp_awareness_hooks:
                suggestion = await self.mcp_awareness_hooks.on_context_change(new_context)
                if suggestion:
                    logger.info(f"MCP context suggestion: {suggestion[:100]}...")
                    
        except Exception as e:
            logger.error(f"Error in MCP context change hook: {e}")
    
    async def _on_mcp_error_occurred(self, context: Dict[str, Any]) -> None:
        """Hook for MCP-aware error resolution."""
        try:
            error_data = context.get("data", {})
            error_message = error_data.get("error", "")
            error_context = error_data.get("context", {})
            
            if error_message and self.mcp_awareness_hooks:
                suggestion = await self.mcp_awareness_hooks.on_error_occurred(error_message, error_context)
                if suggestion:
                    logger.info(f"MCP error resolution suggestion: {suggestion[:100]}...")
                    
        except Exception as e:
            logger.error(f"Error in MCP error occurred hook: {e}")
    
    # Structured thinking hooks and detection methods
    async def _on_structured_thought_process(self, context: Dict[str, Any]) -> None:
        """Hook for structured thought processing."""
        try:
            arguments = context.get("arguments", {})
            session_id = arguments.get("session_id")
            content = arguments.get("content", "")
            
            if session_id and content:
                # Track active thinking session
                self.active_thinking_sessions[session_id] = {
                    "last_activity": datetime.utcnow().isoformat(),
                    "stage": arguments.get("stage", "unknown"),
                    "thought_count": arguments.get("thought_number", 0)
                }
                
                logger.debug(f"AutoCode: Processing structured thought for session {session_id}")
                
                # Auto-suggest next stage if structured thinking extension available
                if self.structured_thinking_extension:
                    next_stage_info = await self.structured_thinking_extension.suggest_next_thinking_stage(session_id)
                    if next_stage_info.get("next_stage") != "complete":
                        logger.info(f"AutoCode: Suggested next thinking stage: {next_stage_info.get('next_stage')}")
                        
        except Exception as e:
            logger.error(f"Error in structured thought processing hook: {e}")
    
    async def _on_thinking_summary_generate(self, context: Dict[str, Any]) -> None:
        """Hook for thinking summary generation."""
        try:
            arguments = context.get("arguments", {})
            session_id = arguments.get("session_id")
            
            if session_id and session_id in self.active_thinking_sessions:
                # Mark session as completing
                self.active_thinking_sessions[session_id]["status"] = "summarizing"
                logger.info(f"AutoCode: Generating thinking summary for session {session_id}")
                
                # Generate action plan if structured thinking extension available
                if self.structured_thinking_extension:
                    action_plan = await self.structured_thinking_extension.generate_coding_action_plan(session_id)
                    if not action_plan.get("error"):
                        logger.info(f"AutoCode: Generated action plan with {len(action_plan.get('action_items', []))} items")
                        
        except Exception as e:
            logger.error(f"Error in thinking summary generation hook: {e}")
    
    async def _on_thinking_process_continue(self, context: Dict[str, Any]) -> None:
        """Hook for continuing thinking processes."""
        try:
            arguments = context.get("arguments", {})
            session_id = arguments.get("session_id")
            
            if session_id and self.structured_thinking_extension:
                # Get continuation context and suggestions
                next_stage_info = await self.structured_thinking_extension.suggest_next_thinking_stage(session_id)
                if next_stage_info.get("next_stage") != "complete":
                    logger.info(f"AutoCode: Continuing thinking process - next stage: {next_stage_info.get('next_stage')}")
                    
        except Exception as e:
            logger.error(f"Error in thinking process continuation hook: {e}")
    
    def _should_trigger_structured_thinking(self, content: str, context: Dict[str, Any] = None) -> bool:
        """
        Detect if user input should trigger structured thinking process.
        
        Similar to mcp-sequential-thinking's automatic detection patterns.
        """
        if not self.proactive_config.get("triggers", {}).get("structured_thinking", True):
            return False
            
        # Check content complexity indicators
        complexity_indicators = [
            "implement", "design", "architect", "plan", "analyze", "solve",
            "build", "create", "develop", "optimize", "refactor", 
            "debug", "troubleshoot", "investigate", "research",
            "how should I", "what's the best way", "help me figure out",
            "I need to understand", "break down", "step by step"
        ]
        
        content_lower = content.lower()
        has_complexity_indicator = any(indicator in content_lower for indicator in complexity_indicators)
        
        # Check for problem-solving language patterns
        problem_patterns = [
            "problem", "issue", "challenge", "difficulty", "stuck",
            "not working", "error", "bug", "failing", "broken"
        ]
        has_problem_pattern = any(pattern in content_lower for pattern in problem_patterns)
        
        # Check content length (longer requests often need structured thinking)
        is_substantial = len(content.split()) > 15
        
        # Check for multiple questions or requirements
        has_multiple_parts = any(marker in content for marker in ["?", "and", "also", "additionally", "furthermore"])
        
        # Scoring system (like complexity detection in existing code)
        score = 0
        if has_complexity_indicator:
            score += 2
        if has_problem_pattern:
            score += 1
        if is_substantial:
            score += 1
        if has_multiple_parts:
            score += 1
            
        # Additional context scoring
        if context:
            # If user is working on a project, more likely to need structured thinking
            if context.get("data", {}).get("project_context"):
                score += 1
            # If there are multiple files involved
            if context.get("data", {}).get("file_count", 0) > 3:
                score += 1
        
        # Trigger if score meets threshold (similar to mcp-sequential-thinking approach)
        return score >= 3
    
    async def _auto_trigger_structured_thinking(self, content: str, context: Dict[str, Any] = None) -> None:
        """
        Automatically trigger structured thinking process with smart context integration.
        
        Enhanced with multi-dimensional analysis similar to memory system's sophistication.
        """
        if not self.structured_thinking_extension:
            logger.warning("AutoCode: Structured thinking extension not available")
            return
            
        try:
            # Enhanced context extraction with multi-dimensional analysis
            enhanced_context = await self._build_enhanced_context(content, context)
            
            # Use proactive thinking suggestions for smarter triggering
            thinking_suggestions = await self.structured_thinking_extension.suggest_proactive_thinking(
                enhanced_context, 
                limit=5
            )
            
            if not thinking_suggestions.get("suggestions"):
                logger.debug("AutoCode: No proactive thinking suggestions found")
                return
            
            # Check if any high-confidence suggestions warrant auto-triggering
            high_confidence_suggestions = [
                s for s in thinking_suggestions["suggestions"]
                if s["confidence"] >= 0.8 and s["priority"] == "high"
            ]
            
            if high_confidence_suggestions:
                # Auto-trigger the best suggestion
                best_suggestion = high_confidence_suggestions[0]
                
                auto_trigger_result = await self.structured_thinking_extension.auto_trigger_thinking_from_context(
                    enhanced_context,
                    threshold=0.8
                )
                
                if auto_trigger_result.get("status") == "auto_triggered":
                    session_id = auto_trigger_result["session_id"]
                    logger.info(f"AutoCode: Auto-started structured thinking session {session_id} ({best_suggestion['type']})")
                    
                    # Track the auto-started session with enhanced metadata
                    self.active_thinking_sessions[session_id] = {
                        "auto_started": True,
                        "trigger_content": content[:100],
                        "started_at": datetime.utcnow().isoformat(),
                        "suggestion_type": best_suggestion["type"],
                        "confidence": best_suggestion["confidence"],
                        "estimated_time": best_suggestion["estimated_time"],
                        "context_complexity": enhanced_context.get("complexity_score", 0),
                        "auto_progression_enabled": True
                    }
                    
                    # Automatically progress to next stage if confidence is very high
                    if best_suggestion["confidence"] >= 0.9:
                        await self._attempt_auto_progression(session_id)
                    
                    return auto_trigger_result
            else:
                logger.debug(f"AutoCode: Thinking suggestions below threshold - highest confidence: {max([s['confidence'] for s in thinking_suggestions['suggestions']], default=0)}")
                
        except Exception as e:
            logger.error(f"Error auto-triggering structured thinking: {e}")
    
    async def _build_enhanced_context(self, content: str, base_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build enhanced context with multi-dimensional analysis.
        
        Similar to memory system's comprehensive context building.
        """
        enhanced_context = {
            "current_task": content,
            "recent_activity": [],
            "files_accessed": [],
            "commands_executed": [],
            "project_context": {},
            "complexity_score": 0.0
        }
        
        # Extract from base context
        if base_context:
            data = base_context.get("data", {})
            enhanced_context.update({
                "project_context": data.get("project_context", {}),
                "recent_activity": data.get("recent_activity", []),
                "files_accessed": data.get("files_accessed", []),
                "commands_executed": data.get("commands_executed", [])
            })
        
        # Use AutoCode hooks to get additional context
        if self.autocode_hooks:
            try:
                # Get project patterns for context enrichment
                project_patterns = await self.autocode_hooks.get_cached_project_patterns()
                if project_patterns:
                    enhanced_context["project_context"].update({
                        "detected_frameworks": project_patterns.get("frameworks", []),
                        "detected_languages": project_patterns.get("languages", []),
                        "project_complexity": project_patterns.get("complexity", 0),
                        "architecture_patterns": project_patterns.get("patterns", [])
                    })
            except Exception as e:
                logger.debug(f"Could not get project patterns for context: {e}")
        
        # Calculate context complexity score
        complexity_factors = []
        
        # Task complexity
        task_complexity = len(content.split()) * 0.01
        complexity_factors.append(min(0.3, task_complexity))
        
        # Project complexity
        if enhanced_context["project_context"]:
            framework_count = len(enhanced_context["project_context"].get("detected_frameworks", []))
            language_count = len(enhanced_context["project_context"].get("detected_languages", []))
            project_complexity = (framework_count + language_count) * 0.05
            complexity_factors.append(min(0.2, project_complexity))
        
        # Activity complexity
        file_complexity = len(enhanced_context["files_accessed"]) * 0.02
        command_complexity = len(enhanced_context["commands_executed"]) * 0.01
        complexity_factors.extend([min(0.15, file_complexity), min(0.1, command_complexity)])
        
        enhanced_context["complexity_score"] = min(1.0, sum(complexity_factors))
        
        # Add contextual intelligence similar to memory system
        enhanced_context.update({
            "intelligence_level": "high" if enhanced_context["complexity_score"] > 0.7 else "medium" if enhanced_context["complexity_score"] > 0.4 else "low",
            "auto_progression_recommended": enhanced_context["complexity_score"] > 0.6,
            "proactive_memory_integration": True,
            "multi_dimensional_analysis": True
        })
        
        return enhanced_context
    
    async def _attempt_auto_progression(self, session_id: str) -> None:
        """
        Attempt automatic progression to next thinking stage.
        
        Similar to memory system's proactive operations.
        """
        try:
            if session_id not in self.active_thinking_sessions:
                return
                
            session_data = self.active_thinking_sessions[session_id]
            if not session_data.get("auto_progression_enabled", False):
                return
            
            # Auto-progress to next stage
            progression_result = await self.structured_thinking_extension.auto_progress_thinking_stage(
                session_id, 
                auto_execute=True
            )
            
            if progression_result.get("status") == "auto_progressed":
                logger.info(f"AutoCode: Auto-progressed thinking session {session_id} to {progression_result.get('stage')} stage")
                
                # Update session tracking
                session_data.update({
                    "last_auto_progression": datetime.utcnow().isoformat(),
                    "current_stage": progression_result.get("stage"),
                    "auto_progression_count": session_data.get("auto_progression_count", 0) + 1
                })
                
                # If we're at synthesis or conclusion, consider generating summary
                if progression_result.get("stage") in ["synthesis", "conclusion"]:
                    await self._attempt_auto_summary(session_id)
                    
        except Exception as e:
            logger.error(f"Error in auto-progression: {e}")
    
    async def _attempt_auto_summary(self, session_id: str) -> None:
        """
        Attempt automatic summary generation for completed thinking sessions.
        
        Similar to memory system's automatic session summaries.
        """
        try:
            if session_id not in self.active_thinking_sessions:
                return
                
            session_data = self.active_thinking_sessions[session_id]
            
            # Generate comprehensive summary
            summary_result = await self.structured_thinking_extension.generate_coding_action_plan(session_id)
            
            if summary_result and not summary_result.get("error"):
                logger.info(f"AutoCode: Auto-generated action plan for thinking session {session_id}")
                
                # Update session as completed
                session_data.update({
                    "completed_at": datetime.utcnow().isoformat(),
                    "action_plan_generated": True,
                    "plan_memory_id": summary_result.get("plan_memory_id"),
                    "total_action_items": len(summary_result.get("action_items", []))
                })
                
        except Exception as e:
            logger.error(f"Error in auto-summary generation: {e}")
    
    async def get_enhanced_thinking_suggestions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get enhanced thinking suggestions with full context integration.
        
        Similar to memory system's suggest_memory_queries with full intelligence.
        """
        try:
            if not self.structured_thinking_extension:
                return {"error": "Structured thinking extension not available"}
            
            # Build enhanced context
            enhanced_context = await self._build_enhanced_context(
                context.get("current_task", ""), 
                context
            )
            
            # Get proactive suggestions
            suggestions = await self.structured_thinking_extension.suggest_proactive_thinking(
                enhanced_context, 
                limit=5
            )
            
            # Add hook-specific enhancements
            if suggestions.get("suggestions"):
                for suggestion in suggestions["suggestions"]:
                    # Add auto-execution capability
                    suggestion["auto_executable"] = suggestion["confidence"] >= 0.8
                    suggestion["hook_integration"] = True
                    suggestion["memory_integration"] = enhanced_context.get("proactive_memory_integration", False)
                    
                    # Add time estimates based on complexity
                    complexity = enhanced_context.get("complexity_score", 0.5)
                    base_time = int(suggestion["estimated_time"].split("-")[0])
                    adjusted_time = int(base_time * (1 + complexity * 0.5))
                    suggestion["adjusted_time_estimate"] = f"{adjusted_time}-{adjusted_time + 5} minutes"
            
            # Add session management suggestions
            active_sessions = len(self.active_thinking_sessions)
            if active_sessions > 0:
                suggestions["active_sessions"] = {
                    "count": active_sessions,
                    "sessions": [
                        {
                            "session_id": sid,
                            "type": data.get("suggestion_type", "unknown"),
                            "stage": data.get("current_stage", "unknown"),
                            "auto_started": data.get("auto_started", False)
                        }
                        for sid, data in self.active_thinking_sessions.items()
                    ]
                }
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting enhanced thinking suggestions: {e}")
            return {"error": f"Failed to get suggestions: {e}"}


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

async def trigger_mcp_user_request_hook(user_request: str, context: Dict[str, Any] = None) -> None:
    """Trigger MCP user request hook for proactive tool suggestions."""
    await HookRegistry.trigger_event_hooks("user_request", {
        "request": user_request,
        "context": context or {}
    })

async def trigger_mcp_tool_about_to_execute_hook(tool_name: str, context: Dict[str, Any] = None) -> None:
    """Trigger MCP tool about to execute hook for alternative suggestions."""
    await HookRegistry.trigger_event_hooks("tool_about_to_execute", {
        "tool_name": tool_name,
        "context": context or {}
    })

async def trigger_mcp_context_change_hook(context: Dict[str, Any]) -> None:
    """Trigger MCP context change hook for context-aware suggestions."""
    await HookRegistry.trigger_event_hooks("mcp_context_change", {
        "context": context
    })

async def trigger_mcp_error_occurred_hook(error: str, context: Dict[str, Any] = None) -> None:
    """Trigger MCP error occurred hook for error resolution suggestions."""
    await HookRegistry.trigger_event_hooks("error_occurred", {
        "error": error,
        "context": context or {}
    })