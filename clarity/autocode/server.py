"""
MCP server extensions for AutoCode command intelligence.
"""

from typing import Any, Dict, List, Optional
from loguru import logger


class AutoCodeServerExtension:
    """
    Extends the MCP server with AutoCode command intelligence tools.
    """
    
    def __init__(self, domain_manager, autocode_hooks):
        """
        Initialize the AutoCode server extension.
        
        Args:
            domain_manager: The memory domain manager
            autocode_hooks: The AutoCode hooks instance
        """
        self.domain_manager = domain_manager
        self.autocode_hooks = autocode_hooks
    
    async def handle_suggest_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle suggest_command tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Command suggestions
        """
        try:
            intent = arguments.get("intent", "")
            context = arguments.get("context", {})
            
            if not intent:
                return {"error": "Intent is required"}
            
            # Get command suggestions
            suggestions = await self.autocode_hooks.suggest_command(intent, context)
            
            # Format response
            return {
                "intent": intent,
                "suggestions": suggestions,
                "context": context,
                "total_suggestions": len(suggestions)
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling suggest_command: {e}")
            return {"error": f"Failed to suggest commands: {str(e)}"}
    
    async def handle_track_bash(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle track_bash tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Tracking confirmation
        """
        try:
            command = arguments.get("command", "")
            exit_code = arguments.get("exit_code", 0)
            output = arguments.get("output", "")
            context = arguments.get("context", {})
            
            if not command:
                return {"error": "Command is required"}
            
            # Track the bash execution
            await self.autocode_hooks.on_bash_execution(
                command=command,
                exit_code=exit_code,
                output=output,
                working_directory=context.get("working_directory")
            )
            
            return {
                "status": "tracked",
                "command": command,
                "exit_code": exit_code,
                "success": exit_code == 0
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling track_bash: {e}")
            return {"error": f"Failed to track bash execution: {str(e)}"}
    
    async def handle_get_session_history(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_session_history tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Session history data
        """
        try:
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            days_back = arguments.get("days_back", 30)
            
            if not query:
                return {"error": "Query is required"}
            
            # Search session summaries
            memories = await self.domain_manager.retrieve_memories(
                query=query,
                memory_types=["session_summary"],
                limit=limit,
                min_similarity=0.4,
                include_metadata=True
            )
            
            # Filter by days_back if needed
            from datetime import datetime, timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            filtered_memories = []
            for memory in memories:
                created_at = memory.get("created_at")
                if created_at:
                    try:
                        memory_date = datetime.fromisoformat(created_at)
                        if memory_date >= cutoff_date:
                            filtered_memories.append(memory)
                    except:
                        filtered_memories.append(memory)  # Include if can't parse
                else:
                    filtered_memories.append(memory)
            
            return {
                "query": query,
                "sessions": filtered_memories,
                "total_found": len(filtered_memories),
                "days_back": days_back
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling get_session_history: {e}")
            return {"error": f"Failed to get session history: {str(e)}"}
    
    async def handle_get_project_patterns(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_project_patterns tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Project patterns data
        """
        try:
            project_path = arguments.get("project_path", "")
            pattern_types = arguments.get("pattern_types", [])
            
            if not project_path:
                return {"error": "Project path is required"}
            
            # Check if project path exists
            import os
            if not os.path.exists(project_path):
                return {"error": f"Project path does not exist: {project_path}"}
            
            # Get project patterns from AutoCode domain
            patterns = await self.domain_manager.autocode_domain.get_project_patterns(project_path)
            
            # Filter by pattern types if specified
            if pattern_types:
                filtered_patterns = {
                    k: v for k, v in patterns.items() 
                    if k in pattern_types
                }
                patterns = filtered_patterns
            
            return {
                "project_path": project_path,
                "patterns": patterns,
                "pattern_types_requested": pattern_types,
                "total_patterns": len(patterns)
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling get_project_patterns: {e}")
            return {"error": f"Failed to get project patterns: {str(e)}"}
    
    async def handle_track_file_access(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle track_file_access tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Tracking confirmation
        """
        try:
            file_path = arguments.get("file_path", "")
            operation = arguments.get("operation", "read")
            content = arguments.get("content", "")
            
            if not file_path:
                return {"error": "File path is required"}
            
            # Track the file access
            await self.autocode_hooks.on_file_read(
                file_path=file_path,
                content=content,
                operation=operation
            )
            
            return {
                "status": "tracked",
                "file_path": file_path,
                "operation": operation,
                "content_length": len(content) if content else 0
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling track_file_access: {e}")
            return {"error": f"Failed to track file access: {str(e)}"}
    
    async def handle_get_command_success_rate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_command_success_rate tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Success rate data
        """
        try:
            command = arguments.get("command", "")
            context = arguments.get("context", {})
            
            if not command:
                return {"error": "Command is required"}
            
            # Get success rate
            success_rate = await self.autocode_hooks.get_command_success_rate(command, context)
            
            return {
                "command": command,
                "success_rate": success_rate,
                "context": context,
                "confidence": "high" if success_rate > 0.8 else "medium" if success_rate > 0.5 else "low"
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling get_command_success_rate: {e}")
            return {"error": f"Failed to get command success rate: {str(e)}"}
    
    async def handle_learn_retry_patterns(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle learn_retry_patterns tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Learning confirmation
        """
        try:
            # Get command learner
            command_learner = getattr(self.domain_manager.autocode_domain, 'command_learner', None)
            
            if not command_learner:
                return {"error": "Command learner not available"}
            
            # Learn retry patterns
            await command_learner.learn_retry_patterns()
            
            return {
                "status": "completed",
                "message": "Retry patterns learned from recent command history"
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling learn_retry_patterns: {e}")
            return {"error": f"Failed to learn retry patterns: {str(e)}"}
    
    async def handle_find_similar_sessions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle find_similar_sessions tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Similar sessions data
        """
        try:
            query = arguments.get("query", "")
            context = arguments.get("context", {})
            time_range_days = arguments.get("time_range_days")
            
            if not query:
                return {"error": "Query is required"}
            
            # Find similar sessions using AutoCode domain
            sessions = await self.domain_manager.autocode_domain.find_similar_sessions(
                query=query,
                context=context,
                time_range_days=time_range_days
            )
            
            return {
                "query": query,
                "context": context,
                "sessions": sessions,
                "total_found": len(sessions),
                "time_range_days": time_range_days
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling find_similar_sessions: {e}")
            return {"error": f"Failed to find similar sessions: {str(e)}"}
    
    async def handle_get_continuation_context(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_continuation_context tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Continuation context data
        """
        try:
            current_task = arguments.get("current_task", "")
            project_context = arguments.get("project_context", {})
            
            if not current_task:
                return {"error": "Current task description is required"}
            
            # Get continuation context
            context = await self.domain_manager.autocode_domain.get_context_for_continuation(
                current_task=current_task,
                project_context=project_context
            )
            
            return {
                "current_task": current_task,
                "project_context": project_context,
                "continuation_context": context
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling get_continuation_context: {e}")
            return {"error": f"Failed to get continuation context: {str(e)}"}
    
    async def handle_suggest_workflow_optimizations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle suggest_workflow_optimizations tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Workflow optimization suggestions
        """
        try:
            current_workflow = arguments.get("current_workflow", [])
            session_context = arguments.get("session_context", {})
            
            if not current_workflow:
                return {"error": "Current workflow steps are required"}
            
            # Get workflow optimization suggestions
            optimizations = await self.domain_manager.autocode_domain.suggest_workflow_optimizations(
                current_workflow=current_workflow,
                session_context=session_context
            )
            
            return {
                "current_workflow": current_workflow,
                "session_context": session_context,
                "optimizations": optimizations,
                "total_suggestions": len(optimizations)
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling suggest_workflow_optimizations: {e}")
            return {"error": f"Failed to suggest workflow optimizations: {str(e)}"}
    
    async def handle_get_learning_progression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_learning_progression tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Learning progression data
        """
        try:
            topic = arguments.get("topic", "")
            time_range_days = arguments.get("time_range_days", 180)
            
            if not topic:
                return {"error": "Topic is required"}
            
            # Get learning progression
            progression = await self.domain_manager.autocode_domain.get_learning_progression(
                topic=topic,
                time_range_days=time_range_days
            )
            
            return {
                "topic": topic,
                "time_range_days": time_range_days,
                "progression": progression
            }
            
        except Exception as e:
            logger.error(f"AutoCode server: Error handling get_learning_progression: {e}")
            return {"error": f"Failed to get learning progression: {str(e)}"}
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for AutoCode features.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "suggest_command",
                "description": "Get intelligent command suggestions based on intent and context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "What you want to accomplish (e.g., 'delete file', 'install dependencies')"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current context (project type, platform, etc.)",
                            "properties": {
                                "project_type": {"type": "string"},
                                "project_path": {"type": "string"},
                                "platform": {"type": "string"}
                            }
                        }
                    },
                    "required": ["intent"]
                }
            },
            {
                "name": "track_bash",
                "description": "Track bash command execution for learning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command that was executed"
                        },
                        "exit_code": {
                            "type": "integer",
                            "description": "Exit code from command execution"
                        },
                        "output": {
                            "type": "string",
                            "description": "Command output or error message"
                        },
                        "context": {
                            "type": "object",
                            "description": "Execution context",
                            "properties": {
                                "project_type": {"type": "string"},
                                "project_path": {"type": "string"},
                                "working_directory": {"type": "string"}
                            }
                        }
                    },
                    "required": ["command", "exit_code"]
                }
            },
            {
                "name": "get_session_history",
                "description": "Search historical session data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for session history"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of sessions to return",
                            "minimum": 1,
                            "maximum": 20
                        },
                        "days_back": {
                            "type": "integer",
                            "description": "How many days back to search",
                            "minimum": 1,
                            "maximum": 90
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_project_patterns",
                "description": "Get detected patterns for a project",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze"
                        },
                        "pattern_types": {
                            "type": "array",
                            "description": "Types of patterns to retrieve",
                            "items": {
                                "type": "string",
                                "enum": ["architectural", "naming", "component", "testing", "build"]
                            }
                        }
                    },
                    "required": ["project_path"]
                }
            },
            {
                "name": "track_file_access",
                "description": "Track file access for pattern learning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file that was accessed"
                        },
                        "operation": {
                            "type": "string",
                            "description": "Type of operation performed",
                            "enum": ["read", "write", "edit", "delete"]
                        },
                        "content": {
                            "type": "string",
                            "description": "File content (for analysis)"
                        }
                    },
                    "required": ["file_path", "operation"]
                }
            },
            {
                "name": "get_command_success_rate",
                "description": "Get success rate for a specific command",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to check success rate for"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current context",
                            "properties": {
                                "project_type": {"type": "string"},
                                "platform": {"type": "string"}
                            }
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "learn_retry_patterns",
                "description": "Analyze recent commands to learn retry patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "find_similar_sessions",
                "description": "Find sessions similar to current context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query describing current task or context"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current context (project type, technologies, etc.)",
                            "properties": {
                                "project_type": {"type": "string"},
                                "technologies": {"type": "array", "items": {"type": "string"}},
                                "project_path": {"type": "string"}
                            }
                        },
                        "time_range_days": {
                            "type": "integer",
                            "description": "Limit search to recent days",
                            "minimum": 1,
                            "maximum": 365
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_continuation_context",
                "description": "Get relevant context for continuing work on a task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "current_task": {
                            "type": "string",
                            "description": "Description of current task"
                        },
                        "project_context": {
                            "type": "object",
                            "description": "Current project context",
                            "properties": {
                                "project_type": {"type": "string"},
                                "project_path": {"type": "string"},
                                "technologies": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "required": ["current_task"]
                }
            },
            {
                "name": "suggest_workflow_optimizations",
                "description": "Suggest workflow optimizations based on historical data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "current_workflow": {
                            "type": "array",
                            "description": "List of current workflow steps",
                            "items": {"type": "string"}
                        },
                        "session_context": {
                            "type": "object",
                            "description": "Current session context"
                        }
                    },
                    "required": ["current_workflow"]
                }
            },
            {
                "name": "get_learning_progression",
                "description": "Track learning progression on a specific topic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to track learning progression for"
                        },
                        "time_range_days": {
                            "type": "integer",
                            "description": "Time range to analyze in days",
                            "minimum": 1,
                            "maximum": 365,
                            "default": 180
                        }
                    },
                    "required": ["topic"]
                }
            }
        ]