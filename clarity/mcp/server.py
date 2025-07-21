"""
MCP server implementation for the memory system.
"""

import json
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from clarity.mcp.tools import MemoryToolDefinitions
from clarity.domains.manager import MemoryDomainManager
from clarity.autocode.server import AutoCodeServerExtension
from clarity.autocode.hooks import AutoCodeHooks
from clarity.autocode.hook_manager import HookManager, HookRegistry


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
        self.app = FastMCP("mcp-alunai-clarity-server")
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

        @self.app.tool()
        async def qdrant_performance_stats() -> str:
            """Get detailed Qdrant performance statistics and optimization recommendations."""
            try:
                # Get comprehensive memory stats from persistence domain
                stats = await self.domain_manager.persistence_domain.get_memory_stats()
                
                # Calculate performance metrics
                total_memories = stats.get("total_memories", 0)
                indexed_memories = stats.get("indexed_memories", 0)
                indexing_ratio = (indexed_memories / max(total_memories, 1)) * 100
                
                # Determine performance rating
                performance_rating = "excellent"
                recommendations = []
                
                if indexing_ratio < 95:
                    performance_rating = "needs_optimization"
                    recommendations.append("Run collection optimization to improve indexing ratio")
                
                if stats.get("disk_data_size", 0) > 1024 * 1024 * 1024:  # > 1GB
                    recommendations.append("Consider archiving old memories to reduce disk usage")
                
                if total_memories > 100000:
                    recommendations.append("Performance may benefit from collection sharding")
                
                # Get memory type distribution
                memory_types = stats.get("memory_types", {})
                most_common_type = max(memory_types.items(), key=lambda x: x[1]) if memory_types else ("none", 0)
                
                performance_stats = {
                    "total_memories": total_memories,
                    "indexed_memories": indexed_memories,
                    "indexing_ratio_percent": round(indexing_ratio, 2),
                    "performance_rating": performance_rating,
                    "disk_size_mb": round(stats.get("disk_data_size", 0) / (1024 * 1024), 2),
                    "ram_size_mb": round(stats.get("ram_data_size", 0) / (1024 * 1024), 2),
                    "collection_status": stats.get("collection_status", "unknown"),
                    "most_common_memory_type": most_common_type[0],
                    "memory_type_distribution": memory_types,
                    "memory_tiers": stats.get("memory_tiers", {}),
                    "recommendations": recommendations,
                    "estimated_search_time_ms": self._estimate_search_time(total_memories),
                }
                
                return json.dumps({
                    "success": True,
                    "performance_stats": performance_stats,
                    "raw_qdrant_stats": stats
                })
                
            except Exception as e:
                logger.error(f"Error in qdrant_performance_stats: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def optimize_qdrant_collection() -> str:
            """Optimize the Qdrant collection for better performance."""
            try:
                # Trigger optimization
                success = await self.domain_manager.persistence_domain.optimize_collection()
                
                if success:
                    # Get updated stats after optimization
                    stats = await self.domain_manager.persistence_domain.get_memory_stats()
                    
                    return json.dumps({
                        "success": True,
                        "message": "Collection optimization triggered successfully",
                        "updated_stats": {
                            "total_memories": stats.get("total_memories", 0),
                            "indexed_memories": stats.get("indexed_memories", 0),
                            "collection_status": stats.get("collection_status", "unknown"),
                            "optimizer_status": stats.get("optimizer_status", "unknown"),
                        }
                    })
                else:
                    return json.dumps({
                        "success": False,
                        "error": "Failed to trigger collection optimization"
                    })
                    
            except Exception as e:
                logger.error(f"Error in optimize_qdrant_collection: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        # Proactive Memory Consultation Tools
        @self.app.tool()
        async def suggest_memory_queries(
            current_context: Dict[str, Any],
            task_description: Optional[str] = None,
            limit: int = 3
        ) -> str:
            """Suggest memory queries that Claude should execute based on current context."""
            try:
                suggestions = []
                
                # Extract keywords from context for query suggestions
                keywords = []
                if task_description:
                    keywords.extend(task_description.split()[:5])
                
                # Add context-based keywords
                for key, value in current_context.items():
                    if isinstance(value, str) and len(value) < 50:
                        keywords.append(value)
                    elif key in ["file_path", "project_path", "command", "intent"]:
                        keywords.append(str(value))
                
                # Generate query suggestions based on context
                if "file_path" in current_context:
                    file_path = current_context["file_path"]
                    suggestions.append({
                        "query": f"file {file_path}",
                        "reason": f"Check for previous work on {file_path}",
                        "types": ["code", "project_pattern"]
                    })
                
                if "command" in current_context:
                    command = current_context["command"]
                    suggestions.append({
                        "query": f"command {command}",
                        "reason": f"Find similar command patterns for {command}",
                        "types": ["bash_execution", "command_pattern"]
                    })
                
                if "project_path" in current_context:
                    project_path = current_context["project_path"]
                    suggestions.append({
                        "query": f"project {project_path}",
                        "reason": f"Retrieve project context for {project_path}",
                        "types": ["project_pattern", "session_summary"]
                    })
                
                if task_description:
                    suggestions.append({
                        "query": task_description[:100],
                        "reason": f"Find memories related to similar tasks",
                        "types": ["session_summary", "reflection"]
                    })
                
                # Limit to requested number of suggestions
                suggestions = suggestions[:limit]
                
                return json.dumps({
                    "success": True,
                    "current_context": current_context,
                    "task_description": task_description,
                    "suggestions": suggestions,
                    "total_suggestions": len(suggestions)
                })
            except Exception as e:
                logger.error(f"Error in suggest_memory_queries: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def check_relevant_memories(
            context: Dict[str, Any],
            auto_execute: bool = True,
            min_similarity: float = 0.6
        ) -> str:
            """Automatically check for and return relevant memories based on current context."""
            try:
                relevant_memories = []
                
                # Generate context-aware queries
                queries = []
                
                # File-based queries
                if "file_path" in context:
                    file_path = context["file_path"]
                    from pathlib import Path
                    path = Path(file_path)
                    queries.append(f"file {path.name} {path.suffix}")
                
                # Command-based queries  
                if "command" in context:
                    command = context["command"]
                    queries.append(f"command {command.split()[0]}")
                
                # Project-based queries
                if "project_path" in context:
                    project_path = context["project_path"]
                    queries.append(f"project {Path(project_path).name}")
                
                # Task-based queries
                if "task" in context:
                    task = context["task"]
                    queries.append(task[:100])
                
                # Directory-based queries
                if "directory" in context:
                    directory = context["directory"]
                    queries.append(f"directory {Path(directory).name}")
                
                # Execute memory retrieval for each query
                if auto_execute:
                    for query in queries:
                        memories = await self.domain_manager.retrieve_memories(
                            query=query,
                            limit=3,
                            min_similarity=min_similarity,
                            include_metadata=True
                        )
                        
                        if memories:
                            relevant_memories.extend([{
                                "query": query,
                                "memories": memories
                            }])
                
                return json.dumps({
                    "success": True,
                    "context": context,
                    "queries_generated": queries,
                    "relevant_memories": relevant_memories,
                    "total_memories": sum(len(rm["memories"]) for rm in relevant_memories),
                    "auto_executed": auto_execute
                })
            except Exception as e:
                logger.error(f"Error in check_relevant_memories: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def configure_proactive_memory(
            enabled: bool = True,
            file_access_triggers: bool = True,
            tool_execution_triggers: bool = True,
            context_change_triggers: bool = True,
            min_similarity_threshold: float = 0.6,
            max_memories_per_trigger: int = 3,
            auto_present_memories: bool = True
        ) -> str:
            """
            Configure proactive memory checking behavior.
            
            This allows customization of when and how Claude automatically
            receives relevant memory context during workflow.
            """
            try:
                config = {
                    "proactive_memory": {
                        "enabled": enabled,
                        "triggers": {
                            "file_access": file_access_triggers,
                            "tool_execution": tool_execution_triggers,
                            "context_change": context_change_triggers
                        },
                        "similarity_threshold": min_similarity_threshold,
                        "max_memories_per_trigger": max_memories_per_trigger,
                        "auto_present": auto_present_memories,
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }
                
                # Store configuration in memory system
                config_memory = {
                    "id": "proactive_memory_config",
                    "type": "system_configuration",
                    "content": config,
                    "importance": 1.0,
                    "metadata": {
                        "config_type": "proactive_memory",
                        "auto_generated": False
                    }
                }
                
                await self.domain_manager.store_memory(config_memory, "long_term")
                
                # Update hook manager if available
                if hasattr(self.domain_manager, 'autocode_domain') and self.domain_manager.autocode_domain:
                    autocode_domain = self.domain_manager.autocode_domain
                    if hasattr(autocode_domain, 'hook_manager') and autocode_domain.hook_manager:
                        autocode_domain.hook_manager.proactive_config = config["proactive_memory"]
                
                return json.dumps({
                    "success": True,
                    "message": "Proactive memory configuration updated successfully",
                    "config": config["proactive_memory"]
                })
                
            except Exception as e:
                logger.error(f"Error configuring proactive memory: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })

        @self.app.tool()
        async def get_proactive_memory_stats() -> str:
            """Get statistics about proactive memory usage and effectiveness."""
            try:
                # Retrieve proactive memory analytics
                analytics_memories = await self.domain_manager.retrieve_memories(
                    query="memory_usage_analytics proactive",
                    memory_types=["memory_usage_analytics"],
                    limit=100,
                    min_similarity=0.3
                )
                
                # Retrieve presented memories
                presented_memories = await self.domain_manager.retrieve_memories(
                    query="proactive_memory auto_presented",
                    memory_types=["proactive_memory"],
                    limit=50,
                    min_similarity=0.3
                )
                
                stats = {
                    "total_proactive_presentations": len(presented_memories),
                    "analytics_entries": len(analytics_memories),
                    "most_common_triggers": self._analyze_trigger_patterns(presented_memories),
                    "memory_effectiveness": self._calculate_memory_effectiveness(analytics_memories),
                    "recent_activity": len([m for m in presented_memories if self._is_recent(m, hours=24)])
                }
                
                return json.dumps({
                    "success": True,
                    "stats": stats
                })
                
            except Exception as e:
                logger.error(f"Error getting proactive memory stats: {str(e)}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        logger.info("AutoCode tools registered successfully")
    
    def _analyze_trigger_patterns(self, presented_memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze which triggers are most commonly used for proactive memory presentation."""
        trigger_counts = {}
        for memory in presented_memories:
            metadata = memory.get("metadata", {})
            trigger = metadata.get("trigger_context", "unknown")
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        return dict(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_memory_effectiveness(self, analytics_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate effectiveness metrics for proactive memory presentations."""
        if not analytics_memories:
            return {"total_presentations": 0, "effectiveness_score": 0.0}
        
        total_presentations = len(analytics_memories)
        # This is a placeholder - in a real implementation, you'd track user engagement
        # with presented memories to calculate true effectiveness
        effectiveness_score = min(0.8, total_presentations / 100.0)  # Simple heuristic
        
        return {
            "total_presentations": total_presentations,
            "effectiveness_score": round(effectiveness_score, 2),
            "average_memories_per_presentation": round(
                sum(content.get("memory_count", 0) for memory in analytics_memories 
                    for content in [memory.get("content", {})] if isinstance(content, dict)) / max(total_presentations, 1), 1
            )
        }
    
    def _is_recent(self, memory: Dict[str, Any], hours: int = 24) -> bool:
        """Check if a memory was created within the specified number of hours."""
        try:
            created_at = memory.get("created_at")
            if not created_at:
                return False
            
            from datetime import datetime, timedelta
            memory_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            return memory_time >= cutoff_time
        except:
            return False

    def _estimate_search_time(self, total_memories: int) -> float:
        """Estimate search time based on collection size."""
        if total_memories < 1000:
            return round(0.1 + (total_memories * 0.0001), 2)  # Very fast for small collections
        elif total_memories < 10000:
            return round(0.5 + (total_memories * 0.00005), 2)  # Sub-millisecond for medium
        elif total_memories < 100000:
            return round(1.0 + (total_memories * 0.00001), 2)  # ~1-2ms for large
        else:
            return round(2.0 + (total_memories * 0.000005), 2)  # ~2-5ms for very large
    
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
