"""
MCP Tool Awareness Hooks.

This module provides hooks that make Claude proactively aware of and suggest
MCP tools instead of using indirect methods like scripts or manual processes.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
from loguru import logger

from ..mcp.tool_indexer import MCPToolIndexer, MCPToolSuggester


class MCPAwarenessHooks:
    """
    Hooks that provide proactive MCP tool awareness and suggestions.
    
    These hooks integrate with the existing hook system to make Claude
    naturally prefer MCP tools over indirect approaches.
    """
    
    def __init__(self, domain_manager):
        """
        Initialize MCP awareness hooks.
        
        Args:
            domain_manager: Memory domain manager
        """
        self.domain_manager = domain_manager
        self.tool_indexer = MCPToolIndexer(domain_manager)
        self.tool_suggester = MCPToolSuggester(self.tool_indexer)
        self.indexed_tools: Dict[str, Any] = {}
        
        # Track suggestion patterns to avoid repetition
        self.recent_suggestions: List[str] = []
        self.max_recent_suggestions = 10
    
    async def initialize(self) -> None:
        """Initialize the MCP awareness system."""
        logger.info("Initializing MCP tool awareness hooks...")
        
        try:
            # Start MCP tool indexing in background (non-blocking)
            import asyncio
            logger.info("Starting background MCP tool indexing...")
            asyncio.create_task(self._background_index_tools())
            
            # Initialize with empty tools for now - will be populated by background task
            self.indexed_tools = {}
            
            logger.info("MCP awareness initialized (background indexing started)")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP awareness: {e}")
            self.indexed_tools = {}
    
    async def _background_index_tools(self) -> None:
        """Index MCP tools in background without blocking initialization."""
        try:
            logger.info("Background MCP tool indexing started...")
            
            # Index available MCP tools
            indexed_tools = await self.tool_indexer.discover_and_index_tools()
            self.indexed_tools = indexed_tools
            
            # Store system-level memory about MCP availability
            await self._store_mcp_system_memory()
            
            logger.info(f"Background MCP indexing completed with {len(self.indexed_tools)} tools")
            
        except Exception as e:
            logger.error(f"Background MCP tool indexing failed: {e}")
            self.indexed_tools = {}
    
    async def on_user_request(self, request: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Hook called when user makes a request.
        
        Proactively suggests MCP tools if the request might lead to indirect methods.
        
        Args:
            request: User's request
            context: Request context
            
        Returns:
            Suggestion message or None
        """
        try:
            # Analyze request for MCP tool opportunities
            suggestion = await self.tool_suggester.analyze_and_suggest(request)
            
            if suggestion and not self._is_recent_suggestion(suggestion):
                self._add_recent_suggestion(suggestion)
                return suggestion
            
        except Exception as e:
            logger.warning(f"Error in MCP suggestion analysis: {e}")
        
        return None
    
    async def on_tool_about_to_execute(self, tool_name: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Hook called before a tool is about to execute.
        
        Suggests MCP alternatives if Claude is about to use indirect methods.
        
        Args:
            tool_name: Name of the tool about to execute
            context: Execution context
            
        Returns:
            Alternative suggestion or None
        """
        try:
            # Check if this is an indirect method that has MCP alternatives
            if self._is_indirect_tool(tool_name, context):
                alternatives = await self._find_mcp_alternatives(tool_name, context)
                
                if alternatives:
                    return self._format_alternative_suggestion(tool_name, alternatives)
            
        except Exception as e:
            logger.warning(f"Error suggesting MCP alternatives: {e}")
        
        return None
    
    async def on_context_change(self, new_context: Dict[str, Any]) -> Optional[str]:
        """
        Hook called when context changes (e.g., new file opened, directory changed).
        
        Suggests relevant MCP tools based on the new context.
        
        Args:
            new_context: New context information
            
        Returns:
            Context-aware suggestions or None
        """
        try:
            # Generate context-aware suggestions
            suggestions = await self._get_context_aware_suggestions(new_context)
            
            if suggestions:
                return self._format_context_suggestions(suggestions)
            
        except Exception as e:
            logger.warning(f"Error generating context-aware suggestions: {e}")
        
        return None
    
    async def on_error_occurred(self, error: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Hook called when an error occurs.
        
        Suggests MCP tools that might help resolve the error.
        
        Args:
            error: Error message
            context: Error context
            
        Returns:
            Error resolution suggestions or None
        """
        try:
            # Check if error might be resolved with MCP tools
            if self._is_mcp_resolvable_error(error):
                suggestions = await self._suggest_error_resolution_tools(error, context)
                
                if suggestions:
                    return self._format_error_resolution_suggestions(suggestions)
            
        except Exception as e:
            logger.warning(f"Error suggesting error resolution tools: {e}")
        
        return None
    
    async def get_enhanced_system_prompt(self) -> str:
        """
        Generate an enhanced system prompt that includes MCP tool awareness.
        
        Returns:
            Enhanced system prompt with MCP tool information
        """
        if not self.indexed_tools:
            return ""
        
        # Group tools by category
        tools_by_category = {}
        for tool_info in self.indexed_tools.values():
            category = tool_info.category
            if category not in tools_by_category:
                tools_by_category[category] = []
            tools_by_category[category].append(tool_info)
        
        # Generate prompt sections
        prompt_sections = [
            "🔧 **MCP TOOLS AVAILABLE** - Use these FIRST before scripts or indirect methods:\n"
        ]
        
        for category, tools in tools_by_category.items():
            prompt_sections.append(f"\n**{category.replace('_', ' ').title()}:**")
            for tool in tools:
                prompt_sections.append(f"- `{tool.name}`: {tool.description}")
        
        prompt_sections.extend([
            "\n**🎯 IMPORTANT MCP USAGE GUIDELINES:**",
            "- ALWAYS check if an MCP tool can handle the task before writing scripts",
            "- Database queries → Use MCP postgres tool, NOT psql scripts",
            "- Web automation → Use MCP playwright tool, NOT manual browsing", 
            "- Memory operations → Use MCP alunai-memory tools proactively",
            "- File operations → Use MCP file tools when available",
            "\n**💡 PROACTIVE BEHAVIOR:**",
            "- Before each action, consider: 'Is there an MCP tool for this?'",
            "- Suggest MCP alternatives when users ask for indirect approaches",
            "- Use memory tools to enhance responses with relevant context"
        ])
        
        return "\n".join(prompt_sections)
    
    def _is_indirect_tool(self, tool_name: str, context: Dict[str, Any]) -> bool:
        """Check if a tool represents an indirect approach that has MCP alternatives."""
        indirect_tools = {
            'bash', 'shell', 'exec', 'run_command', 'write_file', 'create_script'
        }
        
        if tool_name.lower() in indirect_tools:
            # Check if the command being run has MCP alternatives
            command = context.get('command', '').lower()
            
            # Database commands
            if any(cmd in command for cmd in ['psql', 'mysql', 'sqlite3']):
                return True
            
            # File manipulation that could use MCP
            if any(cmd in command for cmd in ['curl', 'wget', 'python -c', 'node -e']):
                return True
        
        return False
    
    async def _find_mcp_alternatives(self, tool_name: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find MCP alternatives for indirect tools."""
        command = context.get('command', '').lower()
        
        # Database alternatives
        if any(cmd in command for cmd in ['psql', 'mysql', 'sqlite3']):
            return await self.tool_indexer.suggest_tools_for_intent("database query sql", limit=2)
        
        # Web/HTTP alternatives
        if any(cmd in command for cmd in ['curl', 'wget']):
            return await self.tool_indexer.suggest_tools_for_intent("web request http api", limit=2)
        
        # File alternatives
        if 'python -c' in command or 'node -e' in command:
            return await self.tool_indexer.suggest_tools_for_intent("file operations read write", limit=2)
        
        return []
    
    def _format_alternative_suggestion(self, tool_name: str, alternatives: List[Dict[str, Any]]) -> str:
        """Format alternative suggestions."""
        if not alternatives:
            return ""
        
        message = f"⚡ **Before using {tool_name}**, consider these MCP alternatives:\n\n"
        
        for alt in alternatives:
            message += f"**{alt['tool_name']}**: {alt['description']}\n"
            message += f"💡 {alt['usage_hint']}\n\n"
        
        message += "MCP tools are often more reliable and integrated than shell commands!"
        return message
    
    async def _get_context_aware_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggestions based on current context."""
        suggestions = []
        
        # File context
        if 'file_path' in context:
            file_path = context['file_path']
            if file_path.endswith('.sql'):
                suggestions.extend(await self.tool_indexer.suggest_tools_for_intent("database sql", limit=1))
            elif file_path.endswith(('.py', '.js', '.ts')):
                suggestions.extend(await self.tool_indexer.suggest_tools_for_intent("code development", limit=1))
        
        # Directory context
        if 'directory' in context:
            directory = context['directory'].lower()
            if any(word in directory for word in ['db', 'database', 'sql']):
                suggestions.extend(await self.tool_indexer.suggest_tools_for_intent("database", limit=1))
        
        # Task context
        if 'task' in context:
            task = context['task']
            suggestions.extend(await self.tool_indexer.suggest_tools_for_intent(task, limit=2))
        
        return suggestions
    
    def _format_context_suggestions(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format context-aware suggestions."""
        if not suggestions:
            return ""
        
        message = "💡 **Context-Aware MCP Tools**: Based on your current context, these tools might be helpful:\n\n"
        
        for suggestion in suggestions:
            message += f"**{suggestion['tool_name']}**: {suggestion['description']}\n"
        
        return message
    
    def _is_mcp_resolvable_error(self, error: str) -> bool:
        """Check if an error might be resolvable with MCP tools."""
        error_lower = error.lower()
        
        # Database connection errors
        if any(word in error_lower for word in ['connection refused', 'database', 'sql']):
            return True
        
        # File not found errors
        if 'no such file' in error_lower or 'file not found' in error_lower:
            return True
        
        # Permission errors
        if 'permission denied' in error_lower:
            return True
        
        return False
    
    async def _suggest_error_resolution_tools(self, error: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest tools that might help resolve an error."""
        error_lower = error.lower()
        
        if any(word in error_lower for word in ['database', 'sql', 'connection']):
            return await self.tool_indexer.suggest_tools_for_intent("database connection", limit=1)
        
        if 'file' in error_lower:
            return await self.tool_indexer.suggest_tools_for_intent("file operations", limit=1)
        
        return []
    
    def _format_error_resolution_suggestions(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format error resolution suggestions."""
        if not suggestions:
            return ""
        
        message = "🔧 **Error Resolution**: These MCP tools might help resolve the error:\n\n"
        
        for suggestion in suggestions:
            message += f"**{suggestion['tool_name']}**: {suggestion['description']}\n"
        
        return message
    
    async def _store_mcp_system_memory(self) -> None:
        """Store system-level memory about MCP tool availability."""
        system_memory = {
            "mcp_system_info": True,
            "available_tools_count": len(self.indexed_tools),
            "tool_categories": list(set(tool.category for tool in self.indexed_tools.values())),
            "servers": list(set(tool.server_name for tool in self.indexed_tools.values())),
            "guidelines": [
                "Always prefer MCP tools over scripts or indirect methods",
                "Check for MCP alternatives before using shell commands",
                "Use memory tools proactively to enhance responses",
                "Suggest MCP tools when users ask for indirect approaches"
            ]
        }
        
        await self.domain_manager.store_memory(
            memory_type="mcp_system_info",
            content=system_memory,
            importance=1.0,  # Maximum importance for system behavior
            metadata={
                "category": "system_behavior",
                "auto_generated": True,
                "system_level": True
            },
            context={
                "purpose": "mcp_awareness_system",
                "behavior_guidance": True
            }
        )
    
    def _is_recent_suggestion(self, suggestion: str) -> bool:
        """Check if this suggestion was made recently."""
        # Simple check based on first few words
        suggestion_key = ' '.join(suggestion.split()[:5])
        return suggestion_key in self.recent_suggestions
    
    def _add_recent_suggestion(self, suggestion: str) -> None:
        """Add suggestion to recent suggestions list."""
        suggestion_key = ' '.join(suggestion.split()[:5])
        self.recent_suggestions.append(suggestion_key)
        
        # Keep only recent suggestions
        if len(self.recent_suggestions) > self.max_recent_suggestions:
            self.recent_suggestions = self.recent_suggestions[-self.max_recent_suggestions:]