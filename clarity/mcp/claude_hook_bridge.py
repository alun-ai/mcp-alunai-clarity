"""
Claude Code Hook Integration Bridge.

This module provides a bridge between Claude Code's native hook system and 
our MCP discovery and learning system, enabling pattern learning and proactive
suggestions based on actual tool usage.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from .tool_indexer import MCPToolIndexer, MCPToolInfo


@dataclass
class HookExecutionContext:
    """Context information from a Claude Code hook execution."""
    hook_type: str  # PreToolUse, PostToolUse, UserPromptSubmit
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    user_prompt: Optional[str]
    exit_code: int
    stdout: str
    stderr: str
    timestamp: datetime
    session_id: Optional[str]


class MCPHookLearningSystem:
    """
    Learns MCP usage patterns from Claude Code hooks.
    
    This system bridges Claude Code's hook system with our MCP discovery
    and learning capabilities, enabling continuous improvement of suggestions.
    """
    
    def __init__(self, domain_manager, tool_indexer: Optional[MCPToolIndexer] = None):
        """
        Initialize the hook learning system.
        
        Args:
            domain_manager: Memory domain manager for storing learned patterns
            tool_indexer: MCP tool indexer for tool discovery and suggestion
        """
        self.domain_manager = domain_manager
        self.tool_indexer = tool_indexer or MCPToolIndexer(domain_manager)
        
        # Track learning patterns
        self.learned_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.usage_statistics: Dict[str, int] = {}
        
        # Configuration for hook integration
        self.hook_config_path = "~/.claude-code/hooks/mcp_learning_hooks.json"
        self.learning_data_path = "~/.cache/alunai-clarity/mcp_hook_learning.json"
    
    async def initialize(self) -> None:
        """Initialize the hook learning system."""
        logger.info("Initializing MCP hook learning system...")
        
        try:
            # Ensure cache directory exists
            cache_dir = Path(self.learning_data_path).parent.expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing learning data
            await self._load_learning_data()
            
            # Set up Claude Code hook configuration
            await self._setup_claude_hook_integration()
            
            logger.info("MCP hook learning system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hook learning system: {e}")
    
    async def analyze_pre_tool_usage(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Analyze tool usage before execution for MCP opportunities.
        
        Args:
            tool_name: Name of the tool about to be executed
            tool_args: Arguments passed to the tool
            context: Additional context information
            
        Returns:
            Optional suggestion message for MCP alternatives
        """
        try:
            logger.debug(f"Analyzing pre-tool usage: {tool_name}")
            
            # Check if this is an indirect method that has MCP alternatives
            if self._is_indirect_tool(tool_name, tool_args):
                alternatives = await self._find_mcp_alternatives(tool_name, tool_args)
                
                if alternatives:
                    suggestion = await self._generate_mcp_suggestion(
                        tool_name, tool_args, alternatives
                    )
                    
                    # Store the analysis as a learning pattern
                    await self._store_pre_usage_pattern(
                        tool_name, tool_args, alternatives, context
                    )
                    
                    return suggestion
            
        except Exception as e:
            logger.warning(f"Error in pre-tool usage analysis: {e}")
        
        return None
    
    async def learn_from_successful_usage(
        self, 
        tool_sequence: List[str], 
        context: Dict[str, Any]
    ) -> None:
        """
        Learn from successful MCP tool usage patterns.
        
        Args:
            tool_sequence: Sequence of tools used successfully
            context: Context information about the usage
        """
        try:
            logger.debug(f"Learning from successful usage: {tool_sequence}")
            
            # Extract MCP tools from the sequence
            mcp_tools = [tool for tool in tool_sequence if self._is_mcp_tool(tool)]
            
            if mcp_tools:
                # Store successful pattern
                pattern = {
                    "tool_sequence": tool_sequence,
                    "mcp_tools_used": mcp_tools,
                    "context": context,
                    "success_timestamp": datetime.now().isoformat(),
                    "pattern_strength": self._calculate_pattern_strength(tool_sequence)
                }
                
                await self._store_successful_pattern(pattern)
                
                # Update usage statistics
                for tool in mcp_tools:
                    self.usage_statistics[tool] = self.usage_statistics.get(tool, 0) + 1
                
                # Store as memory for future retrieval
                await self._store_pattern_as_memory(pattern)
        
        except Exception as e:
            logger.warning(f"Error learning from successful usage: {e}")
    
    async def generate_hook_config(self) -> Dict[str, Any]:
        """
        Generate Claude Code hook configuration for MCP learning.
        
        Returns:
            Hook configuration dictionary
        """
        script_path = str(Path(__file__).parent / "hook_scripts" / "mcp_learning_hook.py")
        
        hook_config = {
            "hooks": {
                "PreToolUse": [{
                    "matcher": "bash|shell|exec|python|node|curl|wget|git",
                    "hooks": [{
                        "type": "command",
                        "command": f"{sys.executable} {script_path} pre-tool-use",
                        "exit_codes": [0, 1]  # Continue execution regardless
                    }]
                }],
                "PostToolUse": [{
                    "matcher": "*",
                    "hooks": [{
                        "type": "command", 
                        "command": f"{sys.executable} {script_path} post-tool-use",
                        "exit_codes": [0]
                    }]
                }],
                "UserPromptSubmit": [{
                    "matcher": "*",
                    "hooks": [{
                        "type": "command",
                        "command": f"{sys.executable} {script_path} prompt-submit",
                        "exit_codes": [0]
                    }]
                }]
            }
        }
        
        return hook_config
    
    def _is_indirect_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """Check if a tool represents an indirect method that might have MCP alternatives."""
        indirect_patterns = {
            "bash": ["psql", "mysql", "curl", "wget", "git"],
            "shell": ["psql", "mysql", "curl", "wget", "git"],
            "exec": ["database", "api", "web"],
            "python": ["requests", "psycopg2", "pymongo"],
            "node": ["axios", "fetch", "pg"],
            "curl": True,
            "wget": True
        }
        
        if tool_name in indirect_patterns:
            if isinstance(indirect_patterns[tool_name], bool):
                return indirect_patterns[tool_name]
            
            # Check if any indirect pattern keywords appear in the arguments
            args_str = str(tool_args).lower()
            return any(pattern in args_str for pattern in indirect_patterns[tool_name])
        
        return False
    
    async def _find_mcp_alternatives(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> List[MCPToolInfo]:
        """Find MCP tool alternatives for indirect methods."""
        alternatives = []
        
        try:
            # Analyze the indirect tool usage to infer intent
            intent_keywords = self._extract_intent_keywords(tool_name, tool_args)
            
            # Search for relevant MCP tools
            for keyword in intent_keywords:
                matching_tools = await self._search_mcp_tools_by_keyword(keyword)
                alternatives.extend(matching_tools)
            
            # Remove duplicates
            seen_names = set()
            unique_alternatives = []
            for tool in alternatives:
                if tool.name not in seen_names:
                    unique_alternatives.append(tool)
                    seen_names.add(tool.name)
            
            return unique_alternatives
        
        except Exception as e:
            logger.warning(f"Error finding MCP alternatives: {e}")
            return []
    
    def _extract_intent_keywords(self, tool_name: str, tool_args: Dict[str, Any]) -> List[str]:
        """Extract intent keywords from tool usage."""
        keywords = []
        
        # Tool-specific keyword extraction
        if tool_name in ["bash", "shell", "exec"]:
            command = tool_args.get("command", "")
            if "psql" in command or "postgresql" in command:
                keywords.extend(["database", "postgres", "sql"])
            elif "mysql" in command:
                keywords.extend(["database", "mysql", "sql"])
            elif "curl" in command or "wget" in command:
                keywords.extend(["api", "web", "http"])
            elif "git" in command:
                keywords.extend(["version_control", "repository"])
        
        elif tool_name == "python":
            code = tool_args.get("code", "")
            if "requests" in code or "urllib" in code:
                keywords.extend(["api", "web", "http"])
            elif "psycopg2" in code or "pg" in code:
                keywords.extend(["database", "postgres"])
            elif "pymongo" in code:
                keywords.extend(["database", "mongodb"])
        
        # Add general keywords based on tool name
        general_keywords = {
            "curl": ["api", "web", "http"],
            "wget": ["web", "download", "http"],
            "git": ["version_control", "repository"],
            "docker": ["container", "deployment"],
            "npm": ["package_manager", "javascript"],
            "pip": ["package_manager", "python"]
        }
        
        if tool_name in general_keywords:
            keywords.extend(general_keywords[tool_name])
        
        return keywords
    
    async def _search_mcp_tools_by_keyword(self, keyword: str) -> List[MCPToolInfo]:
        """Search for MCP tools matching a keyword."""
        try:
            # Query the memory system for MCP tools
            memories = await self.domain_manager.retrieve_memory(
                query=f"MCP tool {keyword}",
                types=["mcp_tool"],
                limit=5
            )
            
            tools = []
            for memory in memories:
                content = memory.get('content', {})
                if content.get('tool_name'):
                    tool = MCPToolInfo(
                        name=content['tool_name'],
                        description=content.get('description', ''),
                        parameters=content.get('parameters', {}),
                        server_name=content.get('server_name', ''),
                        use_cases=content.get('use_cases', []),
                        keywords=set(content.get('keywords', [])),
                        category=content.get('category', '')
                    )
                    tools.append(tool)
            
            return tools
        
        except Exception as e:
            logger.warning(f"Error searching MCP tools by keyword {keyword}: {e}")
            return []
    
    def _is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool name represents an MCP tool."""
        mcp_indicators = [
            "mcp__",  # MCP tool prefix
            "_query", "_execute", "_store", "_retrieve",  # Common MCP tool suffixes
            "postgres", "playwright", "memory", "filesystem"  # Known MCP server types
        ]
        
        return any(indicator in tool_name.lower() for indicator in mcp_indicators)
    
    def _calculate_pattern_strength(self, tool_sequence: List[str]) -> float:
        """Calculate the strength of a usage pattern."""
        mcp_tools = sum(1 for tool in tool_sequence if self._is_mcp_tool(tool))
        total_tools = len(tool_sequence)
        
        if total_tools == 0:
            return 0.0
        
        # Higher strength for sequences with more MCP tools
        mcp_ratio = mcp_tools / total_tools
        
        # Bonus for longer sequences (more context)
        length_bonus = min(total_tools / 10, 0.3)
        
        return min(mcp_ratio + length_bonus, 1.0)
    
    async def _generate_mcp_suggestion(
        self,
        tool_name: str,
        tool_args: Dict[str, Any], 
        alternatives: List[MCPToolInfo]
    ) -> str:
        """Generate a suggestion message for MCP alternatives."""
        if not alternatives:
            return ""
        
        suggestion_lines = [
            f"💡 MCP Alternative Available: Instead of using {tool_name}, consider these MCP tools:",
            ""
        ]
        
        for tool in alternatives[:3]:  # Show top 3 alternatives
            suggestion_lines.extend([
                f"• **{tool.name}**: {tool.description}",
                f"  Use cases: {', '.join(tool.use_cases[:2])}",
                ""
            ])
        
        suggestion_lines.append("MCP tools provide direct integration and better error handling.")
        
        return "\n".join(suggestion_lines)
    
    async def _store_pre_usage_pattern(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        alternatives: List[MCPToolInfo],
        context: Dict[str, Any]
    ) -> None:
        """Store pre-usage analysis pattern."""
        pattern = {
            "type": "pre_usage_analysis",
            "indirect_tool": tool_name,
            "tool_args": tool_args,
            "mcp_alternatives": [tool.name for tool in alternatives],
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in memory
        await self.domain_manager.store_memory(
            memory_type="mcp_learning_pattern",
            content=pattern,
            importance=0.7,
            metadata={
                "category": "pre_usage_analysis",
                "tool_name": tool_name,
                "auto_generated": True
            }
        )
    
    async def _store_successful_pattern(self, pattern: Dict[str, Any]) -> None:
        """Store successful usage pattern."""
        await self.domain_manager.store_memory(
            memory_type="mcp_learning_pattern",
            content=pattern,
            importance=0.8,
            metadata={
                "category": "successful_usage",
                "pattern_strength": pattern.get("pattern_strength", 0.0),
                "auto_generated": True
            }
        )
    
    async def _store_pattern_as_memory(self, pattern: Dict[str, Any]) -> None:
        """Store pattern as searchable memory."""
        memory_content = {
            "pattern_type": "mcp_usage_workflow",
            "workflow_description": f"Successful workflow using tools: {', '.join(pattern['mcp_tools_used'])}",
            "tools_used": pattern['tool_sequence'],
            "mcp_tools": pattern['mcp_tools_used'],
            "context": pattern['context'],
            "success_indicators": ["completed_successfully", "no_errors"],
            "when_to_suggest": self._generate_suggestion_criteria(pattern)
        }
        
        await self.domain_manager.store_memory(
            memory_type="mcp_workflow",
            content=memory_content,
            importance=pattern.get("pattern_strength", 0.5),
            metadata={
                "category": "workflow_pattern",
                "tool_count": len(pattern['mcp_tools_used']),
                "auto_learned": True
            }
        )
    
    def _generate_suggestion_criteria(self, pattern: Dict[str, Any]) -> List[str]:
        """Generate criteria for when to suggest this pattern."""
        criteria = []
        
        context = pattern.get('context', {})
        tools = pattern.get('mcp_tools_used', [])
        
        # Generate criteria based on context and tools
        if 'database' in str(tools).lower():
            criteria.append("when user needs database operations")
        if 'web' in str(tools).lower():
            criteria.append("when user needs web automation")
        if 'memory' in str(tools).lower():
            criteria.append("when user needs to store or retrieve information")
        
        return criteria
    
    async def _setup_claude_hook_integration(self) -> None:
        """Set up Claude Code hook integration configuration."""
        try:
            hook_config = await self.generate_hook_config()
            
            # Ensure hook directory exists
            config_path = Path(self.hook_config_path).expanduser()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write hook configuration
            with open(config_path, 'w') as f:
                json.dump(hook_config, f, indent=2)
            
            logger.info(f"Hook configuration written to {config_path}")
            
            # Create hook script directory
            script_dir = config_path.parent / "hook_scripts"
            script_dir.mkdir(exist_ok=True)
            
            # Create the hook script
            await self._create_hook_script(script_dir / "mcp_learning_hook.py")
            
        except Exception as e:
            logger.warning(f"Could not set up hook integration: {e}")
    
    async def _create_hook_script(self, script_path: Path) -> None:
        """Create the actual hook script that Claude Code will execute."""
        script_content = '''#!/usr/bin/env python3
"""
MCP Learning Hook Script.

This script is executed by Claude Code hooks to analyze tool usage
and learn MCP usage patterns.
"""

import json
import sys
import os
from datetime import datetime

def main():
    hook_type = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    
    # Read hook context from environment or stdin
    context = {
        "hook_type": hook_type,
        "timestamp": datetime.now().isoformat(),
        "tool_name": os.environ.get("CLAUDE_TOOL_NAME"),
        "tool_args": os.environ.get("CLAUDE_TOOL_ARGS"),
        "user_prompt": os.environ.get("CLAUDE_USER_PROMPT"),
        "session_id": os.environ.get("CLAUDE_SESSION_ID")
    }
    
    # Log the hook execution for analysis
    log_file = os.path.expanduser("~/.cache/alunai-clarity/mcp_hook_log.jsonl")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "a") as f:
        f.write(json.dumps(context) + "\\n")
    
    # For pre-tool-use, we could provide suggestions
    if hook_type == "pre-tool-use":
        # This would be where we analyze and potentially suggest MCP alternatives
        # For now, just log and continue
        print("MCP Learning: Analyzing tool usage for MCP opportunities...")
    
    # Exit with code 0 to continue normal execution
    sys.exit(0)

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Hook script created at {script_path}")
    
    async def _load_learning_data(self) -> None:
        """Load existing learning data from cache."""
        try:
            data_path = Path(self.learning_data_path).expanduser()
            if data_path.exists():
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                self.learned_patterns = data.get('learned_patterns', {})
                self.usage_statistics = data.get('usage_statistics', {})
                
                logger.debug(f"Loaded learning data: {len(self.learned_patterns)} patterns, {len(self.usage_statistics)} usage stats")
        
        except Exception as e:
            logger.warning(f"Could not load learning data: {e}")
            self.learned_patterns = {}
            self.usage_statistics = {}
    
    async def save_learning_data(self) -> None:
        """Save learning data to cache."""
        try:
            data_path = Path(self.learning_data_path).expanduser()
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "learned_patterns": self.learned_patterns,
                "usage_statistics": self.usage_statistics,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved learning data to {data_path}")
        
        except Exception as e:
            logger.warning(f"Could not save learning data: {e}")