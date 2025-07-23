"""Slash Command Discovery System for MCP Servers.

This module discovers and manages MCP-exposed slash commands by connecting
to MCP servers and retrieving their available prompts.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Import MCP client if available
try:
    from mcp import ClientSession, StdioClientTransport
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    ClientSession = None
    StdioClientTransport = None

if TYPE_CHECKING:
    from .tool_indexer import MCPToolIndexer

logger = logging.getLogger(__name__)


@dataclass
class SlashCommand:
    """Represents a discoverable MCP slash command."""
    command: str
    server_name: str
    prompt_name: str
    description: str
    arguments: List[Dict[str, Any]]
    usage_examples: List[str]
    category: str
    confidence: float
    last_discovered: str
    usage_count: int
    
    @classmethod
    def from_mcp_prompt(cls, server_name: str, prompt_data: Any) -> 'SlashCommand':
        """Create slash command from MCP prompt data."""
        command_name = f"/mcp__{server_name}__{prompt_data.name}"
        
        # Extract arguments if available
        arguments = []
        if hasattr(prompt_data, 'arguments') and prompt_data.arguments:
            for arg in prompt_data.arguments:
                arg_dict = {
                    'name': getattr(arg, 'name', 'unknown'),
                    'description': getattr(arg, 'description', ''),
                    'required': getattr(arg, 'required', False),
                    'type': getattr(arg, 'type', 'string')
                }
                arguments.append(arg_dict)
        
        return cls(
            command=command_name,
            server_name=server_name,
            prompt_name=prompt_data.name,
            description=getattr(prompt_data, 'description', ''),
            arguments=arguments,
            usage_examples=[],
            category='mcp_prompt',
            confidence=1.0,
            last_discovered=datetime.now(timezone.utc).isoformat(),
            usage_count=0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SlashCommand':
        """Create from dictionary."""
        return cls(**data)
    
    def generate_usage_example(self) -> str:
        """Generate a usage example for the command."""
        if self.arguments:
            arg_examples = []
            for arg in self.arguments[:3]:  # Show up to 3 arguments
                if arg['required']:
                    arg_examples.append(f"--{arg['name']} <{arg['type']}>")
                else:
                    arg_examples.append(f"[--{arg['name']} <{arg['type']}>]")
            
            args_str = " ".join(arg_examples)
            return f"{self.command} {args_str}"
        else:
            return self.command


@dataclass
class SlashCommandSuggestion:
    """Represents a contextual slash command suggestion."""
    command: SlashCommand
    relevance_score: float
    reason: str
    context_match: float
    suggested_arguments: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'command': self.command.to_dict(),
            'relevance_score': self.relevance_score,
            'reason': self.reason,
            'context_match': self.context_match,
            'suggested_arguments': self.suggested_arguments
        }


class SlashCommandDiscovery:
    """Discovers and manages MCP-exposed slash commands."""
    
    def __init__(self, tool_indexer: 'MCPToolIndexer'):
        """Initialize slash command discovery."""
        self.tool_indexer = tool_indexer
        self.slash_commands = {}
        self.discovery_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self._cache_timestamps = {}
        
        # Command categorization patterns
        self.command_categories = {
            'file_operations': ['file', 'read', 'write', 'directory', 'path'],
            'data_processing': ['data', 'process', 'transform', 'analyze', 'convert'],
            'web_requests': ['web', 'http', 'api', 'fetch', 'request'],
            'database': ['database', 'query', 'sql', 'table', 'record'],
            'git_operations': ['git', 'repo', 'commit', 'branch', 'merge'],
            'documentation': ['docs', 'help', 'manual', 'guide', 'readme'],
            'development': ['dev', 'build', 'test', 'deploy', 'lint'],
            'utility': ['util', 'tool', 'helper', 'format', 'validate']
        }
        
        self.usage_patterns = {}
        self.suggestion_history = []
    
    async def discover_slash_commands(self, server_name: str, server_config: Dict[str, Any]) -> List[SlashCommand]:
        """Discover slash commands from MCP server prompts."""
        if not MCP_CLIENT_AVAILABLE:
            logger.debug("MCP client not available for slash command discovery")
            return []
        
        # Check cache first
        cache_key = f"{server_name}:{hash(json.dumps(server_config, sort_keys=True))}"
        if self._is_cache_valid(cache_key):
            return self.discovery_cache[cache_key]
        
        try:
            commands = await self._connect_and_discover(server_name, server_config)
            
            # Cache the results
            self.discovery_cache[cache_key] = commands
            self._cache_timestamps[cache_key] = asyncio.get_event_loop().time()
            
            logger.info(f"Discovered {len(commands)} slash commands from {server_name}")
            return commands
            
        except Exception as e:
            logger.debug(f"Could not discover slash commands from {server_name}: {e}")
            return []
    
    async def _connect_and_discover(self, server_name: str, server_config: Dict[str, Any]) -> List[SlashCommand]:
        """Connect to MCP server and discover available prompts."""
        commands = []
        
        try:
            # Create transport based on server configuration
            transport = None
            
            if 'command' in server_config:
                # Stdio transport
                command = server_config['command']
                args = server_config.get('args', [])
                env = server_config.get('env', {})
                
                transport = StdioClientTransport(
                    command=command,
                    args=args,
                    env=env
                )
            
            elif 'module' in server_config:
                # Python module transport
                module = server_config['module']
                args = server_config.get('args', [])
                env = server_config.get('env', {})
                
                transport = StdioClientTransport(
                    command='python',
                    args=['-m', module] + args,
                    env=env
                )
            
            if not transport:
                logger.debug(f"Could not create transport for {server_name}")
                return []
            
            # Connect with timeout
            async with ClientSession(transport) as session:
                # Initialize the session
                await asyncio.wait_for(session.initialize(), timeout=10.0)
                
                # Get available prompts
                prompts_result = await asyncio.wait_for(
                    session.list_prompts(), timeout=5.0
                )
                
                # Convert prompts to slash commands
                for prompt in prompts_result.prompts:
                    try:
                        command = SlashCommand.from_mcp_prompt(server_name, prompt)
                        command.category = self._categorize_command(command)
                        commands.append(command)
                        
                        # Store in our registry
                        self.slash_commands[command.command] = command
                        
                    except Exception as e:
                        logger.debug(f"Could not process prompt {getattr(prompt, 'name', 'unknown')}: {e}")
                        continue
                
                return commands
                
        except asyncio.TimeoutError:
            logger.debug(f"Timeout connecting to {server_name}")
            return []
        except Exception as e:
            logger.debug(f"Error connecting to {server_name}: {e}")
            return []
    
    def _categorize_command(self, command: SlashCommand) -> str:
        """Categorize a command based on its name and description."""
        text_to_check = f"{command.prompt_name} {command.description}".lower()
        
        # Check each category
        category_scores = {}
        for category, keywords in self.command_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_to_check)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Return category with highest score
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'utility'  # Default category
    
    async def store_slash_commands(self, commands: List[SlashCommand]):
        """Store discovered slash commands as memories."""
        if not self.tool_indexer or not hasattr(self.tool_indexer, 'domain_manager'):
            logger.debug("No domain manager available for storing slash commands")
            return
        
        for cmd in commands:
            try:
                await self.tool_indexer.domain_manager.store_memory(
                    memory_type="mcp_slash_command",
                    content=json.dumps(cmd.to_dict()),
                    importance=0.8,
                    metadata={
                        "category": "mcp_commands",
                        "command_type": "slash_command", 
                        "server": cmd.server_name,
                        "prompt_name": cmd.prompt_name,
                        "command_category": cmd.category,
                        "discoverable": True,
                        "arguments_count": len(cmd.arguments),
                        "has_description": bool(cmd.description)
                    }
                )
                
            except Exception as e:
                logger.debug(f"Could not store slash command {cmd.command}: {e}")
    
    async def get_contextual_suggestions(self, prompt: str, context: Dict[str, Any] = None) -> List[SlashCommandSuggestion]:
        """Get contextual slash command suggestions based on user prompt."""
        context = context or {}
        suggestions = []
        
        # Analyze prompt to determine intent and relevant categories
        relevant_categories = self._analyze_prompt_categories(prompt)
        
        for command_name, command in self.slash_commands.items():
            # Calculate relevance score
            relevance_score = self._calculate_command_relevance(command, prompt, relevant_categories)
            
            if relevance_score > 0.3:  # Only suggest reasonably relevant commands
                # Generate suggested arguments
                suggested_args = self._suggest_command_arguments(command, prompt, context)
                
                # Create suggestion
                suggestion = SlashCommandSuggestion(
                    command=command,
                    relevance_score=relevance_score,
                    reason=self._generate_suggestion_reason(command, prompt, relevance_score),
                    context_match=self._calculate_context_match(command, context),
                    suggested_arguments=suggested_args
                )
                
                suggestions.append(suggestion)
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Store suggestion for learning
        await self._record_suggestion_context(prompt, suggestions[:3], context)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _analyze_prompt_categories(self, prompt: str) -> Dict[str, float]:
        """Analyze prompt to determine relevant command categories."""
        prompt_lower = prompt.lower()
        category_scores = {}
        
        for category, keywords in self.command_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1
            
            if score > 0:
                # Normalize score by number of keywords in category
                category_scores[category] = score / len(keywords)
        
        return category_scores
    
    def _calculate_command_relevance(self, command: SlashCommand, prompt: str, relevant_categories: Dict[str, float]) -> float:
        """Calculate how relevant a command is to the given prompt."""
        relevance = 0.0
        
        # Category match
        if command.category in relevant_categories:
            relevance += relevant_categories[command.category] * 0.5
        
        # Text similarity
        prompt_words = set(prompt.lower().split())
        command_words = set((command.prompt_name + ' ' + command.description).lower().split())
        
        if prompt_words and command_words:
            intersection = prompt_words.intersection(command_words)
            union = prompt_words.union(command_words)
            text_similarity = len(intersection) / len(union) if union else 0
            relevance += text_similarity * 0.3
        
        # Usage history boost
        if command.command in self.usage_patterns:
            usage_data = self.usage_patterns[command.command]
            success_rate = usage_data.get('success_count', 0) / max(1, usage_data.get('usage_count', 1))
            relevance += success_rate * 0.2
        
        return min(1.0, relevance)
    
    def _calculate_context_match(self, command: SlashCommand, context: Dict[str, Any]) -> float:
        """Calculate how well command matches the current context."""
        match_score = 0.0
        
        # Project type match
        project_type = context.get('project_type', '')
        if project_type:
            if any(project_type.lower() in word.lower() for word in [command.prompt_name, command.description]):
                match_score += 0.3
        
        # Available servers match
        available_servers = context.get('available_servers', [])
        if command.server_name in available_servers:
            match_score += 0.4
        
        # Recent tool usage match
        recent_tools = context.get('recent_tools_used', [])
        if any(tool.lower() in command.description.lower() for tool in recent_tools):
            match_score += 0.3
        
        return min(1.0, match_score)
    
    def _suggest_command_arguments(self, command: SlashCommand, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest appropriate arguments for a command based on context."""
        suggested_args = {}
        
        for arg in command.arguments:
            arg_name = arg['name']
            arg_type = arg.get('type', 'string')
            
            # Try to extract values from prompt
            if arg_type == 'string':
                # Look for quoted strings or file paths
                if 'file' in arg_name.lower() or 'path' in arg_name.lower():
                    # Look for path-like strings
                    import re
                    path_matches = re.findall(r'[\'"]([^\'\"]+)[\'"]', prompt)
                    for match in path_matches:
                        if '/' in match or '\\' in match or '.' in match:
                            suggested_args[arg_name] = match
                            break
                
                elif 'name' in arg_name.lower():
                    # Look for names (capitalized words)
                    import re
                    name_matches = re.findall(r'\b[A-Z][a-z]+\b', prompt)
                    if name_matches:
                        suggested_args[arg_name] = name_matches[0]
            
            elif arg_type in ['number', 'integer']:
                # Look for numbers in prompt
                import re
                number_matches = re.findall(r'\b\d+\b', prompt)
                if number_matches:
                    suggested_args[arg_name] = int(number_matches[0])
            
            elif arg_type == 'boolean':
                # Look for boolean indicators
                if any(word in prompt.lower() for word in ['yes', 'true', 'enable', 'on']):
                    suggested_args[arg_name] = True
                elif any(word in prompt.lower() for word in ['no', 'false', 'disable', 'off']):
                    suggested_args[arg_name] = False
        
        return suggested_args
    
    def _generate_suggestion_reason(self, command: SlashCommand, prompt: str, relevance_score: float) -> str:
        """Generate a reason for suggesting this command."""
        if relevance_score > 0.8:
            return f"Highly relevant: '{command.description}' matches your request perfectly"
        elif relevance_score > 0.6:
            return f"Good match: '{command.description}' can help with your task"
        elif relevance_score > 0.4:
            return f"Potentially useful: '{command.description}' might be relevant"
        else:
            return f"Available option: '{command.description}' from {command.server_name}"
    
    async def _record_suggestion_context(self, prompt: str, suggestions: List[SlashCommandSuggestion], context: Dict[str, Any]):
        """Record suggestion context for learning."""
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'prompt': prompt[:200],  # Truncate
            'suggestions_count': len(suggestions),
            'top_suggestion': suggestions[0].command.command if suggestions else None,
            'context': context,
            'categories_detected': list(self._analyze_prompt_categories(prompt).keys())
        }
        
        self.suggestion_history.append(record)
        
        # Keep history manageable
        if len(self.suggestion_history) > 500:
            self.suggestion_history = self.suggestion_history[-250:]
    
    async def learn_from_command_usage(self, command: str, success: bool, execution_time: float, context: Dict[str, Any]):
        """Learn from actual command usage."""
        if command not in self.usage_patterns:
            self.usage_patterns[command] = {
                'usage_count': 0,
                'success_count': 0,
                'total_execution_time': 0.0,
                'contexts': [],
                'last_used': None
            }
        
        pattern = self.usage_patterns[command]
        pattern['usage_count'] += 1
        pattern['total_execution_time'] += execution_time
        pattern['last_used'] = datetime.now(timezone.utc).isoformat()
        
        if success:
            pattern['success_count'] += 1
        
        # Store context (limited)
        context_summary = {
            'project_type': context.get('project_type'),
            'intent': context.get('user_intent', '')[:50],
            'success': success
        }
        pattern['contexts'].append(context_summary)
        
        # Keep contexts manageable
        if len(pattern['contexts']) > 20:
            pattern['contexts'] = pattern['contexts'][-10:]
        
        # Update command confidence
        if command in self.slash_commands:
            success_rate = pattern['success_count'] / pattern['usage_count']
            self.slash_commands[command].confidence = success_rate
            self.slash_commands[command].usage_count = pattern['usage_count']
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.discovery_cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        current_time = asyncio.get_event_loop().time()
        
        return (current_time - timestamp) < self.cache_timeout
    
    def invalidate_cache(self, server_name: str = None):
        """Invalidate discovery cache."""
        if server_name:
            # Invalidate specific server
            keys_to_remove = [key for key in self.discovery_cache.keys() if key.startswith(server_name)]
            for key in keys_to_remove:
                del self.discovery_cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
        else:
            # Invalidate all
            self.discovery_cache.clear()
            self._cache_timestamps.clear()
        
        logger.debug(f"Invalidated slash command cache for {server_name or 'all servers'}")
    
    async def get_command_analytics(self) -> Dict[str, Any]:
        """Get analytics about discovered and used slash commands."""
        analytics = {
            'total_commands_discovered': len(self.slash_commands),
            'commands_by_server': {},
            'commands_by_category': {},
            'usage_statistics': {},
            'top_commands': [],
            'recent_suggestions': 0,
            'suggestion_effectiveness': 0.0
        }
        
        # Commands by server
        for command in self.slash_commands.values():
            server = command.server_name
            analytics['commands_by_server'][server] = analytics['commands_by_server'].get(server, 0) + 1
        
        # Commands by category
        for command in self.slash_commands.values():
            category = command.category
            analytics['commands_by_category'][category] = analytics['commands_by_category'].get(category, 0) + 1
        
        # Usage statistics
        total_usage = 0
        total_success = 0
        
        for command_name, pattern in self.usage_patterns.items():
            total_usage += pattern['usage_count']
            total_success += pattern['success_count']
            
            if pattern['usage_count'] > 0:
                success_rate = pattern['success_count'] / pattern['usage_count']
                avg_execution_time = pattern['total_execution_time'] / pattern['usage_count']
                
                analytics['usage_statistics'][command_name] = {
                    'usage_count': pattern['usage_count'],
                    'success_rate': success_rate,
                    'avg_execution_time': avg_execution_time
                }
        
        # Top commands by usage
        top_commands = sorted(
            [(name, data) for name, data in self.usage_patterns.items()],
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )
        analytics['top_commands'] = [
            {
                'command': name,
                'usage_count': data['usage_count'],
                'success_rate': data['success_count'] / data['usage_count']
            }
            for name, data in top_commands[:10]
        ]
        
        # Recent activity
        current_time = datetime.now(timezone.utc).timestamp()
        recent_cutoff = current_time - 3600  # Last hour
        
        recent_suggestions = 0
        for record in self.suggestion_history:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')).timestamp()
                if timestamp > recent_cutoff:
                    recent_suggestions += 1
            except (ValueError, KeyError):
                continue
        
        analytics['recent_suggestions'] = recent_suggestions
        
        # Suggestion effectiveness (if we have usage data)
        if total_usage > 0:
            analytics['suggestion_effectiveness'] = total_success / total_usage
        
        return analytics
    
    def get_command_by_name(self, command_name: str) -> Optional[SlashCommand]:
        """Get a specific command by name."""
        return self.slash_commands.get(command_name)
    
    def get_commands_by_server(self, server_name: str) -> List[SlashCommand]:
        """Get all commands from a specific server."""
        return [cmd for cmd in self.slash_commands.values() if cmd.server_name == server_name]
    
    def get_commands_by_category(self, category: str) -> List[SlashCommand]:
        """Get all commands in a specific category."""
        return [cmd for cmd in self.slash_commands.values() if cmd.category == category]
    
    async def validate_commands(self) -> Dict[str, Any]:
        """Validate that discovered commands are still accessible."""
        validation_results = {
            'total_commands': len(self.slash_commands),
            'accessible_commands': 0,
            'inaccessible_commands': 0,
            'validation_errors': [],
            'server_status': {}
        }
        
        # Group commands by server for efficient validation
        commands_by_server = {}
        for command in self.slash_commands.values():
            server = command.server_name
            if server not in commands_by_server:
                commands_by_server[server] = []
            commands_by_server[server].append(command)
        
        # Validate each server
        for server_name, commands in commands_by_server.items():
            try:
                # Try to get server configuration
                if hasattr(self.tool_indexer, 'discovered_servers'):
                    servers = await self.tool_indexer.get_discovered_servers()
                    if server_name in servers:
                        server_config = servers[server_name]
                        # Try basic connectivity test
                        test_commands = await self._connect_and_discover(server_name, server_config)
                        
                        validation_results['server_status'][server_name] = {
                            'accessible': len(test_commands) > 0,
                            'commands_count': len(commands),
                            'discovered_count': len(test_commands)
                        }
                        
                        if len(test_commands) > 0:
                            validation_results['accessible_commands'] += len(commands)
                        else:
                            validation_results['inaccessible_commands'] += len(commands)
                    else:
                        validation_results['validation_errors'].append(f"Server {server_name} not found in configuration")
                        validation_results['inaccessible_commands'] += len(commands)
                else:
                    validation_results['validation_errors'].append("Tool indexer not available for validation")
                    
            except Exception as e:
                validation_results['validation_errors'].append(f"Error validating {server_name}: {e}")
                validation_results['inaccessible_commands'] += len(commands)
        
        return validation_results