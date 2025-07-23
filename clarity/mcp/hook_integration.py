"""
Hook Integration Framework for MCP Discovery Enhancement.

This module integrates with Claude Code's hook system to learn from real-time
tool usage patterns and provide proactive MCP tool suggestions.
"""

import asyncio
import json
import os
import re
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from loguru import logger
from datetime import datetime
from .ultrathink_config import UltrathinkConfig

if TYPE_CHECKING:
    from .tool_indexer import MCPToolIndexer


class MCPHookIntegration:
    """Integrates with Claude Code's hook system for MCP learning."""
    
    def __init__(self, tool_indexer: 'MCPToolIndexer'):
        """Initialize hook integration system."""
        self.tool_indexer = tool_indexer
        self.hook_config_path = "~/.config/claude-code/hooks.json"
        self.hook_script_path = os.path.join(os.path.dirname(__file__), "hook_analyzer.py")
        self.learning_patterns = {}
        self.suggestion_history = []
        
        # Patterns for detecting MCP opportunities
        self.opportunity_patterns = {
            'database_queries': [
                r'psql\s+-.*-c',
                r'mysql\s+-.*-e',
                r'sqlite3.*"select',
                r'query.*database',
                r'database.*query',
                r'select.*from',
                r'\bdatabase\b',
                r'\bquery\b',
                r'\bsql\b',
                r'\buser.*authentication\b',
                r'\bauth\b',
                r'\blogin\b',
                r'\bvalidation\b',
                r'\buser.*activity\b',
                r'\banalytics\b'
            ],
            'web_automation': [
                r'curl\s+.*http',
                r'wget\s+.*http',
                r'browse.*to',
                r'navigate.*to',
                r'click.*on',
                r'\bapi\b.*\bcall\b',
                r'\bapi\b',
                r'\bcall\b',
                r'\bhttp\b',
                r'\bweb\b'
            ],
            'file_operations': [
                r'cat\s+.*\|',
                r'grep.*-r',
                r'find.*-name',
                r'read.*file',
                r'file.*read',
                r'write.*to.*file',
                r'\bfile\b',
                r'\bread\b',
                r'\bwrite\b',
                r'\bconfiguration\b',
                r'\bconfig\b',
                r'\breport\b',
                r'\blog\b'
            ],
            'memory_operations': [
                r'remember.*this',
                r'save.*for.*later',
                r'store.*information',
                r'recall.*from'
            ]
        }
        
        # Patterns for detecting structured thinking triggers
        self.structured_thinking_patterns = [
            # Explicit structured thinking requests
            r'\bstructured\s+thinking\b',
            r'\bthinking\s+process\b',
            r'\bstructured\s+approach\b',
            r'\bstep\s+by\s+step\b',
            r'\bstep-by-step\b',
            r'\bsystematic\s+analysis\b',
            r'\bcomprehensive\s+analysis\b',
            r'\bdetailed\s+analysis\b',
            r'\bmethodical\s+approach\b',
            
            # Problem-solving patterns
            r'\bhow\s+should\s+we\s+approach\b',
            r'\bwhat\s+are\s+the\s+steps\b',
            r'\bbreak\s+down\s+the\s+problem\b',
            r'\banalyze\s+this\s+problem\b',
            r'\bsolve\s+this\s+step\s+by\s+step\b',
            r'\bwalk\s+through\s+the\s+process\b',
            r'\bthink\s+through\s+this\b',
            r'\breason\s+through\s+this\b',
            
            # Complex reasoning indicators
            r'\bcomplex\s+(problem|challenge|issue|situation)\b',
            r'\bmultiple\s+(factors|variables|considerations)\b',
            r'\bneed\s+to\s+consider\b.*\bmultiple\b',
            r'\brequires\s+careful\s+(thought|analysis|consideration)\b',
            r'\bcomprehensive\s+(understanding|review|evaluation)\b',
            r'\bin-depth\s+(analysis|review|evaluation)\b',
            r'\bthorough\s+(analysis|review|evaluation)\b',
            
            # Decision-making patterns
            r'\bmake\s+a\s+decision\s+about\b',
            r'\bchoose\s+between\s+options\b',
            r'\bevaluate\s+the\s+alternatives\b',
            r'\bweigh\s+the\s+(pros\s+and\s+cons|options|alternatives)\b',
            r'\bcompare\s+and\s+contrast\b',
            r'\bpros\s+and\s+cons\b',
            
            # Planning and strategy patterns
            r'\bplan\s+for\b.*\b(project|implementation|strategy)\b',
            r'\bdevelop\s+a\s+(strategy|plan|approach)\b',
            r'\bcreate\s+a\s+(roadmap|framework|methodology)\b',
            r'\bdesign\s+an\s+approach\b',
            r'\barchitecture\s+(design|planning)\b',
            
            # Learning and understanding patterns
            r'\bunderstand\s+the\s+(concept|system|process)\b',
            r'\blearn\s+about\b.*\b(complex|advanced|detailed)\b',
            r'\bexplain\s+the\s+(relationship|connection|interaction)\b',
            r'\bhelp\s+me\s+understand\b',
            
            # Creative and innovative thinking
            r'\bbrainstorm\b.*\b(ideas|solutions|approaches)\b',
            r'\bcreative\s+(solutions|approaches|thinking)\b',
            r'\binnovative\s+(ideas|solutions|approaches)\b',
            r'\bthink\s+outside\s+the\s+box\b',
            r'\bunconventional\s+(approach|solution|method)\b'
        ]
        
        # Initialize ultrathink configuration
        self.ultrathink_config_manager = UltrathinkConfig()
        self.ultrathink_config = self.ultrathink_config_manager.get_config()
        
        # Combine built-in patterns with custom patterns
        self.all_thinking_patterns = self.structured_thinking_patterns.copy()
        custom_patterns = self.ultrathink_config_manager.get_custom_patterns()
        for custom_pattern in custom_patterns:
            if isinstance(custom_pattern, dict) and 'pattern' in custom_pattern:
                self.all_thinking_patterns.append(custom_pattern['pattern'])
            elif isinstance(custom_pattern, str):
                self.all_thinking_patterns.append(custom_pattern)
    
    async def setup_hooks(self) -> bool:
        """
        Configure Claude Code hooks for MCP learning.
        
        Returns:
            True if hooks were successfully configured, False otherwise
        """
        try:
            # Create hooks configuration
            hook_config = await self._generate_hook_config()
            
            # Write hook configuration
            success = await self._write_hook_config(hook_config)
            
            if success:
                logger.info("MCP hook integration configured successfully")
                return True
            else:
                logger.warning("Failed to configure MCP hooks")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up hooks: {e}")
            return False
    
    async def _generate_hook_config(self) -> Dict[str, Any]:
        """Generate the hook configuration for Claude Code."""
        python_cmd = "python"
        analyzer_script = os.path.abspath(self.hook_script_path)
        
        hook_config = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "bash|shell|exec|run_command",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"{python_cmd} {analyzer_script} --pre-tool --tool={{tool_name}} --args={{args}}",
                                "timeout_ms": 2000,
                                "continue_on_error": True
                            }
                        ]
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"{python_cmd} {analyzer_script} --post-tool --tool={{tool_name}} --result={{result}}",
                                "timeout_ms": 3000,
                                "continue_on_error": True
                            }
                        ]
                    }
                ],
                "UserPromptSubmit": [
                    {
                        "matcher": "*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"{python_cmd} {analyzer_script} --prompt-submit --prompt={{prompt}}",
                                "timeout_ms": 1500,
                                "continue_on_error": True,
                                "modify_prompt": True
                            }
                        ]
                    }
                ]
            },
            "metadata": {
                "created_by": "mcp-alunai-clarity",
                "version": "1.0.0",
                "description": "MCP discovery enhancement hooks",
                "created_at": datetime.now().isoformat()
            }
        }
        
        return hook_config
    
    async def _write_hook_config(self, config: Dict[str, Any]) -> bool:
        """Write the hook configuration to Claude Code."""
        try:
            config_path = os.path.expanduser(self.hook_config_path)
            config_dir = os.path.dirname(config_path)
            
            # Create config directory if it doesn't exist
            os.makedirs(config_dir, exist_ok=True)
            
            # Check if hooks file already exists and merge
            existing_config = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        existing_config = json.load(f)
                    logger.debug("Found existing hook configuration")
                except Exception as e:
                    logger.warning(f"Could not read existing hooks: {e}")
            
            # Merge configurations (our hooks take precedence)
            merged_config = existing_config.copy()
            merged_config.update(config)
            
            # Write the configuration
            with open(config_path, 'w') as f:
                json.dump(merged_config, f, indent=2)
            
            logger.info(f"Hook configuration written to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write hook configuration: {e}")
            return False
    
    async def analyze_tool_usage(self, event_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze tool usage from hooks for MCP learning opportunities.
        
        Args:
            event_type: Type of hook event (pre_tool, post_tool, prompt_submit)
            data: Event data from hook
            
        Returns:
            Analysis result or None
        """
        try:
            if event_type == "pre_tool":
                return await self._analyze_pre_tool_usage(data)
            elif event_type == "post_tool":
                return await self._learn_from_tool_result(data)
            elif event_type == "prompt_submit":
                suggestion_text = await self._suggest_mcp_opportunities(data)
                if suggestion_text:
                    # Extract opportunities for structured response
                    prompt = data.get('prompt', '')
                    opportunities = []
                    
                    # Analyze prompt for MCP tool opportunities
                    for category, patterns in self.opportunity_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, prompt, re.IGNORECASE):
                                # Get relevant MCP tools for this category
                                mcp_tools = await self._get_relevant_mcp_tools(category)
                                
                                if mcp_tools:
                                    opportunities.extend([tool['name'] for tool in mcp_tools])
                    
                    return {
                        'MCP Tool Suggestion': suggestion_text,
                        'suggested_approach': list(set(opportunities)),  # Remove duplicates
                        'confidence': 0.8 if opportunities else 0.5
                    }
                return None
            else:
                logger.warning(f"Unknown event type: {event_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing tool usage: {e}")
            return None
    
    async def _analyze_pre_tool_usage(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze tool usage before execution to suggest MCP alternatives."""
        tool_name = data.get('tool_name', '')
        tool_args = data.get('args', '')
        
        # Check if this is a tool that could be replaced by MCP
        if tool_name in ['bash', 'shell', 'exec', 'run_command']:
            command = tool_args
            
            # Analyze the command for MCP opportunities
            suggestions = await self._detect_command_opportunities(command)
            
            if suggestions:
                # Log the learning opportunity
                await self._log_learning_opportunity({
                    'type': 'pre_tool_suggestion',
                    'original_tool': tool_name,
                    'original_args': tool_args,
                    'suggested_mcp_tools': suggestions,
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'suggestions': suggestions,
                    'confidence': self._calculate_suggestion_confidence(command, suggestions)
                }
        
        return None
    
    async def _learn_from_tool_result(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Learn from successful tool usage patterns."""
        tool_name = data.get('tool_name', '')
        result = data.get('result', '')
        
        # Analyze if the tool was successful and could have used MCP
        success_indicators = [
            'successfully', 'completed', 'done', 'finished', 'ok', 'success'
        ]
        
        is_successful = any(indicator in str(result).lower() for indicator in success_indicators)
        
        if is_successful and (tool_name in ['bash', 'shell', 'exec'] or 'postgres' in tool_name or 'query' in tool_name):
            # Store successful pattern for future learning
            pattern = {
                'tool_name': tool_name,
                'result_pattern': self._extract_result_pattern(result),
                'success': True,
                'success_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._store_usage_pattern(pattern)
            
            return {'learned_pattern': True, 'success': True}
        
        return None
    
    async def _suggest_mcp_opportunities(self, data: Dict[str, Any]) -> Optional[str]:
        """Suggest MCP opportunities from user prompts."""
        prompt = data.get('prompt', '')
        
        # First, check for structured thinking patterns and enable ultrathink if needed
        enhanced_prompt = await self._enhance_prompt_with_ultrathink(prompt)
        
        # Analyze prompt for MCP tool opportunities
        opportunities = []
        
        for category, patterns in self.opportunity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    # Get relevant MCP tools for this category
                    mcp_tools = await self._get_relevant_mcp_tools(category)
                    
                    if mcp_tools:
                        opportunities.append({
                            'category': category,
                            'pattern': pattern,
                            'suggested_tools': mcp_tools
                        })
        
        if opportunities:
            # Generate suggestion message
            suggestion = await self._format_prompt_suggestion(opportunities, enhanced_prompt)
            
            # Log this suggestion
            await self._log_learning_opportunity({
                'type': 'prompt_suggestion',
                'original_prompt': prompt,
                'opportunities': opportunities,
                'suggestion': suggestion,
                'timestamp': datetime.now().isoformat()
            })
            
            return suggestion
        
        # Return enhanced prompt even if no MCP opportunities found
        return enhanced_prompt if enhanced_prompt != prompt else None
    
    async def _detect_command_opportunities(self, command: str) -> List[Dict[str, Any]]:
        """Detect MCP tool opportunities in bash commands."""
        opportunities = []
        
        # Database commands
        if re.search(r'psql|mysql|sqlite', command, re.IGNORECASE):
            db_tools = await self._get_relevant_mcp_tools('database_queries')
            if db_tools:
                opportunities.append({
                    'category': 'database',
                    'reason': 'Direct database queries available via MCP',
                    'tools': db_tools[:2]  # Limit suggestions
                })
        
        # Web requests
        if re.search(r'curl|wget|http', command, re.IGNORECASE):
            web_tools = await self._get_relevant_mcp_tools('web_automation')
            if web_tools:
                opportunities.append({
                    'category': 'web',
                    'reason': 'Web automation tools available via MCP',
                    'tools': web_tools[:2]
                })
        
        # File operations
        if re.search(r'grep|find|cat.*\||awk|sed', command, re.IGNORECASE):
            file_tools = await self._get_relevant_mcp_tools('file_operations')
            if file_tools:
                opportunities.append({
                    'category': 'files',
                    'reason': 'File operations available via MCP tools',
                    'tools': file_tools[:2]
                })
        
        return opportunities
    
    async def _get_relevant_mcp_tools(self, category: str) -> List[Dict[str, str]]:
        """Get relevant MCP tools for a category."""
        try:
            relevant_tools = []
            
            # Check if we have indexed tools (preferred)
            if hasattr(self.tool_indexer, 'indexed_tools') and self.tool_indexer.indexed_tools:
                # Filter tools by category
                category_map = {
                    'database_queries': 'database',
                    'web_automation': 'web_automation', 
                    'file_operations': 'file_operations',
                    'memory_operations': 'memory_management'
                }
                
                target_category = category_map.get(category, category)
                
                for tool_name, tool_info in self.tool_indexer.indexed_tools.items():
                    if hasattr(tool_info, 'category') and tool_info.category == target_category:
                        relevant_tools.append({
                            'name': tool_info.name,
                        'description': tool_info.description,
                        'server': tool_info.server_name
                    })
            
            # Fallback to discovered_servers if no indexed tools
            elif hasattr(self.tool_indexer, 'discovered_servers') and self.tool_indexer.discovered_servers:
                # Map categories to server types
                server_category_map = {
                    'database_queries': ['postgres', 'sqlite', 'mysql'],
                    'web_automation': ['web', 'browser', 'http'],
                    'file_operations': ['filesystem', 'file'],
                    'memory_operations': ['memory', 'storage']
                }
                
                target_servers = server_category_map.get(category, [])
                
                for server_name, server_info in self.tool_indexer.discovered_servers.items():
                    # Check if server matches category
                    if any(target in server_name.lower() for target in target_servers):
                        tools = server_info.get('tools', [])
                        for tool_name in tools:
                            relevant_tools.append({
                                'name': tool_name,
                                'description': f'{tool_name} from {server_name}',
                                'server': server_name
                            })
            
            return relevant_tools[:3]  # Limit to top 3
            
        except Exception as e:
            logger.debug(f"Error getting relevant MCP tools: {e}")
            return []
    
    def _calculate_suggestion_confidence(self, command: str, suggestions: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for suggestions."""
        base_confidence = 0.5
        
        # Higher confidence for exact pattern matches
        exact_patterns = len([s for s in suggestions if self._has_exact_pattern_match(command, s)])
        pattern_boost = exact_patterns * 0.2
        
        # Higher confidence for multiple suggestions
        multiple_boost = 0.1 if len(suggestions) > 1 else 0
        
        confidence = min(1.0, base_confidence + pattern_boost + multiple_boost)
        return confidence
    
    def _has_exact_pattern_match(self, command: str, suggestion: Dict[str, Any]) -> bool:
        """Check if command has exact pattern match with suggestion."""
        category = suggestion.get('category', '')
        if category == 'database':
            return bool(re.search(r'select|insert|update|delete', command, re.IGNORECASE))
        elif category == 'web':
            return bool(re.search(r'http[s]?://', command))
        elif category == 'files':
            return bool(re.search(r'\.(txt|csv|json|xml)', command))
        return False
    
    async def _format_prompt_suggestion(self, opportunities: List[Dict[str, Any]], original_prompt: str) -> str:
        """Format MCP tool suggestions for prompt modification."""
        if not opportunities:
            return original_prompt
        
        suggestion_lines = []
        suggestion_lines.append("\n🔧 **MCP Tool Suggestion**: Consider using these MCP tools instead of scripts:")
        
        for opportunity in opportunities[:2]:  # Limit to top 2 opportunities
            category = opportunity['category']
            tools = opportunity['suggested_tools'][:2]  # Limit to 2 tools per category
            
            suggestion_lines.append(f"\n**{category.title()} Operations:**")
            for tool in tools:
                suggestion_lines.append(f"- `{tool['name']}` from {tool['server']}: {tool['description']}")
        
        suggestion_lines.append("\nThese tools provide direct access without shell scripts and often have better error handling.")
        
        # Append to original prompt
        return original_prompt + "\n".join(suggestion_lines)
    
    def _extract_result_pattern(self, result: str) -> str:
        """Extract a pattern from tool execution result for learning."""
        if not result:
            return "no_output"
        
        result_str = str(result).lower()
        
        if "error" in result_str or "failed" in result_str:
            return "error"
        elif "success" in result_str or "done" in result_str:
            return "success" 
        elif "warning" in result_str:
            return "warning"
        else:
            return "unknown"
    
    async def _store_usage_pattern(self, pattern: Dict[str, Any]):
        """Store learned usage pattern for future suggestions."""
        try:
            pattern_key = f"{pattern['tool_name']}_{pattern['result_pattern']}"
            
            if pattern_key not in self.learning_patterns:
                self.learning_patterns[pattern_key] = {
                    'count': 0,
                    'success_rate': 0.0,
                    'examples': []
                }
            
            self.learning_patterns[pattern_key]['count'] += 1
            self.learning_patterns[pattern_key]['examples'].append(pattern)
            
            # Calculate success rate
            total = len(self.learning_patterns[pattern_key]['examples'])
            successful = sum(1 for ex in self.learning_patterns[pattern_key]['examples'] if ex.get('success', False))
            self.learning_patterns[pattern_key]['success_rate'] = successful / total if total > 0 else 0
            
            # Store in memory system if available
            if hasattr(self.tool_indexer, 'domain_manager'):
                await self.tool_indexer.domain_manager.store_memory(
                    memory_type="mcp_usage_pattern",
                    content=json.dumps(pattern),
                    importance=0.7,
                    metadata={
                        "category": "hook_learning",
                        "pattern_type": pattern['result_pattern'],
                        "tool_name": pattern['tool_name']
                    }
                )
                
        except Exception as e:
            logger.error(f"Error storing usage pattern: {e}")
    
    async def _log_learning_opportunity(self, opportunity: Dict[str, Any]):
        """Log learning opportunities for analysis."""
        try:
            self.suggestion_history.append(opportunity)
            
            # Keep only recent history (last 100 entries)
            if len(self.suggestion_history) > 100:
                self.suggestion_history = self.suggestion_history[-100:]
            
            # Store in memory system
            if hasattr(self.tool_indexer, 'domain_manager'):
                await self.tool_indexer.domain_manager.store_memory(
                    memory_type="mcp_learning_opportunity", 
                    content=json.dumps(opportunity),
                    importance=0.8,
                    metadata={
                        "category": "hook_analysis",
                        "opportunity_type": opportunity['type'],
                        "auto_generated": True
                    }
                )
                
        except Exception as e:
            logger.error(f"Error logging learning opportunity: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning patterns."""
        # Calculate suggestion success rate from history
        successful_suggestions = sum(1 for s in self.suggestion_history if s.get('accepted', False))
        total_suggestions = len(self.suggestion_history)
        suggestion_success_rate = successful_suggestions / total_suggestions if total_suggestions > 0 else 0
        
        return {
            'total_patterns': len(self.learning_patterns),
            'patterns_learned': len(self.learning_patterns),
            'suggestion_history_count': len(self.suggestion_history),
            'suggestion_success_rate': suggestion_success_rate,
            'pattern_categories': list(self.learning_patterns.keys()),
            'average_success_rate': sum(p['success_rate'] for p in self.learning_patterns.values()) / len(self.learning_patterns) if self.learning_patterns else 0
        }
    
    async def get_proactive_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get proactive MCP tool suggestions based on learned patterns."""
        suggestions = []
        
        current_task = context.get('current_task', '')
        recent_tools = context.get('recent_tools', [])
        
        try:
            # Analyze current context for suggestion opportunities  
            for category, patterns in self.opportunity_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, current_task, re.IGNORECASE):
                        tools = await self._get_relevant_mcp_tools(category)
                        if tools:
                            suggestions.append({
                                'category': category,
                                'confidence': 0.8,
                                'tools': tools[:2],
                                'reason': f'Detected {category} pattern in current task'
                            })
            
            # Consider recent tool usage patterns
            if recent_tools:
                for tool in recent_tools[-3:]:  # Last 3 tools
                    if tool in ['bash', 'shell', 'exec']:
                        # Check if we have learned patterns for this
                        pattern_key = f"{tool}_success"
                        if pattern_key in self.learning_patterns:
                            pattern_info = self.learning_patterns[pattern_key]
                            if pattern_info['success_rate'] < 0.5:  # Low success rate
                                suggestions.append({
                                    'category': 'alternative',
                                    'confidence': 0.6,
                                    'tools': await self._get_relevant_mcp_tools('file_operations'),
                                    'reason': f'Low success rate with {tool}, consider MCP alternatives'
                                })
            
        except Exception as e:
            logger.error(f"Error generating proactive suggestions: {e}")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    async def _enhance_prompt_with_ultrathink(self, prompt: str) -> str:
        """
        Detect structured thinking patterns and enhance prompt with ultrathink directive.
        
        Args:
            prompt: The original user prompt
            
        Returns:
            Enhanced prompt with ultrathink directive if patterns detected, otherwise original prompt
        """
        if not self.ultrathink_config_manager.is_enabled():
            return prompt
        
        # Check for exclusion patterns first
        exclusion_patterns = self.ultrathink_config_manager.get_exclusion_patterns()
        for exclusion_pattern in exclusion_patterns:
            if re.search(exclusion_pattern, prompt, re.IGNORECASE):
                logger.debug(f"Ultrathink disabled due to exclusion pattern: {exclusion_pattern}")
                return prompt
        
        # Check for structured thinking patterns
        thinking_matches = []
        confidence_score = 0.0
        pattern_weights = self.ultrathink_config_manager.get_pattern_weights()
        confidence_scores = self.ultrathink_config_manager.get_confidence_scores()
        
        # Check built-in patterns
        for pattern in self.all_thinking_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                thinking_matches.extend(matches)
                
                # Determine confidence boost based on pattern keywords
                pattern_lower = pattern.lower()
                if any(keyword in pattern_lower for keyword in pattern_weights.get('high_confidence', [])):
                    confidence_score += confidence_scores.get('high_confidence', 0.3)
                elif any(keyword in pattern_lower for keyword in pattern_weights.get('medium_confidence', [])):
                    confidence_score += confidence_scores.get('medium_confidence', 0.2)
                else:
                    confidence_score += confidence_scores.get('low_confidence', 0.1)
        
        # Check custom patterns with their specific weights
        custom_patterns = self.ultrathink_config_manager.get_custom_patterns()
        for custom_pattern in custom_patterns:
            if isinstance(custom_pattern, dict):
                pattern = custom_pattern.get('pattern', '')
                weight_category = custom_pattern.get('weight_category', 'medium_confidence')
            else:
                pattern = custom_pattern
                weight_category = 'medium_confidence'
            
            if pattern:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if matches:
                    thinking_matches.extend(matches)
                    confidence_score += confidence_scores.get(weight_category, 0.1)
        
        # Normalize confidence score (cap at 1.0)
        confidence_score = min(confidence_score, 1.0)
        
        # Check if confidence meets threshold
        minimum_confidence = self.ultrathink_config_manager.get_minimum_confidence()
        
        if confidence_score >= minimum_confidence and thinking_matches:
            # Log the ultrathink enhancement
            await self._log_ultrathink_enhancement(prompt, thinking_matches, confidence_score)
            
            # Add the ultrathink directive
            ultrathink_directive = self.ultrathink_config_manager.get_ultrathink_directive()
            enhanced_prompt = prompt + ultrathink_directive
            
            logger.info(f"Ultrathink mode enabled for prompt with confidence {confidence_score:.2f}")
            
            return enhanced_prompt
        
        return prompt
    
    async def _log_ultrathink_enhancement(self, original_prompt: str, matches: List[str], confidence: float):
        """Log ultrathink enhancements for analysis and learning."""
        try:
            enhancement_data = {
                'type': 'ultrathink_enhancement',
                'original_prompt': original_prompt[:200] + '...' if len(original_prompt) > 200 else original_prompt,
                'matched_patterns': list(set(matches)),  # Remove duplicates
                'confidence_score': confidence,
                'timestamp': datetime.now().isoformat(),
                'enhancement_applied': True
            }
            
            # Add to suggestion history for tracking
            self.suggestion_history.append(enhancement_data)
            
            # Keep only recent history (last 100 entries)
            if len(self.suggestion_history) > 100:
                self.suggestion_history = self.suggestion_history[-100:]
            
            # Store in memory system if available
            if hasattr(self.tool_indexer, 'domain_manager'):
                await self.tool_indexer.domain_manager.store_memory(
                    memory_type="ultrathink_enhancement",
                    content=json.dumps(enhancement_data),
                    importance=0.8,  # High importance for learning patterns
                    metadata={
                        "category": "thinking_enhancement",
                        "enhancement_type": "ultrathink_auto_enable",
                        "confidence_score": confidence,
                        "auto_generated": True
                    }
                )
                
        except Exception as e:
            logger.error(f"Error logging ultrathink enhancement: {e}")
    
    def configure_ultrathink(self, config_updates: Dict[str, Any]) -> None:
        """
        Update ultrathink configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.ultrathink_config_manager.update_config(config_updates)
        self.ultrathink_config = self.ultrathink_config_manager.get_config()
        
        # Update thinking patterns if custom patterns were changed
        self.all_thinking_patterns = self.structured_thinking_patterns.copy()
        custom_patterns = self.ultrathink_config_manager.get_custom_patterns()
        for custom_pattern in custom_patterns:
            if isinstance(custom_pattern, dict) and 'pattern' in custom_pattern:
                self.all_thinking_patterns.append(custom_pattern['pattern'])
            elif isinstance(custom_pattern, str):
                self.all_thinking_patterns.append(custom_pattern)
        
        logger.info(f"Ultrathink configuration updated: {config_updates}")
    
    def get_ultrathink_stats(self) -> Dict[str, Any]:
        """Get statistics about ultrathink enhancements."""
        ultrathink_enhancements = [
            s for s in self.suggestion_history 
            if s.get('type') == 'ultrathink_enhancement'
        ]
        
        if not ultrathink_enhancements:
            stats = {
                'total_enhancements': 0,
                'average_confidence': 0.0,
                'most_common_patterns': [],
                'enhancement_rate': 0.0
            }
        else:
            total_prompts = len(self.suggestion_history)
            enhancement_count = len(ultrathink_enhancements)
            
            # Calculate average confidence
            avg_confidence = sum(e.get('confidence_score', 0) for e in ultrathink_enhancements) / enhancement_count
            
            # Find most common patterns
            all_patterns = []
            for enhancement in ultrathink_enhancements:
                all_patterns.extend(enhancement.get('matched_patterns', []))
            
            from collections import Counter
            pattern_counts = Counter(all_patterns)
            most_common_patterns = pattern_counts.most_common(5)
            
            stats = {
                'total_enhancements': enhancement_count,
                'average_confidence': avg_confidence,
                'most_common_patterns': most_common_patterns,
                'enhancement_rate': enhancement_count / total_prompts if total_prompts > 0 else 0.0
            }
        
        # Add configuration stats
        stats['config_stats'] = self.ultrathink_config_manager.get_stats()
        stats['pattern_counts'] = {
            'built_in_patterns': len(self.structured_thinking_patterns),
            'custom_patterns': len(self.ultrathink_config_manager.get_custom_patterns()),
            'total_patterns': len(self.all_thinking_patterns),
            'exclusion_patterns': len(self.ultrathink_config_manager.get_exclusion_patterns())
        }
        
        return stats