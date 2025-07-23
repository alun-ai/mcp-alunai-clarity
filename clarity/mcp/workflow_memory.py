"""Enhanced Memory System for MCP Workflow Patterns.

This module extends the memory system to store and retrieve MCP workflow
patterns, enabling intelligent suggestions based on successful usage history.
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

if TYPE_CHECKING:
    from clarity.domains.domain_manager import DomainManager

logger = logging.getLogger(__name__)


@dataclass
class MCPWorkflowPattern:
    """Represents a successful MCP workflow pattern."""
    pattern_id: str
    pattern_type: str
    trigger_context: str
    tool_sequence: List[str]
    resource_references: List[str]
    success_indicators: Dict[str, Any]
    contextual_factors: Dict[str, Any]
    usage_frequency: int
    effectiveness_score: float
    created_at: str
    last_used: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPWorkflowPattern':
        """Create pattern from dictionary."""
        return cls(
            pattern_id=data.get('pattern_id', ''),
            pattern_type=data.get('pattern_type', 'mcp_workflow'),
            trigger_context=data.get('trigger_context', ''),
            tool_sequence=data.get('tool_sequence', []),
            resource_references=data.get('resource_references', []),
            success_indicators=data.get('success_indicators', {}),
            contextual_factors=data.get('contextual_factors', {}),
            usage_frequency=data.get('usage_frequency', 1),
            effectiveness_score=data.get('effectiveness_score', 0.8),
            created_at=data.get('created_at', datetime.now(timezone.utc).isoformat()),
            last_used=data.get('last_used', datetime.now(timezone.utc).isoformat())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return asdict(self)


@dataclass 
class MCPInteraction:
    """Represents a single MCP interaction for learning."""
    interaction_id: str
    server_name: str
    tool_name: str
    resource_used: Optional[str]
    context: str
    success: bool
    duration_ms: float
    error_message: Optional[str]
    user_intent: str
    timestamp: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPInteraction':
        """Create interaction from dictionary."""
        return cls(
            interaction_id=data.get('interaction_id', ''),
            server_name=data.get('server_name', ''),
            tool_name=data.get('tool_name', ''),
            resource_used=data.get('resource_used'),
            context=data.get('context', ''),
            success=data.get('success', False),
            duration_ms=data.get('duration_ms', 0.0),
            error_message=data.get('error_message'),
            user_intent=data.get('user_intent', ''),
            timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to dictionary."""
        return asdict(self)


@dataclass
class SuggestionContext:
    """Context for generating MCP suggestions."""
    current_task: str
    user_intent: str
    project_type: Optional[str]
    recent_tools_used: List[str]
    recent_failures: List[str]
    environment_info: Dict[str, Any]
    available_servers: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuggestionContext':
        """Create context from dictionary."""
        return cls(
            current_task=data.get('current_task', ''),
            user_intent=data.get('user_intent', ''),
            project_type=data.get('project_type'),
            recent_tools_used=data.get('recent_tools_used', []),
            recent_failures=data.get('recent_failures', []),
            environment_info=data.get('environment_info', {}),
            available_servers=data.get('available_servers', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return asdict(self)


class WorkflowMemoryEnhancer:
    """Enhances memory system with MCP workflow patterns."""
    
    def __init__(self, domain_manager: 'DomainManager'):
        """Initialize workflow memory enhancer."""
        self.domain_manager = domain_manager
        self.pattern_cache = {}
        self.interaction_history = []
        self.cache_timeout = 300  # 5 minutes
        self._cache_timestamps = {}
        
        # Pattern scoring weights
        self.scoring_weights = {
            'success_rate': 0.3,
            'usage_frequency': 0.2,
            'recency': 0.2,
            'context_similarity': 0.2,
            'effectiveness_score': 0.1
        }
    
    async def store_mcp_workflow_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Store successful MCP usage pattern as memory."""
        try:
            # Create pattern object
            pattern_id = f"mcp_pattern_{asyncio.get_event_loop().time():.0f}"
            
            pattern = MCPWorkflowPattern(
                pattern_id=pattern_id,
                pattern_type="mcp_workflow",
                trigger_context=pattern_data.get("context", ""),
                tool_sequence=pattern_data.get("tools", []),
                resource_references=pattern_data.get("resources", []),
                success_indicators=pattern_data.get("success", {}) if not isinstance(pattern_data.get("success"), bool) else {"success": pattern_data.get("success", True)},
                contextual_factors={
                    "project_type": pattern_data.get("project_type"),
                    "user_intent": pattern_data.get("intent"),
                    "environment": pattern_data.get("env", {}),
                    "complexity": pattern_data.get("complexity", "medium"),
                    "domain": pattern_data.get("domain", "general")
                },
                usage_frequency=1,
                effectiveness_score=pattern_data.get("score", 0.8),
                created_at=datetime.now(timezone.utc).isoformat(),
                last_used=datetime.now(timezone.utc).isoformat()
            )
            
            # Store as memory
            memory_id = await self.domain_manager.store_memory(
                memory_type="mcp_workflow_pattern",
                content=json.dumps(pattern.to_dict()),
                importance=0.9,
                metadata={
                    "category": "workflow_patterns",
                    "pattern_type": "mcp_usage",
                    "tools": pattern_data.get("tools", []),
                    "effectiveness": pattern_data.get("score", 0.8),
                    "auto_learned": True,
                    "pattern_id": pattern_id,
                    "context_keywords": self._extract_keywords(pattern_data.get("context", "")),
                    "success_factors": list(pattern_data.get("success", {}).keys()) if isinstance(pattern_data.get("success"), dict) else []
                }
            )
            
            # Update cache
            self.pattern_cache[pattern_id] = pattern
            self._cache_timestamps[pattern_id] = asyncio.get_event_loop().time()
            
            logger.info(f"Stored MCP workflow pattern: {pattern_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store MCP workflow pattern: {e}")
            raise
    
    async def store_mcp_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """Store individual MCP interaction for learning."""
        try:
            interaction = MCPInteraction.from_dict(interaction_data)
            
            # Store as memory
            memory_id = await self.domain_manager.store_memory(
                memory_type="mcp_interaction",
                content=json.dumps(interaction.to_dict()),
                importance=0.7 if interaction.success else 0.5,
                metadata={
                    "category": "mcp_interactions",
                    "server": interaction.server_name,
                    "tool": interaction.tool_name,
                    "success": interaction.success,
                    "duration_ms": interaction.duration_ms,
                    "user_intent": interaction.user_intent,
                    "auto_learned": True
                }
            )
            
            # Add to history for trend analysis
            self.interaction_history.append(interaction)
            if len(self.interaction_history) > 1000:
                self.interaction_history = self.interaction_history[-500:]
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store MCP interaction: {e}")
            raise
    
    async def find_similar_workflows(self, context: str, limit: int = 5) -> List[MCPWorkflowPattern]:
        """Find similar successful MCP workflows."""
        try:
            # Query memory system
            query_result = await self.domain_manager.retrieve_memories(
                query=context,
                types=["mcp_workflow_pattern"],
                limit=limit * 2,  # Get more to filter and rank
                min_similarity=0.6
            )
            
            patterns = []
            for memory in query_result:
                try:
                    pattern_data = json.loads(memory['content'])
                    pattern = MCPWorkflowPattern.from_dict(pattern_data)
                    patterns.append(pattern)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Could not parse pattern from memory: {e}")
                    continue
            
            # Score and rank patterns
            scored_patterns = []
            for pattern in patterns:
                score = await self._score_pattern_relevance(pattern, context)
                scored_patterns.append((score, pattern))
            
            # Sort by score and return top results
            scored_patterns.sort(key=lambda x: x[0], reverse=True)
            return [pattern for _, pattern in scored_patterns[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to find similar workflows: {e}")
            return []
    
    async def get_workflow_suggestions(self, suggestion_context: SuggestionContext) -> List[Dict[str, Any]]:
        """Get workflow suggestions based on context."""
        try:
            # Find relevant patterns
            similar_patterns = await self.find_similar_workflows(
                suggestion_context.current_task + " " + suggestion_context.user_intent
            )
            
            suggestions = []
            for pattern in similar_patterns:
                suggestion = {
                    "type": "workflow_pattern",
                    "pattern_id": pattern.pattern_id,
                    "confidence": pattern.effectiveness_score,
                    "suggested_tools": pattern.tool_sequence,
                    "resource_references": pattern.resource_references,
                    "context": pattern.trigger_context,
                    "success_factors": pattern.success_indicators,
                    "usage_count": pattern.usage_frequency,
                    "description": self._generate_pattern_description(pattern),
                    "applicability_score": await self._calculate_applicability(
                        pattern, suggestion_context
                    )
                }
                suggestions.append(suggestion)
            
            # Also get server-specific suggestions
            server_suggestions = await self._get_server_specific_suggestions(suggestion_context)
            suggestions.extend(server_suggestions)
            
            # Sort by applicability score
            suggestions.sort(key=lambda x: x.get('applicability_score', 0), reverse=True)
            
            return suggestions[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Failed to get workflow suggestions: {e}")
            return []
    
    async def update_pattern_usage(self, pattern_id: str, success: bool) -> bool:
        """Update pattern usage statistics."""
        try:
            # Update in cache if present
            if pattern_id in self.pattern_cache:
                pattern = self.pattern_cache[pattern_id]
                pattern.usage_frequency += 1
                pattern.last_used = datetime.now(timezone.utc).isoformat()
                
                if success:
                    # Increase effectiveness score slightly
                    pattern.effectiveness_score = min(1.0, pattern.effectiveness_score + 0.05)
                else:
                    # Decrease slightly
                    pattern.effectiveness_score = max(0.0, pattern.effectiveness_score - 0.02)
            
            # Update in memory system
            memories = await self.domain_manager.retrieve_memories(
                query=f"pattern_id:{pattern_id}",
                types=["mcp_workflow_pattern"],
                limit=1
            )
            
            if memories:
                memory = memories[0]
                pattern_data = json.loads(memory['content'])
                pattern_data['usage_frequency'] += 1
                pattern_data['last_used'] = datetime.now(timezone.utc).isoformat()
                
                if success:
                    pattern_data['effectiveness_score'] = min(
                        1.0, pattern_data['effectiveness_score'] + 0.05
                    )
                else:
                    pattern_data['effectiveness_score'] = max(
                        0.0, pattern_data['effectiveness_score'] - 0.02
                    )
                
                # Update memory
                await self.domain_manager.update_memory(
                    memory['id'],
                    {"content": json.dumps(pattern_data)}
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update pattern usage: {e}")
            return False
    
    async def get_pattern_analytics(self) -> Dict[str, Any]:
        """Get analytics about stored patterns and interactions."""
        try:
            # Get all patterns
            pattern_memories = await self.domain_manager.retrieve_memories(
                query="",
                types=["mcp_workflow_pattern"],
                limit=100
            )
            
            # Get all interactions
            interaction_memories = await self.domain_manager.retrieve_memories(
                query="",
                types=["mcp_interaction"],
                limit=500
            )
            
            analytics = {
                "total_patterns": len(pattern_memories),
                "total_interactions": len(interaction_memories),
                "pattern_effectiveness": [],
                "popular_tools": {},
                "success_rate_by_server": {},
                "common_contexts": {},
                "recent_activity": 0
            }
            
            # Analyze patterns
            effectiveness_scores = []
            for memory in pattern_memories:
                try:
                    pattern_data = json.loads(memory['content'])
                    effectiveness_scores.append(pattern_data.get('effectiveness_score', 0))
                    
                    # Count tools
                    for tool in pattern_data.get('tool_sequence', []):
                        analytics["popular_tools"][tool] = analytics["popular_tools"].get(tool, 0) + 1
                    
                    # Count contexts
                    context = pattern_data.get('trigger_context', '')[:50]  # First 50 chars
                    analytics["common_contexts"][context] = analytics["common_contexts"].get(context, 0) + 1
                    
                except (json.JSONDecodeError, KeyError):
                    continue
            
            if effectiveness_scores:
                analytics["pattern_effectiveness"] = {
                    "average": sum(effectiveness_scores) / len(effectiveness_scores),
                    "min": min(effectiveness_scores),
                    "max": max(effectiveness_scores)
                }
            
            # Analyze interactions
            server_stats = {}
            recent_time = datetime.now(timezone.utc).timestamp() - 3600  # Last hour
            
            for memory in interaction_memories:
                try:
                    interaction_data = json.loads(memory['content'])
                    server = interaction_data.get('server_name', 'unknown')
                    success = interaction_data.get('success', False)
                    
                    if server not in server_stats:
                        server_stats[server] = {'total': 0, 'success': 0}
                    
                    server_stats[server]['total'] += 1
                    if success:
                        server_stats[server]['success'] += 1
                    
                    # Check if recent
                    timestamp_str = interaction_data.get('timestamp', '')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
                        if timestamp > recent_time:
                            analytics["recent_activity"] += 1
                    except (ValueError, AttributeError):
                        pass
                        
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Calculate success rates
            for server, stats in server_stats.items():
                if stats['total'] > 0:
                    analytics["success_rate_by_server"][server] = stats['success'] / stats['total']
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get pattern analytics: {e}")
            return {}
    
    async def _score_pattern_relevance(self, pattern: MCPWorkflowPattern, context: str) -> float:
        """Score pattern relevance to current context."""
        score = 0.0
        
        # Context similarity (basic keyword matching)
        context_words = set(context.lower().split())
        pattern_words = set(pattern.trigger_context.lower().split())
        
        if context_words and pattern_words:
            intersection = context_words.intersection(pattern_words)
            union = context_words.union(pattern_words)
            context_similarity = len(intersection) / len(union) if union else 0
            score += self.scoring_weights['context_similarity'] * context_similarity
        
        # Effectiveness score
        score += self.scoring_weights['effectiveness_score'] * pattern.effectiveness_score
        
        # Usage frequency (normalized to 0-1)
        freq_score = min(1.0, pattern.usage_frequency / 10)
        score += self.scoring_weights['usage_frequency'] * freq_score
        
        # Recency (patterns used recently are more relevant)
        try:
            last_used = datetime.fromisoformat(pattern.last_used.replace('Z', '+00:00'))
            hours_ago = (datetime.now(timezone.utc) - last_used).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_ago / 168))  # Decay over a week
            score += self.scoring_weights['recency'] * recency_score
        except (ValueError, AttributeError):
            pass
        
        return min(1.0, score)
    
    async def _calculate_applicability(self, pattern: MCPWorkflowPattern, context: SuggestionContext) -> float:
        """Calculate how applicable a pattern is to the current context."""
        applicability = 0.0
        
        # Check if required tools/servers are available
        available_tools = set(context.recent_tools_used + context.available_servers)
        pattern_tools = set(pattern.tool_sequence)
        
        if pattern_tools:
            tool_availability = len(pattern_tools.intersection(available_tools)) / len(pattern_tools)
            applicability += 0.4 * tool_availability
        
        # Check project type match
        pattern_project_type = pattern.contextual_factors.get('project_type')
        if pattern_project_type and context.project_type:
            if pattern_project_type == context.project_type:
                applicability += 0.2
        
        # Check environment compatibility
        pattern_env = pattern.contextual_factors.get('environment', {})
        context_env = context.environment_info
        
        env_match = 0
        for key, value in pattern_env.items():
            if key in context_env and context_env[key] == value:
                env_match += 1
        
        if pattern_env:
            applicability += 0.2 * (env_match / len(pattern_env))
        
        # Base effectiveness
        applicability += 0.2 * pattern.effectiveness_score
        
        return min(1.0, applicability)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for better matching."""
        import re
        
        # Handle None or empty text
        if not text:
            return []
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]  # Return top 10 keywords
    
    def _generate_pattern_description(self, pattern: MCPWorkflowPattern) -> str:
        """Generate human-readable description of pattern."""
        tools_str = ", ".join(pattern.tool_sequence[:3])
        if len(pattern.tool_sequence) > 3:
            tools_str += f" and {len(pattern.tool_sequence) - 3} more"
        
        context_snippet = pattern.trigger_context[:50] + "..." if len(pattern.trigger_context) > 50 else pattern.trigger_context
        
        return f"Use {tools_str} for '{context_snippet}' (used {pattern.usage_frequency} times, {pattern.effectiveness_score:.1%} effective)"
    
    async def _get_server_specific_suggestions(self, context: SuggestionContext) -> List[Dict[str, Any]]:
        """Get suggestions specific to available MCP servers."""
        suggestions = []
        
        # Analyze recent failures to suggest alternatives
        for failure in context.recent_failures:
            if 'file' in failure.lower() and 'filesystem' in context.available_servers:
                suggestions.append({
                    "type": "server_alternative",
                    "server": "filesystem",
                    "confidence": 0.8,
                    "description": "Consider using filesystem MCP server for file operations",
                    "applicability_score": 0.7,
                    "reason": f"Alternative for failed: {failure}"
                })
            
            elif 'database' in failure.lower() or 'sql' in failure.lower():
                db_servers = [s for s in context.available_servers if 'postgres' in s or 'mysql' in s or 'database' in s]
                for server in db_servers:
                    suggestions.append({
                        "type": "server_alternative", 
                        "server": server,
                        "confidence": 0.8,
                        "description": f"Consider using {server} MCP server for database operations",
                        "applicability_score": 0.8,
                        "reason": f"Alternative for failed: {failure}"
                    })
        
        # Proactive suggestions based on current task and available servers
        task_lower = context.current_task.lower()
        intent_lower = context.user_intent.lower()
        combined_context = f"{task_lower} {intent_lower}"
        
        # Database-related suggestions
        if any(keyword in combined_context for keyword in ['data', 'user', 'analytics', 'query', 'database', 'behavior']):
            db_servers = [s for s in context.available_servers if 'postgres' in s or 'mysql' in s or 'database' in s]
            for server in db_servers:
                suggestions.append({
                    "type": "proactive_suggestion",
                    "server": server,
                    "confidence": 0.85,
                    "suggested_tools": [f"{server}_query"],
                    "description": f"Use {server} MCP server for data analysis queries",
                    "applicability_score": 0.85,
                    "reason": "Data analysis tasks often require database queries"
                })
        
        # File-related suggestions  
        if any(keyword in combined_context for keyword in ['file', 'config', 'read', 'write', 'report', 'csv']):
            if 'filesystem' in context.available_servers:
                # Higher score for explicit file operations
                file_score = 0.95 if any(word in combined_context for word in ['read', 'csv', 'file']) else 0.85
                suggestions.append({
                    "type": "proactive_suggestion", 
                    "server": "filesystem",
                    "confidence": 0.9,
                    "suggested_tools": ["read_file", "write_file"],
                    "description": "Use filesystem MCP server for file operations",
                    "applicability_score": file_score,
                    "reason": "Task involves file operations"
                })
        
        # Web/API-related suggestions
        if any(keyword in combined_context for keyword in ['api', 'http', 'web', 'fetch', 'get', 'post', 'put']):
            if 'web' in context.available_servers:
                # Higher score for explicit API operations
                api_score = 0.95 if any(word in combined_context for word in ['api', 'fetch', 'http']) else 0.85
                suggestions.append({
                    "type": "proactive_suggestion",
                    "server": "web", 
                    "confidence": 0.9,
                    "suggested_tools": ["http_get", "http_post", "http_put"],
                    "description": "Use web MCP server for API operations",
                    "applicability_score": api_score,
                    "reason": "Task involves API operations"
                })
        
        return suggestions