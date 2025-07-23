"""
Enhanced Memory Schema for MCP Workflow Pattern Storage.

This module defines enhanced memory structures and storage patterns for
capturing and learning from MCP usage workflows, tool sequences, and
contextual patterns.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from ..domains.domain_manager import DomainManager


class WorkflowStage(Enum):
    """Stages in a workflow pattern."""
    INITIATION = "initiation"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"


class PatternType(Enum):
    """Types of workflow patterns."""
    MCP_USAGE = "mcp_usage"
    TOOL_SEQUENCE = "tool_sequence"
    RESOURCE_ACCESS = "resource_access"
    ERROR_RECOVERY = "error_recovery"
    OPTIMIZATION = "optimization"


class ContextType(Enum):
    """Types of context information."""
    PROJECT = "project"
    TASK = "task"
    USER_INTENT = "user_intent"
    ENVIRONMENT = "environment"
    TECHNICAL = "technical"


@dataclass
class WorkflowStep:
    """A single step in a workflow pattern."""
    step_number: int
    step_type: str  # "tool_use", "resource_access", "validation", "decision"
    tool_name: Optional[str]
    parameters: Dict[str, Any]
    expected_outcome: str
    actual_outcome: Optional[str] = None
    success: Optional[bool] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class ContextualFactor:
    """A contextual factor that influenced the workflow."""
    factor_type: ContextType
    name: str
    value: Any
    importance: float  # 0.0 to 1.0
    influence_on_outcome: str  # Description of how this factor influenced the workflow


@dataclass
class WorkflowPattern:
    """A complete workflow pattern."""
    pattern_id: str
    pattern_type: PatternType
    pattern_name: str
    description: str
    
    # Workflow structure
    steps: List[WorkflowStep]
    success_indicators: List[str]
    failure_indicators: List[str]
    
    # Context information
    contextual_factors: List[ContextualFactor]
    trigger_conditions: List[str]
    environment_requirements: Dict[str, Any]
    
    # Learning metrics
    success_rate: float
    usage_count: int
    last_used: datetime
    effectiveness_score: float
    
    # Relationships
    related_patterns: List[str]  # IDs of related patterns
    superseded_patterns: List[str]  # Patterns this one replaces
    variations: List[str]  # Pattern variations
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    learned_from: List[str]  # Sources this pattern was learned from
    confidence_score: float


@dataclass
class MCPInteraction:
    """Details of an MCP tool interaction within a workflow."""
    tool_name: str
    server_name: str
    interaction_type: str  # "direct", "via_hook", "suggested", "fallback"
    parameters_used: Dict[str, Any]
    response_data: Any
    response_time_ms: float
    success: bool
    error_details: Optional[Dict[str, Any]] = None
    alternative_tools_considered: List[str] = None


@dataclass
class SuggestionContext:
    """Context for when and how to suggest a pattern."""
    trigger_keywords: List[str]
    context_requirements: Dict[str, Any]
    user_intent_patterns: List[str]
    environmental_conditions: Dict[str, Any]
    success_probability: float
    suggestion_timing: str  # "preemptive", "during_execution", "post_error"


class EnhancedMemorySchema:
    """
    Enhanced memory schema for storing complex MCP workflow patterns.
    
    This class provides structured storage and retrieval of workflow patterns,
    tool usage sequences, and contextual learning from MCP interactions.
    """
    
    def __init__(self, domain_manager: DomainManager):
        """
        Initialize the enhanced memory schema.
        
        Args:
            domain_manager: Domain manager for memory operations
        """
        self.domain_manager = domain_manager
        
        # Memory type definitions
        self.memory_types = {
            "workflow_pattern": "Complete workflow patterns with steps and context",
            "mcp_interaction": "Individual MCP tool interactions",
            "suggestion_context": "Context patterns for when to suggest workflows",
            "pattern_relationship": "Relationships between different patterns",
            "optimization_insight": "Insights about workflow optimizations",
            "error_recovery_pattern": "Patterns for recovering from errors",
            "contextual_learning": "Context-specific learning patterns"
        }
    
    async def store_workflow_pattern(
        self, 
        pattern: WorkflowPattern
    ) -> str:
        """
        Store a complete workflow pattern.
        
        Args:
            pattern: The workflow pattern to store
            
        Returns:
            Memory ID of the stored pattern
        """
        try:
            # Convert pattern to dictionary for storage
            pattern_dict = asdict(pattern)
            
            # Handle datetime serialization
            pattern_dict['last_used'] = pattern.last_used.isoformat()
            pattern_dict['created_at'] = pattern.created_at.isoformat()
            pattern_dict['updated_at'] = pattern.updated_at.isoformat()
            
            # Create rich content description for vector search
            content_description = self._generate_pattern_description(pattern)
            
            memory_content = {
                "pattern": pattern_dict,
                "description": content_description,
                "searchable_keywords": self._extract_searchable_keywords(pattern),
                "workflow_summary": self._generate_workflow_summary(pattern)
            }
            
            memory_id = await self.domain_manager.store_memory(
                memory_type="workflow_pattern",
                content=memory_content,
                importance=self._calculate_pattern_importance(pattern),
                metadata={
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type.value,
                    "pattern_name": pattern.pattern_name,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "effectiveness_score": pattern.effectiveness_score,
                    "step_count": len(pattern.steps),
                    "mcp_tools_used": self._extract_mcp_tools(pattern),
                    "auto_learned": True
                },
                context={
                    "purpose": "workflow_pattern_storage",
                    "pattern_type": pattern.pattern_type.value,
                    "learning_source": "mcp_usage_analysis"
                }
            )
            
            logger.debug(f"Stored workflow pattern {pattern.pattern_id} with memory ID {memory_id}")
            return memory_id
        
        except Exception as e:
            logger.error(f"Failed to store workflow pattern: {e}")
            raise
    
    async def store_mcp_interaction(
        self, 
        interaction: MCPInteraction,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an individual MCP interaction.
        
        Args:
            interaction: The MCP interaction to store
            workflow_context: Optional context about the broader workflow
            
        Returns:
            Memory ID of the stored interaction
        """
        try:
            interaction_dict = asdict(interaction)
            
            memory_content = {
                "interaction": interaction_dict,
                "interaction_summary": self._summarize_interaction(interaction),
                "performance_metrics": {
                    "response_time_ms": interaction.response_time_ms,
                    "success": interaction.success,
                    "tool_effectiveness": self._calculate_tool_effectiveness(interaction)
                },
                "workflow_context": workflow_context or {}
            }
            
            memory_id = await self.domain_manager.store_memory(
                memory_type="mcp_interaction",
                content=memory_content,
                importance=0.7 if interaction.success else 0.5,
                metadata={
                    "tool_name": interaction.tool_name,
                    "server_name": interaction.server_name,
                    "interaction_type": interaction.interaction_type,
                    "success": interaction.success,
                    "response_time_category": self._categorize_response_time(interaction.response_time_ms),
                    "auto_logged": True
                },
                context={
                    "purpose": "mcp_interaction_logging",
                    "tool_category": self._categorize_tool(interaction.tool_name),
                    "performance_tracking": True
                }
            )
            
            return memory_id
        
        except Exception as e:
            logger.error(f"Failed to store MCP interaction: {e}")
            raise
    
    async def store_suggestion_context(
        self, 
        context: SuggestionContext,
        associated_pattern_id: str
    ) -> str:
        """
        Store context patterns for when to suggest workflows.
        
        Args:
            context: Suggestion context information
            associated_pattern_id: ID of the pattern this context applies to
            
        Returns:
            Memory ID of the stored context
        """
        try:
            context_dict = asdict(context)
            
            memory_content = {
                "suggestion_context": context_dict,
                "associated_pattern": associated_pattern_id,
                "context_description": self._describe_suggestion_context(context),
                "trigger_summary": self._summarize_triggers(context)
            }
            
            memory_id = await self.domain_manager.store_memory(
                memory_type="suggestion_context",
                content=memory_content,
                importance=context.success_probability,
                metadata={
                    "associated_pattern": associated_pattern_id,
                    "suggestion_timing": context.suggestion_timing,
                    "trigger_count": len(context.trigger_keywords),
                    "success_probability": context.success_probability,
                    "auto_generated": True
                },
                context={
                    "purpose": "suggestion_optimization",
                    "learning_type": "contextual_triggers"
                }
            )
            
            return memory_id
        
        except Exception as e:
            logger.error(f"Failed to store suggestion context: {e}")
            raise
    
    async def retrieve_workflow_patterns(
        self, 
        query: str,
        pattern_type: Optional[PatternType] = None,
        min_success_rate: float = 0.0,
        limit: int = 10
    ) -> List[WorkflowPattern]:
        """
        Retrieve workflow patterns based on query and criteria.
        
        Args:
            query: Search query for relevant patterns
            pattern_type: Optional pattern type filter
            min_success_rate: Minimum success rate filter
            limit: Maximum number of patterns to return
            
        Returns:
            List of matching workflow patterns
        """
        try:
            # Build query filters
            query_filters = []
            if pattern_type:
                query_filters.append(f"pattern_type:{pattern_type.value}")
            if min_success_rate > 0:
                query_filters.append(f"success_rate:>={min_success_rate}")
            
            search_query = f"{query} {' '.join(query_filters)}".strip()
            
            memories = await self.domain_manager.retrieve_memory(
                query=search_query,
                types=["workflow_pattern"],
                limit=limit
            )
            
            patterns = []
            for memory in memories:
                content = memory.get('content', {})
                pattern_dict = content.get('pattern', {})
                
                if pattern_dict:
                    pattern = self._deserialize_workflow_pattern(pattern_dict)
                    if pattern:
                        patterns.append(pattern)
            
            return patterns
        
        except Exception as e:
            logger.error(f"Failed to retrieve workflow patterns: {e}")
            return []
    
    async def retrieve_similar_interactions(
        self, 
        tool_name: str,
        success_only: bool = True,
        limit: int = 5
    ) -> List[MCPInteraction]:
        """
        Retrieve similar MCP interactions for learning purposes.
        
        Args:
            tool_name: Name of the MCP tool
            success_only: Whether to only return successful interactions
            limit: Maximum number of interactions to return
            
        Returns:
            List of similar MCP interactions
        """
        try:
            query = f"MCP interaction {tool_name}"
            if success_only:
                query += " success"
            
            memories = await self.domain_manager.retrieve_memory(
                query=query,
                types=["mcp_interaction"],
                limit=limit
            )
            
            interactions = []
            for memory in memories:
                content = memory.get('content', {})
                interaction_dict = content.get('interaction', {})
                
                if interaction_dict:
                    interaction = self._deserialize_mcp_interaction(interaction_dict)
                    if interaction:
                        interactions.append(interaction)
            
            return interactions
        
        except Exception as e:
            logger.error(f"Failed to retrieve similar interactions: {e}")
            return []
    
    async def find_optimization_opportunities(
        self, 
        current_workflow: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find optimization opportunities for a current workflow.
        
        Args:
            current_workflow: List of current workflow steps/tools
            
        Returns:
            List of optimization suggestions
        """
        try:
            # Search for patterns that include similar tool sequences
            tool_query = " ".join(current_workflow[:3])  # Use first 3 tools for search
            
            patterns = await self.retrieve_workflow_patterns(
                query=f"workflow optimization {tool_query}",
                min_success_rate=0.8,
                limit=5
            )
            
            optimizations = []
            for pattern in patterns:
                if pattern.effectiveness_score > 0.8:
                    optimization = {
                        "pattern_id": pattern.pattern_id,
                        "optimization_type": "workflow_replacement",
                        "current_steps": current_workflow,
                        "suggested_steps": [step.tool_name for step in pattern.steps if step.tool_name],
                        "expected_improvement": pattern.effectiveness_score,
                        "reasoning": f"Pattern {pattern.pattern_name} has {pattern.effectiveness_score:.0%} effectiveness",
                        "success_rate": pattern.success_rate
                    }
                    optimizations.append(optimization)
            
            return optimizations
        
        except Exception as e:
            logger.error(f"Failed to find optimization opportunities: {e}")
            return []
    
    def _generate_pattern_description(self, pattern: WorkflowPattern) -> str:
        """Generate a rich description of the workflow pattern for search."""
        description_parts = [
            f"Workflow pattern: {pattern.pattern_name}",
            f"Description: {pattern.description}",
            f"Type: {pattern.pattern_type.value}",
            f"Steps: {len(pattern.steps)} step workflow"
        ]
        
        # Add tool information
        tools = [step.tool_name for step in pattern.steps if step.tool_name]
        if tools:
            description_parts.append(f"Tools used: {', '.join(set(tools))}")
        
        # Add success information
        description_parts.append(f"Success rate: {pattern.success_rate:.0%}")
        description_parts.append(f"Effectiveness: {pattern.effectiveness_score:.0%}")
        
        # Add context information
        contexts = [factor.name for factor in pattern.contextual_factors]
        if contexts:
            description_parts.append(f"Context factors: {', '.join(contexts)}")
        
        return ". ".join(description_parts)
    
    def _extract_searchable_keywords(self, pattern: WorkflowPattern) -> List[str]:
        """Extract searchable keywords from a workflow pattern."""
        keywords = set()
        
        # Pattern metadata
        keywords.add(pattern.pattern_type.value)
        keywords.update(pattern.pattern_name.lower().split())
        keywords.update(pattern.description.lower().split())
        
        # Tools and steps
        for step in pattern.steps:
            if step.tool_name:
                keywords.add(step.tool_name.lower())
            keywords.add(step.step_type.lower())
            keywords.update(step.expected_outcome.lower().split())
        
        # Contextual factors
        for factor in pattern.contextual_factors:
            keywords.add(factor.name.lower())
            keywords.add(factor.factor_type.value)
        
        # Trigger conditions
        for trigger in pattern.trigger_conditions:
            keywords.update(trigger.lower().split())
        
        return list(keywords)
    
    def _generate_workflow_summary(self, pattern: WorkflowPattern) -> str:
        """Generate a concise summary of the workflow."""
        tool_chain = []
        for step in pattern.steps:
            if step.tool_name:
                tool_chain.append(step.tool_name)
            else:
                tool_chain.append(f"({step.step_type})")
        
        summary = f"{pattern.pattern_name}: {' → '.join(tool_chain)}"
        
        if pattern.success_rate > 0:
            summary += f" (Success: {pattern.success_rate:.0%})"
        
        return summary
    
    def _calculate_pattern_importance(self, pattern: WorkflowPattern) -> float:
        """Calculate the importance score for a pattern."""
        base_importance = 0.7
        
        # Adjust based on success rate
        success_bonus = pattern.success_rate * 0.2
        
        # Adjust based on usage count
        usage_bonus = min(pattern.usage_count * 0.05, 0.1)
        
        # Adjust based on effectiveness
        effectiveness_bonus = pattern.effectiveness_score * 0.1
        
        return min(base_importance + success_bonus + usage_bonus + effectiveness_bonus, 1.0)
    
    def _extract_mcp_tools(self, pattern: WorkflowPattern) -> List[str]:
        """Extract MCP tool names from a workflow pattern."""
        mcp_tools = []
        
        for step in pattern.steps:
            if step.tool_name and self._is_mcp_tool(step.tool_name):
                mcp_tools.append(step.tool_name)
        
        return mcp_tools
    
    def _is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool name represents an MCP tool."""
        mcp_indicators = [
            "mcp__",
            "_query", "_execute", "_store", "_retrieve",
            "postgres", "playwright", "memory", "filesystem"
        ]
        
        return any(indicator in tool_name.lower() for indicator in mcp_indicators)
    
    def _summarize_interaction(self, interaction: MCPInteraction) -> str:
        """Generate a summary of an MCP interaction."""
        summary = f"{interaction.tool_name} on {interaction.server_name}"
        
        if interaction.success:
            summary += f" (Success in {interaction.response_time_ms:.0f}ms)"
        else:
            summary += " (Failed)"
            if interaction.error_details:
                error_type = interaction.error_details.get('type', 'Unknown')
                summary += f" - {error_type}"
        
        return summary
    
    def _calculate_tool_effectiveness(self, interaction: MCPInteraction) -> float:
        """Calculate the effectiveness of a tool interaction."""
        if not interaction.success:
            return 0.0
        
        # Base effectiveness
        effectiveness = 0.8
        
        # Adjust based on response time
        if interaction.response_time_ms < 100:
            effectiveness += 0.2
        elif interaction.response_time_ms > 5000:
            effectiveness -= 0.2
        
        # Adjust if alternatives were considered (shows this was the best choice)
        if interaction.alternative_tools_considered:
            effectiveness += 0.1
        
        return max(0.0, min(effectiveness, 1.0))
    
    def _categorize_response_time(self, response_time_ms: float) -> str:
        """Categorize response time performance."""
        if response_time_ms < 100:
            return "fast"
        elif response_time_ms < 1000:
            return "normal"
        elif response_time_ms < 5000:
            return "slow"
        else:
            return "very_slow"
    
    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize a tool by its apparent function."""
        tool_categories = {
            "database": ["postgres", "mysql", "sqlite", "query", "db"],
            "web": ["playwright", "browser", "selenium", "web", "http"],
            "file": ["filesystem", "file", "read", "write", "fs"],
            "memory": ["memory", "store", "retrieve", "recall"],
            "api": ["api", "request", "http", "rest"],
            "development": ["git", "build", "test", "deploy", "docker"]
        }
        
        tool_lower = tool_name.lower()
        
        for category, keywords in tool_categories.items():
            if any(keyword in tool_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _describe_suggestion_context(self, context: SuggestionContext) -> str:
        """Generate a description of suggestion context."""
        description = f"Suggest when: {', '.join(context.trigger_keywords[:3])}"
        
        if context.user_intent_patterns:
            description += f" for intents: {', '.join(context.user_intent_patterns[:2])}"
        
        description += f" (Success probability: {context.success_probability:.0%})"
        
        return description
    
    def _summarize_triggers(self, context: SuggestionContext) -> str:
        """Summarize trigger conditions."""
        return f"{len(context.trigger_keywords)} triggers, timing: {context.suggestion_timing}"
    
    def _deserialize_workflow_pattern(self, pattern_dict: Dict[str, Any]) -> Optional[WorkflowPattern]:
        """Deserialize a workflow pattern from stored dictionary."""
        try:
            # Convert datetime strings back to datetime objects
            pattern_dict['last_used'] = datetime.fromisoformat(pattern_dict['last_used'])
            pattern_dict['created_at'] = datetime.fromisoformat(pattern_dict['created_at'])
            pattern_dict['updated_at'] = datetime.fromisoformat(pattern_dict['updated_at'])
            
            # Convert enum strings back to enums
            pattern_dict['pattern_type'] = PatternType(pattern_dict['pattern_type'])
            
            # Deserialize steps
            steps = []
            for step_dict in pattern_dict.get('steps', []):
                step = WorkflowStep(**step_dict)
                steps.append(step)
            pattern_dict['steps'] = steps
            
            # Deserialize contextual factors
            factors = []
            for factor_dict in pattern_dict.get('contextual_factors', []):
                factor_dict['factor_type'] = ContextType(factor_dict['factor_type'])
                factor = ContextualFactor(**factor_dict)
                factors.append(factor)
            pattern_dict['contextual_factors'] = factors
            
            return WorkflowPattern(**pattern_dict)
        
        except Exception as e:
            logger.warning(f"Failed to deserialize workflow pattern: {e}")
            return None
    
    def _deserialize_mcp_interaction(self, interaction_dict: Dict[str, Any]) -> Optional[MCPInteraction]:
        """Deserialize an MCP interaction from stored dictionary."""
        try:
            return MCPInteraction(**interaction_dict)
        except Exception as e:
            logger.warning(f"Failed to deserialize MCP interaction: {e}")
            return None