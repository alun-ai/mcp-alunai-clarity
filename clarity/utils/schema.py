"""
MCP-Enhanced Memory Schema Validation.

This module provides modernized memory schemas with integrated MCP workflow
patterns and cross-system intelligence. Legacy memory types have been
deprecated in favor of MCP-aware memory structures.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class MCPIntegrationType(str, Enum):
    """Types of MCP integration for memory entries."""
    NONE = "none"
    WORKFLOW_PATTERN = "workflow_pattern"
    TOOL_CORRELATION = "tool_correlation"
    RESOURCE_REFERENCE = "resource_reference"
    THINKING_INTEGRATION = "thinking_integration"
    CROSS_SYSTEM_INTELLIGENCE = "cross_system_intelligence"


class MemoryBase(BaseModel):
    """Enhanced base model for MCP-integrated memory objects."""
    id: str
    type: str
    importance: float = 0.5
    mcp_integration: MCPIntegrationType = MCPIntegrationType.NONE
    mcp_context: Optional[Dict[str, Any]] = None
    workflow_patterns: Optional[List[str]] = None
    cache_metadata: Optional[Dict[str, Any]] = None
    
    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v: float) -> float:
        """Validate importance score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")
        return v
    
    @field_validator("workflow_patterns")
    @classmethod
    def validate_workflow_patterns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate workflow pattern references."""
        if v is not None:
            for pattern in v:
                if not isinstance(pattern, str) or not pattern:
                    raise ValueError("Workflow patterns must be non-empty strings")
        return v


class MCPThinkingWorkflowMemory(MemoryBase):
    """MCP-enhanced memory for structured thinking workflows."""
    type: str = "mcp_thinking_workflow"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.THINKING_INTEGRATION
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCP thinking workflow content."""
        required_fields = ["session_id", "thinking_stage", "workflow_pattern", "tool_context"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"MCP thinking workflow must have '{field}' field")
        
        # Validate thinking stage
        valid_stages = ["problem_definition", "research", "analysis", "synthesis", "conclusion", "mcp_integration"]
        if v["thinking_stage"] not in valid_stages:
            raise ValueError(f"Thinking stage must be one of: {valid_stages}")
        
        # Validate workflow pattern structure
        workflow = v["workflow_pattern"]
        if not isinstance(workflow, dict) or "trigger_context" not in workflow:
            raise ValueError("Workflow pattern must be dict with 'trigger_context'")
        
        # Validate tool context
        tool_context = v["tool_context"]
        if not isinstance(tool_context, dict):
            raise ValueError("Tool context must be a dictionary")
            
        return v


class MCPResourcePatternMemory(MemoryBase):
    """MCP-enhanced memory for resource patterns and references."""
    type: str = "mcp_resource_pattern"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.RESOURCE_REFERENCE
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCP resource pattern content."""
        required_fields = ["resource_reference", "access_pattern", "success_metrics", "mcp_server_context"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"MCP resource pattern must have '{field}' field")
        
        # Validate resource reference format
        resource_ref = v["resource_reference"]
        if not isinstance(resource_ref, str) or not resource_ref.startswith("@memory:"):
            raise ValueError("Resource reference must start with '@memory:'")
        
        # Validate success metrics
        success_metrics = v["success_metrics"]
        if not isinstance(success_metrics, dict) or "effectiveness_score" not in success_metrics:
            raise ValueError("Success metrics must be dict with 'effectiveness_score'")
        
        if not 0.0 <= success_metrics["effectiveness_score"] <= 1.0:
            raise ValueError("Effectiveness score must be between 0.0 and 1.0")
            
        return v


class ThinkingMCPIntegrationMemory(MemoryBase):
    """Memory for cross-system thinking integration with MCP."""
    type: str = "thinking_mcp_integration"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.CROSS_SYSTEM_INTELLIGENCE
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate thinking MCP integration content."""
        required_fields = ["integration_type", "correlation_data", "performance_impact", "usage_context"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Thinking MCP integration must have '{field}' field")
        
        # Validate integration type
        valid_types = ["cache_intelligence", "workflow_correlation", "real_time_enhancement", "pattern_discovery"]
        if v["integration_type"] not in valid_types:
            raise ValueError(f"Integration type must be one of: {valid_types}")
        
        # Validate correlation data
        correlation = v["correlation_data"]
        if not isinstance(correlation, dict) or "correlation_strength" not in correlation:
            raise ValueError("Correlation data must be dict with 'correlation_strength'")
        
        if not 0.0 <= correlation["correlation_strength"] <= 1.0:
            raise ValueError("Correlation strength must be between 0.0 and 1.0")
            
        return v


class MCPToolCorrelationMemory(MemoryBase):
    """Memory for correlations between tools and thinking patterns."""
    type: str = "mcp_tool_correlation"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.TOOL_CORRELATION
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCP tool correlation content."""
        required_fields = ["tool_name", "thinking_context", "correlation_patterns", "effectiveness_data"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"MCP tool correlation must have '{field}' field")
        
        # Validate tool name
        if not isinstance(v["tool_name"], str) or not v["tool_name"]:
            raise ValueError("Tool name must be non-empty string")
        
        # Validate correlation patterns
        patterns = v["correlation_patterns"]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError("Correlation patterns must be non-empty list")
        
        # Validate effectiveness data
        effectiveness = v["effectiveness_data"]
        if not isinstance(effectiveness, dict) or "success_rate" not in effectiveness:
            raise ValueError("Effectiveness data must be dict with 'success_rate'")
            
        return v


class MCPWorkflowPatternMemory(MemoryBase):
    """Memory for MCP workflow patterns and their effectiveness."""
    type: str = "mcp_workflow_pattern"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.WORKFLOW_PATTERN
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCP workflow pattern content."""
        required_fields = ["pattern_id", "trigger_context", "tool_sequence", "success_metrics", "context_conditions"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"MCP workflow pattern must have '{field}' field")
        
        # Validate tool sequence
        tool_sequence = v["tool_sequence"]
        if not isinstance(tool_sequence, list) or not tool_sequence:
            raise ValueError("Tool sequence must be non-empty list")
        
        # Validate success metrics
        success_metrics = v["success_metrics"]
        required_metrics = ["completion_rate", "average_duration", "user_satisfaction"]
        for metric in required_metrics:
            if metric not in success_metrics:
                raise ValueError(f"Success metrics must include '{metric}'")
            
        return v


class EnhancedContextMemory(MemoryBase):
    """Enhanced contextual memory with MCP awareness."""
    type: str = "enhanced_context"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.CROSS_SYSTEM_INTELLIGENCE
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate enhanced context content."""
        required_fields = ["context_type", "primary_content", "mcp_correlations", "temporal_context"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Enhanced context must have '{field}' field")
        
        # Validate context type
        valid_types = ["conversation", "code", "document", "reflection", "problem_solving", "decision_making"]
        if v["context_type"] not in valid_types:
            raise ValueError(f"Context type must be one of: {valid_types}")
        
        # Validate MCP correlations
        correlations = v["mcp_correlations"]
        if not isinstance(correlations, dict):
            raise ValueError("MCP correlations must be a dictionary")
            
        return v


# Legacy types removed - use MCP-enhanced equivalents








# MCP-Enhanced Structured Thinking Memory Types
class StructuredThinkingMemory(MemoryBase):
    """Enhanced structured thinking processes with MCP integration."""
    type: str = "structured_thinking"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.THINKING_INTEGRATION
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCP-enhanced structured thinking content."""
        required_fields = ["session_id", "session_title", "total_thoughts", "stages_completed", "mcp_workflows"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Structured thinking must have '{field}' field")
        
        # Validate stages_completed is a list
        if not isinstance(v["stages_completed"], list):
            raise ValueError("Structured thinking 'stages_completed' must be a list")
        
        # Validate MCP workflows integration
        mcp_workflows = v["mcp_workflows"]
        if not isinstance(mcp_workflows, dict):
            raise ValueError("MCP workflows must be a dictionary")
            
        return v


class ThoughtProcessMemory(MemoryBase):
    """Enhanced individual thoughts with MCP context and stage metadata."""
    type: str = "thought_process"
    content: Dict[str, Any]
    mcp_integration: MCPIntegrationType = MCPIntegrationType.THINKING_INTEGRATION
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCP-enhanced thought process content."""
        required_fields = ["thought_number", "thinking_stage", "thought_content", "mcp_tool_context"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Thought process must have '{field}' field")
        
        # Validate thought_number is positive integer
        if not isinstance(v["thought_number"], int) or v["thought_number"] <= 0:
            raise ValueError("Thought process 'thought_number' must be a positive integer")
            
        # Validate thinking_stage is valid (extended for MCP)
        valid_stages = ["problem_definition", "research", "analysis", "synthesis", "conclusion", "mcp_integration", "tool_selection"]
        if v["thinking_stage"] not in valid_stages:
            raise ValueError(f"Thought process 'thinking_stage' must be one of: {valid_stages}")
        
        # Validate MCP tool context
        mcp_context = v["mcp_tool_context"]
        if not isinstance(mcp_context, dict):
            raise ValueError("MCP tool context must be a dictionary")
            
        return v


class ThinkingRelationshipMemory(MemoryBase):
    """Model for relationships between thoughts."""
    type: str = "thinking_relationship"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod  
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate thinking relationship content."""
        required_fields = ["source_thought_id", "target_thought_id", "relationship_type", "strength"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Thinking relationship must have '{field}' field")
        
        # Validate relationship_type
        valid_types = ["builds_on", "challenges", "supports", "contradicts", "extends"]
        if v["relationship_type"] not in valid_types:
            raise ValueError(f"Thinking relationship 'relationship_type' must be one of: {valid_types}")
            
        # Validate strength is between 0 and 1
        if not isinstance(v["strength"], (int, float)) or not 0.0 <= v["strength"] <= 1.0:
            raise ValueError("Thinking relationship 'strength' must be between 0.0 and 1.0")
            
        return v


class ProblemAnalysisMemory(MemoryBase):
    """Model for problem definition stage thoughts.""" 
    type: str = "problem_analysis"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate problem analysis content."""
        required_fields = ["problem_statement", "constraints", "success_criteria"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Problem analysis must have '{field}' field")
        return v


class ResearchNotesMemory(MemoryBase):
    """Model for research stage findings."""
    type: str = "research_notes"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate research notes content."""
        required_fields = ["research_topic", "findings", "sources"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Research notes must have '{field}' field")
        
        # Validate sources is a list
        if not isinstance(v["sources"], list):
            raise ValueError("Research notes 'sources' must be a list")
            
        return v


class AnalysisResultMemory(MemoryBase):
    """Model for analysis stage conclusions."""
    type: str = "analysis_result"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis result content."""
        required_fields = ["analysis_subject", "components", "risk_assessment"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Analysis result must have '{field}' field")
        
        # Validate components is a list
        if not isinstance(v["components"], list):
            raise ValueError("Analysis result 'components' must be a list")
            
        return v


class SolutionSynthesisMemory(MemoryBase):
    """Model for synthesis stage solutions."""
    type: str = "solution_synthesis" 
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate solution synthesis content."""
        required_fields = ["solution_approach", "alternatives_considered", "trade_offs"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Solution synthesis must have '{field}' field")
        
        # Validate alternatives_considered is a list
        if not isinstance(v["alternatives_considered"], list):
            raise ValueError("Solution synthesis 'alternatives_considered' must be a list")
            
        return v


class ConclusionSummaryMemory(MemoryBase):
    """Model for final conclusions and decisions."""
    type: str = "conclusion_summary"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate conclusion summary content."""
        required_fields = ["final_decision", "action_items", "success_metrics"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Conclusion summary must have '{field}' field")
        
        # Validate action_items is a list
        if not isinstance(v["action_items"], list):
            raise ValueError("Conclusion summary 'action_items' must be a list")
            
        return v


# MCP-Enhanced Memory Type Configuration
MCP_ENHANCED_MEMORY_TYPES = {
    # Core MCP Integration Types
    "mcp_thinking_workflow": {
        "description": "MCP-enhanced structured thinking workflows",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True,
        "mcp_integration": "thinking_integration",
        "cache_enabled": True
    },
    "mcp_resource_pattern": {
        "description": "MCP resource patterns and references",
        "tier": "core",
        "retention_days": 180,
        "requires_embedding": True,
        "mcp_integration": "resource_reference",
        "cache_enabled": True
    },
    "thinking_mcp_integration": {
        "description": "Cross-system thinking integration",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True,
        "mcp_integration": "cross_system_intelligence",
        "cache_enabled": True
    },
    "mcp_tool_correlation": {
        "description": "Tool and thinking pattern correlations",
        "tier": "supplementary",
        "retention_days": 180,
        "requires_embedding": True,
        "mcp_integration": "tool_correlation",
        "cache_enabled": True
    },
    "mcp_workflow_pattern": {
        "description": "MCP workflow patterns and effectiveness",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True,
        "mcp_integration": "workflow_pattern",
        "cache_enabled": True
    },
    "enhanced_context": {
        "description": "Enhanced contextual memory with MCP awareness",
        "tier": "core",
        "retention_days": 180,
        "requires_embedding": True,
        "mcp_integration": "cross_system_intelligence",
        "cache_enabled": True
    },
    # Enhanced Structured Thinking Types  
    "structured_thinking": {
        "description": "MCP-enhanced structured thinking process",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True,
        "mcp_integration": "thinking_integration",
        "cache_enabled": True
    },
    "thought_process": {
        "description": "MCP-enhanced individual thought with context",
        "tier": "core",
        "retention_days": 180,
        "requires_embedding": True,
        "mcp_integration": "thinking_integration",
        "cache_enabled": False
    },
    "thinking_relationship": {
        "description": "Enhanced relationships between thoughts",
        "tier": "supplementary",
        "retention_days": 90,
        "requires_embedding": False,
        "mcp_integration": "thinking_integration",
        "cache_enabled": False
    }
}


def validate_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a memory object against its schema.
    
    Args:
        memory: Memory dictionary
        
    Returns:
        Validated memory dictionary
        
    Raises:
        ValueError: If memory is invalid
    """
    if "type" not in memory:
        raise ValueError("Memory must have a 'type' field")
        
    memory_type = memory["type"]
    
    # MCP-Enhanced Memory Validators
    validators = {
        # Core MCP Integration Types
        "mcp_thinking_workflow": MCPThinkingWorkflowMemory,
        "mcp_resource_pattern": MCPResourcePatternMemory,
        "thinking_mcp_integration": ThinkingMCPIntegrationMemory,
        "mcp_tool_correlation": MCPToolCorrelationMemory,
        "mcp_workflow_pattern": MCPWorkflowPatternMemory,
        "enhanced_context": EnhancedContextMemory,
        # Enhanced structured thinking types
        "structured_thinking": StructuredThinkingMemory,
        "thought_process": ThoughtProcessMemory,
        "thinking_relationship": ThinkingRelationshipMemory,
        # Remaining compatible thinking types (modernized)
        "problem_analysis": ProblemAnalysisMemory,
        "research_notes": ResearchNotesMemory,
        "analysis_result": AnalysisResultMemory,
        "solution_synthesis": SolutionSynthesisMemory,
        "conclusion_summary": ConclusionSummaryMemory
    }
    
    if memory_type not in validators:
        raise ValueError(f"Unknown memory type: {memory_type}")
        
    # Validate using Pydantic model
    model = validators[memory_type](**memory)
    
    # Return validated model as dict
    return model.dict()


def validate_iso_timestamp(timestamp: str) -> bool:
    """
    Validate ISO timestamp format.
    
    Args:
        timestamp: Timestamp string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.fromisoformat(timestamp)
        return True
    except ValueError:
        return False
    
    
def validate_memory_id(memory_id: str) -> bool:
    """
    Validate memory ID format.
    
    Args:
        memory_id: Memory ID string
        
    Returns:
        True if valid, False otherwise
    """
    # Memory IDs should start with "mem_" followed by alphanumeric chars
    pattern = r"^mem_[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, memory_id))


def get_mcp_memory_types() -> Dict[str, Dict[str, Any]]:
    """
    Get MCP-enhanced memory type configuration.
    
    Returns:
        Dictionary of memory type configurations with MCP integration
    """
    return MCP_ENHANCED_MEMORY_TYPES


def is_mcp_integrated_type(memory_type: str) -> bool:
    """
    Check if a memory type has MCP integration enabled.
    
    Args:
        memory_type: Memory type string
        
    Returns:
        True if memory type has MCP integration
    """
    config = MCP_ENHANCED_MEMORY_TYPES.get(memory_type, {})
    mcp_integration = config.get("mcp_integration", "none")
    
    # Check if it's an MCP-enhanced type or has explicit MCP integration
    return (mcp_integration != "none" and mcp_integration is not None) or memory_type.startswith("mcp_")


def get_memory_cache_config(memory_type: str) -> bool:
    """
    Check if a memory type should use caching.
    
    Args:
        memory_type: Memory type string
        
    Returns:
        True if caching is enabled for this memory type
    """
    config = MCP_ENHANCED_MEMORY_TYPES.get(memory_type, {})
    return config.get("cache_enabled", False)


# Backwards compatibility - deprecated, will be removed
STRUCTURED_THINKING_MEMORY_TYPES = MCP_ENHANCED_MEMORY_TYPES  # Deprecated alias
