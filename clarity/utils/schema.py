"""
Schema validation utilities for the memory MCP server.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MemoryBase(BaseModel):
    """Base model for memory objects."""
    id: str
    type: str
    importance: float = 0.5
    
    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v: float) -> float:
        """Validate importance score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")
        return v


class ConversationMemory(MemoryBase):
    """Model for conversation memories."""
    type: str = "conversation"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate conversation content."""
        if "role" not in v and "messages" not in v:
            raise ValueError("Conversation must have either 'role' or 'messages'")
        
        if "role" in v and "message" not in v:
            raise ValueError("Conversation with 'role' must have 'message'")
            
        if "messages" in v and not isinstance(v["messages"], list):
            raise ValueError("Conversation 'messages' must be a list")
            
        return v


class FactMemory(MemoryBase):
    """Model for fact memories."""
    type: str = "fact"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate fact content."""
        if "fact" not in v:
            raise ValueError("Fact must have 'fact' field")
            
        if "confidence" in v and not 0.0 <= v["confidence"] <= 1.0:
            raise ValueError("Fact confidence must be between 0.0 and 1.0")
            
        return v


class DocumentMemory(MemoryBase):
    """Model for document memories."""
    type: str = "document"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document content."""
        if "title" not in v or "text" not in v:
            raise ValueError("Document must have 'title' and 'text' fields")
            
        return v


class EntityMemory(MemoryBase):
    """Model for entity memories."""
    type: str = "entity"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entity content."""
        if "name" not in v or "entity_type" not in v:
            raise ValueError("Entity must have 'name' and 'entity_type' fields")
            
        return v


class ReflectionMemory(MemoryBase):
    """Model for reflection memories."""
    type: str = "reflection"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reflection content."""
        if "subject" not in v or "reflection" not in v:
            raise ValueError("Reflection must have 'subject' and 'reflection' fields")
            
        return v


class CodeMemory(MemoryBase):
    """Model for code memories."""
    type: str = "code"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code content."""
        if "language" not in v or "code" not in v:
            raise ValueError("Code must have 'language' and 'code' fields")
            
        return v


class ProjectPatternMemory(MemoryBase):
    """Model for project architectural patterns."""
    type: str = "project_pattern"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate project pattern content."""
        required_fields = ["pattern_type", "framework", "language", "structure"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Project pattern must have '{field}' field")
        return v


class CommandPatternMemory(MemoryBase):
    """Model for bash command patterns and success rates."""
    type: str = "command_pattern"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate command pattern content."""
        required_fields = ["command", "context", "success_rate", "platform"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Command pattern must have '{field}' field")
        
        # Validate success_rate is between 0 and 1
        if not 0.0 <= v["success_rate"] <= 1.0:
            raise ValueError("Command success_rate must be between 0.0 and 1.0")
            
        return v


class SessionSummaryMemory(MemoryBase):
    """Model for session summaries and completed work."""
    type: str = "session_summary"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate session summary content."""
        required_fields = ["session_id", "tasks_completed", "patterns_used", "files_modified"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Session summary must have '{field}' field")
        
        # Validate that lists are actually lists
        list_fields = ["tasks_completed", "patterns_used", "files_modified"]
        for field in list_fields:
            if not isinstance(v[field], list):
                raise ValueError(f"Session summary '{field}' must be a list")
                
        return v


class BashExecutionMemory(MemoryBase):
    """Model for individual bash command executions."""
    type: str = "bash_execution"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bash execution content."""
        required_fields = ["command", "exit_code", "timestamp", "context"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Bash execution must have '{field}' field")
        
        # Validate exit_code is an integer
        if not isinstance(v["exit_code"], int):
            raise ValueError("Bash execution exit_code must be an integer")
            
        # Validate timestamp format
        if not validate_iso_timestamp(v["timestamp"]):
            raise ValueError("Bash execution timestamp must be valid ISO format")
            
        return v


# Structured Thinking Memory Types from Sequential Thinking Integration
class StructuredThinkingMemory(MemoryBase):
    """Model for complete structured thinking processes."""
    type: str = "structured_thinking"
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate structured thinking content."""
        required_fields = ["session_id", "session_title", "total_thoughts", "stages_completed"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Structured thinking must have '{field}' field")
        
        # Validate stages_completed is a list
        if not isinstance(v["stages_completed"], list):
            raise ValueError("Structured thinking 'stages_completed' must be a list")
            
        return v


class ThoughtProcessMemory(MemoryBase):
    """Model for individual thoughts with stage metadata."""
    type: str = "thought_process" 
    content: Dict[str, Any]
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate thought process content."""
        required_fields = ["thought_number", "thinking_stage", "thought_content"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Thought process must have '{field}' field")
        
        # Validate thought_number is positive integer
        if not isinstance(v["thought_number"], int) or v["thought_number"] <= 0:
            raise ValueError("Thought process 'thought_number' must be a positive integer")
            
        # Validate thinking_stage is valid
        valid_stages = ["problem_definition", "research", "analysis", "synthesis", "conclusion"]
        if v["thinking_stage"] not in valid_stages:
            raise ValueError(f"Thought process 'thinking_stage' must be one of: {valid_stages}")
            
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


# Memory type configuration for structured thinking
STRUCTURED_THINKING_MEMORY_TYPES = {
    "structured_thinking": {
        "description": "Complete structured thinking process",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True
    },
    "thought_process": {
        "description": "Individual thought with stage metadata",
        "tier": "core", 
        "retention_days": 180,
        "requires_embedding": True
    },
    "thinking_relationship": {
        "description": "Relationships between thoughts",
        "tier": "supplementary",
        "retention_days": 90,
        "requires_embedding": False
    },
    "problem_analysis": {
        "description": "Problem definition stage thoughts",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True
    },
    "research_notes": {
        "description": "Research stage findings",
        "tier": "core",
        "retention_days": 180,
        "requires_embedding": True
    },
    "analysis_result": {
        "description": "Analysis stage conclusions",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True
    },
    "solution_synthesis": {
        "description": "Synthesis stage solutions",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True
    },
    "conclusion_summary": {
        "description": "Final conclusions and decisions",
        "tier": "core",
        "retention_days": 365,
        "requires_embedding": True
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
    
    # Choose validator based on type
    validators = {
        "conversation": ConversationMemory,
        "fact": FactMemory,
        "document": DocumentMemory,
        "entity": EntityMemory,
        "reflection": ReflectionMemory,
        "code": CodeMemory,
        "project_pattern": ProjectPatternMemory,
        "command_pattern": CommandPatternMemory,
        "session_summary": SessionSummaryMemory,
        "bash_execution": BashExecutionMemory,
        # Structured thinking memory types
        "structured_thinking": StructuredThinkingMemory,
        "thought_process": ThoughtProcessMemory,
        "thinking_relationship": ThinkingRelationshipMemory,
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
