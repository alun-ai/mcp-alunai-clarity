"""
Schema validation utilities for the memory MCP server.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class MemoryBase(BaseModel):
    """Base model for memory objects."""
    id: str
    type: str
    importance: float = 0.5
    
    @validator("importance")
    def validate_importance(cls, v: float) -> float:
        """Validate importance score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")
        return v


class ConversationMemory(MemoryBase):
    """Model for conversation memories."""
    type: str = "conversation"
    content: Dict[str, Any]
    
    @validator("content")
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
    
    @validator("content")
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
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document content."""
        if "title" not in v or "text" not in v:
            raise ValueError("Document must have 'title' and 'text' fields")
            
        return v


class EntityMemory(MemoryBase):
    """Model for entity memories."""
    type: str = "entity"
    content: Dict[str, Any]
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entity content."""
        if "name" not in v or "entity_type" not in v:
            raise ValueError("Entity must have 'name' and 'entity_type' fields")
            
        return v


class ReflectionMemory(MemoryBase):
    """Model for reflection memories."""
    type: str = "reflection"
    content: Dict[str, Any]
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reflection content."""
        if "subject" not in v or "reflection" not in v:
            raise ValueError("Reflection must have 'subject' and 'reflection' fields")
            
        return v


class CodeMemory(MemoryBase):
    """Model for code memories."""
    type: str = "code"
    content: Dict[str, Any]
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code content."""
        if "language" not in v or "code" not in v:
            raise ValueError("Code must have 'language' and 'code' fields")
            
        return v


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
        "code": CodeMemory
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
