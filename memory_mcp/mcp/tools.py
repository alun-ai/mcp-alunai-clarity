"""
MCP tool definitions for the memory system.
"""

from typing import Dict, Any

from memory_mcp.domains.manager import MemoryDomainManager


class MemoryToolDefinitions:
    """
    Defines MCP tools for the memory system.
    
    This class contains the schema definitions and validation for
    the MCP tools exposed by the memory server.
    """
    
    def __init__(self, domain_manager: MemoryDomainManager) -> None:
        """
        Initialize the tool definitions.
        
        Args:
            domain_manager: The memory domain manager
        """
        self.domain_manager = domain_manager
    
    @property
    def store_memory_schema(self) -> Dict[str, Any]:
        """Schema for the store_memory tool."""
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Type of memory to store (conversation, fact, document, entity, reflection)",
                    "enum": ["conversation", "fact", "document", "entity", "reflection", "code"]
                },
                "content": {
                    "type": "object",
                    "description": "Content of the memory (type-specific structure)"
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score (0.0-1.0, higher is more important)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata for the memory"
                },
                "context": {
                    "type": "object",
                    "description": "Contextual information for the memory"
                }
            },
            "required": ["type", "content"]
        }
    
    @property
    def retrieve_memory_schema(self) -> Dict[str, Any]:
        """Schema for the retrieve_memory tool."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query string to search for relevant memories"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to retrieve (default: 5)",
                    "minimum": 1,
                    "maximum": 50
                },
                "types": {
                    "type": "array",
                    "description": "Types of memories to include (null for all types)",
                    "items": {
                        "type": "string",
                        "enum": ["conversation", "fact", "document", "entity", "reflection", "code"]
                    }
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Minimum similarity score (0.0-1.0) for results",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include metadata in the results"
                }
            },
            "required": ["query"]
        }
    
    @property
    def list_memories_schema(self) -> Dict[str, Any]:
        """Schema for the list_memories tool."""
        return {
            "type": "object",
            "properties": {
                "types": {
                    "type": "array",
                    "description": "Types of memories to include (null for all types)",
                    "items": {
                        "type": "string",
                        "enum": ["conversation", "fact", "document", "entity", "reflection", "code"]
                    }
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to retrieve (default: 20)",
                    "minimum": 1,
                    "maximum": 100
                },
                "offset": {
                    "type": "integer",
                    "description": "Offset for pagination (default: 0)",
                    "minimum": 0
                },
                "tier": {
                    "type": "string",
                    "description": "Memory tier to retrieve from (null for all tiers)",
                    "enum": ["short_term", "long_term", "archived"]
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Whether to include memory content in the results (default: false)"
                }
            }
        }
    
    @property
    def update_memory_schema(self) -> Dict[str, Any]:
        """Schema for the update_memory tool."""
        return {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to update"
                },
                "updates": {
                    "type": "object",
                    "description": "Updates to apply to the memory",
                    "properties": {
                        "content": {
                            "type": "object",
                            "description": "New content for the memory"
                        },
                        "importance": {
                            "type": "number",
                            "description": "New importance score (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Updates to memory metadata"
                        },
                        "context": {
                            "type": "object",
                            "description": "Updates to memory context"
                        }
                    }
                }
            },
            "required": ["memory_id", "updates"]
        }
    
    @property
    def delete_memory_schema(self) -> Dict[str, Any]:
        """Schema for the delete_memory tool."""
        return {
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "description": "IDs of memories to delete",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["memory_ids"]
        }
    
    @property
    def memory_stats_schema(self) -> Dict[str, Any]:
        """Schema for the memory_stats tool."""
        return {
            "type": "object",
            "properties": {}
        }
