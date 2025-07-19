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
                    "description": "Type of memory to store",
                    "enum": ["conversation", "fact", "document", "entity", "reflection", "code", "project_pattern", "command_pattern", "session_summary", "bash_execution"]
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
    
    # AutoCode tool schemas
    @property
    def suggest_command_schema(self) -> Dict[str, Any]:
        """Schema for the suggest_command tool."""
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What you want to accomplish (e.g., 'delete file', 'install dependencies')"
                },
                "context": {
                    "type": "object",
                    "description": "Current context (project type, platform, etc.)",
                    "properties": {
                        "project_type": {"type": "string"},
                        "project_path": {"type": "string"},
                        "platform": {"type": "string"}
                    }
                }
            },
            "required": ["intent"]
        }
    
    @property
    def track_bash_schema(self) -> Dict[str, Any]:
        """Schema for the track_bash tool."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command that was executed"
                },
                "exit_code": {
                    "type": "integer",
                    "description": "Exit code from command execution"
                },
                "output": {
                    "type": "string",
                    "description": "Command output or error message"
                },
                "context": {
                    "type": "object",
                    "description": "Execution context",
                    "properties": {
                        "project_type": {"type": "string"},
                        "project_path": {"type": "string"},
                        "current_directory": {"type": "string"}
                    }
                }
            },
            "required": ["command", "exit_code"]
        }
    
    @property
    def get_session_history_schema(self) -> Dict[str, Any]:
        """Schema for the get_session_history tool."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for session history"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to return",
                    "minimum": 1,
                    "maximum": 20
                },
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to search",
                    "minimum": 1,
                    "maximum": 90
                }
            },
            "required": ["query"]
        }
    
    @property
    def get_project_patterns_schema(self) -> Dict[str, Any]:
        """Schema for the get_project_patterns tool."""
        return {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Path to the project to analyze"
                },
                "pattern_types": {
                    "type": "array",
                    "description": "Types of patterns to retrieve",
                    "items": {
                        "type": "string",
                        "enum": ["architectural", "naming", "component", "testing", "build"]
                    }
                }
            },
            "required": ["project_path"]
        }
    
    @property
    def track_file_access_schema(self) -> Dict[str, Any]:
        """Schema for the track_file_access tool."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file that was accessed"
                },
                "operation": {
                    "type": "string",
                    "description": "Type of operation performed",
                    "enum": ["read", "write", "edit", "delete"]
                },
                "content": {
                    "type": "string",
                    "description": "File content (for analysis)"
                }
            },
            "required": ["file_path", "operation"]
        }
