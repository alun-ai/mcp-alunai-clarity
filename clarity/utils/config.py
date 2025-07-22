"""
Configuration utilities for the memory MCP server.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from clarity.shared.infrastructure import get_cache


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file with caching.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = os.path.expanduser(config_path)
    
    # Initialize config cache
    config_cache = get_cache(
        "config",
        max_size=100,       # Cache up to 100 config files
        max_memory_mb=10,   # 10MB for config data
        default_ttl=600.0   # 10 minutes TTL
    )
    
    # Create cache key from file path and modification time
    cache_key = f"cfg_{config_path}"
    if os.path.exists(config_path):
        mtime = os.path.getmtime(config_path)
        cache_key += f"_{int(mtime)}"
    
    # Try cache first
    cached_config = config_cache.get(cache_key)
    if cached_config is not None:
        return cached_config
    
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        config = create_default_config(config_path)
        config_cache.set(cache_key, config)
        return config
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            
            # Validate and merge with defaults
            config = validate_config(config)
            
            # Ensure all instances use shared storage (remove session isolation logic)
            
            # Cache the configuration
            config_cache.set(cache_key, config)
            
            return config
    except json.JSONDecodeError:
        logger.error(f"Error parsing configuration file: {config_path}")
        config = create_default_config(config_path)
        config_cache.set(cache_key, config)
        return config
    except (FileNotFoundError, json.JSONDecodeError, ConfigurationError, PermissionError, OSError) as e:
        logger.error(f"Error loading configuration: {str(e)}")
        config = create_default_config(config_path)
        config_cache.set(cache_key, config)
        return config


def create_default_config(config_path: str) -> Dict[str, Any]:
    """
    Create default configuration.
    
    Args:
        config_path: Path to save the configuration file
        
    Returns:
        Default configuration dictionary
    """
    logger.info(f"Creating default configuration at {config_path}")
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Use shared storage by default for all Claude instances in same project
    base_data_path = os.path.dirname(config_path)  # Use same directory as config
    qdrant_path = os.path.join(base_data_path, "qdrant")
    
    # Default configuration
    config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "quick_start": False
        },
        "qdrant": {
            "path": qdrant_path,
            "host": "localhost",
            "port": 6333,
            "prefer_grpc": False,
            "index_params": {
                "m": 16,
                "ef_construct": 200,
                "full_scan_threshold": 10000
            },
            "optimization": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 0
            }
        },
        "alunai-clarity": {
            "max_short_term_items": 100,
            "max_long_term_items": 1000,
            "max_archival_items": 10000,
            "consolidation_interval_hours": 24,
            "short_term_threshold": 0.3,
            "legacy_file_path": os.path.join(
                os.path.expanduser("~/.clarity/data"),
                "memory.json"
            )
        },
        "embedding": {
            "default_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "cache_dir": os.path.expanduser("~/.clarity/cache")
        },
        "retrieval": {
            "default_top_k": 5,
            "semantic_threshold": 0.75,
            "recency_weight": 0.3,
            "importance_weight": 0.7
        },
        "autocode": {
            "enabled": True,
            "auto_scan_projects": True,
            "track_bash_commands": True,
            "generate_session_summaries": True,
            "command_learning": {
                "enabled": True,
                "min_confidence_threshold": 0.3,
                "max_suggestions": 5,
                "track_failures": True
            },
            "pattern_detection": {
                "enabled": True,
                "supported_languages": ["typescript", "javascript", "python", "rust", "go", "java"],
                "max_scan_depth": 5,
                "ignore_patterns": ["node_modules", ".git", "__pycache__", "target", "build", "dist"]
            },
            "session_analysis": {
                "enabled": True,
                "min_session_length": 3,
                "track_architectural_decisions": True,
                "extract_learning_patterns": True,
                "identify_workflow_improvements": True,
                "confidence_threshold": 0.6
            },
            "history_navigation": {
                "enabled": True,
                "similarity_threshold": 0.6,
                "max_results": 10,
                "context_window_days": 30,
                "prioritize_recent": True,
                "include_incomplete_sessions": True
            },
            "history_retention": {
                "session_summaries_days": 90,
                "command_patterns_days": 30,
                "project_patterns_days": 365
            },
            "mcp_awareness": {
                "enabled": True,
                "index_tools_on_startup": False,
                "proactive_suggestions": True,
                "suggest_alternatives": True,
                "context_aware_suggestions": True,
                "error_resolution_suggestions": True,
                "max_recent_suggestions": 10
            }
        }
    }
    
    # Save default config
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except (OSError, PermissionError, json.JSONEncodeError) as e:
        logger.error(f"Error saving default configuration: {str(e)}")
    
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    # Create default config template
    default_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "quick_start": False
        },
        "qdrant": {
            "path": os.path.expanduser("~/.claude/alunai-clarity/qdrant"),
            "host": "localhost",
            "port": 6333,
            "prefer_grpc": False,
            "index_params": {
                "m": 16,
                "ef_construct": 200,
                "full_scan_threshold": 10000
            },
            "optimization": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 0
            }
        },
        "alunai-clarity": {
            "max_short_term_items": 100,
            "max_long_term_items": 1000,
            "max_archival_items": 10000,
            "consolidation_interval_hours": 24,
            "short_term_threshold": 0.3,
            "legacy_file_path": os.path.join(
                os.path.expanduser("~/.clarity/data"),
                "memory.json"
            )
        },
        "embedding": {
            "default_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "cache_dir": os.path.expanduser("~/.clarity/cache")
        },
        "retrieval": {
            "default_top_k": 5,
            "semantic_threshold": 0.75,
            "recency_weight": 0.3,
            "importance_weight": 0.7
        },
        "autocode": {
            "enabled": True,
            "auto_scan_projects": True,
            "track_bash_commands": True,
            "generate_session_summaries": True,
            "command_learning": {
                "enabled": True,
                "min_confidence_threshold": 0.3,
                "max_suggestions": 5,
                "track_failures": True
            },
            "pattern_detection": {
                "enabled": True,
                "supported_languages": ["typescript", "javascript", "python", "rust", "go", "java"],
                "max_scan_depth": 5,
                "ignore_patterns": ["node_modules", ".git", "__pycache__", "target", "build", "dist"]
            },
            "session_analysis": {
                "enabled": True,
                "min_session_length": 3,
                "track_architectural_decisions": True,
                "extract_learning_patterns": True,
                "identify_workflow_improvements": True,
                "confidence_threshold": 0.6
            },
            "history_navigation": {
                "enabled": True,
                "similarity_threshold": 0.6,
                "max_results": 10,
                "context_window_days": 30,
                "prioritize_recent": True,
                "include_incomplete_sessions": True
            },
            "history_retention": {
                "session_summaries_days": 90,
                "command_patterns_days": 30,
                "project_patterns_days": 365
            },
            "mcp_awareness": {
                "enabled": True,
                "index_tools_on_startup": False,
                "proactive_suggestions": True,
                "suggest_alternatives": True,
                "context_aware_suggestions": True,
                "error_resolution_suggestions": True,
                "max_recent_suggestions": 10
            }
        }
    }
    
    # Merge with user config (deep merge)
    merged_config = deep_merge(default_config, config)
    
    # Convert relative paths to absolute
    if "alunai-clarity" in merged_config and "file_path" in merged_config["alunai-clarity"]:
        file_path = merged_config["alunai-clarity"]["file_path"]
        if not os.path.isabs(file_path):
            merged_config["alunai-clarity"]["file_path"] = os.path.abspath(file_path)
    
    return merged_config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result
