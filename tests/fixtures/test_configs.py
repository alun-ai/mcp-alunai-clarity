"""
Test configurations and sample data for MCP discovery testing.

This module provides various test configurations, sample data, and test patterns
for comprehensive testing of the MCP discovery enhancement system.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone


# Test MCP server configurations
TEST_MCP_CONFIGS = {
    "postgres": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-postgres"],
        "env": {"DATABASE_URL": "postgresql://localhost/test"},
        "source": "test_config"
    },
    "filesystem": {
        "command": "npx", 
        "args": ["@modelcontextprotocol/server-filesystem"],
        "env": {"ALLOWED_DIRS": "/tmp:/test"},
        "source": "test_config"
    },
    "web": {
        "command": "python",
        "args": ["-m", "web_mcp_server"],
        "env": {"API_KEY": "test_api_key", "BASE_URL": "https://api.test.com"},
        "source": "test_config"
    },
    "git": {
        "command": "git-mcp-server",
        "args": ["--repo", "/test/repo"],
        "env": {"GIT_TOKEN": "test_token"},
        "source": "test_config"
    },
    "sqlite": {
        "command": "sqlite-mcp-server",
        "args": ["--db", "/test/data.db"],
        "env": {},
        "source": "test_config"
    },
    "docs": {
        "command": "python",
        "args": ["-m", "docs_mcp_server"],
        "env": {"DOCS_PATH": "/test/docs"},
        "source": "test_config"
    }
}


# Test Claude Desktop configurations
TEST_CLAUDE_DESKTOP_CONFIGS = {
    "basic_config": {
        "mcpServers": {
            "postgres": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-postgres"],
                "env": {"DATABASE_URL": "postgresql://localhost/test"}
            },
            "filesystem": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem"],
                "env": {"ALLOWED_DIRS": "/tmp"}
            }
        }
    },
    "complex_config": {
        "mcpServers": {
            "postgres": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-postgres"],
                "env": {
                    "DATABASE_URL": "postgresql://user:pass@localhost:5432/mydb",
                    "POOL_SIZE": "10"
                }
            },
            "filesystem": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem"],
                "env": {"ALLOWED_DIRS": "/home/user:/tmp:/var/log"}
            },
            "git": {
                "command": "python",
                "args": ["-m", "git_mcp_server"],
                "env": {
                    "GIT_TOKEN": "ghp_test_token",
                    "DEFAULT_REPO": "/home/user/projects"
                }
            },
            "api": {
                "command": "node",
                "args": ["api-mcp-server.js"],
                "env": {
                    "API_KEY": "test_api_key",
                    "BASE_URL": "https://api.example.com",
                    "TIMEOUT": "30000"
                }
            }
        },
        "logging": {
            "level": "debug",
            "file": "/tmp/mcp.log"
        }
    },
    "minimal_config": {
        "mcpServers": {
            "simple": {
                "command": "simple-server",
                "args": []
            }
        }
    }
}


# Test workflow patterns
TEST_WORKFLOW_PATTERNS = [
    {
        "context": "Database query request for user authentication",
        "tools": ["postgres_query"],
        "resources": ["@postgres:query://SELECT * FROM users WHERE email = ?"],
        "success": {
            "rows_returned": 1,
            "execution_time": 0.05,
            "query_plan": "index_scan"
        },
        "score": 0.95,
        "project_type": "web_application",
        "intent": "user_authentication",
        "usage_count": 50,
        "last_used": "2024-01-15T10:30:00Z"
    },
    {
        "context": "Read configuration file for application settings",
        "tools": ["filesystem_read"],
        "resources": ["@filesystem:file://config/app.json"],
        "success": {
            "file_size": 2048,
            "encoding": "utf-8",
            "parse_success": True
        },
        "score": 0.88,
        "project_type": "web_application",
        "intent": "configuration_access",
        "usage_count": 25,
        "last_used": "2024-01-15T09:15:00Z"
    },
    {
        "context": "API request to external service for data retrieval",
        "tools": ["web_request"],
        "resources": ["@web:request://https://api.service.com/data"],
        "success": {
            "status_code": 200,
            "response_time": 150,
            "data_received": True
        },
        "score": 0.82,
        "project_type": "data_integration",
        "intent": "data_retrieval",
        "usage_count": 15,
        "last_used": "2024-01-15T08:45:00Z"
    },
    {
        "context": "Git repository status check and commit history",
        "tools": ["git_status", "git_log"],
        "resources": ["@git:repo://current", "@git:history://main"],
        "success": {
            "files_changed": 3,
            "commits_found": 10,
            "clean_status": True
        },
        "score": 0.79,
        "project_type": "development",
        "intent": "version_control",
        "usage_count": 30,
        "last_used": "2024-01-15T11:00:00Z"
    },
    {
        "context": "Failed database connection attempt",
        "tools": ["postgres_query"],
        "success": False,
        "failure_reason": "Connection timeout",
        "score": 0.1,
        "project_type": "web_application",
        "intent": "database_access",
        "usage_count": 5,
        "last_used": "2024-01-15T07:30:00Z"
    }
]


# Test hook integration scenarios
TEST_HOOK_SCENARIOS = [
    {
        "scenario": "Database access via shell command",
        "pre_tool_data": {
            "tool_name": "bash",
            "args": 'psql -d myapp -c "SELECT count(*) FROM users"'
        },
        "expected_opportunity": {
            "type": "database_query",
            "confidence": 0.9,
            "suggested_tools": ["postgres_query"]
        },
        "post_tool_data": {
            "result": "count: 1500",
            "exit_code": 0,
            "execution_time": 0.2
        },
        "expected_learning": {
            "pattern_learned": True,
            "success": True,
            "improvement_score": 0.1
        }
    },
    {
        "scenario": "File operations via command line",
        "pre_tool_data": {
            "tool_name": "bash",
            "args": "cat /app/config.json | jq '.database'"
        },
        "expected_opportunity": {
            "type": "file_operations",
            "confidence": 0.85,
            "suggested_tools": ["filesystem_read"]
        },
        "post_tool_data": {
            "result": '{"host": "localhost", "port": 5432}',
            "exit_code": 0,
            "execution_time": 0.1
        },
        "expected_learning": {
            "pattern_learned": True,
            "success": True,
            "improvement_score": 0.15
        }
    },
    {
        "scenario": "Web request via curl",
        "pre_tool_data": {
            "tool_name": "bash",
            "args": 'curl -X GET "https://api.service.com/users" -H "Authorization: Bearer token"'
        },
        "expected_opportunity": {
            "type": "web_request",
            "confidence": 0.88,
            "suggested_tools": ["web_request"]
        },
        "post_tool_data": {
            "result": '{"users": [{"id": 1, "name": "test"}]}',
            "exit_code": 0,
            "execution_time": 0.3
        },
        "expected_learning": {
            "pattern_learned": True,
            "success": True,
            "improvement_score": 0.2
        }
    }
]


# Test resource reference patterns
TEST_RESOURCE_PATTERNS = [
    {
        "reference": "@postgres:query://SELECT * FROM users WHERE active = true",
        "context": "Get active users for dashboard",
        "usage_data": {
            "success_rate": 0.95,
            "avg_response_time": 80,
            "usage_count": 100,
            "last_used": "2024-01-15T12:00:00Z"
        }
    },
    {
        "reference": "@filesystem:file://config/database.yaml",
        "context": "Load database configuration",
        "usage_data": {
            "success_rate": 0.98,
            "avg_response_time": 20,
            "usage_count": 50,
            "last_used": "2024-01-15T11:30:00Z"
        }
    },
    {
        "reference": "@web:request://https://api.weather.com/current",
        "context": "Get current weather data",
        "usage_data": {
            "success_rate": 0.88,
            "avg_response_time": 200,
            "usage_count": 25,
            "last_used": "2024-01-15T10:00:00Z"
        }
    },
    {
        "reference": "@git:repo://main/README.md",
        "context": "Read project documentation",
        "usage_data": {
            "success_rate": 0.99,
            "avg_response_time": 50,
            "usage_count": 15,
            "last_used": "2024-01-15T09:30:00Z"
        }
    }
]


# Test slash command configurations
TEST_SLASH_COMMANDS = [
    {
        "command": "/mcp__postgres__query",
        "server_name": "postgres",
        "prompt_name": "query",
        "description": "Execute SQL query on PostgreSQL database",
        "arguments": [
            {"name": "sql", "type": "string", "required": True, "description": "SQL query to execute"},
            {"name": "limit", "type": "integer", "required": False, "description": "Maximum rows to return"}
        ],
        "category": "database",
        "usage_data": {
            "usage_count": 75,
            "success_rate": 0.92,
            "avg_execution_time": 150
        }
    },
    {
        "command": "/mcp__filesystem__read",
        "server_name": "filesystem",
        "prompt_name": "read",
        "description": "Read file contents from filesystem",
        "arguments": [
            {"name": "path", "type": "string", "required": True, "description": "File path to read"},
            {"name": "encoding", "type": "string", "required": False, "description": "File encoding"}
        ],
        "category": "file_operations",
        "usage_data": {
            "usage_count": 45,
            "success_rate": 0.96,
            "avg_execution_time": 80
        }
    },
    {
        "command": "/mcp__web__get",
        "server_name": "web",
        "prompt_name": "get",
        "description": "Make HTTP GET request",
        "arguments": [
            {"name": "url", "type": "string", "required": True, "description": "URL to request"},
            {"name": "headers", "type": "object", "required": False, "description": "HTTP headers"}
        ],
        "category": "web_requests",
        "usage_data": {
            "usage_count": 30,
            "success_rate": 0.85,
            "avg_execution_time": 250
        }
    }
]


# Test environment configurations
TEST_ENVIRONMENTS = {
    "development": {
        "servers": ["postgres", "filesystem", "git"],
        "config_path": "/dev/mcp/config.json",
        "log_level": "debug",
        "cache_enabled": True,
        "discovery_timeout": 5000
    },
    "testing": {
        "servers": ["postgres", "filesystem"],
        "config_path": "/test/mcp/config.json", 
        "log_level": "info",
        "cache_enabled": False,
        "discovery_timeout": 2000
    },
    "production": {
        "servers": ["postgres", "filesystem", "web", "git"],
        "config_path": "/prod/mcp/config.json",
        "log_level": "warn",
        "cache_enabled": True,
        "discovery_timeout": 10000
    }
}


# Test user contexts for suggestions
TEST_USER_CONTEXTS = [
    {
        "context_id": "web_dev_auth",
        "current_task": "User authentication system",
        "user_intent": "Verify user credentials",
        "project_type": "web_application",
        "recent_tools_used": ["bash", "curl"],
        "recent_failures": [],
        "environment_info": {
            "database": "postgres",
            "framework": "fastapi",
            "language": "python"
        },
        "available_servers": ["postgres", "filesystem", "web"]
    },
    {
        "context_id": "data_analysis",
        "current_task": "Analyze user behavior data",
        "user_intent": "Generate insights from user activity",
        "project_type": "data_analysis",
        "recent_tools_used": ["pandas", "jupyter"],
        "recent_failures": ["postgres_connection_failed"],
        "environment_info": {
            "database": "postgres",
            "platform": "jupyter",
            "language": "python"
        },
        "available_servers": ["postgres", "web"]
    },
    {
        "context_id": "devops_deploy",
        "current_task": "Deploy application to production",
        "user_intent": "Ensure safe deployment",
        "project_type": "devops",
        "recent_tools_used": ["git", "docker", "kubectl"],
        "recent_failures": [],
        "environment_info": {
            "platform": "kubernetes",
            "environment": "production",
            "language": "multiple"
        },
        "available_servers": ["git", "filesystem"]
    }
]


# Performance test datasets
PERFORMANCE_TEST_DATA = {
    "small_dataset": {
        "servers": 3,
        "tools_per_server": 5,
        "patterns": 10,
        "resource_references": 20,
        "expected_discovery_time_ms": 100
    },
    "medium_dataset": {
        "servers": 10,
        "tools_per_server": 15,
        "patterns": 100,
        "resource_references": 200,
        "expected_discovery_time_ms": 500
    },
    "large_dataset": {
        "servers": 25,
        "tools_per_server": 30,
        "patterns": 500,
        "resource_references": 1000,
        "expected_discovery_time_ms": 2000
    }
}


# Error simulation configurations
ERROR_SCENARIOS = {
    "connection_timeout": {
        "error_type": "timeout",
        "delay_ms": 5000,
        "expected_fallback": "graceful_degradation"
    },
    "server_unavailable": {
        "error_type": "connection_refused",
        "error_message": "Connection refused",
        "expected_fallback": "skip_server"
    },
    "malformed_response": {
        "error_type": "json_parse_error",
        "response_data": "invalid json {",
        "expected_fallback": "error_logging"
    },
    "auth_failure": {
        "error_type": "authentication_error",
        "status_code": 401,
        "expected_fallback": "auth_retry"
    }
}


# Compatibility test configurations
COMPATIBILITY_SCENARIOS = {
    "legacy_config_format": {
        "config": {
            "servers": {
                "postgres": {
                    "cmd": "postgres-server",  # Legacy field name
                    "arguments": ["--db", "test"],  # Legacy field name
                    "environment": {"DB_URL": "test"}  # Legacy field name
                }
            }
        },
        "expected_migration": {
            "command": "postgres-server",
            "args": ["--db", "test"],
            "env": {"DB_URL": "test"}
        }
    },
    "missing_optional_fields": {
        "config": {
            "mcpServers": {
                "minimal": {
                    "command": "minimal-server"
                    # Missing args and env
                }
            }
        },
        "expected_defaults": {
            "args": [],
            "env": {}
        }
    }
}


# Test utility functions
def get_test_config(config_name: str) -> Dict[str, Any]:
    """Get a test configuration by name."""
    all_configs = {
        **TEST_MCP_CONFIGS,
        **TEST_CLAUDE_DESKTOP_CONFIGS
    }
    return all_configs.get(config_name, {})


def get_test_patterns(pattern_type: str = None, count: int = None) -> List[Dict[str, Any]]:
    """Get test workflow patterns, optionally filtered and limited."""
    patterns = TEST_WORKFLOW_PATTERNS.copy()
    
    if pattern_type:
        patterns = [p for p in patterns if p.get('intent') == pattern_type]
    
    if count:
        patterns = patterns[:count]
    
    return patterns


def get_test_user_context(context_id: str = None) -> Dict[str, Any]:
    """Get a test user context by ID or return default."""
    if context_id:
        for context in TEST_USER_CONTEXTS:
            if context['context_id'] == context_id:
                return context
    
    return TEST_USER_CONTEXTS[0]  # Return first as default


def generate_test_servers(count: int) -> Dict[str, Dict[str, Any]]:
    """Generate a specified number of test server configurations."""
    servers = {}
    for i in range(count):
        server_name = f"test_server_{i}"
        servers[server_name] = {
            "command": f"test-server-{i}",
            "args": [f"--config", f"config{i}.json"],
            "env": {"TEST_VAR": f"value_{i}"},
            "source": "generated_test"
        }
    return servers


def create_test_workflow_pattern(
    context: str,
    tools: List[str],
    score: float = 0.8,
    project_type: str = "test_project"
) -> Dict[str, Any]:
    """Create a test workflow pattern with specified parameters."""
    return {
        "context": context,
        "tools": tools,
        "score": score,
        "project_type": project_type,
        "intent": "test_intent",
        "success": True,
        "usage_count": 1,
        "last_used": datetime.now(timezone.utc).isoformat()
    }


# Export main data structures and functions
__all__ = [
    'TEST_MCP_CONFIGS',
    'TEST_CLAUDE_DESKTOP_CONFIGS',
    'TEST_WORKFLOW_PATTERNS',
    'TEST_HOOK_SCENARIOS',
    'TEST_RESOURCE_PATTERNS',
    'TEST_SLASH_COMMANDS',
    'TEST_ENVIRONMENTS',
    'TEST_USER_CONTEXTS',
    'PERFORMANCE_TEST_DATA',
    'ERROR_SCENARIOS',
    'COMPATIBILITY_SCENARIOS',
    'get_test_config',
    'get_test_patterns',
    'get_test_user_context',
    'generate_test_servers',
    'create_test_workflow_pattern'
]