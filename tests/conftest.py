"""
Test configuration and fixtures for Alunai Clarity test suite.
"""

import asyncio
import json
import os
import tempfile
import uuid
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import numpy as np

from clarity.domains.manager import MemoryDomainManager
from clarity.domains.persistence import PersistenceDomain
from clarity.utils.config import create_default_config
from clarity.utils.embeddings import EmbeddingManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_config_file() -> Generator[str, None, None]:
    """Create a temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            "server": {
                "host": "localhost",
                "port": 8080,
                "log_level": "INFO"
            },
            "alunai-clarity": {
                "max_short_term_items": 100,
                "max_long_term_items": 500,
                "max_archival_items": 1000,
                "short_term_threshold": 0.3,
                "consolidation_interval_hours": 1
            },
            "qdrant": {
                "path": ":memory:",
                "index_params": {
                    "m": 16,
                    "ef_construct": 100,
                    "full_scan_threshold": 1000
                }
            },
            "embedding": {
                "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "cache_dir": None
            },
            "retrieval": {
                "similarity_weight": 0.4,
                "recency_weight": 0.3,
                "importance_weight": 0.3
            },
            "autocode": {
                "enabled": True,
                "command_learning": {
                    "enabled": True,
                    "min_confidence_threshold": 0.3,
                    "max_suggestions": 3
                },
                "pattern_detection": {
                    "enabled": True,
                    "supported_languages": ["python", "javascript", "typescript"],
                    "max_scan_depth": 3
                },
                "session_analysis": {
                    "enabled": True,
                    "track_architectural_decisions": True,
                    "extract_learning_patterns": True
                },
                "history_navigation": {
                    "enabled": True,
                    "similarity_threshold": 0.6,
                    "context_window_days": 7
                }
            }
        }
        json.dump(config, f, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def test_config(temp_config_file: str) -> Dict[str, Any]:
    """Load test configuration from temporary file."""
    with open(temp_config_file, 'r') as f:
        return json.load(f)


@pytest.fixture
def temp_data_dir() -> Generator[str, None, None]:
    """Create temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_embedding_manager() -> EmbeddingManager:
    """Create a mock embedding manager for testing."""
    config = {
        "embedding": {
            "default_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "cache_dir": None
        }
    }
    
    manager = EmbeddingManager(config)
    
    # Mock the model to avoid downloading during tests
    manager.model = MagicMock()
    manager.model.encode = MagicMock(return_value=np.random.rand(384))
    
    return manager


@pytest.fixture
async def mock_persistence_domain(test_config: Dict[str, Any]) -> AsyncGenerator[PersistenceDomain, None]:
    """Create a mock persistence domain for testing."""
    # Override config to use in-memory Qdrant
    test_config["qdrant"]["path"] = ":memory:"
    
    domain = PersistenceDomain(test_config)
    
    # Mock Qdrant client to avoid actual database operations
    domain.qdrant_client = AsyncMock()
    domain.embedding_manager = MagicMock()
    domain.embedding_manager.get_embedding.return_value = np.random.rand(384)
    domain.embedding_manager.calculate_similarity.return_value = 0.8
    
    await domain.initialize()
    yield domain


@pytest.fixture
async def mock_domain_manager(test_config: Dict[str, Any]) -> AsyncGenerator[MemoryDomainManager, None]:
    """Create a mock domain manager for testing."""
    # Override config to use in-memory storage
    test_config["qdrant"]["path"] = ":memory:"
    
    manager = MemoryDomainManager(test_config)
    
    # Mock the persistence domain
    manager.persistence_domain = AsyncMock()
    manager.persistence_domain.initialize = AsyncMock()
    manager.persistence_domain.store_memory = AsyncMock()
    manager.persistence_domain.search_memories = AsyncMock(return_value=[])
    manager.persistence_domain.generate_embedding = AsyncMock(return_value=np.random.rand(384))
    manager.persistence_domain.get_memory = AsyncMock(return_value=None)
    manager.persistence_domain.get_memory_tier = AsyncMock(return_value="short_term")
    manager.persistence_domain.update_memory = AsyncMock()
    manager.persistence_domain.delete_memories = AsyncMock(return_value=True)
    manager.persistence_domain.get_memory_stats = AsyncMock(return_value={})
    manager.persistence_domain.list_memories = AsyncMock(return_value=[])
    manager.persistence_domain.get_metadata = AsyncMock(return_value=None)
    manager.persistence_domain.set_metadata = AsyncMock()
    
    # Mock other domains
    manager.episodic_domain = AsyncMock()
    manager.episodic_domain.initialize = AsyncMock()
    manager.episodic_domain.process_memory = AsyncMock(side_effect=lambda x: x)
    manager.episodic_domain.get_stats = AsyncMock(return_value={})
    
    manager.semantic_domain = AsyncMock()
    manager.semantic_domain.initialize = AsyncMock()
    manager.semantic_domain.process_memory = AsyncMock(side_effect=lambda x: x)
    manager.semantic_domain.get_stats = AsyncMock(return_value={})
    
    manager.temporal_domain = AsyncMock()
    manager.temporal_domain.initialize = AsyncMock()
    manager.temporal_domain.process_new_memory = AsyncMock(side_effect=lambda x: {**x, "created_at": "2023-01-01T00:00:00", "last_accessed": "2023-01-01T00:00:00", "last_modified": "2023-01-01T00:00:00", "access_count": 0})
    manager.temporal_domain.adjust_memory_relevance = AsyncMock(side_effect=lambda x, q: x)
    manager.temporal_domain.update_memory_access = AsyncMock()
    manager.temporal_domain.update_memory_modification = AsyncMock(side_effect=lambda x: x)
    manager.temporal_domain.get_stats = AsyncMock(return_value={})
    
    manager.autocode_domain = AsyncMock()
    manager.autocode_domain.initialize = AsyncMock()
    manager.autocode_domain.set_command_learner = AsyncMock()
    manager.autocode_domain.get_stats = AsyncMock(return_value={})
    
    await manager.initialize()
    yield manager


@pytest.fixture
def sample_memory_data() -> Dict[str, Any]:
    """Sample memory data for testing."""
    return {
        "conversation": {
            "type": "conversation",
            "content": {
                "role": "user",
                "message": "Hello, how are you today?"
            },
            "importance": 0.7,
            "metadata": {"session_id": "test_session_1"},
            "context": {"user_id": "test_user"}
        },
        "fact": {
            "type": "fact",
            "content": {
                "fact": "Python is a high-level programming language",
                "confidence": 0.95,
                "category": "programming"
            },
            "importance": 0.9,
            "metadata": {"source": "knowledge_base"},
            "context": {"topic": "programming"}
        },
        "project_pattern": {
            "type": "project_pattern",
            "content": {
                "pattern_type": "framework",
                "framework": "FastAPI",
                "language": "python",
                "structure": {
                    "directories": ["app", "tests", "docs"],
                    "files": ["main.py", "requirements.txt", "README.md"]
                }
            },
            "importance": 0.8,
            "metadata": {"project_name": "test_api"},
            "context": {"scan_date": "2023-01-01"}
        },
        "command_pattern": {
            "type": "command_pattern",
            "content": {
                "command": "pytest -v",
                "context": {"project_type": "python", "framework": "pytest"},
                "success_rate": 0.95,
                "platform": "linux"
            },
            "importance": 0.6,
            "metadata": {"usage_count": 42},
            "context": {"working_directory": "/test/project"}
        }
    }


@pytest.fixture
def sample_mcp_tools() -> list:
    """Sample MCP tool configurations for testing."""
    return [
        {
            "name": "test_query",
            "description": "Test database query tool",
            "parameters": {
                "query": {"type": "string", "description": "SQL query"},
                "database": {"type": "string", "description": "Database name"}
            },
            "server_name": "test_db",
            "category": "database"
        },
        {
            "name": "test_navigate",
            "description": "Test web navigation tool",
            "parameters": {
                "url": {"type": "string", "description": "URL to navigate"},
                "action": {"type": "string", "description": "Action to perform"}
            },
            "server_name": "test_web",
            "category": "web_automation"
        }
    ]


@pytest.fixture
def sample_claude_config() -> Dict[str, Any]:
    """Sample Claude Desktop configuration for testing."""
    return {
        "mcpServers": {
            "alunai-clarity": {
                "command": "python",
                "args": ["-m", "clarity"],
                "env": {
                    "MEMORY_FILE_PATH": "/test/memory.json"
                },
                "type": "stdio"
            },
            "postgres": {
                "command": "npx",
                "args": ["@postgresql/mcp-server"],
                "type": "stdio"
            },
            "playwright": {
                "command": "npx", 
                "args": ["@playwright/mcp-server"],
                "type": "stdio"
            }
        }
    }


@pytest.fixture
def memory_id_generator():
    """Generate unique memory IDs for testing."""
    def generate():
        return f"mem_{str(uuid.uuid4())}"
    return generate


class AsyncContextManager:
    """Helper class for async context managers in tests."""
    def __init__(self, obj):
        self.obj = obj
    
    async def __aenter__(self):
        return self.obj
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Factory for creating async context managers in tests."""
    return AsyncContextManager


# Performance test utilities
@pytest.fixture
def performance_test_data():
    """Generate test data for performance testing."""
    def generate_memories(count: int):
        memories = []
        for i in range(count):
            memory = {
                "id": f"perf_mem_{i}",
                "type": "fact",
                "content": {
                    "fact": f"Test fact number {i}",
                    "confidence": 0.8
                },
                "importance": 0.5,
                "created_at": "2023-01-01T00:00:00",
                "embedding": np.random.rand(384).tolist()
            }
            memories.append(memory)
        return memories
    return generate_memories


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_qdrant: mark test as requiring Qdrant")
    config.addinivalue_line("markers", "requires_embedding: mark test as requiring embedding model")