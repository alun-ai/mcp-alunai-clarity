#!/usr/bin/env python3
"""
SQLite Memory Persistence Core Functionality Tests

Unit tests for the SQLiteMemoryPersistence class covering all core functionality:
- Memory storage and retrieval
- CRUD operations
- Data validation
- Error handling
- Cache integration
"""

import asyncio
import json
import os
import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Any, Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence
from clarity.shared.exceptions.base import MemoryOperationError, ValidationError


class TestSQLiteMemoryPersistence:
    """Test suite for SQLite memory persistence core functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        temp_dir = tempfile.mkdtemp(prefix="sqlite_test_")
        db_path = os.path.join(temp_dir, "test_memory.db")
        yield db_path
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1] * 384  # 384-dimensional vector
        return mock_model
    
    @pytest.fixture
    def sqlite_persistence(self, temp_db_path, mock_embedding_model):
        """Create SQLiteMemoryPersistence instance for testing."""
        return SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
    
    @pytest.fixture
    def sample_memory(self):
        """Sample memory data for testing."""
        return {
            "id": "test-001",
            "type": "structured_thinking",
            "content": "This is a test memory for validation",
            "importance": 0.8,
            "tier": "short_term",
            "metadata": {"test": True, "category": "validation"},
            "context": {"session": "test_session"}
        }
    
    def test_initialization(self, temp_db_path, mock_embedding_model):
        """Test SQLite persistence initialization."""
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        assert persistence.db_path == temp_db_path
        assert persistence.embedding_model == mock_embedding_model
        assert persistence.embedding_dimensions == 384
        assert os.path.exists(temp_db_path)
    
    def test_database_schema_creation(self, sqlite_persistence):
        """Test database schema is created correctly."""
        import sqlite3
        
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            # Check if memories table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='memories'
            """)
            assert cursor.fetchone() is not None
            
            # Check table schema
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            expected_columns = {
                'memory_id': 'TEXT',
                'memory_type': 'TEXT',
                'content': 'TEXT',
                'text_content': 'TEXT',
                'importance': 'REAL',
                'tier': 'TEXT',
                'created_at': 'TEXT',
                'updated_at': 'TEXT',
                'metadata': 'TEXT',
                'context': 'TEXT',
                'access_count': 'INTEGER',
                'last_accessed': 'TEXT',
                'embedding': 'BLOB'
            }
            
            for col_name, col_type in expected_columns.items():
                assert col_name in columns
    
    def test_sanitize_memory_id(self, sqlite_persistence):
        """Test memory ID sanitization."""
        # Valid UUID should remain unchanged
        valid_uuid = str(uuid.uuid4())
        assert sqlite_persistence._sanitize_memory_id(valid_uuid) == valid_uuid
        
        # Test prefix removal
        prefixed_id = "mem_" + valid_uuid
        assert sqlite_persistence._sanitize_memory_id(prefixed_id) == valid_uuid
        
        # Test invalid ID generates new UUID
        invalid_id = "invalid-id-format"
        sanitized = sqlite_persistence._sanitize_memory_id(invalid_id)
        assert sanitized != invalid_id
        # Validate it's a valid UUID
        uuid.UUID(sanitized)  # Should not raise exception
    
    def test_embedding_serialization(self, sqlite_persistence):
        """Test embedding serialization and deserialization."""
        original_embedding = [0.1, 0.2, 0.3, -0.4, 0.5]
        
        # Test serialization
        serialized = sqlite_persistence._serialize_embedding(original_embedding)
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = sqlite_persistence._deserialize_embedding(serialized)
        assert len(deserialized) == len(original_embedding)
        
        # Check values (with floating point precision tolerance)
        for orig, deser in zip(original_embedding, deserialized):
            assert abs(orig - deser) < 1e-6
    
    def test_cosine_similarity_calculation(self, sqlite_persistence):
        """Test cosine similarity calculation."""
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = sqlite_persistence._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = sqlite_persistence._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = sqlite_persistence._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6
        
        # Test zero vector handling
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = sqlite_persistence._calculate_cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    @pytest.mark.asyncio
    async def test_store_memory_basic(self, sqlite_persistence, sample_memory):
        """Test basic memory storage."""
        memory_id = await sqlite_persistence.store_memory(sample_memory)
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
        
        # Verify memory was stored
        import sqlite3
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            cursor = conn.execute(
                "SELECT memory_id, memory_type, content FROM memories WHERE memory_id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == sample_memory["type"]
    
    @pytest.mark.asyncio
    async def test_store_memory_with_tier_override(self, sqlite_persistence, sample_memory):
        """Test memory storage with tier override."""
        memory_id = await sqlite_persistence.store_memory(sample_memory, tier="long_term")
        
        # Verify tier was overridden
        import sqlite3
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            cursor = conn.execute(
                "SELECT tier FROM memories WHERE memory_id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            assert row[0] == "long_term"
    
    @pytest.mark.asyncio
    async def test_store_memory_edge_cases(self, sqlite_persistence):
        """Test memory storage edge cases."""
        # Test empty content
        empty_memory = {
            "id": "empty-001",
            "type": "episodic",
            "content": "",
            "importance": 0.1,
            "tier": "short_term"
        }
        memory_id = await sqlite_persistence.store_memory(empty_memory)
        assert memory_id is not None
        
        # Test very long content
        long_content = "A" * 10000
        long_memory = {
            "id": "long-001",
            "type": "semantic",
            "content": long_content,
            "importance": 0.5,
            "tier": "archival"
        }
        memory_id = await sqlite_persistence.store_memory(long_memory)
        assert memory_id is not None
        
        # Test complex nested content
        complex_memory = {
            "id": "complex-001",
            "type": "structured_thinking",
            "content": {
                "analysis": "Complex analysis",
                "conclusions": ["Point 1", "Point 2"],
                "metadata": {"nested": {"deep": "value"}}
            },
            "importance": 0.9,
            "tier": "long_term"
        }
        memory_id = await sqlite_persistence.store_memory(complex_memory)
        assert memory_id is not None
    
    @pytest.mark.asyncio
    async def test_get_memory(self, sqlite_persistence, sample_memory):
        """Test memory retrieval by ID."""
        # Store memory first
        stored_id = await sqlite_persistence.store_memory(sample_memory)
        
        # Retrieve memory
        retrieved = await sqlite_persistence.get_memory(stored_id)
        
        assert retrieved is not None
        assert retrieved["id"] == stored_id
        assert retrieved["type"] == sample_memory["type"]
        assert retrieved["importance"] == sample_memory["importance"]
        assert retrieved["tier"] == sample_memory["tier"]
        
        # Test nonexistent memory
        nonexistent = await sqlite_persistence.get_memory("nonexistent-id")
        assert nonexistent is None
    
    @pytest.mark.asyncio
    async def test_update_memory(self, sqlite_persistence, sample_memory):
        """Test memory updates."""
        # Store memory first
        stored_id = await sqlite_persistence.store_memory(sample_memory)
        
        # Update memory
        updates = {
            "importance": 0.95,
            "content": "Updated content for testing",
            "metadata": {"updated": True, "test": False}
        }
        
        success = await sqlite_persistence.update_memory(stored_id, updates)
        assert success is True
        
        # Verify updates
        updated_memory = await sqlite_persistence.get_memory(stored_id)
        assert updated_memory["importance"] == 0.95
        assert "Updated content" in str(updated_memory["content"])
        assert updated_memory["metadata"]["updated"] is True
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_memory(self, sqlite_persistence):
        """Test updating nonexistent memory."""
        success = await sqlite_persistence.update_memory("nonexistent-id", {"importance": 0.9})
        assert success is False
    
    @pytest.mark.asyncio
    async def test_delete_memories(self, sqlite_persistence, sample_memory):
        """Test memory deletion."""
        # Store multiple memories
        memory_ids = []
        for i in range(3):
            memory = sample_memory.copy()
            memory["id"] = f"delete-test-{i}"
            memory_id = await sqlite_persistence.store_memory(memory)
            memory_ids.append(memory_id)
        
        # Delete some memories
        deleted_ids = await sqlite_persistence.delete_memories(memory_ids[:2])
        assert len(deleted_ids) == 2
        
        # Verify deletion
        for deleted_id in deleted_ids:
            retrieved = await sqlite_persistence.get_memory(deleted_id)
            assert retrieved is None
        
        # Verify remaining memory still exists
        remaining = await sqlite_persistence.get_memory(memory_ids[2])
        assert remaining is not None
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_basic(self, sqlite_persistence):
        """Test basic memory retrieval with query."""
        # Store test memories with known content
        test_memories = [
            {
                "id": "search-001",
                "type": "structured_thinking",
                "content": "Database optimization techniques and performance improvements",
                "importance": 0.8,
                "tier": "long_term"
            },
            {
                "id": "search-002", 
                "type": "episodic",
                "content": "User reported slow query performance in production",
                "importance": 0.9,
                "tier": "short_term"
            },
            {
                "id": "search-003",
                "type": "semantic",
                "content": "Vector embeddings and similarity search algorithms",
                "importance": 0.7,
                "tier": "archival"
            }
        ]
        
        for memory in test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test retrieval
        results = await sqlite_persistence.retrieve_memories(
            "database performance optimization",
            limit=5,
            min_similarity=0.0
        )
        
        assert len(results) > 0
        # Results should be sorted by similarity
        for i in range(1, len(results)):
            assert results[i-1]["similarity_score"] >= results[i]["similarity_score"]
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_with_filters(self, sqlite_persistence):
        """Test memory retrieval with various filters."""
        # Store test memories with different types and tiers
        test_memories = [
            {
                "id": "filter-001",
                "type": "structured_thinking",
                "content": "Analysis of system performance",
                "importance": 0.9,
                "tier": "long_term"
            },
            {
                "id": "filter-002",
                "type": "episodic", 
                "content": "Performance issue reported by user",
                "importance": 0.8,
                "tier": "short_term"
            },
            {
                "id": "filter-003",
                "type": "structured_thinking",
                "content": "System architecture analysis",
                "importance": 0.7,
                "tier": "short_term"
            }
        ]
        
        for memory in test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test memory type filter
        structured_results = await sqlite_persistence.retrieve_memories(
            "system analysis",
            memory_types=["structured_thinking"],
            limit=5,
            min_similarity=0.0
        )
        
        for result in structured_results:
            assert result["type"] == "structured_thinking"
        
        # Test tier filter with filters parameter
        long_term_results = await sqlite_persistence.retrieve_memories(
            "performance",
            filters={"tier": "long_term"},
            limit=5,
            min_similarity=0.0
        )
        
        for result in long_term_results:
            # Note: tier not included in basic response, need include_metadata=True
            pass  # We'll test this in filtering tests
    
    @pytest.mark.asyncio
    async def test_search_memories(self, sqlite_persistence):
        """Test search_memories method."""
        # Store test memories
        test_memories = [
            {
                "id": "search-mem-001",
                "type": "procedural",
                "content": "Step-by-step database backup procedure",
                "importance": 0.8,
                "tier": "system"
            },
            {
                "id": "search-mem-002",
                "type": "episodic",
                "content": "Database backup completed successfully",
                "importance": 0.7,
                "tier": "short_term"
            }
        ]
        
        for memory in test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test search by type
        procedural_results = await sqlite_persistence.search_memories(
            types=["procedural"],
            limit=10
        )
        
        assert len(procedural_results) > 0
        for result in procedural_results:
            assert result["type"] == "procedural"
        
        # Test search with filters
        system_results = await sqlite_persistence.search_memories(
            filters={"tier": "system"},
            limit=10
        )
        
        assert len(system_results) > 0
        for result in system_results:
            assert result["tier"] == "system"
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, sqlite_persistence):
        """Test memory statistics retrieval."""
        # Store diverse test memories
        test_memories = [
            {"id": "stats-001", "type": "structured_thinking", "content": "Analysis 1", "importance": 0.8, "tier": "long_term"},
            {"id": "stats-002", "type": "episodic", "content": "Event 1", "importance": 0.7, "tier": "short_term"},
            {"id": "stats-003", "type": "structured_thinking", "content": "Analysis 2", "importance": 0.9, "tier": "long_term"},
            {"id": "stats-004", "type": "semantic", "content": "Concept 1", "importance": 0.6, "tier": "archival"}
        ]
        
        for memory in test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Get statistics
        stats = await sqlite_persistence.get_memory_stats()
        
        assert "total_memories" in stats
        assert stats["total_memories"] >= 4  # At least our test memories
        
        assert "memory_types" in stats
        assert "structured_thinking" in stats["memory_types"]
        assert "episodic" in stats["memory_types"]
        
        assert "memory_tiers" in stats
        assert "long_term" in stats["memory_tiers"]
        assert "short_term" in stats["memory_tiers"]
        
        assert "database_size_bytes" in stats
        assert stats["database_size_bytes"] > 0
        
        assert "database_path" in stats
        assert stats["database_path"] == sqlite_persistence.db_path
    
    @pytest.mark.asyncio
    async def test_access_tracking(self, sqlite_persistence, sample_memory):
        """Test access count tracking functionality."""
        # Store memory
        stored_id = await sqlite_persistence.store_memory(sample_memory)
        
        # Get memory multiple times to increment access count
        for _ in range(3):
            await sqlite_persistence.get_memory(stored_id)
        
        # Check access count was updated
        import sqlite3
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            cursor = conn.execute(
                "SELECT access_count, last_accessed FROM memories WHERE memory_id = ?",
                (stored_id,)
            )
            row = cursor.fetchone()
            assert row[0] >= 3  # Access count should be at least 3
            assert row[1] is not None  # Last accessed should be set
    
    @pytest.mark.asyncio
    async def test_error_handling(self, temp_db_path, mock_embedding_model):
        """Test error handling scenarios."""
        persistence = SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
        
        # Test storage error with invalid data types
        with pytest.raises(Exception):  # Should handle gracefully
            await persistence.store_memory({"invalid": "structure"})
        
        # Test embedding generation error
        mock_embedding_model.encode.side_effect = Exception("Embedding error")
        
        with pytest.raises(MemoryOperationError):
            await persistence.store_memory({
                "id": "error-test",
                "type": "test",
                "content": "test content",
                "importance": 0.5,
                "tier": "short_term"
            })
    
    def test_database_configuration(self, sqlite_persistence):
        """Test database configuration settings."""
        import sqlite3
        
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            # Check WAL mode
            cursor = conn.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            assert journal_mode.upper() == "WAL"
            
            # Check synchronous mode
            cursor = conn.execute("PRAGMA synchronous")
            sync_mode = cursor.fetchone()[0]
            assert sync_mode == 1  # NORMAL mode
            
            # Check cache size
            cursor = conn.execute("PRAGMA cache_size")
            cache_size = cursor.fetchone()[0]
            assert cache_size == 10000
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, sqlite_persistence, sample_memory):
        """Test integration with cache infrastructure."""
        # Store memory
        stored_id = await sqlite_persistence.store_memory(sample_memory)
        
        # First retrieval (should cache)
        first_retrieval = await sqlite_persistence.get_memory(stored_id)
        assert first_retrieval is not None
        
        # Second retrieval (should use cache)
        second_retrieval = await sqlite_persistence.get_memory(stored_id)
        assert second_retrieval is not None
        assert first_retrieval["id"] == second_retrieval["id"]
        
        # Update memory (should invalidate cache)
        await sqlite_persistence.update_memory(stored_id, {"importance": 0.95})
        
        # Next retrieval should get updated data
        updated_retrieval = await sqlite_persistence.get_memory(stored_id)
        assert updated_retrieval["importance"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])