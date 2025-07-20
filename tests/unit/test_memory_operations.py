"""
Unit tests for core memory operations in Alunai Clarity.
"""

import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from clarity.domains.manager import MemoryDomainManager
from clarity.utils.schema import validate_memory


@pytest.mark.unit
class TestMemoryStorage:
    """Test memory storage operations."""
    
    @pytest.mark.asyncio
    async def test_store_conversation_memory(self, mock_domain_manager: MemoryDomainManager, sample_memory_data: Dict[str, Any]):
        """Test storing conversation memory."""
        conversation_data = sample_memory_data["conversation"]
        
        memory_id = await mock_domain_manager.store_memory(
            memory_type=conversation_data["type"],
            content=conversation_data["content"],
            importance=conversation_data["importance"],
            metadata=conversation_data["metadata"],
            context=conversation_data["context"]
        )
        
        assert memory_id.startswith("mem_")
        mock_domain_manager.persistence_domain.store_memory.assert_called_once()
        
        # Verify the memory was processed by the episodic domain
        mock_domain_manager.episodic_domain.process_memory.assert_called_once()
        mock_domain_manager.semantic_domain.process_memory.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_store_fact_memory(self, mock_domain_manager: MemoryDomainManager, sample_memory_data: Dict[str, Any]):
        """Test storing fact memory."""
        fact_data = sample_memory_data["fact"]
        
        memory_id = await mock_domain_manager.store_memory(
            memory_type=fact_data["type"],
            content=fact_data["content"],
            importance=fact_data["importance"],
            metadata=fact_data["metadata"],
            context=fact_data["context"]
        )
        
        assert memory_id.startswith("mem_")
        mock_domain_manager.persistence_domain.store_memory.assert_called_once()
        
        # Verify the memory was processed by the semantic domain
        mock_domain_manager.semantic_domain.process_memory.assert_called_once()
        mock_domain_manager.episodic_domain.process_memory.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_store_code_memory(self, mock_domain_manager: MemoryDomainManager):
        """Test storing code memory (should be processed by both domains)."""
        code_content = {
            "code": "def hello_world():\n    print('Hello, World!')",
            "language": "python",
            "description": "Simple hello world function"
        }
        
        memory_id = await mock_domain_manager.store_memory(
            memory_type="code",
            content=code_content,
            importance=0.8
        )
        
        assert memory_id.startswith("mem_")
        mock_domain_manager.persistence_domain.store_memory.assert_called_once()
        
        # Verify the memory was processed by both domains
        mock_domain_manager.episodic_domain.process_memory.assert_called_once()
        mock_domain_manager.semantic_domain.process_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_autocode_memory(self, mock_domain_manager: MemoryDomainManager, sample_memory_data: Dict[str, Any]):
        """Test storing AutoCode-specific memory types."""
        pattern_data = sample_memory_data["project_pattern"]
        
        memory_id = await mock_domain_manager.store_memory(
            memory_type=pattern_data["type"],
            content=pattern_data["content"],
            importance=pattern_data["importance"],
            metadata=pattern_data["metadata"],
            context=pattern_data["context"]
        )
        
        assert memory_id.startswith("mem_")
        mock_domain_manager.persistence_domain.store_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_tier_assignment(self, mock_domain_manager: MemoryDomainManager):
        """Test memory tier assignment based on importance."""
        # High importance should go to short_term
        high_importance_id = await mock_domain_manager.store_memory(
            memory_type="fact",
            content={"fact": "Important fact"},
            importance=0.8
        )
        
        # Low importance should go to long_term  
        low_importance_id = await mock_domain_manager.store_memory(
            memory_type="fact",
            content={"fact": "Less important fact"},
            importance=0.1
        )
        
        # Check that store_memory was called with appropriate tiers
        calls = mock_domain_manager.persistence_domain.store_memory.call_args_list
        assert len(calls) == 2
        
        # First call (high importance) should use short_term
        first_call_args = calls[0][0]
        assert "short_term" in str(calls[0])
        
        # Second call (low importance) should use long_term
        second_call_args = calls[1][0]
        assert "long_term" in str(calls[1])


@pytest.mark.unit
class TestMemoryRetrieval:
    """Test memory retrieval operations."""
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_basic(self, mock_domain_manager: MemoryDomainManager):
        """Test basic memory retrieval."""
        # Mock return data
        mock_memories = [
            {
                "id": "mem_1",
                "type": "fact",
                "content": {"fact": "Test fact"},
                "similarity": 0.8,
                "created_at": "2023-01-01T00:00:00",
                "importance": 0.7
            }
        ]
        mock_domain_manager.persistence_domain.search_memories.return_value = mock_memories
        
        results = await mock_domain_manager.retrieve_memories(
            query="test query",
            limit=5,
            min_similarity=0.6
        )
        
        assert len(results) == 1
        assert results[0]["id"] == "mem_1"
        assert results[0]["similarity"] == 0.8
        
        # Verify the search was called correctly
        mock_domain_manager.persistence_domain.search_memories.assert_called_once()
        mock_domain_manager.persistence_domain.generate_embedding.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_with_types(self, mock_domain_manager: MemoryDomainManager):
        """Test memory retrieval with type filtering."""
        mock_memories = []
        mock_domain_manager.persistence_domain.search_memories.return_value = mock_memories
        
        results = await mock_domain_manager.retrieve_memories(
            query="test query",
            limit=3,
            memory_types=["fact", "conversation"],
            min_similarity=0.7
        )
        
        assert len(results) == 0
        
        # Verify search was called with correct parameters
        call_args = mock_domain_manager.persistence_domain.search_memories.call_args
        assert call_args[1]["types"] == ["fact", "conversation"]
        assert call_args[1]["min_similarity"] == 0.7
        assert call_args[1]["limit"] == 3
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_with_metadata(self, mock_domain_manager: MemoryDomainManager):
        """Test memory retrieval with metadata inclusion."""
        mock_memories = [
            {
                "id": "mem_1",
                "type": "fact",
                "content": {"fact": "Test fact"},
                "similarity": 0.9,
                "metadata": {"source": "test"},
                "created_at": "2023-01-01T00:00:00",
                "last_accessed": "2023-01-01T01:00:00",
                "importance": 0.8
            }
        ]
        mock_domain_manager.persistence_domain.search_memories.return_value = mock_memories
        
        results = await mock_domain_manager.retrieve_memories(
            query="test",
            include_metadata=True
        )
        
        assert len(results) == 1
        assert "metadata" in results[0]
        assert "created_at" in results[0]
        assert "last_accessed" in results[0]
        assert "importance" in results[0]
        assert results[0]["metadata"]["source"] == "test"
    
    @pytest.mark.asyncio
    async def test_retrieve_updates_access_time(self, mock_domain_manager: MemoryDomainManager):
        """Test that memory retrieval updates access times."""
        mock_memories = [
            {"id": "mem_1", "type": "fact", "content": {"fact": "Test"}, "similarity": 0.8},
            {"id": "mem_2", "type": "fact", "content": {"fact": "Test 2"}, "similarity": 0.7}
        ]
        mock_domain_manager.persistence_domain.search_memories.return_value = mock_memories
        
        await mock_domain_manager.retrieve_memories("test")
        
        # Verify access times were updated for all retrieved memories
        assert mock_domain_manager.temporal_domain.update_memory_access.call_count == 2
        mock_domain_manager.temporal_domain.update_memory_access.assert_any_call("mem_1")
        mock_domain_manager.temporal_domain.update_memory_access.assert_any_call("mem_2")


@pytest.mark.unit
class TestMemoryUpdating:
    """Test memory update operations."""
    
    @pytest.mark.asyncio
    async def test_update_memory_content(self, mock_domain_manager: MemoryDomainManager):
        """Test updating memory content."""
        # Mock existing memory
        existing_memory = {
            "id": "mem_1",
            "type": "fact",
            "content": {"fact": "Old fact"},
            "importance": 0.5,
            "metadata": {},
            "context": {}
        }
        mock_domain_manager.persistence_domain.get_memory.return_value = existing_memory
        mock_domain_manager.persistence_domain.get_memory_tier.return_value = "short_term"
        
        updates = {
            "content": {"fact": "Updated fact", "confidence": 0.9},
            "importance": 0.8
        }
        
        result = await mock_domain_manager.update_memory("mem_1", updates)
        
        assert result is True
        mock_domain_manager.persistence_domain.get_memory.assert_called_once_with("mem_1")
        mock_domain_manager.persistence_domain.update_memory.assert_called_once()
        
        # Verify content changes triggered reprocessing
        mock_domain_manager.semantic_domain.process_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_memory_importance_tier_change(self, mock_domain_manager: MemoryDomainManager):
        """Test updating memory importance that changes tier."""
        existing_memory = {
            "id": "mem_1",
            "type": "fact", 
            "content": {"fact": "Test fact"},
            "importance": 0.5,
            "metadata": {},
            "context": {}
        }
        mock_domain_manager.persistence_domain.get_memory.return_value = existing_memory
        mock_domain_manager.persistence_domain.get_memory_tier.return_value = "long_term"
        
        # Update importance from 0.5 to 0.8 (should move to short_term)
        updates = {"importance": 0.8}
        
        result = await mock_domain_manager.update_memory("mem_1", updates)
        
        assert result is True
        # Memory should be updated with new tier
        mock_domain_manager.persistence_domain.update_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_memory(self, mock_domain_manager: MemoryDomainManager):
        """Test updating non-existent memory."""
        mock_domain_manager.persistence_domain.get_memory.return_value = None
        
        result = await mock_domain_manager.update_memory("nonexistent", {"importance": 0.9})
        
        assert result is False
        mock_domain_manager.persistence_domain.update_memory.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_update_memory_metadata(self, mock_domain_manager: MemoryDomainManager):
        """Test updating memory metadata."""
        existing_memory = {
            "id": "mem_1",
            "type": "fact",
            "content": {"fact": "Test fact"},
            "importance": 0.5,
            "metadata": {"source": "original"},
            "context": {"user": "test"}
        }
        mock_domain_manager.persistence_domain.get_memory.return_value = existing_memory
        mock_domain_manager.persistence_domain.get_memory_tier.return_value = "short_term"
        
        updates = {
            "metadata": {"category": "important"},
            "context": {"session": "new_session"}
        }
        
        result = await mock_domain_manager.update_memory("mem_1", updates)
        
        assert result is True
        mock_domain_manager.temporal_domain.update_memory_modification.assert_called_once()


@pytest.mark.unit
class TestMemoryDeletion:
    """Test memory deletion operations."""
    
    @pytest.mark.asyncio
    async def test_delete_memories_success(self, mock_domain_manager: MemoryDomainManager):
        """Test successful memory deletion."""
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        mock_domain_manager.persistence_domain.delete_memories.return_value = True
        
        result = await mock_domain_manager.delete_memories(memory_ids)
        
        assert result is True
        mock_domain_manager.persistence_domain.delete_memories.assert_called_once_with(memory_ids)
    
    @pytest.mark.asyncio
    async def test_delete_memories_failure(self, mock_domain_manager: MemoryDomainManager):
        """Test failed memory deletion."""
        memory_ids = ["mem_1", "mem_2"]
        mock_domain_manager.persistence_domain.delete_memories.return_value = False
        
        result = await mock_domain_manager.delete_memories(memory_ids)
        
        assert result is False
        mock_domain_manager.persistence_domain.delete_memories.assert_called_once_with(memory_ids)


@pytest.mark.unit
class TestMemoryListing:
    """Test memory listing operations."""
    
    @pytest.mark.asyncio
    async def test_list_memories_basic(self, mock_domain_manager: MemoryDomainManager):
        """Test basic memory listing."""
        mock_memories = [
            {
                "id": "mem_1",
                "type": "fact",
                "content": {"fact": "Test fact 1"},
                "created_at": "2023-01-01T00:00:00",
                "importance": 0.7,
                "tier": "short_term"
            },
            {
                "id": "mem_2", 
                "type": "conversation",
                "content": {"role": "user", "message": "Hello"},
                "created_at": "2023-01-01T01:00:00",
                "importance": 0.5,
                "tier": "short_term"
            }
        ]
        mock_domain_manager.persistence_domain.list_memories.return_value = mock_memories
        
        results = await mock_domain_manager.list_memories(
            limit=10,
            offset=0,
            include_content=True
        )
        
        assert len(results) == 2
        assert results[0]["id"] == "mem_1"
        assert "content" in results[0]
        assert results[1]["id"] == "mem_2"
        assert "content" in results[1]
        
        mock_domain_manager.persistence_domain.list_memories.assert_called_once_with(
            types=None,
            limit=10,
            offset=0,
            tier=None
        )
    
    @pytest.mark.asyncio
    async def test_list_memories_filtered(self, mock_domain_manager: MemoryDomainManager):
        """Test memory listing with filters."""
        mock_memories = []
        mock_domain_manager.persistence_domain.list_memories.return_value = mock_memories
        
        results = await mock_domain_manager.list_memories(
            memory_types=["fact", "document"],
            limit=5,
            offset=10,
            tier="long_term",
            include_content=False
        )
        
        assert len(results) == 0
        
        mock_domain_manager.persistence_domain.list_memories.assert_called_once_with(
            types=["fact", "document"],
            limit=5,
            offset=10,
            tier="long_term"
        )
    
    @pytest.mark.asyncio
    async def test_list_memories_without_content(self, mock_domain_manager: MemoryDomainManager):
        """Test memory listing without content."""
        mock_memories = [
            {
                "id": "mem_1",
                "type": "fact",
                "content": {"fact": "Should not be included"},
                "created_at": "2023-01-01T00:00:00",
                "importance": 0.7
            }
        ]
        mock_domain_manager.persistence_domain.list_memories.return_value = mock_memories
        
        results = await mock_domain_manager.list_memories(include_content=False)
        
        assert len(results) == 1
        assert "content" not in results[0]
        assert results[0]["id"] == "mem_1"
        assert results[0]["type"] == "fact"


@pytest.mark.unit
class TestMemoryStats:
    """Test memory statistics operations."""
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, mock_domain_manager: MemoryDomainManager):
        """Test retrieving memory statistics."""
        mock_persistence_stats = {
            "total_memories": 100,
            "total_size": "10MB",
            "qdrant_stats": {"indexed_count": 95}
        }
        mock_episodic_stats = {"conversation_count": 25}
        mock_semantic_stats = {"fact_count": 50}
        mock_temporal_stats = {"last_consolidation": "2023-01-01T00:00:00"}
        mock_autocode_stats = {"pattern_count": 25}
        
        mock_domain_manager.persistence_domain.get_memory_stats.return_value = mock_persistence_stats
        mock_domain_manager.episodic_domain.get_stats.return_value = mock_episodic_stats
        mock_domain_manager.semantic_domain.get_stats.return_value = mock_semantic_stats
        mock_domain_manager.temporal_domain.get_stats.return_value = mock_temporal_stats
        mock_domain_manager.autocode_domain.get_stats.return_value = mock_autocode_stats
        
        stats = await mock_domain_manager.get_memory_stats()
        
        assert stats["total_memories"] == 100
        assert stats["episodic_domain"]["conversation_count"] == 25
        assert stats["semantic_domain"]["fact_count"] == 50
        assert stats["temporal_domain"]["last_consolidation"] == "2023-01-01T00:00:00"
        assert stats["autocode_domain"]["pattern_count"] == 25


@pytest.mark.unit
class TestAutoCodeMemoryMethods:
    """Test AutoCode-specific memory methods."""
    
    @pytest.mark.asyncio
    async def test_store_project_pattern(self, mock_domain_manager: MemoryDomainManager):
        """Test storing project pattern."""
        pattern_type = "framework"
        framework = "FastAPI"
        language = "python"
        structure = {"files": ["main.py", "requirements.txt"]}
        
        memory_id = await mock_domain_manager.store_project_pattern(
            pattern_type=pattern_type,
            framework=framework,
            language=language,
            structure=structure,
            importance=0.8,
            metadata={"project": "test_api"}
        )
        
        assert memory_id.startswith("mem_")
        
        # Verify store_memory was called with correct parameters
        call_args = mock_domain_manager.store_memory.call_args if hasattr(mock_domain_manager, 'store_memory') else None
        # Since we're mocking the domain manager, we need to verify the actual method call differently
        mock_domain_manager.persistence_domain.store_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_command_pattern(self, mock_domain_manager: MemoryDomainManager):
        """Test storing command pattern."""
        command = "pytest -v"
        context = {"framework": "pytest", "project_type": "python"}
        success_rate = 0.95
        platform = "linux"
        
        memory_id = await mock_domain_manager.store_command_pattern(
            command=command,
            context=context,
            success_rate=success_rate,
            platform=platform,
            importance=0.7
        )
        
        assert memory_id.startswith("mem_")
        mock_domain_manager.persistence_domain.store_memory.assert_called()
    
    @pytest.mark.asyncio
    async def test_store_session_summary(self, mock_domain_manager: MemoryDomainManager):
        """Test storing session summary."""
        session_id = "session_123"
        tasks_completed = [
            {"task": "Created API endpoint", "status": "completed"},
            {"task": "Added tests", "status": "completed"}
        ]
        patterns_used = ["FastAPI", "pytest"]
        files_modified = ["main.py", "test_main.py"]
        
        memory_id = await mock_domain_manager.store_session_summary(
            session_id=session_id,
            tasks_completed=tasks_completed,
            patterns_used=patterns_used,
            files_modified=files_modified,
            importance=0.9
        )
        
        assert memory_id.startswith("mem_")
        mock_domain_manager.persistence_domain.store_memory.assert_called()
    
    @pytest.mark.asyncio
    async def test_store_bash_execution(self, mock_domain_manager: MemoryDomainManager):
        """Test storing bash execution record."""
        command = "pip install fastapi"
        exit_code = 0
        output = "Successfully installed fastapi-0.100.0"
        context = {"timestamp": "2023-01-01T12:00:00", "directory": "/project"}
        
        memory_id = await mock_domain_manager.store_bash_execution(
            command=command,
            exit_code=exit_code,
            output=output,
            context=context,
            importance=0.4
        )
        
        assert memory_id.startswith("mem_")
        mock_domain_manager.persistence_domain.store_memory.assert_called()


@pytest.mark.unit
class TestMemoryValidation:
    """Test memory validation utilities."""
    
    def test_validate_conversation_memory_with_role_message(self):
        """Test validating conversation memory with role/message format."""
        memory = {
            "id": "mem_test1",
            "type": "conversation",
            "importance": 0.8,
            "content": {
                "role": "user",
                "message": "Hello, Claude!"
            }
        }
        
        validated = validate_memory(memory)
        assert validated["id"] == "mem_test1"
        assert validated["type"] == "conversation"
        assert validated["content"]["role"] == "user"
        assert validated["content"]["message"] == "Hello, Claude!"
    
    def test_validate_conversation_memory_with_messages_array(self):
        """Test validating conversation memory with messages array format."""
        memory = {
            "id": "mem_test2",
            "type": "conversation",
            "importance": 0.7,
            "content": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        }
        
        validated = validate_memory(memory)
        assert validated["id"] == "mem_test2"
        assert validated["type"] == "conversation"
        assert len(validated["content"]["messages"]) == 2
    
    def test_validate_fact_memory(self):
        """Test validating fact memory."""
        memory = {
            "id": "mem_test3",
            "type": "fact",
            "importance": 0.9,
            "content": {
                "fact": "The capital of France is Paris.",
                "confidence": 0.95
            }
        }
        
        validated = validate_memory(memory)
        assert validated["id"] == "mem_test3"
        assert validated["type"] == "fact"
        assert validated["content"]["fact"] == "The capital of France is Paris."
        assert validated["content"]["confidence"] == 0.95
    
    def test_validate_invalid_conversation_memory(self):
        """Test validation fails for invalid conversation memory."""
        memory = {
            "id": "mem_test4",
            "type": "conversation",
            "importance": 0.5,
            "content": {}  # Missing required fields
        }
        
        with pytest.raises(ValueError):
            validate_memory(memory)
    
    def test_validate_invalid_fact_memory(self):
        """Test validation fails for invalid fact memory."""
        memory = {
            "id": "mem_test5",
            "type": "fact",
            "importance": 0.7,
            "content": {
                "confidence": 0.8  # Missing fact field
            }
        }
        
        with pytest.raises(ValueError):
            validate_memory(memory)