"""
Unit tests for domain managers in Alunai Clarity.
"""

import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import numpy as np

from clarity.domains.persistence import PersistenceDomain
from clarity.domains.episodic import EpisodicDomain
from clarity.domains.semantic import SemanticDomain
from clarity.domains.temporal import TemporalDomain
from clarity.utils.embeddings import EmbeddingManager


@pytest.mark.unit
class TestPersistenceDomain:
    """Test persistence domain functionality."""
    
    @pytest.mark.asyncio
    async def test_persistence_domain_initialization(self, test_config: Dict[str, Any]):
        """Test persistence domain initialization."""
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant:
            mock_client = AsyncMock()
            mock_qdrant.return_value = mock_client
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            assert domain.qdrant_client is not None
            assert domain.embedding_manager is not None
            mock_client.create_collection.assert_called()
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, test_config: Dict[str, Any]):
        """Test embedding generation."""
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant:
            domain = PersistenceDomain(test_config)
            
            # Mock embedding manager
            domain.embedding_manager = MagicMock()
            test_embedding = np.random.rand(384)
            domain.embedding_manager.get_embedding.return_value = test_embedding
            
            result = await domain.generate_embedding("test text")
            
            assert np.array_equal(result, test_embedding)
            domain.embedding_manager.get_embedding.assert_called_once_with("test text")
    
    @pytest.mark.asyncio
    async def test_store_memory_with_embedding(self, test_config: Dict[str, Any]):
        """Test storing memory with embedding generation."""
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant:
            mock_client = AsyncMock()
            mock_qdrant.return_value = mock_client
            
            domain = PersistenceDomain(test_config)
            domain.embedding_manager = MagicMock()
            domain.embedding_manager.get_embedding.return_value = np.random.rand(384)
            await domain.initialize()
            
            memory = {
                "id": "mem_test",
                "type": "fact",
                "content": {"fact": "Test fact"},
                "importance": 0.7
            }
            
            await domain.store_memory(memory, "short_term")
            
            # Verify Qdrant upsert was called
            mock_client.upsert.assert_called_once()
            
            # Verify embedding was generated
            domain.embedding_manager.get_embedding.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_memories(self, test_config: Dict[str, Any]):
        """Test memory search functionality."""
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant:
            mock_client = AsyncMock()
            mock_qdrant.return_value = mock_client
            
            # Mock search results
            mock_search_result = MagicMock()
            mock_search_result.id = "mem_test"
            mock_search_result.score = 0.85
            mock_search_result.payload = {
                "id": "mem_test",
                "type": "fact",
                "content": {"fact": "Test fact"},
                "importance": 0.7
            }
            mock_client.search.return_value = [mock_search_result]
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            query_embedding = np.random.rand(384)
            results = await domain.search_memories(
                embedding=query_embedding,
                limit=5,
                min_similarity=0.6
            )
            
            assert len(results) == 1
            assert results[0]["id"] == "mem_test"
            assert results[0]["similarity"] == 0.85
            mock_client.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_memory(self, test_config: Dict[str, Any]):
        """Test retrieving specific memory."""
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant:
            mock_client = AsyncMock()
            mock_qdrant.return_value = mock_client
            
            # Mock retrieve result
            mock_result = MagicMock()
            mock_result.payload = {
                "id": "mem_test",
                "type": "fact",
                "content": {"fact": "Test fact"}
            }
            mock_client.retrieve.return_value = [mock_result]
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            memory = await domain.get_memory("mem_test")
            
            assert memory["id"] == "mem_test"
            assert memory["type"] == "fact"
            mock_client.retrieve.assert_called_once_with(
                collection_name="alunai_clarity_memories",
                ids=["mem_test"]
            )
    
    @pytest.mark.asyncio
    async def test_delete_memories(self, test_config: Dict[str, Any]):
        """Test deleting memories."""
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant:
            mock_client = AsyncMock()
            mock_qdrant.return_value = mock_client
            mock_client.delete.return_value = True
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            result = await domain.delete_memories(["mem_1", "mem_2"])
            
            assert result is True
            mock_client.delete.assert_called_once_with(
                collection_name="alunai_clarity_memories",
                points_selector={"ids": ["mem_1", "mem_2"]}
            )


@pytest.mark.unit
class TestEpisodicDomain:
    """Test episodic domain functionality."""
    
    @pytest.mark.asyncio
    async def test_episodic_domain_initialization(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test episodic domain initialization."""
        domain = EpisodicDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        # Should initialize without errors
        assert domain.config == test_config
        assert domain.persistence_domain == mock_persistence_domain
    
    @pytest.mark.asyncio
    async def test_process_conversation_memory(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test processing conversation memory."""
        domain = EpisodicDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        memory = {
            "id": "mem_conv",
            "type": "conversation",
            "content": {
                "role": "user",
                "message": "Hello, how are you?"
            },
            "importance": 0.7
        }
        
        processed = await domain.process_memory(memory)
        
        # Should return memory with episodic processing
        assert processed["id"] == "mem_conv"
        assert processed["type"] == "conversation"
        # Additional episodic metadata should be added
        assert "episodic_metadata" in processed
    
    @pytest.mark.asyncio
    async def test_get_episodic_stats(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test getting episodic domain statistics."""
        domain = EpisodicDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        stats = await domain.get_stats()
        
        assert isinstance(stats, dict)
        assert "conversation_memories" in stats
        assert "reflection_memories" in stats
        assert "total_episodic_memories" in stats


@pytest.mark.unit
class TestSemanticDomain:
    """Test semantic domain functionality."""
    
    @pytest.mark.asyncio
    async def test_semantic_domain_initialization(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test semantic domain initialization."""
        domain = SemanticDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        assert domain.config == test_config
        assert domain.persistence_domain == mock_persistence_domain
    
    @pytest.mark.asyncio
    async def test_process_fact_memory(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test processing fact memory."""
        domain = SemanticDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        memory = {
            "id": "mem_fact",
            "type": "fact",
            "content": {
                "fact": "Python is a programming language",
                "confidence": 0.95
            },
            "importance": 0.8
        }
        
        processed = await domain.process_memory(memory)
        
        assert processed["id"] == "mem_fact"
        assert processed["type"] == "fact"
        # Semantic processing metadata should be added
        assert "semantic_metadata" in processed
    
    @pytest.mark.asyncio
    async def test_process_document_memory(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test processing document memory."""
        domain = SemanticDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        memory = {
            "id": "mem_doc",
            "type": "document",
            "content": {
                "title": "API Documentation",
                "content": "This document describes the API endpoints...",
                "format": "markdown"
            },
            "importance": 0.9
        }
        
        processed = await domain.process_memory(memory)
        
        assert processed["id"] == "mem_doc"
        assert processed["type"] == "document"
        assert "semantic_metadata" in processed
    
    @pytest.mark.asyncio
    async def test_get_semantic_stats(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test getting semantic domain statistics."""
        domain = SemanticDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        stats = await domain.get_stats()
        
        assert isinstance(stats, dict)
        assert "fact_memories" in stats
        assert "document_memories" in stats
        assert "entity_memories" in stats
        assert "total_semantic_memories" in stats


@pytest.mark.unit
class TestTemporalDomain:
    """Test temporal domain functionality."""
    
    @pytest.mark.asyncio
    async def test_temporal_domain_initialization(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test temporal domain initialization."""
        domain = TemporalDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        assert domain.config == test_config
        assert domain.persistence_domain == mock_persistence_domain
        assert isinstance(domain.last_consolidation, datetime)
        assert isinstance(domain.consolidation_interval, timedelta)
    
    @pytest.mark.asyncio
    async def test_process_new_memory(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test processing new memory with temporal information."""
        domain = TemporalDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        memory = {
            "id": "mem_new",
            "type": "fact",
            "content": {"fact": "Test fact"},
            "importance": 0.7
        }
        
        processed = await domain.process_new_memory(memory)
        
        assert "created_at" in processed
        assert "last_accessed" in processed
        assert "last_modified" in processed
        assert "access_count" in processed
        assert processed["access_count"] == 0
        
        # Verify timestamps are valid ISO format
        datetime.fromisoformat(processed["created_at"])
        datetime.fromisoformat(processed["last_accessed"])
        datetime.fromisoformat(processed["last_modified"])
    
    @pytest.mark.asyncio
    async def test_update_memory_access(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test updating memory access time."""
        domain = TemporalDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        # Mock existing memory
        existing_memory = {
            "id": "mem_test",
            "type": "fact",
            "content": {"fact": "Test"},
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 5
        }
        mock_persistence_domain.get_memory.return_value = existing_memory
        mock_persistence_domain.get_memory_tier.return_value = "short_term"
        
        await domain.update_memory_access("mem_test")
        
        # Verify get_memory was called
        mock_persistence_domain.get_memory.assert_called_once_with("mem_test")
        # Verify update_memory was called
        mock_persistence_domain.update_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adjust_memory_relevance(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test adjusting memory relevance with temporal factors."""
        domain = TemporalDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        memories = [
            {
                "id": "mem_1",
                "type": "fact",
                "content": {"fact": "Recent fact"},
                "similarity": 0.8,
                "importance": 0.7,
                "last_accessed": datetime.now().isoformat()
            },
            {
                "id": "mem_2",
                "type": "fact", 
                "content": {"fact": "Old fact"},
                "similarity": 0.8,
                "importance": 0.7,
                "last_accessed": (datetime.now() - timedelta(days=30)).isoformat()
            }
        ]
        
        adjusted = await domain.adjust_memory_relevance(memories, "test query")
        
        assert len(adjusted) == 2
        # Recent memory should have higher adjusted score
        assert adjusted[0]["adjusted_score"] > adjusted[1]["adjusted_score"]
        assert "recency_score" in adjusted[0]
        assert "recency_score" in adjusted[1]
    
    @pytest.mark.asyncio
    async def test_update_memory_modification(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test updating memory modification time."""
        domain = TemporalDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        memory = {
            "id": "mem_test",
            "type": "fact",
            "content": {"fact": "Test fact"}
        }
        
        updated = await domain.update_memory_modification(memory)
        
        assert "last_modified" in updated
        # Verify timestamp is recent
        modification_time = datetime.fromisoformat(updated["last_modified"])
        assert (datetime.now() - modification_time).total_seconds() < 5
    
    @pytest.mark.asyncio
    async def test_get_temporal_stats(self, test_config: Dict[str, Any], mock_persistence_domain):
        """Test getting temporal domain statistics."""
        domain = TemporalDomain(test_config, mock_persistence_domain)
        await domain.initialize()
        
        stats = await domain.get_stats()
        
        assert isinstance(stats, dict)
        assert "last_consolidation" in stats
        assert "next_consolidation" in stats
        assert "status" in stats
        
        # Verify timestamps are valid
        datetime.fromisoformat(stats["last_consolidation"])
        datetime.fromisoformat(stats["next_consolidation"])


@pytest.mark.unit
class TestEmbeddingManager:
    """Test embedding manager functionality."""
    
    def test_embedding_manager_initialization(self, test_config: Dict[str, Any]):
        """Test embedding manager initialization."""
        manager = EmbeddingManager(test_config)
        
        assert manager.model_name == test_config["embedding"]["default_model"]
        assert manager.dimensions == test_config["embedding"]["dimensions"]
        assert manager.cache_dir == test_config["embedding"]["cache_dir"]
        assert manager.model is None  # Model not loaded yet
    
    def test_calculate_similarity_numpy_arrays(self, mock_embedding_manager):
        """Test similarity calculation with numpy arrays."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([1.0, 0.0, 0.0])
        
        # Orthogonal vectors should have low similarity
        similarity_orthogonal = mock_embedding_manager.calculate_similarity(v1, v2)
        assert similarity_orthogonal == pytest.approx(0.0, abs=1e-6)
        
        # Identical vectors should have high similarity
        similarity_identical = mock_embedding_manager.calculate_similarity(v1, v3)
        assert similarity_identical == pytest.approx(1.0, abs=1e-6)
    
    def test_calculate_similarity_lists(self, mock_embedding_manager):
        """Test similarity calculation with lists."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [1.0, 0.0, 0.0]
        
        # Orthogonal vectors
        similarity_orthogonal = mock_embedding_manager.calculate_similarity(v1, v2)
        assert similarity_orthogonal == pytest.approx(0.0, abs=1e-6)
        
        # Identical vectors
        similarity_identical = mock_embedding_manager.calculate_similarity(v1, v3)
        assert similarity_identical == pytest.approx(1.0, abs=1e-6)
    
    def test_calculate_similarity_mixed_types(self, mock_embedding_manager):
        """Test similarity calculation with mixed types."""
        v1_array = np.array([1.0, 1.0, 0.0])
        v2_list = [1.0, 1.0, 0.0]
        
        similarity = mock_embedding_manager.calculate_similarity(v1_array, v2_list)
        assert similarity == pytest.approx(1.0, abs=1e-6)
    
    @patch('clarity.utils.embeddings.SentenceTransformer')
    def test_get_embedding_loads_model(self, mock_sentence_transformer, test_config):
        """Test that get_embedding loads the model when needed."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_sentence_transformer.return_value = mock_model
        
        manager = EmbeddingManager(test_config)
        
        # Model should be None initially
        assert manager.model is None
        
        # Get embedding should load model
        embedding = manager.get_embedding("test text")
        
        # Model should now be loaded
        assert manager.model is not None
        mock_sentence_transformer.assert_called_once_with(manager.model_name)
        mock_model.encode.assert_called_once_with("test text")
        assert isinstance(embedding, np.ndarray)
    
    def test_normalize_vector(self, mock_embedding_manager):
        """Test vector normalization."""
        vector = np.array([3.0, 4.0, 0.0])
        normalized = mock_embedding_manager._normalize_vector(vector)
        
        # Check that magnitude is 1
        magnitude = np.linalg.norm(normalized)
        assert magnitude == pytest.approx(1.0, abs=1e-6)
        
        # Check direction is preserved
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_normalize_zero_vector(self, mock_embedding_manager):
        """Test normalization of zero vector."""
        zero_vector = np.array([0.0, 0.0, 0.0])
        normalized = mock_embedding_manager._normalize_vector(zero_vector)
        
        # Zero vector should remain zero
        np.testing.assert_array_equal(normalized, zero_vector)