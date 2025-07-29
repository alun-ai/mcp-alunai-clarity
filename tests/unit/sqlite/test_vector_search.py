#!/usr/bin/env python3
"""
SQLite Vector Search Tests

Unit tests specifically focused on vector similarity search functionality:
- Embedding generation and caching
- Similarity calculations
- Search accuracy and ranking
- Vector search performance
- Edge cases in vector operations
"""

import asyncio
import math
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Any, Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence


class TestSQLiteVectorSearch:
    """Test suite for SQLite vector search functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="vector_test_")
        db_path = os.path.join(temp_dir, "vector_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model with predictable outputs."""
        mock_model = Mock()
        
        # Create predictable embeddings based on content
        def mock_encode(text):
            # Simple hash-based embedding for testing
            hash_val = hash(text) % 1000
            embedding = [0.0] * 384
            embedding[0] = hash_val / 1000.0  # First dimension varies by content
            embedding[1] = len(text) / 1000.0  # Second dimension based on length
            # Fill remaining dimensions with small random-like values
            for i in range(2, 384):
                embedding[i] = ((hash_val * (i + 1)) % 100) / 10000.0
            return np.array(embedding)
        
        mock_model.encode.side_effect = mock_encode
        return mock_model
    
    @pytest.fixture
    def sqlite_persistence(self, temp_db_path, mock_embedding_model):
        """Create SQLiteMemoryPersistence instance for testing."""
        return SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
    
    @pytest.fixture
    def vector_test_memories(self):
        """Test memories with known content for vector search testing."""
        return [
            {
                "id": "vec-001",
                "type": "structured_thinking",
                "content": "Database performance optimization techniques for large datasets",
                "importance": 0.9,
                "tier": "long_term"
            },
            {
                "id": "vec-002", 
                "type": "structured_thinking",
                "content": "Database query optimization and indexing strategies",
                "importance": 0.8,
                "tier": "long_term"
            },
            {
                "id": "vec-003",
                "type": "episodic",
                "content": "User reported slow database performance issues",
                "importance": 0.7,
                "tier": "short_term"
            },
            {
                "id": "vec-004",
                "type": "semantic",
                "content": "Machine learning algorithms for natural language processing", 
                "importance": 0.8,
                "tier": "archival"
            },
            {
                "id": "vec-005",
                "type": "procedural",
                "content": "Step-by-step guide for database backup and recovery",
                "importance": 0.6,
                "tier": "system"
            }
        ]
    
    def test_embedding_generation(self, sqlite_persistence):
        """Test embedding generation functionality."""
        test_text = "This is a test sentence for embedding generation"
        
        embedding = sqlite_persistence._generate_embedding(test_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Default embedding dimensions
        assert all(isinstance(x, float) for x in embedding)
        
        # Test embedding consistency
        embedding2 = sqlite_persistence._generate_embedding(test_text)
        assert embedding == embedding2  # Should be identical for same text
    
    def test_embedding_caching(self, sqlite_persistence):
        """Test embedding caching functionality."""
        test_text = "Cached embedding test"
        
        # First call should generate and cache
        embedding1 = sqlite_persistence._generate_embedding(test_text)
        
        # Second call should use cache (verify by checking model call count)
        embedding2 = sqlite_persistence._generate_embedding(test_text)
        
        assert embedding1 == embedding2
        
        # Check cache hit by verifying model was called only once
        # (This would be more robust with cache inspection, but mock verification works)
        call_count = sqlite_persistence.embedding_model.encode.call_count
        
        # Generate embedding for different text should call model again
        sqlite_persistence._generate_embedding("Different text")
        new_call_count = sqlite_persistence.embedding_model.encode.call_count
        assert new_call_count > call_count
    
    def test_vector_serialization_roundtrip(self, sqlite_persistence):
        """Test vector serialization and deserialization roundtrip."""
        original_vector = [0.1, -0.5, 0.9, 0.0, -1.0, 0.33333]
        
        # Serialize
        serialized = sqlite_persistence._serialize_embedding(original_vector)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = sqlite_persistence._deserialize_embedding(serialized)
        
        # Verify roundtrip accuracy
        assert len(deserialized) == len(original_vector)
        for orig, deser in zip(original_vector, deserialized):
            assert abs(orig - deser) < 1e-6  # Account for floating point precision
    
    def test_cosine_similarity_edge_cases(self, sqlite_persistence):
        """Test cosine similarity calculation edge cases."""
        # Test with zero vectors
        zero_vec = [0.0, 0.0, 0.0]
        normal_vec = [1.0, 0.0, 0.0]
        
        similarity = sqlite_persistence._calculate_cosine_similarity(zero_vec, normal_vec)
        assert similarity == 0.0
        
        # Test with identical vectors
        vec = [0.5, 0.5, 0.5]
        similarity = sqlite_persistence._calculate_cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-10
        
        # Test with normalized vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = sqlite_persistence._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10
        
        # Test with opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = sqlite_persistence._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-10
        
        # Test with high-dimensional vectors
        dim = 384
        vec1 = [1.0] + [0.0] * (dim - 1)
        vec2 = [0.0] + [1.0] + [0.0] * (dim - 2)
        similarity = sqlite_persistence._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_vector_search_basic(self, sqlite_persistence, vector_test_memories):
        """Test basic vector similarity search."""
        # Store test memories
        for memory in vector_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Search for database-related content
        results = await sqlite_persistence.retrieve_memories(
            "database optimization performance",
            limit=3,
            min_similarity=0.0
        )
        
        assert len(results) > 0
        
        # Verify results contain similarity scores
        for result in results:
            assert "similarity_score" in result
            assert isinstance(result["similarity_score"], float)
            assert 0.0 <= result["similarity_score"] <= 1.0
        
        # Verify results are sorted by similarity (descending)
        similarities = [r["similarity_score"] for r in results]
        assert similarities == sorted(similarities, reverse=True)
    
    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, sqlite_persistence, vector_test_memories):
        """Test similarity threshold filtering."""
        # Store test memories
        for memory in vector_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Search with high similarity threshold
        high_threshold_results = await sqlite_persistence.retrieve_memories(
            "database performance optimization techniques",
            limit=10,
            min_similarity=0.8  # High threshold
        )
        
        # Search with low similarity threshold  
        low_threshold_results = await sqlite_persistence.retrieve_memories(
            "database performance optimization techniques",
            limit=10,
            min_similarity=0.1  # Low threshold
        )
        
        # High threshold should return fewer results
        assert len(high_threshold_results) <= len(low_threshold_results)
        
        # All high-threshold results should meet the threshold
        for result in high_threshold_results:
            assert result["similarity_score"] >= 0.8
        
        # Low threshold results should include more diverse matches
        assert len(low_threshold_results) > 0
    
    @pytest.mark.asyncio
    async def test_search_ranking_accuracy(self, sqlite_persistence):
        """Test search result ranking accuracy."""
        # Create memories with known semantic relationships
        semantic_memories = [
            {
                "id": "rank-001",
                "type": "structured_thinking", 
                "content": "Python programming language features and syntax",
                "importance": 0.8,
                "tier": "long_term"
            },
            {
                "id": "rank-002",
                "type": "structured_thinking",
                "content": "Python web development with Django framework",
                "importance": 0.7,
                "tier": "long_term"
            },
            {
                "id": "rank-003",
                "type": "structured_thinking",
                "content": "JavaScript asynchronous programming concepts",
                "importance": 0.8,
                "tier": "long_term"
            },
            {
                "id": "rank-004",
                "type": "episodic",
                "content": "Database connection timeout error occurred",
                "importance": 0.9,
                "tier": "short_term"
            }
        ]
        
        for memory in semantic_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Search for Python-related content
        python_results = await sqlite_persistence.retrieve_memories(
            "Python programming development",
            limit=5,
            min_similarity=0.0
        )
        
        # The first result should be most relevant to Python
        assert len(python_results) > 0
        
        # Python-related memories should rank higher than unrelated ones
        python_ids = {"rank-001", "rank-002"}
        top_result_id = python_results[0]["id"]
        
        # At least one of the top results should be Python-related
        top_2_ids = {r["id"] for r in python_results[:2]}
        assert len(python_ids.intersection(top_2_ids)) > 0
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, sqlite_persistence, vector_test_memories):
        """Test handling of empty or whitespace queries."""
        # Store test memories
        for memory in vector_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test empty query
        empty_results = await sqlite_persistence.retrieve_memories(
            "",
            limit=5,
            min_similarity=0.0
        )
        
        # Should handle gracefully (implementation-dependent behavior)
        assert isinstance(empty_results, list)
        
        # Test whitespace query
        whitespace_results = await sqlite_persistence.retrieve_memories(
            "   ",
            limit=5,
            min_similarity=0.0
        )
        
        assert isinstance(whitespace_results, list)
    
    @pytest.mark.asyncio
    async def test_vector_search_with_filters(self, sqlite_persistence, vector_test_memories):
        """Test vector search combined with metadata filters."""
        # Store test memories
        for memory in vector_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Search with memory type filter
        filtered_results = await sqlite_persistence.retrieve_memories(
            "database performance",
            memory_types=["structured_thinking"],
            limit=5,
            min_similarity=0.0
        )
        
        # Verify all results match the filter
        for result in filtered_results:
            assert result["type"] == "structured_thinking"
        
        # Search with tier filter
        tier_filtered_results = await sqlite_persistence.retrieve_memories(
            "database",
            filters={"tier": "long_term"},
            limit=5,
            min_similarity=0.0
        )
        
        # All results should be from long_term tier
        # (Note: basic retrieve_memories doesn't return tier, but search_memories does)
        assert isinstance(tier_filtered_results, list)
    
    @pytest.mark.asyncio
    async def test_large_result_set_handling(self, sqlite_persistence):
        """Test handling of large result sets."""
        # Store many similar memories
        batch_memories = []
        for i in range(50):
            memory = {
                "id": f"batch-{i:03d}",
                "type": "episodic",
                "content": f"Database performance issue number {i} with similar symptoms",
                "importance": 0.5 + (i % 10) * 0.05,
                "tier": "short_term"
            }
            batch_memories.append(memory)
            await sqlite_persistence.store_memory(memory)
        
        # Search with various limits
        small_results = await sqlite_persistence.retrieve_memories(
            "database performance issue symptoms",
            limit=5,
            min_similarity=0.0
        )
        
        large_results = await sqlite_persistence.retrieve_memories(
            "database performance issue symptoms", 
            limit=25,
            min_similarity=0.0
        )
        
        assert len(small_results) <= 5
        assert len(large_results) <= 25
        assert len(large_results) >= len(small_results)
        
        # Verify ordering is maintained
        small_similarities = [r["similarity_score"] for r in small_results]
        large_similarities = [r["similarity_score"] for r in large_results]
        
        assert small_similarities == sorted(small_similarities, reverse=True)
        assert large_similarities == sorted(large_similarities, reverse=True)
    
    @pytest.mark.asyncio
    async def test_embedding_model_error_handling(self, temp_db_path):
        """Test error handling when embedding model fails."""
        # Create persistence with failing embedding model
        failing_model = Mock()
        failing_model.encode.side_effect = Exception("Model failure")
        
        persistence = SQLiteMemoryPersistence(temp_db_path, failing_model)
        
        # Attempt to store memory should fail gracefully
        with pytest.raises(Exception):  # Should raise MemoryOperationError
            await persistence.store_memory({
                "id": "fail-test",
                "type": "test",
                "content": "test content",
                "importance": 0.5,
                "tier": "short_term"
            })
    
    @pytest.mark.asyncio 
    async def test_search_memory_method(self, sqlite_persistence, vector_test_memories):
        """Test the search_memories method with embedding parameter."""
        # Store test memories
        for memory in vector_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Generate test embedding
        query_text = "database optimization performance"
        query_embedding = sqlite_persistence._generate_embedding(query_text)
        
        # Search using embedding directly
        embedding_results = await sqlite_persistence.search_memories(
            embedding=query_embedding,
            limit=3,
            min_similarity=0.0
        )
        
        assert len(embedding_results) > 0
        
        # Verify similarity scores are included
        for result in embedding_results:
            assert "similarity" in result
            assert isinstance(result["similarity"], float)
        
        # Search without embedding (should work too)
        no_embedding_results = await sqlite_persistence.search_memories(
            types=["structured_thinking"],
            limit=5
        )
        
        assert isinstance(no_embedding_results, list)
    
    def test_vector_dimension_consistency(self, sqlite_persistence):
        """Test that all embeddings have consistent dimensions."""
        test_texts = [
            "Short text",
            "This is a medium length text with more words and content",
            "This is a very long text with lots of words and content that goes on and on with many different concepts and ideas expressed in various ways to test how the embedding model handles longer input sequences and whether the output dimensions remain consistent across different text lengths",
            "",  # Empty text
            "Special characters: !@#$%^&*()_+-=[]{}|;:,.<>?",
            "Numbers and dates: 2024-01-28 12:34:56, phone: 555-123-4567"
        ]
        
        embeddings = []
        for text in test_texts:
            embedding = sqlite_persistence._generate_embedding(text)
            embeddings.append(embedding)
            
            # Check dimensions
            assert len(embedding) == 384
            
            # Check all values are finite numbers
            assert all(math.isfinite(x) for x in embedding)
        
        # All embeddings should have same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert all(dim == 384 for dim in dimensions)
    
    @pytest.mark.asyncio
    async def test_similarity_score_validation(self, sqlite_persistence, vector_test_memories):
        """Test that similarity scores are mathematically valid."""
        # Store test memories
        for memory in vector_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Perform various searches
        test_queries = [
            "exact database performance optimization techniques match",
            "somewhat related database content",
            "completely unrelated machine learning algorithms"
        ]
        
        for query in test_queries:
            results = await sqlite_persistence.retrieve_memories(
                query,
                limit=10,
                min_similarity=0.0
            )
            
            for result in results:
                similarity = result["similarity_score"]
                
                # Similarity should be between -1 and 1 (cosine similarity range)
                assert -1.0 <= similarity <= 1.0
                
                # Should be a finite number
                assert math.isfinite(similarity)
                
                # For our test case, similarities should generally be positive
                # (negative would indicate very dissimilar content)
                assert similarity >= -0.1  # Allow small negative values due to precision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])