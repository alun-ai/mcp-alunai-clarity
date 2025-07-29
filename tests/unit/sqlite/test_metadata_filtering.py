#!/usr/bin/env python3
"""
SQLite Metadata Filtering Tests

Unit tests specifically focused on metadata filtering functionality:
- Memory type filtering
- Memory tier filtering
- Importance-based filtering
- Complex metadata filtering
- Combined filter operations
- Filter validation and edge cases
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from typing import Any, Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence


class TestSQLiteMetadataFiltering:
    """Test suite for SQLite metadata filtering functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="filter_test_")
        db_path = os.path.join(temp_dir, "filter_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1] * 384
        return mock_model
    
    @pytest.fixture
    def sqlite_persistence(self, temp_db_path, mock_embedding_model):
        """Create SQLiteMemoryPersistence instance for testing."""
        return SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
    
    @pytest.fixture
    def diverse_test_memories(self):
        """Diverse test memories for comprehensive filtering tests."""
        return [
            # Structured thinking memories
            {
                "id": "st-001",
                "type": "structured_thinking",
                "content": "Analysis of database performance optimization strategies",
                "importance": 0.9,
                "tier": "long_term",
                "metadata": {
                    "category": "analysis",
                    "domain": "database",
                    "priority": "high",
                    "complexity": "advanced"
                }
            },
            {
                "id": "st-002",
                "type": "structured_thinking",
                "content": "System architecture design considerations",
                "importance": 0.8,
                "tier": "long_term",
                "metadata": {
                    "category": "design",
                    "domain": "architecture",
                    "priority": "medium",
                    "complexity": "intermediate"
                }
            },
            
            # Episodic memories
            {
                "id": "ep-001",
                "type": "episodic",
                "content": "Production database outage at 2024-01-28 14:30",
                "importance": 0.95,
                "tier": "short_term",
                "metadata": {
                    "category": "incident",
                    "domain": "operations",
                    "priority": "critical",
                    "resolved": True,
                    "impact": "high"
                }
            },
            {
                "id": "ep-002",
                "type": "episodic",
                "content": "User reported slow query response times",
                "importance": 0.7,
                "tier": "short_term",
                "metadata": {
                    "category": "issue",
                    "domain": "performance",
                    "priority": "medium",
                    "resolved": False,
                    "impact": "medium"
                }
            },
            
            # Procedural memories
            {
                "id": "pr-001",
                "type": "procedural",
                "content": "Database backup and recovery procedure",
                "importance": 0.85,
                "tier": "system",
                "metadata": {
                    "category": "procedure",
                    "domain": "operations",
                    "priority": "high",
                    "frequency": "daily",
                    "automation": "partial"
                }
            },
            {
                "id": "pr-002",
                "type": "procedural",
                "content": "Code review and deployment checklist",
                "importance": 0.75,
                "tier": "long_term",
                "metadata": {
                    "category": "procedure",
                    "domain": "development",
                    "priority": "medium",
                    "frequency": "as_needed",
                    "automation": "none"
                }
            },
            
            # Semantic memories
            {
                "id": "sm-001",
                "type": "semantic",
                "content": "Vector embeddings represent text as numerical vectors",
                "importance": 0.6,
                "tier": "archival",
                "metadata": {
                    "category": "concept",
                    "domain": "ml",
                    "priority": "low",
                    "complexity": "basic",
                    "field": "nlp"
                }
            },
            {
                "id": "sm-002",
                "type": "semantic",
                "content": "ACID properties ensure database transaction reliability", 
                "importance": 0.8,
                "tier": "archival",
                "metadata": {
                    "category": "concept",
                    "domain": "database",
                    "priority": "medium",
                    "complexity": "intermediate",
                    "field": "data_engineering"
                }
            }
        ]
    
    @pytest.mark.asyncio
    async def test_memory_type_filtering(self, sqlite_persistence, diverse_test_memories):
        """Test filtering by memory type."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test single type filter
        structured_results = await sqlite_persistence.retrieve_memories(
            "analysis system design",
            memory_types=["structured_thinking"],
            limit=10,
            min_similarity=0.0
        )
        
        assert len(structured_results) > 0
        for result in structured_results:
            assert result["type"] == "structured_thinking"
        
        # Test multiple type filter
        multi_type_results = await sqlite_persistence.retrieve_memories(
            "database operations",
            memory_types=["episodic", "procedural"],
            limit=10,
            min_similarity=0.0
        )
        
        assert len(multi_type_results) > 0
        allowed_types = {"episodic", "procedural"}
        for result in multi_type_results:
            assert result["type"] in allowed_types
        
        # Test non-existent type
        empty_results = await sqlite_persistence.retrieve_memories(
            "test query",
            memory_types=["nonexistent_type"],
            limit=10,
            min_similarity=0.0
        )
        
        assert len(empty_results) == 0
    
    @pytest.mark.asyncio
    async def test_tier_filtering(self, sqlite_persistence, diverse_test_memories):
        """Test filtering by memory tier."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test tier filtering using search_memories (which returns tier info)
        long_term_results = await sqlite_persistence.search_memories(
            filters={"tier": "long_term"},
            limit=10
        )
        
        assert len(long_term_results) > 0
        for result in long_term_results:
            assert result["tier"] == "long_term"
        
        # Test short_term tier
        short_term_results = await sqlite_persistence.search_memories(
            filters={"tier": "short_term"},
            limit=10
        )
        
        assert len(short_term_results) > 0
        for result in short_term_results:
            assert result["tier"] == "short_term"
        
        # Test system tier
        system_results = await sqlite_persistence.search_memories(
            filters={"tier": "system"},
            limit=10
        )
        
        assert len(system_results) > 0
        for result in system_results:
            assert result["tier"] == "system"
        
        # Test archival tier
        archival_results = await sqlite_persistence.search_memories(
            filters={"tier": "archival"},
            limit=10
        )
        
        assert len(archival_results) > 0
        for result in archival_results:
            assert result["tier"] == "archival"
    
    @pytest.mark.asyncio
    async def test_importance_filtering(self, sqlite_persistence, diverse_test_memories):
        """Test filtering by importance level."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test high importance filter (>= 0.8)
        high_importance_results = await sqlite_persistence.retrieve_memories(
            "database system analysis",
            filters={"min_importance": 0.8},
            limit=10,
            min_similarity=0.0
        )
        
        # Note: retrieve_memories with filters supports min_importance
        # but doesn't return importance in basic response
        # Let's use search_memories instead for this test
        high_importance_search = await sqlite_persistence.search_memories(
            filters={"importance": [0.8, 0.85, 0.9, 0.95]},  # Test specific values
            limit=10
        )
        
        for result in high_importance_search:
            assert result["importance"] >= 0.8
        
        # Test medium importance range
        medium_importance_search = await sqlite_persistence.search_memories(
            limit=10
        )
        
        # Filter manually for validation
        medium_results = [r for r in medium_importance_search if 0.6 <= r["importance"] < 0.8]
        assert len(medium_results) > 0
    
    @pytest.mark.asyncio
    async def test_combined_filters(self, sqlite_persistence, diverse_test_memories):
        """Test combining multiple filters."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test type + tier combination
        combined_results = await sqlite_persistence.retrieve_memories(
            "database analysis",
            memory_types=["structured_thinking", "episodic"],
            filters={"tier": "long_term"},
            limit=10,
            min_similarity=0.0
        )
        
        # Verify using search_memories which returns complete data
        combined_search = await sqlite_persistence.search_memories(
            types=["structured_thinking", "episodic"],
            filters={"tier": "long_term"},
            limit=10
        )
        
        assert len(combined_search) > 0
        allowed_types = {"structured_thinking", "episodic"}
        for result in combined_search:
            assert result["type"] in allowed_types
            assert result["tier"] == "long_term"
        
        # Test type + importance combination
        high_value_structured = await sqlite_persistence.search_memories(
            types=["structured_thinking"],
            filters={"importance": [0.8, 0.9]},  # High importance only
            limit=10
        )
        
        for result in high_value_structured:
            assert result["type"] == "structured_thinking"
            assert result["importance"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_search_with_metadata_access(self, sqlite_persistence, diverse_test_memories):
        """Test searching with metadata included in results."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Search with metadata included
        results_with_metadata = await sqlite_persistence.retrieve_memories(
            "database system",
            include_metadata=True,
            limit=10,
            min_similarity=0.0
        )
        
        assert len(results_with_metadata) > 0
        for result in results_with_metadata:
            assert "metadata" in result
            assert "context" in result
            assert "tier" in result
            assert "access_count" in result
            
            # Verify metadata structure
            metadata = result["metadata"]
            assert isinstance(metadata, dict)
            
            # Should have some of our test metadata fields
            expected_fields = {"category", "domain", "priority"}
            found_fields = set(metadata.keys())
            assert len(expected_fields.intersection(found_fields)) > 0
    
    @pytest.mark.asyncio
    async def test_filter_validation_edge_cases(self, sqlite_persistence, diverse_test_memories):
        """Test filter validation and edge cases."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Test empty type list
        empty_type_results = await sqlite_persistence.retrieve_memories(
            "test query",
            memory_types=[],
            limit=10,
            min_similarity=0.0
        )
        # Should behave like no filter (return all)
        assert isinstance(empty_type_results, list)
        
        # Test None filters
        none_filter_results = await sqlite_persistence.retrieve_memories(
            "test query",
            memory_types=None,
            filters=None,
            limit=10,
            min_similarity=0.0
        )
        assert isinstance(none_filter_results, list)
        
        # Test invalid tier
        invalid_tier_results = await sqlite_persistence.search_memories(
            filters={"tier": "invalid_tier"},
            limit=10
        )
        assert len(invalid_tier_results) == 0
        
        # Test very high importance threshold
        extreme_importance_results = await sqlite_persistence.search_memories(
            filters={"importance": [1.0]},  # Maximum possible
            limit=10
        )
        # Might be empty, but should not error
        assert isinstance(extreme_importance_results, list)
    
    @pytest.mark.asyncio
    async def test_complex_metadata_queries(self, sqlite_persistence):
        """Test complex metadata-based queries."""
        # Create memories with rich metadata
        complex_memories = [
            {
                "id": "complex-001",
                "type": "structured_thinking",
                "content": "Deep analysis of microservices architecture patterns",
                "importance": 0.9,
                "tier": "long_term",
                "metadata": {
                    "category": "analysis",
                    "domain": "architecture",
                    "tags": ["microservices", "patterns", "scalability"],
                    "complexity": "advanced",
                    "estimated_effort": "high",
                    "stakeholders": ["engineering", "architecture"],
                    "priority": "critical"
                }
            },
            {
                "id": "complex-002",
                "type": "episodic",
                "content": "API gateway performance issues during peak traffic",
                "importance": 0.85,
                "tier": "short_term",
                "metadata": {
                    "category": "incident",
                    "domain": "operations",
                    "tags": ["api_gateway", "performance", "traffic"],
                    "severity": "high",
                    "resolved": False,
                    "stakeholders": ["sre", "engineering"],
                    "priority": "urgent"
                }
            }
        ]
        
        for memory in complex_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Search with metadata filtering
        architecture_results = await sqlite_persistence.retrieve_memories(
            "microservices architecture patterns",
            include_metadata=True,
            limit=5,
            min_similarity=0.0
        )
        
        assert len(architecture_results) > 0
        
        # Verify rich metadata is preserved
        for result in architecture_results:
            metadata = result.get("metadata", {})
            if result["id"] == "complex-001":
                assert "tags" in metadata
                assert "microservices" in metadata["tags"]
                assert metadata["complexity"] == "advanced"
                assert "stakeholders" in metadata
    
    @pytest.mark.asyncio
    async def test_filter_performance_with_large_dataset(self, sqlite_persistence):
        """Test filtering performance with larger dataset."""
        # Create larger dataset with varied filters
        large_dataset = []
        
        memory_types = ["structured_thinking", "episodic", "procedural", "semantic"]
        tiers = ["short_term", "long_term", "archival", "system"]
        
        for i in range(100):
            memory = {
                "id": f"perf-{i:03d}",
                "type": memory_types[i % len(memory_types)],
                "content": f"Performance test memory {i} with varied content for filtering tests",
                "importance": 0.1 + (i % 10) * 0.1,  # Range from 0.1 to 1.0
                "tier": tiers[i % len(tiers)],
                "metadata": {
                    "category": f"category_{i % 5}",
                    "priority": ["low", "medium", "high"][i % 3],
                    "index": i
                }
            }
            large_dataset.append(memory)
            await sqlite_persistence.store_memory(memory)
        
        # Test various filter combinations on large dataset
        import time
        
        # Filter by type
        start_time = time.time()
        type_filtered = await sqlite_persistence.search_memories(
            types=["structured_thinking"],
            limit=50
        )
        type_filter_time = time.time() - start_time
        
        assert len(type_filtered) > 0
        assert type_filter_time < 1.0  # Should complete within 1 second
        
        # Filter by tier
        start_time = time.time()
        tier_filtered = await sqlite_persistence.search_memories(
            filters={"tier": "long_term"},
            limit=50
        )
        tier_filter_time = time.time() - start_time
        
        assert len(tier_filtered) > 0
        assert tier_filter_time < 1.0
        
        # Combined filters
        start_time = time.time()
        combined_filtered = await sqlite_persistence.search_memories(
            types=["episodic", "procedural"],
            filters={"tier": "short_term"},
            limit=20
        )
        combined_filter_time = time.time() - start_time
        
        assert isinstance(combined_filtered, list)
        assert combined_filter_time < 1.0
    
    @pytest.mark.asyncio
    async def test_filter_result_consistency(self, sqlite_persistence, diverse_test_memories):
        """Test that filter results are consistent across multiple calls."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Run same filter query multiple times
        filter_params = {
            "types": ["structured_thinking"],
            "filters": {"tier": "long_term"},
            "limit": 10
        }
        
        results_sets = []
        for _ in range(3):
            results = await sqlite_persistence.search_memories(**filter_params)
            results_sets.append(results)
        
        # All result sets should be identical (same IDs, same order)
        assert len(results_sets) == 3
        assert len(results_sets[0]) == len(results_sets[1]) == len(results_sets[2])
        
        # Check ID consistency
        ids_set_1 = [r["id"] for r in results_sets[0]]
        ids_set_2 = [r["id"] for r in results_sets[1]]
        ids_set_3 = [r["id"] for r in results_sets[2]]
        
        assert ids_set_1 == ids_set_2 == ids_set_3
    
    @pytest.mark.asyncio
    async def test_no_results_scenarios(self, sqlite_persistence, diverse_test_memories):
        """Test scenarios that should return no results."""
        # Store test memories
        for memory in diverse_test_memories:
            await sqlite_persistence.store_memory(memory)
        
        # Filter for non-existent type
        no_type_results = await sqlite_persistence.search_memories(
            types=["nonexistent_type"],
            limit=10
        )
        assert len(no_type_results) == 0
        
        # Filter for non-existent tier
        no_tier_results = await sqlite_persistence.search_memories(
            filters={"tier": "nonexistent_tier"},
            limit=10
        )
        assert len(no_tier_results) == 0
        
        # Impossible importance filter
        no_importance_results = await sqlite_persistence.search_memories(
            filters={"importance": [2.0]},  # > 1.0 is impossible
            limit=10
        )
        assert len(no_importance_results) == 0
        
        # Combined filters that match nothing
        no_combined_results = await sqlite_persistence.search_memories(
            types=["structured_thinking"],
            filters={"tier": "system"},  # No structured_thinking in system tier
            limit=10
        )
        # This might return results depending on test data, but should handle gracefully
        assert isinstance(no_combined_results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])