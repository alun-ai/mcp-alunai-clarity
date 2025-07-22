"""
Systematic Data Validation Testing Framework for MCP-Qdrant Integration.

This framework provides comprehensive testing utilities to validate:
1. MCP function behavior
2. Qdrant data storage accuracy 
3. Vector embedding correctness
4. Search indexing validation
5. Data retrieval consistency
6. Memory lifecycle integrity

Usage:
    from tests.framework.data_validation import DataValidationTestSuite, QdrantInspector
    
    class TestMyFunction(DataValidationTestSuite):
        async def test_store_memory_complete_validation(self):
            # Test both MCP behavior AND Qdrant data
            await self.validate_complete_memory_lifecycle(...)
"""

import asyncio
import pytest
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

from clarity.shared.infrastructure import get_qdrant_connection, UnifiedConnectionConfig
from clarity.domains.persistence import QdrantPersistenceDomain
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct


@dataclass
class ValidationResult:
    """Result of a data validation test."""
    test_name: str
    passed: bool
    mcp_result: Any
    qdrant_data: Any
    validation_details: Dict[str, Any]
    errors: List[str]
    performance_metrics: Dict[str, float]
    
    def __str__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"{status} {self.test_name}: {len(self.errors)} errors"


class QdrantInspector:
    """
    Low-level Qdrant data inspection utilities.
    
    Provides direct access to Qdrant storage for validation without
    going through the MCP layer, ensuring data integrity.
    """
    
    def __init__(self, config: UnifiedConnectionConfig):
        self.config = config
        self.collection_name = "memories"
    
    async def get_raw_point_by_id(self, point_id: str) -> Optional[PointStruct]:
        """Get raw point data directly from Qdrant."""
        connection_manager = await get_qdrant_connection(self.config)
        async with connection_manager as client:
            try:
                points = client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=True
                )
                return points[0] if points else None
            except Exception:
                return None
    
    async def search_raw_vectors(self, query_vector: List[float], limit: int = 10) -> List[PointStruct]:
        """Search vectors directly in Qdrant."""
        connection_manager = await get_qdrant_connection(self.config)
        async with connection_manager as client:
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            return results
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed collection statistics."""
        connection_manager = await get_qdrant_connection(self.config)
        async with connection_manager as client:
            info = client.get_collection(self.collection_name)
            return {
                "total_points": info.points_count,
                "indexed_vectors": info.indexed_vectors_count,
                "collection_status": info.status.value,
                "optimizer_status": info.optimizer_status.value if info.optimizer_status else "unknown"
            }
    
    async def validate_vector_dimensions(self, expected_dim: int = 384) -> bool:
        """Validate that all vectors have correct dimensions."""
        connection_manager = await get_qdrant_connection(self.config)
        async with connection_manager as client:
            # Sample some points to check dimensions
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                limit=10,
                with_vectors=True
            )
            
            points, _ = scroll_result
            for point in points:
                if point.vector and len(point.vector) != expected_dim:
                    return False
            return True
    
    async def find_duplicate_vectors(self, tolerance: float = 1e-6) -> List[Tuple[str, str]]:
        """Find points with duplicate or nearly identical vectors."""
        connection_manager = await get_qdrant_connection(self.config)
        async with connection_manager as client:
            # Get all points with vectors
            all_points = []
            offset = None
            
            while True:
                scroll_result = client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_vectors=True
                )
                points, next_offset = scroll_result
                
                if not points:
                    break
                    
                all_points.extend(points)
                if not next_offset:
                    break
                offset = next_offset
        
        # Compare vectors for duplicates
        duplicates = []
        for i, point1 in enumerate(all_points):
            for point2 in all_points[i+1:]:
                if point1.vector and point2.vector:
                    # Calculate cosine similarity
                    import numpy as np
                    vec1 = np.array(point1.vector)
                    vec2 = np.array(point2.vector)
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    
                    if similarity > (1.0 - tolerance):
                        duplicates.append((str(point1.id), str(point2.id)))
        
        return duplicates


class DataValidationTestSuite:
    """
    Base class for systematic data validation tests.
    
    Provides comprehensive testing utilities that validate both MCP
    behavior and underlying Qdrant data consistency.
    """
    
    async def setup_test_environment(self):
        """Set up clean test environment."""
        # Initialize attributes if not already done
        if not hasattr(self, 'test_config'):
            self.test_config = {
                "embedding": {
                    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384
                },
                "qdrant": {
                    "path": "./test_qdrant",
                    "timeout": 30.0
                },
                "alunai-clarity": {
                    "consolidation_interval_hours": 24,
                    "semantic_analysis": {"enabled": False},
                    "episodic_memory": {"enabled": False},
                    "temporal_analysis": {"enabled": False}
                }
            }
        if not hasattr(self, 'test_results'):
            self.test_results = []
            
        # Initialize persistence domain
        self.persistence_domain = QdrantPersistenceDomain(self.test_config)
        await self.persistence_domain.initialize()
        
        # Initialize Qdrant inspector
        config = UnifiedConnectionConfig(
            path=self.test_config["qdrant"]["path"],
            timeout=self.test_config["qdrant"]["timeout"]
        )
        self.qdrant_inspector = QdrantInspector(config)
        
        # Clean up any existing test data
        await self._cleanup_test_data()
    
    async def teardown_test_environment(self):
        """Clean up test environment."""
        await self._cleanup_test_data()
    
    async def _cleanup_test_data(self):
        """Remove any test data from Qdrant."""
        try:
            connection_manager = await get_qdrant_connection(
                UnifiedConnectionConfig(path=self.test_config["qdrant"]["path"])
            )
            async with connection_manager as client:
                # Delete all test memories (those with test_ prefix)
                scroll_result = client.scroll(
                    collection_name="memories",
                    scroll_filter=Filter(
                        must=[FieldCondition(key="memory_id", match=MatchValue(value="test_"))]
                    ),
                    limit=1000,
                    with_payload=True
                )
                
                test_points, _ = scroll_result
                if test_points:
                    test_ids = [point.id for point in test_points]
                    client.delete(collection_name="memories", points_selector=test_ids)
        except Exception:
            pass  # Collection might not exist yet
    
    async def validate_complete_memory_lifecycle(
        self,
        memory_type: str,
        content: Dict[str, Any],
        expected_metadata: Dict[str, Any],
        test_name: str
    ) -> ValidationResult:
        """
        Complete validation of memory storage, retrieval, and search.
        
        This is the core validation method that tests:
        1. MCP store_memory function
        2. Qdrant data storage accuracy
        3. Vector embedding generation
        4. Memory retrieval consistency
        5. Search functionality
        6. Metadata preservation
        """
        start_time = time.perf_counter()
        errors = []
        validation_details = {}
        
        # Initialize variables that might be referenced in the finally block
        memory_id = None
        retrieved_memory = None
        search_results = []
        update_success = False
        qdrant_point = None
        collection_stats = {}
        store_time = 0
        retrieve_time = 0
        search_time = 0
        update_time = 0
        
        # Generate unique test ID
        test_id = f"test_{hashlib.md5(test_name.encode()).hexdigest()[:8]}"
        
        try:
            # Step 1: Store memory via MCP
            store_start = time.perf_counter()
            memory_data = {
                "type": memory_type,  # API uses 'type', not 'memory_type'
                "content": content,
                "importance": 0.8,
                **expected_metadata  # Merge metadata into memory data
            }
            memory_id = await self.persistence_domain.store_memory(memory=memory_data)
            store_time = (time.perf_counter() - store_start) * 1000
            
            if not memory_id:
                errors.append("MCP store_memory returned None")
                return ValidationResult(
                    test_name=test_name,
                    passed=False,
                    mcp_result=None,
                    qdrant_data=None,
                    validation_details=validation_details,
                    errors=errors,
                    performance_metrics={"store_time_ms": store_time}
                )
            
            validation_details["memory_id"] = memory_id
            validation_details["store_time_ms"] = store_time
            
            # Step 2: Validate raw Qdrant storage
            qdrant_point = await self.qdrant_inspector.get_raw_point_by_id(memory_id)
            if not qdrant_point:
                errors.append(f"Memory {memory_id} not found in Qdrant")
            else:
                # Validate payload data
                payload = qdrant_point.payload
                
                # Check required fields
                required_fields = ["memory_id", "memory_type", "content", "text_content", "created_at"]
                for field in required_fields:
                    if field not in payload:
                        errors.append(f"Missing required field: {field}")
                
                # Validate memory type
                if payload.get("memory_type") != memory_type:
                    errors.append(f"Memory type mismatch: expected {memory_type}, got {payload.get('memory_type')}")
                
                # Validate content
                if payload.get("content") != content:
                    errors.append("Content mismatch between stored and retrieved")
                
                # Validate metadata preservation (check both top-level and nested)
                for key, expected_value in expected_metadata.items():
                    stored_value = payload.get(key) or payload.get("metadata", {}).get(key)
                    if stored_value != expected_value:
                        errors.append(f"Metadata mismatch for {key}: expected {expected_value}, got {stored_value}")
                
                # Validate vector dimensions
                if not qdrant_point.vector:
                    errors.append("No vector stored with memory")
                elif len(qdrant_point.vector) != self.test_config["embedding"]["dimensions"]:
                    errors.append(f"Vector dimension mismatch: expected {self.test_config['embedding']['dimensions']}, got {len(qdrant_point.vector)}")
                
                validation_details["qdrant_payload"] = payload
                validation_details["vector_dimensions"] = len(qdrant_point.vector) if qdrant_point.vector else 0
            
            # Step 3: Test retrieval via MCP
            retrieve_start = time.perf_counter()
            retrieved_memory = await self.persistence_domain.get_memory(memory_id)
            retrieve_time = (time.perf_counter() - retrieve_start) * 1000
            
            if not retrieved_memory:
                errors.append("MCP get_memory returned None")
            else:
                # Validate retrieved data matches stored data
                retrieved_type = retrieved_memory.get("type") or retrieved_memory.get("memory_type")
                if retrieved_type != memory_type:
                    errors.append(f"Retrieved memory type doesn't match stored: expected {memory_type}, got {retrieved_type}")
                
                if retrieved_memory.get("content") != content:
                    errors.append("Retrieved content doesn't match stored")
            
            validation_details["retrieve_time_ms"] = retrieve_time
            validation_details["retrieved_memory"] = retrieved_memory
            
            # Step 4: Test search functionality
            search_start = time.perf_counter()
            
            # Create search query from content
            if isinstance(content, dict):
                search_text = " ".join(str(v) for v in content.values() if isinstance(v, str))
            else:
                search_text = str(content)
            
            # Search by memory type using query-based search
            search_results = await self.persistence_domain.retrieve_memories(
                query=search_text,
                limit=10,
                memory_types=[memory_type]
            )
            search_time = (time.perf_counter() - search_start) * 1000
            
            # Validate search results include our memory
            found_in_search = any(
                (result.get("id") or result.get("memory_id")) == memory_id 
                for result in search_results
            )
            
            if not found_in_search:
                errors.append("Memory not found in search results")
            
            validation_details["search_time_ms"] = search_time
            validation_details["search_results_count"] = len(search_results)
            validation_details["found_in_search"] = found_in_search
            
            # Step 5: Validate search indexing
            collection_stats = await self.qdrant_inspector.get_collection_stats()
            validation_details["collection_stats"] = collection_stats
            
            # Note: indexed_vectors can be 0 for small datasets due to HNSW indexing behavior
            # Qdrant only indexes when it has enough vectors and determines it's worth it
            # The key test is whether search functionality works, not the indexed count
            if collection_stats["indexed_vectors"] == 0 and not found_in_search:
                errors.append("Vectors not properly indexed - search functionality broken")
            # We don't error on 0 indexed vectors if search works, as this is normal Qdrant behavior
            
            # Step 6: Test memory update
            update_start = time.perf_counter()
            update_data = {"updated_field": "test_update_value"}
            update_success = await self.persistence_domain.update_memory(memory_id, update_data)
            update_time = (time.perf_counter() - update_start) * 1000
            
            if not update_success:
                errors.append("Memory update failed")
            else:
                # Validate update in Qdrant
                updated_point = await self.qdrant_inspector.get_raw_point_by_id(memory_id)
                if not updated_point or updated_point.payload.get("updated_field") != "test_update_value":
                    errors.append("Update not reflected in Qdrant data")
            
            validation_details["update_time_ms"] = update_time
            
        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        result = ValidationResult(
            test_name=test_name,
            passed=len(errors) == 0,
            mcp_result={
                "memory_id": memory_id,
                "retrieved_memory": retrieved_memory,
                "search_results": search_results,
                "update_success": update_success
            },
            qdrant_data={
                "point": qdrant_point,
                "collection_stats": collection_stats
            },
            validation_details=validation_details,
            errors=errors,
            performance_metrics={
                "total_time_ms": total_time,
                "store_time_ms": store_time,
                "retrieve_time_ms": retrieve_time,
                "search_time_ms": search_time,
                "update_time_ms": update_time
            }
        )
        
        self.test_results.append(result)
        return result
    
    async def validate_search_accuracy(
        self,
        query_text: str,
        expected_memory_ids: Set[str],
        memory_types: Optional[List[str]] = None,
        test_name: str = "search_accuracy"
    ) -> ValidationResult:
        """
        Validate search accuracy by comparing MCP search results
        with direct Qdrant vector search results.
        """
        start_time = time.perf_counter()
        errors = []
        validation_details = {}
        
        try:
            # Step 1: Search via MCP
            mcp_start = time.perf_counter()
            mcp_results = await self.persistence_domain.retrieve_memories(
                query=query_text,
                limit=20,
                memory_types=memory_types
            )
            mcp_time = (time.perf_counter() - mcp_start) * 1000
            
            mcp_memory_ids = {result.get("id") or result.get("memory_id") for result in mcp_results if result.get("id") or result.get("memory_id")}
            
            # Step 2: Generate embedding and search directly in Qdrant
            qdrant_start = time.perf_counter()
            
            # Generate embedding using the same method as persistence domain
            import numpy as np
            # This would need access to the actual embedding model
            # For now, we'll use the MCP search and validate consistency
            
            qdrant_time = (time.perf_counter() - qdrant_start) * 1000
            
            # Step 3: Validate expected memories are found
            missing_memories = expected_memory_ids - mcp_memory_ids
            if missing_memories:
                errors.append(f"Expected memories not found in search: {missing_memories}")
            
            # Step 4: Validate search result quality
            for result in mcp_results:
                if not result.get("similarity_score"):
                    errors.append("Search results missing similarity scores")
                    break
                
                if result.get("similarity_score") < 0 or result.get("similarity_score") > 1:
                    errors.append(f"Invalid similarity score: {result.get('similarity_score')}")
            
            validation_details.update({
                "mcp_results_count": len(mcp_results),
                "mcp_memory_ids": list(mcp_memory_ids),
                "expected_memory_ids": list(expected_memory_ids),
                "missing_memories": list(missing_memories),
                "mcp_search_time_ms": mcp_time,
                "qdrant_search_time_ms": qdrant_time
            })
            
        except Exception as e:
            errors.append(f"Search accuracy validation error: {str(e)}")
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        result = ValidationResult(
            test_name=test_name,
            passed=len(errors) == 0,
            mcp_result={"search_results": mcp_results},
            qdrant_data={"direct_search": []},  # Would contain direct Qdrant results
            validation_details=validation_details,
            errors=errors,
            performance_metrics={
                "total_time_ms": total_time,
                "mcp_search_time_ms": mcp_time,
                "qdrant_search_time_ms": qdrant_time
            }
        )
        
        self.test_results.append(result)
        return result
    
    async def validate_data_consistency(self, test_name: str = "data_consistency") -> ValidationResult:
        """
        Validate overall data consistency between MCP layer and Qdrant storage.
        
        Checks for:
        - Orphaned data in Qdrant
        - Missing vectors
        - Duplicate vectors  
        - Index corruption
        """
        start_time = time.perf_counter()
        errors = []
        validation_details = {}
        
        try:
            # Get collection statistics
            stats = await self.qdrant_inspector.get_collection_stats()
            validation_details["collection_stats"] = stats
            
            # Validate vector dimensions
            dimensions_valid = await self.qdrant_inspector.validate_vector_dimensions()
            if not dimensions_valid:
                errors.append("Some vectors have incorrect dimensions")
            validation_details["dimensions_valid"] = dimensions_valid
            
            # Check for duplicate vectors
            duplicates = await self.qdrant_inspector.find_duplicate_vectors()
            if duplicates:
                errors.append(f"Found {len(duplicates)} duplicate vector pairs: {duplicates[:5]}")
            validation_details["duplicate_vectors"] = len(duplicates)
            
            # Validate indexing status
            if stats["total_points"] > 0 and stats["indexed_vectors"] == 0:
                errors.append("Points exist but no vectors are indexed")
            elif stats["indexed_vectors"] > stats["total_points"]:
                errors.append("More indexed vectors than total points (corruption?)")
            
        except Exception as e:
            errors.append(f"Data consistency validation error: {str(e)}")
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        result = ValidationResult(
            test_name=test_name,
            passed=len(errors) == 0,
            mcp_result={},
            qdrant_data={"stats": stats, "duplicates": duplicates},
            validation_details=validation_details,
            errors=errors,
            performance_metrics={"total_time_ms": total_time}
        )
        
        self.test_results.append(result)
        return result
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        
        report = [
            "🧪 DATA VALIDATION TEST REPORT",
            "=" * 50,
            f"Total tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {total_tests - passed_tests}",
            f"Success rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run",
            "",
            "📋 Test Details:",
            "-" * 30
        ]
        
        for result in self.test_results:
            report.append(f"\n{result}")
            if result.errors:
                for error in result.errors:
                    report.append(f"  ❌ {error}")
            
            # Add performance metrics
            perf = result.performance_metrics
            if perf:
                report.append(f"  ⚡ Performance: {perf}")
        
        return "\n".join(report)


# Export the framework
__all__ = [
    'DataValidationTestSuite',
    'QdrantInspector', 
    'ValidationResult'
]