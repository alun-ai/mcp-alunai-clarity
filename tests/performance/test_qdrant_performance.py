"""
Performance tests for Qdrant operations in Alunai Clarity.
"""

import asyncio
import pytest
import time
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import numpy as np

from clarity.domains.persistence import PersistenceDomain
from clarity.domains.manager import MemoryDomainManager


@pytest.mark.performance
class TestQdrantPerformance:
    """Test Qdrant database performance."""
    
    @pytest.mark.asyncio
    async def test_memory_storage_performance(self, test_config: Dict[str, Any], performance_test_data):
        """Test performance of storing large numbers of memories."""
        # Generate test memories
        test_memories = performance_test_data(1000)  # 1000 memories
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            # Mock embedding generation
            domain.embedding_manager = MagicMock()
            domain.embedding_manager.get_embedding.return_value = np.random.rand(384)
            
            # Measure storage performance
            start_time = time.time()
            
            # Store memories in batches to simulate real usage
            batch_size = 50
            for i in range(0, len(test_memories), batch_size):
                batch = test_memories[i:i + batch_size]
                
                # Store batch
                for memory in batch:
                    await domain.store_memory(memory, "short_term")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            memories_per_second = len(test_memories) / total_time
            assert memories_per_second > 100, f"Storage too slow: {memories_per_second:.2f} memories/sec"
            
            # Verify all memories were stored
            assert mock_client.upsert.call_count == len(test_memories)
            
            print(f"Stored {len(test_memories)} memories in {total_time:.2f}s ({memories_per_second:.2f} memories/sec)")
    
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, test_config: Dict[str, Any], performance_test_data):
        """Test performance of searching through large numbers of memories."""
        # Generate test memories
        test_memories = performance_test_data(5000)  # 5000 memories for search
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            
            # Mock search results (simulate finding relevant memories)
            mock_search_results = []
            for i in range(10):  # Return top 10 results
                mock_result = MagicMock()
                mock_result.id = f"perf_mem_{i}"
                mock_result.score = 0.9 - (i * 0.05)  # Decreasing similarity
                mock_result.payload = test_memories[i]
                mock_search_results.append(mock_result)
            
            mock_client.search.return_value = mock_search_results
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            # Mock embedding generation
            domain.embedding_manager = MagicMock()
            domain.embedding_manager.get_embedding.return_value = np.random.rand(384)
            
            # Measure search performance
            search_queries = [
                "FastAPI web framework",
                "Python programming",
                "database integration",
                "API authentication",
                "performance optimization",
                "testing strategies",
                "deployment configuration",
                "error handling",
                "data validation",
                "security best practices"
            ]
            
            start_time = time.time()
            
            for query in search_queries:
                query_embedding = np.random.rand(384)
                results = await domain.search_memories(
                    embedding=query_embedding,
                    limit=10,
                    min_similarity=0.7
                )
                
                # Verify results
                assert len(results) <= 10
                assert all(result.get("similarity", 0) >= 0.7 for result in results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            searches_per_second = len(search_queries) / total_time
            avg_search_time = total_time / len(search_queries) * 1000  # Convert to ms
            
            assert searches_per_second > 50, f"Search too slow: {searches_per_second:.2f} searches/sec"
            assert avg_search_time < 50, f"Average search time too high: {avg_search_time:.2f}ms"
            
            # Verify search was called for each query
            assert mock_client.search.call_count == len(search_queries)
            
            print(f"Performed {len(search_queries)} searches in {total_time:.2f}s ({searches_per_second:.2f} searches/sec, {avg_search_time:.2f}ms avg)")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, test_config: Dict[str, Any], performance_test_data):
        """Test performance of concurrent memory operations."""
        test_memories = performance_test_data(200)
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client.search = MagicMock(return_value=[])
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            # Mock embedding generation
            domain.embedding_manager = MagicMock()
            domain.embedding_manager.get_embedding.return_value = np.random.rand(384)
            
            # Create concurrent operations
            async def store_batch(memories_batch):
                for memory in memories_batch:
                    await domain.store_memory(memory, "short_term")
            
            async def search_batch(queries):
                for query in queries:
                    query_embedding = np.random.rand(384)
                    await domain.search_memories(
                        embedding=query_embedding,
                        limit=5,
                        min_similarity=0.7
                    )
            
            # Prepare concurrent tasks
            store_tasks = []
            search_tasks = []
            
            # Split memories into batches for concurrent storage
            batch_size = 25
            for i in range(0, len(test_memories), batch_size):
                batch = test_memories[i:i + batch_size]
                store_tasks.append(store_batch(batch))
            
            # Create search tasks
            search_queries = [f"query_{i}" for i in range(20)]
            search_batch_size = 5
            for i in range(0, len(search_queries), search_batch_size):
                batch = search_queries[i:i + search_batch_size]
                search_tasks.append(search_batch(batch))
            
            # Measure concurrent performance
            start_time = time.time()
            
            # Run storage and search operations concurrently
            await asyncio.gather(*store_tasks, *search_tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            total_operations = len(test_memories) + len(search_queries)
            operations_per_second = total_operations / total_time
            
            assert operations_per_second > 200, f"Concurrent operations too slow: {operations_per_second:.2f} ops/sec"
            
            print(f"Performed {total_operations} concurrent operations in {total_time:.2f}s ({operations_per_second:.2f} ops/sec)")
    
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self, test_config: Dict[str, Any]):
        """Test performance of embedding generation."""
        with patch('clarity.utils.embeddings.SentenceTransformer') as mock_model_class:
            # Mock the sentence transformer model
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384)
            mock_model_class.return_value = mock_model
            
            from clarity.utils.embeddings import EmbeddingManager
            
            manager = EmbeddingManager(test_config)
            
            # Test texts of varying lengths
            test_texts = [
                "Short text",
                "This is a medium length text that contains some technical information about FastAPI and web development.",
                "This is a much longer text that contains detailed technical information about web development, including topics such as API design, database integration, authentication mechanisms, performance optimization, testing strategies, deployment configurations, error handling patterns, data validation techniques, security best practices, and monitoring approaches. This text is designed to test the performance of embedding generation with longer input sequences that might be typical in real-world documentation or conversation contexts.",
                "Single word",
                "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
            ]
            
            # Multiply to get more test cases
            extended_texts = test_texts * 50  # 250 total texts
            
            # Measure embedding generation performance
            start_time = time.time()
            
            embeddings = []
            for text in extended_texts:
                embedding = manager.get_embedding(text)
                embeddings.append(embedding)
                
                # Verify embedding properties
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (384,)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            embeddings_per_second = len(extended_texts) / total_time
            avg_embedding_time = total_time / len(extended_texts) * 1000  # Convert to ms
            
            assert embeddings_per_second > 100, f"Embedding generation too slow: {embeddings_per_second:.2f} embeddings/sec"
            assert avg_embedding_time < 50, f"Average embedding time too high: {avg_embedding_time:.2f}ms"
            
            print(f"Generated {len(extended_texts)} embeddings in {total_time:.2f}s ({embeddings_per_second:.2f} embeddings/sec, {avg_embedding_time:.2f}ms avg)")
    
    @pytest.mark.asyncio
    async def test_similarity_calculation_performance(self, test_config: Dict[str, Any]):
        """Test performance of similarity calculations."""
        from clarity.utils.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(test_config)
        
        # Generate test vectors
        num_vectors = 1000
        vector_dim = 384
        vectors = [np.random.rand(vector_dim) for _ in range(num_vectors)]
        
        # Test pairwise similarity calculations
        start_time = time.time()
        
        similarity_results = []
        for i in range(0, min(100, num_vectors)):  # Test first 100 vectors
            for j in range(i + 1, min(i + 20, num_vectors)):  # Compare with next 20
                similarity = manager.calculate_similarity(vectors[i], vectors[j])
                similarity_results.append(similarity)
                
                # Verify similarity properties
                assert 0.0 <= similarity <= 1.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        calculations_per_second = len(similarity_results) / total_time
        avg_calculation_time = total_time / len(similarity_results) * 1000000  # Convert to microseconds
        
        assert calculations_per_second > 10000, f"Similarity calculation too slow: {calculations_per_second:.2f} calcs/sec"
        assert avg_calculation_time < 100, f"Average calculation time too high: {avg_calculation_time:.2f}μs"
        
        print(f"Performed {len(similarity_results)} similarity calculations in {total_time:.2f}s ({calculations_per_second:.2f} calcs/sec, {avg_calculation_time:.2f}μs avg)")


@pytest.mark.performance
class TestMemoryManagerPerformance:
    """Test memory manager performance with realistic workloads."""
    
    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, test_config: Dict[str, Any], performance_test_data):
        """Test performance with mixed memory operations."""
        test_memories = performance_test_data(500)
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client.search = MagicMock(return_value=[])
            mock_client.retrieve = MagicMock(return_value=[])
            mock_client.update = MagicMock()
            mock_client.delete = MagicMock(return_value=True)
            
            manager = MemoryDomainManager(test_config)
            await manager.initialize()
            
            # Mock domain operations
            manager.persistence_domain.generate_embedding = MagicMock(return_value=np.random.rand(384))
            manager.persistence_domain.search_memories = MagicMock(return_value=[])
            manager.persistence_domain.get_memory = MagicMock(return_value=test_memories[0])
            manager.persistence_domain.get_memory_tier = MagicMock(return_value="short_term")
            
            # Simulate realistic mixed workload
            start_time = time.time()
            
            operations_count = 0
            
            # 1. Store initial memories (40% of workload)
            store_count = int(len(test_memories) * 0.4)
            for i in range(store_count):
                memory = test_memories[i]
                await manager.store_memory(
                    memory_type=memory["type"],
                    content=memory["content"],
                    importance=memory["importance"]
                )
                operations_count += 1
            
            # 2. Perform searches (30% of workload)
            search_queries = [
                "Python programming",
                "web development",
                "API design",
                "database integration",
                "testing",
                "deployment",
                "performance",
                "security",
                "authentication",
                "monitoring"
            ]
            
            search_count = int(len(search_queries) * 3)  # Repeat queries for more load
            for i in range(search_count):
                query = search_queries[i % len(search_queries)]
                await manager.retrieve_memories(
                    query=query,
                    limit=10,
                    min_similarity=0.7
                )
                operations_count += 1
            
            # 3. Update memories (20% of workload)
            update_count = int(len(test_memories) * 0.2)
            for i in range(update_count):
                memory_id = f"mem_{i}"
                await manager.update_memory(
                    memory_id,
                    {"importance": 0.8, "metadata": {"updated": True}}
                )
                operations_count += 1
            
            # 4. List memories (5% of workload)
            list_count = 10
            for i in range(list_count):
                await manager.list_memories(
                    limit=20,
                    offset=i * 20,
                    include_content=True
                )
                operations_count += 1
            
            # 5. Delete some memories (5% of workload)
            delete_count = int(len(test_memories) * 0.05)
            delete_batch_size = 5
            for i in range(0, delete_count, delete_batch_size):
                memory_ids = [f"mem_{j}" for j in range(i, min(i + delete_batch_size, delete_count))]
                await manager.delete_memories(memory_ids)
                operations_count += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            operations_per_second = operations_count / total_time
            
            assert operations_per_second > 50, f"Mixed workload too slow: {operations_per_second:.2f} ops/sec"
            
            print(f"Performed {operations_count} mixed operations in {total_time:.2f}s ({operations_per_second:.2f} ops/sec)")
    
    @pytest.mark.asyncio
    async def test_autocode_operations_performance(self, test_config: Dict[str, Any]):
        """Test performance of AutoCode-specific operations."""
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            
            manager = MemoryDomainManager(test_config)
            await manager.initialize()
            
            # Measure AutoCode operations performance
            start_time = time.time()
            
            operations_count = 0
            
            # Store project patterns
            for i in range(50):
                await manager.store_project_pattern(
                    pattern_type="framework",
                    framework=f"Framework{i}",
                    language="python",
                    structure={"files": [f"file{j}.py" for j in range(5)]},
                    importance=0.8
                )
                operations_count += 1
            
            # Store command patterns
            for i in range(100):
                await manager.store_command_pattern(
                    command=f"command_{i}",
                    context={"type": "test", "framework": f"Framework{i % 10}"},
                    success_rate=0.9,
                    platform="linux",
                    importance=0.7
                )
                operations_count += 1
            
            # Store session summaries
            for i in range(25):
                await manager.store_session_summary(
                    session_id=f"session_{i}",
                    tasks_completed=[{"task": f"task_{j}", "status": "completed"} for j in range(3)],
                    patterns_used=[f"Pattern{j}" for j in range(2)],
                    files_modified=[f"file{j}.py" for j in range(4)],
                    importance=0.85
                )
                operations_count += 1
            
            # Store bash executions
            for i in range(75):
                await manager.store_bash_execution(
                    command=f"bash_command_{i}",
                    exit_code=0 if i % 10 != 0 else 1,  # 10% failure rate
                    output=f"output for command {i}",
                    context={"timestamp": f"2023-01-{i:02d}T12:00:00"},
                    importance=0.5
                )
                operations_count += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            operations_per_second = operations_count / total_time
            
            assert operations_per_second > 30, f"AutoCode operations too slow: {operations_per_second:.2f} ops/sec"
            
            print(f"Performed {operations_count} AutoCode operations in {total_time:.2f}s ({operations_per_second:.2f} ops/sec)")


@pytest.mark.performance  
@pytest.mark.slow
class TestScalabilityPerformance:
    """Test system performance at scale."""
    
    @pytest.mark.asyncio
    async def test_large_scale_memory_operations(self, test_config: Dict[str, Any], performance_test_data):
        """Test performance with large numbers of memories."""
        # Generate large dataset
        large_dataset = performance_test_data(10000)  # 10K memories
        
        with patch('clarity.domains.persistence.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_qdrant_class.return_value = mock_client
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client.search = MagicMock(return_value=[])
            
            domain = PersistenceDomain(test_config)
            await domain.initialize()
            
            # Mock embedding generation
            domain.embedding_manager = MagicMock()
            domain.embedding_manager.get_embedding.return_value = np.random.rand(384)
            
            # Test large-scale storage
            start_time = time.time()
            
            # Store in larger batches for efficiency
            batch_size = 100
            for i in range(0, len(large_dataset), batch_size):
                batch = large_dataset[i:i + batch_size]
                
                # Process batch concurrently
                tasks = []
                for memory in batch:
                    task = domain.store_memory(memory, "short_term")
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            storage_time = time.time() - start_time
            
            # Test large-scale search
            search_start = time.time()
            
            # Perform multiple concurrent searches
            search_tasks = []
            for i in range(100):  # 100 concurrent searches
                query_embedding = np.random.rand(384)
                task = domain.search_memories(
                    embedding=query_embedding,
                    limit=20,
                    min_similarity=0.6
                )
                search_tasks.append(task)
            
            await asyncio.gather(*search_tasks)
            
            search_time = time.time() - search_start
            total_time = time.time() - start_time
            
            # Performance assertions
            storage_rate = len(large_dataset) / storage_time
            search_rate = 100 / search_time
            
            # At scale, we expect some degradation but still reasonable performance
            assert storage_rate > 500, f"Large-scale storage too slow: {storage_rate:.2f} memories/sec"
            assert search_rate > 20, f"Large-scale search too slow: {search_rate:.2f} searches/sec"
            
            print(f"Large scale test: {len(large_dataset)} memories stored in {storage_time:.2f}s ({storage_rate:.2f}/sec)")
            print(f"Large scale test: 100 searches in {search_time:.2f}s ({search_rate:.2f}/sec)")
            print(f"Total large scale test time: {total_time:.2f}s")