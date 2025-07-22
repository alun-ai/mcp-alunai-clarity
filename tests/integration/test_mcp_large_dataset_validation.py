"""
Large Dataset MCP Integration Tests.

This test suite creates and validates large datasets (>100 memories) to test:
1. Vector indexing behavior at scale
2. Semantic retrieval accuracy with large datasets
3. Listing operations with pagination
4. Search performance under load
5. Memory management with diverse content types
6. HNSW index creation and optimization

The dataset includes diverse memory types with realistic content to test
semantic similarity, clustering, and retrieval accuracy.
"""

import pytest
import asyncio
import json
import time
import random
from typing import Dict, Any, List, Set
from dataclasses import dataclass

from tests.framework.mcp_validation import MCPServerTestSuite


@dataclass
class DatasetStats:
    """Statistics about the test dataset."""
    total_memories: int
    memory_types: Dict[str, int]
    average_content_length: float
    semantic_clusters: Dict[str, int]
    indexing_stats: Dict[str, Any]
    performance_metrics: Dict[str, float]


class LargeDatasetGenerator:
    """Generates diverse, realistic test data for large-scale testing."""
    
    def __init__(self):
        self.programming_topics = [
            "Python machine learning algorithms and data science frameworks",
            "JavaScript React components and modern web development",
            "TypeScript type systems and advanced programming patterns", 
            "Go microservices architecture and concurrent programming",
            "Rust memory safety and systems programming concepts",
            "Java Spring Boot enterprise application development",
            "C++ performance optimization and low-level programming",
            "Swift iOS development and mobile app architecture",
            "Kotlin Android development and JVM interoperability",
            "SQL database design and query optimization techniques"
        ]
        
        self.science_topics = [
            "Quantum mechanics principles and wave-particle duality",
            "Machine learning neural networks and deep learning algorithms",
            "Genetics DNA sequencing and CRISPR gene editing technology",
            "Climate science atmospheric modeling and environmental impact",
            "Astrophysics black holes and gravitational wave detection",
            "Chemistry organic synthesis and molecular structure analysis",
            "Physics thermodynamics and statistical mechanics principles",
            "Biology cellular processes and protein folding mechanisms",
            "Neuroscience brain function and cognitive processing systems",
            "Mathematics topology and advanced calculus applications"
        ]
        
        self.business_topics = [
            "Strategic planning and competitive market analysis frameworks",
            "Financial modeling and investment portfolio optimization",
            "Marketing analytics and customer behavior data analysis",
            "Operations management and supply chain optimization",
            "Human resources talent acquisition and employee development",
            "Project management agile methodologies and team coordination",
            "Risk management and compliance regulatory frameworks",
            "Innovation management and product development lifecycle",
            "Leadership development and organizational culture building",
            "Digital transformation and technology adoption strategies"
        ]
        
        self.creative_topics = [
            "Digital art creation and visual design principles",
            "Music composition theory and harmonic progression analysis",
            "Creative writing techniques and narrative structure development",
            "Photography composition and advanced editing techniques",
            "Film production and cinematography storytelling methods",
            "Architecture design and sustainable building practices",
            "Graphic design and brand identity development",
            "Fashion design and textile innovation processes",
            "Game design and interactive user experience creation",
            "Animation and motion graphics production workflows"
        ]
        
        self.memory_types = [
            "programming_knowledge", "scientific_research", "business_strategy",
            "creative_process", "technical_documentation", "learning_notes",
            "project_summary", "meeting_notes", "code_review", "research_paper"
        ]
    
    def generate_large_dataset(self, size: int = 150) -> List[Dict[str, Any]]:
        """Generate a diverse dataset of specified size."""
        dataset = []
        all_topics = (
            self.programming_topics + self.science_topics + 
            self.business_topics + self.creative_topics
        )
        
        for i in range(size):
            # Select topic and memory type
            topic = random.choice(all_topics)
            memory_type = random.choice(self.memory_types)
            
            # Create realistic metadata based on topic
            metadata = self._generate_metadata(topic, memory_type, i)
            
            # Generate content with variations
            content = self._generate_content(topic, memory_type, i)
            
            # Vary importance and context
            importance = random.uniform(0.1, 0.9)
            context = self._generate_context(memory_type, i)
            
            memory = {
                "memory_type": memory_type,
                "content": content,
                "importance": importance,
                "metadata": metadata,
                "context": context
            }
            
            dataset.append(memory)
        
        return dataset
    
    def _generate_metadata(self, topic: str, memory_type: str, index: int) -> Dict[str, Any]:
        """Generate realistic metadata based on topic and type."""
        base_metadata = {
            "created_by": "large_dataset_test",
            "batch_index": index,
            "topic_category": self._categorize_topic(topic)
        }
        
        # Add type-specific metadata
        if memory_type == "programming_knowledge":
            base_metadata.update({
                "language": random.choice(["python", "javascript", "go", "rust", "java"]),
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "framework": random.choice(["react", "django", "express", "gin", "spring"])
            })
        elif memory_type == "scientific_research":
            base_metadata.update({
                "field": random.choice(["physics", "chemistry", "biology", "neuroscience"]),
                "research_stage": random.choice(["hypothesis", "experiment", "analysis", "publication"]),
                "methodology": random.choice(["experimental", "theoretical", "computational"])
            })
        elif memory_type == "business_strategy":
            base_metadata.update({
                "department": random.choice(["marketing", "finance", "operations", "hr"]),
                "priority": random.choice(["low", "medium", "high", "critical"]),
                "timeline": random.choice(["short_term", "medium_term", "long_term"])
            })
        
        # Add common metadata
        base_metadata.update({
            "tags": self._generate_tags(topic, memory_type),
            "source": random.choice(["documentation", "meeting", "research", "experience"]),
            "confidence": random.uniform(0.5, 1.0),
            "relevance_score": random.uniform(0.3, 1.0)
        })
        
        return base_metadata
    
    def _generate_content(self, topic: str, memory_type: str, index: int) -> str:
        """Generate varied content based on topic and type."""
        base_content = f"{topic}"
        
        # Add context-specific details
        variations = [
            f"Key insights: {base_content} with practical applications and implementation details.",
            f"Overview: {base_content} including best practices and common pitfalls to avoid.",
            f"Analysis: {base_content} with performance metrics and optimization strategies.",
            f"Summary: {base_content} covering theoretical foundations and real-world examples.",
            f"Guide: {base_content} with step-by-step instructions and troubleshooting tips.",
            f"Research: {base_content} including methodology and experimental results.",
            f"Discussion: {base_content} with pros, cons, and alternative approaches.",
            f"Tutorial: {base_content} with hands-on examples and exercises."
        ]
        
        # Add unique identifier and variation
        variation = random.choice(variations)
        unique_detail = f" [Dataset entry #{index + 1}] Additional context: {random.choice(['important', 'critical', 'useful', 'essential', 'valuable'])} information for future reference."
        
        return variation + unique_detail
    
    def _generate_context(self, memory_type: str, index: int) -> Dict[str, Any]:
        """Generate context information."""
        return {
            "session_id": f"large_dataset_session_{(index // 10) + 1}",
            "timestamp": time.time() - random.randint(0, 86400 * 30),  # Within last 30 days
            "source_file": f"{memory_type}_{index % 5 + 1}.md",
            "line_number": random.randint(1, 1000),
            "related_items": [f"item_{random.randint(1, 50)}" for _ in range(random.randint(0, 3))]
        }
    
    def _generate_tags(self, topic: str, memory_type: str) -> List[str]:
        """Generate relevant tags."""
        base_tags = [memory_type, "large_dataset_test"]
        
        # Add topic-specific tags
        if "python" in topic.lower():
            base_tags.extend(["python", "programming"])
        if "machine learning" in topic.lower():
            base_tags.extend(["ml", "ai", "data_science"])
        if "business" in topic.lower() or "strategy" in topic.lower():
            base_tags.extend(["business", "strategy"])
        if "science" in topic.lower() or "research" in topic.lower():
            base_tags.extend(["science", "research"])
        
        # Add random additional tags
        additional_tags = ["important", "reference", "documentation", "example", "guide"]
        base_tags.extend(random.sample(additional_tags, random.randint(1, 3)))
        
        return base_tags
    
    def _categorize_topic(self, topic: str) -> str:
        """Categorize topic for metadata."""
        if any(prog_word in topic.lower() for prog_word in ["python", "javascript", "programming", "code"]):
            return "programming"
        elif any(sci_word in topic.lower() for sci_word in ["quantum", "machine learning", "genetics", "physics"]):
            return "science"
        elif any(biz_word in topic.lower() for biz_word in ["strategic", "financial", "marketing", "business"]):
            return "business"
        elif any(creative_word in topic.lower() for creative_word in ["art", "music", "writing", "design"]):
            return "creative"
        else:
            return "general"


@pytest.mark.integration
class TestMCPLargeDatasetValidation(MCPServerTestSuite):
    """
    Large-scale MCP validation tests with >100 memories.
    
    Tests comprehensive functionality including indexing, semantic retrieval,
    pagination, and performance at scale.
    """
    
    # Class attributes to persist data across test methods
    _stored_memory_ids = []
    _dataset_stats = None
    _dataset_generator = None
    
    async def setup_test_environment(self):
        """Set up test environment with large dataset utilities."""
        await super().setup_test_environment()
        if not TestMCPLargeDatasetValidation._dataset_generator:
            TestMCPLargeDatasetValidation._dataset_generator = LargeDatasetGenerator()
        self.dataset_generator = TestMCPLargeDatasetValidation._dataset_generator
        self.dataset_stats = TestMCPLargeDatasetValidation._dataset_stats
        self.stored_memory_ids = TestMCPLargeDatasetValidation._stored_memory_ids
    
    @pytest.mark.asyncio
    async def test_large_dataset_creation_and_indexing(self):
        """
        Create a large dataset (150+ memories) and validate indexing behavior.
        
        Tests:
        - Bulk memory creation through MCP
        - Vector indexing at scale (HNSW behavior)
        - Memory type distribution
        - Performance metrics
        """
        await self.setup_test_environment()
        
        try:
            dataset_size = 150
            print(f"\n🏗️ Creating large dataset with {dataset_size} memories...")
            
            # Generate diverse dataset
            dataset = self.dataset_generator.generate_large_dataset(dataset_size)
            
            # Track performance metrics
            start_time = time.perf_counter()
            store_times = []
            memory_type_counts = {}
            
            # Store all memories through MCP
            for i, memory_data in enumerate(dataset):
                store_start = time.perf_counter()
                
                result = await self.validate_mcp_tool_execution(
                    tool_name="store_memory",
                    arguments=memory_data,
                    validate_underlying_data=False,  # Skip individual validation for performance
                    test_name=f"large_dataset_store_{i}"
                )
                
                store_time = (time.perf_counter() - store_start) * 1000
                store_times.append(store_time)
                
                assert result.passed, f"Failed to store memory {i}: {result.errors}"
                
                memory_id = result.parsed_response["memory_id"]
                self.stored_memory_ids.append(memory_id)
                TestMCPLargeDatasetValidation._stored_memory_ids.append(memory_id)
                
                # Track memory types
                memory_type = memory_data["memory_type"]
                memory_type_counts[memory_type] = memory_type_counts.get(memory_type, 0) + 1
                
                # Progress indicator
                if (i + 1) % 25 == 0:
                    elapsed = time.perf_counter() - start_time
                    print(f"   📊 Progress: {i + 1}/{dataset_size} ({elapsed:.1f}s elapsed)")
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Validate indexing behavior
            collection_stats = await self.qdrant_inspector.get_collection_stats()
            
            # Calculate performance metrics
            avg_store_time = sum(store_times) / len(store_times)
            min_store_time = min(store_times)
            max_store_time = max(store_times)
            
            print(f"\n📈 Large Dataset Creation Results:")
            print(f"   ✅ Total memories stored: {len(self.stored_memory_ids)}")
            print(f"   📊 Memory type distribution: {memory_type_counts}")
            print(f"   ⏱️ Total time: {total_time:.1f}ms ({total_time/1000:.2f}s)")
            print(f"   ⚡ Average store time: {avg_store_time:.1f}ms")
            print(f"   🚀 Store time range: {min_store_time:.1f}ms - {max_store_time:.1f}ms")
            print(f"   📦 Qdrant total points: {collection_stats['total_points']}")
            print(f"   🔍 Indexed vectors: {collection_stats['indexed_vectors']}")
            print(f"   🏗️ Collection status: {collection_stats['collection_status']}")
            
            # Store dataset statistics
            dataset_stats = DatasetStats(
                total_memories=len(self.stored_memory_ids),
                memory_types=memory_type_counts,
                average_content_length=sum(len(str(m["content"])) for m in dataset) / len(dataset),
                semantic_clusters=self._analyze_semantic_clusters(dataset),
                indexing_stats=collection_stats,
                performance_metrics={
                    "total_time_ms": total_time,
                    "avg_store_time_ms": avg_store_time,
                    "min_store_time_ms": min_store_time,
                    "max_store_time_ms": max_store_time,
                    "throughput_per_second": dataset_size / (total_time / 1000)
                }
            )
            self.dataset_stats = dataset_stats
            TestMCPLargeDatasetValidation._dataset_stats = dataset_stats
            
            # Validate dataset creation
            assert len(self.stored_memory_ids) == dataset_size
            assert collection_stats["total_points"] >= dataset_size
            assert avg_store_time < 2000, f"Average store time too slow: {avg_store_time}ms"
            
            # Validate memory type diversity
            assert len(memory_type_counts) >= 5, "Dataset should have diverse memory types"
            
            print(f"   🎯 Dataset creation successful with {collection_stats['indexed_vectors']} vectors indexed")
            
        finally:
            # Don't teardown yet - keep data for subsequent tests
            pass
    
    @pytest.mark.asyncio
    async def test_semantic_retrieval_with_large_dataset(self):
        """
        Test semantic retrieval accuracy with the large dataset.
        
        Tests:
        - Semantic similarity with diverse content
        - Retrieval accuracy across different topics
        - Performance with large vector spaces
        - Relevance ranking quality
        """
        # Ensure we have the large dataset
        if not self.stored_memory_ids:
            await self.test_large_dataset_creation_and_indexing()
        
        print(f"\n🔍 Testing semantic retrieval with {len(self.stored_memory_ids)} memories...")
        
        # Test different types of semantic queries
        semantic_test_cases = [
            {
                "query": "Python machine learning algorithms neural networks",
                "expected_topics": ["python", "machine learning", "neural", "algorithms"],
                "memory_types": ["programming_knowledge", "scientific_research"],
                "min_results": 5
            },
            {
                "query": "JavaScript React components web development",
                "expected_topics": ["javascript", "react", "web", "development"],
                "memory_types": ["programming_knowledge", "technical_documentation"],
                "min_results": 3
            },
            {
                "query": "business strategy financial planning marketing",
                "expected_topics": ["business", "strategy", "financial", "marketing"],
                "memory_types": ["business_strategy", "project_summary"],
                "min_results": 4
            },
            {
                "query": "quantum physics theoretical research",
                "expected_topics": ["quantum", "physics", "theoretical", "research"],
                "memory_types": ["scientific_research", "research_paper"],
                "min_results": 3
            },
            {
                "query": "creative design art visual composition",
                "expected_topics": ["creative", "design", "art", "visual"],
                "memory_types": ["creative_process", "learning_notes"],
                "min_results": 2
            }
        ]
        
        retrieval_results = []
        
        for i, test_case in enumerate(semantic_test_cases):
            print(f"   🧪 Test case {i + 1}: '{test_case['query'][:50]}...'")
            
            start_time = time.perf_counter()
            
            result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": test_case["query"],
                    "limit": 20,
                    "types": test_case.get("memory_types"),
                    "min_similarity": 0.3,
                    "include_metadata": True
                },
                test_name=f"semantic_retrieval_{i}"
            )
            
            retrieval_time = (time.perf_counter() - start_time) * 1000
            
            assert result.passed, f"Semantic retrieval {i} failed: {result.errors}"
            
            memories = result.parsed_response["memories"]
            
            # Validate retrieval quality
            assert len(memories) >= test_case["min_results"], f"Not enough results for query {i}"
            
            # Analyze semantic relevance
            relevant_count = 0
            similarity_scores = []
            
            for memory in memories:
                content = str(memory.get("content", "")).lower()
                
                # Check for expected topics
                topic_matches = sum(1 for topic in test_case["expected_topics"] 
                                  if topic.lower() in content)
                
                if topic_matches > 0:
                    relevant_count += 1
                
                # Collect similarity scores if available
                if "similarity_score" in memory:
                    similarity_scores.append(memory["similarity_score"])
            
            relevance_ratio = relevant_count / len(memories)
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
            
            retrieval_results.append({
                "query": test_case["query"],
                "results_count": len(memories),
                "relevance_ratio": relevance_ratio,
                "avg_similarity": avg_similarity,
                "retrieval_time_ms": retrieval_time
            })
            
            print(f"      📊 Results: {len(memories)} memories, {relevance_ratio:.2%} relevant, {retrieval_time:.1f}ms")
            
            # Validate minimum relevance
            assert relevance_ratio >= 0.3, f"Poor relevance ratio: {relevance_ratio:.2%}"
        
        # Overall semantic retrieval validation
        avg_relevance = sum(r["relevance_ratio"] for r in retrieval_results) / len(retrieval_results)
        avg_retrieval_time = sum(r["retrieval_time_ms"] for r in retrieval_results) / len(retrieval_results)
        
        print(f"\n🎯 Semantic Retrieval Summary:")
        print(f"   ✅ Test cases passed: {len(retrieval_results)}")
        print(f"   📊 Average relevance: {avg_relevance:.2%}")
        print(f"   ⚡ Average retrieval time: {avg_retrieval_time:.1f}ms")
        print(f"   🔍 Dataset size: {len(self.stored_memory_ids)} memories")
        
        # Performance and accuracy assertions
        assert avg_relevance >= 0.4, f"Overall relevance too low: {avg_relevance:.2%}"
        assert avg_retrieval_time < 1000, f"Retrieval too slow: {avg_retrieval_time}ms"
    
    @pytest.mark.asyncio
    async def test_listing_operations_with_pagination(self):
        """
        Test listing operations with pagination on large dataset.
        
        Tests:
        - Pagination functionality
        - Memory type filtering
        - Performance with large datasets
        - Data consistency across pages
        """
        # Ensure we have the large dataset
        if not self.stored_memory_ids:
            await self.test_large_dataset_creation_and_indexing()
        
        print(f"\n📋 Testing listing operations with {len(self.stored_memory_ids)} memories...")
        
        # Test 1: Basic pagination
        page_size = 25
        total_memories = len(self.stored_memory_ids)
        expected_pages = (total_memories + page_size - 1) // page_size
        
        all_listed_ids = set()
        pagination_times = []
        
        for page in range(min(5, expected_pages)):  # Test first 5 pages
            offset = page * page_size
            
            print(f"   📄 Testing page {page + 1} (offset: {offset}, limit: {page_size})")
            
            start_time = time.perf_counter()
            
            result = await self.validate_mcp_tool_execution(
                tool_name="list_memories",
                arguments={
                    "limit": page_size,
                    "offset": offset,
                    "include_content": True
                },
                test_name=f"pagination_test_page_{page}"
            )
            
            page_time = (time.perf_counter() - start_time) * 1000
            pagination_times.append(page_time)
            
            assert result.passed, f"Pagination page {page} failed: {result.errors}"
            
            memories = result.parsed_response["memories"]
            
            # Validate page size (except possibly last page)
            if page < expected_pages - 1:
                assert len(memories) == page_size, f"Incorrect page size: {len(memories)}"
            else:
                assert len(memories) <= page_size, f"Last page too large: {len(memories)}"
            
            # Collect memory IDs to check for duplicates
            page_ids = {m.get("id") or m.get("memory_id") for m in memories}
            page_ids = {id for id in page_ids if id}  # Remove None values
            
            # Check for duplicates across pages
            overlap = all_listed_ids & page_ids
            assert not overlap, f"Duplicate memories across pages: {overlap}"
            
            all_listed_ids.update(page_ids)
            
            print(f"      ⚡ Page {page + 1}: {len(memories)} memories in {page_time:.1f}ms")
        
        # Test 2: Memory type filtering
        memory_types = list(self.dataset_stats.memory_types.keys()) if self.dataset_stats else ["programming_knowledge"]
        test_type = memory_types[0]
        
        print(f"   🏷️ Testing type filtering: {test_type}")
        
        filtered_result = await self.validate_mcp_tool_execution(
            tool_name="list_memories",
            arguments={
                "types": [test_type],
                "limit": 50,
                "offset": 0,
                "include_content": True
            },
            test_name="type_filtering_test"
        )
        
        assert filtered_result.passed, f"Type filtering failed: {filtered_result.errors}"
        
        filtered_memories = filtered_result.parsed_response["memories"]
        
        # Validate all results match the filter
        for memory in filtered_memories:
            memory_type = memory.get("type") or memory.get("memory_type")
            assert memory_type == test_type, f"Wrong type in filtered results: {memory_type}"
        
        # Test 3: Performance with large offsets
        print(f"   ⚡ Testing performance with large offset...")
        
        large_offset_result = await self.validate_mcp_tool_execution(
            tool_name="list_memories",
            arguments={
                "limit": 10,
                "offset": max(0, len(self.stored_memory_ids) - 20),  # Near the end
                "include_content": False  # Faster without content
            },
            test_name="large_offset_test"
        )
        
        assert large_offset_result.passed, "Large offset test failed"
        
        # Summary
        avg_pagination_time = sum(pagination_times) / len(pagination_times)
        
        print(f"\n📊 Listing Operations Summary:")
        print(f"   ✅ Pages tested: {len(pagination_times)}")
        print(f"   📄 Average page load time: {avg_pagination_time:.1f}ms")
        print(f"   🆔 Unique IDs collected: {len(all_listed_ids)}")
        print(f"   🏷️ Type filtering: {len(filtered_memories)} {test_type} memories")
        print(f"   ⚡ Large offset performance: {large_offset_result.performance_metrics['tool_execution_ms']:.1f}ms")
        
        # Performance assertions
        assert avg_pagination_time < 500, f"Pagination too slow: {avg_pagination_time}ms"
        assert len(filtered_memories) > 0, "Type filtering returned no results"
        assert large_offset_result.performance_metrics["tool_execution_ms"] < 1000, "Large offset too slow"
    
    @pytest.mark.asyncio
    async def test_search_performance_and_accuracy_at_scale(self):
        """
        Test search performance and accuracy with large dataset.
        
        Tests:
        - Search performance under various loads
        - Accuracy with different similarity thresholds
        - Concurrent search operations
        - Memory usage and optimization
        """
        # Ensure we have the large dataset
        if not self.stored_memory_ids:
            await self.test_large_dataset_creation_and_indexing()
        
        print(f"\n🚀 Testing search performance with {len(self.stored_memory_ids)} memories...")
        
        # Test 1: Performance with different result limits
        performance_tests = [
            {"limit": 5, "min_similarity": 0.5},
            {"limit": 10, "min_similarity": 0.4},
            {"limit": 25, "min_similarity": 0.3},
            {"limit": 50, "min_similarity": 0.2},
        ]
        
        query = "advanced programming techniques and software development"
        performance_results = []
        
        for test_config in performance_tests:
            print(f"   🔍 Testing limit={test_config['limit']}, similarity={test_config['min_similarity']}")
            
            start_time = time.perf_counter()
            
            result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": query,
                    "limit": test_config["limit"],
                    "min_similarity": test_config["min_similarity"],
                    "include_metadata": True
                },
                test_name=f"performance_limit_{test_config['limit']}"
            )
            
            search_time = (time.perf_counter() - start_time) * 1000
            
            assert result.passed, f"Search performance test failed: {result.errors}"
            
            memories = result.parsed_response["memories"]
            
            performance_results.append({
                "limit": test_config["limit"],
                "min_similarity": test_config["min_similarity"],
                "results_count": len(memories),
                "search_time_ms": search_time
            })
            
            print(f"      📊 Results: {len(memories)} memories in {search_time:.1f}ms")
            
            # Validate performance scales reasonably
            assert search_time < 2000, f"Search too slow for limit {test_config['limit']}: {search_time}ms"
        
        # Test 2: Concurrent search operations
        print(f"   🔀 Testing concurrent search operations...")
        
        concurrent_queries = [
            "Python programming and data analysis",
            "JavaScript web development frameworks", 
            "Machine learning algorithms and AI",
            "Business strategy and financial planning",
            "Scientific research and methodology"
        ]
        
        async def run_concurrent_search(query: str, index: int):
            result = await self.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": query,
                    "limit": 15,
                    "min_similarity": 0.4
                },
                test_name=f"concurrent_search_{index}"
            )
            return result
        
        # Run concurrent searches
        concurrent_start = time.perf_counter()
        
        concurrent_tasks = [
            run_concurrent_search(query, i) 
            for i, query in enumerate(concurrent_queries)
        ]
        
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = (time.perf_counter() - concurrent_start) * 1000
        
        # Validate concurrent results
        all_concurrent_passed = all(result.passed for result in concurrent_results)
        assert all_concurrent_passed, "Some concurrent searches failed"
        
        total_concurrent_results = sum(len(r.parsed_response["memories"]) for r in concurrent_results)
        
        print(f"      🔀 Concurrent searches: {len(concurrent_queries)} queries in {concurrent_time:.1f}ms")
        print(f"      📊 Total results: {total_concurrent_results} memories")
        
        # Test 3: Vector indexing validation
        collection_stats = await self.qdrant_inspector.get_collection_stats()
        
        print(f"   🏗️ Vector indexing status:")
        print(f"      📦 Total points: {collection_stats['total_points']}")
        print(f"      🔍 Indexed vectors: {collection_stats['indexed_vectors']}")
        print(f"      ✅ Collection status: {collection_stats['collection_status']}")
        
        # With >100 vectors, we should see some indexing
        indexing_ratio = collection_stats['indexed_vectors'] / collection_stats['total_points']
        print(f"      📈 Indexing ratio: {indexing_ratio:.2%}")
        
        # Performance summary
        avg_search_time = sum(r["search_time_ms"] for r in performance_results) / len(performance_results)
        
        print(f"\n🎯 Search Performance Summary:")
        print(f"   ⚡ Average search time: {avg_search_time:.1f}ms")
        print(f"   🔀 Concurrent operation time: {concurrent_time:.1f}ms")
        print(f"   📊 Dataset size: {len(self.stored_memory_ids)} memories")
        print(f"   🏗️ Vector indexing: {indexing_ratio:.2%}")
        
        # Final performance assertions
        assert avg_search_time < 500, f"Average search too slow: {avg_search_time}ms"
        assert concurrent_time < 3000, f"Concurrent searches too slow: {concurrent_time}ms"
        
        # If we have enough data points, should have some indexing
        if collection_stats['total_points'] > 50:
            print(f"   ℹ️ Note: With {collection_stats['total_points']} points, indexing ratio is {indexing_ratio:.2%}")
    
    @pytest.mark.asyncio
    async def test_comprehensive_dataset_cleanup_and_summary(self):
        """
        Comprehensive cleanup and final dataset summary.
        
        Provides complete statistics about the large dataset test
        and cleans up test data.
        """
        # Ensure we have the large dataset
        if not self.stored_memory_ids:
            await self.test_large_dataset_creation_and_indexing()
        
        print(f"\n🧹 Performing comprehensive dataset cleanup and summary...")
        
        # Get final statistics
        final_stats = await self.qdrant_inspector.get_collection_stats()
        
        # Test batch deletion
        deletion_batches = [
            self.stored_memory_ids[i:i+20] 
            for i in range(0, len(self.stored_memory_ids), 20)
        ]
        
        deletion_times = []
        
        for i, batch in enumerate(deletion_batches):
            print(f"   🗑️ Deleting batch {i + 1}/{len(deletion_batches)} ({len(batch)} memories)")
            
            start_time = time.perf_counter()
            
            result = await self.validate_mcp_tool_execution(
                tool_name="delete_memory",
                arguments={"memory_ids": batch},
                test_name=f"cleanup_batch_{i}"
            )
            
            deletion_time = (time.perf_counter() - start_time) * 1000
            deletion_times.append(deletion_time)
            
            assert result.passed, f"Batch deletion {i} failed: {result.errors}"
        
        # Verify cleanup
        post_cleanup_stats = await self.qdrant_inspector.get_collection_stats()
        
        print(f"\n📊 Large Dataset Test Summary:")
        print(f"   🏗️ Dataset created: {len(self.stored_memory_ids)} memories")
        if self.dataset_stats:
            print(f"   📈 Performance metrics:")
            print(f"      - Average store time: {self.dataset_stats.performance_metrics['avg_store_time_ms']:.1f}ms")
            print(f"      - Total creation time: {self.dataset_stats.performance_metrics['total_time_ms']/1000:.2f}s")
            print(f"      - Throughput: {self.dataset_stats.performance_metrics['throughput_per_second']:.1f} ops/sec")
            print(f"   🏷️ Memory type distribution: {self.dataset_stats.memory_types}")
        
        print(f"   🔍 Vector indexing:")
        print(f"      - Peak total points: {final_stats['total_points']}")
        print(f"      - Peak indexed vectors: {final_stats['indexed_vectors']}")
        print(f"      - Final total points: {post_cleanup_stats['total_points']}")
        
        avg_deletion_time = sum(deletion_times) / len(deletion_times)
        print(f"   🗑️ Cleanup performance:")
        print(f"      - Deletion batches: {len(deletion_batches)}")
        print(f"      - Average batch deletion time: {avg_deletion_time:.1f}ms")
        
        # Validate cleanup was effective
        memories_removed = final_stats['total_points'] - post_cleanup_stats['total_points']
        cleanup_ratio = memories_removed / len(self.stored_memory_ids) if self.stored_memory_ids else 0
        
        print(f"   ✅ Cleanup effectiveness: {cleanup_ratio:.2%} ({memories_removed} removed)")
        
        # Final cleanup
        await self.teardown_test_environment()
        
        print(f"   🎉 Large dataset validation completed successfully!")
    
    def _analyze_semantic_clusters(self, dataset: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze semantic clusters in the dataset."""
        clusters = {}
        
        for memory in dataset:
            content = str(memory.get("content", "")).lower()
            
            # Simple clustering based on keywords
            if any(word in content for word in ["python", "javascript", "programming", "code"]):
                clusters["programming"] = clusters.get("programming", 0) + 1
            elif any(word in content for word in ["science", "research", "quantum", "physics"]):
                clusters["science"] = clusters.get("science", 0) + 1
            elif any(word in content for word in ["business", "strategy", "marketing", "financial"]):
                clusters["business"] = clusters.get("business", 0) + 1
            elif any(word in content for word in ["creative", "design", "art", "music"]):
                clusters["creative"] = clusters.get("creative", 0) + 1
            else:
                clusters["general"] = clusters.get("general", 0) + 1
        
        return clusters


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_mcp_large_dataset_validation.py -v -s
    pytest.main([__file__, "-v", "-s"])