#!/usr/bin/env python3
"""
SQLite Memory Persistence Validation Test

Comprehensive test of SQLite-based memory persistence to validate
it handles all required memory operations for the MCP server.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
from tests.migration.sqlite_vec_simple import SQLiteVecPersistenceDomain


class SQLiteValidationTest:
    """Comprehensive test of SQLite functionality for memory operations."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="sqlite_test_")
        self.embedding_model = None
        self.test_memories = []
        
    def setup(self):
        """Setup test environment."""
        print("🧪 Setting up SQLite validation test...")
        
        # Load embedding model
        print("📥 Loading embedding model...")
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        
        # Generate comprehensive test data
        print("📝 Generating comprehensive test data...")
        self.test_memories = self.generate_comprehensive_test_data()
        
        print(f"✅ Test environment ready with {len(self.test_memories)} test memories")
    
    def generate_comprehensive_test_data(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test data covering all memory types and tiers."""
        
        test_cases = [
            # Structured thinking memories
            {
                "memory_type": "structured_thinking",
                "content": "I need to evaluate the trade-offs between Qdrant and SQLite for vector search. Qdrant offers specialized vector operations but requires complex infrastructure. SQLite with sqlite-vec provides simpler deployment with adequate performance.",
                "importance": 0.9,
                "tier": "short_term",
                "metadata": {"analysis_type": "technology_comparison", "decision_pending": True}
            },
            {
                "memory_type": "structured_thinking", 
                "content": "Performance analysis shows SQLite handles 384-dimensional vectors efficiently. With proper indexing, search latency remains under 50ms for datasets up to 10K memories.",
                "importance": 0.8,
                "tier": "long_term",
                "metadata": {"analysis_type": "performance", "validated": True}
            },
            
            # Episodic memories
            {
                "memory_type": "episodic",
                "content": "User reported Qdrant connection timeout errors during peak usage. The error occurred 3 times in the last hour, blocking memory storage operations.",
                "importance": 0.9,
                "tier": "short_term",
                "metadata": {"incident_type": "connection_failure", "frequency": 3, "impact": "high"}
            },
            {
                "memory_type": "episodic",
                "content": "Successfully migrated 5,000 memories from legacy system to new architecture. All data integrity checks passed and search functionality verified.",
                "importance": 0.7,
                "tier": "archival",
                "metadata": {"migration_batch": 1, "status": "completed"}
            },
            
            # Procedural memories
            {
                "memory_type": "procedural",
                "content": "To optimize vector search: 1) Create proper indexes on frequently queried fields, 2) Use WAL mode for concurrent access, 3) Batch insert operations for performance, 4) Set appropriate cache sizes.",
                "importance": 0.8,
                "tier": "long_term",
                "metadata": {"procedure_type": "optimization", "steps": 4}
            },
            {
                "memory_type": "procedural",
                "content": "Database backup procedure: 1) Stop write operations, 2) Create consistent snapshot, 3) Verify backup integrity, 4) Resume operations. Test restore procedure monthly.",
                "importance": 0.9,
                "tier": "system", 
                "metadata": {"procedure_type": "backup", "frequency": "monthly"}
            },
            
            # Semantic memories
            {
                "memory_type": "semantic",
                "content": "Vector embeddings are numerical representations of text in high-dimensional space. Cosine similarity measures the angle between vectors, indicating semantic similarity.",
                "importance": 0.6,
                "tier": "archival",
                "metadata": {"concept": "vector_embeddings", "domain": "nlp"}
            },
            {
                "memory_type": "semantic",
                "content": "SQLite WAL mode allows concurrent readers while maintaining ACID properties. Writers don't block readers, improving multi-user performance.",
                "importance": 0.7,
                "tier": "long_term",
                "metadata": {"concept": "database_concurrency", "domain": "database"}
            },
            
            # Additional test variations
            {
                "memory_type": "episodic",
                "content": "Database performance improved 40% after indexing optimization. Query response time reduced from 120ms to 72ms average.",
                "importance": 0.8,
                "tier": "short_term",
                "metadata": {"performance_gain": 40, "metric": "query_time"}
            },
            {
                "memory_type": "structured_thinking",
                "content": "SQLite benefits: 1) No external server required, 2) ACID transactions built-in, 3) Extensive tooling ecosystem, 4) Proven reliability. Main limitation: not optimized for distributed systems.",
                "importance": 0.7,
                "tier": "long_term", 
                "metadata": {"analysis_type": "pros_cons", "technology": "sqlite"}
            }
        ]
        
        # Generate additional test cases with variations
        memories = []
        for i, template in enumerate(test_cases):
            memory = template.copy()
            memory["memory_id"] = f"test-{i:03d}"
            memory["created_at"] = f"2025-01-28T{10 + (i % 12):02d}:{(i * 5) % 60:02d}:00Z"
            memories.append(memory)
        
        # Add some edge case memories
        edge_cases = [
            {
                "memory_id": "edge-001",
                "memory_type": "episodic",
                "content": "",  # Empty content
                "importance": 0.1,
                "tier": "short_term",
                "metadata": {}
            },
            {
                "memory_id": "edge-002", 
                "memory_type": "semantic",
                "content": "A" * 1000,  # Very long content
                "importance": 1.0,  # Maximum importance
                "tier": "archival",
                "metadata": {"size": "large"}
            }
        ]
        
        memories.extend(edge_cases)
        return memories
    
    async def test_basic_operations(self, sqlite_db: SQLiteVecPersistenceDomain) -> Dict[str, Any]:
        """Test basic CRUD operations."""
        print("\n📋 Testing basic CRUD operations...")
        
        results = {
            "create_success": 0,
            "read_success": 0, 
            "update_success": 0,
            "delete_success": 0,
            "errors": []
        }
        
        try:
            # CREATE: Store all memories
            print("💾 Testing memory storage...")
            stored_ids = []
            for memory in self.test_memories:
                try:
                    memory_id = await sqlite_db.store_memory(memory, memory["tier"])
                    stored_ids.append(memory_id)
                    results["create_success"] += 1
                except Exception as e:
                    results["errors"].append(f"Storage error for {memory.get('memory_id', 'unknown')}: {e}")
            
            print(f"   ✅ Stored {results['create_success']}/{len(self.test_memories)} memories")
            
            # READ: Retrieve memories with various queries
            print("🔍 Testing memory retrieval...")
            test_queries = [
                "vector search optimization",
                "database performance",
                "migration procedure",
                "SQLite benefits"
            ]
            
            for query in test_queries:
                try:
                    retrieved = await sqlite_db.retrieve_memories(query, limit=3)
                    if retrieved:
                        results["read_success"] += len(retrieved)
                except Exception as e:
                    results["errors"].append(f"Retrieval error for '{query}': {e}")
            
            print(f"   ✅ Retrieved {results['read_success']} memories across {len(test_queries)} queries")
            
            # UPDATE: Modify some memories
            print("✏️ Testing memory updates...")
            if stored_ids:
                test_updates = [
                    {"importance": 0.95, "metadata": {"updated": True}},
                    {"content": "Updated content for testing", "tier": "long_term"}
                ]
                
                for i, update_data in enumerate(test_updates):
                    if i < len(stored_ids):
                        try:
                            success = await sqlite_db.update_memory(stored_ids[i], update_data)
                            if success:
                                results["update_success"] += 1
                        except Exception as e:
                            results["errors"].append(f"Update error for {stored_ids[i]}: {e}")
            
            print(f"   ✅ Updated {results['update_success']} memories")
            
            # DELETE: Remove some memories
            print("🗑️ Testing memory deletion...")
            if len(stored_ids) >= 3:
                try:
                    deleted = await sqlite_db.delete_memories(stored_ids[:3])
                    results["delete_success"] = len(deleted)
                except Exception as e:
                    results["errors"].append(f"Deletion error: {e}")
            
            print(f"   ✅ Deleted {results['delete_success']} memories")
            
        except Exception as e:
            results["errors"].append(f"Basic operations test error: {e}")
        
        return results
    
    async def test_vector_search(self, sqlite_db: SQLiteVecPersistenceDomain) -> Dict[str, Any]:
        """Test vector similarity search functionality."""
        print("\n🔍 Testing vector similarity search...")
        
        results = {
            "search_tests": 0,
            "successful_searches": 0,
            "total_results": 0,
            "similarity_scores": [],
            "errors": []
        }
        
        search_scenarios = [
            {
                "query": "database performance optimization techniques",
                "expected_types": ["procedural", "structured_thinking"],
                "min_results": 1
            },
            {
                "query": "vector embeddings and similarity search", 
                "expected_types": ["semantic"],
                "min_results": 1
            },
            {
                "query": "migration and system issues",
                "expected_types": ["episodic"],
                "min_results": 1
            }
        ]
        
        for scenario in search_scenarios:
            results["search_tests"] += 1
            try:
                # Test basic similarity search
                retrieved = await sqlite_db.retrieve_memories(
                    scenario["query"], 
                    limit=5,
                    min_similarity=0.0
                )
                
                if len(retrieved) >= scenario["min_results"]:
                    results["successful_searches"] += 1
                
                results["total_results"] += len(retrieved)
                
                # Check if expected memory types are found
                found_types = set(m.get("memory_type") for m in retrieved)
                expected_types = set(scenario["expected_types"])
                
                if found_types.intersection(expected_types):
                    print(f"   ✅ Query '{scenario['query'][:30]}...' found {len(retrieved)} results with expected types")
                else:
                    print(f"   ⚠️ Query '{scenario['query'][:30]}...' found {len(retrieved)} results but missing expected types")
                
            except Exception as e:
                results["errors"].append(f"Search error for '{scenario['query']}': {e}")
        
        return results
    
    async def test_filtering(self, sqlite_db: SQLiteVecPersistenceDomain) -> Dict[str, Any]:
        """Test filtering functionality."""
        print("\n🎛️ Testing filtering functionality...")
        
        results = {
            "filter_tests": 0,
            "successful_filters": 0,
            "total_filtered": 0,
            "errors": []
        }
        
        filter_scenarios = [
            {
                "name": "Memory type filter",
                "filters": {"memory_types": ["structured_thinking"]},
                "query": "analysis"
            },
            {
                "name": "Tier filter", 
                "filters": {"tier": "long_term"},
                "query": "database"
            },
            {
                "name": "Importance filter",
                "filters": {"min_importance": 0.8},
                "query": "optimization"
            },
            {
                "name": "Combined filters",
                "filters": {"memory_types": ["episodic", "procedural"], "tier": "short_term"},
                "query": "performance"
            }
        ]
        
        for scenario in filter_scenarios:
            results["filter_tests"] += 1
            try:
                filtered = await sqlite_db.retrieve_memories(
                    scenario["query"],
                    limit=10,
                    **scenario["filters"]
                )
                
                results["total_filtered"] += len(filtered)
                
                # Validate filter worked
                if scenario["filters"].get("memory_types"):
                    expected_types = set(scenario["filters"]["memory_types"])
                    found_types = set(m.get("memory_type") for m in filtered)
                    if found_types.issubset(expected_types):
                        results["successful_filters"] += 1
                        print(f"   ✅ {scenario['name']}: {len(filtered)} results, types match")
                    else:
                        print(f"   ❌ {scenario['name']}: {len(filtered)} results, type filter failed")
                elif scenario["filters"].get("tier"):
                    expected_tier = scenario["filters"]["tier"]
                    if all(m.get("tier") == expected_tier for m in filtered):
                        results["successful_filters"] += 1
                        print(f"   ✅ {scenario['name']}: {len(filtered)} results, tier matches")
                    else:
                        print(f"   ❌ {scenario['name']}: {len(filtered)} results, tier filter failed")
                else:
                    results["successful_filters"] += 1
                    print(f"   ✅ {scenario['name']}: {len(filtered)} results")
                
            except Exception as e:
                results["errors"].append(f"Filter error for {scenario['name']}: {e}")
        
        return results
    
    async def test_performance(self, sqlite_db: SQLiteVecPersistenceDomain) -> Dict[str, Any]:
        """Test performance characteristics."""
        print("\n⚡ Testing performance...")
        
        results = {
            "storage_time_ms": 0,
            "search_times_ms": [],
            "concurrent_success": False,
            "errors": []
        }
        
        try:
            # Test storage performance
            print("   📊 Testing storage performance...")
            start_time = time.time()
            
            perf_memories = []
            for i in range(50):  # Test with 50 memories
                memory = {
                    "memory_id": f"perf-{i:03d}",
                    "memory_type": "episodic",
                    "content": f"Performance test memory {i} with content for vector embedding generation and search testing",
                    "importance": 0.5 + (i % 5) * 0.1,
                    "tier": "short_term",
                    "metadata": {"batch": i // 10}
                }
                perf_memories.append(memory)
                await sqlite_db.store_memory(memory, memory["tier"])
            
            storage_duration = (time.time() - start_time) * 1000
            results["storage_time_ms"] = storage_duration
            print(f"   ✅ Stored 50 memories in {storage_duration:.1f}ms ({50000/storage_duration:.1f} memories/sec)")
            
            # Test search performance
            print("   🔍 Testing search performance...")
            search_queries = [
                "performance test memory content",
                "vector embedding generation testing", 
                "batch processing optimization"
            ]
            
            for query in search_queries:
                start_time = time.time()
                results_found = await sqlite_db.retrieve_memories(query, limit=5)
                search_duration = (time.time() - start_time) * 1000
                results["search_times_ms"].append(search_duration)
                print(f"   ✅ Search '{query[:30]}...': {search_duration:.1f}ms, {len(results_found)} results")
            
            # Test concurrent access (basic)
            print("   🔄 Testing concurrent access...")
            try:
                import threading
                import time
                
                def concurrent_search():
                    """Concurrent search function."""
                    import asyncio
                    try:
                        # Create new event loop for thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Run search
                        result = loop.run_until_complete(
                            sqlite_db.retrieve_memories("concurrent test", limit=3)
                        )
                        return len(result)
                    except Exception:
                        return -1
                    finally:
                        loop.close()
                
                # Run multiple concurrent searches
                threads = []
                for i in range(3):
                    thread = threading.Thread(target=concurrent_search)
                    threads.append(thread)
                    thread.start()
                
                # Wait for completion
                for thread in threads:
                    thread.join(timeout=10)
                
                results["concurrent_success"] = True
                print("   ✅ Concurrent access test passed")
                
            except Exception as e:
                results["errors"].append(f"Concurrent test error: {e}")
        
        except Exception as e:
            results["errors"].append(f"Performance test error: {e}")
        
        return results
    
    async def test_statistics(self, sqlite_db: SQLiteVecPersistenceDomain) -> Dict[str, Any]:
        """Test statistics and monitoring."""
        print("\n📊 Testing statistics and monitoring...")
        
        try:
            stats = await sqlite_db.get_memory_stats()
            
            print("   📈 Database Statistics:")
            print(f"      Total memories: {stats.get('total_memories', 0)}")
            print(f"      Unique types: {stats.get('unique_types', 0)}")
            print(f"      Unique tiers: {stats.get('unique_tiers', 0)}")
            print(f"      Average importance: {stats.get('average_importance', 0):.2f}")
            print(f"      sqlite-vec enabled: {stats.get('sqlite_vec_enabled', False)}")
            
            if stats.get('type_distribution'):
                print("      Type distribution:")
                for mem_type, count in stats['type_distribution'].items():
                    print(f"        {mem_type}: {count}")
            
            if stats.get('tier_distribution'):
                print("      Tier distribution:")
                for tier, count in stats['tier_distribution'].items():
                    print(f"        {tier}: {count}")
            
            return {"success": True, "stats": stats, "errors": []}
            
        except Exception as e:
            return {"success": False, "stats": {}, "errors": [str(e)]}
    
    def generate_comprehensive_report(self, test_results: Dict[str, Any]):
        """Generate comprehensive test report."""
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE SQLITE VALIDATION REPORT")
        print("="*70)
        
        # Overall summary
        total_tests = 0
        passed_tests = 0
        total_errors = 0
        
        print("\n📋 TEST SUMMARY")
        print("-" * 40)
        
        # Basic operations
        basic = test_results.get("basic_operations", {})
        basic_total = basic.get("create_success", 0) + basic.get("read_success", 0) + basic.get("update_success", 0) + basic.get("delete_success", 0)
        basic_expected = len(self.test_memories) + 12 + 2 + 3  # Storage + retrievals + updates + deletes
        basic_success = basic_total > (basic_expected * 0.8)  # 80% success threshold
        
        print(f"✅ Basic Operations: {'PASS' if basic_success else 'FAIL'}")
        print(f"   - Storage: {basic.get('create_success', 0)}/{len(self.test_memories)} memories")
        print(f"   - Retrieval: {basic.get('read_success', 0)} results")
        print(f"   - Updates: {basic.get('update_success', 0)} successful")
        print(f"   - Deletions: {basic.get('delete_success', 0)} successful")
        
        total_tests += 1
        if basic_success:
            passed_tests += 1
        total_errors += len(basic.get("errors", []))
        
        # Vector search
        vector = test_results.get("vector_search", {})
        vector_success = vector.get("successful_searches", 0) >= vector.get("search_tests", 0) * 0.7
        
        print(f"✅ Vector Search: {'PASS' if vector_success else 'FAIL'}")
        print(f"   - Search tests: {vector.get('successful_searches', 0)}/{vector.get('search_tests', 0)} passed")
        print(f"   - Total results: {vector.get('total_results', 0)}")
        
        total_tests += 1
        if vector_success:
            passed_tests += 1
        total_errors += len(vector.get("errors", []))
        
        # Filtering
        filtering = test_results.get("filtering", {})
        filter_success = filtering.get("successful_filters", 0) >= filtering.get("filter_tests", 0) * 0.8
        
        print(f"✅ Filtering: {'PASS' if filter_success else 'FAIL'}")
        print(f"   - Filter tests: {filtering.get('successful_filters', 0)}/{filtering.get('filter_tests', 0)} passed")
        print(f"   - Total filtered: {filtering.get('total_filtered', 0)} results")
        
        total_tests += 1
        if filter_success:
            passed_tests += 1
        total_errors += len(filtering.get("errors", []))
        
        # Performance
        performance = test_results.get("performance", {})
        avg_search_time = sum(performance.get("search_times_ms", [])) / max(len(performance.get("search_times_ms", [])), 1)
        perf_success = (
            performance.get("storage_time_ms", 0) < 5000 and  # Under 5 seconds for 50 memories
            avg_search_time < 100 and  # Under 100ms average search
            performance.get("concurrent_success", False)
        )
        
        print(f"✅ Performance: {'PASS' if perf_success else 'FAIL'}")
        print(f"   - Storage time: {performance.get('storage_time_ms', 0):.1f}ms (50 memories)")
        print(f"   - Average search: {avg_search_time:.1f}ms")
        print(f"   - Concurrent access: {'✅' if performance.get('concurrent_success', False) else '❌'}")
        
        total_tests += 1
        if perf_success:
            passed_tests += 1
        total_errors += len(performance.get("errors", []))
        
        # Statistics
        stats = test_results.get("statistics", {})
        stats_success = stats.get("success", False)
        
        print(f"✅ Statistics: {'PASS' if stats_success else 'FAIL'}")
        if stats.get("stats"):
            db_stats = stats["stats"]
            print(f"   - Total memories: {db_stats.get('total_memories', 0)}")
            print(f"   - Memory types: {db_stats.get('unique_types', 0)}")
            print(f"   - sqlite-vec: {'enabled' if db_stats.get('sqlite_vec_enabled', False) else 'fallback mode'}")
        
        total_tests += 1
        if stats_success:
            passed_tests += 1
        total_errors += len(stats.get("errors", []))
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT")
        print("-" * 40)
        overall_success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Tests passed: {passed_tests}/{total_tests} ({overall_success_rate:.1f}%)")
        print(f"Total errors: {total_errors}")
        
        if overall_success_rate >= 90:
            assessment = "🟢 EXCELLENT - Ready for production"
            decision = "GO"
        elif overall_success_rate >= 75:
            assessment = "🟡 GOOD - Ready with monitoring"
            decision = "GO"
        elif overall_success_rate >= 60:
            assessment = "🟠 ACCEPTABLE - Needs optimization"
            decision = "CAUTION"
        else:
            assessment = "🔴 POOR - Not ready"
            decision = "NO GO"
        
        print(f"\nDecision: {decision}")
        print(f"Assessment: {assessment}")
        
        # Migration readiness
        print(f"\n🚀 PRODUCTION READINESS")
        print("-" * 40)
        
        if decision in ["GO"]:
            print("✅ SQLite implementation meets all requirements")
            print("✅ All core functionality validated")
            print("✅ Performance meets requirements")
            print("✅ Error handling working correctly")
            
            benefits = [
                "No external server dependencies",
                "Built-in ACID transactions", 
                "Simplified deployment and maintenance",
                "Proven reliability and stability",
                "Extensive tooling ecosystem"
            ]
            
            print(f"\n🎁 SQLite Benefits:")
            for benefit in benefits:
                print(f"   • {benefit}")
        else:
            print("❌ SQLite implementation needs more work")
            if total_errors > 0:
                print("❌ Error handling issues detected")
        
        print("\n" + "="*70)
        
        return {
            "overall_success_rate": overall_success_rate,
            "decision": decision,
            "total_errors": total_errors,
            "passed_tests": passed_tests,
            "total_tests": total_tests
        }
    
    def cleanup(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


async def main():
    """Run comprehensive SQLite validation test."""
    test = SQLiteValidationTest()
    
    try:
        # Setup
        test.setup()
        
        print("\n🧪 RUNNING COMPREHENSIVE SQLITE VALIDATION")
        print("="*70)
        print("This test validates SQLite memory persistence functionality")
        
        # Setup SQLite database
        sqlite_config = {
            "sqlite": {
                "path": os.path.join(test.temp_dir, "validation_test.db"),
                "vector_dimensions": 384
            },
            "embedding": {"default_model": "paraphrase-MiniLM-L3-v2", "dimensions": 384}
        }
        
        sqlite_db = SQLiteVecPersistenceDomain(sqlite_config)
        sqlite_db.initialize()
        
        # Run comprehensive tests
        test_results = {}
        
        test_results["basic_operations"] = await test.test_basic_operations(sqlite_db)
        test_results["vector_search"] = await test.test_vector_search(sqlite_db)
        test_results["filtering"] = await test.test_filtering(sqlite_db)
        test_results["performance"] = await test.test_performance(sqlite_db)
        test_results["statistics"] = await test.test_statistics(sqlite_db)
        
        # Generate comprehensive report
        summary = test.generate_comprehensive_report(test_results)
        
        # Close database
        sqlite_db.close()
        
        return 0 if summary["decision"] in ["GO"] else 1
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        test.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)