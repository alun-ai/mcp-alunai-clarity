"""
Focused Large Dataset Test for MCP Integration.

A simplified version of the large dataset test to debug issues.
"""

import pytest
import asyncio
import time
from tests.framework.mcp_validation import MCPServerTestSuite
from tests.integration.test_mcp_large_dataset_validation import LargeDatasetGenerator


@pytest.mark.asyncio
async def test_focused_large_dataset():
    """Focused test with smaller dataset to debug issues."""
    suite = MCPServerTestSuite()
    await suite.setup_test_environment()
    
    try:
        # Start with a larger dataset for comprehensive testing
        dataset_size = 150
        print(f"\n🏗️ Creating focused dataset with {dataset_size} memories...")
        
        # Generate dataset
        generator = LargeDatasetGenerator()
        dataset = generator.generate_large_dataset(dataset_size)
        
        # Track performance metrics
        start_time = time.perf_counter()
        store_times = []
        stored_memory_ids = []
        
        # Store memories one by one
        for i, memory_data in enumerate(dataset):
            store_start = time.perf_counter()
            
            result = await suite.validate_mcp_tool_execution(
                tool_name="store_memory",
                arguments=memory_data,
                validate_underlying_data=False,
                test_name=f"focused_store_{i}"
            )
            
            store_time = (time.perf_counter() - store_start) * 1000
            store_times.append(store_time)
            
            if result.passed:
                memory_id = result.parsed_response["memory_id"]
                stored_memory_ids.append(memory_id)
                if (i + 1) % 25 == 0:
                    print(f"   📊 Progress: {i+1}/{dataset_size} memories stored")
            else:
                print(f"   ❌ Failed to store memory {i+1}: {result.errors}")
                assert False, f"Memory store {i} failed: {result.errors}"
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Get collection stats
        collection_stats = await suite.qdrant_inspector.get_collection_stats()
        
        # Calculate performance metrics
        avg_store_time = sum(store_times) / len(store_times)
        
        print(f"\n📈 Focused Dataset Results:")
        print(f"   ✅ Total memories stored: {len(stored_memory_ids)}")
        print(f"   ⏱️ Total time: {total_time:.1f}ms ({total_time/1000:.2f}s)")
        print(f"   ⚡ Average store time: {avg_store_time:.1f}ms")
        print(f"   📦 Qdrant total points: {collection_stats['total_points']}")
        print(f"   🔍 Indexed vectors: {collection_stats['indexed_vectors']}")
        print(f"   🏗️ Collection status: {collection_stats['collection_status']}")
        
        # Validate results
        assert len(stored_memory_ids) == dataset_size
        assert collection_stats["total_points"] >= dataset_size
        assert avg_store_time < 2000, f"Average store time too slow: {avg_store_time}ms"
        
        print(f"   🎯 Focused dataset test successful!")
        
        # Test basic retrieval
        print(f"\n🔍 Testing retrieval...")
        try:
            retrieve_result = await suite.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": "programming",
                    "limit": 5,
                    "min_similarity": 0.3,
                    "include_metadata": True
                },
                test_name="focused_retrieval_test"
            )
            
            if retrieve_result.passed:
                memories = retrieve_result.parsed_response.get("memories", [])
                print(f"   📊 Retrieved {len(memories)} memories")
                retrieval_success = True
            else:
                print(f"   ❌ Retrieval failed: {retrieve_result.errors}")
                retrieval_success = False
        except Exception as e:
            print(f"   ⚠️ Retrieval test encountered error: {e}")
            retrieval_success = False
        
        return {
            "stored_memories": len(stored_memory_ids),
            "total_time_ms": total_time,
            "avg_store_time_ms": avg_store_time,
            "collection_stats": collection_stats,
            "retrieval_success": retrieval_success
        }
        
    finally:
        await suite.teardown_test_environment()


if __name__ == "__main__":
    # Run directly
    result = asyncio.run(test_focused_large_dataset())
    print(f"\n🎉 Test completed: {result}")