"""
More detailed debugging of retrieve_memory issue.
"""

import asyncio
import traceback
from tests.framework.mcp_validation import MCPServerTestSuite


async def debug_retrieve_detailed():
    """Debug each step of the retrieve process."""
    print("🔍 Detailed debugging of retrieve_memory...")
    
    suite = MCPServerTestSuite()
    await suite.setup_test_environment()
    
    try:
        # First store a test memory
        print("\n1️⃣ Storing test memory...")
        store_result = await suite.validate_mcp_tool_execution(
            tool_name="store_memory",
            arguments={
                "memory_type": "debug_test",
                "content": "This is debug content for retrieval testing",
                "importance": 0.8,
                "metadata": {"test_key": "test_value"}
            },
            validate_underlying_data=False,
            test_name="debug_store"
        )
        
        if store_result.passed:
            memory_id = store_result.parsed_response["memory_id"]
            print(f"   ✅ Stored memory: {memory_id}")
        else:
            print(f"   ❌ Store failed: {store_result.errors}")
            return
        
        # Test each step of retrieval manually
        print(f"\n2️⃣ Testing domain_manager.retrieve_memories directly...")
        
        try:
            memories = await suite.mcp_server.domain_manager.retrieve_memories(
                query="debug content testing",
                limit=5,
                memory_types=None,
                min_similarity=0.3,
                include_metadata=True
            )
            print(f"   ✅ Domain manager returned: {len(memories)} memories")
            for i, mem in enumerate(memories):
                print(f"      Memory {i}: {list(mem.keys())}")
        except Exception as e:
            print(f"   ❌ Domain manager error: {e}")
            print(f"   📋 Traceback:")
            traceback.print_exc()
            return
            
        # Test MCPResponseBuilder
        print(f"\n3️⃣ Testing MCPResponseBuilder...")
        
        try:
            from clarity.shared.utils import MCPResponseBuilder
            response = MCPResponseBuilder.memories_retrieved(memories)
            print(f"   ✅ MCPResponseBuilder response: {response[:200]}...")
        except Exception as e:
            print(f"   ❌ MCPResponseBuilder error: {e}")
            traceback.print_exc()
            
    finally:
        await suite.teardown_test_environment()


if __name__ == "__main__":
    asyncio.run(debug_retrieve_detailed())