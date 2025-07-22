"""
Debug script to understand the retrieve memory response format issue.
"""

import asyncio
import json
from tests.framework.mcp_validation import MCPServerTestSuite


async def debug_retrieve_response():
    """Debug the actual vs expected response format for retrieve_memory."""
    print("🔍 Debugging retrieve_memory response format...")
    
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
        
        # Now test retrieve_memory directly
        print(f"\n2️⃣ Testing retrieve_memory directly...")
        
        # Call the tool method directly to see raw response
        raw_response = await suite._call_retrieve_memory_tool(
            query="debug content testing",
            limit=5,
            min_similarity=0.3,
            include_metadata=True
        )
        
        print(f"📄 Raw MCP Response:")
        print(f"   Type: {type(raw_response)}")
        print(f"   Content: {raw_response}")
        
        # Try to parse it
        try:
            parsed = json.loads(raw_response)
            print(f"\n📋 Parsed Response Structure:")
            for key, value in parsed.items():
                print(f"   {key}: {type(value)} = {value if len(str(value)) < 100 else str(value)[:100] + '...'}")
        except Exception as e:
            print(f"   ❌ JSON Parse Error: {e}")
        
        # Now test with MCP validation
        print(f"\n3️⃣ Testing with MCP validation...")
        retrieve_result = await suite.validate_mcp_tool_execution(
            tool_name="retrieve_memory",
            arguments={
                "query": "debug content testing", 
                "limit": 5,
                "min_similarity": 0.3,
                "include_metadata": True
            },
            validate_underlying_data=False,
            test_name="debug_retrieve"
        )
        
        print(f"🔍 Validation Result:")
        print(f"   Passed: {retrieve_result.passed}")
        print(f"   Errors: {retrieve_result.errors}")
        if retrieve_result.response_validation:
            print(f"   Response Validation: {retrieve_result.response_validation}")
        
        # Check expected schema
        print(f"\n📏 Expected Schema for retrieve_memory:")
        expected_schema = suite.protocol_validator.response_schemas.get("retrieve_memory")
        print(f"   {expected_schema}")
        
    finally:
        await suite.teardown_test_environment()


if __name__ == "__main__":
    asyncio.run(debug_retrieve_response())