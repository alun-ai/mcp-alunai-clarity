"""
Simple MCP test to debug issues.
"""

import pytest
import asyncio
from tests.framework.mcp_validation import MCPServerTestSuite


@pytest.mark.asyncio
async def test_simple_mcp_store():
    """Simple test of MCP store functionality."""
    suite = MCPServerTestSuite()
    await suite.setup_test_environment()
    
    try:
        # Simple store test
        result = await suite.validate_mcp_tool_execution(
            tool_name="store_memory",
            arguments={
                "memory_type": "simple_test",
                "content": "Simple test content",
                "importance": 0.5
            },
            validate_underlying_data=False,  # Skip complex validation first
            test_name="simple_store_test"
        )
        
        print(f"Result passed: {result.passed}")
        print(f"Errors: {result.errors}")
        print(f"Response: {result.mcp_response}")
        
        assert result.passed, f"Simple store test failed: {result.errors}"
        
    finally:
        await suite.teardown_test_environment()


if __name__ == "__main__":
    # Run directly
    asyncio.run(test_simple_mcp_store())