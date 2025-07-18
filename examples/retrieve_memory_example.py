#!/usr/bin/env python3
"""
Example script showing how to retrieve memories using the Memory MCP Server API.
"""

import json
import asyncio
import argparse
import sys
import os
import subprocess
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


async def retrieve_memory_example(query: str, limit: int = 5, memory_types: List[str] = None, 
                                 min_similarity: float = 0.6) -> None:
    """
    Example of retrieving memories using subprocess to communicate with the MCP server.
    
    Args:
        query: Query string to search for memories
        limit: Maximum number of memories to retrieve
        memory_types: Types of memories to include (None for all types)
        min_similarity: Minimum similarity score for results
    """
    # Construct the request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "executeFunction",
        "params": {
            "name": "retrieve_memory",
            "arguments": {
                "query": query,
                "limit": limit,
                "types": memory_types,
                "min_similarity": min_similarity,
                "include_metadata": True
            }
        }
    }
    
    # Convert to JSON
    request_json = json.dumps(request)
    
    # Execute MCP server process
    process = subprocess.Popen(
        ["python", "-m", "memory_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send request
    stdout, stderr = process.communicate(input=request_json + "\n")
    
    # Parse response
    try:
        response = json.loads(stdout)
        if "result" in response and "value" in response["result"]:
            result = json.loads(response["result"]["value"][0]["text"])
            if result.get("success"):
                memories = result.get("memories", [])
                if not memories:
                    print(f"No memories found for query: '{query}'")
                else:
                    print(f"Found {len(memories)} memories for query: '{query}'")
                    for i, memory in enumerate(memories):
                        print(f"\nMemory {i+1}:")
                        print(f"  Type: {memory['type']}")
                        print(f"  Similarity: {memory.get('similarity', 0.0):.2f}")
                        
                        if memory["type"] == "fact":
                            print(f"  Fact: {memory['content'].get('fact', 'N/A')}")
                        elif memory["type"] == "entity":
                            print(f"  Entity: {memory['content'].get('name', 'N/A')}")
                            print(f"  Attributes: {memory['content'].get('attributes', {})}")
                        elif memory["type"] == "conversation":
                            print(f"  Role: {memory['content'].get('role', 'N/A')}")
                            print(f"  Message: {memory['content'].get('message', 'N/A')}")
                        
                        if "metadata" in memory:
                            print(f"  Created: {memory.get('created_at', 'N/A')}")
                            print(f"  Last Accessed: {memory.get('last_accessed', 'N/A')}")
                            print(f"  Importance: {memory.get('importance', 0.0)}")
            else:
                print(f"Error retrieving memories: {result.get('error')}")
        else:
            print(f"Unexpected response: {response}")
    except json.JSONDecodeError:
        print(f"Error parsing response: {stdout}")
        print(f"Error output: {stderr}")


def main() -> None:
    """Main function for the example script."""
    parser = argparse.ArgumentParser(description="Memory MCP Retrieve Example")
    parser.add_argument("--query", default="user preferences", help="Query string to search for memories")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of memories to retrieve")
    parser.add_argument("--types", nargs="+", choices=["conversation", "fact", "entity", "reflection", "code"], 
                      help="Types of memories to include")
    parser.add_argument("--min-similarity", type=float, default=0.6, help="Minimum similarity score (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Run the example
    asyncio.run(retrieve_memory_example(args.query, args.limit, args.types, args.min_similarity))


if __name__ == "__main__":
    main()