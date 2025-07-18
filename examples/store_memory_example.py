#!/usr/bin/env python3
"""
Example script showing how to store a memory using the Memory MCP Server API.
"""

import json
import asyncio
import argparse
import sys
import os
import subprocess
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


async def store_memory_example(memory_type: str, content: Dict[str, Any], importance: float) -> None:
    """
    Example of storing a memory using subprocess to communicate with the MCP server.
    
    Args:
        memory_type: Type of memory (conversation, fact, entity, etc.)
        content: Memory content as a dictionary
        importance: Importance score (0.0-1.0)
    """
    # Construct the request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "executeFunction",
        "params": {
            "name": "store_memory",
            "arguments": {
                "type": memory_type,
                "content": content,
                "importance": importance
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
                print(f"Memory stored successfully with ID: {result.get('memory_id')}")
            else:
                print(f"Error storing memory: {result.get('error')}")
        else:
            print(f"Unexpected response: {response}")
    except json.JSONDecodeError:
        print(f"Error parsing response: {stdout}")
        print(f"Error output: {stderr}")


def main() -> None:
    """Main function for the example script."""
    parser = argparse.ArgumentParser(description="Memory MCP Store Example")
    parser.add_argument("--type", choices=["conversation", "fact", "entity", "reflection", "code"], default="fact")
    parser.add_argument("--content", help="Content string for the memory")
    parser.add_argument("--importance", type=float, default=0.7, help="Importance score (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Construct memory content based on type
    if args.type == "fact":
        content = {
            "fact": args.content or "Paris is the capital of France",
            "confidence": 0.95,
            "domain": "geography"
        }
    elif args.type == "entity":
        content = {
            "name": "user",
            "entity_type": "person",
            "attributes": {
                "preference": args.content or "Python programming language"
            }
        }
    elif args.type == "conversation":
        content = {
            "role": "user",
            "message": args.content or "I really enjoy machine learning and data science."
        }
    elif args.type == "reflection":
        content = {
            "subject": "user preferences",
            "reflection": args.content or "The user seems to prefer technical discussions about AI and programming."
        }
    elif args.type == "code":
        content = {
            "language": "python",
            "code": args.content or "print('Hello, world!')",
            "description": "Simple hello world program"
        }
    
    # Run the example
    asyncio.run(store_memory_example(args.type, content, args.importance))


if __name__ == "__main__":
    main()