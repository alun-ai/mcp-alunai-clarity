#!/usr/bin/env python3
"""
Test script to simulate MCP interaction with production Docker image
to reproduce the store_memory failure.
"""

import json
import subprocess
import time
import sys

def test_mcp_interaction():
    """Test MCP server interaction step by step."""
    
    print("Testing MCP server initialization and tool calls...")
    
    # Create the docker command
    docker_cmd = [
        "docker", "run", "-i", "--rm",
        "-v", f"{sys.argv[1] if len(sys.argv) > 1 else './test-data'}:/app/data",
        "mcp-alunai-clarity-fixed:test"
    ]
    
    print(f"Docker command: {' '.join(docker_cmd)}")
    
    # Start the docker container
    process = subprocess.Popen(
        docker_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    try:
        # Step 1: Send initialize request
        print("\n1. Sending initialize request...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-01-07",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        print(f"Initialize response: {response_line.strip()}")
        
        # Step 2: Send initialized notification
        print("\n2. Sending initialized notification...")
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        process.stdin.write(json.dumps(initialized_notif) + "\n")
        process.stdin.flush()
        
        # Wait a bit for initialization to complete
        time.sleep(2)
        
        # Step 3: List tools to see what's available
        print("\n3. Listing available tools...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        print(f"Tools list response: {response_line.strip()}")
        
        # Step 4: Call store_memory
        print("\n4. Calling store_memory tool...")
        store_memory_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "store_memory",
                "arguments": {
                    "memory_type": "integration_test",
                    "content": "Testing memory storage after proper MCP initialization",
                    "importance": 0.8
                }
            }
        }
        
        process.stdin.write(json.dumps(store_memory_request) + "\n")
        process.stdin.flush()
        
        # Read response with timeout
        response_line = process.stdout.readline()
        print(f"Store memory response: {response_line.strip()}")
        
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Clean up
        process.stdin.close()
        process.terminate()
        process.wait()
        
        # Print any stderr output
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"\nStderr output:\n{stderr_output}")

if __name__ == "__main__":
    test_mcp_interaction()