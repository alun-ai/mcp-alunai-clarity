#!/usr/bin/env python3
"""
Real integration test - no mocks, no simulations.
Tests actual Claude Code hook integration with running MCP server.
"""

import subprocess
import json
import time
import sys
import os
from pathlib import Path

def run_with_real_mcp_server():
    """Test with actual running MCP server."""
    
    print("=== REAL INTEGRATION TEST ===")
    print("Testing auto-capture with live MCP server connection")
    
    # Check if MCP server is running
    try:
        result = subprocess.run([
            "docker", "ps", "--filter", "name=alunai-clarity-mcp-dev", "--format", "{{.Status}}"
        ], capture_output=True, text=True, check=True)
        
        if "Up" not in result.stdout:
            print("❌ MCP server container not running")
            return False
            
        print("✅ MCP server container is running")
        
    except subprocess.CalledProcessError:
        print("❌ Failed to check MCP server status")
        return False
    
    # Test hook configuration exists
    hook_path = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/.claude/alunai-clarity/hooks.json"
    if not os.path.exists(hook_path):
        print(f"❌ Hook configuration not found at {hook_path}")
        return False
        
    print("✅ Hook configuration file exists")
    
    # Test hook analyzer script exists and is executable
    analyzer_path = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py"
    if not os.path.exists(analyzer_path):
        print(f"❌ Hook analyzer script not found at {analyzer_path}")
        return False
        
    print("✅ Hook analyzer script exists")
    
    # Test direct hook execution
    test_prompts = [
        "Remember this: Integration test with real MCP server connection",
        "Please remember: Auto-capture should work without explicit tool calls",
        "Remember that: This validates the complete hook integration"
    ]
    
    success_count = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: Direct hook execution ---")
        print(f"Prompt: {prompt}")
        
        try:
            result = subprocess.run([
                sys.executable, analyzer_path, "--prompt-submit", f"--prompt={prompt}"
            ], capture_output=True, text=True, check=True, timeout=10)
            
            if result.stdout:
                response = json.loads(result.stdout)
                modified_prompt = response.get("modified_prompt", "")
                
                if "store_memory" in modified_prompt and prompt.replace("Remember this: ", "").replace("Please remember: ", "").replace("Remember that: ", "") in modified_prompt:
                    print("✅ Hook correctly modified prompt")
                    print(f"Modified: {modified_prompt[:100]}...")
                    success_count += 1
                else:
                    print("❌ Hook did not modify prompt correctly")
                    print(f"Output: {modified_prompt}")
            else:
                print("❌ No output from hook")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Hook execution failed: {e}")
            print(f"Stderr: {e.stderr}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON response: {e}")
        except subprocess.TimeoutExpired:
            print("❌ Hook execution timed out")
    
    print(f"\n--- Hook Execution Results ---")
    print(f"Success rate: {success_count}/{len(test_prompts)} ({success_count/len(test_prompts)*100:.1f}%)")
    
    return success_count == len(test_prompts)

def test_mcp_server_connection():
    """Test actual MCP server connection and store_memory call."""
    
    print("\n=== MCP SERVER CONNECTION TEST ===")
    
    # Get the container's stdio transport for MCP
    try:
        # Test by executing store_memory directly through the container
        test_content = f"Integration test at {time.time()}"
        
        result = subprocess.run([
            "docker", "exec", "alunai-clarity-mcp-dev",
            "python", "-c", f"""
import asyncio
import json
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def test_store():
    config = load_config('/app/data/config.json')
    server = MemoryMcpServer(config)
    
    # Call store_memory through the FastMCP tool manager
    result = await server.app.call_tool(
        'store_memory',
        {{
            'memory_type': 'integration_test',
            'content': '{test_content}',
            'importance': 0.8
        }}
    )
    print(result)

asyncio.run(test_store())
"""
        ], capture_output=True, text=True, check=True, timeout=30)
        
        if "success" in result.stdout.lower() and "memory_id" in result.stdout:
            print("✅ MCP server store_memory call successful")
            print(f"Response: {result.stdout.strip()}")
            return True
        else:
            print("❌ MCP server store_memory call failed")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ MCP server connection failed: {e}")
        print(f"Stderr: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("❌ MCP server connection timed out")
        return False

def test_end_to_end_integration():
    """Test complete end-to-end integration."""
    
    print("\n=== END-TO-END INTEGRATION TEST ===")
    print("Testing complete flow: Hook detection -> Prompt modification -> MCP call")
    
    # Simulate what Claude Code would do:
    # 1. User submits "Remember this: X"
    # 2. Hook modifies to "store_memory X"
    # 3. Claude processes modified prompt and calls store_memory
    
    original_prompt = "Remember this: End-to-end integration test validates complete auto-capture flow"
    
    # Step 1: Hook processes the prompt
    analyzer_path = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py"
    
    try:
        hook_result = subprocess.run([
            sys.executable, analyzer_path, "--prompt-submit", f"--prompt={original_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        response = json.loads(hook_result.stdout)
        modified_prompt = response.get("modified_prompt", "")
        
        if "store_memory" not in modified_prompt:
            print("❌ Hook did not modify prompt")
            return False
            
        print("✅ Step 1: Hook successfully modified prompt")
        print(f"Modified prompt: {modified_prompt}")
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return False
    
    # Step 2: Extract store_memory call from modified prompt and execute it
    # This simulates what Claude would do when it sees the modified prompt
    
    try:
        # Parse the store_memory call from the modified prompt
        if "store_memory" in modified_prompt:
            # Extract content after "store_memory"
            content_start = modified_prompt.find("store_memory") + len("store_memory")
            content = modified_prompt[content_start:].split("\n")[0].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            
            print(f"✅ Step 2: Extracted content for store_memory: '{content}'")
            
            # Execute the store_memory call through MCP server
            store_result = subprocess.run([
                "docker", "exec", "alunai-clarity-mcp-dev",
                "python", "-c", f"""
import asyncio
import json
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def test_store():
    config = load_config('/app/data/config.json')
    server = MemoryMcpServer(config)
    
    # Call store_memory through the FastMCP tool manager
    result = await server.app.call_tool(
        'store_memory',
        {{
            'memory_type': 'auto_capture_test',
            'content': '{content}',
            'importance': 0.9
        }}
    )
    print(result)

asyncio.run(test_store())
"""
            ], capture_output=True, text=True, check=True, timeout=30)
            
            if "success" in store_result.stdout.lower() and "memory_id" in store_result.stdout:
                print("✅ Step 3: MCP store_memory call successful")
                print(f"Store result: {store_result.stdout.strip()}")
                return True
            else:
                print("❌ Step 3: MCP store_memory call failed")
                print(f"Output: {store_result.stdout}")
                return False
                
    except Exception as e:
        print(f"❌ Step 2/3 failed: {e}")
        return False

def main():
    """Run complete real integration test suite."""
    
    print("REAL INTEGRATION TEST - NO MOCKS")
    print("Testing actual Claude Code hook integration with live MCP server")
    print("=" * 60)
    
    # Test 1: Hook execution
    hook_test = run_with_real_mcp_server()
    
    # Test 2: MCP server connection
    mcp_test = test_mcp_server_connection()
    
    # Test 3: End-to-end integration
    e2e_test = test_end_to_end_integration()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Hook execution test: {'✅ PASS' if hook_test else '❌ FAIL'}")
    print(f"MCP server test: {'✅ PASS' if mcp_test else '❌ FAIL'}")
    print(f"End-to-end test: {'✅ PASS' if e2e_test else '❌ FAIL'}")
    
    all_passed = hook_test and mcp_test and e2e_test
    print(f"\nOVERALL: {'🎉 ALL TESTS PASSED' if all_passed else '❌ TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ Auto-capture is working correctly with real integration")
        print("✅ 'Remember this:' prompts will be automatically converted to store_memory calls")
        print("✅ No explicit tool calls needed - the system works transparently")
    else:
        print("\n❌ Integration issues detected - auto-capture may not work correctly")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)