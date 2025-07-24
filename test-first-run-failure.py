#!/usr/bin/env python3
"""
Test first-run auto-capture failure - exactly replicating user logs.
Tests auto-capture on completely fresh startup before any explicit store_memory calls.
"""

import subprocess
import json
import time
import sys
import os

def test_fresh_startup_auto_capture():
    """Test auto-capture on completely fresh MCP server startup."""
    
    print("=== TESTING FIRST-RUN AUTO-CAPTURE FAILURE ===")
    print("Replicating exact scenario from user logs")
    
    # Test the hook analyzer directly (simulates what Claude Code would do)
    test_prompt = "Remember this: ## MANDATORY: DIRECT INTEGRATION ONLY"
    
    print(f"\n--- Testing Fresh Auto-Capture ---")
    print(f"Prompt: {test_prompt[:50]}...")
    
    try:
        # Test hook analyzer (this should work - pattern detection)
        result = subprocess.run([
            sys.executable, "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py", 
            "--prompt-submit", f"--prompt={test_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        if result.stdout:
            response = json.loads(result.stdout)
            modified_prompt = response.get("modified_prompt", "")
            
            if "store_memory" in modified_prompt:
                print("✅ Hook correctly detected pattern and modified prompt")
                print(f"Modified: {modified_prompt[:100]}...")
                
                # Now test if the MCP server can handle the store_memory call
                # This simulates what Claude would do when it sees the modified prompt
                print("\n--- Testing MCP Server Call (Fresh State) ---")
                
                # Extract content for store_memory call
                content_start = modified_prompt.find("store_memory") + len("store_memory")
                content = modified_prompt[content_start:].split("\n")[0].strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                
                # Make the actual MCP call through the container
                mcp_result = subprocess.run([
                    "docker", "exec", "alunai-clarity-mcp-dev",
                    "python", "-c", f"""
import asyncio
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def test_fresh_store():
    try:
        config = load_config('/app/data/config.json')
        server = MemoryMcpServer(config)
        
        # This should simulate what happens when Claude processes the modified prompt
        result = await server.app.call_tool(
            'store_memory',
            {{
                'memory_type': 'fresh_test',
                'content': '{content[:100]}...',
                'importance': 0.9
            }}
        )
        
        print("SUCCESS:", result)
        
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()

asyncio.run(test_fresh_store())
"""
                ], capture_output=True, text=True, timeout=30)
                
                print(f"MCP Result Stdout: {mcp_result.stdout}")
                print(f"MCP Result Stderr: {mcp_result.stderr}")
                print(f"MCP Return Code: {mcp_result.returncode}")
                
                if mcp_result.returncode == 0 and "SUCCESS" in mcp_result.stdout:
                    print("✅ UNEXPECTED: Auto-capture worked on first run!")
                    return True
                else:
                    print("❌ EXPECTED: Auto-capture failed on first run")
                    print("This confirms the lazy initialization issue")
                    return False
                    
            else:
                print("❌ Hook failed to detect pattern")
                return False
        else:
            print("❌ No output from hook")
            return False
            
    except Exception as e:
        print(f"❌ Hook execution failed: {e}")
        return False

def test_after_explicit_store():
    """Test auto-capture after explicit store_memory call (should work)."""
    
    print("\n=== TESTING AUTO-CAPTURE AFTER EXPLICIT STORE ===")
    
    # First, make an explicit store_memory call to initialize domains
    print("--- Making Explicit store_memory Call ---")
    
    explicit_result = subprocess.run([
        "docker", "exec", "alunai-clarity-mcp-dev",
        "python", "-c", """
import asyncio
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def explicit_store():
    try:
        config = load_config('/app/data/config.json')
        server = MemoryMcpServer(config)
        
        result = await server.app.call_tool(
            'store_memory',
            {
                'memory_type': 'explicit_init',
                'content': 'Explicit call to initialize domains',
                'importance': 0.8
            }
        )
        
        print("EXPLICIT SUCCESS:", result)
        
    except Exception as e:
        print("EXPLICIT ERROR:", str(e))

asyncio.run(explicit_store())
"""
    ], capture_output=True, text=True, timeout=30)
    
    print(f"Explicit call result: {explicit_result.returncode}")
    print(f"Explicit stdout: {explicit_result.stdout}")
    
    if explicit_result.returncode != 0:
        print("❌ Explicit store_memory failed - can't test second auto-capture")
        return False
    
    # Now test auto-capture (should work after domains are initialized)
    print("\n--- Testing Auto-Capture After Initialization ---")
    
    test_prompt = "Remember this: Auto-capture should work now that domains are initialized"
    
    try:
        # Hook processing
        hook_result = subprocess.run([
            sys.executable, "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py", 
            "--prompt-submit", f"--prompt={test_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        response = json.loads(hook_result.stdout)
        modified_prompt = response.get("modified_prompt", "")
        
        if "store_memory" in modified_prompt:
            print("✅ Hook processed prompt correctly")
            
            # Extract content
            content_start = modified_prompt.find("store_memory") + len("store_memory")
            content = modified_prompt[content_start:].split("\n")[0].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            
            # Test MCP call (should work now)
            mcp_result = subprocess.run([
                "docker", "exec", "alunai-clarity-mcp-dev",
                "python", "-c", f"""
import asyncio
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def test_post_init_store():
    try:
        config = load_config('/app/data/config.json')
        server = MemoryMcpServer(config)
        
        result = await server.app.call_tool(
            'store_memory',
            {{
                'memory_type': 'post_init_test',
                'content': '{content[:100]}...',
                'importance': 0.9  
            }}
        )
        
        print("POST-INIT SUCCESS:", result)
        
    except Exception as e:
        print("POST-INIT ERROR:", str(e))

asyncio.run(test_post_init_store())
"""
            ], capture_output=True, text=True, timeout=30)
            
            print(f"Post-init MCP result: {mcp_result.returncode}")
            print(f"Post-init stdout: {mcp_result.stdout}")
            
            if mcp_result.returncode == 0 and "POST-INIT SUCCESS" in mcp_result.stdout:
                print("✅ EXPECTED: Auto-capture works after explicit initialization")
                return True
            else:
                print("❌ UNEXPECTED: Auto-capture still failed after initialization")
                return False
        else:
            print("❌ Hook failed to process prompt")
            return False
            
    except Exception as e:
        print(f"❌ Post-init test failed: {e}")
        return False

def main():
    """Run the first-run failure test suite."""
    
    print("FIRST-RUN AUTO-CAPTURE FAILURE TEST")
    print("Exactly replicating the user's scenario")
    print("=" * 60)
    
    # Test 1: Fresh startup auto-capture (should fail)
    fresh_test = test_fresh_startup_auto_capture()
    
    # Test 2: Auto-capture after explicit store (should work)
    post_init_test = test_after_explicit_store()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS RESULTS:")
    print(f"Fresh startup auto-capture: {'✅ WORKED (unexpected)' if fresh_test else '❌ FAILED (expected)'}")
    print(f"Post-initialization auto-capture: {'✅ WORKED (expected)' if post_init_test else '❌ FAILED (unexpected)'}")
    
    if not fresh_test and post_init_test:
        print("\n🎯 CONFIRMED: Lazy initialization prevents first-run auto-capture")
        print("💡 SOLUTION: Move domain initialization to constructor") 
        return True
    elif fresh_test and post_init_test:
        print("\n❓ UNEXPECTED: Auto-capture works on first run - issue may be elsewhere")
        return False
    else:
        print("\n❌ BROADER ISSUE: Auto-capture not working at all")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)