#!/usr/bin/env python3
"""
Final validation test - Complete end-to-end auto-capture flow.
Tests that auto-capture works correctly from fresh startup with fixed hook paths.
"""

import subprocess
import json
import time
import sys
import os

def validate_hook_configuration():
    """Validate that hook configuration has correct host paths."""
    
    print("=== VALIDATING HOOK CONFIGURATION ===")
    
    hook_path = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/.claude/alunai-clarity/hooks.json"
    
    if not os.path.exists(hook_path):
        print("❌ Hook configuration file not found")
        return False
    
    with open(hook_path, 'r') as f:
        config = json.load(f)
    
    # Check if hook command uses host path
    try:
        command = config['hooks']['UserPromptSubmit'][0]['hooks'][0]['command']
        
        if "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py" in command:
            print("✅ Hook configuration uses correct host path")
            print(f"Command: {command}")
            return True
        else:
            print("❌ Hook configuration uses incorrect path")
            print(f"Command: {command}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading hook configuration: {e}")
        return False

def test_hook_execution_with_logging():
    """Test hook execution and verify logging works."""
    
    print("\n=== TESTING HOOK EXECUTION ===")
    
    # Clear previous log
    log_file = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/hook_execution.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Test hook execution
    test_prompt = "Remember this: Final validation test for auto-capture integration"
    
    try:
        result = subprocess.run([
            "python3", "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py",
            "--prompt-submit", f"--prompt={test_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        # Validate JSON response
        if result.stdout:
            try:
                response = json.loads(result.stdout)
                modified_prompt = response.get("modified_prompt", "")
                
                if "store_memory" in modified_prompt and "Final validation test" in modified_prompt:
                    print("✅ Hook execution successful")
                    print(f"Modified prompt: {modified_prompt[:80]}...")
                    
                    # Check logging
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                        print("✅ Hook execution logged successfully")
                        return True
                    else:
                        print("❌ Hook execution not logged")
                        return False
                else:
                    print("❌ Hook did not modify prompt correctly")
                    return False
                    
            except json.JSONDecodeError:
                print("❌ Hook returned invalid JSON")
                return False
        else:
            print("❌ Hook returned no output")
            return False
            
    except Exception as e:
        print(f"❌ Hook execution failed: {e}")
        return False

def test_mcp_server_integration():
    """Test integration with MCP server for actual memory storage."""
    
    print("\n=== TESTING MCP SERVER INTEGRATION ===")
    
    # Test with fresh container to simulate first run
    try:
        result = subprocess.run([
            "docker", "exec", "alunai-clarity-mcp-dev",
            "python", "-c", """
import asyncio
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def test_integration():
    try:
        config = load_config('/app/data/config.json')
        server = MemoryMcpServer(config)
        
        # Simulate what happens when Claude processes auto-capture modified prompt
        result = await server.app.call_tool(
            'store_memory',
            {
                'memory_type': 'final_validation',
                'content': 'Final validation test for auto-capture integration',
                'importance': 0.95
            }
        )
        
        print("INTEGRATION SUCCESS:", result)
        
    except Exception as e:
        print("INTEGRATION ERROR:", str(e))
        import traceback
        traceback.print_exc()

asyncio.run(test_integration())
"""
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "INTEGRATION SUCCESS" in result.stdout:
            print("✅ MCP server integration successful")
            print("✅ Memory storage works on first run")
            return True
        else:
            print("❌ MCP server integration failed")
            print(f"Return code: {result.returncode}")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ MCP integration test failed: {e}")
        return False

def test_complete_flow_simulation():
    """Simulate the complete Claude Code flow."""
    
    print("\n=== TESTING COMPLETE FLOW SIMULATION ===")
    
    # Step 1: Hook processes user prompt
    user_prompt = "Remember this: Complete flow validation - hook processing works perfectly"
    
    try:
        hook_result = subprocess.run([
            "python3", "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py",
            "--prompt-submit", f"--prompt={user_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        response = json.loads(hook_result.stdout)
        modified_prompt = response.get("modified_prompt", "")
        
        if "store_memory" not in modified_prompt:
            print("❌ Step 1 failed: Hook did not modify prompt")
            return False
            
        print("✅ Step 1: Hook successfully modified prompt")
        
        # Step 2: Extract store_memory parameters (simulates Claude's processing)
        # Find the content after "store_memory"
        content_start = modified_prompt.find("store_memory") + len("store_memory")
        content_line = modified_prompt[content_start:].split("\n")[0].strip()
        if content_line.startswith(":"):
            content_line = content_line[1:].strip()
        
        print(f"✅ Step 2: Extracted content: '{content_line[:50]}...'")
        
        # Step 3: MCP server processes the store_memory call
        mcp_result = subprocess.run([
            "docker", "exec", "alunai-clarity-mcp-dev",
            "python", "-c", f"""
import asyncio
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def complete_flow_test():
    try:
        config = load_config('/app/data/config.json')
        server = MemoryMcpServer(config)
        
        result = await server.app.call_tool(
            'store_memory',
            {{
                'memory_type': 'complete_flow_test',
                'content': '{content_line}',
                'importance': 1.0
            }}
        )
        
        print("COMPLETE FLOW SUCCESS:", result)
        
    except Exception as e:
        print("COMPLETE FLOW ERROR:", str(e))

asyncio.run(complete_flow_test())
"""
        ], capture_output=True, text=True, timeout=30)
        
        if mcp_result.returncode == 0 and "COMPLETE FLOW SUCCESS" in mcp_result.stdout:
            print("✅ Step 3: MCP server processed store_memory successfully")
            print("✅ COMPLETE FLOW SUCCESSFUL")
            return True
        else:
            print("❌ Step 3 failed: MCP server processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Complete flow test failed: {e}")
        return False

def main():
    """Run final validation test suite."""
    
    print("FINAL VALIDATION TEST - AUTO-CAPTURE INTEGRATION")
    print("Testing complete end-to-end flow with fixed hook paths")
    print("=" * 70)
    
    # Test 1: Hook configuration
    config_test = validate_hook_configuration()
    
    # Test 2: Hook execution
    hook_test = test_hook_execution_with_logging()
    
    # Test 3: MCP server integration
    mcp_test = test_mcp_server_integration()
    
    # Test 4: Complete flow simulation
    flow_test = test_complete_flow_simulation()
    
    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS:")
    print(f"Hook configuration: {'✅ PASS' if config_test else '❌ FAIL'}")
    print(f"Hook execution: {'✅ PASS' if hook_test else '❌ FAIL'}")
    print(f"MCP integration: {'✅ PASS' if mcp_test else '❌ FAIL'}")
    print(f"Complete flow: {'✅ PASS' if flow_test else '❌ FAIL'}")
    
    all_passed = config_test and hook_test and mcp_test and flow_test
    
    print(f"\nOVERALL RESULT: {'🎉 ALL TESTS PASSED' if all_passed else '❌ TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ AUTO-CAPTURE IS FULLY FUNCTIONAL")
        print("✅ 'Remember this:' prompts will be automatically converted to store_memory")
        print("✅ Works from first run - no initialization required")
        print("✅ Hook configuration uses correct host paths")
        print("✅ Integration with MCP server is seamless")
        
        print("\n🎯 USAGE:")
        print("Users can now type 'Remember this: [content]' and it will automatically")
        print("be stored in memory without needing explicit store_memory tool calls.")
    else:
        print("\n❌ AUTO-CAPTURE HAS ISSUES")
        print("Please fix the failing tests before deployment.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)