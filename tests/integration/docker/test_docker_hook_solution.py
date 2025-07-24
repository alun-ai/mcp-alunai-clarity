#!/usr/bin/env python3
"""
Test the Docker-based hook solution.
Validates that hooks work for all users using Docker container execution.
"""

import subprocess
import json
import time
import sys
import os

def test_container_availability():
    """Test that the MCP container is running and accessible."""
    
    print("=== TESTING CONTAINER AVAILABILITY ===")
    
    container_name = "alunai-clarity-mcp-dev"
    
    try:
        # Check if container is running
        result = subprocess.run([
            "docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"
        ], capture_output=True, text=True, check=True)
        
        if container_name in result.stdout:
            print(f"✅ Container {container_name} is running")
            return True
        else:
            print(f"❌ Container {container_name} is not running")
            
            # Try to find any running MCP containers
            all_containers = subprocess.run([
                "docker", "ps", "--format", "{{.Names}}"
            ], capture_output=True, text=True, check=True)
            
            mcp_containers = [name for name in all_containers.stdout.split('\n') 
                             if 'alunai' in name or 'clarity' in name or 'mcp' in name]
            
            if mcp_containers:
                print(f"Found possible MCP containers: {mcp_containers}")
                return mcp_containers[0]  # Return first found container
            else:
                print("No MCP containers found running")
                return False
            
    except Exception as e:
        print(f"❌ Error checking container: {e}")
        return False

def test_docker_hook_execution(container_name):
    """Test that hook execution works via Docker."""
    
    print(f"\n=== TESTING DOCKER HOOK EXECUTION ===")
    
    test_prompt = "Remember this: Docker hook execution test"
    
    try:
        result = subprocess.run([
            "docker", "exec", container_name,
            "python", "/app/clarity/mcp/hook_analyzer.py",
            "--prompt-submit", f"--prompt={test_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        if result.stdout:
            try:
                response = json.loads(result.stdout)
                modified_prompt = response.get("modified_prompt", "")
                
                if "store_memory" in modified_prompt and "Docker hook execution test" in modified_prompt:
                    print("✅ Docker hook execution successful")
                    print(f"Modified prompt: {modified_prompt[:80]}...")
                    return True
                else:
                    print("❌ Hook did not modify prompt correctly")
                    print(f"Response: {result.stdout}")
                    return False
                    
            except json.JSONDecodeError:
                print("❌ Hook returned invalid JSON")
                print(f"Output: {result.stdout}")
                return False
        else:
            print("❌ Hook returned no output")
            return False
            
    except Exception as e:
        print(f"❌ Docker hook execution failed: {e}")
        return False

def test_automatic_hook_creation():
    """Test that hooks are created automatically when MCP server starts."""
    
    print(f"\n=== TESTING AUTOMATIC HOOK CREATION ===")
    
    # Remove existing hooks to test automatic creation
    hook_path = "./.claude/alunai-clarity/hooks.json"
    if os.path.exists(hook_path):
        os.remove(hook_path)
        print("Removed existing hooks for clean test")
    
    # Start MCP server in container to trigger hook creation
    try:
        result = subprocess.run([
            "docker", "exec", "alunai-clarity-mcp-dev",
            "python", "-c", """
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

# Initialize server (this should create hooks)
config = load_config('/app/data/config.json')
server = MemoryMcpServer(config)
print("Server initialized - hooks should be created")
"""
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ MCP server initialization completed")
            
            # Check if hooks were created
            if os.path.exists(hook_path):
                with open(hook_path, 'r') as f:
                    hook_config = json.load(f)
                
                # Validate hook configuration
                try:
                    command = hook_config['hooks']['UserPromptSubmit'][0]['hooks'][0]['command']
                    
                    if "docker exec" in command and "alunai-clarity-mcp" in command:
                        print("✅ Hooks created automatically with Docker execution")
                        print(f"Command: {command}")
                        return True
                    else:
                        print("❌ Hooks created but with incorrect command")
                        print(f"Command: {command}")
                        return False
                        
                except Exception as e:
                    print(f"❌ Invalid hook configuration: {e}")
                    return False
            else:
                print("❌ Hooks were not created automatically")
                return False
        else:
            print("❌ MCP server initialization failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Hook creation test failed: {e}")
        return False

def test_end_to_end_flow():
    """Test complete end-to-end flow with Docker hooks."""
    
    print(f"\n=== TESTING END-TO-END FLOW ===")
    
    # Simulate user prompt being processed by Claude Code hook
    user_prompt = "Remember this: End-to-end Docker hook validation test"
    
    # Step 1: Hook processes prompt via Docker
    try:
        hook_result = subprocess.run([
            "docker", "exec", "alunai-clarity-mcp-dev",
            "python", "/app/clarity/mcp/hook_analyzer.py",
            "--prompt-submit", f"--prompt={user_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        response = json.loads(hook_result.stdout)
        modified_prompt = response.get("modified_prompt", "")
        
        if "store_memory" not in modified_prompt:
            print("❌ Step 1 failed: Hook did not modify prompt")
            return False
            
        print("✅ Step 1: Hook successfully modified prompt via Docker")
        
        # Step 2: MCP server processes store_memory
        mcp_result = subprocess.run([
            "docker", "exec", "alunai-clarity-mcp-dev",
            "python", "-c", f"""
import asyncio
import sys
sys.path.insert(0, '/app')

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config

async def end_to_end_test():
    try:
        config = load_config('/app/data/config.json')
        server = MemoryMcpServer(config)
        
        result = await server.app.call_tool(
            'store_memory',
            {{
                'memory_type': 'docker_hook_test',
                'content': 'End-to-end Docker hook validation test',
                'importance': 1.0
            }}
        )
        
        print("END_TO_END_SUCCESS:", result)
        
    except Exception as e:
        print("END_TO_END_ERROR:", str(e))

asyncio.run(end_to_end_test())
"""
        ], capture_output=True, text=True, timeout=30)
        
        if mcp_result.returncode == 0 and "END_TO_END_SUCCESS" in mcp_result.stdout:
            print("✅ Step 2: MCP server processed store_memory successfully")
            print("✅ END-TO-END FLOW SUCCESSFUL")
            return True
        else:
            print("❌ Step 2 failed: MCP server processing failed")
            print(f"Output: {mcp_result.stdout}")
            print(f"Error: {mcp_result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

def main():
    """Run Docker hook solution test suite."""
    
    print("DOCKER HOOK SOLUTION TEST")
    print("Testing automatic hook creation using Docker container execution")
    print("=" * 70)
    
    # Test 1: Container availability
    container_result = test_container_availability()
    if not container_result:
        print("\n❌ CONTAINER NOT AVAILABLE - Cannot proceed with tests")
        return False
    
    container_name = container_result if isinstance(container_result, str) else "alunai-clarity-mcp-dev"
    
    # Test 2: Docker hook execution
    hook_test = test_docker_hook_execution(container_name)
    
    # Test 3: Automatic hook creation
    creation_test = test_automatic_hook_creation()
    
    # Test 4: End-to-end flow
    flow_test = test_end_to_end_flow()
    
    print("\n" + "=" * 70)
    print("DOCKER HOOK SOLUTION RESULTS:")
    print(f"Container availability: {'✅ PASS' if container_result else '❌ FAIL'}")
    print(f"Docker hook execution: {'✅ PASS' if hook_test else '❌ FAIL'}")
    print(f"Automatic hook creation: {'✅ PASS' if creation_test else '❌ FAIL'}")
    print(f"End-to-end flow: {'✅ PASS' if flow_test else '❌ FAIL'}")
    
    all_passed = container_result and hook_test and creation_test and flow_test
    
    print(f"\nOVERALL RESULT: {'🎉 ALL TESTS PASSED' if all_passed else '❌ TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ DOCKER HOOK SOLUTION IS WORKING")
        print("✅ Hooks are created automatically when MCP server starts")
        print("✅ Hooks execute via Docker container - works for all users")
        print("✅ No manual setup required - just like Qdrant directory creation")
        print("✅ 'Remember this:' prompts are automatically converted to store_memory")
        
        print("\n🎯 USER EXPERIENCE:")
        print("Users simply:")
        print("1. Run their MCP server via Docker (as they already do)")
        print("2. Type 'Remember this: [content]' in Claude Code")
        print("3. Memory is automatically stored - no setup needed")
    else:
        print("\n❌ DOCKER HOOK SOLUTION HAS ISSUES")
        print("Please fix the failing tests before deployment.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)