#!/usr/bin/env python3
"""
Test the REAL user workflow - not container internals.
Tests what actually matters: Can Claude Code on the HOST use the auto-capture feature?
"""

import subprocess
import json
import time
import sys
import os
from pathlib import Path

def test_hooks_file_exists_on_host():
    """Test that hooks.json file exists where Claude Code expects it on the HOST."""
    
    print("=== TESTING HOOKS FILE EXISTS ON HOST ===")
    
    # This is where Claude Code looks for hooks in the project directory
    hooks_path = "./.claude/alunai-clarity/hooks.json"
    
    if not os.path.exists(hooks_path):
        print(f"❌ Hooks file not found at: {os.path.abspath(hooks_path)}")
        print("This means Claude Code cannot find the hooks file")
        return False
    
    print(f"✅ Hooks file exists at: {os.path.abspath(hooks_path)}")
    
    # Test file is readable
    try:
        with open(hooks_path, 'r') as f:
            hooks_config = json.load(f)
        print("✅ Hooks file is readable and valid JSON")
    except Exception as e:
        print(f"❌ Hooks file exists but cannot be read: {e}")
        return False
    
    # Test file has correct structure for Claude Code
    try:
        command = hooks_config['hooks']['UserPromptSubmit'][0]['hooks'][0]['command']
        if 'docker exec' in command and 'hook_analyzer.py' in command:
            print("✅ Hooks file has correct Claude Code structure")
            print(f"Command: {command}")
        else:
            print(f"❌ Hooks file has incorrect structure. Command: {command}")
            return False
    except KeyError as e:
        print(f"❌ Hooks file missing required structure: {e}")
        return False
    
    return True

def test_hooks_created_automatically():
    """Test that hooks are created automatically when MCP server starts."""
    
    print("\n=== TESTING AUTOMATIC HOOKS CREATION ===")
    
    hooks_path = "./.claude/alunai-clarity/hooks.json"
    
    # Remove existing hooks to test fresh creation
    if os.path.exists(hooks_path):
        os.remove(hooks_path)
        print("Removed existing hooks for clean test")
    
    # Start MCP server (this should create hooks automatically)
    try:
        result = subprocess.run([
            "docker", "exec", "alunai-clarity-mcp-dev",
            "python", "-c", """
import sys
sys.path.insert(0, '/app')
from clarity.mcp.server import MemoryMcpServer  
from clarity.utils.config import load_config

# This should automatically create hooks
config = load_config('/app/data/config.json')
server = MemoryMcpServer(config)
print("MCP server initialized")
"""
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ MCP server failed to start: {result.stderr}")
            return False
            
        print("✅ MCP server started successfully")
        
        # Give it a moment for file system sync
        time.sleep(1)
        
        # Check if hooks were created on HOST
        if os.path.exists(hooks_path):
            print(f"✅ Hooks automatically created at: {os.path.abspath(hooks_path)}")
            return True
        else:
            print(f"❌ Hooks were NOT created automatically at: {os.path.abspath(hooks_path)}")
            print("This means users won't get auto-capture without manual setup")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test automatic creation: {e}")
        return False

def test_hook_command_works_from_host():
    """Test that the hook command can actually be executed from the HOST."""
    
    print("\n=== TESTING HOOK COMMAND FROM HOST ===")
    
    hooks_path = "./.claude/alunai-clarity/hooks.json"
    
    if not os.path.exists(hooks_path):
        print("❌ No hooks file found - cannot test command execution")
        return False
    
    # Read the actual hook command that Claude Code would execute
    try:
        with open(hooks_path, 'r') as f:
            hooks_config = json.load(f)
        
        hook_command = hooks_config['hooks']['UserPromptSubmit'][0]['hooks'][0]['command']
        print(f"Testing hook command: {hook_command}")
        
        # Replace the {prompt} placeholder with a test prompt
        test_prompt = "Remember this: Host-side hook execution test"
        actual_command = hook_command.replace("{prompt}", test_prompt)
        
        print(f"Executing: {actual_command}")
        
        # Execute the command exactly as Claude Code would
        result = subprocess.run(
            actual_command,
            shell=True,  # Claude Code uses shell execution
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"❌ Hook command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
        
        # Check if we got valid JSON response
        if result.stdout:
            try:
                response = json.loads(result.stdout)
                modified_prompt = response.get("modified_prompt", "")
                
                if "store_memory" in modified_prompt and "Host-side hook execution test" in modified_prompt:
                    print("✅ Hook command executed successfully from host")
                    print(f"Response: {modified_prompt[:80]}...")
                    return True
                else:
                    print("❌ Hook command ran but didn't modify prompt correctly")
                    print(f"Response: {result.stdout}")
                    return False
            except json.JSONDecodeError:
                print("❌ Hook command returned invalid JSON")
                print(f"Output: {result.stdout}")
                return False
        else:
            print("❌ Hook command returned no output")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test hook command: {e}")
        return False

def test_file_permissions_and_access():
    """Test that the hooks file has correct permissions for Claude Code."""
    
    print("\n=== TESTING FILE PERMISSIONS ===")
    
    hooks_path = "./.claude/alunai-clarity/hooks.json"
    
    if not os.path.exists(hooks_path):
        print("❌ Hooks file doesn't exist - cannot test permissions")
        return False
    
    # Test file is readable
    if os.access(hooks_path, os.R_OK):
        print("✅ Hooks file is readable")
    else:
        print("❌ Hooks file is not readable - Claude Code won't be able to use it")
        return False
    
    # Test file ownership/permissions
    import stat
    file_stat = os.stat(hooks_path)
    mode = stat.filemode(file_stat.st_mode)
    print(f"File permissions: {mode}")
    
    # Should be readable by owner (which Claude Code runs as)
    if file_stat.st_mode & stat.S_IRUSR:
        print("✅ File has correct read permissions")
        return True
    else:
        print("❌ File doesn't have read permissions for owner")
        return False

def test_directory_structure():
    """Test that the .claude directory structure matches what Claude Code expects."""
    
    print("\n=== TESTING DIRECTORY STRUCTURE ===")
    
    # Check .claude directory exists
    claude_dir = "./.claude"
    if not os.path.exists(claude_dir):
        print(f"❌ .claude directory doesn't exist at: {os.path.abspath(claude_dir)}")
        return False
    
    print(f"✅ .claude directory exists at: {os.path.abspath(claude_dir)}")
    
    # Check alunai-clarity subdirectory
    project_dir = "./.claude/alunai-clarity"
    if not os.path.exists(project_dir):
        print(f"❌ Project directory doesn't exist at: {os.path.abspath(project_dir)}")
        return False
    
    print(f"✅ Project directory exists at: {os.path.abspath(project_dir)}")
    
    # Check that it contains expected files (like qdrant, config)
    expected_files = ['hooks.json']
    expected_dirs = ['qdrant', '.qdrant_coordination']
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(f"{project_dir}/{file}"):
            missing_files.append(file)
    
    missing_dirs = []
    for dir in expected_dirs:
        if not os.path.exists(f"{project_dir}/{dir}"):
            missing_dirs.append(dir)
    
    if missing_files:
        print(f"❌ Missing expected files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"⚠️  Missing expected directories: {missing_dirs} (may be OK)")
    
    print("✅ Directory structure is correct")
    return True

def main():
    """Run the REAL user workflow tests."""
    
    print("REAL USER WORKFLOW TEST")
    print("Testing what actually matters: Can Claude Code on the HOST use auto-capture?")
    print("=" * 80)
    
    # Test 1: Basic file existence and structure
    file_exists = test_hooks_file_exists_on_host()
    
    # Test 2: Automatic creation
    auto_creation = test_hooks_created_automatically()
    
    # Test 3: Command execution from host
    command_works = test_hook_command_works_from_host()
    
    # Test 4: File permissions
    permissions_ok = test_file_permissions_and_access()
    
    # Test 5: Directory structure
    structure_ok = test_directory_structure()
    
    print("\n" + "=" * 80)
    print("REAL USER WORKFLOW RESULTS:")
    print(f"Hooks file exists on host: {'✅ PASS' if file_exists else '❌ FAIL'}")
    print(f"Automatic creation works: {'✅ PASS' if auto_creation else '❌ FAIL'}")
    print(f"Hook command works from host: {'✅ PASS' if command_works else '❌ FAIL'}")
    print(f"File permissions correct: {'✅ PASS' if permissions_ok else '❌ FAIL'}")
    print(f"Directory structure correct: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    
    all_passed = file_exists and auto_creation and command_works and permissions_ok and structure_ok
    
    print(f"\nOVERALL RESULT: {'🎉 ALL TESTS PASSED' if all_passed else '❌ TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ REAL USER WORKFLOW IS WORKING")
        print("✅ Claude Code on the host can find and use the hooks")
        print("✅ Auto-capture will work for real users")
        print("✅ No manual setup required")
        
        print("\n🎯 WHAT THIS MEANS FOR USERS:")
        print("1. Users start their MCP server via Docker")
        print("2. Hooks are automatically created in the correct location")
        print("3. Claude Code can find and execute the hooks")
        print("4. 'Remember this:' prompts automatically store memory")
        print("5. Everything works without any manual steps")
    else:
        print("\n❌ REAL USER WORKFLOW IS BROKEN")
        print("The feature will NOT work for real users despite passing container tests")
        print("Focus on fixing the failing tests above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)