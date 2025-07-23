#!/usr/bin/env python3
"""
Create Claude Code hooks with correct relative paths.
This runs on the host machine to create proper hook configuration.
"""

import os
import json
from datetime import datetime

def create_hook_configuration():
    """Create hook configuration with project-relative paths."""
    
    print("Creating Claude Code hooks with relative paths...")
    
    # Get current working directory (should be project root)
    project_root = os.getcwd()
    
    # Create portable relative path from current working directory
    analyzer_script = "./clarity/mcp/hook_analyzer.py"
    
    # Verify the script exists
    if not os.path.exists(analyzer_script):
        print(f"❌ Hook analyzer not found at: {analyzer_script}")
        return False
    
    print(f"✅ Using hook analyzer at: {analyzer_script}")
    
    # Create hook configuration
    hook_config = {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python {analyzer_script} --prompt-submit --prompt={{prompt}}",
                            "timeout_ms": 1500,
                            "continue_on_error": True,
                            "modify_prompt": True
                        }
                    ]
                }
            ]
        },
        "metadata": {
            "created_by": "mcp-alunai-clarity",
            "version": "1.0.0", 
            "description": "MCP auto-capture hooks with relative paths",
            "created_at": datetime.now().isoformat()
        }
    }
    
    # Create hooks directory
    hooks_dir = "./.claude/alunai-clarity"
    os.makedirs(hooks_dir, exist_ok=True)
    
    # Write hook configuration
    hooks_file = os.path.join(hooks_dir, "hooks.json")
    
    with open(hooks_file, 'w') as f:
        json.dump(hook_config, f, indent=2)
    
    print(f"✅ Created hooks configuration at: {hooks_file}")
    print(f"✅ Hook command: {hook_config['hooks']['UserPromptSubmit'][0]['hooks'][0]['command']}")
    
    return True

def test_hook_execution():
    """Test that the hook can be executed."""
    
    print("\nTesting hook execution...")
    
    analyzer_script = "./clarity/mcp/hook_analyzer.py"
    
    import subprocess
    
    try:
        result = subprocess.run([
            "python3", analyzer_script, "--prompt-submit", 
            "--prompt=Remember this: Testing relative path hooks"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        if result.stdout and "store_memory" in result.stdout:
            print("✅ Hook execution successful")
            return True
        else:
            print("❌ Hook execution failed - no output")
            return False
            
    except Exception as e:
        print(f"❌ Hook execution failed: {e}")
        return False

def main():
    """Create hooks and test execution."""
    
    print("CREATING CLAUDE CODE HOOKS WITH RELATIVE PATHS")
    print("=" * 50)
    
    # Create hooks
    hooks_created = create_hook_configuration()
    
    if not hooks_created:
        print("❌ Failed to create hooks")
        return False
    
    # Test execution
    execution_works = test_hook_execution()
    
    print("\n" + "=" * 50)
    
    if hooks_created and execution_works:
        print("✅ SUCCESS: Hooks created with relative paths and tested")
        print("✅ Auto-capture should now work with 'Remember this:' prompts")
        return True
    else:
        print("❌ FAILED: Issues with hook creation or execution")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)