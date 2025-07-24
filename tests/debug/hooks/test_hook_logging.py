#!/usr/bin/env python3
"""
Test to add logging to hook execution to see if Claude Code calls it.
"""

import os
import json
from datetime import datetime

def add_hook_logging():
    """Add logging to the hook analyzer to track when it's called."""
    
    hook_analyzer_path = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py"
    
    # Read the current hook analyzer
    with open(hook_analyzer_path, 'r') as f:
        content = f.read()
    
    # Add logging at the start of main()
    logging_code = '''    # Log hook execution for debugging
    import os
    log_file = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/hook_execution.log"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat()}: Hook executed with args: {sys.argv}\\n")
'''
    
    # Find the main function and add logging
    if "def main():" in content and "# Log hook execution for debugging" not in content:
        # Insert logging right after the main() function definition
        main_pos = content.find("def main():")
        next_line_pos = content.find("\n", main_pos) + 1
        
        # Find the next line with actual code (skip docstring if present)
        while next_line_pos < len(content):
            line_start = next_line_pos
            line_end = content.find("\n", line_start)
            if line_end == -1:
                line_end = len(content)
            line = content[line_start:line_end].strip()
            
            if line and not line.startswith('"""') and not line.startswith("'''"):
                break
            next_line_pos = line_end + 1
        
        # Insert logging code
        modified_content = content[:next_line_pos] + logging_code + content[next_line_pos:]
        
        # Write back
        with open(hook_analyzer_path, 'w') as f:
            f.write(modified_content)
        
        print("✅ Added logging to hook analyzer")
        return True
    else:
        print("✅ Logging already present or main() not found")
        return True

def test_hook_execution():
    """Test if the hook gets executed by directly calling it."""
    
    print("=== TESTING HOOK EXECUTION LOGGING ===")
    
    # Clear the log file
    log_file = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/hook_execution.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Add logging to hook analyzer
    add_hook_logging()
    
    # Test direct hook execution
    import subprocess
    
    test_prompt = "Remember this: Testing hook execution logging"
    
    print(f"Testing hook with prompt: {test_prompt[:50]}...")
    
    try:
        result = subprocess.run([
            "python", "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py",
            "--prompt-submit", f"--prompt={test_prompt}"
        ], capture_output=True, text=True, check=True, timeout=10)
        
        print("✅ Hook executed successfully")
        print(f"Output: {result.stdout[:100]}...")
        
        # Check if log file was created
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            print("✅ Hook execution logged:")
            print(log_content)
            return True
        else:
            print("❌ No log file created")
            return False
            
    except Exception as e:
        print(f"❌ Hook execution failed: {e}")
        return False

def main():
    """Test hook execution logging."""
    
    print("HOOK EXECUTION LOGGING TEST")
    print("Testing if Claude Code actually calls the hook")
    print("=" * 50)
    
    success = test_hook_execution()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Hook logging test successful")
        print("Now try using 'Remember this:' in Claude Code and check the log file")
        print("Log file: /Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/hook_execution.log")
    else:
        print("❌ Hook logging test failed")
    
    return success

if __name__ == "__main__":
    main()