#!/usr/bin/env python3
"""
Extract hooks from config.json and create hooks.json for Claude Code.
This bridges the gap between our config.json approach and Claude Code's expected hooks.json format.
"""

import json
import os
import sys

def extract_hooks_from_config():
    """Extract hooks from config.json and create hooks.json."""
    
    config_path = "./.claude/alunai-clarity/config.json"
    hooks_path = "./.claude/alunai-clarity/hooks.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        # Read config.json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if it has claude_code_hooks section
        if "claude_code_hooks" not in config:
            print("❌ No claude_code_hooks section found in config.json")
            return False
        
        hooks_config = config["claude_code_hooks"]
        
        # Write to hooks.json in the format Claude Code expects
        with open(hooks_path, 'w') as f:
            json.dump(hooks_config, f, indent=2)
        
        print(f"✅ Extracted hooks from config.json to {hooks_path}")
        print(f"✅ Container: {hooks_config.get('metadata', {}).get('container_name', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting hooks: {e}")
        return False

def main():
    """Main entry point."""
    
    print("Extracting Claude Code hooks from config.json...")
    
    success = extract_hooks_from_config()
    
    if success:
        print("\n🎉 Hooks extracted successfully!")
        print("Claude Code should now be able to use auto-capture.")
    else:
        print("\n❌ Failed to extract hooks")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())