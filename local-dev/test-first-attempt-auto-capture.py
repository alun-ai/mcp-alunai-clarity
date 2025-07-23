#!/usr/bin/env python3
"""Test that auto-capture works on the first attempt without needing a manual trigger."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clarity.auto_memory.auto_capture import should_store_memory, extract_memory_content
from clarity.mcp.hook_analyzer import HookAnalyzerCLI


async def test_first_attempt_auto_capture():
    """Test that auto-capture works immediately on first attempt."""
    
    print("=== Testing First-Attempt Auto-Capture ===")
    
    # Test prompts that should trigger auto-capture
    test_prompts = [
        "Remember this: Always validate user input before database operations",
        "Please remember: I prefer TypeScript over JavaScript for new projects",
        "Remember that: Error messages should be user-friendly and actionable"
    ]
    
    # Test hook analyzer functionality (simulates Claude Code hook execution)
    analyzer = HookAnalyzerCLI()
    
    print("\n--- Testing Hook Analyzer (Claude Code Integration) ---")
    
    success_count = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: '{prompt[:50]}...'")
        
        try:
            # Test the hook analyzer's auto-capture detection
            modified_prompt = await analyzer._check_auto_memory_capture(prompt)
            
            if modified_prompt and modified_prompt != prompt:
                print(f"  ✅ Auto-capture triggered!")
                print(f"  📝 Modified prompt preview: {modified_prompt[:100]}...")
                success_count += 1
            else:
                print(f"  ❌ Auto-capture NOT triggered")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n--- Results ---")
    print(f"Auto-capture success rate: {success_count}/{len(test_prompts)} ({success_count/len(test_prompts)*100:.1f}%)")
    
    if success_count == len(test_prompts):
        print("🎉 SUCCESS: Auto-capture works on first attempt!")
        return True
    else:
        print("❌ FAILURE: Auto-capture not working consistently")
        return False


async def test_hook_configuration():
    """Test that the Claude Code hook configuration exists."""
    
    print("\n=== Testing Hook Configuration ===")
    
    hook_config_path = "/root/.config/claude-code/hooks.json"
    
    if os.path.exists(hook_config_path):
        print(f"✅ Hook config file exists: {hook_config_path}")
        
        try:
            with open(hook_config_path, 'r') as f:
                config = json.load(f)
            
            if 'hooks' in config and 'UserPromptSubmit' in config['hooks']:
                print("✅ UserPromptSubmit hook configured")
                
                # Check the hook command
                user_hooks = config['hooks']['UserPromptSubmit']
                if user_hooks and len(user_hooks) > 0:
                    hook_cmd = user_hooks[0]['hooks'][0]['command']
                    if 'hook_analyzer.py' in hook_cmd and '--prompt-submit' in hook_cmd:
                        print("✅ Hook command correctly configured for auto-capture")
                        return True
                    else:
                        print(f"❌ Hook command incorrect: {hook_cmd}")
                else:
                    print("❌ No UserPromptSubmit hooks found")
            else:
                print("❌ UserPromptSubmit hook not configured")
        except Exception as e:
            print(f"❌ Error reading hook config: {e}")
    else:
        print(f"❌ Hook config file not found: {hook_config_path}")
    
    return False


async def main():
    """Run all tests."""
    print("Testing immediate auto-capture functionality...")
    
    # Test 1: Hook configuration
    config_ok = await test_hook_configuration()
    
    # Test 2: Auto-capture functionality  
    capture_ok = await test_first_attempt_auto_capture()
    
    print("\n" + "="*50)
    if config_ok and capture_ok:
        print("🎉 ALL TESTS PASSED - Auto-capture ready from first use!")
        return True
    else:
        print("❌ TESTS FAILED - Auto-capture needs fixes")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)