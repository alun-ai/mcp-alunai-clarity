#!/usr/bin/env python3
"""Test automatic memory capture functionality."""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clarity.auto_memory.auto_capture import should_store_memory, extract_memory_content
from clarity.mcp.hook_analyzer import HookAnalyzerCLI

async def test_auto_capture():
    """Test the auto-capture functionality directly."""
    
    test_prompts = [
        "Remember this: Always use async/await for database operations",
        "Please remember: I prefer TypeScript over JavaScript", 
        "Remember that error handling should be transparent",
        "Can you help me with this code?",  # Should not trigger
        "Hello there, how are you?"  # Should not trigger
    ]
    
    print("=== Testing Auto-Capture Pattern Detection ===")
    
    for prompt in test_prompts:
        print(f"\nTesting: '{prompt}'")
        
        # Test pattern detection
        should_store = should_store_memory(prompt)
        print(f"  should_store_memory: {should_store}")
        
        if should_store:
            memory_type, content, importance = extract_memory_content(prompt)
            print(f"  memory_type: {memory_type}")
            print(f"  content: {content}")
            print(f"  importance: {importance}")
        
        # Test hook analyzer 
        analyzer = HookAnalyzerCLI()
        modified_prompt = await analyzer._check_auto_memory_capture(prompt)
        if modified_prompt:
            print(f"  modified_prompt: {modified_prompt}")
        else:
            print(f"  modified_prompt: None (no change)")

if __name__ == "__main__":
    asyncio.run(test_auto_capture())