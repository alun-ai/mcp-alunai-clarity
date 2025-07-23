#!/usr/bin/env python3
"""Test MCPHooks initialization by triggering a memory operation."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clarity.utils.config import load_config
from clarity.mcp.server import MemoryMcpServer


async def test_mcp_hooks_initialization():
    """Test that MCPHooks are properly initialized when domains are loaded."""
    
    print("=== Testing MCPHooks Initialization ===")
    
    # Load config
    config_path = "/app/data/config.json"
    config = load_config(config_path)
    
    print(f"Config loaded from: {config_path}")
    
    # Create server instance
    server = MemoryMcpServer(config)
    print("Server instance created")
    
    # Check initial state
    print(f"Initial domains_initialized: {server._domains_initialized}")
    print(f"Initial autocode_hooks: {hasattr(server, 'autocode_hooks')}")
    print(f"Initial mcp_hooks: {hasattr(server, 'mcp_hooks')}")
    
    # Trigger lazy initialization directly
    print("\nTriggering lazy initialization...")
    try:
        # Call the lazy initialization method directly
        await server._lazy_initialize_domains()
        print("Lazy initialization completed")
        
        # Check post-initialization state
        print(f"\nPost-init domains_initialized: {server._domains_initialized}")
        print(f"Post-init autocode_hooks: {hasattr(server, 'autocode_hooks')}")
        print(f"Post-init mcp_hooks: {hasattr(server, 'mcp_hooks')}")
        
        if hasattr(server, 'mcp_hooks'):
            print("✅ MCPHooks successfully initialized!")
            
            # Check if hooks were set up
            if hasattr(server.mcp_hooks, 'tool_indexer'):
                indexer = server.mcp_hooks.tool_indexer
                if hasattr(indexer, 'hook_integration'):
                    print("✅ Hook integration available")
                    # Check if hook config path exists
                    hook_config_path = os.path.expanduser("~/.config/claude-code/hooks.json")
                    if os.path.exists(hook_config_path):
                        print(f"✅ Claude Code hooks configured at: {hook_config_path}")
                        with open(hook_config_path, 'r') as f:
                            hook_config = json.load(f)
                            if 'UserPromptSubmit' in hook_config.get('hooks', {}):
                                print("✅ UserPromptSubmit hook configured for auto-capture!")
                            else:
                                print("❌ UserPromptSubmit hook not found in config")
                    else:
                        print(f"❌ Hook config file not found: {hook_config_path}")
                else:
                    print("❌ Hook integration not available")
            else:
                print("❌ Tool indexer not available")
        else:
            print("❌ MCPHooks not initialized")
            
    except Exception as e:
        print(f"Error during memory operation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_hooks_initialization())