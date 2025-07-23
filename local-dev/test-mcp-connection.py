#!/usr/bin/env python3
"""
Test script to validate MCP server connection and hook system functionality.
Run this inside the container to test hook execution issues.
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, '/app')

def test_basic_imports():
    """Test that all basic modules can be imported."""
    print("=" * 50)
    print("TESTING BASIC IMPORTS")
    print("=" * 50)
    
    try:
        from clarity.mcp.server import MCPServer
        print("✓ MCPServer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MCPServer: {e}")
        traceback.print_exc()
    
    try:
        from clarity.mcp.tool_indexer import MCPToolIndexer
        print("✓ MCPToolIndexer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MCPToolIndexer: {e}")
        traceback.print_exc()
    
    try:
        from clarity.mcp.hook_integration import MCPHookIntegration
        print("✓ MCPHookIntegration imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MCPHookIntegration: {e}")
        traceback.print_exc()

def test_domain_manager():
    """Test domain manager initialization."""
    print("\n" + "=" * 50)
    print("TESTING DOMAIN MANAGER")
    print("=" * 50)
    
    try:
        from clarity.domains.manager import DomainManager
        domain_manager = DomainManager()
        print("✓ DomainManager initialized successfully")
        return domain_manager
    except Exception as e:
        print(f"✗ Failed to initialize DomainManager: {e}")
        traceback.print_exc()
        return None

def test_hook_system(domain_manager):
    """Test hook system initialization and basic functionality."""
    print("\n" + "=" * 50)
    print("TESTING HOOK SYSTEM")
    print("=" * 50)
    
    if not domain_manager:
        print("✗ Cannot test hook system without domain manager")
        return
    
    try:
        from clarity.mcp.tool_indexer import MCPToolIndexer
        from clarity.mcp.hook_integration import MCPHookIntegration
        
        # Initialize tool indexer with domain manager
        tool_indexer = MCPToolIndexer(domain_manager)
        print("✓ MCPToolIndexer initialized successfully")
        
        # Initialize hook integration
        hook_integration = MCPHookIntegration(tool_indexer)
        print("✓ MCPHookIntegration initialized successfully")
        
        # Test basic hook functionality
        print("\nTesting hook functionality:")
        print(f"  - Hook config path: {hook_integration.hook_config_path}")
        print(f"  - Hook script path: {hook_integration.hook_script_path}")
        print(f"  - Opportunity patterns loaded: {len(hook_integration.opportunity_patterns)}")
        
        return hook_integration
        
    except Exception as e:
        print(f"✗ Hook system test failed: {e}")
        traceback.print_exc()
        return None

def test_mcp_server_initialization():
    """Test MCP server initialization."""
    print("\n" + "=" * 50)
    print("TESTING MCP SERVER INITIALIZATION")
    print("=" * 50)
    
    try:
        from clarity.mcp.server import MCPServer
        
        # Test basic server creation (don't actually start it)
        print("Testing MCPServer initialization...")
        
        # This might fail due to missing configuration, but we can catch specifics
        server = MCPServer()
        print("✓ MCPServer created successfully")
        
        # Test tool registration
        if hasattr(server, '_register_structured_thinking_tools'):
            print("✓ Structured thinking tools registration method available")
        
        if hasattr(server, '_register_autocode_tools'):
            print("✓ AutoCode tools registration method available")
        
        return server
        
    except Exception as e:
        print(f"✗ MCP server test failed: {e}")
        traceback.print_exc()
        return None

def test_hook_execution_simulation(hook_integration):
    """Simulate hook execution to identify potential issues."""
    print("\n" + "=" * 50)
    print("TESTING HOOK EXECUTION SIMULATION")
    print("=" * 50)
    
    if not hook_integration:
        print("✗ Cannot test hook execution without hook integration")
        return
    
    try:
        # Simulate analyzing a command that might trigger MCP suggestions
        test_commands = [
            "git status",
            "docker ps",
            "npm install",
            "python manage.py migrate",
            "curl -X GET https://api.example.com/users"
        ]
        
        for cmd in test_commands:
            print(f"\nTesting command analysis: '{cmd}'")
            try:
                # This would normally be called by the hook system
                # We're just testing that the pattern matching works
                for category, patterns in hook_integration.opportunity_patterns.items():
                    for pattern in patterns:
                        import re
                        if re.search(pattern, cmd, re.IGNORECASE):
                            print(f"  ✓ Matched pattern '{pattern}' in category '{category}'")
                            break
                    else:
                        continue
                    break
                else:
                    print(f"  - No patterns matched for '{cmd}'")
                    
            except Exception as e:
                print(f"  ✗ Error analyzing command '{cmd}': {e}")
        
        print("\n✓ Hook execution simulation completed")
        
    except Exception as e:
        print(f"✗ Hook execution simulation failed: {e}")
        traceback.print_exc()

def main():
    """Run all tests."""
    print(f"Starting MCP Connection and Hook System Tests")
    print(f"Time: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Run tests
    test_basic_imports()
    domain_manager = test_domain_manager()
    hook_integration = test_hook_system(domain_manager)
    server = test_mcp_server_initialization()
    test_hook_execution_simulation(hook_integration)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print("If you see ✗ errors above, those indicate the specific issues")
    print("that are preventing the hook system from working properly.")
    print("Focus on fixing those imports and initialization issues first.")
    print("=" * 50)

if __name__ == "__main__":
    main()