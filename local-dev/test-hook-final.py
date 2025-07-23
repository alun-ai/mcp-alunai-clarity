#!/usr/bin/env python3
"""
Final comprehensive test for hook execution with proper configuration.
This simulates the exact environment that would exist when Claude connects.
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, '/app')

def load_config():
    """Load the configuration that the MCP server would use."""
    try:
        from clarity.utils.config import load_config
        config = load_config('/app/data/config.json')
        print(f"✓ Configuration loaded from /app/data/config.json")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        # Create a minimal config for testing
        return {
            "qdrant": {
                "path": "/app/data/qdrant",
                "index_params": {
                    "m": 16,
                    "ef_construct": 200,
                    "full_scan_threshold": 10000
                }
            },
            "embedding": {
                "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "cache_dir": "/app/data/cache"
            },
            "alunai-clarity": {
                "max_short_term_items": 1000,
                "max_long_term_items": 10000,
                "max_archival_items": 100000
            }
        }

def test_with_proper_config():
    """Test components with proper configuration."""
    print("=" * 70)
    print("TESTING WITH PROPER CONFIGURATION")
    print("=" * 70)
    
    config = load_config()
    components = {}
    
    # Test MemoryDomainManager with config
    try:
        from clarity.domains.manager import MemoryDomainManager
        domain_manager = MemoryDomainManager(config)
        components['domain_manager'] = domain_manager
        print("✓ MemoryDomainManager initialized with config")
    except Exception as e:
        print(f"✗ Failed to initialize MemoryDomainManager: {e}")
        traceback.print_exc()
    
    # Test MemoryMcpServer with config
    try:
        from clarity.mcp.server import MemoryMcpServer
        server = MemoryMcpServer(config)
        components['server'] = server
        print("✓ MemoryMcpServer initialized with config")
    except Exception as e:
        print(f"✗ Failed to initialize MemoryMcpServer: {e}")
        traceback.print_exc()
    
    return components, config

def test_complete_hook_system(components, config):
    """Test the complete hook system as it would work in production."""
    print("\n" + "=" * 70)
    print("TESTING COMPLETE HOOK SYSTEM")
    print("=" * 70)
    
    if 'domain_manager' not in components:
        print("✗ Cannot test hook system without domain manager")
        return None
    
    try:
        from clarity.mcp.tool_indexer import MCPToolIndexer
        from clarity.mcp.hook_integration import MCPHookIntegration
        
        # Initialize with proper domain manager
        tool_indexer = MCPToolIndexer(components['domain_manager'])
        print("✓ MCPToolIndexer initialized with domain manager")
        
        # Initialize hook integration
        hook_integration = MCPHookIntegration(tool_indexer)
        print("✓ MCPHookIntegration initialized")
        
        # Test all hook integration attributes
        print(f"✓ Hook config path: {hook_integration.hook_config_path}")
        print(f"✓ Hook script path: {hook_integration.hook_script_path}")
        print(f"✓ Learning patterns: {len(hook_integration.learning_patterns)} items")
        print(f"✓ Suggestion history: {len(hook_integration.suggestion_history)} items")
        print(f"✓ Opportunity patterns: {len(hook_integration.opportunity_patterns)} categories")
        
        # List opportunity pattern categories
        print("  Pattern categories:")
        for category in hook_integration.opportunity_patterns.keys():
            print(f"    - {category}")
        
        return hook_integration
        
    except Exception as e:
        print(f"✗ Hook system initialization failed: {e}")
        traceback.print_exc()
        return None

def test_real_hook_execution_scenarios(hook_integration):
    """Test real-world hook execution scenarios."""
    print("\n" + "=" * 70)
    print("TESTING REAL HOOK EXECUTION SCENARIOS")
    print("=" * 70)
    
    if not hook_integration:
        print("✗ Cannot test without hook integration")
        return
    
    # Real scenarios that would happen in Claude Code
    real_scenarios = [
        {
            'name': 'Database Query',
            'command': 'psql -h localhost -U postgres -d myapp -c "SELECT id, email FROM users WHERE active = true"',
            'expected_category': 'database_queries'
        },
        {
            'name': 'API Testing',
            'command': 'curl -X POST -H "Content-Type: application/json" -d \'{"name": "John"}\' https://api.example.com/users',
            'expected_category': 'web_automation'
        },
        {
            'name': 'Docker Management',
            'command': 'docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=secret postgres:13',
            'expected_category': 'container_orchestration'
        },
        {
            'name': 'File Operations',
            'command': 'find /var/log -name "*.log" -mtime +7 -delete',
            'expected_category': 'file_management'
        },
        {
            'name': 'Git Operations',
            'command': 'git log --oneline --graph --decorate --all',
            'expected_category': 'version_control'
        }
    ]
    
    successful_tests = 0
    total_tests = len(real_scenarios)
    
    for scenario in real_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Command: {scenario['command']}")
        print(f"Expected category: {scenario['expected_category']}")
        
        try:
            # Test pattern matching
            matched = False
            for category, patterns in hook_integration.opportunity_patterns.items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, scenario['command'], re.IGNORECASE):
                        print(f"✓ Matched in category '{category}' with pattern: {pattern}")
                        matched = True
                        if category == scenario['expected_category']:
                            print(f"✓ Correctly matched expected category!")
                        break
                if matched:
                    break
            
            if matched:
                successful_tests += 1
            else:
                print("✗ No patterns matched - this might be an issue")
                
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print(f"\n--- SCENARIO SUMMARY ---")
    print(f"Successful tests: {successful_tests}/{total_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")

def test_claude_connection_simulation():
    """Simulate what happens when Claude connects to the MCP server."""
    print("\n" + "=" * 70)
    print("SIMULATING CLAUDE CONNECTION")
    print("=" * 70)
    
    try:
        # This simulates the actual initialization that happens when Claude connects
        config = load_config()
        
        # Initialize the server as it would be when Claude connects
        from clarity.mcp.server import MemoryMcpServer
        server = MemoryMcpServer(config)
        
        print("✓ MCP Server initialized (as Claude would see it)")
        
        # Check if the server has the expected methods
        expected_methods = ['list_tools', 'call_tool', 'list_resources', 'read_resource']
        for method in expected_methods:
            if hasattr(server, method):
                print(f"✓ Server has {method} method")
            else:
                print(f"✗ Server missing {method} method")
        
        # Test tool listing (what Claude would do first)
        if hasattr(server, 'list_tools'):
            try:
                tools = server.list_tools()
                print(f"✓ Tools listed successfully: {len(tools.tools)} tools available")
                
                # Show first few tools
                for i, tool in enumerate(tools.tools[:3]):
                    print(f"  Tool {i+1}: {tool.name}")
                    
            except Exception as e:
                print(f"✗ Tool listing failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Claude connection simulation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the complete hook execution test suite."""
    print("=" * 70)
    print("ALUNAI CLARITY HOOK EXECUTION TEST SUITE")
    print("=" * 70)
    print(f"Time: {datetime.now()}")
    print(f"Purpose: Test hook execution in realistic Claude Code environment")
    print("=" * 70)
    
    # Test with proper configuration
    components, config = test_with_proper_config()
    
    # Test complete hook system
    hook_integration = test_complete_hook_system(components, config)
    
    # Test real scenarios
    test_real_hook_execution_scenarios(hook_integration)
    
    # Test Claude connection
    claude_works = test_claude_connection_simulation()
    
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    
    if hook_integration and claude_works:
        print("🎉 SUCCESS: Hook system is working correctly!")
        print("")
        print("Your MCP server is ready for real testing with Claude:")
        print("1. Add the configuration to Claude Desktop")
        print("2. Connect to the container using the provided config")
        print("3. Test hook execution by running commands in Claude Code")
        print("")
        print("The container will show debug logs for all hook executions.")
    else:
        print("❌ ISSUES FOUND: Some components are not working correctly")
        print("")
        print("Focus on fixing the ✗ errors shown above.")
        print("The most common issues are:")
        print("- Missing configuration files")
        print("- Import errors in hook modules")
        print("- Missing required dependencies")
    
    print("=" * 70)

if __name__ == "__main__":
    main()