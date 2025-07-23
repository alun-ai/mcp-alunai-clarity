#!/usr/bin/env python3
"""
Test script specifically for hook execution issues.
This focuses on the actual hook system functionality that's failing.
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, '/app')

def test_core_components():
    """Test the core components needed for hook execution."""
    print("=" * 60)
    print("TESTING CORE COMPONENTS FOR HOOK EXECUTION")
    print("=" * 60)
    
    components = {}
    
    # Test MemoryMcpServer (the actual MCP server)
    try:
        from clarity.mcp.server import MemoryMcpServer
        server = MemoryMcpServer()
        components['server'] = server
        print("✓ MemoryMcpServer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize MemoryMcpServer: {e}")
        traceback.print_exc()
    
    # Test MemoryDomainManager
    try:
        from clarity.domains.manager import MemoryDomainManager
        domain_manager = MemoryDomainManager()
        components['domain_manager'] = domain_manager
        print("✓ MemoryDomainManager initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize MemoryDomainManager: {e}")
        traceback.print_exc()
    
    return components

def test_hook_system_with_correct_classes(components):
    """Test hook system with the correct class names."""
    print("\n" + "=" * 60)
    print("TESTING HOOK SYSTEM WITH CORRECT CLASSES")
    print("=" * 60)
    
    if 'domain_manager' not in components:
        print("✗ Cannot test hook system without domain manager")
        return None
    
    try:
        from clarity.mcp.tool_indexer import MCPToolIndexer
        from clarity.mcp.hook_integration import MCPHookIntegration
        
        # Initialize tool indexer with domain manager
        tool_indexer = MCPToolIndexer(components['domain_manager'])
        print("✓ MCPToolIndexer initialized successfully")
        
        # Initialize hook integration
        hook_integration = MCPHookIntegration(tool_indexer)
        print("✓ MCPHookIntegration initialized successfully")
        
        # Test hook methods
        print("\nTesting hook methods:")
        if hasattr(hook_integration, 'analyze_command'):
            print("✓ analyze_command method available")
        if hasattr(hook_integration, 'generate_mcp_suggestions'):
            print("✓ generate_mcp_suggestions method available")
        if hasattr(hook_integration, 'update_learning_patterns'):
            print("✓ update_learning_patterns method available")
        
        return hook_integration
        
    except Exception as e:
        print(f"✗ Hook system initialization failed: {e}")
        traceback.print_exc()
        return None

def test_hook_execution_flow(hook_integration):
    """Test the actual hook execution flow that's failing."""
    print("\n" + "=" * 60)
    print("TESTING HOOK EXECUTION FLOW")
    print("=" * 60)
    
    if not hook_integration:
        print("✗ Cannot test hook execution without hook integration")
        return
    
    # Test commands that should trigger hook execution
    test_scenarios = [
        {
            'command': 'git status',
            'context': 'user working with git repository',
            'expected': 'should not trigger MCP suggestions'
        },
        {
            'command': 'docker ps -a',
            'context': 'user managing containers',
            'expected': 'might suggest container management tools'
        },
        {
            'command': 'curl -X POST https://api.example.com/users -d "{\\"name\\": \\"test\\"}"',
            'context': 'user making API calls',
            'expected': 'should suggest API testing tools'
        },
        {
            'command': 'psql -h localhost -U user -d database -c "SELECT * FROM users"',
            'context': 'user querying database',
            'expected': 'should suggest database tools'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- Testing scenario: {scenario['command'][:50]}... ---")
        print(f"Context: {scenario['context']}")
        print(f"Expected: {scenario['expected']}")
        
        try:
            # Test pattern matching
            matched_patterns = []
            for category, patterns in hook_integration.opportunity_patterns.items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, scenario['command'], re.IGNORECASE):
                        matched_patterns.append((category, pattern))
            
            if matched_patterns:
                print(f"✓ Matched {len(matched_patterns)} patterns:")
                for category, pattern in matched_patterns[:3]:  # Show first 3
                    print(f"  - Category '{category}': pattern '{pattern}'")
            else:
                print("- No patterns matched")
            
            # Test if analyze_command method exists and works
            if hasattr(hook_integration, 'analyze_command'):
                try:
                    # This might fail, but we want to see the specific error
                    result = hook_integration.analyze_command(scenario['command'])
                    print(f"✓ analyze_command executed successfully: {result}")
                except Exception as e:
                    print(f"✗ analyze_command failed: {e}")
            
        except Exception as e:
            print(f"✗ Scenario test failed: {e}")
            traceback.print_exc()

def test_hook_integration_methods(hook_integration):
    """Test specific hook integration methods that might be failing."""
    print("\n" + "=" * 60)
    print("TESTING HOOK INTEGRATION METHODS")
    print("=" * 60)
    
    if not hook_integration:
        print("✗ Cannot test methods without hook integration")
        return
    
    # Test method existence and basic functionality
    methods_to_test = [
        'analyze_command',
        'generate_mcp_suggestions', 
        'update_learning_patterns',
        'install_hooks',
        'setup_claude_integration'
    ]
    
    for method_name in methods_to_test:
        if hasattr(hook_integration, method_name):
            print(f"✓ Method '{method_name}' is available")
            
            # Try to get method signature if possible
            try:
                import inspect
                method = getattr(hook_integration, method_name)
                sig = inspect.signature(method)
                print(f"  Signature: {method_name}{sig}")
            except Exception as e:
                print(f"  Could not get signature: {e}")
        else:
            print(f"✗ Method '{method_name}' is NOT available")
    
    # Test specific problematic methods
    try:
        print("\nTesting specific method calls:")
        
        # Test pattern loading
        patterns = hook_integration.opportunity_patterns
        print(f"✓ Opportunity patterns loaded: {len(patterns)} categories")
        
        # Test suggestion history
        history = hook_integration.suggestion_history
        print(f"✓ Suggestion history initialized: {len(history)} items")
        
    except Exception as e:
        print(f"✗ Method testing failed: {e}")
        traceback.print_exc()

def main():
    """Run comprehensive hook execution tests."""
    print(f"Hook Execution Test Suite")
    print(f"Time: {datetime.now()}")
    print(f"Purpose: Identify specific issues with hook execution")
    
    # Test core components
    components = test_core_components()
    
    # Test hook system
    hook_integration = test_hook_system_with_correct_classes(components)
    
    # Test hook execution flow
    test_hook_execution_flow(hook_integration)
    
    # Test specific methods
    test_hook_integration_methods(hook_integration)
    
    print("\n" + "=" * 60)
    print("HOOK EXECUTION TEST SUMMARY")
    print("=" * 60)
    print("This test identifies the specific components that are failing")
    print("during hook execution. Focus on fixing the ✗ errors above.")
    print("")
    print("Next steps:")
    print("1. Fix any import/initialization errors")
    print("2. Test individual hook methods")
    print("3. Verify pattern matching logic")
    print("4. Test end-to-end hook execution with Claude")
    print("=" * 60)

if __name__ == "__main__":
    main()