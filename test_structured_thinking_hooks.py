#!/usr/bin/env python3
"""
Test script for structured thinking hook integration.

Validates that the hook system properly triggers structured thinking processes
similar to mcp-sequential-thinking approach.
"""

import asyncio
import json
import sys
from loguru import logger
from clarity.autocode.hook_manager import HookManager
from clarity.autocode.hooks import AutoCodeHooks
from clarity.domains.memory import MemoryDomain


async def test_structured_thinking_detection():
    """Test automatic structured thinking detection and triggering."""
    
    logger.info("Testing structured thinking hook integration...")
    
    # Create mock domain manager
    class MockDomainManager:
        def __init__(self):
            self.persistence_domain = MemoryDomain()
    
    # Initialize hook manager
    domain_manager = MockDomainManager()
    autocode_hooks = AutoCodeHooks()
    hook_manager = HookManager(domain_manager, autocode_hooks)
    
    # Test cases that should trigger structured thinking
    test_cases = [
        {
            "content": "I need to implement a user authentication system with JWT tokens and password hashing",
            "should_trigger": True,
            "description": "Complex implementation task"
        },
        {
            "content": "Help me debug this error in my API endpoint",
            "should_trigger": True,
            "description": "Problem-solving request"
        },
        {
            "content": "What's the best way to architect a microservices system for handling user data?",
            "should_trigger": True,
            "description": "Design and architecture question"
        },
        {
            "content": "How do I plan and build a real-time chat application with websockets?",
            "should_trigger": True,
            "description": "Multi-component planning task"
        },
        {
            "content": "Hello",
            "should_trigger": False,
            "description": "Simple greeting"
        },
        {
            "content": "What is Python?",
            "should_trigger": False,
            "description": "Simple factual question"
        }
    ]
    
    logger.info(f"Testing {len(test_cases)} structured thinking detection scenarios...")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        content = test_case["content"]
        expected = test_case["should_trigger"]
        description = test_case["description"]
        
        # Test the detection logic
        should_trigger = hook_manager._should_trigger_structured_thinking(content)
        
        result = {
            "test_case": i,
            "description": description,
            "content": content[:50] + "..." if len(content) > 50 else content,
            "expected": expected,
            "actual": should_trigger,
            "passed": should_trigger == expected
        }
        
        results.append(result)
        
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        logger.info(f"Test {i}: {status} - {description}")
        logger.info(f"  Expected: {expected}, Got: {should_trigger}")
        
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    logger.info(f"\nStructured Thinking Detection Results: {passed}/{total} tests passed")
    
    return results


async def test_hook_registration():
    """Test that structured thinking hooks are properly registered."""
    
    logger.info("Testing structured thinking hook registration...")
    
    # Create mock domain manager
    class MockDomainManager:
        def __init__(self):
            self.persistence_domain = MemoryDomain()
    
    # Initialize hook manager
    domain_manager = MockDomainManager()
    autocode_hooks = AutoCodeHooks()
    hook_manager = HookManager(domain_manager, autocode_hooks)
    
    # Check if structured thinking hooks are registered
    expected_hooks = [
        "process_structured_thought",
        "generate_thinking_summary", 
        "continue_thinking_process"
    ]
    
    registered_hooks = list(hook_manager.tool_hooks.keys())
    logger.info(f"Registered tool hooks: {registered_hooks}")
    
    results = []
    for hook_name in expected_hooks:
        is_registered = hook_name in hook_manager.tool_hooks
        result = {
            "hook_name": hook_name,
            "registered": is_registered,
            "passed": is_registered
        }
        results.append(result)
        
        status = "✓ PASS" if is_registered else "✗ FAIL"
        logger.info(f"Hook '{hook_name}': {status}")
    
    # Check structured thinking extension initialization
    extension_initialized = hook_manager.structured_thinking_extension is not None
    logger.info(f"Structured thinking extension initialized: {extension_initialized}")
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    logger.info(f"Hook Registration Results: {passed}/{total} hooks properly registered")
    
    return results, extension_initialized


async def test_conversation_hook_integration():
    """Test integration with conversation message hooks."""
    
    logger.info("Testing conversation message hook integration...")
    
    # Create mock domain manager
    class MockDomainManager:
        def __init__(self):
            self.persistence_domain = MemoryDomain()
    
    # Initialize hook manager
    domain_manager = MockDomainManager()
    autocode_hooks = AutoCodeHooks()
    hook_manager = HookManager(domain_manager, autocode_hooks)
    
    # Test message that should trigger structured thinking
    test_message = "I need to design and implement a comprehensive user authentication system with JWT, OAuth2, and multi-factor authentication support"
    
    # Simulate conversation message hook
    context = {
        "data": {
            "role": "user",
            "content": test_message,
            "message_id": "test_msg_123"
        }
    }
    
    logger.info(f"Testing conversation hook with message: '{test_message[:50]}...'")
    
    # This would normally be called by the MCP server
    try:
        await hook_manager._on_conversation_message(context)
        logger.info("✓ Conversation message hook executed successfully")
        integration_success = True
    except Exception as e:
        logger.error(f"✗ Conversation message hook failed: {e}")
        integration_success = False
    
    return integration_success


async def main():
    """Run all structured thinking hook tests."""
    
    logger.info("Starting structured thinking hook integration tests...")
    
    # Test 1: Detection logic
    detection_results = await test_structured_thinking_detection()
    
    # Test 2: Hook registration 
    hook_results, extension_init = await test_hook_registration()
    
    # Test 3: Conversation integration
    conversation_success = await test_conversation_hook_integration()
    
    # Overall results
    detection_passed = sum(1 for r in detection_results if r["passed"])
    detection_total = len(detection_results)
    
    hook_passed = sum(1 for r in hook_results if r["passed"])
    hook_total = len(hook_results)
    
    logger.info("\n" + "="*50)
    logger.info("STRUCTURED THINKING HOOK INTEGRATION TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Detection Logic: {detection_passed}/{detection_total} tests passed")
    logger.info(f"Hook Registration: {hook_passed}/{hook_total} hooks registered")
    logger.info(f"Extension Initialization: {'✓ SUCCESS' if extension_init else '✗ FAILED'}")
    logger.info(f"Conversation Integration: {'✓ SUCCESS' if conversation_success else '✗ FAILED'}")
    
    # Determine overall success
    all_detection_passed = detection_passed == detection_total
    all_hooks_registered = hook_passed == hook_total
    
    overall_success = (all_detection_passed and all_hooks_registered and 
                      extension_init and conversation_success)
    
    logger.info(f"\nOverall Integration: {'✓ SUCCESS' if overall_success else '✗ FAILED'}")
    
    if overall_success:
        logger.info("Structured thinking hooks are properly integrated and working like mcp-sequential-thinking!")
        return 0
    else:
        logger.error("Some structured thinking hook integration tests failed.")
        return 1


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)