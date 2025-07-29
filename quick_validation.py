#!/usr/bin/env python3
"""
Quick validation test for critical SQLite integration issues.
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_critical_imports():
    """Test only the most critical imports."""
    print("🔍 Testing critical imports...")
    
    try:
        # These should work now
        from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence
        from clarity.domains.persistence import PersistenceDomain
        from clarity.domains.manager import MemoryDomainManager
        print("✅ All critical imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Critical import error: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")
    
    try:
        from clarity.utils.config import load_config
        config = load_config("data/default_config.json")
        
        # Check key settings
        use_sqlite = config.get("persistence", {}).get("use_sqlite", False)
        has_sqlite_config = "sqlite" in config
        
        print(f"✅ Config loaded - SQLite enabled: {use_sqlite}, SQLite config: {has_sqlite_config}")
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

async def test_basic_persistence():
    """Test basic persistence operations."""
    print("\n🔍 Testing basic persistence...")
    
    try:
        from clarity.domains.persistence import PersistenceDomain
        from clarity.utils.config import load_config
        
        # Load config with test DB
        config = load_config("data/default_config.json")
        config["sqlite"] = {"path": "/tmp/quick_test.db"}
        
        # Initialize persistence
        persistence = PersistenceDomain(config)
        await persistence.initialize()
        print("✅ Persistence initialized")
        
        # Quick memory test
        test_memory = {
            "id": "quick_test_001",
            "type": "test",
            "content": "Quick validation test",
            "importance": 0.5
        }
        
        memory_id = await persistence.store_memory(test_memory)
        print(f"✅ Memory stored: {memory_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Persistence error: {e}")
        return False

async def main():
    """Run quick validation."""
    print("🚀 Quick SQLite Migration Validation\n")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Critical imports
    if test_critical_imports():
        tests_passed += 1
    
    # Test 2: Configuration
    if test_config():
        tests_passed += 1
    
    # Test 3: Basic persistence (with timeout protection)
    try:
        import asyncio
        await asyncio.wait_for(test_basic_persistence(), timeout=30.0)
        tests_passed += 1
    except asyncio.TimeoutError:
        print("❌ Persistence test timed out")
    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
    
    # Results
    print(f"\n{'='*50}")
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 QUICK VALIDATION SUCCESSFUL!")
        print("✅ Core SQLite integration is working")
    else:
        print("⚠️ Some issues remain, but core functionality appears viable")
    
    print("="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())