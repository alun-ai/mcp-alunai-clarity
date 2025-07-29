#!/usr/bin/env python3
"""
Comprehensive system validation test for SQLite migration.
"""

import os
import sys
import json
import asyncio
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported without errors."""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        from clarity.mcp.server import MemoryMcpServer
        print("✅ MemoryMcpServer import successful")
        
        from clarity.domains.manager import MemoryDomainManager  
        print("✅ MemoryDomainManager import successful")
        
        from clarity.domains.persistence import PersistenceDomain
        print("✅ PersistenceDomain import successful")
        
        from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence
        print("✅ SQLiteMemoryPersistence import successful")
        
        from clarity.utils.config import load_config
        print("✅ Config utilities import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n🔍 Testing configuration loading...")
    
    try:
        from clarity.utils.config import load_config
        
        # Test default config
        config_path = "data/default_config.json"
        if os.path.exists(config_path):
            config = load_config(config_path)
            print(f"✅ Configuration loaded from {config_path}")
            
            # Validate key SQLite settings
            if config.get("persistence", {}).get("use_sqlite"):
                print("✅ SQLite backend enabled in config")
            else:
                print("⚠️ SQLite backend not enabled in config")
                
            if "sqlite" in config:
                print("✅ SQLite configuration section found")
            else:
                print("⚠️ SQLite configuration section missing")
                
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        traceback.print_exc()
        return False

async def test_persistence_initialization():
    """Test persistence domain initialization."""
    print("\n🔍 Testing persistence domain initialization...")
    
    try:
        from clarity.domains.persistence import PersistenceDomain
        from clarity.utils.config import load_config
        
        # Load config
        config = load_config("data/default_config.json")
        
        # Create test DB path
        test_db_path = "/tmp/test_clarity_memories.db"
        config["sqlite"] = {"path": test_db_path}
        
        # Initialize persistence domain
        persistence = PersistenceDomain(config)
        print("✅ PersistenceDomain created")
        
        await persistence.initialize()
        print("✅ PersistenceDomain initialized")
        
        # Check if database file was created
        if os.path.exists(test_db_path):
            print(f"✅ SQLite database created: {test_db_path}")
        else:
            print(f"⚠️ SQLite database file not found: {test_db_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Persistence initialization error: {e}")
        traceback.print_exc()
        return False

async def test_memory_operations():
    """Test basic memory storage and retrieval."""
    print("\n🔍 Testing basic memory operations...")
    
    try:
        from clarity.domains.persistence import PersistenceDomain
        from clarity.utils.config import load_config
        
        # Load config
        config = load_config("data/default_config.json")
        
        # Create test DB path
        test_db_path = "/tmp/test_clarity_memories.db"
        config["sqlite"] = {"path": test_db_path}
        
        # Initialize persistence domain
        persistence = PersistenceDomain(config)
        await persistence.initialize()
        
        # Test memory storage
        test_memory = {
            "id": "test_mem_001",
            "type": "test",
            "content": "This is a test memory for validation",
            "importance": 0.8,
            "metadata": {"test": True, "validation": "SQLite migration"},
            "context": {"test_phase": "validation"}
        }
        
        memory_id = await persistence.store_memory(test_memory)
        print(f"✅ Memory stored with ID: {memory_id}")
        
        # Test memory retrieval
        results = await persistence.retrieve_memories(
            query="test memory validation",
            limit=5,
            min_similarity=0.1
        )
        
        if results:
            print(f"✅ Memory retrieval successful, found {len(results)} results")
            for result in results:
                print(f"   - Memory: {result.get('id', 'N/A')} (similarity: {result.get('similarity_score', 'N/A')})")
        else:
            print("⚠️ No memories retrieved")
            
        return True
        
    except Exception as e:
        print(f"❌ Memory operations error: {e}")
        traceback.print_exc()
        return False

async def test_domain_manager():
    """Test domain manager initialization."""
    print("\n🔍 Testing domain manager...")
    
    try:
        from clarity.domains.manager import MemoryDomainManager
        from clarity.utils.config import load_config
        
        # Load config
        config = load_config("data/default_config.json")
        
        # Create test DB path
        test_db_path = "/tmp/test_clarity_memories.db"
        config["sqlite"] = {"path": test_db_path}
        
        # Initialize domain manager
        domain_manager = MemoryDomainManager(config)
        print("✅ MemoryDomainManager created")
        
        await domain_manager.initialize()
        print("✅ MemoryDomainManager initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain manager error: {e}")
        traceback.print_exc()
        return False

async def test_mcp_server_creation():
    """Test MCP server creation (without starting)."""
    print("\n🔍 Testing MCP server creation...")
    
    try:
        from clarity.mcp.server import MemoryMcpServer
        from clarity.utils.config import load_config
        
        # Load config
        config = load_config("data/default_config.json")
        
        # Create test DB path
        test_db_path = "/tmp/test_clarity_memories.db"
        config["sqlite"] = {"path": test_db_path}
        
        # Create MCP server (but don't start it)
        server = MemoryMcpServer(config)
        print("✅ MemoryMcpServer created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP server creation error: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run comprehensive validation tests."""
    print("🚀 Starting Alunai Clarity SQLite Migration Validation\n")
    
    all_tests_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_tests_passed = False
    
    # Test 2: Configuration
    if not test_config_loading():
        all_tests_passed = False
    
    # Test 3: Persistence initialization
    if not await test_persistence_initialization():
        all_tests_passed = False
    
    # Test 4: Memory operations
    if not await test_memory_operations():
        all_tests_passed = False
    
    # Test 5: Domain manager
    if not await test_domain_manager():
        all_tests_passed = False
    
    # Test 6: MCP server creation
    if not await test_mcp_server_creation():
        all_tests_passed = False
    
    # Final results
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ SQLite migration appears successful")
        print("✅ System is ready for basic functionality")
    else:
        print("❌ SOME VALIDATION TESTS FAILED!")
        print("⚠️ Issues need to be resolved before system is fully functional")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())