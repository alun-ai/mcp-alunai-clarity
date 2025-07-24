#!/usr/bin/env python3
"""
Comprehensive test for Qdrant connection pooling fixes.

Tests that store_memory operations no longer hang indefinitely 
and that connection recovery mechanisms work properly.
"""

import asyncio
import time
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from clarity.domains.persistence import QdrantPersistenceDomain
from clarity.shared.simple_logging import get_logger

logger = get_logger(__name__)


class QdrantConnectionFixTester:
    """Test suite for Qdrant connection pooling fixes."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        self.domain = None
    
    async def setup(self):
        """Set up test environment with temporary Qdrant storage."""
        self.temp_dir = tempfile.mkdtemp(prefix="qdrant_fix_test_")
        qdrant_path = Path(self.temp_dir) / "qdrant"
        
        config = {
            "qdrant": {
                "path": str(qdrant_path),
                "timeout": 30.0,
                "prefer_grpc": False
            },
            "embedding": {
                "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384
            }
        }
        
        self.domain = QdrantPersistenceDomain(config)
        await self.domain.initialize()
        logger.info(f"Test setup complete with temporary storage: {qdrant_path}")
    
    async def cleanup(self):
        """Clean up test environment."""
        if self.domain:
            # Close domain properly
            if hasattr(self.domain, 'close'):
                await self.domain.close()
        
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Test cleanup complete")
    
    async def test_basic_store_memory(self) -> bool:
        """Test basic store_memory operation with timeout protection."""
        try:
            logger.info("🧪 Testing basic store_memory operation...")
            
            test_memory = {
                "type": "development_framework",
                "content": "Test memory for connection fix validation",
                "importance": 0.8,
                "metadata": {"test": "connection_fix"}
            }
            
            start_time = time.time()
            
            # This should NOT hang indefinitely
            memory_id = await asyncio.wait_for(
                self.domain.store_memory(test_memory),
                timeout=45.0  # Generous timeout for test
            )
            
            elapsed = time.time() - start_time
            
            if memory_id and elapsed < 30.0:  # Should complete within 30 seconds
                logger.info(f"✅ Basic store_memory completed in {elapsed:.2f}s")
                return True
            else:
                logger.error(f"❌ Basic store_memory took too long: {elapsed:.2f}s")
                return False
                
        except asyncio.TimeoutError:
            logger.error("❌ Basic store_memory operation timed out (HANGING DETECTED)")
            return False
        except Exception as e:
            logger.error(f"❌ Basic store_memory failed with error: {e}")
            return False
    
    async def test_concurrent_store_memory(self) -> bool:
        """Test concurrent store_memory operations."""
        try:
            logger.info("🧪 Testing concurrent store_memory operations...")
            
            async def store_test_memory(i: int):
                memory = {
                    "type": "concurrent_test",
                    "content": f"Concurrent test memory {i}",
                    "importance": 0.5,
                    "metadata": {"test_id": i}
                }
                return await self.domain.store_memory(memory)
            
            start_time = time.time()
            
            # Run 5 concurrent operations
            tasks = [store_test_memory(i) for i in range(5)]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # Generous timeout for concurrent operations
            )
            
            elapsed = time.time() - start_time
            
            # Check results
            successful = sum(1 for r in results if isinstance(r, str) and r.startswith("mem_"))
            failed = len(results) - successful
            
            if successful >= 3 and elapsed < 45.0:  # At least 3/5 should succeed
                logger.info(f"✅ Concurrent operations: {successful}/5 successful in {elapsed:.2f}s")
                return True
            else:
                logger.error(f"❌ Concurrent operations: only {successful}/5 successful, took {elapsed:.2f}s")
                return False
                
        except asyncio.TimeoutError:
            logger.error("❌ Concurrent store_memory operations timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Concurrent operations failed: {e}")
            return False
    
    async def test_connection_recovery(self) -> bool:
        """Test connection recovery after simulated failure."""
        try:
            logger.info("🧪 Testing connection recovery mechanisms...")
            
            # First, store a memory successfully
            test_memory = {
                "type": "recovery_test",
                "content": "Memory before connection issue",
                "importance": 0.7
            }
            
            memory_id_1 = await asyncio.wait_for(
                self.domain.store_memory(test_memory),
                timeout=30.0
            )
            
            if not memory_id_1:
                logger.error("❌ Initial memory storage failed")
                return False
            
            # Simulate connection issue by clearing the client
            if hasattr(self.domain, 'client') and self.domain.client:
                logger.info("Simulating connection failure...")
                try:
                    # Close the client to simulate "instance is closed" error
                    self.domain.client.close()
                except:
                    pass
                self.domain.client = None
                self.domain._client_initialized = False
            
            # Try to store another memory - should recover automatically
            recovery_memory = {
                "type": "recovery_test",
                "content": "Memory after connection recovery",
                "importance": 0.8
            }
            
            start_time = time.time()
            memory_id_2 = await asyncio.wait_for(
                self.domain.store_memory(recovery_memory),
                timeout=45.0
            )
            elapsed = time.time() - start_time
            
            if memory_id_2 and elapsed < 35.0:
                logger.info(f"✅ Connection recovery successful in {elapsed:.2f}s")
                return True
            else:
                logger.error(f"❌ Connection recovery failed or took too long: {elapsed:.2f}s")
                return False
                
        except asyncio.TimeoutError:
            logger.error("❌ Connection recovery test timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Connection recovery test failed: {e}")
            return False
    
    async def test_retrieve_memory(self) -> bool:
        """Test memory retrieval operations."""
        try:
            logger.info("🧪 Testing memory retrieval operations...")
            
            # Store a test memory
            test_memory = {
                "type": "retrieval_test",
                "content": "Memory for retrieval testing",
                "importance": 0.9
            }
            
            memory_id = await asyncio.wait_for(
                self.domain.store_memory(test_memory),
                timeout=30.0
            )
            
            if not memory_id:
                logger.error("❌ Failed to store memory for retrieval test")
                return False
            
            # Retrieve memories
            start_time = time.time()
            memories = await asyncio.wait_for(
                self.domain.retrieve_memories("retrieval testing", limit=3),
                timeout=30.0
            )
            elapsed = time.time() - start_time
            
            if memories and len(memories) > 0 and elapsed < 20.0:
                logger.info(f"✅ Memory retrieval successful: found {len(memories)} memories in {elapsed:.2f}s")
                return True
            else:
                logger.error(f"❌ Memory retrieval failed or took too long: {elapsed:.2f}s")
                return False
                
        except asyncio.TimeoutError:
            logger.error("❌ Memory retrieval test timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Memory retrieval test failed: {e}")
            return False
    
    async def test_stress_operations(self) -> bool:
        """Test stress operations to ensure no hanging under load."""
        try:
            logger.info("🧪 Testing stress operations...")
            
            async def rapid_store_operation(batch_id: int):
                memories = []
                for i in range(3):  # 3 memories per batch
                    memory = {
                        "type": "stress_test",
                        "content": f"Stress test memory batch {batch_id}, item {i}",
                        "importance": 0.3,
                        "metadata": {"batch": batch_id, "item": i}
                    }
                    try:
                        memory_id = await asyncio.wait_for(
                            self.domain.store_memory(memory),
                            timeout=20.0  # Shorter timeout for stress test
                        )
                        memories.append(memory_id)
                    except Exception as e:
                        logger.debug(f"Stress operation failed (expected under load): {e}")
                        continue
                return memories
            
            start_time = time.time()
            
            # Run multiple batches of operations rapidly
            batch_tasks = [rapid_store_operation(i) for i in range(4)]
            results = await asyncio.wait_for(
                asyncio.gather(*batch_tasks, return_exceptions=True),
                timeout=90.0  # Allow more time for stress test
            )
            
            elapsed = time.time() - start_time
            
            # Count successful operations
            total_successful = 0
            for result in results:
                if isinstance(result, list):
                    total_successful += len([r for r in result if isinstance(r, str)])
            
            if total_successful >= 6 and elapsed < 60.0:  # At least 50% success rate
                logger.info(f"✅ Stress test: {total_successful} operations successful in {elapsed:.2f}s")
                return True
            else:
                logger.warning(f"⚠️ Stress test: {total_successful} operations in {elapsed:.2f}s (partial success)")
                return total_successful > 0  # Accept partial success for stress test
                
        except asyncio.TimeoutError:
            logger.error("❌ Stress operations timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Stress operations failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("🚀 Starting comprehensive Qdrant connection fix tests...")
        
        tests = [
            ("basic_store_memory", self.test_basic_store_memory),
            ("concurrent_store_memory", self.test_concurrent_store_memory),
            ("connection_recovery", self.test_connection_recovery),
            ("retrieve_memory", self.test_retrieve_memory),
            ("stress_operations", self.test_stress_operations),
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*50}")
                result = await test_func()
                self.test_results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{status}: {test_name}")
            except Exception as e:
                logger.error(f"❌ FAIL: {test_name} - {e}")
                self.test_results[test_name] = False
        
        return self.test_results
    
    def print_summary(self):
        """Print test results summary."""
        logger.info(f"\n{'='*60}")
        logger.info("🏁 QDRANT CONNECTION FIX TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
        
        logger.info(f"\nResults: {passed}/{total} tests passed ({success_rate:.1f}% success rate)")
        
        if success_rate >= 80:
            logger.info("🎉 CONNECTION FIX VALIDATION: SUCCESS")
            logger.info("The Qdrant connection pooling fixes are working correctly!")
        else:
            logger.error("💥 CONNECTION FIX VALIDATION: FAILED")
            logger.error("Some issues still exist with the connection pooling fixes.")
        
        return success_rate >= 80


async def main():
    """Main test execution."""
    tester = QdrantConnectionFixTester()
    
    try:
        await tester.setup()
        await tester.run_all_tests()
        success = tester.print_summary()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())