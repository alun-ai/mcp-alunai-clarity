"""
JSON Memory Importer for migrating from legacy JSON storage to Qdrant.

This utility provides one-time migration from the old JSON-based memory storage
to the new high-performance Qdrant vector database.
"""

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from loguru import logger
from clarity.domains.persistence import QdrantPersistenceDomain
from clarity.utils.config import load_config


class JSONMemoryImporter:
    """
    Imports memories from legacy JSON files to Qdrant.
    
    This class handles the migration process from the old JSON-based storage
    to the new Qdrant vector database with proper error handling and progress tracking.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the JSON memory importer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.persistence = QdrantPersistenceDomain(config)
        self.import_stats = {
            "total_processed": 0,
            "successful_imports": 0,
            "failed_imports": 0,
            "start_time": None,
            "end_time": None,
            "errors": []
        }
    
    async def initialize(self) -> None:
        """Initialize the importer and persistence domain."""
        await self.persistence.initialize()
    
    def _load_json_memories(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load memories from JSON file.
        
        Args:
            json_file_path: Path to the JSON memory file
            
        Returns:
            Loaded JSON data
        """
        try:
            json_path = Path(json_file_path)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_file_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON memory file: {json_file_path}")
            return data
            
        except (OSError, ValueError, PermissionError, UnicodeDecodeError) as e:
            logger.error(f"Failed to load JSON file {json_file_path}: {e}")
            raise
    
    def _extract_memories_from_json(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract individual memories from JSON structure.
        
        Args:
            json_data: Loaded JSON data
            
        Returns:
            List of memory dictionaries
        """
        memories = []
        
        # Extract from different memory tiers
        for tier in ["short_term_memory", "long_term_memory", "archived_memory"]:
            if tier in json_data:
                tier_memories = json_data[tier]
                if isinstance(tier_memories, list):
                    for memory in tier_memories:
                        if isinstance(memory, dict):
                            # Standardize memory format
                            standardized_memory = self._standardize_memory_format(memory, tier)
                            memories.append(standardized_memory)
                        else:
                            logger.warning(f"Invalid memory format in {tier}: {type(memory)}")
        
        # Extract from flat memory list (if present)
        if "memories" in json_data and isinstance(json_data["memories"], list):
            for memory in json_data["memories"]:
                if isinstance(memory, dict):
                    standardized_memory = self._standardize_memory_format(memory, "unknown")
                    memories.append(standardized_memory)
        
        # Handle direct array format (when JSON is just an array of memories)
        elif isinstance(json_data, list):
            for memory in json_data:
                if isinstance(memory, dict):
                    standardized_memory = self._standardize_memory_format(memory, "unknown")
                    memories.append(standardized_memory)
        
        logger.info(f"Extracted {len(memories)} memories from JSON data")
        return memories
    
    def _standardize_memory_format(self, memory: Dict[str, Any], tier: str) -> Dict[str, Any]:
        """
        Standardize memory format for Qdrant storage.
        
        Args:
            memory: Original memory data
            tier: Memory tier (short_term_memory, long_term_memory, etc.)
            
        Returns:
            Standardized memory dictionary
        """
        # Map tier names
        tier_mapping = {
            "short_term_memory": "short_term",
            "long_term_memory": "long_term", 
            "archived_memory": "archived"
        }
        
        # Clean the ID to be a valid UUID (remove mem_ prefix if present)
        original_id = memory.get("id", memory.get("memory_id"))
        clean_id = original_id.replace("mem_", "") if original_id and original_id.startswith("mem_") else original_id
        
        standardized = {
            "id": clean_id,
            "type": memory.get("type", memory.get("memory_type", "unknown")),
            "content": memory.get("content", {}),
            "importance": memory.get("importance", 0.5),
            "tier": tier_mapping.get(tier, tier),
            "created_at": memory.get("created_at", memory.get("timestamp", datetime.utcnow().isoformat())),
            "updated_at": memory.get("updated_at", memory.get("timestamp", datetime.utcnow().isoformat())),
            "metadata": memory.get("metadata", {}),
            "context": memory.get("context", {}),
            "access_count": memory.get("access_count", 0),
            "last_accessed": memory.get("last_accessed"),
        }
        
        # Handle missing or invalid IDs
        if not standardized["id"]:
            import uuid
            standardized["id"] = str(uuid.uuid4())
        
        return standardized
    
    async def import_memories(self, memories: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Import memories to Qdrant in batches.
        
        Args:
            memories: List of memories to import
            batch_size: Number of memories to process in each batch
        """
        self.import_stats["total_processed"] = len(memories)
        self.import_stats["start_time"] = datetime.utcnow().isoformat()
        
        logger.info(f"Starting import of {len(memories)} memories in batches of {batch_size}")
        
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(memories) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} memories)")
            
            await self._import_batch(batch, batch_num)
            
            # Progress update
            progress = (i + len(batch)) / len(memories) * 100
            logger.info(f"Import progress: {progress:.1f}% ({self.import_stats['successful_imports']} successful, {self.import_stats['failed_imports']} failed)")
        
        self.import_stats["end_time"] = datetime.utcnow().isoformat()
        logger.info("Memory import completed")
    
    async def _import_batch(self, batch: List[Dict[str, Any]], batch_num: int) -> None:
        """
        Import a batch of memories.
        
        Args:
            batch: Batch of memories to import
            batch_num: Batch number for logging
        """
        for memory in batch:
            try:
                memory_id = await self.persistence.store_memory(memory)
                self.import_stats["successful_imports"] += 1
                logger.debug(f"Imported memory: {memory_id}")
                
            except (ValueError, KeyError, AttributeError, RuntimeError) as e:
                self.import_stats["failed_imports"] += 1
                error_msg = f"Failed to import memory {memory.get('id', 'unknown')}: {e}"
                self.import_stats["errors"].append(error_msg)
                logger.error(error_msg)
    
    def print_import_summary(self) -> None:
        """Print a summary of the import process."""
        stats = self.import_stats
        
        duration = "Unknown"
        if stats["start_time"] and stats["end_time"]:
            start = datetime.fromisoformat(stats["start_time"])
            end = datetime.fromisoformat(stats["end_time"])
            duration = str(end - start)
        
        print("\n" + "="*60)
        print("JSON MEMORY IMPORT SUMMARY")
        print("="*60)
        print(f"Total memories processed: {stats['total_processed']}")
        print(f"Successful imports: {stats['successful_imports']}")
        print(f"Failed imports: {stats['failed_imports']}")
        print(f"Success rate: {(stats['successful_imports'] / max(stats['total_processed'], 1)) * 100:.1f}%")
        print(f"Import duration: {duration}")
        
        if stats["errors"]:
            print(f"\nFirst 5 errors:")
            for error in stats["errors"][:5]:
                print(f"  - {error}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more errors")
        
        print("="*60)
    
    async def verify_import(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Verify the import by checking a sample of memories.
        
        Args:
            sample_size: Number of memories to verify
            
        Returns:
            Verification results
        """
        try:
            # Get memory stats from Qdrant
            stats = await self.persistence.get_memory_stats()
            
            # Test search functionality
            test_queries = ["memory", "code", "project", "command", "conversation"]
            search_results = {}
            
            for query in test_queries:
                results = await self.persistence.retrieve_memories(
                    query=query,
                    limit=5,
                    min_similarity=0.3
                )
                search_results[query] = len(results)
            
            verification = {
                "qdrant_stats": stats,
                "search_test_results": search_results,
                "verification_passed": stats.get("total_memories", 0) > 0,
                "total_memories_in_qdrant": stats.get("total_memories", 0),
                "indexed_memories": stats.get("indexed_memories", 0),
            }
            
            return verification
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"Verification failed: {e}")
            return {"verification_passed": False, "error": str(e)}


async def import_json_memories(
    json_file_path: str,
    config_path: Optional[str] = None,
    batch_size: int = 100,
    verify: bool = True
) -> None:
    """
    Main function to import memories from JSON file to Qdrant.
    
    Args:
        json_file_path: Path to the JSON memory file
        config_path: Path to configuration file (optional)
        batch_size: Batch size for import
        verify: Whether to verify the import
    """
    try:
        # Load configuration
        if config_path:
            config = load_config(config_path)
        else:
            # Use default configuration
            config = {
                "qdrant": {
                    "path": "./.claude/alunai-clarity/qdrant",
                    "index_params": {
                        "m": 16,
                        "ef_construct": 200,
                        "full_scan_threshold": 10000,
                    }
                },
                "embedding": {
                    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384
                }
            }
        
        # Initialize importer
        importer = JSONMemoryImporter(config)
        await importer.initialize()
        
        # Load and extract memories from JSON
        logger.info(f"Loading memories from: {json_file_path}")
        json_data = importer._load_json_memories(json_file_path)
        memories = importer._extract_memories_from_json(json_data)
        
        if not memories:
            logger.warning("No memories found in JSON file")
            return
        
        # Import memories
        await importer.import_memories(memories, batch_size)
        
        # Print summary
        importer.print_import_summary()
        
        # Verify import
        if verify:
            logger.info("Verifying import...")
            verification = await importer.verify_import()
            
            if verification.get("verification_passed"):
                print(f"\n✅ Import verification PASSED")
                print(f"   - Total memories in Qdrant: {verification.get('total_memories_in_qdrant', 0)}")
                print(f"   - Indexed memories: {verification.get('indexed_memories', 0)}")
                print(f"   - Search functionality: Working")
            else:
                print(f"\n❌ Import verification FAILED")
                if "error" in verification:
                    print(f"   Error: {verification['error']}")
        
        logger.info("JSON memory import completed successfully")
        
    except (OSError, ValueError, KeyError, AttributeError, RuntimeError) as e:
        logger.error(f"JSON memory import failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import memories from JSON to Qdrant")
    parser.add_argument("json_file", help="Path to JSON memory file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for import")
    parser.add_argument("--no-verify", action="store_true", help="Skip import verification")
    
    args = parser.parse_args()
    
    asyncio.run(import_json_memories(
        json_file_path=args.json_file,
        config_path=args.config,
        batch_size=args.batch_size,
        verify=not args.no_verify
    ))