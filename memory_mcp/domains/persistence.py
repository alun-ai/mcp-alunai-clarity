"""
Persistence Domain for storage and retrieval of memories.

The Persistence Domain is responsible for:
- File system operations
- Vector embedding generation and storage
- Index management
- Memory file structure 
- Backup and recovery
- Efficient storage formats
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class PersistenceDomain:
    """
    Manages the storage and retrieval of memories.
    
    This domain handles file operations, embedding generation,
    and index management for the memory system.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the persistence domain.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.memory_file_path = self.config["memory"].get("file_path", "memory.json")
        self.embedding_model_name = self.config["embedding"].get("default_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dimensions = self.config["embedding"].get("dimensions", 384)
        
        # Will be initialized during initialize()
        self.embedding_model = None
        self.memory_data = None
    
    async def initialize(self) -> None:
        """Initialize the persistence domain."""
        logger.info("Initializing Persistence Domain")
        logger.info(f"Using memory file: {self.memory_file_path}")
        
        # Create memory file directory if it doesn't exist
        os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
        
        # Load memory file or create if it doesn't exist
        self.memory_data = await self._load_memory_file()
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        logger.info("Persistence Domain initialized")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        # Generate embedding
        embedding = self.embedding_model.encode(text)
        
        # Convert to list of floats for JSON serialization
        return embedding.tolist()
    
    async def store_memory(self, memory: Dict[str, Any], tier: str = "short_term") -> None:
        """
        Store a memory.
        
        Args:
            memory: Memory to store
            tier: Memory tier (short_term, long_term, archived)
        """
        # Ensure memory has all required fields
        if "id" not in memory:
            raise ValueError("Memory must have an ID")
        
        # Add to appropriate tier
        valid_tiers = ["short_term", "long_term", "archived"]
        if tier not in valid_tiers:
            raise ValueError(f"Invalid tier: {tier}. Must be one of {valid_tiers}")
        
        tier_key = f"{tier}_memory"
        if tier_key not in self.memory_data:
            self.memory_data[tier_key] = []
        
        # Check for existing memory with same ID
        existing_index = None
        for i, existing_memory in enumerate(self.memory_data[tier_key]):
            if existing_memory.get("id") == memory["id"]:
                existing_index = i
                break
        
        if existing_index is not None:
            # Update existing memory
            self.memory_data[tier_key][existing_index] = memory
        else:
            # Add new memory
            self.memory_data[tier_key].append(memory)
        
        # Update memory index if embedding exists
        if "embedding" in memory:
            await self._update_memory_index(memory, tier)
        
        # Update memory stats
        self._update_memory_stats()
        
        # Save memory file
        await self._save_memory_file()
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory dict or None if not found
        """
        # Check all tiers
        for tier in ["short_term_memory", "long_term_memory", "archived_memory"]:
            if tier not in self.memory_data:
                continue
                
            for memory in self.memory_data[tier]:
                if memory.get("id") == memory_id:
                    return memory
        
        return None
    
    async def get_memory_tier(self, memory_id: str) -> Optional[str]:
        """
        Get the tier of a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory tier or None if not found
        """
        # Check all tiers
        for tier_key in ["short_term_memory", "long_term_memory", "archived_memory"]:
            if tier_key not in self.memory_data:
                continue
                
            for memory in self.memory_data[tier_key]:
                if memory.get("id") == memory_id:
                    # Convert tier_key to tier name
                    return tier_key.replace("_memory", "")
        
        return None
    
    async def update_memory(self, memory: Dict[str, Any], tier: str) -> None:
        """
        Update an existing memory.
        
        Args:
            memory: Updated memory dict
            tier: Memory tier
        """
        # Get current tier
        current_tier = await self.get_memory_tier(memory["id"])
        
        if current_tier is None:
            # Memory doesn't exist, store as new
            await self.store_memory(memory, tier)
            return
        
        if current_tier == tier:
            # Same tier, just update the memory
            tier_key = f"{tier}_memory"
            for i, existing_memory in enumerate(self.memory_data[tier_key]):
                if existing_memory.get("id") == memory["id"]:
                    self.memory_data[tier_key][i] = memory
                    break
            
            # Update memory index if embedding exists
            if "embedding" in memory:
                await self._update_memory_index(memory, tier)
            
            # Save memory file
            await self._save_memory_file()
        else:
            # Different tier, remove from old tier and add to new tier
            old_tier_key = f"{current_tier}_memory"
            
            # Remove from old tier
            self.memory_data[old_tier_key] = [
                m for m in self.memory_data[old_tier_key]
                if m.get("id") != memory["id"]
            ]
            
            # Add to new tier
            await self.store_memory(memory, tier)
    
    async def delete_memories(self, memory_ids: List[str]) -> bool:
        """
        Delete memories.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            Success flag
        """
        deleted_count = 0
        
        # Check all tiers
        for tier_key in ["short_term_memory", "long_term_memory", "archived_memory"]:
            if tier_key not in self.memory_data:
                continue
            
            # Filter out memories to delete
            original_count = len(self.memory_data[tier_key])
            self.memory_data[tier_key] = [
                memory for memory in self.memory_data[tier_key]
                if memory.get("id") not in memory_ids
            ]
            deleted_count += original_count - len(self.memory_data[tier_key])
        
        # Update memory index
        for memory_id in memory_ids:
            await self._remove_from_memory_index(memory_id)
        
        # Update memory stats
        self._update_memory_stats()
        
        # Save memory file
        await self._save_memory_file()
        
        return deleted_count > 0
    
    async def search_memories(
        self,
        embedding: List[float],
        limit: int = 5,
        types: Optional[List[str]] = None,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for memories using vector similarity.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            types: Memory types to include (None for all)
            min_similarity: Minimum similarity score
            
        Returns:
            List of matching memories with similarity scores
        """
        # Convert embedding to numpy array
        query_embedding = np.array(embedding)
        
        # Get all memories with embeddings
        memories_with_embeddings = []
        
        for tier_key in ["short_term_memory", "long_term_memory", "archived_memory"]:
            if tier_key not in self.memory_data:
                continue
                
            for memory in self.memory_data[tier_key]:
                if "embedding" in memory:
                    # Filter by type if specified
                    if types and memory.get("type") not in types:
                        continue
                        
                    memories_with_embeddings.append(memory)
        
        # Calculate similarities
        results_with_scores = []
        
        for memory in memories_with_embeddings:
            memory_embedding = np.array(memory["embedding"])
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, memory_embedding)
            
            if similarity >= min_similarity:
                # Create a copy to avoid modifying the original
                result = memory.copy()
                result["similarity"] = float(similarity)
                results_with_scores.append(result)
        
        # Sort by similarity
        results_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit results
        return results_with_scores[:limit]
    
    async def list_memories(
        self,
        types: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
        tier: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List memories with filtering options.
        
        Args:
            types: Memory types to include (None for all)
            limit: Maximum number of memories to return
            offset: Offset for pagination
            tier: Memory tier to filter by (None for all)
            
        Returns:
            List of memories
        """
        all_memories = []
        
        # Determine which tiers to include
        tiers_to_include = []
        if tier:
            tiers_to_include = [f"{tier}_memory"]
        else:
            tiers_to_include = ["short_term_memory", "long_term_memory", "archived_memory"]
        
        # Collect memories from selected tiers
        for tier_key in tiers_to_include:
            if tier_key not in self.memory_data:
                continue
                
            for memory in self.memory_data[tier_key]:
                # Filter by type if specified
                if types and memory.get("type") not in types:
                    continue
                    
                # Add tier info
                memory_copy = memory.copy()
                memory_copy["tier"] = tier_key.replace("_memory", "")
                all_memories.append(memory_copy)
        
        # Sort by creation time (newest first)
        all_memories.sort(
            key=lambda m: m.get("created_at", ""),
            reverse=True
        )
        
        # Apply pagination
        paginated_memories = all_memories[offset:offset+limit]
        
        return paginated_memories
    
    async def get_metadata(self, key: str) -> Optional[str]:
        """
        Get metadata value.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        metadata = self.memory_data.get("metadata", {})
        return metadata.get(key)
    
    async def set_metadata(self, key: str, value: str) -> None:
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        if "metadata" not in self.memory_data:
            self.memory_data["metadata"] = {}
            
        self.memory_data["metadata"][key] = value
        
        # Save memory file
        await self._save_memory_file()
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Memory statistics
        """
        return self.memory_data.get("metadata", {}).get("memory_stats", {})
    
    async def _load_memory_file(self) -> Dict[str, Any]:
        """
        Load the memory file.
        
        Returns:
            Memory data
        """
        if not os.path.exists(self.memory_file_path):
            logger.info(f"Memory file not found, creating new file: {self.memory_file_path}")
            return self._create_empty_memory_file()
        
        try:
            with open(self.memory_file_path, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded memory file with {self._count_memories(data)} memories")
                return data
        except json.JSONDecodeError:
            logger.error(f"Error parsing memory file: {self.memory_file_path}")
            logger.info("Creating new memory file")
            return self._create_empty_memory_file()
    
    def _create_empty_memory_file(self) -> Dict[str, Any]:
        """
        Create an empty memory file structure.
        
        Returns:
            Empty memory data
        """
        return {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "memory_stats": {
                    "total_memories": 0,
                    "active_memories": 0,
                    "archived_memories": 0
                }
            },
            "memory_index": {
                "index_type": "hnsw",
                "index_parameters": {
                    "m": 16,
                    "ef_construction": 200,
                    "ef": 50
                },
                "entries": {}
            },
            "short_term_memory": [],
            "long_term_memory": [],
            "archived_memory": [],
            "memory_schema": {
                "conversation": {
                    "required_fields": ["role", "message"],
                    "optional_fields": ["summary", "entities", "sentiment", "intent"]
                },
                "fact": {
                    "required_fields": ["fact", "confidence"],
                    "optional_fields": ["domain", "entities", "references"]
                },
                "document": {
                    "required_fields": ["title", "text"],
                    "optional_fields": ["summary", "chunks", "metadata"]
                },
                "code": {
                    "required_fields": ["language", "code"],
                    "optional_fields": ["description", "purpose", "dependencies"]
                }
            },
            "config": {
                "memory_management": {
                    "max_short_term_memories": 100,
                    "max_long_term_memories": 10000,
                    "archival_threshold_days": 30,
                    "deletion_threshold_days": 365,
                    "importance_decay_rate": 0.01,
                    "minimum_importance_threshold": 0.2
                },
                "retrieval": {
                    "default_top_k": 5,
                    "semantic_threshold": 0.75,
                    "recency_weight": 0.3,
                    "importance_weight": 0.7
                },
                "embedding": {
                    "default_model": self.embedding_model_name,
                    "dimensions": self.embedding_dimensions,
                    "batch_size": 8
                }
            }
        }
    
    async def _save_memory_file(self) -> None:
        """Save the memory file."""
        # Update metadata
        self.memory_data["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Create temp file
        temp_file = f"{self.memory_file_path}.tmp"
        
        try:
            with open(temp_file, "w") as f:
                json.dump(self.memory_data, f, indent=2)
            
            # Rename temp file to actual file (atomic operation)
            os.replace(temp_file, self.memory_file_path)
            logger.debug(f"Memory file saved: {self.memory_file_path}")
        except Exception as e:
            logger.error(f"Error saving memory file: {str(e)}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _count_memories(self, data: Dict[str, Any]) -> int:
        """
        Count the total number of memories.
        
        Args:
            data: Memory data
            
        Returns:
            Total number of memories
        """
        count = 0
        for tier in ["short_term_memory", "long_term_memory", "archived_memory"]:
            if tier in data:
                count += len(data[tier])
        return count
    
    def _update_memory_stats(self) -> None:
        """Update memory statistics."""
        # Initialize stats if not present
        if "metadata" not in self.memory_data:
            self.memory_data["metadata"] = {}
        
        if "memory_stats" not in self.memory_data["metadata"]:
            self.memory_data["metadata"]["memory_stats"] = {}
        
        # Count memories in each tier
        short_term_count = len(self.memory_data.get("short_term_memory", []))
        long_term_count = len(self.memory_data.get("long_term_memory", []))
        archived_count = len(self.memory_data.get("archived_memory", []))
        
        # Update stats
        stats = self.memory_data["metadata"]["memory_stats"]
        stats["total_memories"] = short_term_count + long_term_count + archived_count
        stats["active_memories"] = short_term_count + long_term_count
        stats["archived_memories"] = archived_count
        stats["short_term_count"] = short_term_count
        stats["long_term_count"] = long_term_count
    
    async def _update_memory_index(self, memory: Dict[str, Any], tier: str) -> None:
        """
        Update the memory index.
        
        Args:
            memory: Memory to index
            tier: Memory tier
        """
        if "memory_index" not in self.memory_data:
            self.memory_data["memory_index"] = {
                "index_type": "hnsw",
                "index_parameters": {
                    "m": 16,
                    "ef_construction": 200,
                    "ef": 50
                },
                "entries": {}
            }
        
        if "entries" not in self.memory_data["memory_index"]:
            self.memory_data["memory_index"]["entries"] = {}
        
        # Add to index
        memory_id = memory["id"]
        
        self.memory_data["memory_index"]["entries"][memory_id] = {
            "tier": tier,
            "type": memory.get("type", "unknown"),
            "importance": memory.get("importance", 0.5),
            "recency": memory.get("created_at", datetime.now().isoformat())
        }
    
    async def _remove_from_memory_index(self, memory_id: str) -> None:
        """
        Remove a memory from the index.
        
        Args:
            memory_id: Memory ID
        """
        if "memory_index" not in self.memory_data or "entries" not in self.memory_data["memory_index"]:
            return
        
        if memory_id in self.memory_data["memory_index"]["entries"]:
            del self.memory_data["memory_index"]["entries"][memory_id]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (0.0-1.0)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
