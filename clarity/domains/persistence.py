"""
SQLite-based Persistence Domain for simplified, high-performance memory storage and retrieval.

The Persistence Domain is responsible for:
- SQLite vector database operations with sqlite-vec
- Vector embedding generation and indexing
- High-performance similarity search with cosine distance
- Memory metadata storage and filtering with JSON columns  
- Simple connection management (no pools required)
- Cache integration for performance optimization

This replaces the complex Qdrant infrastructure with a 90% simpler SQLite solution
that provides the same functionality with significantly reduced complexity.
"""

import os
import uuid
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from clarity.shared.lazy_imports import ml_deps
from clarity.shared.exceptions.base import MemoryOperationError, ValidationError
from clarity.shared.infrastructure import get_cache, cached

# Import our new SQLite implementation
from .sqlite_persistence import SQLiteMemoryPersistence


class PersistenceDomain:
    """
    High-performance SQLite-based memory persistence.
    
    This domain provides vector database operations with advanced
    similarity search, filtering, and simplified memory management.
    
    Replaces the complex Qdrant infrastructure with a single SQLite backend.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the persistence domain with SQLite backend.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get SQLite configuration
        sqlite_config = config.get("sqlite", {})
        if not sqlite_config:
            # Fallback to alunai-clarity section
            sqlite_config = config.get("alunai-clarity", {}).get("sqlite", {})
        
        # SQLite database path
        self.db_path = sqlite_config.get("path", "./.claude/alunai-clarity/memories.db")
        
        # Embedding configuration
        embedding_config = config.get("embedding", {})
        self.embedding_model_name = embedding_config.get("default_model", "paraphrase-MiniLM-L3-v2")
        self.embedding_dimensions = embedding_config.get("dimensions", 384)
        
        # Feature flag for migration compatibility
        self.use_sqlite = config.get("persistence", {}).get("use_sqlite", True)  # Default to SQLite
        
        # Initialize embedding model (lazy-loaded)
        self.embedding_model = None
        
        # Initialize SQLite backend
        self.backend = SQLiteMemoryPersistence(
            db_path=self.db_path,
            embedding_model=self.embedding_model
        )
        
        logger.info(f"Persistence domain initialized with SQLite backend: {self.db_path}")
    
    async def initialize(self) -> None:
        """Initialize the persistence domain."""
        try:
            logger.info("Initializing SQLite persistence domain...")
            
            # SQLite backend initializes automatically in constructor
            # No additional initialization required
            
            logger.info("SQLite persistence domain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize persistence domain: {e}")
            raise MemoryOperationError(f"Persistence initialization failed: {e}")
    
    async def close(self) -> None:
        """Clean up resources."""
        # SQLite connections are closed automatically per-operation
        # No background tasks or persistent connections to clean up
        logger.info("Persistence domain closed")
    
    async def store_memory(self, memory: Dict[str, Any], tier: str = "short_term") -> str:
        """
        Store a memory with vector embedding.
        
        Args:
            memory: Memory data to store
            tier: Memory tier (short_term, long_term, archival)
            
        Returns:
            Memory ID
        """
        return await self.backend.store_memory(memory, tier)
    
    async def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_similarity: float = 0.6,
        include_metadata: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using vector similarity search with optional filtering.
        
        Args:
            query: Search query
            limit: Maximum number of results
            memory_types: Filter by memory types
            min_similarity: Minimum similarity score
            include_metadata: Whether to include metadata
            filters: Additional filters
            
        Returns:
            List of matching memories
        """
        return await self.backend.retrieve_memories(
            query=query,
            limit=limit,
            memory_types=memory_types,
            min_similarity=min_similarity,
            include_metadata=include_metadata,
            filters=filters
        )
    
    async def search_memories(
        self,
        embedding: Optional[List[float]] = None,
        limit: int = 10,
        types: Optional[List[str]] = None,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories by embedding or with filters.
        
        Args:
            embedding: Vector embedding for similarity search (optional)
            limit: Maximum number of results
            types: Filter by memory types
            min_similarity: Minimum similarity threshold
            filters: Additional filters
            
        Returns:
            List of matching memories
        """
        return await self.backend.search_memories(
            embedding=embedding,
            limit=limit,
            types=types,
            min_similarity=min_similarity,
            filters=filters
        )
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            updates: Updates to apply
            
        Returns:
            Success status
        """
        return await self.backend.update_memory(memory_id, updates)
    
    async def delete_memories(self, memory_ids: List[str]) -> List[str]:
        """
        Delete memories.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            List of successfully deleted memory IDs
        """
        return await self.backend.delete_memories(memory_ids)
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return await self.backend.get_memory_stats()
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID with caching.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory dictionary if found, None otherwise
        """
        return await self.backend.get_memory(memory_id)
    
    async def get_memory_tier(self, memory_id: str) -> Optional[str]:
        """
        Get the tier of a specific memory.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory tier (short_term, long_term, archival) if found, None otherwise
        """
        memory = await self.get_memory(memory_id)
        return memory.get("tier") if memory else None
    
    async def list_memories(
        self,
        types: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
        tier: Optional[str] = None,
        include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List memories with optional filtering.
        
        Args:
            types: Filter by memory types
            limit: Maximum number of memories to return
            offset: Number of memories to skip
            tier: Filter by memory tier (short_term, long_term, archival)
            include_content: Whether to include full content in results
            
        Returns:
            List of memory dictionaries
        """
        # Build filters for search_memories
        filters = {}
        if tier:
            filters["tier"] = tier
        
        # Use search_memories as the backend
        results = await self.backend.search_memories(
            limit=limit,
            types=types,
            filters=filters
        )
        
        # Apply offset (simple implementation)
        if offset > 0:
            results = results[offset:]
        
        # Filter content if not requested
        if not include_content:
            for result in results:
                if "content" in result:
                    del result["content"]
        
        return results
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using the configured embedding model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the text embedding
        """
        return self.backend._generate_embedding(text)
    
    # Metadata management methods
    async def get_metadata(self, key: str) -> Optional[Any]:
        """
        Get metadata value by key.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        try:
            # Search for metadata memories
            results = await self.backend.search_memories(
                types=["metadata"],
                filters={"metadata_key": key},
                limit=1
            )
            
            if results:
                return results[0].get("metadata_value")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metadata {key}: {e}")
            return None
    
    async def set_metadata(self, key: str, value: Any) -> bool:
        """
        Set metadata value by key.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Success flag
        """
        try:
            # Create metadata memory
            metadata_memory = {
                "id": f"metadata_{key}",
                "type": "metadata",
                "content": f"Metadata: {key} = {value}",
                "importance": 0.1,  # Low importance for metadata
                "metadata": {
                    "metadata_key": key,
                    "metadata_value": value,
                    "is_system_metadata": True
                },
                "tier": "system"
            }
            
            await self.backend.store_memory(metadata_memory)
            logger.debug(f"Set metadata: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set metadata {key}: {e}")
            return False
    
    # Structured thinking integration methods (for compatibility)
    async def store_structured_thought(
        self,
        thought: Any,  # StructuredThought object
        session_id: Optional[str] = None
    ) -> str:
        """
        Store a structured thought with enhanced metadata.
        
        Args:
            thought: StructuredThought instance
            session_id: Optional session ID to group related thoughts
            
        Returns:
            Memory ID for the stored thought
        """
        try:
            # Convert structured thought to memory format
            memory_data = {
                "id": f"mem_{thought.id}",
                "type": "structured_thinking",
                "content": thought.content,
                "importance": getattr(thought, 'importance', 0.7),
                "metadata": {
                    "thought_id": thought.id,
                    "thinking_stage": getattr(thought, 'stage', 'unknown'),
                    "thinking_session_id": session_id,
                    "structured_thinking": True,
                    "thought_number": getattr(thought, 'thought_number', 1),
                    "tags": getattr(thought, 'tags', []),
                    "axioms": getattr(thought, 'axioms', []),
                    "assumptions_challenged": getattr(thought, 'assumptions_challenged', [])
                },
                "tier": "long_term"  # Structured thoughts are valuable
            }
            
            memory_id = await self.store_memory(memory_data)
            logger.info(f"Stored structured thought: {thought.id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store structured thought: {e}")
            raise MemoryOperationError(f"Could not store structured thought: {e}")
    
    async def find_similar_thinking_sessions(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find thinking sessions similar to a query using vector similarity.
        
        Args:
            query: Search query for finding similar sessions
            limit: Maximum number of sessions to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar thinking session memories
        """
        try:
            # Search for structured thinking memories
            sessions = await self.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=["structured_thinking"],
                min_similarity=min_similarity,
                include_metadata=True,
                filters={"structured_thinking": True}
            )
            
            logger.info(f"Found {len(sessions)} similar thinking sessions for query: {query}")
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to find similar thinking sessions: {e}")
            return []
    
    # Performance and monitoring methods
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            "embedding_cache": self.backend.embedding_cache.get_info(),
            "memory_cache": self.backend.memory_cache.get_info()
        }
    
    async def backup_memories(self, backup_path: str) -> bool:
        """Create a backup of the SQLite database."""
        try:
            backup_file = Path(backup_path) / f"sqlite_memories_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Simple file copy for SQLite backup
            import shutil
            shutil.copy2(self.db_path, str(backup_file))
            
            logger.info(f"SQLite memory backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False


# Maintain compatibility with existing code that imports specific class names
QdrantPersistenceDomain = PersistenceDomain  # Alias for backward compatibility
PersistenceDomain = PersistenceDomain