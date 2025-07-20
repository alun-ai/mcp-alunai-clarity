"""
Qdrant-based Persistence Domain for high-performance memory storage and retrieval.

The Persistence Domain is responsible for:
- Qdrant vector database operations
- Vector embedding generation and indexing
- High-performance similarity search
- Memory metadata storage and filtering
- Collection management and optimization
"""

import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    MatchValue,
    SearchRequest,
)


class QdrantPersistenceDomain:
    """
    High-performance Qdrant-based memory persistence.
    
    This domain provides vector database operations with advanced
    similarity search, filtering, and scalable memory management.
    """
    
    COLLECTION_NAME = "memories"
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Qdrant persistence domain.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.qdrant_config = self.config.get("qdrant", {})
        
        # Qdrant configuration
        self.qdrant_host = self.qdrant_config.get("host", "localhost")
        self.qdrant_port = self.qdrant_config.get("port", 6333)
        self.qdrant_path = self.qdrant_config.get("path", "./qdrant_data")
        self.prefer_grpc = self.qdrant_config.get("prefer_grpc", False)
        
        # Embedding configuration
        self.embedding_model_name = self.config["embedding"].get(
            "default_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_dimensions = self.config["embedding"].get("dimensions", 384)
        
        # Performance settings
        self.index_params = self.qdrant_config.get("index_params", {
            "m": 16,
            "ef_construct": 200,
            "full_scan_threshold": 10000,
        })
        
        # Will be initialized during initialize()
        self.client = None
        self.embedding_model = None
        
    async def initialize(self) -> None:
        """Initialize the Qdrant persistence domain."""
        try:
            logger.info("Initializing Qdrant persistence domain...")
            
            # Initialize Qdrant client (embedded mode)
            self.client = QdrantClient(path=self.qdrant_path)
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Ensure collection exists
            await self._ensure_collection()
            
            logger.info("Qdrant persistence domain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant persistence domain: {e}")
            raise
    
    async def _ensure_collection(self) -> None:
        """Ensure the memories collection exists with proper configuration."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.COLLECTION_NAME not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME}")
                
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.embedding_dimensions,
                        distance=Distance.COSINE,
                        hnsw_config={
                            "m": self.index_params["m"],
                            "ef_construct": self.index_params["ef_construct"],
                            "full_scan_threshold": self.index_params["full_scan_threshold"],
                        }
                    ),
                )
                logger.info("Qdrant collection created successfully")
            else:
                logger.info(f"Qdrant collection '{self.COLLECTION_NAME}' already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _prepare_memory_payload(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare memory data for Qdrant storage."""
        # Extract text content for embedding
        content = memory.get("content", "")
        if isinstance(content, dict):
            # Combine all string values from content dict
            text_parts = []
            for value in content.values():
                if isinstance(value, str):
                    text_parts.append(value)
            text_content = " ".join(text_parts)
        else:
            text_content = str(content)
        
        # Prepare payload (metadata)
        payload = {
            "memory_id": memory.get("id", str(uuid.uuid4())),
            "memory_type": memory.get("type", "unknown"),
            "content": memory.get("content", {}),
            "importance": memory.get("importance", 0.5),
            "tier": memory.get("tier", "short_term"),
            "created_at": memory.get("created_at", datetime.utcnow().isoformat()),
            "updated_at": memory.get("updated_at", datetime.utcnow().isoformat()),
            "metadata": memory.get("metadata", {}),
            "context": memory.get("context", {}),
            "access_count": memory.get("access_count", 0),
            "last_accessed": memory.get("last_accessed"),
            "text_content": text_content,  # For full-text search
        }
        
        return payload, text_content
    
    async def store_memory(self, memory: Dict[str, Any]) -> str:
        """
        Store a memory in Qdrant.
        
        Args:
            memory: Memory data to store
            
        Returns:
            Memory ID
        """
        try:
            # Prepare payload and extract text
            payload, text_content = self._prepare_memory_payload(memory)
            memory_id = payload["memory_id"]
            
            # Generate embedding
            embedding = self._generate_embedding(text_content)
            
            # Create point for Qdrant
            point = PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )
            
            logger.debug(f"Stored memory in Qdrant: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
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
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build filters
            search_filter = self._build_search_filter(
                memory_types=memory_types,
                min_similarity=min_similarity,
                additional_filters=filters
            )
            
            # Perform vector search
            search_results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=min_similarity,
                with_payload=True,
                with_vectors=False
            )
            
            # Process results
            memories = []
            for result in search_results:
                memory_data = {
                    "id": result.payload["memory_id"],
                    "type": result.payload["memory_type"],
                    "content": result.payload["content"],
                    "importance": result.payload["importance"],
                    "similarity_score": float(result.score),
                    "created_at": result.payload["created_at"],
                    "updated_at": result.payload["updated_at"],
                }
                
                if include_metadata:
                    memory_data.update({
                        "metadata": result.payload.get("metadata", {}),
                        "context": result.payload.get("context", {}),
                        "tier": result.payload.get("tier", "short_term"),
                        "access_count": result.payload.get("access_count", 0),
                        "last_accessed": result.payload.get("last_accessed"),
                    })
                
                memories.append(memory_data)
            
            # Update access tracking
            if memories:
                await self._update_access_tracking([m["id"] for m in memories])
            
            logger.debug(f"Retrieved {len(memories)} memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise
    
    def _build_search_filter(
        self,
        memory_types: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Filter]:
        """Build Qdrant search filter from parameters."""
        conditions = []
        
        # Memory type filter
        if memory_types:
            conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchValue(value=memory_types[0]) if len(memory_types) == 1
                    else MatchValue(any=memory_types)
                )
            )
        
        # Additional filters
        if additional_filters:
            for key, value in additional_filters.items():
                if isinstance(value, (int, float)):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(gte=value)
                        )
                    )
                elif isinstance(value, str):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                elif isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(any=value)
                        )
                    )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    async def _update_access_tracking(self, memory_ids: List[str]) -> None:
        """Update access tracking for retrieved memories."""
        try:
            current_time = datetime.utcnow().isoformat()
            
            for memory_id in memory_ids:
                # Get current memory
                points = self.client.retrieve(
                    collection_name=self.COLLECTION_NAME,
                    ids=[memory_id],
                    with_payload=True
                )
                
                if points:
                    payload = points[0].payload
                    payload["access_count"] = payload.get("access_count", 0) + 1
                    payload["last_accessed"] = current_time
                    
                    # Update point
                    self.client.set_payload(
                        collection_name=self.COLLECTION_NAME,
                        payload=payload,
                        points=[memory_id]
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to update access tracking: {e}")
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            updates: Updates to apply
            
        Returns:
            Success status
        """
        try:
            # Get current memory
            points = self.client.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[memory_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                logger.warning(f"Memory not found for update: {memory_id}")
                return False
            
            current_point = points[0]
            current_payload = current_point.payload
            current_vector = current_point.vector
            
            # Apply updates
            for key, value in updates.items():
                if key == "content":
                    current_payload["content"] = value
                    # Regenerate embedding if content changed
                    if isinstance(value, dict):
                        text_parts = [str(v) for v in value.values() if isinstance(v, str)]
                        text_content = " ".join(text_parts)
                    else:
                        text_content = str(value)
                    current_payload["text_content"] = text_content
                    current_vector = self._generate_embedding(text_content)
                else:
                    current_payload[key] = value
            
            current_payload["updated_at"] = datetime.utcnow().isoformat()
            
            # Update point
            updated_point = PointStruct(
                id=memory_id,
                vector=current_vector,
                payload=current_payload
            )
            
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[updated_point]
            )
            
            logger.debug(f"Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memories(self, memory_ids: List[str]) -> List[str]:
        """
        Delete memories from Qdrant.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            List of successfully deleted memory IDs
        """
        try:
            # Delete points
            operation_info = self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=memory_ids
            )
            
            logger.debug(f"Deleted {len(memory_ids)} memories from Qdrant")
            return memory_ids
            
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}")
            return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.COLLECTION_NAME)
            
            # Count memories by type
            type_counts = {}
            scroll_result = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                with_payload=["memory_type"],
                limit=10000  # Adjust based on expected memory count
            )
            
            for point in scroll_result[0]:
                memory_type = point.payload.get("memory_type", "unknown")
                type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            
            # Count by tier
            tier_counts = {}
            scroll_result = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                with_payload=["tier"],
                limit=10000
            )
            
            for point in scroll_result[0]:
                tier = point.payload.get("tier", "unknown")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            stats = {
                "total_memories": collection_info.points_count,
                "indexed_memories": collection_info.indexed_vectors_count,
                "memory_types": type_counts,
                "memory_tiers": tier_counts,
                "collection_status": collection_info.status.value,
                "optimizer_status": collection_info.optimizer_status,
                "disk_data_size": getattr(collection_info, 'disk_data_size', 0),
                "ram_data_size": getattr(collection_info, 'ram_data_size', 0),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def optimize_collection(self) -> bool:
        """Optimize the Qdrant collection for better performance."""
        try:
            self.client.update_collection(
                collection_name=self.COLLECTION_NAME,
                optimizer_config={
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 0,
                }
            )
            
            logger.info("Qdrant collection optimization triggered")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False
    
    async def backup_memories(self, backup_path: str) -> bool:
        """Create a backup snapshot of memories."""
        try:
            # Create snapshot
            snapshot_info = self.client.create_snapshot(
                collection_name=self.COLLECTION_NAME
            )
            
            # Move snapshot to backup location
            import shutil
            snapshot_path = Path(self.qdrant_path) / "snapshots" / snapshot_info.name
            backup_file = Path(backup_path) / f"memories_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.snapshot"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(snapshot_path), str(backup_file))
            
            logger.info(f"Memory backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False


# Maintain compatibility with existing code
PersistenceDomain = QdrantPersistenceDomain