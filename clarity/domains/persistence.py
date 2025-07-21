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
import sys
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import redirect_stderr
from io import StringIO

from loguru import logger
from qdrant_client.models import (
    Filter, VectorParams, Distance, PointStruct, FieldCondition, 
    MatchValue, Range, MatchAny
)
from clarity.shared.lazy_imports import ml_deps, db_deps
from clarity.shared.exceptions.base import QdrantConnectionError, MemoryOperationError, ValidationError
from clarity.shared.infrastructure import (
    ConnectionConfig, 
    get_connection_pool, 
    qdrant_connection,
    get_cache,
    cached
)
from clarity.shared.infrastructure.shared_qdrant import get_shared_qdrant_client
# Qdrant models will be imported lazily when needed


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
        # Get qdrant config from alunai-clarity section if available, otherwise use global
        self.qdrant_config = self.config.get("alunai-clarity", {}).get("qdrant", {})
        if not self.qdrant_config:
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
        self.connection_pool = None
        
        # Initialize caches
        self.embedding_cache = get_cache(
            "embeddings",
            max_size=5000,      # Cache up to 5000 embeddings
            max_memory_mb=200,  # 200MB for embeddings
            default_ttl=7200.0  # 2 hours TTL
        )
        self.memory_cache = get_cache(
            "memories", 
            max_size=1000,      # Cache up to 1000 memory objects
            max_memory_mb=50,   # 50MB for memory data
            default_ttl=1800.0  # 30 minutes TTL
        )
        
    async def initialize(self) -> None:
        """Initialize the Qdrant persistence domain."""
        try:
            logger.info("Initializing Qdrant persistence domain...")
            
            # Initialize connection pool
            pool_config = ConnectionConfig(
                url=self.qdrant_config.get("url"),
                path=self.qdrant_path if not self.qdrant_config.get("url") else None,
                api_key=self.qdrant_config.get("api_key"),
                timeout=self.qdrant_config.get("timeout", 30.0),
                prefer_grpc=self.qdrant_config.get("prefer_grpc", True)
            )
            
            self.connection_pool = await get_connection_pool(pool_config)
            await self.connection_pool.initialize()
            
            # Use shared client for concurrent access support
            if self.qdrant_config.get("url"):
                # Remote Qdrant - use direct client
                QdrantClientClass = db_deps.QdrantClient
                if QdrantClientClass is None:
                    raise ImportError("qdrant-client not available")
                self.client = QdrantClientClass(
                    url=self.qdrant_config["url"],
                    api_key=self.qdrant_config.get("api_key"),
                    timeout=pool_config.timeout
                )
            else:
                # Local Qdrant - use shared client for concurrent access
                self.client = await get_shared_qdrant_client(
                    self.qdrant_path, 
                    pool_config.timeout
                )
            
            # Initialize embedding model lazily (will be loaded on first use)
            self.embedding_model = None
            logger.debug(f"Embedding model {self.embedding_model_name} will be loaded lazily")
            
            # Ensure collection exists
            await self._ensure_collection()
            
            logger.info("Qdrant persistence domain initialized successfully")
            
        except (QdrantConnectionError, ImportError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to initialize Qdrant persistence domain: {e}")
            raise
    
    async def _ensure_collection(self) -> None:
        """Ensure the memories collection exists with proper configuration."""
        try:
            async with qdrant_connection() as client:
                collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.COLLECTION_NAME not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME}")
                
                async with qdrant_connection() as client:
                    client.create_collection(
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
                
        except (QdrantConnectionError, ConnectionError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _lazy_load_embedding_model(self) -> None:
        """Lazy load the embedding model when first needed."""
        if self.embedding_model is not None:
            return
            
        SentenceTransformerClass = ml_deps.SentenceTransformer
        if SentenceTransformerClass is None:
            raise ImportError("sentence-transformers not available")
        
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        with redirect_stderr(StringIO()):
            self.embedding_model = SentenceTransformerClass(self.embedding_model_name)
        
        logger.info("Embedding model loaded successfully")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with caching."""
        if not self.embedding_model:
            self._lazy_load_embedding_model()
        
        # Create cache key from text hash (for privacy)
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"emb_{text_hash}_{len(text)}"
        
        # Try cache first
        cached_embedding = self.embedding_cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding
        with redirect_stderr(StringIO()):
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        embedding_list = embedding.tolist()
        
        # Cache the result
        self.embedding_cache.set(cache_key, embedding_list)
        
        return embedding_list
    
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
        original_id = memory.get("id", str(uuid.uuid4()))
        # Clean the ID to be a valid UUID (remove mem_ prefix if present)
        clean_id = original_id.replace("mem_", "") if original_id and original_id.startswith("mem_") else original_id
        
        payload = {
            "memory_id": clean_id,
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
    
    async def store_memory(self, memory: Dict[str, Any], tier: str = "short_term") -> str:
        """
        Store a memory in Qdrant.
        
        Args:
            memory: Memory data to store
            tier: Memory tier (short_term, long_term, archival) - stored in metadata
            
        Returns:
            Memory ID
        """
        try:
            # Prepare payload and extract text
            payload, text_content = self._prepare_memory_payload(memory)
            memory_id = payload["memory_id"]
            
            # Add tier to payload metadata
            if "metadata" not in payload:
                payload["metadata"] = {}
            payload["metadata"]["tier"] = tier
            
            # Generate embedding
            embedding = self._generate_embedding(text_content)
            
            # Create point for Qdrant
            point = PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload
            )
            
            # Store in Qdrant using connection pool
            async with qdrant_connection() as client:
                client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=[point]
                )
            
            logger.debug(f"Stored memory in Qdrant: {memory_id}")
            return memory_id
            
        except (QdrantConnectionError, MemoryOperationError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to store memory: {e}")
            raise MemoryOperationError("Failed to store memory", cause=e)
    
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
            
            # Perform vector search using connection pool
            async with qdrant_connection() as client:
                search_results = client.search(
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
            
        except (QdrantConnectionError, MemoryOperationError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise MemoryOperationError("Failed to retrieve memories", cause=e)
    
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
                async with qdrant_connection() as client:
                    points = client.retrieve(
                    collection_name=self.COLLECTION_NAME,
                    ids=[memory_id],
                    with_payload=True
                )
                
                if points:
                    payload = points[0].payload
                    payload["access_count"] = payload.get("access_count", 0) + 1
                    payload["last_accessed"] = current_time
                    
                    # Update point
                    async with qdrant_connection() as client:
                        client.set_payload(
                        collection_name=self.COLLECTION_NAME,
                        payload=payload,
                        points=[memory_id]
                    )
                    
        except (QdrantConnectionError, MemoryOperationError, ValueError) as e:
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
            
            async with qdrant_connection() as client:
                client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[updated_point]
            )
            
            # Invalidate cache for updated memory
            cache_key = f"mem_{memory_id}"
            self.memory_cache.delete(cache_key)
            
            logger.debug(f"Updated memory: {memory_id}")
            return True
            
        except (QdrantConnectionError, MemoryOperationError, KeyError, ValueError) as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
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
        try:
            # Build filters
            search_filters = []
            
            if types:
                search_filters.append(FieldCondition(
                    key="memory_type",
                    match=MatchAny(any=types)
                ))
            
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        search_filters.append(FieldCondition(
                            key=key,
                            match=MatchAny(any=value)
                        ))
                    else:
                        search_filters.append(FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        ))
            
            # Combine filters
            search_filter = None
            if search_filters:
                if len(search_filters) == 1:
                    search_filter = Filter(must=[search_filters[0]])
                else:
                    search_filter = Filter(must=search_filters)
            
            # If embedding is provided, do vector search
            if embedding:
                async with qdrant_connection() as client:
                    points = client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=embedding,
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=min_similarity
                )
            else:
                # No embedding provided, use scroll to get filtered results
                async with qdrant_connection() as client:
                    points, _ = client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    scroll_filter=search_filter,
                    limit=limit
                )
            
            # Convert results
            results = []
            for point in points:
                memory = dict(point.payload)
                # Fix field name mapping
                memory["id"] = memory.get("memory_id")
                memory["type"] = memory.get("memory_type")
                if hasattr(point, 'score'):
                    memory["similarity"] = point.score
                results.append(memory)
            
            return results
            
        except (QdrantConnectionError, MemoryOperationError, ValueError, RuntimeError) as e:
            logger.error(f"Error searching memories: {e}")
            return []

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
            async with qdrant_connection() as client:
                operation_info = client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=memory_ids
            )
            
            # Invalidate cache for deleted memories
            for memory_id in memory_ids:
                cache_key = f"mem_{memory_id}"
                self.memory_cache.delete(cache_key)
            
            logger.debug(f"Deleted {len(memory_ids)} memories from Qdrant")
            return memory_ids
            
        except (QdrantConnectionError, MemoryOperationError, KeyError, ValueError) as e:
            logger.error(f"Failed to delete memories: {e}")
            return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            # Get collection info
            async with qdrant_connection() as client:
                collection_info = client.get_collection(self.COLLECTION_NAME)
            
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
            
        except (RuntimeError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def optimize_collection(self) -> bool:
        """Optimize the Qdrant collection for better performance."""
        try:
            async with qdrant_connection() as client:
                client.update_collection(
                    collection_name=self.COLLECTION_NAME,
                    optimizer_config={
                        "deleted_threshold": 0.2,
                        "vacuum_min_vector_number": 1000,
                        "default_segment_number": 0,
                    }
                )
            
            logger.info("Qdrant collection optimization triggered")
            return True
            
        except (RuntimeError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool performance statistics"""
        if self.connection_pool:
            return self.connection_pool.get_stats()
        return {"error": "Connection pool not initialized"}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "embedding_cache": self.embedding_cache.get_info(),
            "memory_cache": self.memory_cache.get_info()
        }
    
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
            
        except (OSError, PermissionError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    async def get_metadata(self, key: str) -> Optional[Any]:
        """
        Get metadata value by key.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        try:
            # Search for metadata points with specific key
            search_result = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="memory_type",
                            match=MatchValue(value="metadata")
                        ),
                        FieldCondition(
                            key="metadata_key",
                            match=MatchValue(value=key)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            if search_result[0]:
                return search_result[0][0].payload.get("metadata_value")
            
            return None
            
        except (RuntimeError, AttributeError, KeyError, ValueError, IndexError) as e:
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
            # Create metadata point
            metadata_id = f"metadata_{key}"
            
            point = PointStruct(
                id=metadata_id,
                vector=[0.0] * self.embedding_dimensions,  # Dummy vector for metadata
                payload={
                    "memory_type": "metadata",
                    "metadata_key": key,
                    "metadata_value": value,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "tier": "system"
                }
            )
            
            async with qdrant_connection() as client:
                client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )
            
            logger.debug(f"Set metadata: {key} = {value}")
            return True
            
        except (RuntimeError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Failed to set metadata {key}: {e}")
            return False

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using the configured embedding model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the text embedding
            
        Raises:
            RuntimeError: If embedding model is not initialized
        """
        try:
            return self._generate_embedding(text)
        except (RuntimeError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
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
        try:
            # Build filter conditions
            must_conditions = []
            
            if types:
                must_conditions.append(
                    FieldCondition(key="memory_type", match=MatchAny(any=types))
                )
            
            if tier:
                must_conditions.append(
                    FieldCondition(key="tier", match=MatchValue(value=tier))
                )
            
            # Create filter if we have conditions
            search_filter = None
            if must_conditions:
                search_filter = Filter(must=must_conditions)
            
            # Search without vector (get all matching records)
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            memories = []
            for point in results[0]:  # results is (points, next_page_offset)
                payload = point.payload
                memory = {
                    "id": payload.get("memory_id"),
                    "type": payload.get("memory_type"),
                    "importance": payload.get("importance"),
                    "tier": payload.get("tier"),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "access_count": payload.get("access_count", 0),
                    "last_accessed": payload.get("last_accessed"),
                    "metadata": payload.get("metadata", {})
                }
                
                if include_content:
                    memory["content"] = payload.get("content")
                    
                memories.append(memory)
            
            logger.debug(f"Listed {len(memories)} memories with filters: types={types}, tier={tier}")
            return memories
            
        except (RuntimeError, AttributeError, KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to list memories: {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID with caching.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory dictionary if found, None otherwise
        """
        # Try cache first
        cache_key = f"mem_{memory_id}"
        cached_memory = self.memory_cache.get(cache_key)
        if cached_memory is not None:
            return cached_memory
        
        try:
            # Search for memory by ID
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if not results[0]:  # No memories found
                return None
            
            point = results[0][0]
            payload = point.payload
            
            # Update access tracking
            await self._update_access_tracking([memory_id])
            
            memory = {
                "id": payload.get("memory_id"),
                "type": payload.get("memory_type"),
                "content": payload.get("content"),
                "importance": payload.get("importance"),
                "tier": payload.get("tier"),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "access_count": payload.get("access_count", 0),
                "last_accessed": payload.get("last_accessed"),
                "metadata": payload.get("metadata", {})
            }
            
            # Cache the memory for future access
            self.memory_cache.set(cache_key, memory)
            
            logger.debug(f"Retrieved memory: {memory_id}")
            return memory
            
        except (RuntimeError, AttributeError, KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    async def get_memory_tier(self, memory_id: str) -> Optional[str]:
        """
        Get the tier of a specific memory.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory tier (short_term, long_term, archival) if found, None otherwise
        """
        try:
            # Search for memory by ID, only retrieve tier field
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
                ),
                limit=1,
                with_payload=["tier"],  # Only retrieve tier field
                with_vectors=False
            )
            
            if not results[0]:  # No memories found
                return None
            
            point = results[0][0]
            tier = point.payload.get("tier")
            
            logger.debug(f"Retrieved tier for memory {memory_id}: {tier}")
            return tier
            
        except (RuntimeError, AttributeError, KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to get memory tier for {memory_id}: {e}")
            return None


# Maintain compatibility with existing code
PersistenceDomain = QdrantPersistenceDomain