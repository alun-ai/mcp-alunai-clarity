#!/usr/bin/env python3
"""
Simple SQLite + sqlite-vec Implementation for Memory Persistence

A synchronous, straightforward implementation that maintains API compatibility
with the existing Qdrant persistence domain while using SQLite for storage.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SQLiteVecPersistenceDomain:
    """
    Simple SQLite-based memory persistence domain.
    
    API-compatible replacement for QdrantPersistenceDomain with
    simplified implementation using SQLite + sqlite-vec.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SQLite persistence domain."""
        self.config = config
        self.sqlite_config = config.get("sqlite", {})
        self.embedding_config = config.get("embedding", {})
        
        # Database configuration
        self.db_path = self.sqlite_config.get("path", "./memory.db")
        self.vector_dimensions = self.sqlite_config.get("vector_dimensions", 384)
        
        # Embedding configuration
        self.embedding_model_name = self.embedding_config.get("default_model", "paraphrase-MiniLM-L3-v2")
        self.embedding_model = None
        
        # Thread-safe connection handling
        self._connection_lock = threading.RLock()
        self._connection = None
        self.sqlite_vec_available = False
        
        logger.info(f"SQLiteVec persistence domain configured: {self.db_path}")
    
    def initialize(self):
        """Initialize the SQLite database and connection."""
        # Create database directory if needed
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database schema
        self._initialize_database()
        
        # Load embedding model
        self._load_embedding_model()
        
        logger.info("SQLiteVec persistence domain initialized successfully")
    
    def _initialize_database(self):
        """Create database schema with vector support."""
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            
            # Try to load sqlite-vec extension
            try:
                conn.enable_load_extension(True)
                try:
                    import sqlite_vec
                    sqlite_vec.load(conn)
                    self.sqlite_vec_available = True
                    logger.info("sqlite-vec extension loaded successfully")
                except (ImportError, sqlite3.OperationalError) as e:
                    logger.warning(f"sqlite-vec extension not available: {e}, using fallback")
                    self.sqlite_vec_available = False
                conn.enable_load_extension(False)
            except AttributeError:
                logger.warning("SQLite extension loading not supported, using fallback")
                self.sqlite_vec_available = False
            
            # Create main memories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    tier TEXT NOT NULL DEFAULT 'short_term',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT,
                    text_content TEXT,
                    embedding BLOB
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
            
            # Create composite indexes for common filter combinations
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type_tier ON memories(memory_type, tier)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type_importance ON memories(memory_type, importance)")
            
            conn.commit()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get a thread-safe database connection."""
        with self._connection_lock:
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.db_path, 
                    timeout=30.0,
                    check_same_thread=False
                )
                self._connection.row_factory = sqlite3.Row
            
            yield self._connection
    
    def _serialize_vector(self, vector: List[float]) -> bytes:
        """Serialize vector to bytes for storage."""
        if self.sqlite_vec_available:
            try:
                from sqlite_vec import serialize_float32
                return serialize_float32(vector)
            except ImportError:
                pass
        
        # Fallback: use numpy
        return np.array(vector, dtype=np.float32).tobytes()
    
    def _deserialize_vector(self, vector_bytes: bytes) -> List[float]:
        """Deserialize vector from bytes."""
        return np.frombuffer(vector_bytes, dtype=np.float32).tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def store_memory(self, memory: Dict[str, Any], tier: str = "short_term") -> str:
        """Store a memory with vector embedding."""
        memory_id = memory.get("memory_id", str(uuid.uuid4()))
        
        # Generate embedding
        content = memory.get("content", "")
        if content and self.embedding_model:
            embedding = self.embedding_model.encode(content).tolist()
        else:
            embedding = [0.0] * self.vector_dimensions
        
        # Prepare data
        now = datetime.utcnow().isoformat()
        serialized_embedding = self._serialize_vector(embedding)
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (memory_id, memory_type, content, importance, tier, created_at, updated_at, 
                 access_count, last_accessed, metadata, text_content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                memory.get("memory_type", "episodic"),
                content,
                memory.get("importance", 0.5),
                tier,
                memory.get("created_at", now),
                now,
                memory.get("access_count", 0),
                memory.get("last_accessed"),
                json.dumps(memory.get("metadata", {})),
                content,  # For full-text search
                serialized_embedding
            ))
            conn.commit()
        
        logger.debug(f"Stored memory: {memory_id}")
        return memory_id
    
    async def retrieve_memories(self, query: str, limit: int = 5, **filters) -> List[Dict[str, Any]]:
        """Retrieve memories using vector similarity search."""
        if not query and not filters:
            return []
        
        results = []
        
        with self._get_connection() as conn:
            if query and self.embedding_model:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query).tolist()
                
                # Build SQL query
                sql_parts = ["SELECT * FROM memories"]
                params = []
                conditions = []
                
                # Add filters
                if filters.get("memory_types"):
                    placeholders = ",".join("?" * len(filters["memory_types"]))
                    conditions.append(f"memory_type IN ({placeholders})")
                    params.extend(filters["memory_types"])
                
                if filters.get("tier"):
                    conditions.append("tier = ?")
                    params.append(filters["tier"])
                
                if filters.get("min_importance"):
                    conditions.append("importance >= ?")
                    params.append(filters["min_importance"])
                
                if conditions:
                    sql_parts.append("WHERE " + " AND ".join(conditions))
                
                sql = " ".join(sql_parts)
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                # Calculate similarities and sort
                candidates = []
                for row in rows:
                    if row["embedding"]:
                        stored_embedding = self._deserialize_vector(row["embedding"])
                        similarity = self._cosine_similarity(query_embedding, stored_embedding)
                        
                        # Apply similarity threshold
                        min_similarity = filters.get("min_similarity", 0.0)
                        if similarity >= min_similarity:
                            candidates.append((similarity, dict(row)))
                
                # Sort by similarity and limit results
                candidates.sort(key=lambda x: x[0], reverse=True)
                results = [item[1] for item in candidates[:limit]]
            
            else:
                # No query embedding, just filter
                sql_parts = ["SELECT * FROM memories"]
                params = []
                conditions = []
                
                if filters.get("memory_types"):
                    placeholders = ",".join("?" * len(filters["memory_types"]))
                    conditions.append(f"memory_type IN ({placeholders})")
                    params.extend(filters["memory_types"])
                
                if filters.get("tier"):
                    conditions.append("tier = ?")
                    params.append(filters["tier"])
                
                if filters.get("min_importance"):
                    conditions.append("importance >= ?")
                    params.append(filters["min_importance"])
                
                if conditions:
                    sql_parts.append("WHERE " + " AND ".join(conditions))
                
                sql_parts.append("ORDER BY created_at DESC")
                sql_parts.append("LIMIT ?")
                params.append(limit)
                
                sql = " ".join(sql_parts)
                cursor = conn.execute(sql, params)
                results = [dict(row) for row in cursor.fetchall()]
        
        # Parse metadata back to dict
        for result in results:
            if result.get("metadata"):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except (json.JSONDecodeError, TypeError):
                    result["metadata"] = {}
        
        logger.debug(f"Retrieved {len(results)} memories for query: {query[:50]}")
        return results
    
    async def search_memories(self, embedding: Optional[List[float]] = None, **filters) -> List[Dict[str, Any]]:
        """Search memories with optional embedding and filters."""
        if embedding:
            # Create a dummy query for the embedding-based search
            return await self.retrieve_memories("", embedding=embedding, **filters)
        else:
            # Filter-only search
            return await self.retrieve_memories("", **filters)
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory."""
        with self._get_connection() as conn:
            # Check if memory exists
            cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE memory_id = ?", (memory_id,))
            if cursor.fetchone()[0] == 0:
                return False
            
            # Build update query
            set_parts = []
            params = []
            
            for key, value in updates.items():
                if key == "metadata":
                    set_parts.append("metadata = ?")
                    params.append(json.dumps(value))
                elif key in ["memory_type", "content", "importance", "tier"]:
                    set_parts.append(f"{key} = ?")
                    params.append(value)
            
            if set_parts:
                set_parts.append("updated_at = ?")
                params.append(datetime.utcnow().isoformat())
                params.append(memory_id)
                
                sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE memory_id = ?"
                conn.execute(sql, params)
                conn.commit()
                
                logger.debug(f"Updated memory: {memory_id}")
                return True
        
        return False
    
    async def delete_memories(self, memory_ids: List[str]) -> List[str]:
        """Delete memories by IDs."""
        deleted = []
        
        with self._get_connection() as conn:
            for memory_id in memory_ids:
                cursor = conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                if cursor.rowcount > 0:
                    deleted.append(memory_id)
            conn.commit()
        
        logger.debug(f"Deleted {len(deleted)} memories")
        return deleted
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(DISTINCT memory_type) as unique_types,
                    COUNT(DISTINCT tier) as unique_tiers,
                    AVG(importance) as avg_importance,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM memories
            """)
            row = cursor.fetchone()
            
            # Get type distribution
            cursor = conn.execute("""
                SELECT memory_type, COUNT(*) as count 
                FROM memories 
                GROUP BY memory_type
            """)
            type_dist = {row["memory_type"]: row["count"] for row in cursor.fetchall()}
            
            # Get tier distribution
            cursor = conn.execute("""
                SELECT tier, COUNT(*) as count 
                FROM memories 
                GROUP BY tier
            """)
            tier_dist = {row["tier"]: row["count"] for row in cursor.fetchall()}
            
            return {
                "total_memories": row["total_count"],
                "unique_types": row["unique_types"],
                "unique_tiers": row["unique_tiers"],
                "average_importance": row["avg_importance"],
                "oldest_memory": row["oldest"],
                "newest_memory": row["newest"],
                "type_distribution": type_dist,
                "tier_distribution": tier_dist,
                "sqlite_vec_enabled": self.sqlite_vec_available
            }
    
    def close(self):
        """Close the database connection."""
        with self._connection_lock:
            if self._connection:
                self._connection.close()
                self._connection = None
        logger.info("SQLiteVec persistence domain closed")