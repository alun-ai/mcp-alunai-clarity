#!/usr/bin/env python3
"""
SQLite Migration Tests

Tests for migrating data to/from SQLite memory persistence:
- Data migration from legacy formats
- Schema version management
- Data integrity validation during migration
- Rollback capabilities
- Performance during migration
- Edge cases and error handling
"""

import asyncio
import json
import os
import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Any, Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from clarity.domains.sqlite_persistence import SQLiteMemoryPersistence


class TestSQLiteMigration:
    """Test suite for SQLite migration functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="migration_test_")
        db_path = os.path.join(temp_dir, "migration_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def legacy_db_path(self):
        """Create temporary legacy database path for testing."""
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="legacy_test_")
        db_path = os.path.join(temp_dir, "legacy_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing."""
        mock_model = Mock()
        
        def consistent_encode(text):
            # Generate consistent embeddings for migration testing
            import hashlib
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            embedding = [0.0] * 384
            
            # Use hash to generate deterministic embedding
            for i in range(min(384, len(text_hash) // 2)):
                hex_pair = text_hash[i*2:(i*2)+2]
                embedding[i] = int(hex_pair, 16) / 255.0 - 0.5  # Range -0.5 to 0.5
            
            return embedding
        
        mock_model.encode.side_effect = consistent_encode
        return mock_model
    
    @pytest.fixture
    def sqlite_persistence(self, temp_db_path, mock_embedding_model):
        """Create SQLiteMemoryPersistence instance for testing."""
        return SQLiteMemoryPersistence(temp_db_path, mock_embedding_model)
    
    def create_legacy_data_format(self) -> List[Dict[str, Any]]:
        """Create test data in legacy format for migration."""
        return [
            {
                "memory_id": "legacy-001",
                "type": "structured_thinking",
                "content": {
                    "analysis": "Legacy system performance analysis",
                    "conclusion": "Needs optimization",
                    "confidence": 0.8
                },
                "importance": 0.9,
                "tier": "long_term",
                "timestamp": "2024-01-15T10:30:00Z",
                "metadata": {
                    "source": "legacy_system",
                    "analyst": "system",
                    "version": "1.0"
                },
                "tags": ["performance", "analysis", "legacy"]
            },
            {
                "memory_id": "legacy-002",
                "type": "episodic",
                "content": "User reported login issues on 2024-01-15",
                "importance": 0.85,
                "tier": "short_term", 
                "timestamp": "2024-01-15T14:45:00Z",
                "metadata": {
                    "source": "support_ticket",
                    "ticket_id": "TK-12345",
                    "user_id": "user_789"
                },
                "tags": ["login", "issue", "support"]
            },
            {
                "memory_id": "invalid-uuid-format",  # Invalid ID to test sanitization
                "type": "semantic",
                "content": "Database ACID properties ensure consistency",
                "importance": 0.7,
                "tier": "archival",
                "timestamp": "2024-01-10T09:00:00Z",
                "metadata": {
                    "source": "knowledge_base",
                    "topic": "database_theory"
                },
                "tags": ["database", "theory"]
            },
            {
                # Missing required fields to test validation
                "memory_id": "legacy-incomplete",
                "content": "Incomplete legacy memory entry",
                "timestamp": "2024-01-12T16:20:00Z"
            }
        ]
    
    def create_legacy_database(self, db_path: str, legacy_data: List[Dict[str, Any]]):
        """Create a legacy database format for migration testing."""
        import sqlite3
        
        with sqlite3.connect(db_path) as conn:
            # Create legacy table schema (different from new schema)
            conn.executescript("""
                CREATE TABLE legacy_memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT,
                    content_json TEXT,
                    importance REAL,
                    tier TEXT,
                    created_timestamp TEXT,
                    metadata_json TEXT,
                    tags_json TEXT
                );
                
                CREATE INDEX idx_legacy_type ON legacy_memories(memory_type);
                CREATE INDEX idx_legacy_tier ON legacy_memories(tier);
            """)
            
            # Insert legacy data
            for memory in legacy_data:
                conn.execute("""
                    INSERT INTO legacy_memories 
                    (id, memory_type, content_json, importance, tier, created_timestamp, metadata_json, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.get("memory_id", ""),
                    memory.get("type", "unknown"),
                    json.dumps(memory.get("content", "")),
                    memory.get("importance", 0.5),
                    memory.get("tier", "short_term"),
                    memory.get("timestamp", datetime.utcnow().isoformat()),
                    json.dumps(memory.get("metadata", {})),
                    json.dumps(memory.get("tags", []))
                ))
    
    @pytest.mark.asyncio
    async def test_basic_data_migration(self, sqlite_persistence, legacy_db_path):
        """Test basic data migration from legacy format."""
        # Create legacy data and database
        legacy_data = self.create_legacy_data_format()
        self.create_legacy_database(legacy_db_path, legacy_data)
        
        # Perform migration
        migrated_count = await self.migrate_from_legacy(sqlite_persistence, legacy_db_path)
        
        assert migrated_count > 0
        
        # Verify migration by checking stored memories
        stats = await sqlite_persistence.get_memory_stats()
        assert stats["total_memories"] >= 3  # At least the valid entries
        
        # Test that migrated data is searchable
        results = await sqlite_persistence.retrieve_memories(
            "legacy system performance analysis",
            limit=5,
            min_similarity=0.0
        )
        
        assert len(results) > 0
        
        # Verify specific migrated memory
        legacy_001_found = False
        for result in results:
            if "legacy system" in str(result.get("content", "")).lower():
                legacy_001_found = True
                assert result["type"] == "structured_thinking"
                assert result["importance"] == 0.9
        
        assert legacy_001_found
    
    async def migrate_from_legacy(self, sqlite_persistence: SQLiteMemoryPersistence, legacy_db_path: str) -> int:
        """Migrate data from legacy database format."""
        import sqlite3
        
        migrated_count = 0
        errors = []
        
        with sqlite3.connect(legacy_db_path) as legacy_conn:
            cursor = legacy_conn.execute("""
                SELECT id, memory_type, content_json, importance, tier, 
                       created_timestamp, metadata_json, tags_json
                FROM legacy_memories
            """)
            
            for row in cursor.fetchall():
                try:
                    # Parse legacy data
                    legacy_id, memory_type, content_json, importance, tier, timestamp, metadata_json, tags_json = row
                    
                    # Convert to new format
                    content = json.loads(content_json) if content_json else ""
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    tags = json.loads(tags_json) if tags_json else []
                    
                    # Enhance metadata with migration info
                    metadata.update({
                        "migrated_from": "legacy_system",
                        "migration_timestamp": datetime.utcnow().isoformat(),
                        "original_tags": tags
                    })
                    
                    # Create new memory format
                    new_memory = {
                        "id": legacy_id,
                        "type": memory_type or "unknown",
                        "content": content,
                        "importance": importance or 0.5,
                        "tier": tier or "short_term",
                        "created_at": timestamp,
                        "metadata": metadata,
                        "context": {"migration_source": "legacy_database"}
                    }
                    
                    # Store migrated memory
                    await sqlite_persistence.store_memory(new_memory)
                    migrated_count += 1
                    
                except Exception as e:
                    errors.append(f"Failed to migrate {legacy_id}: {e}")
        
        if errors:
            print(f"Migration completed with {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  {error}")
        
        return migrated_count
    
    @pytest.mark.asyncio
    async def test_data_integrity_validation(self, sqlite_persistence, legacy_db_path):
        """Test data integrity validation during migration."""
        # Create legacy data with various integrity issues
        problematic_data = [
            {
                "memory_id": "integrity-001",
                "type": "structured_thinking",
                "content": {"analysis": "Valid content"},
                "importance": 1.5,  # Invalid: > 1.0
                "tier": "long_term",
                "timestamp": "2024-01-15T10:30:00Z"
            },
            {
                "memory_id": "",  # Invalid: empty ID
                "type": "episodic",
                "content": "Valid content",
                "importance": 0.8,
                "tier": "short_term",
                "timestamp": "2024-01-15T11:00:00Z"
            },
            {
                "memory_id": "integrity-003",
                "type": "invalid_type",  # Non-standard type
                "content": "",  # Empty content
                "importance": -0.1,  # Invalid: negative
                "tier": "invalid_tier",  # Invalid tier
                "timestamp": "invalid-timestamp"  # Invalid timestamp
            },
            {
                "memory_id": "integrity-004",
                "type": "semantic",
                "content": "Valid content with proper structure",
                "importance": 0.7,
                "tier": "archival",
                "timestamp": "2024-01-15T12:00:00Z"
            }
        ]
        
        self.create_legacy_database(legacy_db_path, problematic_data)
        
        # Perform migration with validation
        migrated_count = await self.migrate_with_validation(sqlite_persistence, legacy_db_path)
        
        # Should migrate valid entries and handle invalid ones gracefully
        assert migrated_count >= 1  # At least the valid entry should migrate
        
        # Verify that valid data was migrated correctly
        results = await sqlite_persistence.retrieve_memories(
            "valid content proper structure",
            limit=5,
            min_similarity=0.0
        )
        
        assert len(results) > 0
        
        # Check that invalid data was either corrected or skipped
        stats = await sqlite_persistence.get_memory_stats()
        assert stats["total_memories"] == migrated_count
    
    async def migrate_with_validation(self, sqlite_persistence: SQLiteMemoryPersistence, legacy_db_path: str) -> int:
        """Migrate data with integrity validation and correction."""
        import sqlite3
        
        migrated_count = 0
        corrected_count = 0
        skipped_count = 0
        
        with sqlite3.connect(legacy_db_path) as legacy_conn:
            cursor = legacy_conn.execute("""
                SELECT id, memory_type, content_json, importance, tier, 
                       created_timestamp, metadata_json, tags_json
                FROM legacy_memories
            """)
            
            for row in cursor.fetchall():
                try:
                    legacy_id, memory_type, content_json, importance, tier, timestamp, metadata_json, tags_json = row
                    
                    # Validate and correct data
                    corrections = []
                    
                    # Fix empty or invalid ID
                    if not legacy_id or legacy_id.strip() == "":
                        legacy_id = str(uuid.uuid4())
                        corrections.append("generated_new_id")
                    
                    # Validate importance
                    if importance is None or importance < 0:
                        importance = 0.5
                        corrections.append("corrected_importance")
                    elif importance > 1.0:
                        importance = 1.0
                        corrections.append("clamped_importance")
                    
                    # Validate memory type
                    valid_types = ["structured_thinking", "episodic", "procedural", "semantic", "unknown"]
                    if memory_type not in valid_types:
                        memory_type = "unknown"
                        corrections.append("corrected_type")
                    
                    # Validate tier
                    valid_tiers = ["short_term", "long_term", "archival", "system"]
                    if tier not in valid_tiers:
                        tier = "short_term"
                        corrections.append("corrected_tier")
                    
                    # Validate timestamp
                    try:
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        timestamp = datetime.utcnow().isoformat()
                        corrections.append("corrected_timestamp")
                    
                    # Parse content
                    try:
                        content = json.loads(content_json) if content_json else ""
                    except json.JSONDecodeError:
                        content = content_json or ""
                        corrections.append("simplified_content")
                    
                    # Skip if content is completely empty
                    if not content or (isinstance(content, str) and content.strip() == ""):
                        skipped_count += 1
                        continue
                    
                    # Parse metadata
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except json.JSONDecodeError:
                        metadata = {}
                        corrections.append("reset_metadata")
                    
                    # Add migration tracking
                    metadata.update({
                        "migrated_from": "legacy_system",
                        "migration_timestamp": datetime.utcnow().isoformat(),
                        "migration_corrections": corrections
                    })
                    
                    # Create corrected memory
                    new_memory = {
                        "id": legacy_id,
                        "type": memory_type,
                        "content": content,
                        "importance": importance,
                        "tier": tier,
                        "created_at": timestamp,
                        "metadata": metadata,
                        "context": {
                            "migration_source": "legacy_database",
                            "validation_applied": True,
                            "corrections_made": len(corrections)
                        }
                    }
                    
                    # Store migrated memory
                    await sqlite_persistence.store_memory(new_memory)
                    migrated_count += 1
                    
                    if corrections:
                        corrected_count += 1
                    
                except Exception as e:
                    print(f"Failed to migrate {legacy_id}: {e}")
                    skipped_count += 1
        
        print(f"Migration summary: {migrated_count} migrated, {corrected_count} corrected, {skipped_count} skipped")
        return migrated_count
    
    @pytest.mark.asyncio
    async def test_schema_version_management(self, sqlite_persistence):
        """Test schema version management during migration."""
        # Test current schema version
        import sqlite3
        
        with sqlite3.connect(sqlite_persistence.db_path) as conn:
            # Check if we can add version tracking
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL,
                        description TEXT
                    )
                """)
                
                # Add current version
                conn.execute("""
                    INSERT OR REPLACE INTO schema_version (version, applied_at, description)
                    VALUES (1, ?, 'Initial SQLite memory persistence schema')
                """, (datetime.utcnow().isoformat(),))
                
                # Verify version
                cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
                current_version = cursor.fetchone()[0]
                assert current_version == 1
                
                print(f"Schema version management: Current version {current_version}")
                
            except Exception as e:
                pytest.fail(f"Schema version management failed: {e}")
    
    @pytest.mark.asyncio
    async def test_rollback_capability(self, sqlite_persistence, legacy_db_path):
        """Test rollback capability during migration."""
        # Create backup before migration
        backup_path = sqlite_persistence.db_path + ".backup"
        
        # Store some initial data
        initial_memory = {
            "id": "initial-001",
            "type": "semantic",
            "content": "Initial data before migration",
            "importance": 0.6,
            "tier": "short_term"
        }
        await sqlite_persistence.store_memory(initial_memory)
        
        # Create backup
        import shutil
        shutil.copy2(sqlite_persistence.db_path, backup_path)
        
        # Get initial stats
        initial_stats = await sqlite_persistence.get_memory_stats()
        initial_count = initial_stats["total_memories"]
        
        # Perform migration
        legacy_data = self.create_legacy_data_format()
        self.create_legacy_database(legacy_db_path, legacy_data)
        
        migrated_count = await self.migrate_from_legacy(sqlite_persistence, legacy_db_path)
        
        # Verify migration increased memory count
        post_migration_stats = await sqlite_persistence.get_memory_stats()
        assert post_migration_stats["total_memories"] > initial_count
        
        # Simulate rollback
        print(f"Rolling back from {post_migration_stats['total_memories']} to {initial_count} memories")
        
        # Restore from backup
        shutil.copy2(backup_path, sqlite_persistence.db_path)
        
        # Re-initialize persistence to pick up restored database
        sqlite_persistence._ensure_database()
        
        # Verify rollback
        rollback_stats = await sqlite_persistence.get_memory_stats()
        assert rollback_stats["total_memories"] == initial_count
        
        # Verify original data is still accessible
        results = await sqlite_persistence.retrieve_memories(
            "initial data before migration",
            limit=5,
            min_similarity=0.0
        )
        
        assert len(results) > 0
        assert any("initial data" in str(r.get("content", "")).lower() for r in results)
        
        # Cleanup
        os.unlink(backup_path)
    
    @pytest.mark.asyncio
    async def test_migration_performance(self, sqlite_persistence, legacy_db_path):
        """Test migration performance with larger datasets."""
        # Create large legacy dataset
        large_legacy_data = []
        
        for i in range(500):
            memory = {
                "memory_id": f"perf-{i:04d}",
                "type": ["structured_thinking", "episodic", "procedural", "semantic"][i % 4],
                "content": f"Performance test memory {i} with content for migration testing",
                "importance": 0.1 + (i % 10) * 0.09,
                "tier": ["short_term", "long_term", "archival", "system"][i % 4],
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "batch": i // 100,
                    "index": i,
                    "category": f"category_{i % 20}"
                },
                "tags": [f"tag_{i % 10}", f"batch_{i // 100}"]
            }
            large_legacy_data.append(memory)
        
        self.create_legacy_database(legacy_db_path, large_legacy_data)
        
        # Measure migration performance
        import time
        
        migration_start = time.perf_counter()
        migrated_count = await self.migrate_from_legacy(sqlite_persistence, legacy_db_path)
        migration_time = time.perf_counter() - migration_start
        
        migration_rate = migrated_count / migration_time
        
        print(f"\nMigration Performance:")
        print(f"  Migrated memories: {migrated_count}")
        print(f"  Migration time: {migration_time:.2f}s")
        print(f"  Migration rate: {migration_rate:.1f} memories/sec")
        
        # Performance assertions
        assert migrated_count >= 450  # Should migrate most memories
        assert migration_rate > 10  # Should migrate at least 10 memories/sec
        
        # Verify migrated data is searchable
        search_start = time.perf_counter()
        search_results = await sqlite_persistence.retrieve_memories(
            "performance test memory migration",
            limit=10,
            min_similarity=0.0
        )
        search_time = time.perf_counter() - search_start
        
        print(f"  Search time: {search_time * 1000:.2f}ms")
        print(f"  Search results: {len(search_results)}")
        
        assert len(search_results) > 0
        assert search_time < 0.1  # Search should be fast even after large migration
    
    @pytest.mark.asyncio
    async def test_incremental_migration(self, sqlite_persistence, legacy_db_path):
        """Test incremental migration capability."""
        # Initial migration batch
        initial_batch = self.create_legacy_data_format()[:2]  # First 2 memories
        self.create_legacy_database(legacy_db_path, initial_batch)
        
        # Perform initial migration
        initial_count = await self.migrate_from_legacy(sqlite_persistence, legacy_db_path)
        assert initial_count == 2
        
        # Add more data to legacy database (simulating incremental data)
        additional_data = [
            {
                "memory_id": "incremental-001",
                "type": "procedural",
                "content": "New procedural memory added later",
                "importance": 0.8,
                "tier": "long_term",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"source": "incremental_batch"}
            },
            {
                "memory_id": "incremental-002",
                "type": "episodic",
                "content": "New episodic memory from second batch",
                "importance": 0.7,
                "tier": "short_term",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"source": "incremental_batch"}
            }
        ]
        
        # Add to existing legacy database
        import sqlite3
        with sqlite3.connect(legacy_db_path) as conn:
            for memory in additional_data:
                conn.execute("""
                    INSERT INTO legacy_memories 
                    (id, memory_type, content_json, importance, tier, created_timestamp, metadata_json, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory["memory_id"],
                    memory["type"],
                    json.dumps(memory["content"]),
                    memory["importance"],
                    memory["tier"],
                    memory["timestamp"],
                    json.dumps(memory.get("metadata", {})),
                    json.dumps([])
                ))
        
        # Perform incremental migration (only new data)
        incremental_count = await self.migrate_incremental(sqlite_persistence, legacy_db_path)
        
        # Verify incremental migration
        assert incremental_count == 2  # Should migrate only the new memories
        
        # Verify total count
        final_stats = await sqlite_persistence.get_memory_stats()
        assert final_stats["total_memories"] >= 4  # Initial + incremental
        
        # Verify incremental data is searchable
        incremental_results = await sqlite_persistence.retrieve_memories(
            "incremental batch procedural episodic",
            limit=5,
            min_similarity=0.0
        )
        
        assert len(incremental_results) > 0
    
    async def migrate_incremental(self, sqlite_persistence: SQLiteMemoryPersistence, legacy_db_path: str) -> int:
        """Perform incremental migration (only new records)."""
        import sqlite3
        
        # Get existing memory IDs to avoid duplicates
        existing_stats = await sqlite_persistence.get_memory_stats()
        
        # For simplicity, track by checking metadata for migration timestamp
        # In a real implementation, you'd maintain a migration log
        
        migrated_count = 0
        
        with sqlite3.connect(legacy_db_path) as legacy_conn:
            # Get all legacy records
            cursor = legacy_conn.execute("""
                SELECT id, memory_type, content_json, importance, tier, 
                       created_timestamp, metadata_json, tags_json
                FROM legacy_memories
            """)
            
            for row in cursor.fetchall():
                legacy_id, memory_type, content_json, importance, tier, timestamp, metadata_json, tags_json = row
                
                # Check if already migrated by trying to retrieve
                existing_memory = await sqlite_persistence.get_memory(legacy_id)
                if existing_memory:
                    continue  # Skip already migrated
                
                try:
                    # Parse and migrate new record
                    content = json.loads(content_json) if content_json else ""
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    metadata.update({
                        "migrated_from": "legacy_system",
                        "migration_timestamp": datetime.utcnow().isoformat(),
                        "migration_type": "incremental"
                    })
                    
                    new_memory = {
                        "id": legacy_id,
                        "type": memory_type or "unknown",
                        "content": content,
                        "importance": importance or 0.5,
                        "tier": tier or "short_term",
                        "created_at": timestamp,
                        "metadata": metadata,
                        "context": {"migration_source": "legacy_database_incremental"}
                    }
                    
                    await sqlite_persistence.store_memory(new_memory)
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"Failed to migrate incremental record {legacy_id}: {e}")
        
        return migrated_count
    
    @pytest.mark.asyncio
    async def test_migration_error_handling(self, sqlite_persistence, legacy_db_path):
        """Test error handling during migration."""
        # Create problematic legacy data
        problematic_data = [
            {
                "memory_id": "error-001",
                "type": "structured_thinking",
                "content": None,  # Invalid: None content
                "importance": 0.8,
                "tier": "long_term",
                "timestamp": "2024-01-15T10:30:00Z"
            },
            {
                "memory_id": None,  # Invalid: None ID
                "type": "episodic",
                "content": "Valid content",
                "importance": 0.7,
                "tier": "short_term",
                "timestamp": "2024-01-15T11:00:00Z"
            }
        ]
        
        # Create legacy database with problematic data
        import sqlite3
        with sqlite3.connect(legacy_db_path) as conn:
            conn.execute("""
                CREATE TABLE legacy_memories (
                    id TEXT,
                    memory_type TEXT,
                    content_json TEXT,
                    importance REAL,
                    tier TEXT,
                    created_timestamp TEXT,
                    metadata_json TEXT,
                    tags_json TEXT
                )
            """)
            
            # Insert problematic data
            for memory in problematic_data:
                conn.execute("""
                    INSERT INTO legacy_memories 
                    (id, memory_type, content_json, importance, tier, created_timestamp, metadata_json, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.get("memory_id"),
                    memory.get("type"),
                    json.dumps(memory.get("content")),
                    memory.get("importance"),
                    memory.get("tier"),
                    memory.get("timestamp"),
                    json.dumps({}),
                    json.dumps([])
                ))
        
        # Attempt migration with error handling
        migrated_count = await self.migrate_with_error_handling(sqlite_persistence, legacy_db_path)
        
        # Should handle errors gracefully and continue processing
        # Even if some records fail, the process should not crash
        assert isinstance(migrated_count, int)
        assert migrated_count >= 0
    
    async def migrate_with_error_handling(self, sqlite_persistence: SQLiteMemoryPersistence, legacy_db_path: str) -> int:
        """Migrate with comprehensive error handling."""
        import sqlite3
        
        migrated_count = 0
        error_count = 0
        errors = []
        
        try:
            with sqlite3.connect(legacy_db_path) as legacy_conn:
                cursor = legacy_conn.execute("""
                    SELECT id, memory_type, content_json, importance, tier, 
                           created_timestamp, metadata_json, tags_json
                    FROM legacy_memories
                """)
                
                for row in cursor.fetchall():
                    try:
                        legacy_id, memory_type, content_json, importance, tier, timestamp, metadata_json, tags_json = row
                        
                        # Skip records with critical missing data
                        if not legacy_id:
                            error_count += 1
                            errors.append("Missing memory ID")
                            continue
                        
                        # Handle content parsing errors
                        try:
                            content = json.loads(content_json) if content_json else ""
                        except json.JSONDecodeError:
                            content = str(content_json) if content_json else ""
                        
                        # Skip if content is None or empty after parsing
                        if content is None or (isinstance(content, str) and not content.strip()):
                            error_count += 1
                            errors.append(f"Empty content for {legacy_id}")
                            continue
                        
                        # Attempt to store memory
                        new_memory = {
                            "id": legacy_id,
                            "type": memory_type or "unknown",
                            "content": content,
                            "importance": importance or 0.5,
                            "tier": tier or "short_term",
                            "created_at": timestamp or datetime.utcnow().isoformat(),
                            "metadata": {"migrated_from": "legacy_system"},
                            "context": {"migration_source": "legacy_database"}
                        }
                        
                        await sqlite_persistence.store_memory(new_memory)
                        migrated_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        errors.append(f"Migration error for {legacy_id}: {str(e)}")
                        continue  # Continue processing other records
                        
        except Exception as e:
            errors.append(f"Critical migration error: {str(e)}")
        
        print(f"Migration completed: {migrated_count} success, {error_count} errors")
        if errors:
            print("Sample errors:", errors[:3])
        
        return migrated_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])