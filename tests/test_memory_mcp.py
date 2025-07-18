"""
Tests for the Memory MCP Server.
"""

import os
import json
import tempfile
import unittest
from typing import Dict, Any

from memory_mcp.utils.config import load_config, create_default_config
from memory_mcp.utils.schema import validate_memory
from memory_mcp.utils.embeddings import EmbeddingManager


class TestConfig(unittest.TestCase):
    """Tests for configuration utilities."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            try:
                # Create default config
                config = create_default_config(temp.name)
                
                # Check if config file was created
                self.assertTrue(os.path.exists(temp.name))
                
                # Check if config has expected sections
                self.assertIn("server", config)
                self.assertIn("memory", config)
                self.assertIn("embedding", config)
                
                # Load the created config
                loaded_config = load_config(temp.name)
                
                # Check if loaded config matches
                self.assertEqual(config, loaded_config)
            finally:
                # Clean up
                os.unlink(temp.name)
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration."""
        # Use a path that doesn't exist
        with tempfile.NamedTemporaryFile(suffix=".json") as temp:
            pass  # File is deleted on close
        
        # Load config (should create default)
        config = load_config(temp.name)
        
        # Check if config has expected sections
        self.assertIn("server", config)
        self.assertIn("memory", config)
        self.assertIn("embedding", config)
        
        # Clean up
        if os.path.exists(temp.name):
            os.unlink(temp.name)


class TestSchema(unittest.TestCase):
    """Tests for schema validation utilities."""
    
    def test_validate_conversation_memory(self):
        """Test validating conversation memory."""
        # Valid conversation with role/message
        memory = {
            "id": "mem_test1",
            "type": "conversation",
            "importance": 0.8,
            "content": {
                "role": "user",
                "message": "Hello, Claude!"
            }
        }
        
        validated = validate_memory(memory)
        self.assertEqual(validated["id"], "mem_test1")
        self.assertEqual(validated["type"], "conversation")
        
        # Valid conversation with messages array
        memory = {
            "id": "mem_test2",
            "type": "conversation",
            "importance": 0.7,
            "content": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        }
        
        validated = validate_memory(memory)
        self.assertEqual(validated["id"], "mem_test2")
        self.assertEqual(validated["type"], "conversation")
        
        # Invalid: missing required fields
        memory = {
            "id": "mem_test3",
            "type": "conversation",
            "importance": 0.5,
            "content": {}
        }
        
        with self.assertRaises(ValueError):
            validate_memory(memory)
    
    def test_validate_fact_memory(self):
        """Test validating fact memory."""
        # Valid fact
        memory = {
            "id": "mem_test4",
            "type": "fact",
            "importance": 0.9,
            "content": {
                "fact": "The capital of France is Paris.",
                "confidence": 0.95
            }
        }
        
        validated = validate_memory(memory)
        self.assertEqual(validated["id"], "mem_test4")
        self.assertEqual(validated["type"], "fact")
        
        # Invalid: missing fact field
        memory = {
            "id": "mem_test5",
            "type": "fact",
            "importance": 0.7,
            "content": {
                "confidence": 0.8
            }
        }
        
        with self.assertRaises(ValueError):
            validate_memory(memory)


class TestEmbeddings(unittest.TestCase):
    """Tests for embedding utilities."""
    
    def test_embedding_manager_init(self):
        """Test initializing the embedding manager."""
        config = {
            "embedding": {
                "model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
                "dimensions": 384,
                "cache_dir": None
            }
        }
        
        manager = EmbeddingManager(config)
        self.assertEqual(manager.model_name, "sentence-transformers/paraphrase-MiniLM-L3-v2")
        self.assertEqual(manager.dimensions, 384)
        self.assertIsNone(manager.model)  # Model should be None initially
    
    def test_similarity_calculation(self):
        """Test similarity calculation between embeddings."""
        config = {
            "embedding": {
                "model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
                "dimensions": 384
            }
        }
        
        manager = EmbeddingManager(config)
        
        # Test with numpy arrays
        import numpy as np
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([1.0, 1.0, 0.0])
        
        # Orthogonal vectors should have similarity 0
        self.assertAlmostEqual(manager.calculate_similarity(v1, v2), 0.0)
        
        # Same vector should have similarity 1
        self.assertAlmostEqual(manager.calculate_similarity(v1, v1), 1.0)
        
        # Test with lists
        v1_list = [1.0, 0.0, 0.0]
        v2_list = [0.0, 1.0, 0.0]
        
        # Orthogonal vectors should have similarity 0
        self.assertAlmostEqual(manager.calculate_similarity(v1_list, v2_list), 0.0)


if __name__ == "__main__":
    unittest.main()
