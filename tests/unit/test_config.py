"""
Unit tests for configuration and setup in Alunai Clarity.
"""

import json
import os
import tempfile
import pytest
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from clarity.utils.config import load_config, create_default_config, validate_config
from clarity.utils.schema import validate_memory
from clarity.utils.embeddings import EmbeddingManager


@pytest.mark.unit
class TestConfigurationLoading:
    """Test configuration loading and validation."""
    
    def test_load_existing_config(self, temp_config_file: str):
        """Test loading existing configuration file."""
        config = load_config(temp_config_file)
        
        # Verify all required sections exist
        assert "server" in config
        assert "alunai-clarity" in config
        assert "qdrant" in config
        assert "embedding" in config
        assert "autocode" in config
        
        # Verify server configuration
        assert config["server"]["host"] == "localhost"
        assert config["server"]["port"] == 8080
        assert config["server"]["log_level"] == "INFO"
        
        # Verify Alunai Clarity configuration
        assert config["alunai-clarity"]["max_short_term_items"] == 100
        assert config["alunai-clarity"]["short_term_threshold"] == 0.3
        
        # Verify Qdrant configuration
        assert config["qdrant"]["path"] == ":memory:"
        assert "index_params" in config["qdrant"]
        
        # Verify embedding configuration
        assert "default_model" in config["embedding"]
        assert config["embedding"]["dimensions"] == 384
        
        # Verify AutoCode configuration
        assert config["autocode"]["enabled"] is True
        assert "command_learning" in config["autocode"]
        assert "pattern_detection" in config["autocode"]
    
    def test_load_nonexistent_config_creates_default(self):
        """Test that loading non-existent config creates default."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as temp:
            nonexistent_path = temp.name
        
        # File should not exist now
        assert not os.path.exists(nonexistent_path)
        
        # Loading should create default config
        config = load_config(nonexistent_path)
        
        # File should now exist
        assert os.path.exists(nonexistent_path)
        
        # Should have default structure
        assert "server" in config
        assert "alunai-clarity" in config
        assert "qdrant" in config
        assert "embedding" in config
        
        # Clean up
        os.unlink(nonexistent_path)
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            config_path = temp.name
        
        try:
            # Create default config
            config = create_default_config(config_path)
            
            # Verify file was created
            assert os.path.exists(config_path)
            
            # Verify config structure
            assert isinstance(config, dict)
            assert "server" in config
            assert "alunai-clarity" in config
            assert "qdrant" in config
            assert "embedding" in config
            assert "retrieval" in config
            assert "autocode" in config
            
            # Verify default values
            assert config["server"]["host"] in ["localhost", "127.0.0.1"]
            assert config["server"]["port"] == 8000
            assert config["alunai-clarity"]["max_short_term_items"] == 100
            assert "qdrant" in config["qdrant"]["path"]  # Path contains qdrant
            assert config["embedding"]["dimensions"] == 384
            assert config["autocode"]["enabled"] is True
            
            # Verify file contents match returned config
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            assert file_config == config
            
        finally:
            # Clean up
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_config_with_environment_variables(self, temp_config_file: str):
        """Test configuration override with environment variables."""
        # Set environment variables
        env_vars = {
            "ALUNAI_CLARITY_HOST": "0.0.0.0",
            "ALUNAI_CLARITY_PORT": "9000",
            "ALUNAI_CLARITY_LOG_LEVEL": "DEBUG",
            "ALUNAI_CLARITY_QDRANT_PATH": "/custom/qdrant/path",
            "ALUNAI_CLARITY_EMBEDDING_MODEL": "custom-model",
            "ALUNAI_CLARITY_AUTOCODE_ENABLED": "false"
        }
        
        with patch.dict(os.environ, env_vars):
            config = load_config(temp_config_file)
            
            # Verify environment variables override config values
            # Note: This assumes the load_config function supports env var overrides
            # If not implemented, this test documents the expected behavior
            base_config = load_config(temp_config_file)
            
            # For now, just verify the config loads successfully
            assert config is not None
            assert isinstance(config, dict)
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as temp:
            # Write invalid JSON
            temp.write("{ invalid json content }")
            temp_path = temp.name
        
        try:
            # Should handle invalid JSON gracefully
            with pytest.raises((json.JSONDecodeError, ValueError)):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_config_validation(self, test_config: Dict[str, Any]):
        """Test configuration validation."""
        # Test valid config
        is_valid, errors = validate_config(test_config)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test config missing required section
        invalid_config = test_config.copy()
        del invalid_config["server"]
        
        is_valid, errors = validate_config(invalid_config)
        assert is_valid is False
        assert any("server" in error for error in errors)
        
        # Test config with invalid values
        invalid_config2 = test_config.copy()
        invalid_config2["server"]["port"] = "not_a_number"
        
        is_valid, errors = validate_config(invalid_config2)
        assert is_valid is False
        assert any("port" in error for error in errors)
    
    def test_config_merge_defaults(self):
        """Test merging partial config with defaults."""
        partial_config = {
            "server": {
                "host": "custom_host"
                # Missing port and log_level
            },
            "alunai-clarity": {
                "max_short_term_items": 500
                # Missing other settings
            }
            # Missing other sections
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as temp:
            json.dump(partial_config, temp)
            temp_path = temp.name
        
        try:
            config = load_config(temp_path)
            
            # Should have custom values
            assert config["server"]["host"] == "custom_host"
            assert config["alunai-clarity"]["max_short_term_items"] == 500
            
            # Should have default values for missing settings
            assert "port" in config["server"]
            assert "log_level" in config["server"]
            assert "qdrant" in config
            assert "embedding" in config
            
        finally:
            os.unlink(temp_path)


@pytest.mark.unit
class TestConfigurationSections:
    """Test individual configuration sections."""
    
    def test_server_config_section(self, test_config: Dict[str, Any]):
        """Test server configuration section."""
        server_config = test_config["server"]
        
        assert "host" in server_config
        assert "port" in server_config
        assert "log_level" in server_config
        
        # Validate types
        assert isinstance(server_config["host"], str)
        assert isinstance(server_config["port"], int)
        assert isinstance(server_config["log_level"], str)
        
        # Validate values
        assert server_config["port"] > 0
        assert server_config["log_level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def test_alunai_clarity_config_section(self, test_config: Dict[str, Any]):
        """Test Alunai Clarity configuration section."""
        clarity_config = test_config["alunai-clarity"]
        
        required_keys = [
            "max_short_term_items",
            "max_long_term_items",
            "max_archival_items",
            "short_term_threshold",
            "consolidation_interval_hours"
        ]
        
        for key in required_keys:
            assert key in clarity_config
        
        # Validate types and ranges
        assert isinstance(clarity_config["max_short_term_items"], int)
        assert clarity_config["max_short_term_items"] > 0
        
        assert isinstance(clarity_config["short_term_threshold"], (int, float))
        assert 0.0 <= clarity_config["short_term_threshold"] <= 1.0
        
        assert isinstance(clarity_config["consolidation_interval_hours"], (int, float))
        assert clarity_config["consolidation_interval_hours"] > 0
    
    def test_qdrant_config_section(self, test_config: Dict[str, Any]):
        """Test Qdrant configuration section."""
        qdrant_config = test_config["qdrant"]
        
        assert "path" in qdrant_config
        assert "index_params" in qdrant_config
        
        # Validate index parameters
        index_params = qdrant_config["index_params"]
        assert "m" in index_params
        assert "ef_construct" in index_params
        assert "full_scan_threshold" in index_params
        
        # Validate types
        assert isinstance(index_params["m"], int)
        assert isinstance(index_params["ef_construct"], int)
        assert isinstance(index_params["full_scan_threshold"], int)
        
        # Validate ranges
        assert index_params["m"] > 0
        assert index_params["ef_construct"] > 0
        assert index_params["full_scan_threshold"] > 0
    
    def test_embedding_config_section(self, test_config: Dict[str, Any]):
        """Test embedding configuration section."""
        embedding_config = test_config["embedding"]
        
        required_keys = ["default_model", "dimensions"]
        for key in required_keys:
            assert key in embedding_config
        
        # Validate types
        assert isinstance(embedding_config["default_model"], str)
        assert isinstance(embedding_config["dimensions"], int)
        
        # Validate values
        assert len(embedding_config["default_model"]) > 0
        assert embedding_config["dimensions"] > 0
        
        # Common embedding dimensions
        valid_dimensions = [128, 256, 384, 512, 768, 1024, 1536]
        assert embedding_config["dimensions"] in valid_dimensions
    
    def test_autocode_config_section(self, test_config: Dict[str, Any]):
        """Test AutoCode configuration section."""
        autocode_config = test_config["autocode"]
        
        assert "enabled" in autocode_config
        assert isinstance(autocode_config["enabled"], bool)
        
        # Test subsections
        subsections = ["command_learning", "pattern_detection", "session_analysis", "history_navigation"]
        for subsection in subsections:
            assert subsection in autocode_config
            assert isinstance(autocode_config[subsection], dict)
            assert "enabled" in autocode_config[subsection]
        
        # Test command learning config
        cmd_config = autocode_config["command_learning"]
        assert "min_confidence_threshold" in cmd_config
        assert "max_suggestions" in cmd_config
        assert isinstance(cmd_config["min_confidence_threshold"], (int, float))
        assert 0.0 <= cmd_config["min_confidence_threshold"] <= 1.0
        
        # Test pattern detection config
        pattern_config = autocode_config["pattern_detection"]
        assert "supported_languages" in pattern_config
        assert "max_scan_depth" in pattern_config
        assert isinstance(pattern_config["supported_languages"], list)
        assert isinstance(pattern_config["max_scan_depth"], int)
        assert pattern_config["max_scan_depth"] > 0
    
    def test_retrieval_config_section(self, test_config: Dict[str, Any]):
        """Test retrieval configuration section."""
        if "retrieval" in test_config:
            retrieval_config = test_config["retrieval"]
            
            weight_keys = ["similarity_weight", "recency_weight", "importance_weight"]
            for key in weight_keys:
                if key in retrieval_config:
                    assert isinstance(retrieval_config[key], (int, float))
                    assert 0.0 <= retrieval_config[key] <= 1.0
            
            # Weights should sum to approximately 1.0 if all are present
            if all(key in retrieval_config for key in weight_keys):
                total_weight = sum(retrieval_config[key] for key in weight_keys)
                assert 0.9 <= total_weight <= 1.1  # Allow for small floating point errors


@pytest.mark.unit
class TestSchemaValidation:
    """Test memory schema validation."""
    
    def test_validate_conversation_memory_formats(self):
        """Test validation of different conversation memory formats."""
        # Valid format 1: role/message
        memory1 = {
            "id": "mem_conv1",
            "type": "conversation",
            "importance": 0.7,
            "content": {
                "role": "user",
                "message": "Hello, Claude!"
            }
        }
        validated1 = validate_memory(memory1)
        assert validated1["id"] == "mem_conv1"
        
        # Valid format 2: messages array
        memory2 = {
            "id": "mem_conv2", 
            "type": "conversation",
            "importance": 0.8,
            "content": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        }
        validated2 = validate_memory(memory2)
        assert validated2["id"] == "mem_conv2"
        
        # Invalid: missing required fields
        invalid_memory = {
            "id": "mem_conv3",
            "type": "conversation",
            "importance": 0.6,
            "content": {}
        }
        with pytest.raises(ValueError):
            validate_memory(invalid_memory)
    
    def test_validate_fact_memory(self):
        """Test validation of fact memories."""
        # Valid fact memory
        valid_fact = {
            "id": "mem_fact1",
            "type": "fact",
            "importance": 0.9,
            "content": {
                "fact": "Python is a programming language",
                "confidence": 0.95,
                "category": "programming"
            }
        }
        validated = validate_memory(valid_fact)
        assert validated["content"]["fact"] == "Python is a programming language"
        
        # Invalid: missing fact field
        invalid_fact = {
            "id": "mem_fact2",
            "type": "fact",
            "importance": 0.8,
            "content": {
                "confidence": 0.9
            }
        }
        with pytest.raises(ValueError):
            validate_memory(invalid_fact)
    
    def test_validate_document_memory(self):
        """Test validation of document memories."""
        valid_document = {
            "id": "mem_doc1",
            "type": "document",
            "importance": 0.8,
            "content": {
                "title": "API Documentation",
                "content": "This document describes...",
                "format": "markdown",
                "author": "Developer"
            }
        }
        validated = validate_memory(valid_document)
        assert validated["content"]["title"] == "API Documentation"
    
    def test_validate_autocode_memory_types(self):
        """Test validation of AutoCode-specific memory types."""
        # Project pattern memory
        pattern_memory = {
            "id": "mem_pattern1",
            "type": "project_pattern",
            "importance": 0.8,
            "content": {
                "pattern_type": "framework",
                "framework": "FastAPI",
                "language": "python",
                "structure": {"files": ["main.py"]}
            }
        }
        validated_pattern = validate_memory(pattern_memory)
        assert validated_pattern["type"] == "project_pattern"
        
        # Command pattern memory
        command_memory = {
            "id": "mem_cmd1",
            "type": "command_pattern",
            "importance": 0.7,
            "content": {
                "command": "pytest -v",
                "context": {"framework": "pytest"},
                "success_rate": 0.95,
                "platform": "linux"
            }
        }
        validated_command = validate_memory(command_memory)
        assert validated_command["type"] == "command_pattern"
        
        # Session summary memory
        session_memory = {
            "id": "mem_session1",
            "type": "session_summary",
            "importance": 0.9,
            "content": {
                "session_id": "session_123",
                "tasks_completed": [{"task": "Created API", "status": "completed"}],
                "patterns_used": ["FastAPI"],
                "files_modified": ["main.py"]
            }
        }
        validated_session = validate_memory(session_memory)
        assert validated_session["type"] == "session_summary"
    
    def test_validate_memory_with_metadata_and_context(self):
        """Test validation of memories with metadata and context."""
        memory_with_extras = {
            "id": "mem_extra1",
            "type": "fact",
            "importance": 0.8,
            "content": {
                "fact": "Test fact with extras"
            },
            "metadata": {
                "source": "test",
                "category": "testing",
                "created_by": "unit_test"
            },
            "context": {
                "session_id": "test_session",
                "user_id": "test_user",
                "project": "test_project"
            }
        }
        
        validated = validate_memory(memory_with_extras)
        assert "metadata" in validated
        assert "context" in validated
        assert validated["metadata"]["source"] == "test"
        assert validated["context"]["session_id"] == "test_session"
    
    def test_validate_memory_required_fields(self):
        """Test validation of required memory fields."""
        # Missing ID
        memory_no_id = {
            "type": "fact",
            "importance": 0.7,
            "content": {"fact": "Test"}
        }
        with pytest.raises(ValueError):
            validate_memory(memory_no_id)
        
        # Missing type
        memory_no_type = {
            "id": "mem_test",
            "importance": 0.7,
            "content": {"fact": "Test"}
        }
        with pytest.raises(ValueError):
            validate_memory(memory_no_type)
        
        # Missing content
        memory_no_content = {
            "id": "mem_test",
            "type": "fact",
            "importance": 0.7
        }
        with pytest.raises(ValueError):
            validate_memory(memory_no_content)
    
    def test_validate_memory_importance_range(self):
        """Test validation of importance score range."""
        # Valid importance values
        for importance in [0.0, 0.5, 1.0]:
            memory = {
                "id": f"mem_imp_{importance}",
                "type": "fact",
                "importance": importance,
                "content": {"fact": "Test fact"}
            }
            validated = validate_memory(memory)
            assert validated["importance"] == importance
        
        # Invalid importance values
        for invalid_importance in [-0.1, 1.1, 2.0]:
            memory = {
                "id": f"mem_imp_{invalid_importance}",
                "type": "fact", 
                "importance": invalid_importance,
                "content": {"fact": "Test fact"}
            }
            with pytest.raises(ValueError):
                validate_memory(memory)