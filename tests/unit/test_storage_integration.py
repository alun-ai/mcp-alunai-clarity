"""
Test structured thinking storage integration with enhanced persistence domain.

This module validates the storage and retrieval of structured thinking
components using the enhanced persistence domain.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

from clarity.domains.structured_thinking import (
    StructuredThought, 
    ThinkingSession, 
    ThinkingStage, 
    ThoughtRelationship
)
from clarity.utils.schema import validate_memory


class TestStorageIntegration:
    """Test storage integration for structured thinking components"""
    
    def test_structured_thinking_memory_validation(self):
        """Test validation of structured thinking memory types"""
        
        # Test structured thinking memory validation
        structured_thinking_memory = {
            "id": "mem_test_123",
            "type": "structured_thinking",
            "importance": 0.8,
            "content": {
                "session_id": "session_123",
                "session_title": "Test Thinking Session",
                "total_thoughts": 5,
                "stages_completed": ["problem_definition", "research"]
            }
        }
        
        validated = validate_memory(structured_thinking_memory)
        assert validated["type"] == "structured_thinking"
        assert validated["content"]["session_id"] == "session_123"
        assert validated["content"]["total_thoughts"] == 5
    
    def test_thought_process_memory_validation(self):
        """Test validation of thought process memory"""
        
        thought_process_memory = {
            "id": "mem_thought_456", 
            "type": "thought_process",
            "importance": 0.7,
            "content": {
                "thought_number": 2,
                "thinking_stage": "research",
                "thought_content": "Research findings about authentication methods"
            }
        }
        
        validated = validate_memory(thought_process_memory)
        assert validated["type"] == "thought_process"
        assert validated["content"]["thought_number"] == 2
        assert validated["content"]["thinking_stage"] == "research"
    
    def test_thinking_relationship_memory_validation(self):
        """Test validation of thinking relationship memory"""
        
        relationship_memory = {
            "id": "mem_rel_789",
            "type": "thinking_relationship", 
            "importance": 0.8,
            "content": {
                "source_thought_id": "thought_1",
                "target_thought_id": "thought_2",
                "relationship_type": "builds_on",
                "strength": 0.9
            }
        }
        
        validated = validate_memory(relationship_memory)
        assert validated["type"] == "thinking_relationship"
        assert validated["content"]["relationship_type"] == "builds_on"
        assert validated["content"]["strength"] == 0.9
    
    def test_stage_specific_memory_validation(self):
        """Test validation of stage-specific memory types"""
        
        # Test problem analysis memory
        problem_memory = {
            "id": "mem_problem_001",
            "type": "problem_analysis",
            "importance": 0.9,
            "content": {
                "problem_statement": "Need to implement user authentication",
                "constraints": ["Security requirements", "Performance needs"],
                "success_criteria": ["Secure login", "Fast response"]
            }
        }
        
        validated = validate_memory(problem_memory)
        assert validated["type"] == "problem_analysis"
        
        # Test research notes memory
        research_memory = {
            "id": "mem_research_001",
            "type": "research_notes",
            "importance": 0.7,
            "content": {
                "research_topic": "JWT Authentication",
                "findings": ["JWT is stateless", "Requires secure secret"],
                "sources": ["RFC 7519", "OWASP Guidelines"]
            }
        }
        
        validated = validate_memory(research_memory)
        assert validated["type"] == "research_notes"
        assert len(validated["content"]["sources"]) == 2
    
    def test_memory_validation_errors(self):
        """Test memory validation error handling"""
        
        # Test missing required field
        with pytest.raises(ValueError, match="must have 'session_id' field"):
            invalid_memory = {
                "id": "mem_invalid",
                "type": "structured_thinking",
                "importance": 0.5,
                "content": {
                    "session_title": "Invalid Session",
                    "total_thoughts": 3,
                    "stages_completed": []
                    # Missing session_id
                }
            }
            validate_memory(invalid_memory)
        
        # Test invalid thinking stage
        with pytest.raises(ValueError, match="must be one of"):
            invalid_thought = {
                "id": "mem_invalid_thought",
                "type": "thought_process",
                "importance": 0.5,
                "content": {
                    "thought_number": 1,
                    "thinking_stage": "invalid_stage",  # Invalid stage
                    "thought_content": "Some content"
                }
            }
            validate_memory(invalid_thought)
        
        # Test invalid relationship type
        with pytest.raises(ValueError, match="must be one of"):
            invalid_relationship = {
                "id": "mem_invalid_rel",
                "type": "thinking_relationship",
                "importance": 0.5,
                "content": {
                    "source_thought_id": "thought_1",
                    "target_thought_id": "thought_2", 
                    "relationship_type": "invalid_relationship",  # Invalid type
                    "strength": 0.7
                }
            }
            validate_memory(invalid_relationship)
    
    @pytest.mark.asyncio
    async def test_memory_mapper_integration(self):
        """Test integration with ThinkingMemoryMapper"""
        
        from clarity.domains.structured_thinking_utils import ThinkingMemoryMapper
        
        # Create test thought
        thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.ANALYSIS,
            content="Analysis of authentication requirements and security considerations",
            tags=["authentication", "security", "analysis"],
            axioms=["Security first"],
            assumptions_challenged=["Users always use strong passwords"],
            importance=0.8
        )
        
        # Test memory type mapping
        memory_type = ThinkingMemoryMapper.thought_to_memory_type(thought)
        assert memory_type == "analysis_result"
        
        # Test metadata preparation
        metadata = ThinkingMemoryMapper.prepare_memory_metadata(thought, "session_123")
        assert metadata["thinking_stage"] == "analysis"
        assert metadata["thought_number"] == 1
        assert metadata["tags"] == ["authentication", "security", "analysis"]
        assert metadata["axioms"] == ["Security first"]
        assert metadata["assumptions_challenged"] == ["Users always use strong passwords"]
        assert metadata["structured_thinking"] is True
        assert metadata["thinking_session_id"] == "session_123"
    
    def test_schema_memory_type_coverage(self):
        """Test that all structured thinking memory types are covered in schema"""
        
        from clarity.utils.schema import STRUCTURED_THINKING_MEMORY_TYPES
        
        # Verify all expected memory types are defined
        expected_types = [
            "structured_thinking",
            "thought_process", 
            "thinking_relationship",
            "problem_analysis",
            "research_notes",
            "analysis_result",
            "solution_synthesis",
            "conclusion_summary"
        ]
        
        for memory_type in expected_types:
            assert memory_type in STRUCTURED_THINKING_MEMORY_TYPES
            
            config = STRUCTURED_THINKING_MEMORY_TYPES[memory_type]
            assert "description" in config
            assert "tier" in config
            assert "retention_days" in config
            assert "requires_embedding" in config
    
    def test_memory_type_configurations(self):
        """Test memory type configuration values"""
        
        from clarity.utils.schema import STRUCTURED_THINKING_MEMORY_TYPES
        
        # Core thinking types should have long retention
        core_types = ["structured_thinking", "problem_analysis", "analysis_result", 
                     "solution_synthesis", "conclusion_summary"]
        
        for memory_type in core_types:
            config = STRUCTURED_THINKING_MEMORY_TYPES[memory_type]
            assert config["tier"] == "core"
            assert config["retention_days"] >= 365
            assert config["requires_embedding"] is True
        
        # Relationships should be supplementary with shorter retention
        rel_config = STRUCTURED_THINKING_MEMORY_TYPES["thinking_relationship"]
        assert rel_config["tier"] == "supplementary"
        assert rel_config["retention_days"] == 90
        assert rel_config["requires_embedding"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])