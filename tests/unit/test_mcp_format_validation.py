"""
Unit tests for MCP response format validation.

These tests specifically validate the fixes for the critical MCP format issues:
1. Temporal domain configuration access errors
2. Division by zero in recency calculations  
3. Field naming mismatches in MCP validation

These are isolated unit tests that can run quickly to catch regressions.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from clarity.domains.temporal import TemporalDomain
from clarity.domains.persistence import PersistenceDomain
from tests.framework.mcp_validation import MCPProtocolValidator, MCPServerTestSuite


class TestMCPFormatValidation:
    """Unit tests for MCP format validation fixes."""

    def test_temporal_domain_config_access(self):
        """
        Test that temporal domain handles missing retrieval configuration.
        
        Validates fix for: KeyError: 'retrieval' in temporal.py:140
        """
        # Test with missing retrieval config
        config_without_retrieval = {
            "alunai-clarity": {
                "consolidation_interval_hours": 24
            }
            # Note: Missing 'retrieval' section
        }
        
        mock_persistence = MagicMock(spec=PersistenceDomain)
        temporal_domain = TemporalDomain(config_without_retrieval, mock_persistence)
        
        # Create test memory data
        test_memories = [
            {
                "last_accessed": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(), 
                "importance": 0.8,
                "similarity": 0.7
            }
        ]
        
        # This should not raise KeyError anymore
        result = asyncio.run(temporal_domain.adjust_memory_relevance(
            memories=test_memories,
            query="test query"
        ))
        
        assert isinstance(result, list), "Should return list of memories"
        assert len(result) == 1, "Should return one memory"
        assert "adjusted_score" in result[0], "Should have adjusted score"
        assert "recency_score" in result[0], "Should have recency score"

    def test_temporal_domain_recency_calculation(self):
        """
        Test that recency calculation handles same-day memories without division by zero.
        
        Validates fix for: ZeroDivisionError in temporal.py:155
        """
        config = {
            "retrieval": {
                "recency_weight": 0.3,
                "importance_weight": 0.7
            },
            "alunai-clarity": {
                "consolidation_interval_hours": 24
            }
        }
        
        mock_persistence = MagicMock(spec=PersistenceDomain)
        temporal_domain = TemporalDomain(config, mock_persistence)
        
        # Create memory with same-day access (should trigger division by zero without fix)
        now = datetime.now()
        test_memories = [
            {
                "last_accessed": now.isoformat(),
                "created_at": now.isoformat(),
                "importance": 0.5,
                "similarity": 0.6
            }
        ]
        
        # This should not raise ZeroDivisionError
        result = asyncio.run(temporal_domain.adjust_memory_relevance(
            memories=test_memories,
            query="test query"
        ))
        
        assert isinstance(result, list), "Should return list of memories"
        memory = result[0]
        assert "recency_score" in memory, "Should have recency score"
        assert memory["recency_score"] > 0, "Recency score should be positive"
        assert memory["recency_score"] <= 1, "Recency score should not exceed 1"

    def test_mcp_protocol_validator_field_flexibility(self):
        """
        Test that MCP validator accepts both old and new field naming conventions.
        
        Validates fix for field naming mismatch in mcp_validation.py:330-343
        """
        validator = MCPProtocolValidator()
        
        # Test response with old field names
        old_format_response = json.dumps({
            "success": True,
            "memories": [
                {
                    "id": "test_id_1",
                    "content": "Test content 1", 
                    "type": "test_memory",  # Old field name
                    "similarity": 0.85      # Old field name
                },
                {
                    "id": "test_id_2", 
                    "content": "Test content 2",
                    "type": "test_memory",
                    "similarity": 0.75
                }
            ]
        })
        
        # Test response with new field names  
        new_format_response = json.dumps({
            "success": True,
            "memories": [
                {
                    "id": "test_id_3",
                    "content": "Test content 3",
                    "memory_type": "test_memory",    # New field name
                    "similarity_score": 0.92        # New field name
                },
                {
                    "id": "test_id_4",
                    "content": "Test content 4", 
                    "memory_type": "test_memory",
                    "similarity_score": 0.68
                }
            ]
        })
        
        # Test mixed field names
        mixed_format_response = json.dumps({
            "success": True,
            "memories": [
                {
                    "id": "test_id_5",
                    "content": "Test content 5",
                    "type": "test_memory",           # Old
                    "similarity_score": 0.88        # New
                },
                {
                    "id": "test_id_6",
                    "content": "Test content 6",
                    "memory_type": "test_memory",    # New  
                    "similarity": 0.73              # Old
                }
            ]
        })
        
        # All formats should validate successfully
        old_validation = validator.validate_response_format("retrieve_memory", old_format_response)
        new_validation = validator.validate_response_format("retrieve_memory", new_format_response) 
        mixed_validation = validator.validate_response_format("retrieve_memory", mixed_format_response)
        
        assert old_validation["valid_json"], "Old format should have valid JSON"
        assert old_validation["schema_compliant"], f"Old format should be schema compliant: {old_validation['errors']}"
        assert len(old_validation["errors"]) == 0, f"Old format should have no errors: {old_validation['errors']}"
        
        assert new_validation["valid_json"], "New format should have valid JSON"
        assert new_validation["schema_compliant"], f"New format should be schema compliant: {new_validation['errors']}"
        assert len(new_validation["errors"]) == 0, f"New format should have no errors: {new_validation['errors']}"
        
        assert mixed_validation["valid_json"], "Mixed format should have valid JSON" 
        assert mixed_validation["schema_compliant"], f"Mixed format should be schema compliant: {mixed_validation['errors']}"
        assert len(mixed_validation["errors"]) == 0, f"Mixed format should have no errors: {mixed_validation['errors']}"

    def test_mcp_protocol_validator_error_responses(self):
        """Test that error responses are properly validated."""
        validator = MCPProtocolValidator()
        
        # Valid error response
        error_response = json.dumps({
            "success": False,
            "error": "Memory not found"
        })
        
        validation = validator.validate_error_response(error_response)
        
        assert validation["valid_json"], "Error response should have valid JSON"
        assert validation["schema_compliant"], f"Error response should be schema compliant: {validation['errors']}"
        assert len(validation["errors"]) == 0, f"Error response should have no errors: {validation['errors']}"

    def test_mcp_protocol_validator_missing_fields(self):
        """Test validation when required fields are missing."""
        validator = MCPProtocolValidator()
        
        # Response missing top-level required fields
        incomplete_response = json.dumps({
            "memories": [
                {
                    "id": "test_id",
                    "content": "test content",
                    "type": "test"
                }
            ]
            # Missing: success field
        })
        
        validation = validator.validate_response_format("retrieve_memory", incomplete_response)
        
        assert validation["valid_json"], "Should have valid JSON"
        assert not validation["schema_compliant"], "Should not be schema compliant due to missing fields"
        assert len(validation["errors"]) > 0, "Should have validation errors for missing fields"

    @pytest.mark.asyncio
    async def test_end_to_end_format_validation(self):
        """
        End-to-end test to ensure format validation works in real scenarios.
        
        This test simulates the exact conditions that were failing before the fixes.
        """
        suite = MCPServerTestSuite()
        await suite.setup_test_environment()
        
        try:
            # Store a test memory
            store_result = await suite.validate_mcp_tool_execution(
                tool_name="store_memory",
                arguments={
                    "memory_type": "format_validation_test",
                    "content": "This memory tests end-to-end format validation",
                    "importance": 0.7,
                    "metadata": {"test_type": "format_validation"}
                },
                test_name="format_validation_store"
            )
            
            assert store_result.passed, f"Store operation failed: {store_result.errors}"
            memory_id = store_result.parsed_response["memory_id"]
            
            # Retrieve the memory to trigger all the fixed code paths
            retrieve_result = await suite.validate_mcp_tool_execution(
                tool_name="retrieve_memory",
                arguments={
                    "query": "format validation test",
                    "limit": 3,
                    "min_similarity": 0.3,
                    "include_metadata": True
                },
                test_name="format_validation_retrieve"
            )
            
            assert retrieve_result.passed, f"Retrieve operation failed: {retrieve_result.errors}"
            
            # Validate response structure
            assert retrieve_result.response_validation["valid_json"], "Response should be valid JSON"
            assert retrieve_result.response_validation["schema_compliant"], "Response should be schema compliant"
            
            memories = retrieve_result.parsed_response.get("memories", [])
            assert len(memories) > 0, "Should retrieve at least one memory"
            
            # Validate the retrieved memory has the expected structure
            memory = memories[0]
            assert "id" in memory, "Memory should have 'id' field"
            assert "content" in memory, "Memory should have 'content' field"
            
            # Check that either field naming convention works
            has_type = "memory_type" in memory or "type" in memory
            has_similarity = "similarity_score" in memory or "similarity" in memory
            
            assert has_type, "Memory should have type field (either 'memory_type' or 'type')"
            assert has_similarity, "Memory should have similarity field (either 'similarity_score' or 'similarity')"
            
        finally:
            await suite.teardown_test_environment()


if __name__ == "__main__":
    # Allow running directly for debugging
    import asyncio
    
    async def run_format_tests():
        """Run format validation tests directly."""
        test_suite = TestMCPFormatValidation()
        
        print("🧪 Running MCP format validation tests...")
        
        print("\n1️⃣ Testing temporal domain config access...")
        test_suite.test_temporal_domain_config_access()
        print("✅ Temporal domain config access test passed")
        
        print("\n2️⃣ Testing temporal domain recency calculation...")
        test_suite.test_temporal_domain_recency_calculation()
        print("✅ Temporal domain recency calculation test passed")
        
        print("\n3️⃣ Testing MCP protocol validator field flexibility...")
        test_suite.test_mcp_protocol_validator_field_flexibility()
        print("✅ MCP protocol validator flexibility test passed")
        
        print("\n4️⃣ Testing MCP protocol validator error responses...")
        test_suite.test_mcp_protocol_validator_error_responses()
        print("✅ MCP protocol validator error response test passed")
        
        print("\n5️⃣ Testing MCP protocol validator missing fields...")
        test_suite.test_mcp_protocol_validator_missing_fields()
        print("✅ MCP protocol validator missing fields test passed")
        
        print("\n6️⃣ Testing end-to-end format validation...")
        await test_suite.test_end_to_end_format_validation()
        print("✅ End-to-end format validation test passed")
        
        print("\n🎉 All MCP format validation tests passed!")
    
    asyncio.run(run_format_tests())