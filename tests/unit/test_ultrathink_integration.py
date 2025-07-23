"""
Test suite for Ultrathink Auto-Enablement Integration.

This module tests the automatic ultrathink enablement feature when
structured thinking patterns are detected in user prompts.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock
from pathlib import Path

# Add project root to Python path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from clarity.mcp.hook_integration import MCPHookIntegration
from clarity.mcp.ultrathink_config import UltrathinkConfig
from clarity.mcp.tool_indexer import MCPToolIndexer


class TestUltrathinkIntegration:
    """Test ultrathink auto-enablement functionality."""
    
    @pytest.fixture
    async def mock_domain_manager(self):
        """Create a mock domain manager."""
        manager = Mock()
        manager.store_memory = AsyncMock(return_value="test_memory_id")
        manager.retrieve_memories = AsyncMock(return_value=[])
        return manager
    
    @pytest.fixture
    async def mock_tool_indexer(self, mock_domain_manager):
        """Create a mock tool indexer."""
        indexer = Mock(spec=MCPToolIndexer)
        indexer.domain_manager = mock_domain_manager
        indexer.indexed_tools = {}
        indexer.discovered_servers = {}
        return indexer
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "enabled": True,
                "minimum_confidence": 0.5,  # Lower threshold for testing
                "ultrathink_directive": "\\n\\n🧠 **TEST MODE**: Ultrathink enabled for testing.",
                "custom_patterns": [
                    {
                        "pattern": r"\\btest\\s+pattern\\b",
                        "weight_category": "high_confidence"
                    }
                ],
                "exclusion_patterns": [
                    r"\\bquick\\s+test\\b"
                ]
            }
            json.dump(test_config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    async def hook_integration(self, mock_tool_indexer, temp_config_file):
        """Create hook integration with test configuration."""
        # Override config path
        integration = MCPHookIntegration(mock_tool_indexer)
        integration.ultrathink_config_manager = UltrathinkConfig(temp_config_file)
        integration.ultrathink_config = integration.ultrathink_config_manager.get_config()
        
        # Update patterns
        integration.all_thinking_patterns = integration.structured_thinking_patterns.copy()
        custom_patterns = integration.ultrathink_config_manager.get_custom_patterns()
        for custom_pattern in custom_patterns:
            if isinstance(custom_pattern, dict) and 'pattern' in custom_pattern:
                integration.all_thinking_patterns.append(custom_pattern['pattern'])
        
        return integration
    
    @pytest.mark.asyncio
    async def test_structured_thinking_detection(self, hook_integration):
        """Test detection of structured thinking patterns."""
        test_prompts = [
            ("I need a step by step approach to solve this problem", True),
            ("Can you break down the problem systematically?", True),
            ("Let's think through this comprehensive analysis", True),
            ("I need a structured thinking process for this", True),
            ("Help me analyze this complex situation", True),
            ("Just a quick question about syntax", False),
            ("Simple file read operation needed", False),
            ("What's the weather like?", False)
        ]
        
        for prompt, should_trigger in test_prompts:
            enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
            
            if should_trigger:
                assert enhanced_prompt != prompt, f"Expected ultrathink for: {prompt}"
                assert "🧠 **TEST MODE**" in enhanced_prompt, f"Missing ultrathink directive in: {enhanced_prompt}"
            else:
                assert enhanced_prompt == prompt, f"Unexpected ultrathink for: {prompt}"
    
    @pytest.mark.asyncio
    async def test_custom_pattern_detection(self, hook_integration):
        """Test detection of custom patterns."""
        prompt = "I need to test pattern recognition functionality"
        enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        assert enhanced_prompt != prompt
        assert "🧠 **TEST MODE**" in enhanced_prompt
    
    @pytest.mark.asyncio
    async def test_exclusion_patterns(self, hook_integration):
        """Test that exclusion patterns prevent ultrathink enablement."""
        prompt = "This is a structured approach but also a quick test"
        enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        # Should be excluded despite containing structured thinking patterns
        assert enhanced_prompt == prompt
    
    @pytest.mark.asyncio
    async def test_confidence_threshold(self, hook_integration):
        """Test confidence threshold functionality."""
        # Update config to higher threshold
        hook_integration.configure_ultrathink({'minimum_confidence': 0.9})
        
        # Low confidence prompt
        prompt = "I need to consider this"
        enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        # Should not trigger due to high threshold
        assert enhanced_prompt == prompt
        
        # High confidence prompt
        prompt = "I need a comprehensive systematic structured analysis"
        enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        # Should trigger even with high threshold
        assert enhanced_prompt != prompt
        assert "🧠 **TEST MODE**" in enhanced_prompt
    
    @pytest.mark.asyncio
    async def test_disabled_ultrathink(self, hook_integration):
        """Test that disabled ultrathink doesn't enhance prompts."""
        hook_integration.configure_ultrathink({'enabled': False})
        
        prompt = "I need a step by step structured approach"
        enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        assert enhanced_prompt == prompt
    
    @pytest.mark.asyncio
    async def test_prompt_submission_integration(self, hook_integration):
        """Test integration with prompt submission workflow."""
        test_data = {
            'prompt': 'I need a comprehensive analysis of this complex problem'
        }
        
        result = await hook_integration.analyze_tool_usage('prompt_submit', test_data)
        
        assert result is not None
        assert "🧠 **TEST MODE**" in result
        
        # Verify logging occurred
        hook_integration.tool_indexer.domain_manager.store_memory.assert_called()
        call_args = hook_integration.tool_indexer.domain_manager.store_memory.call_args
        assert call_args[1]['memory_type'] == 'ultrathink_enhancement'
    
    @pytest.mark.asyncio
    async def test_ultrathink_stats(self, hook_integration):
        """Test ultrathink statistics collection."""
        # Trigger some enhancements
        prompts = [
            "I need a structured approach",
            "Complex analysis required",
            "Step by step solution needed"
        ]
        
        for prompt in prompts:
            await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        stats = hook_integration.get_ultrathink_stats()
        
        assert stats['total_enhancements'] == 3
        assert stats['average_confidence'] > 0
        assert len(stats['most_common_patterns']) > 0
        assert stats['enhancement_rate'] > 0
        assert 'config_stats' in stats
        assert 'pattern_counts' in stats
    
    @pytest.mark.asyncio
    async def test_configuration_updates(self, hook_integration):
        """Test configuration updates."""
        # Test directive update
        new_directive = "\\n\\nCustom ultrathink directive for testing"
        hook_integration.configure_ultrathink({
            'ultrathink_directive': new_directive
        })
        
        prompt = "I need structured thinking"
        enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        assert new_directive in enhanced_prompt
        
        # Test custom pattern addition
        hook_integration.ultrathink_config_manager.add_custom_pattern(
            r"\\bcustom\\s+test\\s+pattern\\b", 
            'high_confidence'
        )
        
        # Reload patterns
        hook_integration.configure_ultrathink({})
        
        prompt = "This is a custom test pattern example"
        enhanced_prompt = await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        assert enhanced_prompt != prompt
    
    def test_ultrathink_config_creation(self, temp_config_file):
        """Test ultrathink configuration loading and saving."""
        config = UltrathinkConfig(temp_config_file)
        
        assert config.is_enabled()
        assert config.get_minimum_confidence() == 0.5
        assert "TEST MODE" in config.get_ultrathink_directive()
        assert len(config.get_custom_patterns()) == 1
        assert len(config.get_exclusion_patterns()) == 1
    
    def test_pattern_management(self, temp_config_file):
        """Test custom pattern management."""
        config = UltrathinkConfig(temp_config_file)
        
        # Add pattern
        config.add_custom_pattern(r"\\bnew\\s+pattern\\b", 'medium_confidence')
        
        patterns = config.get_custom_patterns()
        assert len(patterns) == 2
        
        # Remove pattern
        success = config.remove_custom_pattern(r"\\btest\\s+pattern\\b")
        assert success
        
        patterns = config.get_custom_patterns()
        assert len(patterns) == 1
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, hook_integration):
        """Test edge cases and error handling."""
        # Empty prompt
        enhanced = await hook_integration._enhance_prompt_with_ultrathink("")
        assert enhanced == ""
        
        # Very long prompt
        long_prompt = "I need structured thinking " * 100
        enhanced = await hook_integration._enhance_prompt_with_ultrathink(long_prompt)
        assert enhanced != long_prompt
        
        # Prompt with special characters
        special_prompt = "I need a step-by-step approach with ñ and 中文 characters"
        enhanced = await hook_integration._enhance_prompt_with_ultrathink(special_prompt)
        assert enhanced != special_prompt
    
    @pytest.mark.asyncio
    async def test_logging_integration(self, hook_integration):
        """Test logging of ultrathink enhancements."""
        prompt = "I need comprehensive analysis"
        await hook_integration._enhance_prompt_with_ultrathink(prompt)
        
        # Check that memory was stored
        hook_integration.tool_indexer.domain_manager.store_memory.assert_called()
        
        # Verify the logged data structure
        call_args = hook_integration.tool_indexer.domain_manager.store_memory.call_args
        stored_content = json.loads(call_args[1]['content'])
        
        assert stored_content['type'] == 'ultrathink_enhancement'
        assert 'confidence_score' in stored_content
        assert 'matched_patterns' in stored_content
        assert stored_content['enhancement_applied'] is True


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])