#!/usr/bin/env python3
"""
Ultrathink Auto-Enablement Demo

This demo shows how the ultrathink integration automatically detects
structured thinking patterns and enhances prompts to enable Claude's
extended thinking capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from clarity.mcp.hook_integration import MCPHookIntegration
from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.mcp.ultrathink_config import UltrathinkConfig


class DemoMCP:
    """Demo MCP integration for ultrathink testing."""
    
    def __init__(self):
        """Initialize demo MCP."""
        self.setup_complete = False
    
    async def setup(self):
        """Set up the demo environment."""
        print("🚀 Setting up Ultrathink Demo Environment...")
        
        # Create mock domain manager
        class MockDomainManager:
            async def store_memory(self, **kwargs):
                print(f"   📝 Stored memory: {kwargs.get('memory_type', 'unknown')}")
                return "demo_memory_id"
            
            async def retrieve_memories(self, **kwargs):
                return []
        
        # Initialize components
        domain_manager = MockDomainManager()
        tool_indexer = MCPToolIndexer(domain_manager)
        self.hook_integration = MCPHookIntegration(tool_indexer)
        
        print("✅ Demo environment ready!")
        self.setup_complete = True
    
    async def demo_structured_thinking_detection(self):
        """Demonstrate structured thinking pattern detection."""
        print("\\n🧠 DEMO: Structured Thinking Detection")
        print("=" * 50)
        
        test_prompts = [
            # Should trigger ultrathink
            "I need a step-by-step approach to solve this complex problem",
            "Can you help me analyze this situation systematically?",
            "Let's break down this challenge using structured thinking",
            "I need a comprehensive analysis of the trade-offs",
            "How should we approach this multi-faceted issue?",
            
            # Should NOT trigger ultrathink
            "What's the current time?",
            "Quick question about Python syntax",
            "Just need a brief summary",
            "Simple file operation needed"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\\n📝 Test {i}: {prompt}")
            
            enhanced_prompt = await self.hook_integration._enhance_prompt_with_ultrathink(prompt)
            
            if enhanced_prompt != prompt:
                print("   ✅ ULTRATHINK ENABLED")
                directive_start = enhanced_prompt.find("🧠")
                if directive_start != -1:
                    print(f"   📋 Added directive: {enhanced_prompt[directive_start:directive_start+100]}...")
            else:
                print("   ⏭️  Standard processing")
    
    async def demo_configuration_options(self):
        """Demonstrate configuration customization."""
        print("\\n⚙️  DEMO: Configuration Options")
        print("=" * 50)
        
        # Show current config
        config = self.hook_integration.ultrathink_config_manager
        print("\\n📊 Current Configuration:")
        print(f"   Enabled: {config.is_enabled()}")
        print(f"   Minimum Confidence: {config.get_minimum_confidence()}")
        print(f"   Custom Patterns: {len(config.get_custom_patterns())}")
        print(f"   Exclusion Patterns: {len(config.get_exclusion_patterns())}")
        
        # Test custom pattern
        print("\\n🔧 Adding custom pattern...")
        config.add_custom_pattern(r"\\bdemo\\s+pattern\\b", 'high_confidence')
        
        # Update hook integration
        self.hook_integration.configure_ultrathink({})
        
        # Test the custom pattern
        test_prompt = "This contains a demo pattern for testing"
        enhanced = await self.hook_integration._enhance_prompt_with_ultrathink(test_prompt)
        
        if enhanced != test_prompt:
            print("   ✅ Custom pattern detected and triggered ultrathink!")
        else:
            print("   ❌ Custom pattern not detected")
        
        # Test exclusion pattern
        print("\\n🚫 Testing exclusion pattern...")
        exclusion_prompt = "I need structured thinking but this is a quick question"
        enhanced = await self.hook_integration._enhance_prompt_with_ultrathink(exclusion_prompt)
        
        if enhanced == exclusion_prompt:
            print("   ✅ Exclusion pattern prevented ultrathink (as expected)")
        else:
            print("   ❌ Exclusion pattern didn't work")
    
    async def demo_confidence_scoring(self):
        """Demonstrate confidence scoring system."""
        print("\\n📈 DEMO: Confidence Scoring")
        print("=" * 50)
        
        test_cases = [
            ("I need help", "Low confidence - generic request"),
            ("I need to analyze this", "Medium confidence - analysis keyword"),
            ("I need a systematic comprehensive analysis", "High confidence - multiple strong keywords"),
            ("This requires a structured step-by-step methodical approach", "Very high confidence - many strong keywords")
        ]
        
        for prompt, description in test_cases:
            # Temporarily lower threshold to see all scores
            original_threshold = self.hook_integration.ultrathink_config_manager.get_minimum_confidence()
            self.hook_integration.configure_ultrathink({'minimum_confidence': 0.0})
            
            enhanced = await self.hook_integration._enhance_prompt_with_ultrathink(prompt)
            
            print(f"\\n📝 {description}")
            print(f"   Prompt: {prompt}")
            
            if enhanced != prompt:
                print("   ✅ Would trigger ultrathink")
            else:
                print("   ⏭️  Below threshold")
            
            # Restore original threshold
            self.hook_integration.configure_ultrathink({'minimum_confidence': original_threshold})
    
    async def demo_real_world_examples(self):
        """Demonstrate with real-world example prompts."""
        print("\\n🌍 DEMO: Real-World Examples")
        print("=" * 50)
        
        real_world_prompts = [
            "I'm designing a new microservices architecture and need to evaluate different patterns. Can you help me think through the trade-offs systematically?",
            
            "Our team is facing a complex decision about whether to migrate our monolith to microservices. What factors should we analyze?",
            
            "I need to troubleshoot a performance issue that involves multiple systems. How should I approach this methodically?",
            
            "We're planning a major refactoring of our codebase. Can you help me break down the planning process step by step?",
            
            "Just checking the syntax for async/await in Python",
            
            "Quick question: what's the difference between let and const in JavaScript?"
        ]
        
        for i, prompt in enumerate(real_world_prompts, 1):
            print(f"\\n🎯 Example {i}:")
            print(f"   {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            
            enhanced = await self.hook_integration._enhance_prompt_with_ultrathink(prompt)
            
            if enhanced != prompt:
                print("   ✅ ULTRATHINK ENABLED - Complex reasoning detected")
            else:
                print("   ⏭️  STANDARD MODE - Simple query")
    
    async def demo_statistics(self):
        """Show ultrathink usage statistics."""
        print("\\n📊 DEMO: Usage Statistics")
        print("=" * 50)
        
        stats = self.hook_integration.get_ultrathink_stats()
        
        print("\\n📈 Enhancement Statistics:")
        print(f"   Total Enhancements: {stats['total_enhancements']}")
        print(f"   Average Confidence: {stats['average_confidence']:.2f}")
        print(f"   Enhancement Rate: {stats['enhancement_rate']:.1%}")
        
        print("\\n🔍 Pattern Statistics:")
        print(f"   Built-in Patterns: {stats['pattern_counts']['built_in_patterns']}")
        print(f"   Custom Patterns: {stats['pattern_counts']['custom_patterns']}")
        print(f"   Total Active Patterns: {stats['pattern_counts']['total_patterns']}")
        
        if stats['most_common_patterns']:
            print("\\n🏆 Most Common Triggered Patterns:")
            for pattern, count in stats['most_common_patterns'][:3]:
                print(f"   {pattern}: {count} times")
    
    async def run_demo(self):
        """Run the complete demo."""
        print("🎭 ULTRATHINK AUTO-ENABLEMENT DEMO")
        print("=" * 60)
        
        if not self.setup_complete:
            await self.setup()
        
        try:
            await self.demo_structured_thinking_detection()
            await self.demo_configuration_options()
            await self.demo_confidence_scoring()
            await self.demo_real_world_examples()
            await self.demo_statistics()
            
            print("\\n🎉 Demo Complete!")
            print("\\n💡 Key Takeaways:")
            print("   • Ultrathink automatically enables for complex reasoning tasks")
            print("   • Pattern detection is configurable and extensible")
            print("   • Confidence scoring prevents false positives")
            print("   • Exclusion patterns handle edge cases")
            print("   • Usage statistics help optimize the system")
            
        except Exception as e:
            print(f"\\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demo entry point."""
    demo = DemoMCP()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())