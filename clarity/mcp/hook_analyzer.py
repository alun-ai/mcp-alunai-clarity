#!/usr/bin/env python3
"""CLI script called by Claude Code hooks for MCP analysis.

This script is invoked by Claude Code's hook system to analyze tool usage
and provide MCP learning and suggestions.
"""

import asyncio
import argparse
import json
import sys
import logging
import os
from pathlib import Path
from datetime import datetime

# Add the project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from clarity.mcp.hook_integration import MCPHookIntegration
    from clarity.mcp.tool_indexer import MCPToolIndexer
    from clarity.auto_memory.auto_capture import should_store_memory, extract_memory_content
    # Mock domain manager for standalone hook testing
    class MockDomainManager:
        async def store_memory(self, **kwargs):
            return "mock_id"
        async def retrieve_memories(self, **kwargs):
            return []
except ImportError as e:
    # Fallback for when running in isolation
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(0)  # Silent exit to avoid cluttering Claude Code output

# Configure minimal logging for hook context
logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger(__name__)


class HookAnalyzerCLI:
    """CLI interface for hook-based MCP analysis."""
    
    def __init__(self):
        """Initialize the hook analyzer."""
        self.hook_integration = None
        self.manager = None
        self._initialized = False
    
    async def _initialize(self):
        """Initialize the clarity manager and hook integration."""
        if self._initialized:
            return
        
        try:
            # Initialize with mock domain manager for standalone testing
            mock_domain_manager = MockDomainManager()
            tool_indexer = MCPToolIndexer(mock_domain_manager)
            
            self.hook_integration = MCPHookIntegration(tool_indexer)
            self._initialized = True
                
        except Exception as e:
            logger.error(f"Failed to initialize hook analyzer: {e}")
            # Continue with limited functionality
    
    async def handle_pre_tool(self, tool_name: str, args: str):
        """Handle pre-tool execution analysis."""
        try:
            await self._initialize()
            
            if self.hook_integration:
                result = await self.hook_integration.analyze_tool_usage('pre_tool', {
                    'tool_name': tool_name,
                    'args': args
                })
                
                if result and result.get('suggestions'):
                    # Output suggestions in a format Claude Code can use
                    for suggestion in result['suggestions']:
                        print(f"MCP Suggestion: {suggestion.get('reason', '')}", file=sys.stderr)
        
        except Exception as e:
            logger.debug(f"Pre-tool analysis error: {e}")
            # Silent failure to avoid disrupting Claude Code
    
    async def handle_post_tool(self, tool_name: str, result: str, success: bool):
        """Handle post-tool execution learning."""
        try:
            await self._initialize()
            
            if self.hook_integration:
                await self.hook_integration.analyze_tool_usage('post_tool', {
                    'tool_name': tool_name,
                    'result': result,
                    'success': success
                })
        
        except Exception as e:
            logger.debug(f"Post-tool analysis error: {e}")
            # Silent failure
    
    async def handle_prompt_submit(self, prompt: str, additional_context: dict = None):
        """Handle prompt submission analysis and enhancement with enhanced Claude Code context."""
        try:
            await self._initialize()
            
            # Enhanced context available in Claude Code v1.0.59+
            session_id = additional_context.get('session_id') if additional_context else None
            transcript_path = additional_context.get('transcript_path') if additional_context else None
            cwd = additional_context.get('cwd') if additional_context else None
            
            # Log enhanced context for debugging
            if additional_context:
                logger.debug(f"Enhanced context: session_id={session_id}, cwd={cwd}")
            
            # Check for automatic memory storage and modify prompt if needed
            modified_prompt = await self._check_auto_memory_capture(prompt)
            if modified_prompt:
                prompt = modified_prompt
            
            # Enhanced hook integration with additional context
            if self.hook_integration:
                enhanced_prompt = await self.hook_integration.analyze_tool_usage('prompt_submit', {
                    'prompt': prompt,
                    'session_id': session_id,
                    'transcript_path': transcript_path,
                    'cwd': cwd,
                    'additional_context': additional_context
                })
                
                if enhanced_prompt:
                    # Output modified prompt with enhanced context for Claude Code
                    output = {
                        "modified_prompt": enhanced_prompt,
                        "additionalContext": {
                            "memory_enhanced": True,
                            "session_id": session_id,
                            "analysis_timestamp": datetime.now().isoformat()
                        }
                    }
                    print(json.dumps(output))
                    return
        
        except Exception as e:
            logger.debug(f"Prompt analysis error: {e}")
        
        # Return original prompt if analysis fails
        output = {
            "modified_prompt": prompt,
            "additionalContext": {
                "memory_enhanced": False,
                "error": str(e) if 'e' in locals() else None
            }
        }
        print(json.dumps(output))
    
    async def _check_auto_memory_capture(self, prompt: str):
        """Check if prompt should trigger automatic memory storage and modify prompt accordingly."""
        try:
            if should_store_memory(prompt):
                memory_type, content, importance = extract_memory_content(prompt)
                
                # Extract the content to store
                content_to_store = content.get("fact", content.get("message", str(content)))
                
                # Create modified prompt that includes store_memory call
                modified_prompt = f"""store_memory {content_to_store}

Original request: {prompt}"""
                
                logger.debug(f"Auto-triggered store_memory for: {memory_type}")
                return modified_prompt
                    
        except Exception as e:
            logger.debug(f"Auto-memory capture error: {e}")
        
        return None


async def main():
    """Main CLI entry point."""
    # Log hook execution for debugging
    import os
    try:
        # Try container log path first, fallback to host path
        if os.path.exists("/app/data"):
            log_file = "/app/data/hook_execution.log"
        else:
            log_file = "/Users/chadupton/Documents/Github/alun-ai/mcp-alunai-clarity/hook_execution.log"
        
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()}: Hook executed with args: {sys.argv}\n")
    except Exception:
        # Ignore logging errors to avoid breaking hook execution
        pass
    parser = argparse.ArgumentParser(
        description="MCP Hook Analyzer for Claude Code Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Event type flags
    parser.add_argument('--pre-tool', action='store_true',
                       help='Analyze tool usage before execution')
    parser.add_argument('--post-tool', action='store_true',
                       help='Learn from tool execution results')
    parser.add_argument('--prompt-submit', action='store_true',
                       help='Analyze and enhance user prompts')
    
    # Event data arguments
    parser.add_argument('--tool', type=str,
                       help='Tool name being used')
    parser.add_argument('--args', type=str, default='',
                       help='Tool arguments')
    parser.add_argument('--result', type=str, default='',
                       help='Tool execution result')
    parser.add_argument('--prompt', type=str, default='',
                       help='User prompt text')
    parser.add_argument('--success', type=str, default='true',
                       help='Whether tool execution was successful')
    
    # Enhanced Claude Code context (v1.0.59+)
    parser.add_argument('--session-id', type=str, default='',
                       help='Claude Code session identifier')
    parser.add_argument('--transcript-path', type=str, default='',
                       help='Path to conversation transcript')
    parser.add_argument('--cwd', type=str, default='',
                       help='Current working directory')
    parser.add_argument('--additional-context', type=str, default='',
                       help='Additional context data as JSON')
    
    # Utility flags
    parser.add_argument('--timeout', type=int, default=5,
                       help='Timeout for analysis in seconds')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Create analyzer instance
    analyzer = HookAnalyzerCLI()
    
    try:
        # Handle different event types with timeout
        if args.pre_tool and args.tool:
            await asyncio.wait_for(
                analyzer.handle_pre_tool(args.tool, args.args),
                timeout=args.timeout
            )
        
        elif args.post_tool and args.tool:
            success = args.success.lower() in ('true', '1', 'yes', 'success')
            await asyncio.wait_for(
                analyzer.handle_post_tool(args.tool, args.result, success),
                timeout=args.timeout
            )
        
        elif args.prompt_submit and args.prompt:
            # Parse additional context if provided
            additional_context = {}
            if args.session_id:
                additional_context['session_id'] = args.session_id
            if args.transcript_path:
                additional_context['transcript_path'] = args.transcript_path
            if args.cwd:
                additional_context['cwd'] = args.cwd
            if args.additional_context:
                try:
                    extra_context = json.loads(args.additional_context)
                    additional_context.update(extra_context)
                except json.JSONDecodeError:
                    logger.debug(f"Invalid additional context JSON: {args.additional_context}")
            
            await asyncio.wait_for(
                analyzer.handle_prompt_submit(args.prompt, additional_context or None),
                timeout=args.timeout
            )
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except asyncio.TimeoutError:
        logger.debug(f"Analysis timed out after {args.timeout}s")
        # Exit gracefully without disrupting Claude Code
        sys.exit(0)
    
    except KeyboardInterrupt:
        logger.debug("Analysis interrupted")
        sys.exit(0)
    
    except Exception as e:
        logger.debug(f"Analysis failed: {e}")
        sys.exit(0)  # Silent exit to avoid disrupting Claude Code


def cli_main():
    """Synchronous CLI wrapper for async main."""
    try:
        # Handle different Python versions and event loops
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're in an existing event loop, create a new thread
            import threading
            import concurrent.futures
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(main())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                future.result(timeout=10)  # 10 second timeout
        else:
            # Standard case - run in current thread
            asyncio.run(main())
    
    except Exception as e:
        logger.debug(f"CLI execution error: {e}")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()