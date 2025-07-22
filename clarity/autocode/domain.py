"""
AutoCode domain for code project intelligence and command learning.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from ..shared.async_utils import AsyncBatcher, async_timed, async_timer, parallel_map
from ..shared.logging import get_logger, log_operation, logged_operation
from ..shared.audit_trail import AuditEventType, AuditSeverity

from ..domains.persistence import PersistenceDomain
from .command_learner import CommandLearner
from .pattern_detector import PatternDetector
from .session_analyzer import SessionAnalyzer
from .history_navigator import HistoryNavigator
from .hook_manager import HookManager
from .structured_thinking_extension import StructuredThinkingExtension


class AutoCodeDomain:
    """
    Domain for code project intelligence and command learning.
    
    This domain handles:
    - Project pattern recognition and indexing
    - Command execution learning and suggestion
    - Session summary generation
    - Cross-project pattern application
    """
    
    def __init__(self, config: Dict[str, Any], persistence_domain: PersistenceDomain):
        """
        Initialize the AutoCode domain.
        
        Args:
            config: Configuration dictionary
            persistence_domain: Persistence domain for storage operations
        """
        self.config = config
        self.persistence_domain = persistence_domain
        self.pattern_cache = {}
        self.command_patterns = {}
        
        # Initialize comprehensive logging
        self.logger = get_logger(__name__)
        
        # AutoCode specific configuration
        self.autocode_config = config.get("autocode", {
            "enabled": True,
            "auto_scan_projects": True,
            "track_bash_commands": True,
            "generate_session_summaries": True
        })
        
        # Initialize command learner, pattern detector, session analyzer, and history navigator
        self.command_learner = None
        self.pattern_detector = None
        self.session_analyzer = None
        self.history_navigator = None
        self.hook_manager = None
        
        # Add structured thinking extension
        self.structured_thinking = StructuredThinkingExtension(persistence_domain)
        
        # Enhanced configuration for structured thinking
        self.enable_structured_thinking = config.get("autocode", {}).get("structured_thinking", {}).get("enabled", True)
        self.thinking_integration_level = config.get("autocode", {}).get("structured_thinking", {}).get("integration_level", "enhanced")
        
    @log_operation(
        operation_name="autocode_domain_initialization",
        actor="system",
        audit_event_type=AuditEventType.SYSTEM_START
    )
    async def initialize(self) -> None:
        """Initialize the AutoCode domain."""
        if not self.autocode_config.get("enabled", True):
            self.logger.info("AutoCode Domain disabled in configuration")
            return
        
        # Pattern detector will be initialized in set_command_learner method
        
        # Initialize session analyzer
        if self.autocode_config.get("session_analysis", {}).get("enabled", True):
            self.session_analyzer = SessionAnalyzer(self.autocode_config)
            self.logger.info("Session analyzer initialized")
        
        # Initialize command learner
        # Note: We need to pass the domain manager, but we don't have it yet
        # This will be set by the domain manager after initialization
        
        await self._load_existing_patterns()
        
        self.logger.info(
            f"AutoCode Domain initialized successfully - "
            f"session_analyzer: {self.session_analyzer is not None}, "
            f"pattern_cache: {len(self.pattern_cache)}, "
            f"command_patterns: {len(self.command_patterns)}"
        )
    
    async def set_command_learner(self, domain_manager):
        """Set the command learner with domain manager reference."""
        try:
            self.command_learner = CommandLearner(domain_manager)
            self.logger.info("AutoCode Domain: Command learner initialized")
            
            # Initialize pattern detector (needs domain manager)
            if self.autocode_config.get("pattern_detection", {}).get("enabled", True):
                self.pattern_detector = PatternDetector(domain_manager)
                self.logger.info("AutoCode Domain: Pattern detector initialized")
            
            # Initialize history navigator (needs domain manager)
            if self.autocode_config.get("history_navigation", {}).get("enabled", True):
                self.history_navigator = HistoryNavigator(domain_manager, self.autocode_config)
                self.logger.info("AutoCode Domain: History navigator initialized")
            
            # Initialize hook manager with MCP awareness (needs domain manager)
            if self.autocode_config.get("mcp_awareness", {}).get("enabled", True):
                self.hook_manager = HookManager(domain_manager, None)  # No autocode_hooks needed for MCP awareness
                await self.hook_manager.initialize()
                self.logger.info("AutoCode Domain: Hook manager with MCP awareness initialized")
                
        except (AttributeError, ImportError, ValueError, RuntimeError) as e:
            self.logger.error(f"AutoCode Domain: Error initializing command learner and navigation: {e}")
    
    async def process_file_access(
        self, 
        file_path: str, 
        content: str, 
        operation: str = "read"
    ) -> None:
        """
        Process file access to extract patterns.
        
        Args:
            file_path: Path to the accessed file
            content: File content (if available)
            operation: Type of operation (read, write, edit)
        """
        if not self.autocode_config.get("enabled", True):
            return
            
        try:
            # Extract patterns from file access
            patterns = await self._extract_file_patterns(file_path, content)
            
            # Store discovered patterns
            for pattern in patterns:
                await self._store_pattern(pattern)
                
            logger.debug(f"Processed file access: {file_path} ({operation})")
            
        except (OSError, ValueError, AttributeError, KeyError) as e:
            logger.error(f"Error processing file access {file_path}: {e}")
    
    async def process_bash_execution(
        self, 
        command: str, 
        exit_code: int, 
        output: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Process bash command execution for learning.
        
        Args:
            command: The bash command executed
            exit_code: Exit code (0 = success)
            output: Command output
            context: Execution context (project type, current directory, etc.)
        """
        if not self.autocode_config.get("track_bash_commands", True):
            return
            
        try:
            success = exit_code == 0
            
            # Store execution record
            execution_data = {
                "command": command,
                "exit_code": exit_code,
                "output": output[:1000] if output else "",  # Truncate long output
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
                "success": success
            }
            
            await self._store_bash_execution(execution_data)
            
            # Use command learner for advanced tracking if available
            if self.command_learner:
                await self.command_learner.track_bash_execution(command, exit_code, output, context)
            else:
                # Fallback to basic pattern tracking
                await self._update_command_patterns(command, success, context)
            
            logger.debug(f"Processed bash execution: {command} (exit_code: {exit_code})")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error processing bash execution {command}: {e}")
    
    async def generate_session_summary(self, conversation_log: List[Dict]) -> str:
        """
        Generate session summary from conversation log.
        
        Args:
            conversation_log: List of conversation messages
            
        Returns:
            Session summary ID
        """
        if not self.autocode_config.get("generate_session_summaries", True):
            return ""
            
        try:
            if self.session_analyzer:
                # Use advanced session analyzer - create session data for the generate_summary method
                session_data = {
                    "conversation_log": conversation_log,
                    "session_id": f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    "start_time": datetime.utcnow().isoformat(),
                    "duration_minutes": 0,  # Placeholder since we don't track real duration here
                    "file_access_log": [],  # Would be populated in a real session
                    "bash_execution_log": []  # Would be populated in a real session
                }
                summary = await self.session_analyzer.generate_summary(session_data)
            else:
                # Fallback to basic analysis
                summary = await self._analyze_session(conversation_log)
            
            return await self._store_session_summary(summary)
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error generating session summary: {e}")
            return ""
    
    async def suggest_command(
        self, 
        intent: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Enhanced command suggestion with optional structured thinking analysis.
        
        Args:
            intent: What the user wants to accomplish
            context: Current context (project type, platform, etc.)
            
        Returns:
            List of suggested commands with metadata, ranked by success probability
        """
        try:
            # Get base suggestions using existing logic
            base_suggestions = await self._get_base_command_suggestions(intent, context)
            
            # Apply structured thinking if enabled and context is complex
            if self.enable_structured_thinking and self._is_complex_context(intent, context):
                thinking_analysis = await self.structured_thinking.analyze_problem_with_stages(
                    problem=intent,
                    project_context=context
                )
                
                # Enhance suggestions with thinking insights
                enhanced_suggestions = []
                for suggestion in base_suggestions:
                    enhanced_suggestion = suggestion.copy()
                    enhanced_suggestion["thinking_analysis"] = {
                        "session_id": thinking_analysis["session_id"],
                        "confidence_boost": 0.1,  # Boost confidence due to structured analysis
                        "research_backing": len(thinking_analysis["research_findings"]),
                        "component_analysis": thinking_analysis["analysis_components"]
                    }
                    enhanced_suggestion["confidence"] = min(suggestion.get("confidence", 0.5) + 0.1, 1.0)
                    enhanced_suggestions.append(enhanced_suggestion)
                
                return enhanced_suggestions
            
            return base_suggestions
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            await self.logger.audit_error(
                f"Error suggesting commands for intent '{intent}': {e}",
                event_type=AuditEventType.ERROR_OCCURRED,
                actor="system",
                resource="command_suggestions",
                action="suggest_command",
                error=e,
                context={'intent': intent, 'context_keys': list(context.keys()) if context else []}
            )
            return []
    
    async def _get_base_command_suggestions(self, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get base command suggestions using existing logic."""
        if self.command_learner:
            # Use advanced command learner
            return await self.command_learner.suggest_command(intent, context)
        else:
            # Fallback to basic suggestions
            basic_suggestions = await self._get_command_suggestions(intent, context)
            # Convert to expected format
            return [{"command": cmd, "confidence": 0.5, "reasoning": "Basic suggestion"} 
                   for cmd in basic_suggestions]
    
    def _is_complex_context(self, intent: str, context: Dict[str, Any]) -> bool:
        """Determine if context warrants structured thinking analysis."""
        complexity_indicators = [
            len(intent.split()) > 5,  # Multi-word intent
            len(context) > 3,  # Rich context
            any(keyword in intent.lower() for keyword in ["implement", "design", "architecture", "solution"]),
            context.get("project_type") in ["enterprise", "complex", "multi-service"]
        ]
        
        return sum(complexity_indicators) >= 2
    
    @log_operation(
        operation_name="get_project_patterns",
        actor="system",
        audit_event_type=AuditEventType.PROJECT_ANALYSIS
    )
    async def get_project_patterns(self, project_path: str) -> Dict[str, Any]:
        """
        Enhanced project pattern detection with structured thinking analysis.
        
        Args:
            project_path: Path to the project
            
        Returns:
            Dictionary of known patterns for the project with structured thinking insights
        """
        try:
            async with self.logger.operation_context(
                "retrieve_project_patterns",
                actor="system",
                resource=project_path,
                audit_event_type=AuditEventType.PROJECT_ANALYSIS
            ):
                # Get base patterns using existing logic  
                base_patterns = await self._detect_base_patterns(project_path)
                
                if self.enable_structured_thinking:
                    # Apply structured thinking to pattern analysis
                    pattern_analysis = await self.structured_thinking.analyze_problem_with_stages(
                        problem=f"Analyze patterns in project {project_path}",
                        project_context={"project_path": project_path, "patterns": base_patterns}
                    )
                    
                    # Enhance patterns with thinking insights
                    enhanced_patterns = base_patterns.copy()
                    enhanced_patterns["structured_analysis"] = {
                        "thinking_session_id": pattern_analysis["session_id"],
                        "analysis_depth": "structured_thinking_applied",
                        "pattern_confidence": self._calculate_pattern_confidence(base_patterns, pattern_analysis)
                    }
                    
                    return enhanced_patterns
                
                return base_patterns
                
        except (OSError, ValueError, AttributeError, KeyError, PermissionError) as e:
            await self.logger.audit_error(
                f"Error retrieving project patterns for {project_path}: {e}",
                event_type=AuditEventType.ERROR_OCCURRED,
                actor="system",
                resource=project_path,
                action="get_project_patterns",
                error=e
            )
            return {}
    
    async def _detect_base_patterns(self, project_path: str) -> Dict[str, Any]:
        """Detect base patterns using existing logic."""
        # Validate project path exists
        import os
        from pathlib import Path
        
        self.logger.debug("Validating project path", context={
            'project_path': project_path,
            'path_exists_os': os.path.exists(project_path),
            'path_exists_pathlib': Path(project_path).exists()
        })
        
        if not os.path.exists(project_path) or not Path(project_path).exists():
            self.logger.error("Project path does not exist", context={
                'project_path': project_path
            })
            return {}
        
        # First try to get cached patterns
        self.logger.debug("Attempting to retrieve cached patterns")
        cached_patterns = await self._retrieve_project_patterns(project_path)
        
        # Validate cached patterns - ensure they don't contain error data
        if cached_patterns and isinstance(cached_patterns, dict):
            # Check if this is an error response that got cached
            if "error" in cached_patterns:
                self.logger.warning("Found corrupted cached patterns, clearing cache", context={
                    'project_path': project_path,
                    'error': cached_patterns.get('error')
                })
                cached_patterns = {}
        
        # If pattern detector is available and we don't have recent patterns, scan the project
        if self.pattern_detector and not cached_patterns:
            self.logger.info("Scanning project for new patterns")
            
            # Audit the pattern detection operation
            await self.logger.audit_info(
                "Starting pattern detection for project",
                event_type=AuditEventType.DATA_READ,
                actor="pattern_detector",
                resource=project_path,
                action="scan_project",
                context={'operation': 'pattern_detection'}
            )
            
            detected_patterns = await self.pattern_detector.scan_project(project_path)
            
            # Validate detected patterns before storing
            if detected_patterns and isinstance(detected_patterns, dict) and "error" not in detected_patterns:
                # Store detected patterns for future use
                await self._store_detected_patterns(project_path, detected_patterns)
                
                self.logger.info("Successfully detected and stored project patterns", context={
                    'project_path': project_path,
                    'pattern_count': len(detected_patterns),
                    'pattern_types': list(detected_patterns.keys())
                })
                
                return detected_patterns
            elif detected_patterns and "error" in detected_patterns:
                self.logger.error("Pattern detection failed", context={
                    'project_path': project_path,
                    'error': detected_patterns.get('error')
                })
                return {}
            else:
                self.logger.warning("Pattern detection returned invalid data", context={
                    'project_path': project_path
                })
                return {}
        
        return cached_patterns
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, Any], thinking_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in pattern detection based on structured analysis."""
        base_confidence = patterns.get("detection_confidence", 0.5)
        
        # Boost confidence based on research findings
        research_boost = min(len(thinking_analysis["research_findings"]) * 0.1, 0.3)
        
        # Boost based on analysis components
        analysis_boost = min(len(thinking_analysis["analysis_components"]["components"]) * 0.05, 0.2)
        
        return min(base_confidence + research_boost + analysis_boost, 1.0)
    
    async def find_similar_sessions(
        self, 
        query: str, 
        context: Dict[str, Any] = None,
        time_range_days: int = None
    ) -> List[Dict[str, Any]]:
        """
        Find sessions similar to the current query and context.
        
        Args:
            query: Search query describing current task or context
            context: Current context (project type, technologies, etc.)
            time_range_days: Limit search to recent days
            
        Returns:
            List of similar sessions with relevance scores
        """
        try:
            if self.history_navigator:
                return await self.history_navigator.find_similar_sessions(query, context, time_range_days)
            else:
                # Fallback to basic search
                return await self.domain_manager.retrieve_memories(
                    query=query,
                    memory_types=["session_summary"],
                    limit=5,
                    min_similarity=0.6
                )
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error finding similar sessions: {e}")
            return []
    
    async def get_context_for_continuation(
        self, 
        current_task: str,
        project_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get relevant context for continuing work on a task.
        
        Args:
            current_task: Description of current task
            project_context: Current project context
            
        Returns:
            Relevant context from previous sessions
        """
        try:
            if self.history_navigator:
                return await self.history_navigator.get_context_for_continuation(current_task, project_context)
            else:
                return {"error": "History navigator not available"}
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error getting continuation context: {e}")
            return {"error": str(e)}
    
    async def suggest_workflow_optimizations(
        self, 
        current_workflow: List[str],
        session_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest workflow optimizations based on historical data.
        
        Args:
            current_workflow: List of current workflow steps
            session_context: Current session context
            
        Returns:
            List of optimization suggestions
        """
        try:
            if self.history_navigator:
                return await self.history_navigator.suggest_workflow_optimizations(current_workflow, session_context)
            else:
                return []
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error suggesting workflow optimizations: {e}")
            return []
    
    async def get_learning_progression(
        self, 
        topic: str,
        time_range_days: int = 180
    ) -> Dict[str, Any]:
        """
        Track learning progression on a specific topic across sessions.
        
        Args:
            topic: Topic to track learning progression for
            time_range_days: Time range to analyze
            
        Returns:
            Learning progression data
        """
        try:
            if self.history_navigator:
                return await self.history_navigator.get_learning_progression(topic, time_range_days)
            else:
                return {"topic": topic, "error": "History navigator not available"}
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error getting learning progression for {topic}: {e}")
            return {"topic": topic, "error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get AutoCode domain statistics."""
        try:
            stats = {
                "enabled": self.autocode_config.get("enabled", True),
                "total_patterns": len(self.pattern_cache),
                "command_patterns": len(self.command_patterns),
                "last_updated": datetime.utcnow().isoformat(),
                "configuration": self.autocode_config,
                "components": {
                    "pattern_detector": self.pattern_detector is not None,
                    "session_analyzer": self.session_analyzer is not None,
                    "command_learner": self.command_learner is not None,
                    "history_navigator": self.history_navigator is not None
                }
            }
            
            # Add component-specific stats if available
            if self.history_navigator:
                stats["context_cache_size"] = len(getattr(self.history_navigator, 'context_cache', {}))
            
            return stats
        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Error getting AutoCode stats: {e}")
            return {"error": str(e)}
    
    # Private implementation methods
    @async_timed("pattern_loading")
    async def _load_existing_patterns(self) -> None:
        """Load existing patterns from storage with async optimization and timeout protection."""
        async with async_timer("pattern_loading"):
            try:
                # Skip pattern loading if persistence domain isn't fully ready to avoid initialization deadlock
                if not hasattr(self.persistence_domain, 'client') or self.persistence_domain.client is None:
                    logger.info("Skipping pattern loading during initialization - will load lazily on first use")
                    return
                
                # Load both pattern types concurrently
                async def load_project_patterns():
                    """Load project patterns from storage"""
                    try:
                        return await self.persistence_domain.search_memories(
                            embedding=None,
                            limit=100,
                            types=["project_pattern"],
                            min_similarity=0.0
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load project patterns: {e}")
                        return []
                
                async def load_command_patterns():
                    """Load command patterns from storage"""
                    try:
                        return await self.persistence_domain.search_memories(
                            embedding=None,
                            limit=100,
                            types=["command_pattern"],
                            min_similarity=0.0
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load command patterns: {e}")
                        return []
                
                # Execute both pattern loading operations concurrently
                project_patterns, command_patterns = await parallel_map(
                    lambda x: x(),
                    [load_project_patterns, load_command_patterns],
                    max_concurrency=2
                )
                
                # Process project patterns
                project_count = 0
                for pattern in project_patterns:
                    try:
                        pattern_id = pattern.get("id")
                        if pattern_id:
                            self.pattern_cache[pattern_id] = pattern
                            project_count += 1
                    except (AttributeError, KeyError, TypeError) as e:
                        logger.warning(f"Skipping invalid project pattern: {e}")
                
                # Process command patterns
                command_count = 0
                for pattern in command_patterns:
                    try:
                        command = pattern.get("content", {}).get("command")
                        if command:
                            self.command_patterns[command] = pattern
                            command_count += 1
                    except (AttributeError, KeyError, TypeError) as e:
                        self.logger.warning(f"Skipping invalid command pattern: {e}")
                        
                self.logger.info(f"Loaded {project_count} project patterns and {command_count} command patterns")
                
            except (ValueError, AttributeError, KeyError, RuntimeError) as e:
                self.logger.error(f"Error loading existing patterns: {e}")
    
    async def _extract_file_patterns(
        self, 
        file_path: str, 
        content: str
    ) -> List[Dict]:
        """Extract patterns from file content."""
        patterns = []
        
        try:
            # Use pattern detector if available for enhanced pattern extraction
            if self.pattern_detector:
                detected_patterns = await self.pattern_detector.analyze_file_content(file_path, content)
                patterns.extend(detected_patterns)
            else:
                # Fallback to basic pattern extraction
                # Basic file type pattern
                if file_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
                    # TypeScript/JavaScript patterns
                    if 'React' in content or 'jsx' in file_path:
                        patterns.append({
                            "type": "framework_usage",
                            "framework": "react",
                            "file_path": file_path,
                            "confidence": 0.8
                        })
                    
                    if 'component' in file_path.lower() or 'Component' in content:
                        patterns.append({
                            "type": "component_pattern",
                            "language": "typescript" if file_path.endswith(('.ts', '.tsx')) else "javascript",
                            "file_path": file_path,
                            "confidence": 0.7
                        })
                
                elif file_path.endswith('.py'):
                    # Python patterns
                    if 'django' in content.lower() or 'from django' in content:
                        patterns.append({
                            "type": "framework_usage",
                            "framework": "django",
                            "file_path": file_path,
                            "confidence": 0.8
                        })
                    
                    if 'flask' in content.lower() or 'from flask' in content:
                        patterns.append({
                            "type": "framework_usage",
                            "framework": "flask",
                            "file_path": file_path,
                            "confidence": 0.8
                        })
            
        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Error extracting patterns from {file_path}: {e}")
        
        return patterns
    
    async def _store_pattern(self, pattern: Dict) -> None:
        """Store a discovered pattern."""
        try:
            memory_id = f"mem_{str(uuid.uuid4())}"
            
            memory = {
                "id": memory_id,
                "type": "project_pattern",
                "content": {
                    "pattern_type": pattern.get("type", "unknown"),
                    "framework": pattern.get("framework", "unknown"),
                    "language": pattern.get("language", "unknown"),
                    "structure": pattern
                },
                "importance": pattern.get("confidence", 0.5),
                "metadata": {
                    "file_path": pattern.get("file_path"),
                    "detected_at": datetime.utcnow().isoformat()
                }
            }
            
            # Store in persistence layer
            await self.persistence_domain.store_memory(memory, "short_term")
            
            # Cache locally
            self.pattern_cache[memory_id] = memory
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error storing pattern: {e}")
    
    async def _store_bash_execution(self, execution_data: Dict) -> None:
        """Store bash execution record."""
        try:
            memory_id = f"mem_{str(uuid.uuid4())}"
            
            memory = {
                "id": memory_id,
                "type": "bash_execution",
                "content": execution_data,
                "importance": 0.4,  # Lower importance for individual executions
                "metadata": {
                    "category": "bash_execution",
                    "platform": execution_data.get("context", {}).get("platform", "unknown")
                }
            }
            
            # Store in persistence layer
            await self.persistence_domain.store_memory(memory, "short_term")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error storing bash execution: {e}")
    
    async def _update_command_patterns(
        self, 
        command: str, 
        success: bool, 
        context: Dict
    ) -> None:
        """Update command success patterns."""
        try:
            # Extract base command for pattern matching
            base_command = command.split()[0] if command.split() else command
            
            # Update or create command pattern
            if base_command in self.command_patterns:
                pattern = self.command_patterns[base_command]
                content = pattern.get("content", {})
                
                # Update success rate (simple averaging for now)
                current_rate = content.get("success_rate", 0.5)
                execution_count = content.get("execution_count", 1)
                new_rate = (current_rate * execution_count + (1.0 if success else 0.0)) / (execution_count + 1)
                
                content["success_rate"] = new_rate
                content["execution_count"] = execution_count + 1
                content["last_used"] = datetime.utcnow().isoformat()
                
            else:
                # Create new command pattern
                memory_id = f"mem_{str(uuid.uuid4())}"
                
                memory = {
                    "id": memory_id,
                    "type": "command_pattern",
                    "content": {
                        "command": base_command,
                        "context": context,
                        "success_rate": 1.0 if success else 0.0,
                        "platform": context.get("platform", "unknown"),
                        "execution_count": 1,
                        "last_used": datetime.utcnow().isoformat()
                    },
                    "importance": 0.6,
                    "metadata": {
                        "category": "command_pattern",
                        "created_at": datetime.utcnow().isoformat()
                    }
                }
                
                await self.persistence_domain.store_memory(memory, "short_term")
                self.command_patterns[base_command] = memory
                
        except (ValueError, AttributeError, KeyError, RuntimeError, TypeError) as e:
            logger.error(f"Error updating command patterns for {command}: {e}")
    
    async def _analyze_session(self, conversation_log: List[Dict]) -> Dict:
        """Analyze conversation log to generate summary."""
        try:
            # Basic session analysis (to be enhanced with SessionAnalyzer)
            session_data = {
                "session_id": f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "message_count": len(conversation_log),
                "tasks_completed": [],  # To be extracted from conversation
                "patterns_used": [],    # To be extracted from conversation
                "files_modified": []    # To be extracted from conversation
            }
            
            # Extract basic information from conversation
            for message in conversation_log:
                content = message.get("content", "")
                role = message.get("role", "")
                
                if role == "assistant":
                    # Look for file mentions
                    import re
                    file_matches = re.findall(r'[`"]([^`"]*\.[a-zA-Z0-9]+)[`"]', content)
                    session_data["files_modified"].extend(file_matches)
                    
                    # Look for task completion indicators
                    if any(indicator in content.lower() for indicator in ["completed", "implemented", "created", "fixed"]):
                        session_data["tasks_completed"].append({
                            "description": content[:100] + "...",  # Truncated description
                            "approach": "standard_implementation",
                            "outcome": "completed"
                        })
            
            return session_data
            
        except (ValueError, AttributeError, KeyError, TypeError, IndexError) as e:
            logger.error(f"Error analyzing session: {e}")
            return {}
    
    async def _store_session_summary(self, summary: Dict) -> str:
        """Store session summary."""
        try:
            memory_id = f"mem_{str(uuid.uuid4())}"
            
            # Create properly formatted memory dictionary for PersistenceDomain.store_memory
            memory = {
                "id": memory_id,
                "type": "session_summary",
                "content": summary,
                "importance": 0.8,  # High importance for session summaries
                "metadata": {
                    "category": "session_summary",
                    "session_id": summary.get("session_id"),
                    "created_at": datetime.utcnow().isoformat(),
                    "autocode_generated": True
                }
            }
            
            # Store using persistence domain's store_memory method
            stored_memory_id = await self.persistence_domain.store_memory(memory, "short_term")
            
            logger.info(f"Session summary stored successfully with ID: {stored_memory_id}")
            return stored_memory_id
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error storing session summary: {e}")
            return ""
    
    async def _get_command_suggestions(
        self, 
        intent: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Get command suggestions for intent."""
        try:
            suggestions = []
            
            # Simple intent-based suggestions (to be enhanced with CommandLearner)
            intent_lower = intent.lower()
            
            if "delete" in intent_lower or "remove" in intent_lower:
                if context.get("platform") == "posix":
                    suggestions = ["rm -f", "rm -rf"]
                else:
                    suggestions = ["del", "rmdir"]
            
            elif "install" in intent_lower:
                project_type = context.get("project_type", "")
                if "node" in project_type or "javascript" in project_type:
                    suggestions = ["npm install", "yarn install"]
                elif "python" in project_type:
                    suggestions = ["pip install", "poetry install"]
                elif "rust" in project_type:
                    suggestions = ["cargo install"]
            
            elif "build" in intent_lower:
                project_type = context.get("project_type", "")
                if "node" in project_type:
                    suggestions = ["npm run build", "yarn build"]
                elif "rust" in project_type:
                    suggestions = ["cargo build"]
                elif "java" in project_type:
                    suggestions = ["mvn compile", "gradle build"]
            
            return suggestions[:5]  # Limit to top 5 suggestions
            
        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Error getting command suggestions: {e}")
            return []
    
    async def _retrieve_project_patterns(self, project_path: str) -> Dict[str, Any]:
        """Retrieve known patterns for a project."""
        try:
            # Search for patterns related to this project
            patterns = await self.persistence_domain.search_memories(
                embedding=None,
                limit=50,
                types=["project_pattern"],
                min_similarity=0.0
            )
            
            project_patterns = {}
            for pattern in patterns:
                metadata = pattern.get("metadata", {})
                pattern_file_path = metadata.get("file_path", "")
                
                # Check if pattern is related to this project
                if project_path in pattern_file_path or any(
                    part in pattern_file_path for part in project_path.split("/")[-2:]
                ):
                    content = pattern.get("content", {})
                    pattern_type = content.get("pattern_type", "unknown")
                    
                    if pattern_type not in project_patterns:
                        project_patterns[pattern_type] = []
                    
                    project_patterns[pattern_type].append(content)
            
            return project_patterns
            
        except (ValueError, AttributeError, KeyError, RuntimeError, TypeError) as e:
            logger.error(f"Error retrieving project patterns: {e}")
            return {}
    
    async def _store_detected_patterns(self, project_path: str, patterns: Dict[str, Any]) -> None:
        """Store detected patterns for a project."""
        try:
            memory_id = f"mem_{str(uuid.uuid4())}"
            
            memory = {
                "id": memory_id,
                "type": "project_pattern",
                "content": {
                    "project_path": project_path,
                    "patterns": patterns,
                    "scan_timestamp": datetime.utcnow().isoformat(),
                    "pattern_version": "1.0"
                },
                "importance": 0.9,  # High importance for comprehensive project patterns
                "metadata": {
                    "category": "project_scan",
                    "project_root": project_path,
                    "framework": patterns.get("framework", {}).get("primary", "unknown"),
                    "language": patterns.get("language", {}).get("primary", "unknown"),
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            # Store in persistence layer
            await self.persistence_domain.store_memory(memory, "long_term")  # Use long_term for project patterns
            
            # Cache locally
            self.pattern_cache[memory_id] = memory
            
            logger.info(f"Stored detected patterns for project: {project_path}")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error storing detected patterns for {project_path}: {e}")
    
    # New methods specific to structured thinking integration
    
    async def create_thinking_session(self, title: str, description: str = None) -> str:
        """Create a new structured thinking session for complex problems."""
        from ..domains.structured_thinking import ThinkingSession
        
        session = ThinkingSession(
            title=title,
            description=description,
            project_context={"autocode_integration": True}
        )
        
        # Store initial session memory
        session_memory_id = await self.persistence_domain.store_memory(
            memory_type="structured_thinking",
            content=f"Started structured thinking session: {title}",
            importance=0.7,
            metadata={
                "session_id": session.id,
                "session_title": title,
                "thinking_session": True,
                "autocode_integration": True
            }
        )
        
        return session.id
    
    async def apply_structured_thinking_to_session_analysis(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply structured thinking to session analysis."""
        
        # Create thinking session for session analysis
        thinking_session_id = await self.create_thinking_session(
            title=f"Session Analysis: {session_data.get('title', 'Unnamed Session')}",
            description="Applying structured thinking to session analysis"
        )
        
        # Process session data through thinking stages
        analysis_result = await self.structured_thinking.analyze_problem_with_stages(
            problem=f"Analyze coding session: {session_data.get('summary', '')}",
            project_context=session_data.get("context", {})
        )
        
        # Generate enhanced session insights
        enhanced_insights = {
            "original_analysis": session_data,
            "structured_analysis": analysis_result,
            "thinking_session_id": thinking_session_id,
            "enhanced_patterns": await self._extract_enhanced_patterns(session_data, analysis_result),
            "learning_insights": await self._generate_learning_insights(session_data, analysis_result)
        }
        
        return enhanced_insights
    
    async def _extract_enhanced_patterns(self, session_data: Dict[str, Any], thinking_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract enhanced patterns from structured thinking analysis."""
        
        patterns = []
        
        # Pattern extraction based on thinking components
        for component in thinking_analysis["analysis_components"]["components"]:
            pattern = {
                "pattern_type": component,
                "confidence": 0.8,  # High confidence due to structured analysis
                "source": "structured_thinking_analysis",
                "metadata": {
                    "thinking_session": thinking_analysis["session_id"],
                    "analysis_stage": "component_identification"
                }
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _generate_learning_insights(self, session_data: Dict[str, Any], thinking_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate learning insights from structured thinking."""
        
        insights = {
            "structured_approach_applied": True,
            "thinking_stages_used": 3,  # Based on analyze_problem_with_stages
            "research_depth": len(thinking_analysis["research_findings"]),
            "component_analysis_depth": len(thinking_analysis["analysis_components"]["components"]),
            "learning_recommendations": [
                "Continue using structured thinking for complex problems",
                "Build on identified patterns in future sessions",
                "Document thinking process for knowledge retention"
            ]
        }
        
        return insights