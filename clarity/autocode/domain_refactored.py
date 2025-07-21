"""
Refactored AutoCode domain using modular components.
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from clarity.shared.exceptions import AutoCodeError
from .interfaces import (
    ProjectPatternManager, 
    SessionManager, 
    LearningEngine, 
    StatsCollector
)
from .components import (
    ProjectPatternManagerImpl,
    SessionManagerImpl,
    LearningEngineImpl,
    StatsCollectorImpl
)


class AutoCodeDomainRefactored:
    """
    Refactored AutoCode domain with modular architecture.
    
    This domain handles:
    - Project pattern recognition and indexing
    - Command execution learning and suggestion  
    - Session summary generation
    - Cross-project pattern application
    - Performance monitoring and statistics
    """
    
    def __init__(self, config: Dict[str, Any], persistence_domain):
        """
        Initialize the AutoCode domain with modular components.
        
        Args:
            config: Configuration dictionary
            persistence_domain: Persistence domain for storage operations
        """
        self.config = config
        self.persistence_domain = persistence_domain
        
        # AutoCode specific configuration
        self.autocode_config = config.get("autocode", {
            "enabled": True,
            "auto_scan_projects": True,
            "track_bash_commands": True,
            "generate_session_summaries": True
        })
        
        # Initialize components
        self.pattern_manager: ProjectPatternManager = ProjectPatternManagerImpl(
            config, persistence_domain
        )
        self.session_manager: SessionManager = SessionManagerImpl(
            config, persistence_domain
        )
        self.learning_engine: LearningEngine = LearningEngineImpl(
            config, persistence_domain
        )
        self.stats_collector: StatsCollector = StatsCollectorImpl(
            config, persistence_domain
        )
        
        # Register components for stats collection
        self.stats_collector.register_component("pattern_manager", self.pattern_manager)
        self.stats_collector.register_component("session_manager", self.session_manager)
        self.stats_collector.register_component("learning_engine", self.learning_engine)
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the AutoCode domain and all components."""
        if self._initialized:
            return
            
        logger.info("Initializing AutoCode Domain (Refactored)")
        
        if not self.autocode_config.get("enabled", True):
            logger.info("AutoCode Domain disabled in configuration")
            return
        
        try:
            # Initialize all components in order
            await self.stats_collector.initialize()
            await self.pattern_manager.initialize()
            await self.session_manager.initialize()
            await self.learning_engine.initialize()
            
            # Set domain manager reference for learning engine
            if hasattr(self.learning_engine, 'domain_manager'):
                self.learning_engine.domain_manager = getattr(self, 'domain_manager', None)
            
            self._initialized = True
            logger.info("AutoCode Domain (Refactored) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AutoCode Domain: {e}")
            raise AutoCodeError("AutoCode domain initialization failed", cause=e)
    
    async def shutdown(self) -> None:
        """Shutdown the AutoCode domain and all components."""
        logger.info("Shutting down AutoCode Domain")
        
        try:
            # Shutdown components in reverse order
            await self.learning_engine.shutdown()
            await self.session_manager.shutdown()
            await self.pattern_manager.shutdown()
            await self.stats_collector.shutdown()
            
            self._initialized = False
            logger.info("AutoCode Domain shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during AutoCode Domain shutdown: {e}")
    
    def set_domain_manager(self, domain_manager) -> None:
        """Set the domain manager reference for components that need it."""
        if hasattr(self.learning_engine, 'domain_manager'):
            self.learning_engine.domain_manager = domain_manager
    
    # Project Pattern Management Methods
    
    async def get_project_patterns(self, project_path: str, pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get patterns for a specific project."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            result = await self.pattern_manager.get_project_patterns(project_path, pattern_types)
            await self.stats_collector.track_operation(
                "get_project_patterns", 
                time.time() - start_time, 
                True,
                {"project_path": project_path, "pattern_types": pattern_types}
            )
            return result
        except Exception as e:
            await self.stats_collector.track_operation(
                "get_project_patterns", 
                time.time() - start_time, 
                False
            )
            raise
    
    # Session Management Methods
    
    async def process_file_access(self, file_path: str, access_type: str, 
                                project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process file access events."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            await self.session_manager.process_file_access(file_path, access_type, project_context)
            await self.stats_collector.track_operation(
                "process_file_access", 
                time.time() - start_time, 
                True
            )
        except Exception as e:
            await self.stats_collector.track_operation(
                "process_file_access", 
                time.time() - start_time, 
                False
            )
            # Don't re-raise for file access tracking - it's not critical
            logger.warning(f"Failed to process file access: {e}")
    
    async def process_bash_execution(self, command: str, working_directory: str, 
                                   success: bool, output: str, 
                                   project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process bash command execution for learning."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            await self.learning_engine.process_bash_execution(
                command, working_directory, success, output, project_context
            )
            await self.stats_collector.track_operation(
                "process_bash_execution", 
                time.time() - start_time, 
                True
            )
        except Exception as e:
            await self.stats_collector.track_operation(
                "process_bash_execution", 
                time.time() - start_time, 
                False
            )
            # Don't re-raise for bash tracking - it's not critical
            logger.warning(f"Failed to process bash execution: {e}")
    
    async def generate_session_summary(self, conversation_log: List[Dict]) -> str:
        """Generate a summary of the current session."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            result = await self.session_manager.generate_session_summary(conversation_log)
            await self.stats_collector.track_operation(
                "generate_session_summary", 
                time.time() - start_time, 
                True
            )
            return result
        except Exception as e:
            await self.stats_collector.track_operation(
                "generate_session_summary", 
                time.time() - start_time, 
                False
            )
            raise
    
    async def find_similar_sessions(self, query: str, context: Optional[Dict[str, Any]] = None, 
                                  time_range_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find sessions similar to the current context."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            result = await self.session_manager.find_similar_sessions(query, context, time_range_days)
            await self.stats_collector.track_operation(
                "find_similar_sessions", 
                time.time() - start_time, 
                True
            )
            return result
        except Exception as e:
            await self.stats_collector.track_operation(
                "find_similar_sessions", 
                time.time() - start_time, 
                False
            )
            raise
    
    async def get_context_for_continuation(self, current_task: str, 
                                         project_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get relevant context for continuing work on a task."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            result = await self.session_manager.get_context_for_continuation(current_task, project_context)
            await self.stats_collector.track_operation(
                "get_context_for_continuation", 
                time.time() - start_time, 
                True
            )
            return result
        except Exception as e:
            await self.stats_collector.track_operation(
                "get_context_for_continuation", 
                time.time() - start_time, 
                False
            )
            raise
    
    # Learning and Suggestion Methods
    
    async def suggest_command(self, intent: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get intelligent command suggestions based on intent and context."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            result = await self.learning_engine.suggest_command(intent, context)
            await self.stats_collector.track_operation(
                "suggest_command", 
                time.time() - start_time, 
                True
            )
            return result
        except Exception as e:
            await self.stats_collector.track_operation(
                "suggest_command", 
                time.time() - start_time, 
                False
            )
            raise
    
    async def suggest_workflow_optimizations(self, current_workflow: List[str], 
                                           session_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Suggest workflow optimizations based on historical data."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            result = await self.learning_engine.suggest_workflow_optimizations(current_workflow, session_context)
            await self.stats_collector.track_operation(
                "suggest_workflow_optimizations", 
                time.time() - start_time, 
                True
            )
            return result
        except Exception as e:
            await self.stats_collector.track_operation(
                "suggest_workflow_optimizations", 
                time.time() - start_time, 
                False
            )
            raise
    
    async def get_learning_progression(self, topic: str, time_range_days: int = 180) -> Dict[str, Any]:
        """Track learning progression on a specific topic."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        try:
            result = await self.learning_engine.get_learning_progression(topic, time_range_days)
            await self.stats_collector.track_operation(
                "get_learning_progression", 
                time.time() - start_time, 
                True
            )
            return result
        except Exception as e:
            await self.stats_collector.track_operation(
                "get_learning_progression", 
                time.time() - start_time, 
                False
            )
            raise
    
    # Statistics and Monitoring Methods
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        if not self._initialized:
            await self.initialize()
        
        try:
            stats = await self.stats_collector.get_stats()
            
            # Add domain-level information
            stats["domain_info"] = {
                "name": "AutoCodeDomainRefactored",
                "initialized": self._initialized,
                "config": self.autocode_config,
                "components": {
                    "pattern_manager": "ProjectPatternManagerImpl",
                    "session_manager": "SessionManagerImpl", 
                    "learning_engine": "LearningEngineImpl",
                    "stats_collector": "StatsCollectorImpl"
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get AutoCode stats: {e}")
            raise AutoCodeError("Statistics collection failed", cause=e)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self._initialized:
            await self.initialize()
        
        return await self.stats_collector.get_performance_metrics()
    
    # Legacy compatibility methods (for backward compatibility)
    
    async def set_command_learner(self, domain_manager):
        """Legacy method for backward compatibility."""
        self.set_domain_manager(domain_manager)
        logger.debug("Set domain manager for legacy compatibility")


# Import time for performance tracking
import time