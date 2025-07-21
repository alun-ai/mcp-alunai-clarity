"""
Interfaces for AutoCode domain components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class ProjectPatternManager(ABC):
    """Interface for project pattern management"""
    
    @abstractmethod
    async def get_project_patterns(self, project_path: str, pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get patterns for a specific project"""
        pass
    
    @abstractmethod
    async def detect_project_patterns(self, project_path: str) -> Dict[str, Any]:
        """Detect and analyze patterns in a project"""
        pass
    
    @abstractmethod
    async def cache_project_patterns(self, project_path: str, patterns: Dict[str, Any]) -> None:
        """Cache patterns for a project"""
        pass
    
    @abstractmethod
    async def load_existing_patterns(self) -> None:
        """Load existing patterns from storage"""
        pass


class SessionManager(ABC):
    """Interface for session management"""
    
    @abstractmethod
    async def generate_session_summary(self, conversation_log: List[Dict]) -> str:
        """Generate a summary of the current session"""
        pass
    
    @abstractmethod
    async def find_similar_sessions(self, query: str, context: Optional[Dict[str, Any]] = None, 
                                  time_range_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find sessions similar to the current context"""
        pass
    
    @abstractmethod
    async def get_context_for_continuation(self, current_task: str, 
                                         project_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get relevant context for continuing work on a task"""
        pass
    
    @abstractmethod
    async def process_file_access(self, file_path: str, access_type: str, 
                                project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process file access events"""
        pass


class LearningEngine(ABC):
    """Interface for learning and suggestion engine"""
    
    @abstractmethod
    async def suggest_command(self, intent: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get intelligent command suggestions based on intent and context"""
        pass
    
    @abstractmethod
    async def suggest_workflow_optimizations(self, current_workflow: List[str], 
                                           session_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Suggest workflow optimizations based on historical data"""
        pass
    
    @abstractmethod
    async def get_learning_progression(self, topic: str, time_range_days: int = 180) -> Dict[str, Any]:
        """Track learning progression on a specific topic"""
        pass
    
    @abstractmethod
    async def process_bash_execution(self, command: str, working_directory: str, 
                                   success: bool, output: str, project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process bash command execution for learning"""
        pass


class StatsCollector(ABC):
    """Interface for statistics collection"""
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        pass
    
    @abstractmethod
    async def track_operation(self, operation: str, duration: float, success: bool, context: Dict[str, Any] = None) -> None:
        """Track operation performance"""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass


class AutoCodeComponent(ABC):
    """Base interface for all AutoCode components"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component gracefully"""
        pass
    
    @abstractmethod
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        pass