"""
Session management component for AutoCode domain.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from clarity.shared.infrastructure import get_cache
from clarity.shared.exceptions import AutoCodeError
from ..interfaces import SessionManager, AutoCodeComponent
from ..session_analyzer import SessionAnalyzer


class SessionManagerImpl(SessionManager, AutoCodeComponent):
    """Implementation of session management"""
    
    def __init__(self, config: Dict[str, Any], persistence_domain):
        """Initialize session manager"""
        self.config = config
        self.persistence_domain = persistence_domain
        self.autocode_config = config.get("autocode", {})
        
        # Session caching
        self.session_cache = get_cache(
            "sessions",
            max_size=200,       # Cache 200 sessions
            max_memory_mb=50,   # 50MB for session data
            default_ttl=1800.0  # 30 minutes TTL
        )
        
        # Context caching for continuation
        self.context_cache = get_cache(
            "session_context",
            max_size=100,       # Cache 100 contexts
            max_memory_mb=25,   # 25MB for context data
            default_ttl=3600.0  # 1 hour TTL
        )
        
        # Initialize session analyzer
        self.session_analyzer = None
        self._initialized = False
        self._current_session_id = None
        self._session_start_time = None
        self._file_access_log = []
        self._bash_execution_log = []
    
    async def initialize(self) -> None:
        """Initialize the session manager"""
        if self._initialized:
            return
            
        logger.info("Initializing Session Manager")
        
        try:
            # Initialize session analyzer if enabled
            if self.autocode_config.get("session_analysis", {}).get("enabled", True):
                self.session_analyzer = SessionAnalyzer(self.autocode_config)
                await self.session_analyzer.initialize()
            
            # Start new session
            await self._start_new_session()
            
            self._initialized = True
            logger.info("Session Manager initialized successfully")
            
        except (AttributeError, ImportError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize Session Manager: {e}")
            raise AutoCodeError("Session manager initialization failed", cause=e)
    
    async def shutdown(self) -> None:
        """Shutdown the session manager"""
        if self._current_session_id:
            await self._end_current_session()
        
        if self.session_analyzer:
            await self.session_analyzer.shutdown()
            
        self._initialized = False
        logger.info("Session Manager shutdown complete")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": "SessionManager",
            "initialized": self._initialized,
            "current_session_id": self._current_session_id,
            "session_start_time": self._session_start_time,
            "file_accesses": len(self._file_access_log),
            "bash_executions": len(self._bash_execution_log),
            "session_cache_info": self.session_cache.get_info(),
            "context_cache_info": self.context_cache.get_info(),
            "session_analyzer_enabled": self.session_analyzer is not None
        }
    
    async def generate_session_summary(self, conversation_log: List[Dict]) -> str:
        """Generate a summary of the current session"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create cache key for the summary
            cache_key = f"summary_{self._current_session_id}_{len(conversation_log)}"
            
            # Try cache first
            cached_summary = self.session_cache.get(cache_key)
            if cached_summary is not None:
                return cached_summary
            
            # Collect session data
            session_data = {
                "session_id": self._current_session_id,
                "start_time": self._session_start_time,
                "conversation_log": conversation_log,
                "file_access_log": self._file_access_log,
                "bash_execution_log": self._bash_execution_log,
                "duration_minutes": self._get_session_duration_minutes()
            }
            
            # Generate summary using session analyzer
            if self.session_analyzer:
                summary = await self.session_analyzer.generate_summary(session_data)
            else:
                summary = await self._generate_basic_summary(session_data)
            
            # Cache the summary
            self.session_cache.set(cache_key, summary)
            
            logger.debug(f"Generated session summary for session {self._current_session_id}")
            return summary
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to generate session summary: {e}")
            raise AutoCodeError("Session summary generation failed", cause=e)
    
    async def find_similar_sessions(self, query: str, context: Optional[Dict[str, Any]] = None, 
                                  time_range_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find sessions similar to the current context"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create cache key
            cache_key = f"similar_{hash(query)}_{hash(str(context))}_{time_range_days}"
            
            # Try cache first
            cached_sessions = self.session_cache.get(cache_key)
            if cached_sessions is not None:
                return cached_sessions
            
            # Query for similar sessions from persistence domain
            search_query = f"session summary: {query}"
            if context:
                # Add context information to search
                for key, value in context.items():
                    search_query += f" {key}:{value}"
            
            # Retrieve similar session memories
            similar_memories = await self.persistence_domain.retrieve_memories(
                query=search_query,
                types=["session_summary", "autocode_session"],
                limit=20,
                min_similarity=0.7
            )
            
            sessions = []
            for memory in similar_memories:
                try:
                    content = memory.get("content", {})
                    if isinstance(content, dict):
                        session_info = {
                            "session_id": content.get("session_id"),
                            "summary": content.get("summary", ""),
                            "start_time": content.get("start_time"),
                            "duration_minutes": content.get("duration_minutes", 0),
                            "file_accesses": len(content.get("file_access_log", [])),
                            "bash_executions": len(content.get("bash_execution_log", [])),
                            "similarity_score": memory.get("score", 0.0),
                            "technologies": content.get("technologies", []),
                            "project_path": content.get("project_path")
                        }
                        
                        # Filter by time range if specified
                        if time_range_days:
                            session_date = datetime.fromisoformat(session_info["start_time"])
                            cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
                            if session_date < cutoff_date:
                                continue
                        
                        sessions.append(session_info)
                        
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to process session memory {memory.get('id')}: {e}")
                    continue
            
            # Sort by similarity score
            sessions.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Cache the result
            self.session_cache.set(cache_key, sessions, ttl=600.0)  # 10 minutes TTL
            
            logger.debug(f"Found {len(sessions)} similar sessions for query: {query[:50]}...")
            return sessions
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to find similar sessions: {e}")
            raise AutoCodeError("Similar session search failed", cause=e)
    
    async def get_context_for_continuation(self, current_task: str, 
                                         project_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get relevant context for continuing work on a task"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create cache key
            cache_key = f"context_{hash(current_task)}_{hash(str(project_context))}"
            
            # Try cache first
            cached_context = self.context_cache.get(cache_key)
            if cached_context is not None:
                return cached_context
            
            # Build context from current session and historical data
            context = {
                "current_session": {
                    "session_id": self._current_session_id,
                    "duration_minutes": self._get_session_duration_minutes(),
                    "recent_files": self._get_recent_file_accesses(10),
                    "recent_commands": self._get_recent_bash_executions(10),
                    "current_task": current_task
                },
                "project_context": project_context or {},
                "similar_sessions": [],
                "relevant_patterns": [],
                "suggestions": []
            }
            
            # Find similar sessions for this task
            similar_sessions = await self.find_similar_sessions(
                current_task, 
                project_context, 
                time_range_days=30
            )
            context["similar_sessions"] = similar_sessions[:5]  # Top 5 similar sessions
            
            # Get relevant patterns if project context available
            if project_context and "project_path" in project_context:
                try:
                    # This would typically call the pattern manager
                    # For now, we'll add a placeholder
                    context["relevant_patterns"] = []
                except (ValueError, AttributeError, KeyError, RuntimeError) as e:
                    logger.warning(f"Failed to get patterns for context: {e}")
            
            # Generate suggestions based on context
            context["suggestions"] = await self._generate_continuation_suggestions(context)
            
            # Cache the context
            self.context_cache.set(cache_key, context)
            
            logger.debug(f"Generated continuation context for task: {current_task[:50]}...")
            return context
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to get continuation context: {e}")
            raise AutoCodeError("Context generation failed", cause=e)
    
    async def process_file_access(self, file_path: str, access_type: str, 
                                project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process file access events"""
        if not self._initialized:
            await self.initialize()
        
        try:
            access_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "file_path": file_path,
                "access_type": access_type,  # read, write, create, delete
                "project_context": project_context or {},
                "session_id": self._current_session_id
            }
            
            # Add to current session log
            self._file_access_log.append(access_event)
            
            # Store in persistence domain for long-term analysis
            await self.persistence_domain.store_memory(
                memory_type="file_access",
                content=access_event,
                importance=0.3,  # Lower importance for individual file accesses
                metadata={
                    "session_id": self._current_session_id,
                    "access_type": access_type,
                    "file_extension": os.path.splitext(file_path)[1].lower()
                }
            )
            
            logger.debug(f"Processed file access: {access_type} {file_path}")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to process file access event: {e}")
    
    async def _start_new_session(self) -> None:
        """Start a new session"""
        self._current_session_id = str(uuid.uuid4())
        self._session_start_time = datetime.utcnow().isoformat()
        self._file_access_log = []
        self._bash_execution_log = []
        
        logger.info(f"Started new session: {self._current_session_id}")
    
    async def _end_current_session(self) -> None:
        """End the current session and store summary"""
        if not self._current_session_id:
            return
        
        try:
            # Create session summary
            session_data = {
                "session_id": self._current_session_id,
                "start_time": self._session_start_time,
                "end_time": datetime.utcnow().isoformat(),
                "duration_minutes": self._get_session_duration_minutes(),
                "file_access_log": self._file_access_log,
                "bash_execution_log": self._bash_execution_log,
                "total_file_accesses": len(self._file_access_log),
                "total_bash_executions": len(self._bash_execution_log)
            }
            
            # Store session data
            await self.persistence_domain.store_memory(
                memory_type="session_summary",
                content=session_data,
                importance=0.7,
                metadata={
                    "session_id": self._current_session_id,
                    "duration_minutes": session_data["duration_minutes"]
                }
            )
            
            logger.info(f"Ended session: {self._current_session_id}")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to end session: {e}")
    
    def _get_session_duration_minutes(self) -> float:
        """Get current session duration in minutes"""
        if not self._session_start_time:
            return 0.0
        
        start = datetime.fromisoformat(self._session_start_time)
        return (datetime.utcnow() - start).total_seconds() / 60.0
    
    def _get_recent_file_accesses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent file accesses"""
        return self._file_access_log[-limit:] if self._file_access_log else []
    
    def _get_recent_bash_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent bash executions"""
        return self._bash_execution_log[-limit:] if self._bash_execution_log else []
    
    async def _generate_basic_summary(self, session_data: Dict[str, Any]) -> str:
        """Generate a basic session summary when session analyzer is not available"""
        duration = session_data.get("duration_minutes", 0)
        file_accesses = len(session_data.get("file_access_log", []))
        bash_executions = len(session_data.get("bash_execution_log", []))
        
        summary = f"Session lasted {duration:.1f} minutes with {file_accesses} file accesses and {bash_executions} command executions."
        
        # Add file type analysis
        if session_data.get("file_access_log"):
            file_types = {}
            for access in session_data["file_access_log"]:
                ext = os.path.splitext(access["file_path"])[1].lower()
                if ext:
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            if file_types:
                top_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
                types_str = ", ".join([f"{ext} ({count})" for ext, count in top_types])
                summary += f" Primary file types: {types_str}."
        
        return summary
    
    async def _generate_continuation_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Generate suggestions for continuing work"""
        suggestions = []
        
        # Analyze recent activity
        recent_files = context["current_session"].get("recent_files", [])
        recent_commands = context["current_session"].get("recent_commands", [])
        
        # Suggest based on file patterns
        if recent_files:
            file_types = set()
            for access in recent_files:
                ext = os.path.splitext(access["file_path"])[1].lower()
                if ext:
                    file_types.add(ext)
            
            if ".py" in file_types:
                suggestions.append("Consider running tests with pytest or python -m unittest")
            if ".js" in file_types or ".ts" in file_types:
                suggestions.append("Consider running npm test or yarn test")
            if ".md" in file_types:
                suggestions.append("Consider building documentation or running spell check")
        
        # Suggest based on command patterns
        if recent_commands:
            commands = [cmd.get("command", "") for cmd in recent_commands]
            if any("git" in cmd for cmd in commands):
                suggestions.append("Consider committing changes or checking git status")
            if any("install" in cmd for cmd in commands):
                suggestions.append("Consider running the application or tests after installation")
        
        # Suggest based on similar sessions
        similar_sessions = context.get("similar_sessions", [])
        if similar_sessions:
            suggestions.append("Check similar sessions for patterns and best practices")
        
        return suggestions[:5]  # Limit to 5 suggestions