# AutoCodeIndex Implementation Plan
## Extending MCP Persistent Memory for Code Intelligence

**Project Location**: `/Users/chadupton/Documents/Github/alun-ai/mcp-persistent-memory`
**Created**: 2025-07-19
**Status**: Planning Phase

## Overview

AutoCodeIndex extends the existing MCP Persistent Memory server to provide intelligent code project awareness, command learning, and session history tracking. This eliminates the need for Claude to repeatedly rediscover project patterns, retry failed commands, or lose context between sessions.

### Key Features
- **Project Pattern Recognition** - Index architectural patterns, naming conventions, and component relationships
- **Command Intelligence** - Learn from bash command successes/failures and suggest optimal commands
- **Session History** - Track what was accomplished in previous sessions with searchable summaries
- **Cross-Project Learning** - Apply successful patterns across different projects
- **Automatic Indexing** - Zero-friction learning during normal Claude operations

## Architecture Analysis

### Existing Foundation
- **Domain-based architecture** with 4 specialized domains (Episodic, Semantic, Temporal, Persistence)
- **Memory types**: conversation, fact, document, entity, reflection, code
- **Tiered storage**: short_term, long_term, archived
- **Semantic search** with vector embeddings
- **Pydantic schemas** for validation
- **MCP protocol** integration

### Integration Strategy
Extend existing architecture by adding a new **AutoCode Domain** that integrates with all existing domains while adding specialized intelligence for code projects and command execution.

## Implementation Phases

## Phase 1: Core AutoCodeIndex Foundation (Week 1-2)

### 1.1 New Memory Types
**File**: `memory_mcp/utils/schema.py`

Add new memory types to existing schema:

```python
class ProjectPatternMemory(MemoryBase):
    """Memory for project architectural patterns."""
    type: str = "project_pattern"
    content: Dict[str, Any]
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["pattern_type", "framework", "language", "structure"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Project pattern must have '{field}' field")
        return v

class CommandPatternMemory(MemoryBase):
    """Memory for bash command patterns and success rates."""
    type: str = "command_pattern"
    content: Dict[str, Any]
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["command", "context", "success_rate", "platform"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Command pattern must have '{field}' field")
        return v

class SessionSummaryMemory(MemoryBase):
    """Memory for session summaries and completed work."""
    type: str = "session_summary"
    content: Dict[str, Any]
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["session_id", "tasks_completed", "patterns_used", "files_modified"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Session summary must have '{field}' field")
        return v

class BashExecutionMemory(MemoryBase):
    """Memory for individual bash command executions."""
    type: str = "bash_execution"
    content: Dict[str, Any]
    
    @validator("content")
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["command", "exit_code", "timestamp", "context"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Bash execution must have '{field}' field")
        return v
```

### 1.2 AutoCode Domain
**New File**: `memory_mcp/domains/autocode.py`

```python
from typing import Any, Dict, List, Optional
from loguru import logger
from .persistence import PersistenceDomain

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
        self.config = config
        self.persistence_domain = persistence_domain
        self.pattern_cache = {}
        self.command_patterns = {}
        
    async def initialize(self) -> None:
        """Initialize the AutoCode domain."""
        logger.info("Initializing AutoCode Domain")
        await self._load_existing_patterns()
        logger.info("AutoCode Domain initialized")
    
    async def process_file_access(self, file_path: str, content: str, operation: str) -> None:
        """
        Process file access to extract patterns.
        
        Args:
            file_path: Path to the accessed file
            content: File content (if read operation)
            operation: Type of operation (read, write, edit)
        """
        # Extract patterns from file access
        patterns = await self._extract_file_patterns(file_path, content)
        
        # Store discovered patterns
        for pattern in patterns:
            await self._store_pattern(pattern)
    
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
        success = exit_code == 0
        
        # Store execution record
        execution_data = {
            "command": command,
            "exit_code": exit_code,
            "output": output[:1000],  # Truncate long output
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "success": success
        }
        
        await self._store_bash_execution(execution_data)
        
        # Update command success patterns
        await self._update_command_patterns(command, success, context)
    
    async def generate_session_summary(self, conversation_log: List[Dict]) -> str:
        """
        Generate session summary from conversation log.
        
        Args:
            conversation_log: List of conversation messages
            
        Returns:
            Session summary ID
        """
        summary = await self._analyze_session(conversation_log)
        return await self._store_session_summary(summary)
    
    async def suggest_command(self, intent: str, context: Dict[str, Any]) -> List[str]:
        """
        Suggest optimal commands for given intent.
        
        Args:
            intent: What the user wants to accomplish
            context: Current context (project type, platform, etc.)
            
        Returns:
            List of suggested commands, ranked by success probability
        """
        return await self._get_command_suggestions(intent, context)
    
    async def get_project_patterns(self, project_path: str) -> Dict[str, Any]:
        """
        Get known patterns for a project.
        
        Args:
            project_path: Path to the project
            
        Returns:
            Dictionary of known patterns for the project
        """
        return await self._retrieve_project_patterns(project_path)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get AutoCode domain statistics."""
        return {
            "total_patterns": len(self.pattern_cache),
            "command_patterns": len(self.command_patterns),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # Private implementation methods
    async def _extract_file_patterns(self, file_path: str, content: str) -> List[Dict]:
        """Extract patterns from file content."""
        # Implementation details...
        pass
    
    async def _store_pattern(self, pattern: Dict) -> None:
        """Store a discovered pattern."""
        # Implementation details...
        pass
    
    async def _store_bash_execution(self, execution_data: Dict) -> None:
        """Store bash execution record."""
        # Implementation details...
        pass
    
    async def _update_command_patterns(self, command: str, success: bool, context: Dict) -> None:
        """Update command success patterns."""
        # Implementation details...
        pass
    
    async def _analyze_session(self, conversation_log: List[Dict]) -> Dict:
        """Analyze conversation log to generate summary."""
        # Implementation details...
        pass
    
    async def _store_session_summary(self, summary: Dict) -> str:
        """Store session summary."""
        # Implementation details...
        pass
```

### 1.3 Enhanced Manager Integration
**File**: `memory_mcp/domains/manager.py`

Add AutoCode domain to existing MemoryDomainManager:

```python
# Add to imports
from memory_mcp.domains.autocode import AutoCodeDomain

class MemoryDomainManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the memory domain manager."""
        self.config = config
        
        # Initialize existing domains
        self.persistence_domain = PersistenceDomain(config)
        self.episodic_domain = EpisodicDomain(config, self.persistence_domain)
        self.semantic_domain = SemanticDomain(config, self.persistence_domain)
        self.temporal_domain = TemporalDomain(config, self.persistence_domain)
        
        # Add AutoCode domain
        self.autocode_domain = AutoCodeDomain(config, self.persistence_domain)
    
    async def initialize(self) -> None:
        """Initialize all domains."""
        logger.info("Initializing Memory Domain Manager")
        
        # Initialize domains in order (persistence first)
        await self.persistence_domain.initialize()
        await self.episodic_domain.initialize()
        await self.semantic_domain.initialize()
        await self.temporal_domain.initialize()
        await self.autocode_domain.initialize()  # Add AutoCode initialization
        
        logger.info("Memory Domain Manager initialized")
    
    # Add new AutoCode-specific methods
    async def store_project_pattern(
        self,
        pattern_type: str,
        framework: str,
        language: str,
        structure: Dict[str, Any],
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a project pattern."""
        return await self.store_memory(
            memory_type="project_pattern",
            content={
                "pattern_type": pattern_type,
                "framework": framework,
                "language": language,
                "structure": structure
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_command_pattern(
        self,
        command: str,
        context: Dict[str, Any],
        success_rate: float,
        platform: str,
        importance: float = 0.6,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a command pattern."""
        return await self.store_memory(
            memory_type="command_pattern",
            content={
                "command": command,
                "context": context,
                "success_rate": success_rate,
                "platform": platform
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_session_summary(
        self,
        session_id: str,
        tasks_completed: List[Dict],
        patterns_used: List[str],
        files_modified: List[str],
        importance: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a session summary."""
        return await self.store_memory(
            memory_type="session_summary",
            content={
                "session_id": session_id,
                "tasks_completed": tasks_completed,
                "patterns_used": patterns_used,
                "files_modified": files_modified
            },
            importance=importance,
            metadata=metadata
        )
```

## Phase 2: Command Intelligence System (Week 3-4)

### 2.1 Command Learning Infrastructure
**New File**: `memory_mcp/autocode/command_learner.py`

```python
import asyncio
import platform
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

class CommandLearner:
    """
    Learns from bash command executions and suggests optimal commands.
    
    This class analyzes command execution patterns, learns from failures,
    and provides intelligent command suggestions based on context.
    """
    
    def __init__(self, domain_manager):
        self.domain_manager = domain_manager
        self.command_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self.context_weights = {
            "project_type": 0.3,
            "platform": 0.2,
            "recent_success": 0.4,
            "user_preference": 0.1
        }
        self.platform = platform.system().lower()
    
    async def track_bash_execution(
        self, 
        command: str, 
        exit_code: int, 
        output: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Track command execution results for learning.
        
        Args:
            command: The bash command that was executed
            exit_code: Exit code (0 = success, non-zero = failure)
            output: Command output/error message
            context: Execution context (project type, current directory, etc.)
        """
        success = exit_code == 0
        timestamp = datetime.utcnow()
        
        # Store execution record
        execution_record = {
            "command": command,
            "exit_code": exit_code,
            "output": output[:500],  # Truncate long output
            "success": success,
            "timestamp": timestamp.isoformat(),
            "context": context,
            "platform": self.platform
        }
        
        # Store in persistent memory
        await self.domain_manager.autocode_domain.process_bash_execution(
            command, exit_code, output, context
        )
        
        # Update local patterns for quick access
        intent = self._extract_intent(command)
        if success:
            self.command_patterns[intent].append(execution_record)
        else:
            self.failure_patterns[intent].append(execution_record)
        
        # Learn from retry patterns if this might be a retry
        await self._detect_retry_pattern(command, success, context)
    
    async def suggest_command(
        self, 
        intent: str, 
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimal commands for given intent.
        
        Args:
            intent: What the user wants to accomplish (e.g., "delete file", "install deps")
            context: Current context (project type, platform, etc.)
            
        Returns:
            List of command suggestions with confidence scores
        """
        context = context or {}
        context["platform"] = self.platform
        
        # Get historical patterns for this intent
        patterns = await self._get_patterns_for_intent(intent, context)
        
        # Rank suggestions by success rate and context relevance
        suggestions = []
        for pattern in patterns:
            confidence = self._calculate_confidence(pattern, context)
            if confidence > 0.3:  # Only suggest if reasonably confident
                suggestions.append({
                    "command": pattern["command"],
                    "confidence": confidence,
                    "success_rate": pattern.get("success_rate", 0.5),
                    "last_used": pattern.get("timestamp"),
                    "context": pattern.get("context", {})
                })
        
        # Sort by confidence descending
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    async def learn_retry_patterns(self) -> None:
        """
        Analyze failed->success command sequences to learn retry patterns.
        
        This method identifies patterns where a command failed and was then
        modified to succeed, learning the correction pattern.
        """
        # Get recent bash executions
        recent_executions = await self._get_recent_executions(hours=24)
        
        # Look for failure->success sequences
        retry_patterns = self._detect_retry_sequences(recent_executions)
        
        # Store learned retry patterns
        for pattern in retry_patterns:
            await self._store_retry_pattern(pattern)
    
    async def get_command_context(self, project_path: str = None) -> Dict[str, Any]:
        """
        Get current command context for better suggestions.
        
        Args:
            project_path: Current project path
            
        Returns:
            Context dictionary with project info, platform, etc.
        """
        context = {
            "platform": self.platform,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if project_path:
            # Detect project type
            project_type = await self._detect_project_type(project_path)
            context["project_type"] = project_type
            context["project_path"] = project_path
        
        return context
    
    # Private helper methods
    def _extract_intent(self, command: str) -> str:
        """Extract intent from command (e.g., 'rm file.txt' -> 'delete_file')."""
        # Simple intent extraction - can be enhanced
        command_parts = command.split()
        if not command_parts:
            return "unknown"
        
        base_command = command_parts[0]
        intent_mapping = {
            "rm": "delete_file",
            "mkdir": "create_directory",
            "cp": "copy_file",
            "mv": "move_file",
            "npm": "package_management",
            "yarn": "package_management",
            "git": "version_control",
            "python": "run_python",
            "node": "run_node",
            "cargo": "rust_build",
            "docker": "container_management"
        }
        
        return intent_mapping.get(base_command, base_command)
    
    async def _get_patterns_for_intent(
        self, 
        intent: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get command patterns for specific intent and context."""
        # Query persistent memory for command patterns
        query = f"intent:{intent} platform:{context.get('platform', '')}"
        
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["command_pattern", "bash_execution"],
            limit=20,
            min_similarity=0.5
        )
        
        return [memory["content"] for memory in memories]
    
    def _calculate_confidence(
        self, 
        pattern: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a command suggestion."""
        confidence = 0.0
        
        # Base success rate
        success_rate = pattern.get("success_rate", 0.5)
        confidence += success_rate * 0.4
        
        # Platform match
        if pattern.get("platform") == context.get("platform"):
            confidence += 0.2
        
        # Project type match
        pattern_project = pattern.get("context", {}).get("project_type")
        context_project = context.get("project_type")
        if pattern_project and pattern_project == context_project:
            confidence += 0.2
        
        # Recency bonus
        if pattern.get("timestamp"):
            try:
                pattern_time = datetime.fromisoformat(pattern["timestamp"])
                days_ago = (datetime.utcnow() - pattern_time).days
                recency_bonus = max(0, 0.2 - (days_ago * 0.01))
                confidence += recency_bonus
            except:
                pass
        
        return min(1.0, confidence)
    
    async def _detect_retry_pattern(
        self, 
        command: str, 
        success: bool, 
        context: Dict[str, Any]
    ) -> None:
        """Detect if this command might be a retry of a failed command."""
        if not success:
            return  # Only interested in successful retries
        
        # Get recent failed commands (last 5 minutes)
        recent_time = datetime.utcnow() - timedelta(minutes=5)
        recent_failures = await self._get_failed_commands_since(recent_time)
        
        # Look for similar commands that failed recently
        for failed_cmd in recent_failures:
            similarity = self._calculate_command_similarity(command, failed_cmd["command"])
            if similarity > 0.7:  # Commands are similar
                # This might be a successful retry
                await self._store_retry_pattern({
                    "failed_command": failed_cmd["command"],
                    "successful_command": command,
                    "context": context,
                    "similarity": similarity
                })
    
    def _calculate_command_similarity(self, cmd1: str, cmd2: str) -> float:
        """Calculate similarity between two commands."""
        # Simple similarity based on shared words
        words1 = set(cmd1.split())
        words2 = set(cmd2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
```

### 2.2 Enhanced MCP Tools
**File**: `memory_mcp/mcp/tools.py`

Add new tool schemas to existing MemoryToolDefinitions class:

```python
class MemoryToolDefinitions:
    # ... existing methods ...
    
    @property
    def suggest_command_schema(self) -> Dict[str, Any]:
        """Schema for the suggest_command tool."""
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What you want to accomplish (e.g., 'delete file', 'install dependencies')"
                },
                "context": {
                    "type": "object",
                    "description": "Current context (project type, platform, etc.)",
                    "properties": {
                        "project_type": {"type": "string"},
                        "project_path": {"type": "string"},
                        "platform": {"type": "string"}
                    }
                }
            },
            "required": ["intent"]
        }
    
    @property
    def track_bash_schema(self) -> Dict[str, Any]:
        """Schema for the track_bash tool."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command that was executed"
                },
                "exit_code": {
                    "type": "integer",
                    "description": "Exit code from command execution"
                },
                "output": {
                    "type": "string",
                    "description": "Command output or error message"
                },
                "context": {
                    "type": "object",
                    "description": "Execution context",
                    "properties": {
                        "project_type": {"type": "string"},
                        "project_path": {"type": "string"},
                        "current_directory": {"type": "string"}
                    }
                }
            },
            "required": ["command", "exit_code"]
        }
    
    @property
    def get_session_history_schema(self) -> Dict[str, Any]:
        """Schema for the get_session_history tool."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for session history"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to return",
                    "minimum": 1,
                    "maximum": 20
                },
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to search",
                    "minimum": 1,
                    "maximum": 90
                }
            },
            "required": ["query"]
        }
    
    @property
    def get_project_patterns_schema(self) -> Dict[str, Any]:
        """Schema for the get_project_patterns tool."""
        return {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Path to the project to analyze"
                },
                "pattern_types": {
                    "type": "array",
                    "description": "Types of patterns to retrieve",
                    "items": {
                        "type": "string",
                        "enum": ["architectural", "naming", "component", "testing", "build"]
                    }
                }
            },
            "required": ["project_path"]
        }
```

## Phase 3: Project Pattern Recognition (Week 5-6)

### 3.1 Pattern Detection Engine
**New File**: `memory_mcp/autocode/pattern_detector.py`

```python
import os
import json
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

@dataclass
class DetectedPattern:
    """Represents a detected code pattern."""
    pattern_type: str
    name: str
    details: Dict[str, Any]
    confidence: float
    file_paths: List[str]

class PatternDetector:
    """
    Detects and indexes code project patterns.
    
    This class scans projects to identify:
    - Architectural patterns (MVC, Provider, Component-based, etc.)
    - Naming conventions (file names, functions, variables)
    - Framework usage patterns
    - Component relationships and dependencies
    - Testing patterns
    - Build and deployment patterns
    """
    
    def __init__(self, domain_manager):
        self.domain_manager = domain_manager
        self.supported_languages = {
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript", 
            ".jsx": "javascript",
            ".py": "python",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".rb": "ruby"
        }
        
        self.framework_indicators = {
            "react": ["package.json:react", "*.tsx", "*.jsx", "React."],
            "vue": ["package.json:vue", "*.vue", "Vue."],
            "angular": ["package.json:@angular", "*.component.ts", "angular.json"],
            "django": ["manage.py", "settings.py", "django"],
            "flask": ["app.py", "from flask", "Flask("],
            "fastapi": ["from fastapi", "FastAPI("],
            "express": ["package.json:express", "app.use(", "express()"],
            "nextjs": ["next.config.js", "pages/", "app/"],
            "svelte": ["package.json:svelte", "*.svelte"],
            "rails": ["Gemfile:rails", "config/routes.rb"],
            "spring": ["pom.xml:spring", "*.java", "@SpringBootApplication"]
        }
    
    async def scan_project(self, project_root: str) -> Dict[str, Any]:
        """
        Perform comprehensive project scan for patterns.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary containing all detected patterns
        """
        project_path = Path(project_root)
        if not project_path.exists():
            raise ValueError(f"Project path {project_root} does not exist")
        
        scan_results = {
            "project_path": str(project_path),
            "scan_timestamp": datetime.utcnow().isoformat(),
            "framework": await self.detect_framework(project_root),
            "language": await self.detect_primary_language(project_root),
            "architecture": await self.detect_architecture_patterns(project_root),
            "naming_conventions": await self.extract_naming_conventions(project_root),
            "component_relationships": await self.map_component_relationships(project_root),
            "testing_patterns": await self.detect_testing_patterns(project_root),
            "build_patterns": await self.detect_build_patterns(project_root),
            "file_structure": await self.analyze_file_structure(project_root)
        }
        
        # Store patterns in memory
        for pattern_type, patterns in scan_results.items():
            if patterns and pattern_type != "project_path" and pattern_type != "scan_timestamp":
                await self._store_project_pattern(project_root, pattern_type, patterns)
        
        return scan_results
    
    async def detect_framework(self, project_root: str) -> Optional[str]:
        """
        Detect the primary framework used in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Detected framework name or None
        """
        project_path = Path(project_root)
        detected_frameworks = []
        
        for framework, indicators in self.framework_indicators.items():
            confidence = 0
            
            for indicator in indicators:
                if ":" in indicator:
                    # File content indicator (e.g., "package.json:react")
                    file_name, content = indicator.split(":", 1)
                    file_path = project_path / file_name
                    
                    if file_path.exists():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                if content in f.read():
                                    confidence += 1
                        except:
                            pass
                elif indicator.startswith("*."):
                    # File extension indicator
                    extension = indicator[1:]  # Remove *
                    if list(project_path.rglob(f"*{extension}")):
                        confidence += 0.5
                else:
                    # File name indicator
                    if (project_path / indicator).exists():
                        confidence += 1
                    # Or check if it appears in any files
                    elif self._search_in_files(project_path, indicator):
                        confidence += 0.5
            
            if confidence > 0:
                detected_frameworks.append((framework, confidence))
        
        if detected_frameworks:
            # Return framework with highest confidence
            detected_frameworks.sort(key=lambda x: x[1], reverse=True)
            return detected_frameworks[0][0]
        
        return None
    
    async def detect_primary_language(self, project_root: str) -> str:
        """
        Detect the primary programming language of the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Primary language name
        """
        project_path = Path(project_root)
        language_counts = {}
        
        for file_path in project_path.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in self.supported_languages:
                    lang = self.supported_languages[suffix]
                    language_counts[lang] = language_counts.get(lang, 0) + 1
        
        if language_counts:
            return max(language_counts, key=language_counts.get)
        
        return "unknown"
    
    async def detect_architecture_patterns(self, project_root: str) -> Dict[str, Any]:
        """
        Detect architectural patterns in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of detected architectural patterns
        """
        project_path = Path(project_root)
        patterns = {}
        
        # Detect common directory structures
        common_patterns = {
            "mvc": ["models", "views", "controllers"],
            "mvvm": ["models", "views", "viewmodels"],
            "component_based": ["components", "pages", "layouts"],
            "layered": ["domain", "application", "infrastructure"],
            "clean_architecture": ["entities", "usecases", "adapters"],
            "microservices": ["services", "api", "gateway"]
        }
        
        for pattern_name, directories in common_patterns.items():
            matches = 0
            for directory in directories:
                if any(project_path.rglob(directory)):
                    matches += 1
            
            if matches >= len(directories) * 0.6:  # 60% match threshold
                patterns[pattern_name] = {
                    "confidence": matches / len(directories),
                    "matched_directories": [d for d in directories if any(project_path.rglob(d))]
                }
        
        # Detect specific patterns by file analysis
        patterns.update(await self._detect_code_patterns(project_path))
        
        return patterns
    
    async def extract_naming_conventions(self, project_root: str) -> Dict[str, Any]:
        """
        Extract naming conventions from the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of naming conventions
        """
        project_path = Path(project_root)
        conventions = {
            "files": {"patterns": [], "examples": []},
            "directories": {"patterns": [], "examples": []},
            "functions": {"patterns": [], "examples": []},
            "classes": {"patterns": [], "examples": []},
            "variables": {"patterns": [], "examples": []}
        }
        
        # Analyze file and directory names
        files = []
        directories = []
        
        for path in project_path.rglob("*"):
            if path.is_file() and not self._should_ignore_path(path):
                files.append(path.name)
            elif path.is_dir() and not self._should_ignore_path(path):
                directories.append(path.name)
        
        # Detect naming patterns
        conventions["files"] = self._analyze_naming_patterns(files)
        conventions["directories"] = self._analyze_naming_patterns(directories)
        
        # Analyze code for function/class/variable names
        for file_path in project_path.rglob("*"):
            if file_path.suffix in self.supported_languages:
                try:
                    code_conventions = await self._extract_code_naming_conventions(file_path)
                    for key in ["functions", "classes", "variables"]:
                        if key in code_conventions:
                            conventions[key]["examples"].extend(code_conventions[key])
                except:
                    pass
        
        # Analyze patterns for code elements
        for key in ["functions", "classes", "variables"]:
            if conventions[key]["examples"]:
                conventions[key].update(
                    self._analyze_naming_patterns(conventions[key]["examples"])
                )
        
        return conventions
    
    async def map_component_relationships(self, project_root: str) -> Dict[str, Any]:
        """
        Map relationships between components in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary mapping component relationships
        """
        project_path = Path(project_root)
        relationships = {
            "imports": {},
            "exports": {},
            "dependencies": {},
            "component_tree": {}
        }
        
        # Analyze import/export relationships
        for file_path in project_path.rglob("*"):
            if file_path.suffix in [".ts", ".tsx", ".js", ".jsx", ".py"]:
                try:
                    file_relationships = await self._analyze_file_relationships(file_path)
                    relative_path = str(file_path.relative_to(project_path))
                    relationships["imports"][relative_path] = file_relationships.get("imports", [])
                    relationships["exports"][relative_path] = file_relationships.get("exports", [])
                except:
                    pass
        
        # Build dependency graph
        relationships["dependencies"] = self._build_dependency_graph(relationships["imports"])
        
        return relationships
    
    async def detect_testing_patterns(self, project_root: str) -> Dict[str, Any]:
        """
        Detect testing patterns and conventions.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of testing patterns
        """
        project_path = Path(project_root)
        patterns = {
            "test_framework": None,
            "test_structure": {},
            "naming_conventions": {},
            "coverage_tools": []
        }
        
        # Detect test frameworks
        framework_indicators = {
            "jest": ["jest.config.js", "package.json:jest", "*.test.js", "*.spec.js"],
            "vitest": ["vitest.config.ts", "package.json:vitest"],
            "pytest": ["pytest.ini", "conftest.py", "test_*.py"],
            "mocha": ["mocha.opts", "package.json:mocha"],
            "jasmine": ["jasmine.json", "package.json:jasmine"],
            "rspec": ["spec/", ".rspec", "Gemfile:rspec"],
            "cargo_test": ["Cargo.toml", "tests/", "*.rs"]
        }
        
        for framework, indicators in framework_indicators.items():
            if self._check_indicators(project_path, indicators):
                patterns["test_framework"] = framework
                break
        
        # Analyze test structure
        test_dirs = list(project_path.rglob("test*")) + list(project_path.rglob("spec*"))
        patterns["test_structure"] = {
            "test_directories": [str(d.relative_to(project_path)) for d in test_dirs if d.is_dir()],
            "test_files": self._find_test_files(project_path)
        }
        
        return patterns
    
    async def detect_build_patterns(self, project_root: str) -> Dict[str, Any]:
        """
        Detect build and deployment patterns.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of build patterns
        """
        project_path = Path(project_root)
        patterns = {
            "build_tools": [],
            "package_managers": [],
            "ci_cd": [],
            "containerization": {},
            "deployment": {}
        }
        
        # Detect build tools
        build_indicators = {
            "webpack": ["webpack.config.js", "package.json:webpack"],
            "vite": ["vite.config.ts", "package.json:vite"],
            "rollup": ["rollup.config.js", "package.json:rollup"],
            "parcel": ["package.json:parcel"],
            "cargo": ["Cargo.toml"],
            "maven": ["pom.xml"],
            "gradle": ["build.gradle", "gradle.properties"],
            "make": ["Makefile"],
            "cmake": ["CMakeLists.txt"]
        }
        
        for tool, indicators in build_indicators.items():
            if self._check_indicators(project_path, indicators):
                patterns["build_tools"].append(tool)
        
        # Detect package managers
        pm_indicators = {
            "npm": ["package.json", "package-lock.json"],
            "yarn": ["yarn.lock"],
            "pnpm": ["pnpm-lock.yaml"],
            "pip": ["requirements.txt", "pyproject.toml"],
            "poetry": ["pyproject.toml", "poetry.lock"],
            "cargo": ["Cargo.lock"],
            "composer": ["composer.json"],
            "bundler": ["Gemfile"]
        }
        
        for pm, indicators in pm_indicators.items():
            if self._check_indicators(project_path, indicators):
                patterns["package_managers"].append(pm)
        
        # Detect CI/CD
        ci_indicators = {
            "github_actions": [".github/workflows/"],
            "gitlab_ci": [".gitlab-ci.yml"],
            "travis": [".travis.yml"],
            "circle_ci": [".circleci/"],
            "jenkins": ["Jenkinsfile"]
        }
        
        for ci, indicators in ci_indicators.items():
            if self._check_indicators(project_path, indicators):
                patterns["ci_cd"].append(ci)
        
        # Detect containerization
        if (project_path / "Dockerfile").exists():
            patterns["containerization"]["docker"] = True
        if (project_path / "docker-compose.yml").exists():
            patterns["containerization"]["docker_compose"] = True
        if (project_path / "k8s").exists() or (project_path / "kubernetes").exists():
            patterns["containerization"]["kubernetes"] = True
        
        return patterns
    
    async def analyze_file_structure(self, project_root: str) -> Dict[str, Any]:
        """
        Analyze overall file structure and organization.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary describing file structure
        """
        project_path = Path(project_root)
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "directory_depth": 0,
            "largest_directories": [],
            "file_size_distribution": {}
        }
        
        file_counts = {}
        dir_sizes = {}
        max_depth = 0
        
        for path in project_path.rglob("*"):
            if self._should_ignore_path(path):
                continue
                
            # Calculate depth
            depth = len(path.relative_to(project_path).parts)
            max_depth = max(max_depth, depth)
            
            if path.is_file():
                structure["total_files"] += 1
                
                # Count file types
                suffix = path.suffix.lower()
                file_counts[suffix] = file_counts.get(suffix, 0) + 1
                
                # Track directory sizes
                parent = str(path.parent.relative_to(project_path))
                dir_sizes[parent] = dir_sizes.get(parent, 0) + 1
                
            elif path.is_dir():
                structure["total_directories"] += 1
        
        structure["file_types"] = dict(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        structure["directory_depth"] = max_depth
        structure["largest_directories"] = sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return structure
    
    # Private helper methods
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored during analysis."""
        ignore_patterns = {
            "node_modules", ".git", "__pycache__", ".pytest_cache",
            "target", "build", "dist", ".next", ".nuxt",
            "coverage", ".coverage", "htmlcov"
        }
        
        path_parts = path.parts
        return any(pattern in path_parts for pattern in ignore_patterns)
    
    def _search_in_files(self, project_path: Path, search_term: str) -> bool:
        """Search for term in project files."""
        for file_path in project_path.rglob("*.{js,ts,py,json}"):
            if self._should_ignore_path(file_path):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if search_term in f.read():
                        return True
            except:
                pass
        return False
    
    def _check_indicators(self, project_path: Path, indicators: List[str]) -> bool:
        """Check if any indicators are present in the project."""
        for indicator in indicators:
            if ":" in indicator:
                file_name, content = indicator.split(":", 1)
                file_path = project_path / file_name
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if content in f.read():
                                return True
                    except:
                        pass
            elif indicator.endswith("/"):
                if (project_path / indicator.rstrip("/")).is_dir():
                    return True
            elif "*" in indicator:
                if list(project_path.rglob(indicator)):
                    return True
            else:
                if (project_path / indicator).exists():
                    return True
        return False
    
    async def _store_project_pattern(
        self, 
        project_root: str, 
        pattern_type: str, 
        pattern_data: Any
    ) -> None:
        """Store detected pattern in memory."""
        await self.domain_manager.store_project_pattern(
            pattern_type=pattern_type,
            framework=pattern_data.get("framework", "unknown") if isinstance(pattern_data, dict) else "unknown",
            language=pattern_data.get("language", "unknown") if isinstance(pattern_data, dict) else "unknown",
            structure=pattern_data if isinstance(pattern_data, dict) else {"data": pattern_data},
            metadata={
                "project_root": project_root,
                "detected_at": datetime.utcnow().isoformat()
            }
        )
```

## Phase 4: Session History & Context (Week 7-8)

### 4.1 Session Summary Generator
**New File**: `memory_mcp/autocode/session_analyzer.py`

```python
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CompletedTask:
    """Represents a completed task from a session."""
    description: str
    approach: str
    outcome: str
    patterns_used: List[str]
    files_affected: List[str]
    commands_executed: List[str]

@dataclass
class ArchitecturalDecision:
    """Represents an architectural decision made during a session."""
    decision: str
    reasoning: str
    alternatives_considered: List[str]
    context: Dict[str, Any]

@dataclass
class SessionSummary:
    """Complete session summary."""
    session_id: str
    start_time: str
    end_time: str
    tasks_completed: List[CompletedTask]
    files_modified: List[str]
    patterns_introduced: List[str]
    architectural_decisions: List[ArchitecturalDecision]
    issues_encountered: List[str]
    next_steps: List[str]
    project_context: Dict[str, Any]

class SessionAnalyzer:
    """
    Analyzes Claude conversations and generates comprehensive session summaries.
    
    This class processes conversation logs to extract:
    - Completed tasks and their approaches
    - Files that were modified or created
    - New patterns or conventions introduced
    - Architectural decisions and their reasoning
    - Issues encountered and how they were resolved
    - Suggested next steps
    """
    
    def __init__(self, domain_manager):
        self.domain_manager = domain_manager
        
        # Patterns for extracting information from conversation
        self.task_patterns = [
            r"I(?:'ve|'ll|\s+will|\s+have)\s+(implemented|created|built|added|fixed|updated|refactored)",
            r"(?:Task|Goal):\s*(.+?)(?:\n|$)",
            r"(?:Completed|Done|Finished):\s*(.+?)(?:\n|$)"
        ]
        
        self.file_patterns = [
            r"(?:Created|Modified|Updated|Edited)\s+(?:file\s+)?[`']?([^\s`']+\.[a-zA-Z0-9]+)[`']?",
            r"(?:File|Path):\s*[`']?([^\s`']+\.[a-zA-Z0-9]+)[`']?",
            r"[`']([^\s`']+\.[a-zA-Z0-9]+)[`']"
        ]
        
        self.decision_patterns = [
            r"(?:I chose|I decided|We decided|Decision|Approach):\s*(.+?)(?:\n|$)",
            r"(?:Reasoning|Because|Why):\s*(.+?)(?:\n|$)",
            r"(?:Alternative|Option|Could have):\s*(.+?)(?:\n|$)"
        ]
    
    async def generate_summary(self, messages: List[Dict[str, Any]]) -> SessionSummary:
        """
        Generate comprehensive session summary from conversation messages.
        
        Args:
            messages: List of conversation messages with role and content
            
        Returns:
            Complete session summary
        """
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract timing information
        start_time = datetime.utcnow().isoformat()  # Approximate
        end_time = datetime.utcnow().isoformat()
        
        # Combine all messages into analysis text
        conversation_text = self._combine_messages(messages)
        
        # Extract different types of information
        tasks_completed = await self.extract_completed_tasks(messages)
        files_modified = self._extract_modified_files(conversation_text)
        patterns_introduced = await self.identify_patterns_introduced(conversation_text, files_modified)
        architectural_decisions = self.track_architectural_decisions(messages)
        issues_encountered = self._extract_issues_and_resolutions(conversation_text)
        next_steps = self._extract_next_steps(conversation_text)
        project_context = await self._determine_project_context(files_modified)
        
        summary = SessionSummary(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            tasks_completed=tasks_completed,
            files_modified=files_modified,
            patterns_introduced=patterns_introduced,
            architectural_decisions=architectural_decisions,
            issues_encountered=issues_encountered,
            next_steps=next_steps,
            project_context=project_context
        )
        
        # Store summary in memory
        await self._store_session_summary(summary)
        
        return summary
    
    async def extract_completed_tasks(self, messages: List[Dict[str, Any]]) -> List[CompletedTask]:
        """
        Extract completed tasks from conversation messages.
        
        Args:
            messages: Conversation messages
            
        Returns:
            List of completed tasks
        """
        tasks = []
        current_task = None
        
        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                # Look for task indicators
                for pattern in self.task_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        if current_task:
                            tasks.append(current_task)
                        
                        current_task = CompletedTask(
                            description=match,
                            approach=self._extract_approach_from_context(content, match),
                            outcome="completed",  # Default, could be enhanced
                            patterns_used=self._extract_patterns_from_content(content),
                            files_affected=self._extract_files_from_content(content),
                            commands_executed=self._extract_commands_from_content(content)
                        )
        
        if current_task:
            tasks.append(current_task)
        
        return tasks
    
    async def identify_patterns_introduced(
        self, 
        conversation_text: str, 
        files_modified: List[str]
    ) -> List[str]:
        """
        Identify new patterns or conventions introduced in the session.
        
        Args:
            conversation_text: Full conversation text
            files_modified: List of modified files
            
        Returns:
            List of pattern names introduced
        """
        patterns = []
        
        # Look for explicit pattern mentions
        pattern_indicators = [
            r"(?:pattern|Pattern):\s*(.+?)(?:\n|$)",
            r"(?:I(?:'ll|'m)\s+use|Using|Implementing)\s+(.+?)\s+pattern",
            r"(?:Following|Applied|Added)\s+(.+?)\s+(?:pattern|convention|approach)"
        ]
        
        for pattern in pattern_indicators:
            matches = re.findall(pattern, conversation_text, re.IGNORECASE | re.MULTILINE)
            patterns.extend([match.strip() for match in matches])
        
        # Analyze modified files for architectural patterns
        for file_path in files_modified:
            if self._is_component_file(file_path):
                patterns.append("component_architecture")
            elif self._is_hook_file(file_path):
                patterns.append("custom_hooks")
            elif self._is_provider_file(file_path):
                patterns.append("provider_pattern")
            elif self._is_service_file(file_path):
                patterns.append("service_layer")
        
        return list(set(patterns))  # Remove duplicates
    
    def track_architectural_decisions(self, messages: List[Dict[str, Any]]) -> List[ArchitecturalDecision]:
        """
        Track architectural decisions and their reasoning.
        
        Args:
            messages: Conversation messages
            
        Returns:
            List of architectural decisions
        """
        decisions = []
        
        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                # Look for decision patterns
                decision_text = ""
                reasoning = ""
                alternatives = []
                
                for line in content.split('\n'):
                    if re.search(r"(?:I chose|I decided|Decision|Approach):", line, re.IGNORECASE):
                        decision_text = re.sub(r"(?:I chose|I decided|Decision|Approach):\s*", "", line, flags=re.IGNORECASE).strip()
                    elif re.search(r"(?:Reasoning|Because|Why):", line, re.IGNORECASE):
                        reasoning = re.sub(r"(?:Reasoning|Because|Why):\s*", "", line, flags=re.IGNORECASE).strip()
                    elif re.search(r"(?:Alternative|Option|Could have):", line, re.IGNORECASE):
                        alt = re.sub(r"(?:Alternative|Option|Could have):\s*", "", line, flags=re.IGNORECASE).strip()
                        alternatives.append(alt)
                
                if decision_text and reasoning:
                    decisions.append(ArchitecturalDecision(
                        decision=decision_text,
                        reasoning=reasoning,
                        alternatives_considered=alternatives,
                        context={"message_index": len(decisions)}
                    ))
        
        return decisions
    
    def _combine_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Combine all messages into a single text for analysis."""
        combined = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            combined.append(f"{role}: {content}")
        return "\n".join(combined)
    
    def _extract_modified_files(self, conversation_text: str) -> List[str]:
        """Extract list of files that were modified during the session."""
        files = set()
        
        for pattern in self.file_patterns:
            matches = re.findall(pattern, conversation_text)
            files.update(matches)
        
        # Filter out invalid file paths
        valid_files = []
        for file_path in files:
            if self._is_valid_file_path(file_path):
                valid_files.append(file_path)
        
        return sorted(valid_files)
    
    def _extract_approach_from_context(self, content: str, task: str) -> str:
        """Extract the approach used for a specific task."""
        # Look for approach indicators near the task mention
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if task.lower() in line.lower():
                # Look in surrounding lines for approach indicators
                context_lines = lines[max(0, i-2):i+3]
                for context_line in context_lines:
                    if re.search(r"(?:approach|method|solution|way):", context_line, re.IGNORECASE):
                        return context_line.strip()
        
        return "standard_implementation"
    
    def _extract_patterns_from_content(self, content: str) -> List[str]:
        """Extract patterns mentioned in content."""
        patterns = []
        pattern_keywords = [
            "pattern", "provider", "hook", "component", "service", 
            "factory", "singleton", "observer", "strategy", "adapter"
        ]
        
        for keyword in pattern_keywords:
            if keyword in content.lower():
                patterns.append(keyword)
        
        return patterns
    
    def _extract_files_from_content(self, content: str) -> List[str]:
        """Extract file paths from content."""
        files = set()
        for pattern in self.file_patterns:
            matches = re.findall(pattern, content)
            files.update(matches)
        return list(files)
    
    def _extract_commands_from_content(self, content: str) -> List[str]:
        """Extract bash commands from content."""
        commands = []
        
        # Look for code blocks with bash commands
        bash_pattern = r"```(?:bash|shell|sh)?\n(.*?)\n```"
        matches = re.findall(bash_pattern, content, re.DOTALL)
        
        for match in matches:
            commands.extend([cmd.strip() for cmd in match.split('\n') if cmd.strip()])
        
        # Look for inline commands
        inline_pattern = r"`([a-zA-Z][^`]*)`"
        inline_matches = re.findall(inline_pattern, content)
        
        for match in inline_matches:
            if self._looks_like_command(match):
                commands.append(match)
        
        return commands
    
    def _extract_issues_and_resolutions(self, conversation_text: str) -> List[str]:
        """Extract issues encountered and their resolutions."""
        issues = []
        
        issue_patterns = [
            r"(?:Error|Issue|Problem|Failed):\s*(.+?)(?:\n|$)",
            r"(?:Encountered|Found)\s+(?:an?\s+)?(?:error|issue|problem):\s*(.+?)(?:\n|$)",
            r"(?:Fixed|Resolved|Solved):\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in issue_patterns:
            matches = re.findall(pattern, conversation_text, re.IGNORECASE | re.MULTILINE)
            issues.extend([match.strip() for match in matches])
        
        return issues
    
    def _extract_next_steps(self, conversation_text: str) -> List[str]:
        """Extract suggested next steps."""
        next_steps = []
        
        step_patterns = [
            r"(?:Next|Next step|TODO|To do):\s*(.+?)(?:\n|$)",
            r"(?:Should|Need to|Could):\s*(.+?)(?:\n|$)",
            r"(?:Future|Later|Eventually):\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, conversation_text, re.IGNORECASE | re.MULTILINE)
            next_steps.extend([match.strip() for match in matches])
        
        return next_steps
    
    async def _determine_project_context(self, files_modified: List[str]) -> Dict[str, Any]:
        """Determine project context from modified files."""
        context = {
            "languages": set(),
            "frameworks": set(),
            "file_types": set()
        }
        
        for file_path in files_modified:
            # Determine language from extension
            extension = file_path.split('.')[-1].lower()
            language_map = {
                "ts": "typescript", "tsx": "typescript",
                "js": "javascript", "jsx": "javascript",
                "py": "python", "rs": "rust", "go": "go",
                "java": "java", "rb": "ruby"
            }
            
            if extension in language_map:
                context["languages"].add(language_map[extension])
            
            context["file_types"].add(extension)
            
            # Detect frameworks from file patterns
            if "component" in file_path.lower() or extension in ["tsx", "jsx"]:
                context["frameworks"].add("react")
            elif "pages" in file_path or "app" in file_path:
                context["frameworks"].add("nextjs")
            elif ".vue" in file_path:
                context["frameworks"].add("vue")
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) if isinstance(v, set) else v for k, v in context.items()}
    
    async def _store_session_summary(self, summary: SessionSummary) -> str:
        """Store session summary in memory."""
        summary_data = {
            "session_id": summary.session_id,
            "start_time": summary.start_time,
            "end_time": summary.end_time,
            "tasks_completed": [
                {
                    "description": task.description,
                    "approach": task.approach,
                    "outcome": task.outcome,
                    "patterns_used": task.patterns_used,
                    "files_affected": task.files_affected,
                    "commands_executed": task.commands_executed
                }
                for task in summary.tasks_completed
            ],
            "files_modified": summary.files_modified,
            "patterns_introduced": summary.patterns_introduced,
            "architectural_decisions": [
                {
                    "decision": decision.decision,
                    "reasoning": decision.reasoning,
                    "alternatives_considered": decision.alternatives_considered,
                    "context": decision.context
                }
                for decision in summary.architectural_decisions
            ],
            "issues_encountered": summary.issues_encountered,
            "next_steps": summary.next_steps,
            "project_context": summary.project_context
        }
        
        return await self.domain_manager.store_session_summary(
            session_id=summary.session_id,
            tasks_completed=summary_data["tasks_completed"],
            patterns_used=summary.patterns_introduced,
            files_modified=summary.files_modified,
            metadata={
                "architectural_decisions": summary_data["architectural_decisions"],
                "issues_encountered": summary.issues_encountered,
                "next_steps": summary.next_steps,
                "project_context": summary.project_context
            }
        )
    
    def _is_valid_file_path(self, file_path: str) -> bool:
        """Check if a string looks like a valid file path."""
        return (
            '.' in file_path and
            len(file_path) > 2 and
            not file_path.startswith('.') and
            ' ' not in file_path and
            file_path.count('.') <= 3
        )
    
    def _is_component_file(self, file_path: str) -> bool:
        """Check if file is a component file."""
        return (
            "component" in file_path.lower() or
            file_path.endswith((".tsx", ".jsx")) or
            "components/" in file_path.lower()
        )
    
    def _is_hook_file(self, file_path: str) -> bool:
        """Check if file is a custom hook."""
        return (
            file_path.startswith("use") or
            "hooks/" in file_path.lower() or
            "/use" in file_path.lower()
        )
    
    def _is_provider_file(self, file_path: str) -> bool:
        """Check if file is a provider."""
        return (
            "provider" in file_path.lower() or
            "context" in file_path.lower()
        )
    
    def _is_service_file(self, file_path: str) -> bool:
        """Check if file is a service layer file."""
        return (
            "service" in file_path.lower() or
            "api/" in file_path.lower() or
            "services/" in file_path.lower()
        )
    
    def _looks_like_command(self, text: str) -> bool:
        """Check if text looks like a bash command."""
        command_indicators = [
            "npm", "yarn", "git", "cd", "ls", "mkdir", "rm", "cp", "mv",
            "python", "node", "cargo", "docker", "kubectl", "pip", "curl"
        ]
        
        return any(text.startswith(cmd) for cmd in command_indicators)
```

### 4.2 Historical Context Retrieval
**New File**: `memory_mcp/autocode/history_navigator.py`

```python
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class HistoricalContext:
    """Historical context for current work."""
    similar_sessions: List[Dict[str, Any]]
    pattern_usage: List[Dict[str, Any]]
    architectural_decisions: List[Dict[str, Any]]
    known_issues: List[Dict[str, Any]]

class HistoryNavigator:
    """
    Navigate and search historical session data.
    
    This class provides intelligent access to historical context,
    helping Claude understand what was done before and why.
    """
    
    def __init__(self, domain_manager):
        self.domain_manager = domain_manager
    
    async def find_similar_work(
        self, 
        current_task: str,
        context: Dict[str, Any] = None,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find previous sessions with similar work.
        
        Args:
            current_task: Description of current task
            context: Current project context
            days_back: How many days back to search
            
        Returns:
            List of similar session summaries
        """
        # Search for similar session summaries
        query = f"task:{current_task}"
        if context and context.get("project_type"):
            query += f" project:{context['project_type']}"
        
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["session_summary"],
            limit=10,
            min_similarity=0.6,
            include_metadata=True
        )
        
        # Filter by time if needed
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        recent_memories = []
        
        for memory in memories:
            created_at = memory.get("created_at")
            if created_at:
                try:
                    memory_date = datetime.fromisoformat(created_at)
                    if memory_date >= cutoff_date:
                        recent_memories.append(memory)
                except:
                    recent_memories.append(memory)  # Include if can't parse date
            else:
                recent_memories.append(memory)
        
        return recent_memories
    
    async def get_pattern_history(
        self, 
        pattern_name: str,
        project_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history of pattern usage across sessions.
        
        Args:
            pattern_name: Name of the pattern to search for
            project_context: Current project context
            
        Returns:
            List of pattern usage instances
        """
        query = f"pattern:{pattern_name}"
        if project_context and project_context.get("language"):
            query += f" language:{project_context['language']}"
        
        # Search both session summaries and project patterns
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["session_summary", "project_pattern"],
            limit=15,
            min_similarity=0.5,
            include_metadata=True
        )
        
        pattern_usages = []
        for memory in memories:
            content = memory.get("content", {})
            
            if memory["type"] == "session_summary":
                if pattern_name in content.get("patterns_used", []):
                    pattern_usages.append({
                        "type": "session_usage",
                        "session_id": content.get("session_id"),
                        "tasks": content.get("tasks_completed", []),
                        "files": content.get("files_modified", []),
                        "timestamp": memory.get("created_at")
                    })
            elif memory["type"] == "project_pattern":
                if pattern_name in content.get("pattern_type", ""):
                    pattern_usages.append({
                        "type": "pattern_definition",
                        "framework": content.get("framework"),
                        "language": content.get("language"),
                        "structure": content.get("structure"),
                        "timestamp": memory.get("created_at")
                    })
        
        return pattern_usages
    
    async def explain_past_decisions(
        self, 
        component: str,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Explain reasoning behind past architectural decisions.
        
        Args:
            component: Component or area to explain
            context: Current context
            
        Returns:
            List of past decisions and their reasoning
        """
        query = f"decision:{component}"
        
        # Search session summaries for architectural decisions
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["session_summary"],
            limit=10,
            min_similarity=0.4,
            include_metadata=True
        )
        
        decisions = []
        for memory in memories:
            content = memory.get("content", {})
            metadata = memory.get("metadata", {})
            
            # Extract architectural decisions from metadata
            arch_decisions = metadata.get("architectural_decisions", [])
            for decision in arch_decisions:
                if component.lower() in decision.get("decision", "").lower():
                    decisions.append({
                        "decision": decision.get("decision"),
                        "reasoning": decision.get("reasoning"),
                        "alternatives": decision.get("alternatives_considered", []),
                        "session_id": content.get("session_id"),
                        "timestamp": memory.get("created_at"),
                        "context": decision.get("context", {})
                    })
        
        return decisions
    
    async def get_known_issues(
        self, 
        technology: str,
        error_pattern: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get known issues and their resolutions for a technology.
        
        Args:
            technology: Technology or framework name
            error_pattern: Specific error pattern to search for
            
        Returns:
            List of known issues and resolutions
        """
        query = f"technology:{technology}"
        if error_pattern:
            query += f" error:{error_pattern}"
        
        # Search session summaries for issues
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["session_summary"],
            limit=15,
            min_similarity=0.4,
            include_metadata=True
        )
        
        issues = []
        for memory in memories:
            content = memory.get("content", {})
            metadata = memory.get("metadata", {})
            
            # Extract issues from metadata
            encountered_issues = metadata.get("issues_encountered", [])
            for issue in encountered_issues:
                if technology.lower() in issue.lower():
                    issues.append({
                        "issue": issue,
                        "session_id": content.get("session_id"),
                        "timestamp": memory.get("created_at"),
                        "context": content.get("project_context", {}),
                        "resolution_approach": self._extract_resolution_from_tasks(
                            content.get("tasks_completed", []), 
                            issue
                        )
                    })
        
        return issues
    
    async def get_command_success_history(
        self, 
        command_pattern: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get success history for command patterns.
        
        Args:
            command_pattern: Command pattern to analyze
            context: Current context
            
        Returns:
            Command success analysis
        """
        query = f"command:{command_pattern}"
        if context and context.get("platform"):
            query += f" platform:{context['platform']}"
        
        # Search bash execution memories
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["bash_execution", "command_pattern"],
            limit=50,
            min_similarity=0.3,
            include_metadata=True
        )
        
        total_executions = 0
        successful_executions = 0
        common_failures = []
        successful_variants = []
        
        for memory in memories:
            content = memory.get("content", {})
            
            if memory["type"] == "bash_execution":
                total_executions += 1
                if content.get("success", False):
                    successful_executions += 1
                    successful_variants.append(content.get("command"))
                else:
                    common_failures.append({
                        "command": content.get("command"),
                        "error": content.get("output", "")[:200],
                        "context": content.get("context", {})
                    })
            elif memory["type"] == "command_pattern":
                # Use stored success rate
                success_rate = content.get("success_rate", 0.5)
                return {
                    "pattern": command_pattern,
                    "stored_success_rate": success_rate,
                    "recommended_command": content.get("command"),
                    "context": content.get("context", {})
                }
        
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        return {
            "pattern": command_pattern,
            "success_rate": success_rate,
            "total_executions": total_executions,
            "successful_variants": list(set(successful_variants)),
            "common_failures": common_failures[:5],  # Top 5 failures
            "recommendation": self._get_command_recommendation(successful_variants)
        }
    
    async def get_session_context(
        self, 
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete context for a specific session.
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            Complete session context or None if not found
        """
        query = f"session_id:{session_id}"
        
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["session_summary"],
            limit=1,
            min_similarity=0.9,
            include_metadata=True
        )
        
        if memories:
            return memories[0]
        
        return None
    
    async def get_related_work(
        self, 
        file_path: str,
        task_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get work related to a specific file or component.
        
        Args:
            file_path: File path to search for
            task_type: Type of task being performed
            
        Returns:
            List of related work sessions
        """
        # Extract component name from file path
        component_name = file_path.split('/')[-1].split('.')[0]
        
        query = f"file:{component_name}"
        if task_type:
            query += f" task:{task_type}"
        
        memories = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["session_summary"],
            limit=10,
            min_similarity=0.4,
            include_metadata=True
        )
        
        related_work = []
        for memory in memories:
            content = memory.get("content", {})
            files_modified = content.get("files_modified", [])
            
            # Check if any modified files are related
            for modified_file in files_modified:
                if (component_name in modified_file or 
                    any(part in modified_file for part in file_path.split('/'))):
                    related_work.append({
                        "session_id": content.get("session_id"),
                        "tasks": content.get("tasks_completed", []),
                        "files": files_modified,
                        "timestamp": memory.get("created_at"),
                        "relevance": self._calculate_file_relevance(file_path, modified_file)
                    })
                    break
        
        # Sort by relevance
        related_work.sort(key=lambda x: x["relevance"], reverse=True)
        
        return related_work
    
    # Private helper methods
    def _extract_resolution_from_tasks(
        self, 
        tasks: List[Dict[str, Any]], 
        issue: str
    ) -> Optional[str]:
        """Extract how an issue was resolved from completed tasks."""
        issue_keywords = issue.lower().split()
        
        for task in tasks:
            task_desc = task.get("description", "").lower()
            if any(keyword in task_desc for keyword in issue_keywords):
                return task.get("approach", "unknown_approach")
        
        return None
    
    def _get_command_recommendation(self, successful_variants: List[str]) -> str:
        """Get command recommendation from successful variants."""
        if not successful_variants:
            return "No successful variants found"
        
        # Return most common successful variant
        from collections import Counter
        variant_counts = Counter(successful_variants)
        return variant_counts.most_common(1)[0][0]
    
    def _calculate_file_relevance(self, target_file: str, related_file: str) -> float:
        """Calculate relevance score between two file paths."""
        target_parts = target_file.split('/')
        related_parts = related_file.split('/')
        
        # Exact match
        if target_file == related_file:
            return 1.0
        
        # Same file name
        if target_parts[-1] == related_parts[-1]:
            return 0.9
        
        # Same directory
        if target_parts[:-1] == related_parts[:-1]:
            return 0.7
        
        # Shared directory components
        shared_dirs = len(set(target_parts[:-1]) & set(related_parts[:-1]))
        total_dirs = len(set(target_parts[:-1]) | set(related_parts[:-1]))
        
        if total_dirs > 0:
            return 0.3 + (shared_dirs / total_dirs) * 0.4
        
        return 0.1
```

## Phase 5: Integration & MCP Hooks (Week 9-10)

### 5.1 Automatic Triggering System
**New File**: `memory_mcp/autocode/hooks.py`

```python
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

class AutoCodeHooks:
    """
    Hooks that automatically trigger indexing during normal Claude operations.
    
    This class provides seamless integration with Claude's normal workflow,
    automatically capturing and learning from interactions without user intervention.
    """
    
    def __init__(self, domain_manager):
        self.domain_manager = domain_manager
        self.session_data = {
            "files_accessed": [],
            "commands_executed": [],
            "start_time": datetime.utcnow(),
            "conversation_log": []
        }
        self.project_cache = {}
    
    async def on_file_read(
        self, 
        file_path: str, 
        content: str,
        operation: str = "read"
    ) -> None:
        """
        Automatically triggered when Claude reads files.
        
        Args:
            file_path: Path to the file that was read
            content: File content (if available)
            operation: Type of operation (read, write, edit)
        """
        # Track file access
        self.session_data["files_accessed"].append({
            "path": file_path,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Extract project context if this is the first file in a new project
        project_root = self._detect_project_root(file_path)
        if project_root and project_root not in self.project_cache:
            await self._cache_project_context(project_root)
        
        # Process file for pattern extraction
        if content and self._should_analyze_file(file_path):
            await self.domain_manager.autocode_domain.process_file_access(
                file_path, content, operation
            )
    
    async def on_bash_execution(
        self, 
        command: str, 
        exit_code: int,
        output: str = "",
        working_directory: str = None
    ) -> None:
        """
        Automatically triggered on bash command execution.
        
        Args:
            command: The bash command that was executed
            exit_code: Exit code from the command
            output: Command output
            working_directory: Directory where command was executed
        """
        # Track command execution
        execution_record = {
            "command": command,
            "exit_code": exit_code,
            "output": output[:500] if output else "",
            "working_directory": working_directory,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.session_data["commands_executed"].append(execution_record)
        
        # Get context for command learning
        context = await self._get_command_context(working_directory)
        
        # Process command for learning
        await self.domain_manager.autocode_domain.process_bash_execution(
            command, exit_code, output, context
        )
        
        # If this was a project setup command, trigger project scan
        if self._is_project_setup_command(command) and exit_code == 0:
            project_root = working_directory or os.getcwd()
            await self._trigger_project_scan(project_root)
    
    async def on_conversation_message(
        self, 
        role: str, 
        content: str,
        message_id: str = None
    ) -> None:
        """
        Track conversation messages for session analysis.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            message_id: Unique message identifier
        """
        self.session_data["conversation_log"].append({
            "role": role,
            "content": content,
            "message_id": message_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def on_conversation_end(self, conversation_id: str = None) -> None:
        """
        Automatically triggered at conversation end to generate session summary.
        
        Args:
            conversation_id: Unique conversation identifier
        """
        if len(self.session_data["conversation_log"]) > 3:  # Only for substantial conversations
            # Generate session summary
            session_analyzer = self.domain_manager.autocode_domain.session_analyzer
            if hasattr(self.domain_manager.autocode_domain, 'session_analyzer'):
                summary = await session_analyzer.generate_summary(
                    self.session_data["conversation_log"]
                )
                
                # Store additional session metadata
                await self._store_session_metadata(summary.session_id, conversation_id)
        
        # Reset session data for next conversation
        self._reset_session_data()
    
    async def on_project_detection(self, project_root: str) -> None:
        """
        Triggered when a new project is detected.
        
        Args:
            project_root: Root directory of the detected project
        """
        # Trigger comprehensive project scan
        pattern_detector = self.domain_manager.autocode_domain.pattern_detector
        if hasattr(self.domain_manager.autocode_domain, 'pattern_detector'):
            await pattern_detector.scan_project(project_root)
        
        # Cache project context
        await self._cache_project_context(project_root)
    
    async def on_error_encountered(
        self, 
        error_type: str, 
        error_message: str,
        context: Dict[str, Any] = None
    ) -> None:
        """
        Track errors for learning and pattern recognition.
        
        Args:
            error_type: Type of error (syntax, runtime, build, etc.)
            error_message: Error message content
            context: Additional context about the error
        """
        error_record = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store error information for later analysis
        await self.domain_manager.store_memory(
            memory_type="fact",
            content={
                "fact": f"Error encountered: {error_type}",
                "error_details": error_record,
                "confidence": 0.9
            },
            importance=0.6,
            metadata={
                "category": "error_tracking",
                "project_context": await self._get_current_project_context()
            }
        )
    
    async def suggest_next_action(
        self, 
        current_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest next action based on accumulated knowledge.
        
        Args:
            current_context: Current working context
            
        Returns:
            Suggested action or None
        """
        # Analyze current context against historical patterns
        history_navigator = self.domain_manager.autocode_domain.history_navigator
        if hasattr(self.domain_manager.autocode_domain, 'history_navigator'):
            
            # Look for similar past work
            similar_work = await history_navigator.find_similar_work(
                current_task=current_context.get("current_task", ""),
                context=current_context
            )
            
            if similar_work:
                # Extract next steps from similar work
                for work in similar_work[:3]:  # Check top 3 similar sessions
                    content = work.get("content", {})
                    metadata = work.get("metadata", {})
                    next_steps = metadata.get("next_steps", [])
                    
                    if next_steps:
                        return {
                            "suggestion": next_steps[0],
                            "confidence": work.get("similarity", 0.5),
                            "based_on": f"Session {content.get('session_id')}",
                            "alternatives": next_steps[1:3]
                        }
        
        return None
    
    # Private helper methods
    def _detect_project_root(self, file_path: str) -> Optional[str]:
        """Detect project root from file path."""
        path = Path(file_path)
        
        # Look for common project indicators
        indicators = [
            "package.json", "Cargo.toml", "pyproject.toml", 
            "requirements.txt", "pom.xml", "build.gradle",
            ".git", "composer.json", "Gemfile"
        ]
        
        current_path = path.parent if path.is_file() else path
        
        while current_path != current_path.parent:
            for indicator in indicators:
                if (current_path / indicator).exists():
                    return str(current_path)
            current_path = current_path.parent
        
        return None
    
    async def _cache_project_context(self, project_root: str) -> None:
        """Cache project context for quick access."""
        if project_root in self.project_cache:
            return
        
        # Basic project detection
        project_path = Path(project_root)
        context = {
            "root": project_root,
            "framework": await self._detect_framework_quick(project_path),
            "language": await self._detect_language_quick(project_path),
            "cached_at": datetime.utcnow().isoformat()
        }
        
        self.project_cache[project_root] = context
    
    async def _detect_framework_quick(self, project_path: Path) -> str:
        """Quick framework detection."""
        if (project_path / "package.json").exists():
            try:
                import json
                with open(project_path / "package.json") as f:
                    package_data = json.load(f)
                    deps = {**package_data.get("dependencies", {}), 
                           **package_data.get("devDependencies", {})}
                    
                    if "react" in deps:
                        return "react"
                    elif "vue" in deps:
                        return "vue"
                    elif "@angular/core" in deps:
                        return "angular"
                    elif "express" in deps:
                        return "express"
            except:
                pass
        
        if (project_path / "Cargo.toml").exists():
            return "rust"
        elif (project_path / "requirements.txt").exists() or (project_path / "pyproject.toml").exists():
            return "python"
        elif (project_path / "pom.xml").exists():
            return "java"
        elif (project_path / "Gemfile").exists():
            return "rails"
        
        return "unknown"
    
    async def _detect_language_quick(self, project_path: Path) -> str:
        """Quick language detection."""
        extensions = {
            ".ts": "typescript", ".tsx": "typescript",
            ".js": "javascript", ".jsx": "javascript",
            ".py": "python", ".rs": "rust",
            ".java": "java", ".rb": "ruby",
            ".go": "go", ".php": "php"
        }
        
        counts = {}
        for file_path in project_path.rglob("*"):
            if file_path.is_file() and not self._should_ignore_path(file_path):
                ext = file_path.suffix.lower()
                if ext in extensions:
                    lang = extensions[ext]
                    counts[lang] = counts.get(lang, 0) + 1
        
        if counts:
            return max(counts, key=counts.get)
        
        return "unknown"
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """Check if file should be analyzed for patterns."""
        analyze_extensions = {
            ".ts", ".tsx", ".js", ".jsx", ".py", ".rs", 
            ".java", ".rb", ".go", ".php", ".vue", ".svelte"
        }
        
        path = Path(file_path)
        return (
            path.suffix.lower() in analyze_extensions and
            not self._should_ignore_path(path)
        )
    
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored."""
        ignore_patterns = {
            "node_modules", ".git", "__pycache__", ".pytest_cache",
            "target", "build", "dist", ".next", ".nuxt", "coverage"
        }
        
        return any(pattern in path.parts for pattern in ignore_patterns)
    
    async def _get_command_context(self, working_directory: str = None) -> Dict[str, Any]:
        """Get context for command execution."""
        context = {
            "platform": os.name,
            "working_directory": working_directory or os.getcwd(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add project context if available
        if working_directory:
            project_root = self._detect_project_root(working_directory)
            if project_root and project_root in self.project_cache:
                project_context = self.project_cache[project_root]
                context.update({
                    "project_root": project_root,
                    "project_framework": project_context.get("framework"),
                    "project_language": project_context.get("language")
                })
        
        return context
    
    def _is_project_setup_command(self, command: str) -> bool:
        """Check if command is a project setup command."""
        setup_commands = [
            "npm init", "yarn init", "cargo init", "git init",
            "pip install", "npm install", "yarn install",
            "cargo build", "mvn install", "gradle build"
        ]
        
        return any(command.startswith(setup_cmd) for setup_cmd in setup_commands)
    
    async def _trigger_project_scan(self, project_root: str) -> None:
        """Trigger a project scan if not recently done."""
        # Check if project was recently scanned
        query = f"project_root:{project_root}"
        recent_scans = await self.domain_manager.retrieve_memories(
            query=query,
            memory_types=["project_pattern"],
            limit=1,
            min_similarity=0.9
        )
        
        # Only scan if no recent scan found
        if not recent_scans:
            await self.on_project_detection(project_root)
    
    async def _store_session_metadata(
        self, 
        session_id: str, 
        conversation_id: str = None
    ) -> None:
        """Store additional session metadata."""
        metadata = {
            "conversation_id": conversation_id,
            "files_accessed_count": len(self.session_data["files_accessed"]),
            "commands_executed_count": len(self.session_data["commands_executed"]),
            "session_duration": (
                datetime.utcnow() - self.session_data["start_time"]
            ).total_seconds(),
            "projects_involved": list(self.project_cache.keys())
        }
        
        await self.domain_manager.store_memory(
            memory_type="fact",
            content={
                "fact": f"Session metadata for {session_id}",
                "metadata": metadata
            },
            importance=0.5,
            metadata={"category": "session_metadata"}
        )
    
    async def _get_current_project_context(self) -> Dict[str, Any]:
        """Get current project context."""
        if self.project_cache:
            # Return most recently accessed project context
            return list(self.project_cache.values())[-1]
        
        return {"framework": "unknown", "language": "unknown"}
    
    def _reset_session_data(self) -> None:
        """Reset session data for next conversation."""
        self.session_data = {
            "files_accessed": [],
            "commands_executed": [],
            "start_time": datetime.utcnow(),
            "conversation_log": []
        }
```

### 5.2 Enhanced Server Integration
**File**: `memory_mcp/mcp/server.py`

Update existing server to integrate AutoCode hooks:

```python
# Add imports
from memory_mcp.autocode.hooks import AutoCodeHooks
from memory_mcp.autocode.command_learner import CommandLearner
from memory_mcp.autocode.pattern_detector import PatternDetector
from memory_mcp.autocode.session_analyzer import SessionAnalyzer
from memory_mcp.autocode.history_navigator import HistoryNavigator

class MemoryMCPServer:
    def __init__(self):
        # ... existing initialization ...
        
        # Initialize AutoCode components
        self.autocode_hooks = AutoCodeHooks(self.domain_manager)
        self.command_learner = CommandLearner(self.domain_manager)
        self.pattern_detector = PatternDetector(self.domain_manager)
        self.session_analyzer = SessionAnalyzer(self.domain_manager)
        self.history_navigator = HistoryNavigator(self.domain_manager)
        
        # Add AutoCode components to domain
        self.domain_manager.autocode_domain.command_learner = self.command_learner
        self.domain_manager.autocode_domain.pattern_detector = self.pattern_detector
        self.domain_manager.autocode_domain.session_analyzer = self.session_analyzer
        self.domain_manager.autocode_domain.history_navigator = self.history_navigator
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict) -> Dict[str, Any]:
        """Handle MCP tool calls with AutoCode integration."""
        
        # Track file operations automatically
        if tool_name in ["read_file", "write_file", "edit_file"]:
            file_path = arguments.get("file_path") or arguments.get("path")
            content = arguments.get("content", "")
            operation = {
                "read_file": "read",
                "write_file": "write", 
                "edit_file": "edit"
            }.get(tool_name, "unknown")
            
            if file_path:
                await self.autocode_hooks.on_file_read(file_path, content, operation)
        
        # Track bash executions automatically
        elif tool_name == "bash":
            command = arguments.get("command", "")
            # Note: We'll need to capture the result to track exit_code
            result = await self._execute_bash_command(arguments)
            
            # Extract exit code and output from result
            exit_code = result.get("exit_code", 0)
            output = result.get("output", "")
            working_dir = arguments.get("working_directory")
            
            await self.autocode_hooks.on_bash_execution(
                command, exit_code, output, working_dir
            )
            
            return result
        
        # AutoCode-specific tools
        elif tool_name == "suggest_command":
            intent = arguments.get("intent")
            context = arguments.get("context", {})
            
            suggestions = await self.command_learner.suggest_command(intent, context)
            
            return {
                "suggestions": suggestions,
                "context": context
            }
        
        elif tool_name == "track_bash":
            command = arguments.get("command")
            exit_code = arguments.get("exit_code")
            output = arguments.get("output", "")
            context = arguments.get("context", {})
            
            await self.command_learner.track_bash_execution(
                command, exit_code, output, context
            )
            
            return {"status": "tracked"}
        
        elif tool_name == "get_session_history":
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            days_back = arguments.get("days_back", 30)
            
            history = await self.history_navigator.find_similar_work(
                current_task=query,
                days_back=days_back
            )
            
            return {
                "sessions": history[:limit],
                "total_found": len(history)
            }
        
        elif tool_name == "get_project_patterns":
            project_path = arguments.get("project_path")
            pattern_types = arguments.get("pattern_types", [])
            
            if not os.path.exists(project_path):
                return {"error": f"Project path {project_path} does not exist"}
            
            patterns = await self.pattern_detector.scan_project(project_path)
            
            # Filter by requested pattern types if specified
            if pattern_types:
                filtered_patterns = {
                    k: v for k, v in patterns.items() 
                    if k in pattern_types
                }
                return {"patterns": filtered_patterns}
            
            return {"patterns": patterns}
        
        elif tool_name == "explain_decision":
            component = arguments.get("component")
            context = arguments.get("context", {})
            
            decisions = await self.history_navigator.explain_past_decisions(
                component, context
            )
            
            return {
                "decisions": decisions,
                "component": component
            }
        
        elif tool_name == "get_command_history":
            command_pattern = arguments.get("command_pattern")
            context = arguments.get("context", {})
            
            history = await self.history_navigator.get_command_success_history(
                command_pattern, context
            )
            
            return history
        
        # Delegate to existing tool handling
        else:
            return await self._handle_existing_tool(tool_name, arguments)
    
    async def handle_conversation_message(
        self, 
        role: str, 
        content: str,
        message_id: str = None
    ) -> None:
        """Handle conversation messages for session tracking."""
        await self.autocode_hooks.on_conversation_message(role, content, message_id)
    
    async def handle_conversation_end(self, conversation_id: str = None) -> None:
        """Handle conversation end for session summary generation."""
        await self.autocode_hooks.on_conversation_end(conversation_id)
    
    # Add new tool definitions to existing list
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions including AutoCode tools."""
        existing_tools = self._get_existing_tool_definitions()
        
        autocode_tools = [
            {
                "name": "suggest_command",
                "description": "Get command suggestions based on intent and context",
                "schema": self.tool_definitions.suggest_command_schema
            },
            {
                "name": "track_bash", 
                "description": "Track bash command execution for learning",
                "schema": self.tool_definitions.track_bash_schema
            },
            {
                "name": "get_session_history",
                "description": "Search historical session data",
                "schema": self.tool_definitions.get_session_history_schema
            },
            {
                "name": "get_project_patterns",
                "description": "Get detected patterns for a project",
                "schema": self.tool_definitions.get_project_patterns_schema
            },
            {
                "name": "explain_decision",
                "description": "Explain past architectural decisions",
                "schema": {
                    "type": "object",
                    "properties": {
                        "component": {
                            "type": "string",
                            "description": "Component or area to explain"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current context"
                        }
                    },
                    "required": ["component"]
                }
            },
            {
                "name": "get_command_history",
                "description": "Get success history for command patterns",
                "schema": {
                    "type": "object",
                    "properties": {
                        "command_pattern": {
                            "type": "string",
                            "description": "Command pattern to analyze"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current context"
                        }
                    },
                    "required": ["command_pattern"]
                }
            }
        ]
        
        return existing_tools + autocode_tools
```

## Implementation Timeline

### Week 1-2: Foundation
- [ ] Implement new memory types in schema.py
- [ ] Create AutoCode domain structure
- [ ] Integrate with existing MemoryDomainManager
- [ ] Basic file access tracking

### Week 3-4: Command Intelligence
- [ ] Implement CommandLearner class
- [ ] Add bash execution tracking
- [ ] Create command suggestion system
- [ ] Add MCP tools for command intelligence

### Week 5-6: Pattern Recognition
- [ ] Implement PatternDetector class
- [ ] Add framework/language detection
- [ ] Create architectural pattern recognition
- [ ] Add project scanning capabilities

### Week 7-8: Session & History
- [ ] Implement SessionAnalyzer class
- [ ] Create HistoryNavigator class
- [ ] Add session summary generation
- [ ] Build historical context retrieval

### Week 9-10: Integration & Hooks
- [ ] Implement AutoCodeHooks class
- [ ] Integrate with MCP server
- [ ] Add automatic triggering
- [ ] Test end-to-end functionality

### Week 11-12: Polish & Testing
- [ ] Add comprehensive error handling
- [ ] Optimize performance
- [ ] Add configuration options
- [ ] Create documentation and examples

## Configuration

Add to `memory_mcp/utils/config.py`:

```python
AUTOCODE_CONFIG = {
    "enabled": True,
    "auto_scan_projects": True,
    "track_bash_commands": True,
    "generate_session_summaries": True,
    "command_learning": {
        "enabled": True,
        "min_confidence_threshold": 0.3,
        "max_suggestions": 5,
        "track_failures": True
    },
    "pattern_detection": {
        "enabled": True,
        "supported_languages": ["typescript", "javascript", "python", "rust", "go", "java"],
        "max_scan_depth": 5,
        "ignore_patterns": ["node_modules", ".git", "__pycache__", "target", "build"]
    },
    "session_analysis": {
        "enabled": True,
        "min_conversation_length": 3,
        "track_architectural_decisions": True,
        "extract_next_steps": True
    },
    "history_retention": {
        "session_summaries_days": 90,
        "command_patterns_days": 30,
        "project_patterns_days": 365
    }
}
```

## Expected Benefits

1. **Immediate Impact**
   - Eliminate repeated command failures (rm vs rm -f)
   - Remember existing project components and patterns
   - Reduce context switching between sessions

2. **Medium-term Benefits**
   - Faster onboarding to new projects
   - Consistent architectural patterns across projects
   - Intelligent command suggestions based on context

3. **Long-term Value**
   - Cross-project learning and pattern application
   - Accumulated knowledge base of successful approaches
   - Reduced need for repetitive explanations

This plan provides a comprehensive, phased approach to implementing AutoCodeIndex as an extension to your existing MCP Persistent Memory server, creating a truly intelligent coding assistant that learns and improves with every interaction.