"""
AutoCode hooks for automatic triggering during Claude operations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger


class AutoCodeHooks:
    """
    Hooks that automatically trigger indexing during normal Claude operations.
    
    This class provides seamless integration with Claude's normal workflow,
    automatically capturing and learning from interactions without user intervention.
    """
    
    def __init__(self, domain_manager):
        """
        Initialize AutoCode hooks.
        
        Args:
            domain_manager: The memory domain manager instance
        """
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
        content: str = "",
        operation: str = "read"
    ) -> None:
        """
        Automatically triggered when Claude reads files.
        
        Args:
            file_path: Path to the file that was read
            content: File content (if available)
            operation: Type of operation (read, write, edit)
        """
        try:
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
            
            # Process file for pattern extraction if content is available
            if content and self._should_analyze_file(file_path):
                await self.domain_manager.autocode_domain.process_file_access(
                    file_path, content, operation
                )
                
            logger.debug(f"AutoCode: Tracked file access {file_path} ({operation})")
            
        except Exception as e:
            logger.error(f"AutoCode: Error tracking file access {file_path}: {e}")
    
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
        try:
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
                
            logger.debug(f"AutoCode: Tracked bash execution {command} (exit: {exit_code})")
            
        except Exception as e:
            logger.error(f"AutoCode: Error tracking bash execution {command}: {e}")
    
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
        try:
            self.session_data["conversation_log"].append({
                "role": role,
                "content": content,
                "message_id": message_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"AutoCode: Error tracking conversation message: {e}")
    
    async def on_conversation_end(self, conversation_id: str = None) -> None:
        """
        Automatically triggered at conversation end to generate session summary.
        
        Args:
            conversation_id: Unique conversation identifier
        """
        try:
            # Only generate summary for substantial conversations
            if len(self.session_data["conversation_log"]) > 3:
                summary_id = await self.domain_manager.autocode_domain.generate_session_summary(
                    self.session_data["conversation_log"]
                )
                
                if summary_id:
                    logger.info(f"AutoCode: Generated session summary {summary_id}")
            
            # Reset session data for next conversation
            self._reset_session_data()
            
        except Exception as e:
            logger.error(f"AutoCode: Error generating session summary: {e}")
    
    async def on_project_detection(self, project_root: str) -> None:
        """
        Triggered when a new project is detected.
        
        Args:
            project_root: Root directory of the detected project
        """
        try:
            # Cache project context
            await self._cache_project_context(project_root)
            
            logger.info(f"AutoCode: Detected new project at {project_root}")
            
        except Exception as e:
            logger.error(f"AutoCode: Error processing project detection {project_root}: {e}")
    
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
        try:
            # Look for similar past work in session summaries
            memories = await self.domain_manager.retrieve_memories(
                query=current_context.get("current_task", ""),
                memory_types=["session_summary"],
                limit=3,
                min_similarity=0.6
            )
            
            if memories:
                # Extract next steps from similar work
                for memory in memories:
                    content = memory.get("content", {})
                    metadata = memory.get("metadata", {})
                    next_steps = metadata.get("next_steps", [])
                    
                    if next_steps:
                        return {
                            "suggestion": next_steps[0],
                            "confidence": memory.get("similarity", 0.5),
                            "based_on": f"Session {content.get('session_id')}",
                            "alternatives": next_steps[1:3]
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"AutoCode: Error suggesting next action: {e}")
            return None
    
    async def suggest_command(
        self, 
        intent: str, 
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest commands for a given intent using the command learner.
        
        Args:
            intent: What the user wants to accomplish
            context: Current context
            
        Returns:
            List of command suggestions
        """
        try:
            # Get current context if not provided
            if not context:
                context = await self._get_current_context()
            
            # Use the AutoCode domain's command suggestion
            return await self.domain_manager.autocode_domain.suggest_command(intent, context)
            
        except Exception as e:
            logger.error(f"AutoCode: Error suggesting command for '{intent}': {e}")
            return []
    
    async def get_command_success_rate(
        self, 
        command: str, 
        context: Dict[str, Any] = None
    ) -> float:
        """
        Get success rate for a command in given context.
        
        Args:
            command: Command to check
            context: Current context
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        try:
            if not context:
                context = await self._get_current_context()
            
            # Use command learner if available
            command_learner = getattr(self.domain_manager.autocode_domain, 'command_learner', None)
            if command_learner:
                return await command_learner.get_success_rate(command, context)
            
            # Fallback to basic calculation
            return 0.5
            
        except Exception as e:
            logger.error(f"AutoCode: Error getting success rate for '{command}': {e}")
            return 0.5
    
    # Private helper methods
    def _detect_project_root(self, file_path: str) -> Optional[str]:
        """Detect project root from file path."""
        try:
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
            
        except Exception as e:
            logger.error(f"AutoCode: Error detecting project root for {file_path}: {e}")
            return None
    
    async def _cache_project_context(self, project_root: str) -> None:
        """Cache project context for quick access."""
        try:
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
            
        except Exception as e:
            logger.error(f"AutoCode: Error caching project context {project_root}: {e}")
    
    async def _detect_framework_quick(self, project_path: Path) -> str:
        """Quick framework detection."""
        try:
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
            
        except Exception as e:
            logger.error(f"AutoCode: Error detecting framework for {project_path}: {e}")
            return "unknown"
    
    async def _detect_language_quick(self, project_path: Path) -> str:
        """Quick language detection."""
        try:
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
            
        except Exception as e:
            logger.error(f"AutoCode: Error detecting language for {project_path}: {e}")
            return "unknown"
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """Check if file should be analyzed for patterns."""
        try:
            analyze_extensions = {
                ".ts", ".tsx", ".js", ".jsx", ".py", ".rs", 
                ".java", ".rb", ".go", ".php", ".vue", ".svelte"
            }
            
            path = Path(file_path)
            return (
                path.suffix.lower() in analyze_extensions and
                not self._should_ignore_path(path)
            )
            
        except Exception as e:
            logger.error(f"AutoCode: Error checking if should analyze {file_path}: {e}")
            return False
    
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored."""
        try:
            ignore_patterns = {
                "node_modules", ".git", "__pycache__", ".pytest_cache",
                "target", "build", "dist", ".next", ".nuxt", "coverage"
            }
            
            return any(pattern in path.parts for pattern in ignore_patterns)
            
        except Exception as e:
            logger.error(f"AutoCode: Error checking ignore path {path}: {e}")
            return True  # Err on side of caution
    
    async def _get_command_context(self, working_directory: str = None) -> Dict[str, Any]:
        """Get context for command execution."""
        try:
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
            
        except Exception as e:
            logger.error(f"AutoCode: Error getting command context: {e}")
            return {"platform": "unknown", "timestamp": datetime.utcnow().isoformat()}
    
    def _is_project_setup_command(self, command: str) -> bool:
        """Check if command is a project setup command."""
        try:
            setup_commands = [
                "npm init", "yarn init", "cargo init", "git init",
                "pip install", "npm install", "yarn install",
                "cargo build", "mvn install", "gradle build"
            ]
            
            return any(command.startswith(setup_cmd) for setup_cmd in setup_commands)
            
        except Exception as e:
            logger.error(f"AutoCode: Error checking setup command {command}: {e}")
            return False
    
    async def _trigger_project_scan(self, project_root: str) -> None:
        """Trigger a project scan if not recently done."""
        try:
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
                
        except Exception as e:
            logger.error(f"AutoCode: Error triggering project scan for {project_root}: {e}")
    
    def _reset_session_data(self) -> None:
        """Reset session data for next conversation."""
        try:
            self.session_data = {
                "files_accessed": [],
                "commands_executed": [],
                "start_time": datetime.utcnow(),
                "conversation_log": []
            }
            
        except Exception as e:
            logger.error(f"AutoCode: Error resetting session data: {e}")
    
    async def _get_current_context(self) -> Dict[str, Any]:
        """Get current context for command suggestions."""
        try:
            context = {
                "platform": os.name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add project context if available
            if self.project_cache:
                # Use most recent project context
                latest_project = list(self.project_cache.values())[-1]
                context.update({
                    "project_root": latest_project.get("root"),
                    "project_framework": latest_project.get("framework"),
                    "project_language": latest_project.get("language")
                })
            
            return context
            
        except Exception as e:
            logger.error(f"AutoCode: Error getting current context: {e}")
            return {"platform": os.name, "timestamp": datetime.utcnow().isoformat()}