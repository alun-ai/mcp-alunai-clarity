"""
Command learning infrastructure for the AutoCode system.
"""

import asyncio
import platform
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from loguru import logger


class CommandLearner:
    """
    Learns from bash command executions and suggests optimal commands.
    
    This class analyzes command execution patterns, learns from failures,
    and provides intelligent command suggestions based on context.
    """
    
    def __init__(self, domain_manager):
        """
        Initialize the command learner.
        
        Args:
            domain_manager: The memory domain manager instance
        """
        self.domain_manager = domain_manager
        self.command_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self.retry_sequences = []
        
        # Context weights for command scoring
        self.context_weights = {
            "project_type": 0.3,
            "platform": 0.2,
            "recent_success": 0.4,
            "user_preference": 0.1
        }
        
        # Current platform info
        self.platform = platform.system().lower()
        self.platform_details = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine()
        }
        
        # Command categorization
        self.command_categories = {
            "file_operations": ["rm", "cp", "mv", "mkdir", "rmdir", "touch", "chmod", "chown"],
            "package_management": ["npm", "yarn", "pip", "cargo", "mvn", "gradle", "composer"],
            "version_control": ["git", "svn", "hg", "bzr"],
            "build_tools": ["make", "cmake", "ninja", "bazel"],
            "process_management": ["ps", "kill", "killall", "top", "htop"],
            "network": ["curl", "wget", "ping", "ssh", "scp", "rsync"],
            "text_processing": ["grep", "sed", "awk", "sort", "uniq", "cut"],
            "system_info": ["ls", "pwd", "whoami", "id", "uname", "df", "du"]
        }
        
        # Intent mapping
        self.intent_patterns = {
            "delete_file": ["rm", "del", "remove", "delete"],
            "copy_file": ["cp", "copy"],
            "move_file": ["mv", "move", "rename"],
            "create_directory": ["mkdir", "md"],
            "list_files": ["ls", "dir"],
            "install_package": ["install", "add"],
            "build_project": ["build", "compile", "make"],
            "run_tests": ["test", "spec", "check"],
            "start_server": ["start", "serve", "run"],
            "git_operations": ["commit", "push", "pull", "clone", "merge"]
        }
    
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
        try:
            success = exit_code == 0
            timestamp = datetime.utcnow()
            
            # Create execution record
            execution_record = {
                "command": command,
                "exit_code": exit_code,
                "output": output[:500] if output else "",  # Truncate long output
                "success": success,
                "timestamp": timestamp.isoformat(),
                "context": context,
                "platform": self.platform_details
            }
            
            # Store in persistent memory
            await self.domain_manager.store_bash_execution(
                command=command,
                exit_code=exit_code,
                output=output,
                context={**context, "timestamp": timestamp.isoformat()},
                metadata={
                    "category": "command_execution",
                    "platform": self.platform,
                    "success": success
                }
            )
            
            # Extract intent and update patterns
            intent = self._extract_intent(command)
            base_command = self._extract_base_command(command)
            
            if success:
                self.command_patterns[intent].append(execution_record)
                await self._update_success_pattern(base_command, command, context)
            else:
                self.failure_patterns[intent].append(execution_record)
                await self._record_failure_pattern(base_command, command, output, context)
            
            # Detect retry patterns
            await self._detect_retry_pattern(command, success, context, timestamp)
            
            logger.debug(f"CommandLearner: Tracked execution {command} (success: {success})")
            
        except (ValueError, KeyError, AttributeError, OSError) as e:
            logger.error(f"CommandLearner: Error tracking execution {command}: {e}")
    
    async def suggest_command(
        self, 
        intent: str, 
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimal commands for given intent.
        
        Args:
            intent: What the user wants to accomplish
            context: Current context (project type, platform, etc.)
            
        Returns:
            List of command suggestions with confidence scores
        """
        try:
            context = context or {}
            context["platform"] = self.platform
            
            # Normalize intent
            normalized_intent = self._normalize_intent(intent)
            
            # Get command patterns for this intent
            patterns = await self._get_patterns_for_intent(normalized_intent, context)
            
            # Generate suggestions based on patterns
            suggestions = []
            
            # Add historical successful commands
            for pattern in patterns:
                confidence = self._calculate_confidence(pattern, context)
                if confidence > 0.3:  # Minimum confidence threshold
                    suggestions.append({
                        "command": pattern.get("command", ""),
                        "confidence": confidence,
                        "success_rate": pattern.get("success_rate", 0.5),
                        "last_used": pattern.get("timestamp"),
                        "context": pattern.get("context", {}),
                        "reasoning": self._generate_reasoning(pattern, context)
                    })
            
            # Add platform-specific suggestions
            platform_suggestions = await self._get_platform_suggestions(normalized_intent, context)
            suggestions.extend(platform_suggestions)
            
            # Add context-aware suggestions
            context_suggestions = await self._get_context_suggestions(normalized_intent, context)
            suggestions.extend(context_suggestions)
            
            # Remove duplicates and sort by confidence
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            unique_suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Limit to top 5 suggestions
            return unique_suggestions[:5]
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error suggesting commands for '{intent}': {e}")
            return []
    
    async def learn_retry_patterns(self) -> None:
        """
        Analyze failed->success command sequences to learn retry patterns.
        """
        try:
            # Get recent bash executions from memory
            recent_executions = await self._get_recent_executions(hours=24)
            
            # Detect retry sequences
            retry_patterns = self._detect_retry_sequences(recent_executions)
            
            # Store learned retry patterns
            for pattern in retry_patterns:
                await self._store_retry_pattern(pattern)
                
            logger.info(f"CommandLearner: Learned {len(retry_patterns)} retry patterns")
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error learning retry patterns: {e}")
    
    async def get_command_context(self, project_path: str = None) -> Dict[str, Any]:
        """
        Get current command context for better suggestions.
        
        Args:
            project_path: Current project path
            
        Returns:
            Context dictionary with project info, platform, etc.
        """
        try:
            context = {
                "platform": self.platform,
                "platform_details": self.platform_details,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if project_path:
                # Detect project type
                project_type = await self._detect_project_type(project_path)
                context.update({
                    "project_type": project_type,
                    "project_path": project_path,
                    "project_language": await self._detect_project_language(project_path)
                })
            
            return context
            
        except (OSError, ValueError, AttributeError, KeyError) as e:
            logger.error(f"CommandLearner: Error getting command context: {e}")
            return {"platform": self.platform, "timestamp": datetime.utcnow().isoformat()}
    
    async def get_success_rate(self, command: str, context: Dict[str, Any] = None) -> float:
        """
        Get success rate for a specific command in given context.
        
        Args:
            command: Command to check
            context: Execution context
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        try:
            base_command = self._extract_base_command(command)
            
            # Query recent executions for this command
            query = f"command:{base_command}"
            if context and context.get("project_type"):
                query += f" project:{context['project_type']}"
            
            memories = await self.domain_manager.retrieve_memories(
                query=query,
                memory_types=["bash_execution"],
                limit=50,
                min_similarity=0.5
            )
            
            if not memories:
                return 0.5  # Default neutral rate
            
            # Calculate success rate
            total = len(memories)
            successful = sum(1 for memory in memories 
                           if memory.get("content", {}).get("exit_code") == 0)
            
            return successful / total if total > 0 else 0.5
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error calculating success rate for {command}: {e}")
            return 0.5
    
    # Private implementation methods
    def _extract_intent(self, command: str) -> str:
        """Extract intent from command."""
        try:
            # Split command and get base command
            parts = command.strip().split()
            if not parts:
                return "unknown"
            
            base_cmd = parts[0]
            
            # Check for compound commands (with &&, ||, ;)
            if any(op in command for op in ["&&", "||", ";"]):
                # For compound commands, use the first command's intent
                first_cmd = command.split("&&")[0].split("||")[0].split(";")[0].strip()
                return self._extract_intent(first_cmd)
            
            # Map base commands to intents
            intent_mapping = {
                # File operations
                "rm": "delete_file",
                "rmdir": "delete_directory", 
                "cp": "copy_file",
                "mv": "move_file",
                "mkdir": "create_directory",
                "touch": "create_file",
                "chmod": "change_permissions",
                "chown": "change_ownership",
                
                # Package management
                "npm": "package_management",
                "yarn": "package_management",
                "pip": "package_management",
                "cargo": "package_management",
                "mvn": "package_management",
                "gradle": "package_management",
                
                # Version control
                "git": "version_control",
                "svn": "version_control",
                
                # Build tools
                "make": "build_project",
                "cmake": "build_project",
                "ninja": "build_project",
                
                # Process management
                "ps": "list_processes",
                "kill": "kill_process",
                "killall": "kill_processes",
                
                # System info
                "ls": "list_files",
                "pwd": "show_directory",
                "cd": "change_directory",
                "find": "find_files",
                "grep": "search_text",
                
                # Network
                "curl": "http_request",
                "wget": "download_file",
                "ssh": "remote_access",
                "scp": "secure_copy",
                
                # Development
                "python": "run_python",
                "node": "run_node",
                "java": "run_java",
                "docker": "container_management"
            }
            
            # First check direct mapping
            if base_cmd in intent_mapping:
                # For package managers, be more specific based on arguments
                if base_cmd in ["npm", "yarn"] and len(parts) > 1:
                    subcommand = parts[1]
                    if subcommand in ["install", "add"]:
                        return "install_package"
                    elif subcommand in ["run", "start"]:
                        return "run_script"
                    elif subcommand in ["build"]:
                        return "build_project"
                    elif subcommand in ["test"]:
                        return "run_tests"
                    else:
                        return f"npm_{subcommand}"
                
                elif base_cmd == "pip" and len(parts) > 1:
                    subcommand = parts[1]
                    if subcommand == "install":
                        return "install_package"
                    elif subcommand in ["list", "show"]:
                        return "list_packages"
                    else:
                        return f"pip_{subcommand}"
                
                elif base_cmd == "git" and len(parts) > 1:
                    subcommand = parts[1]
                    return f"git_{subcommand}"
                
                elif base_cmd == "docker" and len(parts) > 1:
                    subcommand = parts[1]
                    return f"docker_{subcommand}"
                
                return intent_mapping[base_cmd]
            
            # Check for intent keywords in the full command
            command_lower = command.lower()
            for intent, keywords in self.intent_patterns.items():
                if any(keyword in command_lower for keyword in keywords):
                    return intent
            
            # Default to the base command name
            return base_cmd
            
        except (ValueError, AttributeError, IndexError, KeyError) as e:
            logger.error(f"CommandLearner: Error extracting intent from '{command}': {e}")
            return "unknown"
    
    def _extract_base_command(self, command: str) -> str:
        """Extract the base command (first word) from a command string."""
        try:
            parts = command.strip().split()
            return parts[0] if parts else ""
        except (ValueError, AttributeError, IndexError) as e:
            logger.error(f"CommandLearner: Error extracting base command from '{command}': {e}")
            return ""
    
    def _normalize_intent(self, intent: str) -> str:
        """Normalize intent string for consistent matching."""
        try:
            # Convert to lowercase and replace spaces/underscores
            normalized = intent.lower().replace(" ", "_").replace("-", "_")
            
            # Map common variations
            intent_aliases = {
                "remove_file": "delete_file",
                "delete": "delete_file",
                "copy": "copy_file",
                "move": "move_file",
                "rename": "move_file",
                "create_folder": "create_directory",
                "make_directory": "create_directory",
                "install": "install_package",
                "build": "build_project",
                "compile": "build_project",
                "test": "run_tests",
                "start": "run_script"
            }
            
            return intent_aliases.get(normalized, normalized)
            
        except (ValueError, AttributeError, KeyError) as e:
            logger.error(f"CommandLearner: Error normalizing intent '{intent}': {e}")
            return intent.lower()
    
    async def _get_patterns_for_intent(
        self, 
        intent: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get command patterns for specific intent and context."""
        try:
            # Query persistent memory for command patterns
            query = f"intent:{intent}"
            if context.get("platform"):
                query += f" platform:{context['platform']}"
            if context.get("project_type"):
                query += f" project:{context['project_type']}"
            
            memories = await self.domain_manager.retrieve_memories(
                query=query,
                memory_types=["command_pattern", "bash_execution"],
                limit=30,
                min_similarity=0.4
            )
            
            # Convert memories to patterns
            patterns = []
            for memory in memories:
                content = memory.get("content", {})
                if memory["type"] == "bash_execution":
                    # Convert execution record to pattern format
                    patterns.append({
                        "command": content.get("command", ""),
                        "success_rate": 1.0 if content.get("exit_code") == 0 else 0.0,
                        "timestamp": content.get("timestamp"),
                        "context": content.get("context", {}),
                        "platform": content.get("platform", {}),
                        "similarity": memory.get("similarity", 0.5)
                    })
                elif memory["type"] == "command_pattern":
                    # Use stored pattern directly
                    patterns.append({
                        **content,
                        "similarity": memory.get("similarity", 0.5)
                    })
            
            return patterns
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error getting patterns for intent '{intent}': {e}")
            return []
    
    def _calculate_confidence(
        self, 
        pattern: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a command suggestion."""
        try:
            confidence = 0.0
            
            # Base success rate (40% weight)
            success_rate = pattern.get("success_rate", 0.5)
            confidence += success_rate * 0.4
            
            # Platform match (20% weight)
            pattern_platform = pattern.get("platform", {})
            if isinstance(pattern_platform, dict):
                pattern_platform_name = pattern_platform.get("system", "").lower()
            else:
                pattern_platform_name = str(pattern_platform).lower()
            
            if pattern_platform_name == context.get("platform", "").lower():
                confidence += 0.2
            
            # Project type match (20% weight)
            pattern_context = pattern.get("context", {})
            pattern_project = pattern_context.get("project_type", "")
            context_project = context.get("project_type", "")
            if pattern_project and pattern_project == context_project:
                confidence += 0.2
            
            # Recency bonus (10% weight)
            timestamp = pattern.get("timestamp")
            if timestamp:
                try:
                    pattern_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    days_ago = (datetime.utcnow() - pattern_time.replace(tzinfo=None)).days
                    recency_bonus = max(0, 0.1 - (days_ago * 0.005))  # Decay over time
                    confidence += recency_bonus
                except:
                    pass
            
            # Similarity bonus (10% weight)
            similarity = pattern.get("similarity", 0.5)
            confidence += similarity * 0.1
            
            return min(1.0, confidence)
            
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"CommandLearner: Error calculating confidence: {e}")
            return 0.5
    
    def _generate_reasoning(
        self, 
        pattern: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for command suggestion."""
        try:
            reasons = []
            
            success_rate = pattern.get("success_rate", 0.5)
            if success_rate > 0.8:
                reasons.append(f"High success rate ({success_rate:.1%})")
            elif success_rate > 0.6:
                reasons.append(f"Good success rate ({success_rate:.1%})")
            
            # Platform matching
            pattern_platform = pattern.get("platform", {})
            if isinstance(pattern_platform, dict):
                pattern_platform_name = pattern_platform.get("system", "")
            else:
                pattern_platform_name = str(pattern_platform)
            
            if pattern_platform_name.lower() == context.get("platform", "").lower():
                reasons.append("Matches your platform")
            
            # Project type matching
            pattern_context = pattern.get("context", {})
            pattern_project = pattern_context.get("project_type", "")
            context_project = context.get("project_type", "")
            if pattern_project and pattern_project == context_project:
                reasons.append(f"Optimized for {pattern_project} projects")
            
            # Recent usage
            timestamp = pattern.get("timestamp")
            if timestamp:
                try:
                    pattern_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    days_ago = (datetime.utcnow() - pattern_time.replace(tzinfo=None)).days
                    if days_ago < 7:
                        reasons.append("Recently used successfully")
                except:
                    pass
            
            return "; ".join(reasons) if reasons else "Based on historical patterns"
            
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"CommandLearner: Error generating reasoning: {e}")
            return "Based on historical patterns"
    
    async def _get_platform_suggestions(
        self, 
        intent: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get platform-specific command suggestions."""
        try:
            suggestions = []
            platform = context.get("platform", self.platform)
            
            # Platform-specific command mappings
            platform_commands = {
                "darwin": {  # macOS
                    "delete_file": ["rm -f", "trash"],
                    "copy_file": ["cp", "ditto"],
                    "list_files": ["ls -la", "ls -lah"],
                    "install_package": ["brew install"],
                    "open_file": ["open"],
                    "show_processes": ["ps aux", "top"]
                },
                "linux": {
                    "delete_file": ["rm -f", "rm -rf"],
                    "copy_file": ["cp -r", "rsync -av"],
                    "list_files": ["ls -la", "ll"],
                    "install_package": ["apt install", "yum install", "dnf install"],
                    "open_file": ["xdg-open"],
                    "show_processes": ["ps aux", "htop"]
                },
                "windows": {
                    "delete_file": ["del", "rm -f"],
                    "copy_file": ["copy", "xcopy"],
                    "list_files": ["dir", "ls"],
                    "install_package": ["choco install", "winget install"],
                    "open_file": ["start"],
                    "show_processes": ["tasklist", "Get-Process"]
                }
            }
            
            platform_cmds = platform_commands.get(platform, {})
            commands_for_intent = platform_cmds.get(intent, [])
            
            for cmd in commands_for_intent:
                suggestions.append({
                    "command": cmd,
                    "confidence": 0.7,
                    "success_rate": 0.8,  # Assumed good for platform defaults
                    "last_used": None,
                    "context": {"platform_specific": True},
                    "reasoning": f"Platform-optimized for {platform}"
                })
            
            return suggestions
            
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"CommandLearner: Error getting platform suggestions: {e}")
            return []
    
    async def _get_context_suggestions(
        self, 
        intent: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get context-aware command suggestions based on project type."""
        try:
            suggestions = []
            project_type = context.get("project_type", "")
            
            # Project-specific command mappings
            project_commands = {
                "node": {
                    "install_package": ["npm install", "yarn add"],
                    "run_tests": ["npm test", "yarn test"],
                    "build_project": ["npm run build", "yarn build"],
                    "start_server": ["npm start", "yarn start", "npm run dev"]
                },
                "python": {
                    "install_package": ["pip install", "poetry add"],
                    "run_tests": ["pytest", "python -m pytest"],
                    "build_project": ["python setup.py build", "poetry build"],
                    "start_server": ["python manage.py runserver", "flask run"]
                },
                "rust": {
                    "install_package": ["cargo add"],
                    "run_tests": ["cargo test"],
                    "build_project": ["cargo build", "cargo build --release"],
                    "start_server": ["cargo run"]
                },
                "java": {
                    "install_package": ["mvn dependency:resolve", "gradle dependencies"],
                    "run_tests": ["mvn test", "gradle test"],
                    "build_project": ["mvn compile", "gradle build"],
                    "start_server": ["mvn spring-boot:run", "gradle bootRun"]
                }
            }
            
            project_cmds = project_commands.get(project_type, {})
            commands_for_intent = project_cmds.get(intent, [])
            
            for cmd in commands_for_intent:
                suggestions.append({
                    "command": cmd,
                    "confidence": 0.8,
                    "success_rate": 0.85,
                    "last_used": None,
                    "context": {"project_specific": True, "project_type": project_type},
                    "reasoning": f"Optimized for {project_type} projects"
                })
            
            return suggestions
            
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"CommandLearner: Error getting context suggestions: {e}")
            return []
    
    def _deduplicate_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate command suggestions, keeping the highest confidence."""
        try:
            seen_commands = {}
            
            for suggestion in suggestions:
                command = suggestion.get("command", "")
                if not command:
                    continue
                
                if command not in seen_commands:
                    seen_commands[command] = suggestion
                else:
                    # Keep the suggestion with higher confidence
                    if suggestion.get("confidence", 0) > seen_commands[command].get("confidence", 0):
                        seen_commands[command] = suggestion
            
            return list(seen_commands.values())
            
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"CommandLearner: Error deduplicating suggestions: {e}")
            return suggestions
    
    async def _update_success_pattern(
        self, 
        base_command: str, 
        full_command: str, 
        context: Dict[str, Any]
    ) -> None:
        """Update success patterns for a command."""
        try:
            # Store or update command pattern
            await self.domain_manager.store_command_pattern(
                command=base_command,
                context=context,
                success_rate=1.0,  # This execution was successful
                platform=self.platform,
                metadata={
                    "full_command": full_command,
                    "category": "success_pattern",
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error updating success pattern for {base_command}: {e}")
    
    async def _record_failure_pattern(
        self, 
        base_command: str, 
        full_command: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> None:
        """Record failure patterns for learning."""
        try:
            # Store failure information
            await self.domain_manager.store_memory(
                memory_type="fact",
                content={
                    "fact": f"Command failure: {base_command}",
                    "command": full_command,
                    "error_output": output[:200],  # Truncate error
                    "failure_context": context
                },
                importance=0.6,
                metadata={
                    "category": "command_failure",
                    "base_command": base_command,
                    "platform": self.platform
                }
            )
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error recording failure pattern for {base_command}: {e}")
    
    async def _detect_retry_pattern(
        self, 
        command: str, 
        success: bool, 
        context: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """Detect if this command might be a retry of a failed command."""
        try:
            if not success:
                return  # Only interested in successful retries
            
            # Look for recent failures of similar commands
            base_command = self._extract_base_command(command)
            recent_time = timestamp - timedelta(minutes=5)
            
            # Query recent failures
            query = f"command:{base_command} failure"
            recent_failures = await self.domain_manager.retrieve_memories(
                query=query,
                memory_types=["bash_execution"],
                limit=10,
                min_similarity=0.5
            )
            
            for failure_memory in recent_failures:
                failure_content = failure_memory.get("content", {})
                failure_time_str = failure_content.get("timestamp", "")
                
                try:
                    failure_time = datetime.fromisoformat(failure_time_str.replace("Z", "+00:00"))
                    failure_time = failure_time.replace(tzinfo=None)
                    
                    # Check if failure was recent (within 5 minutes)
                    if failure_time >= recent_time:
                        failed_command = failure_content.get("command", "")
                        similarity = self._calculate_command_similarity(command, failed_command)
                        
                        if similarity > 0.7:  # Commands are similar
                            # This might be a successful retry
                            await self._store_retry_pattern({
                                "failed_command": failed_command,
                                "successful_command": command,
                                "context": context,
                                "similarity": similarity,
                                "time_between": (timestamp - failure_time).total_seconds(),
                                "retry_timestamp": timestamp.isoformat()
                            })
                            break
                            
                except (ValueError, TypeError, AttributeError) as parse_error:
                    logger.error(f"CommandLearner: Error parsing failure timestamp: {parse_error}")
                    continue
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error detecting retry pattern for {command}: {e}")
    
    def _calculate_command_similarity(self, cmd1: str, cmd2: str) -> float:
        """Calculate similarity between two commands."""
        try:
            # Split commands into words
            words1 = set(cmd1.lower().split())
            words2 = set(cmd2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            jaccard = len(intersection) / len(union)
            
            # Give extra weight to same base command
            base1 = cmd1.split()[0] if cmd1.split() else ""
            base2 = cmd2.split()[0] if cmd2.split() else ""
            
            if base1 == base2:
                jaccard += 0.2  # Boost for same base command
            
            return min(1.0, jaccard)
            
        except (ValueError, AttributeError, IndexError, TypeError) as e:
            logger.error(f"CommandLearner: Error calculating command similarity: {e}")
            return 0.0
    
    async def _store_retry_pattern(self, pattern: Dict[str, Any]) -> None:
        """Store a detected retry pattern."""
        try:
            await self.domain_manager.store_memory(
                memory_type="fact",
                content={
                    "fact": "Command retry pattern detected",
                    "retry_pattern": pattern
                },
                importance=0.7,
                metadata={
                    "category": "retry_pattern",
                    "failed_command": pattern.get("failed_command", ""),
                    "successful_command": pattern.get("successful_command", ""),
                    "similarity": pattern.get("similarity", 0.0)
                }
            )
            
            # Add to local cache
            self.retry_sequences.append(pattern)
            
            logger.info(f"CommandLearner: Stored retry pattern: {pattern['failed_command']} -> {pattern['successful_command']}")
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error storing retry pattern: {e}")
    
    async def _get_recent_executions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent bash executions from memory."""
        try:
            # Query recent bash executions
            memories = await self.domain_manager.retrieve_memories(
                query="bash_execution",
                memory_types=["bash_execution"],
                limit=100,
                min_similarity=0.3
            )
            
            # Filter by time if possible
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_executions = []
            
            for memory in memories:
                content = memory.get("content", {})
                timestamp_str = content.get("timestamp", "")
                
                try:
                    execution_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    execution_time = execution_time.replace(tzinfo=None)
                    
                    if execution_time >= cutoff_time:
                        recent_executions.append(content)
                except:
                    # Include if we can't parse timestamp
                    recent_executions.append(content)
            
            return recent_executions
            
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"CommandLearner: Error getting recent executions: {e}")
            return []
    
    def _detect_retry_sequences(self, executions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect retry sequences in execution history."""
        try:
            retry_patterns = []
            
            # Sort executions by timestamp
            sorted_executions = sorted(
                executions,
                key=lambda x: x.get("timestamp", ""),
                reverse=False
            )
            
            # Look for failure -> success patterns
            for i in range(len(sorted_executions) - 1):
                current = sorted_executions[i]
                next_exec = sorted_executions[i + 1]
                
                # Check if current failed and next succeeded
                if (current.get("exit_code", 0) != 0 and 
                    next_exec.get("exit_code", 0) == 0):
                    
                    current_cmd = current.get("command", "")
                    next_cmd = next_exec.get("command", "")
                    
                    # Check similarity
                    similarity = self._calculate_command_similarity(current_cmd, next_cmd)
                    
                    if similarity > 0.6:  # Similar commands
                        retry_patterns.append({
                            "failed_command": current_cmd,
                            "successful_command": next_cmd,
                            "similarity": similarity,
                            "failed_output": current.get("output", ""),
                            "success_context": next_exec.get("context", {}),
                            "pattern_type": "failure_to_success_retry"
                        })
            
            return retry_patterns
            
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"CommandLearner: Error detecting retry sequences: {e}")
            return []
    
    async def _detect_project_type(self, project_path: str) -> str:
        """Detect project type from path."""
        try:
            import os
            from pathlib import Path
            
            path = Path(project_path)
            
            # Check for project indicators
            if (path / "package.json").exists():
                return "node"
            elif (path / "Cargo.toml").exists():
                return "rust"
            elif (path / "requirements.txt").exists() or (path / "pyproject.toml").exists():
                return "python"
            elif (path / "pom.xml").exists():
                return "java"
            elif (path / "Gemfile").exists():
                return "ruby"
            elif (path / "go.mod").exists():
                return "go"
            elif (path / "composer.json").exists():
                return "php"
            
            return "unknown"
            
        except (OSError, ValueError, AttributeError, PermissionError) as e:
            logger.error(f"CommandLearner: Error detecting project type for {project_path}: {e}")
            return "unknown"
    
    async def _detect_project_language(self, project_path: str) -> str:
        """Detect primary language from project path."""
        try:
            from pathlib import Path
            
            path = Path(project_path)
            
            # Count file extensions
            extensions = {}
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in [".js", ".ts", ".jsx", ".tsx", ".py", ".rs", ".java", ".go", ".rb", ".php"]:
                        extensions[ext] = extensions.get(ext, 0) + 1
            
            if not extensions:
                return "unknown"
            
            # Map extensions to languages
            ext_to_lang = {
                ".js": "javascript",
                ".jsx": "javascript", 
                ".ts": "typescript",
                ".tsx": "typescript",
                ".py": "python",
                ".rs": "rust",
                ".java": "java",
                ".go": "go",
                ".rb": "ruby",
                ".php": "php"
            }
            
            # Find most common extension
            most_common_ext = max(extensions, key=extensions.get)
            return ext_to_lang.get(most_common_ext, "unknown")
            
        except (OSError, ValueError, AttributeError, PermissionError, KeyError) as e:
            logger.error(f"CommandLearner: Error detecting project language for {project_path}: {e}")
            return "unknown"