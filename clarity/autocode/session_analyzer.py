"""
Session Analyzer for AutoCode domain.

This module provides intelligent analysis of Claude conversation sessions
to extract meaningful summaries, track progress, and identify patterns.
"""

import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger


class SessionAnalyzer:
    """
    Analyzes Claude conversation sessions to extract meaningful insights.
    
    This class processes conversation logs to identify:
    - Tasks completed and approaches used
    - Files modified and architectural decisions made
    - Patterns and workflows that emerged
    - Potential next steps and recommendations
    - Technical context and project evolution
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Session Analyzer.
        
        Args:
            config: Configuration dictionary for analysis parameters
        """
        self.config = config or {}
        
        # Analysis configuration
        self.analysis_config = self.config.get("session_analysis", {
            "min_session_length": 3,  # Minimum messages for analysis
            "track_architectural_decisions": True,
            "extract_learning_patterns": True,
            "identify_workflow_improvements": True,
            "confidence_threshold": 0.6
        })
        
        # Pattern matchers for different types of session content
        self.task_patterns = {
            "implementation": [
                r"implement(?:ed|ing|s)?\s+(.+?)(?:\.|$)",
                r"creat(?:ed|ing|es|e)\s+(.+?)(?:\.|$)",
                r"add(?:ed|ing|s)?\s+(.+?)(?:\.|$)",
                r"build(?:ing|s|t)?\s+(.+?)(?:\.|$)"
            ],
            "fixes": [
                r"fix(?:ed|ing|es)?\s+(.+?)(?:\.|$)",
                r"resolv(?:ed|ing|es)?\s+(.+?)(?:\.|$)",
                r"correct(?:ed|ing|s)?\s+(.+?)(?:\.|$)",
                r"debug(?:ged|ging|s)?\s+(.+?)(?:\.|$)"
            ],
            "refactoring": [
                r"refactor(?:ed|ing|s)?\s+(.+?)(?:\.|$)",
                r"reorganiz(?:ed|ing|es)?\s+(.+?)(?:\.|$)",
                r"restructur(?:ed|ing|es)?\s+(.+?)(?:\.|$)",
                r"improv(?:ed|ing|es)?\s+(.+?)(?:\.|$)"
            ],
            "testing": [
                r"test(?:ed|ing|s)?\s+(.+?)(?:\.|$)",
                r"verif(?:ied|ying|ies|y)?\s+(.+?)(?:\.|$)",
                r"validat(?:ed|ing|es|e)?\s+(.+?)(?:\.|$)"
            ]
        }
        
        self.file_patterns = {
            "code_files": re.compile(r'[`"]([^`"]*\.(ts|tsx|js|jsx|py|rs|java|go|php|rb|cpp|c|h))[`"]'),
            "config_files": re.compile(r'[`"]([^`"]*\.(json|yaml|yml|toml|ini|env|config))[`"]'),
            "documentation": re.compile(r'[`"]([^`"]*\.(md|txt|rst|doc))[`"]')
        }
        
        self.architectural_patterns = [
            r"(?:implement|use|adopt|follow)\s+(.+?)\s+(?:pattern|architecture|approach)",
            r"(?:design|structure|organize)\s+(.+?)\s+(?:using|with|as)",
            r"(?:mvc|microservice|component|layered|hexagonal|clean)\s+architecture",
            r"(?:separation of concerns|dependency injection|inversion of control)"
        ]
        
        # Command tracking patterns
        self.command_patterns = {
            "successful": re.compile(r"(?:successfully\s+)?(?:ran|executed|completed)\s+[`']([^`']+)[`']"),
            "failed": re.compile(r"(?:failed|error)\s+(?:running|executing)\s+[`']([^`']+)[`']"),
            "retry": re.compile(r"(?:retry|try again|re-run)\s+[`']([^`']+)[`']")
        }
    
    async def analyze_session(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a conversation session.
        
        Args:
            conversation_log: List of conversation messages with role, content, timestamp
            
        Returns:
            Comprehensive session analysis with extracted insights
        """
        try:
            if len(conversation_log) < self.analysis_config.get("min_session_length", 3):
                logger.debug("Session too short for meaningful analysis")
                return self._create_minimal_summary(conversation_log)
            
            # Extract basic session metadata
            session_metadata = self._extract_session_metadata(conversation_log)
            
            # Analyze different aspects of the session
            analysis_results = {
                "session_metadata": session_metadata,
                "tasks_analysis": await self._analyze_tasks(conversation_log),
                "files_analysis": await self._analyze_file_interactions(conversation_log),
                "commands_analysis": await self._analyze_command_patterns(conversation_log),
                "architectural_analysis": await self._analyze_architectural_decisions(conversation_log),
                "learning_patterns": await self._extract_learning_patterns(conversation_log),
                "workflow_insights": await self._analyze_workflow_patterns(conversation_log),
                "context_evolution": await self._track_context_evolution(conversation_log),
                "recommendations": await self._generate_recommendations(conversation_log)
            }
            
            # Calculate overall session quality and impact
            analysis_results["session_quality"] = self._calculate_session_quality(analysis_results)
            analysis_results["generated_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Session analysis complete: {len(analysis_results['tasks_analysis']['completed_tasks'])} tasks, "
                       f"{len(analysis_results['files_analysis']['modified_files'])} files modified")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing session: {e}")
            return self._create_error_summary(conversation_log, str(e))
    
    async def _analyze_tasks(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tasks completed during the session."""
        tasks_by_type = {task_type: [] for task_type in self.task_patterns.keys()}
        task_outcomes = {"completed": [], "in_progress": [], "failed": []}
        approaches_used = []
        
        for message in conversation_log:
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                # Extract tasks by type
                for task_type, patterns in self.task_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            task_info = {
                                "description": match.strip(),
                                "type": task_type,
                                "timestamp": message.get("timestamp"),
                                "confidence": self._calculate_task_confidence(content, match)
                            }
                            tasks_by_type[task_type].append(task_info)
                
                # Determine task outcomes
                if any(indicator in content.lower() for indicator in 
                       ["completed", "finished", "done", "successfully", "implemented"]):
                    task_outcomes["completed"].extend(self._extract_task_descriptions(content))
                elif any(indicator in content.lower() for indicator in 
                         ["working on", "implementing", "in progress"]):
                    task_outcomes["in_progress"].extend(self._extract_task_descriptions(content))
                elif any(indicator in content.lower() for indicator in 
                         ["failed", "error", "couldn't", "unable"]):
                    task_outcomes["failed"].extend(self._extract_task_descriptions(content))
                
                # Extract approaches and methodologies
                approaches_used.extend(self._extract_approaches(content))
        
        return {
            "tasks_by_type": tasks_by_type,
            "task_outcomes": task_outcomes,
            "approaches_used": list(set(approaches_used)),
            "total_tasks": sum(len(tasks) for tasks in tasks_by_type.values()),
            "completion_rate": self._calculate_completion_rate(task_outcomes)
        }
    
    async def _analyze_file_interactions(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze file interactions and modifications."""
        files_by_type = {"code_files": set(), "config_files": set(), "documentation": set()}
        file_operations = {"created": [], "modified": [], "deleted": [], "read": []}
        project_structure_changes = []
        
        for message in conversation_log:
            content = message.get("content", "")
            
            # Extract files by type
            for file_type, pattern in self.file_patterns.items():
                matches = pattern.findall(content)
                for match in matches:
                    file_path = match[0] if isinstance(match, tuple) else match
                    files_by_type[file_type].add(file_path)
            
            # Extract file operations
            operations = {
                "created": re.findall(r"creat(?:ed|ing)\s+(?:file\s+)?[`']([^`']+)[`']", content, re.IGNORECASE),
                "modified": re.findall(r"(?:modif|updat|chang)(?:ied|ed|ing)\s+[`']([^`']+)[`']", content, re.IGNORECASE),
                "deleted": re.findall(r"delet(?:ed|ing)\s+[`']([^`']+)[`']", content, re.IGNORECASE),
                "read": re.findall(r"read(?:ing)?\s+[`']([^`']+)[`']", content, re.IGNORECASE)
            }
            
            for op_type, files in operations.items():
                file_operations[op_type].extend([{
                    "file": file,
                    "timestamp": message.get("timestamp"),
                    "context": content[:200] + "..." if len(content) > 200 else content
                } for file in files])
            
            # Detect project structure changes
            if any(indicator in content.lower() for indicator in 
                   ["directory structure", "project organization", "file structure", "reorganize"]):
                project_structure_changes.append({
                    "description": content[:300] + "..." if len(content) > 300 else content,
                    "timestamp": message.get("timestamp")
                })
        
        # Convert sets to lists for JSON serialization
        for file_type in files_by_type:
            files_by_type[file_type] = list(files_by_type[file_type])
        
        return {
            "files_by_type": files_by_type,
            "file_operations": file_operations,
            "project_structure_changes": project_structure_changes,
            "total_files_touched": len(set().union(*[files for files in files_by_type.values()])),
            "modification_patterns": self._analyze_modification_patterns(file_operations)
        }
    
    async def _analyze_command_patterns(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze command execution patterns and success rates."""
        commands_executed = {"successful": [], "failed": [], "retried": []}
        command_categories = {"build": [], "test": [], "install": [], "git": [], "file_ops": []}
        retry_patterns = []
        
        for message in conversation_log:
            content = message.get("content", "")
            
            # Extract command executions
            for result_type, pattern in self.command_patterns.items():
                matches = pattern.findall(content)
                for match in matches:
                    command_info = {
                        "command": match,
                        "timestamp": message.get("timestamp"),
                        "context": content[:150] + "..." if len(content) > 150 else content
                    }
                    commands_executed[result_type].append(command_info)
                    
                    # Categorize commands
                    category = self._categorize_command(match)
                    if category in command_categories:
                        command_categories[category].append(command_info)
            
            # Detect retry patterns
            if "retry" in content.lower() or "try again" in content.lower():
                retry_patterns.append({
                    "context": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": message.get("timestamp")
                })
        
        return {
            "commands_executed": commands_executed,
            "command_categories": command_categories,
            "retry_patterns": retry_patterns,
            "success_rate": self._calculate_command_success_rate(commands_executed),
            "most_common_commands": self._get_most_common_commands(commands_executed)
        }
    
    async def _analyze_architectural_decisions(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze architectural decisions and design patterns used."""
        if not self.analysis_config.get("track_architectural_decisions", True):
            return {}
        
        architectural_decisions = []
        design_patterns = []
        technology_choices = []
        
        for message in conversation_log:
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                # Extract architectural decisions
                for pattern in self.architectural_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        architectural_decisions.append({
                            "decision": match if isinstance(match, str) else " ".join(match),
                            "context": content[:200] + "..." if len(content) > 200 else content,
                            "timestamp": message.get("timestamp")
                        })
                
                # Extract design patterns
                pattern_keywords = ["singleton", "factory", "observer", "strategy", "decorator", 
                                  "adapter", "facade", "proxy", "command", "template"]
                for keyword in pattern_keywords:
                    if keyword in content.lower():
                        design_patterns.append({
                            "pattern": keyword,
                            "context": content[:150] + "..." if len(content) > 150 else content,
                            "timestamp": message.get("timestamp")
                        })
                
                # Extract technology choices
                tech_patterns = [
                    r"(?:using|with|chose|selected)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
                    r"framework:\s*([^,\n]+)",
                    r"library:\s*([^,\n]+)"
                ]
                for pattern in tech_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        technology_choices.append({
                            "technology": match,
                            "timestamp": message.get("timestamp")
                        })
        
        return {
            "architectural_decisions": architectural_decisions,
            "design_patterns": design_patterns,
            "technology_choices": technology_choices,
            "architecture_evolution": self._track_architecture_evolution(architectural_decisions)
        }
    
    async def _extract_learning_patterns(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns of learning and problem-solving approaches."""
        if not self.analysis_config.get("extract_learning_patterns", True):
            return {}
        
        learning_moments = []
        problem_solving_approaches = []
        knowledge_gaps = []
        discoveries = []
        
        for message in conversation_log:
            content = message.get("content", "")
            role = message.get("role")
            
            # Detect learning moments
            learning_indicators = ["learned", "discovered", "found out", "realized", "understood"]
            if any(indicator in content.lower() for indicator in learning_indicators):
                learning_moments.append({
                    "insight": content[:300] + "..." if len(content) > 300 else content,
                    "timestamp": message.get("timestamp"),
                    "source": role
                })
            
            # Detect problem-solving approaches
            if role == "assistant" and any(indicator in content.lower() for indicator in 
                                         ["approach", "strategy", "method", "technique", "solution"]):
                problem_solving_approaches.append({
                    "approach": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": message.get("timestamp")
                })
            
            # Detect knowledge gaps
            gap_indicators = ["don't know", "unclear", "need to research", "not sure", "investigate"]
            if any(indicator in content.lower() for indicator in gap_indicators):
                knowledge_gaps.append({
                    "gap": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": message.get("timestamp")
                })
            
            # Detect discoveries
            discovery_indicators = ["discovered", "found", "noticed", "observed", "detected"]
            if any(indicator in content.lower() for indicator in discovery_indicators):
                discoveries.append({
                    "discovery": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": message.get("timestamp")
                })
        
        return {
            "learning_moments": learning_moments,
            "problem_solving_approaches": problem_solving_approaches,
            "knowledge_gaps": knowledge_gaps,
            "discoveries": discoveries,
            "learning_velocity": len(learning_moments) / max(len(conversation_log), 1)
        }
    
    async def _analyze_workflow_patterns(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze workflow patterns and identify optimization opportunities."""
        workflow_steps = []
        bottlenecks = []
        efficiency_patterns = []
        
        # Track workflow progression
        for i, message in enumerate(conversation_log):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                # Identify workflow steps
                step_indicators = ["first", "next", "then", "after", "finally", "step"]
                if any(indicator in content.lower() for indicator in step_indicators):
                    workflow_steps.append({
                        "step": content[:150] + "..." if len(content) > 150 else content,
                        "order": i,
                        "timestamp": message.get("timestamp")
                    })
                
                # Identify bottlenecks
                bottleneck_indicators = ["stuck", "problem", "issue", "difficulty", "challenge"]
                if any(indicator in content.lower() for indicator in bottleneck_indicators):
                    bottlenecks.append({
                        "bottleneck": content[:200] + "..." if len(content) > 200 else content,
                        "timestamp": message.get("timestamp")
                    })
                
                # Identify efficiency patterns
                efficiency_indicators = ["automated", "scripted", "optimized", "streamlined"]
                if any(indicator in content.lower() for indicator in efficiency_indicators):
                    efficiency_patterns.append({
                        "optimization": content[:200] + "..." if len(content) > 200 else content,
                        "timestamp": message.get("timestamp")
                    })
        
        return {
            "workflow_steps": workflow_steps,
            "bottlenecks": bottlenecks,
            "efficiency_patterns": efficiency_patterns,
            "workflow_complexity": len(workflow_steps),
            "optimization_opportunities": self._identify_optimization_opportunities(workflow_steps, bottlenecks)
        }
    
    async def _track_context_evolution(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track how the context and understanding evolved during the session."""
        context_shifts = []
        scope_changes = []
        requirement_evolution = []
        
        previous_topics = set()
        
        for message in conversation_log:
            content = message.get("content", "")
            
            # Extract current topics
            current_topics = self._extract_topics(content)
            
            # Detect context shifts
            new_topics = current_topics - previous_topics
            if new_topics and previous_topics:
                context_shifts.append({
                    "new_topics": list(new_topics),
                    "timestamp": message.get("timestamp"),
                    "context": content[:100] + "..." if len(content) > 100 else content
                })
            
            previous_topics = current_topics
            
            # Detect scope changes
            scope_indicators = ["expand", "narrow", "scope", "broader", "focus"]
            if any(indicator in content.lower() for indicator in scope_indicators):
                scope_changes.append({
                    "change": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": message.get("timestamp")
                })
            
            # Track requirement evolution
            req_indicators = ["requirement", "need", "should", "must", "change"]
            if any(indicator in content.lower() for indicator in req_indicators):
                requirement_evolution.append({
                    "requirement": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": message.get("timestamp")
                })
        
        return {
            "context_shifts": context_shifts,
            "scope_changes": scope_changes,
            "requirement_evolution": requirement_evolution,
            "context_stability": 1.0 - (len(context_shifts) / max(len(conversation_log), 1))
        }
    
    async def _generate_recommendations(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations for future sessions based on current analysis."""
        recommendations = {
            "next_steps": [],
            "potential_improvements": [],
            "patterns_to_reuse": [],
            "areas_for_exploration": []
        }
        
        # Extract explicit next steps mentioned
        for message in conversation_log:
            content = message.get("content", "")
            
            # Find next steps
            next_step_patterns = [
                r"next(?:\s+step)?:\s*(.+?)(?:\.|$)",
                r"(?:should|need to|will)\s+(.+?)(?:\.|$)",
                r"(?:todo|to do):\s*(.+?)(?:\.|$)"
            ]
            
            for pattern in next_step_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match.strip()) > 10:  # Filter out very short matches
                        recommendations["next_steps"].append({
                            "step": match.strip(),
                            "source": "explicit_mention",
                            "confidence": 0.8
                        })
        
        # Infer potential improvements
        if any("error" in msg.get("content", "").lower() for msg in conversation_log):
            recommendations["potential_improvements"].append({
                "improvement": "Add error handling and validation",
                "reason": "Errors encountered during session",
                "confidence": 0.7
            })
        
        # Identify reusable patterns
        tech_stack = self._extract_technology_stack(conversation_log)
        if tech_stack:
            recommendations["patterns_to_reuse"].append({
                "pattern": f"Technology stack: {', '.join(tech_stack)}",
                "reason": "Successfully used in this session",
                "confidence": 0.8
            })
        
        return recommendations
    
    # Helper methods
    def _extract_session_metadata(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract basic session metadata."""
        if not conversation_log:
            return {}
        
        timestamps = [msg.get("timestamp") for msg in conversation_log if msg.get("timestamp")]
        
        return {
            "session_id": f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            "start_time": timestamps[0] if timestamps else datetime.utcnow().isoformat(),
            "end_time": timestamps[-1] if timestamps else datetime.utcnow().isoformat(),
            "duration_minutes": self._calculate_session_duration(timestamps),
            "message_count": len(conversation_log),
            "user_messages": len([m for m in conversation_log if m.get("role") == "user"]),
            "assistant_messages": len([m for m in conversation_log if m.get("role") == "assistant"])
        }
    
    def _calculate_task_confidence(self, content: str, task: str) -> float:
        """Calculate confidence score for extracted task."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for explicit task indicators
        if any(indicator in content.lower() for indicator in ["completed", "implemented", "created"]):
            confidence += 0.3
        
        # Increase confidence for specific technical terms
        if any(term in task.lower() for term in ["function", "class", "component", "api", "database"]):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_task_descriptions(self, content: str) -> List[str]:
        """Extract task descriptions from content."""
        # Simple extraction - could be enhanced with NLP
        sentences = content.split('.')
        tasks = []
        
        task_verbs = ["implement", "create", "add", "build", "fix", "update", "modify"]
        
        for sentence in sentences:
            if any(verb in sentence.lower() for verb in task_verbs):
                tasks.append(sentence.strip())
        
        return tasks[:3]  # Limit to avoid noise
    
    def _extract_approaches(self, content: str) -> List[str]:
        """Extract approaches and methodologies mentioned."""
        approaches = []
        approach_patterns = [
            r"(?:using|with|via)\s+([A-Za-z\s]+?)(?:\s+(?:to|for|in)|\.)",
            r"approach:\s*([^.\n]+)",
            r"method:\s*([^.\n]+)"
        ]
        
        for pattern in approach_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            approaches.extend([match.strip() for match in matches if len(match.strip()) > 3])
        
        return approaches
    
    def _calculate_completion_rate(self, task_outcomes: Dict[str, List]) -> float:
        """Calculate task completion rate."""
        total_tasks = sum(len(tasks) for tasks in task_outcomes.values())
        if total_tasks == 0:
            return 0.0
        
        completed_tasks = len(task_outcomes.get("completed", []))
        return completed_tasks / total_tasks
    
    def _analyze_modification_patterns(self, file_operations: Dict[str, List]) -> Dict[str, Any]:
        """Analyze file modification patterns."""
        patterns = {}
        
        # Most frequently modified file types
        all_files = []
        for op_files in file_operations.values():
            all_files.extend([f["file"] for f in op_files])
        
        if all_files:
            file_extensions = [Path(f).suffix.lower() for f in all_files]
            ext_counts = {}
            for ext in file_extensions:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            
            patterns["most_modified_types"] = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)
        
        return patterns
    
    def _categorize_command(self, command: str) -> str:
        """Categorize a command by its purpose."""
        command_lower = command.lower()
        
        if any(cmd in command_lower for cmd in ["npm", "yarn", "pip", "cargo install"]):
            return "install"
        elif any(cmd in command_lower for cmd in ["test", "pytest", "jest"]):
            return "test"
        elif any(cmd in command_lower for cmd in ["build", "compile", "make"]):
            return "build"
        elif any(cmd in command_lower for cmd in ["git", "commit", "push", "pull"]):
            return "git"
        elif any(cmd in command_lower for cmd in ["rm", "cp", "mv", "mkdir"]):
            return "file_ops"
        else:
            return "other"
    
    def _calculate_command_success_rate(self, commands_executed: Dict[str, List]) -> float:
        """Calculate command success rate."""
        successful = len(commands_executed.get("successful", []))
        failed = len(commands_executed.get("failed", []))
        total = successful + failed
        
        if total == 0:
            return 1.0
        
        return successful / total
    
    def _get_most_common_commands(self, commands_executed: Dict[str, List]) -> List[Dict[str, Any]]:
        """Get most commonly used commands."""
        all_commands = []
        for cmd_list in commands_executed.values():
            all_commands.extend([cmd["command"] for cmd in cmd_list])
        
        command_counts = {}
        for cmd in all_commands:
            # Extract base command (first word)
            base_cmd = cmd.split()[0] if cmd.split() else cmd
            command_counts[base_cmd] = command_counts.get(base_cmd, 0) + 1
        
        return [{"command": cmd, "count": count} 
                for cmd, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _track_architecture_evolution(self, architectural_decisions: List[Dict[str, Any]]) -> List[str]:
        """Track how architecture evolved during the session."""
        evolution = []
        
        # Group decisions by timestamp and extract evolution
        if architectural_decisions:
            sorted_decisions = sorted(architectural_decisions, 
                                    key=lambda x: x.get("timestamp", ""))
            
            for decision in sorted_decisions:
                evolution.append(decision["decision"])
        
        return evolution
    
    def _identify_optimization_opportunities(self, workflow_steps: List[Dict], bottlenecks: List[Dict]) -> List[str]:
        """Identify workflow optimization opportunities."""
        opportunities = []
        
        if len(bottlenecks) > 2:
            opportunities.append("Consider automating repetitive tasks to reduce bottlenecks")
        
        if len(workflow_steps) > 10:
            opportunities.append("Workflow complexity is high - consider breaking into smaller sessions")
        
        return opportunities
    
    def _extract_topics(self, content: str) -> set:
        """Extract topics from content using simple keyword extraction."""
        # Simple topic extraction - could be enhanced with NLP
        tech_keywords = {
            "react", "vue", "angular", "typescript", "javascript", "python", "rust",
            "api", "database", "authentication", "testing", "deployment", "docker",
            "aws", "git", "component", "service", "model", "controller"
        }
        
        content_lower = content.lower()
        found_topics = {keyword for keyword in tech_keywords if keyword in content_lower}
        
        return found_topics
    
    def _extract_technology_stack(self, conversation_log: List[Dict[str, Any]]) -> List[str]:
        """Extract technology stack used in the session."""
        technologies = set()
        
        tech_patterns = [
            r"(?:using|with|built with)\s+([A-Z][a-zA-Z]+)",
            r"framework:\s*([^,\n]+)",
            r"language:\s*([^,\n]+)"
        ]
        
        for message in conversation_log:
            content = message.get("content", "")
            for pattern in tech_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                technologies.update(match.strip() for match in matches)
        
        return list(technologies)
    
    def _calculate_session_duration(self, timestamps: List[str]) -> int:
        """Calculate session duration in minutes."""
        if len(timestamps) < 2:
            return 0
        
        try:
            start = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
            end = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            duration = end - start
            return int(duration.total_seconds() / 60)
        except:
            return 0
    
    def _calculate_session_quality(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall session quality metrics."""
        quality_metrics = {
            "overall_score": 0.0,
            "productivity_score": 0.0,
            "learning_score": 0.0,
            "technical_depth": 0.0,
            "documentation_quality": 0.0
        }
        
        try:
            # Productivity score based on tasks completed
            tasks_analysis = analysis_results.get("tasks_analysis", {})
            completion_rate = tasks_analysis.get("completion_rate", 0)
            total_tasks = tasks_analysis.get("total_tasks", 0)
            quality_metrics["productivity_score"] = min(completion_rate + (total_tasks * 0.1), 1.0)
            
            # Learning score based on insights and discoveries
            learning_patterns = analysis_results.get("learning_patterns", {})
            learning_velocity = learning_patterns.get("learning_velocity", 0)
            quality_metrics["learning_score"] = min(learning_velocity * 2, 1.0)
            
            # Technical depth based on architectural decisions and complexity
            arch_analysis = analysis_results.get("architectural_analysis", {})
            arch_decisions = len(arch_analysis.get("architectural_decisions", []))
            quality_metrics["technical_depth"] = min(arch_decisions * 0.2, 1.0)
            
            # Documentation quality based on file interactions
            files_analysis = analysis_results.get("files_analysis", {})
            docs_touched = len(files_analysis.get("files_by_type", {}).get("documentation", []))
            quality_metrics["documentation_quality"] = min(docs_touched * 0.3, 1.0)
            
            # Overall score as weighted average
            weights = [0.4, 0.3, 0.2, 0.1]  # Productivity, learning, technical, documentation
            scores = [quality_metrics["productivity_score"], quality_metrics["learning_score"],
                     quality_metrics["technical_depth"], quality_metrics["documentation_quality"]]
            
            quality_metrics["overall_score"] = sum(w * s for w, s in zip(weights, scores))
            
        except Exception as e:
            logger.error(f"Error calculating session quality: {e}")
        
        return quality_metrics
    
    def _create_minimal_summary(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create minimal summary for short sessions."""
        return {
            "session_metadata": self._extract_session_metadata(conversation_log),
            "summary_type": "minimal",
            "reason": "Session too short for detailed analysis",
            "message_count": len(conversation_log),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _create_error_summary(self, conversation_log: List[Dict[str, Any]], error: str) -> Dict[str, Any]:
        """Create error summary when analysis fails."""
        return {
            "session_metadata": self._extract_session_metadata(conversation_log),
            "summary_type": "error",
            "error": error,
            "message_count": len(conversation_log),
            "generated_at": datetime.utcnow().isoformat()
        }