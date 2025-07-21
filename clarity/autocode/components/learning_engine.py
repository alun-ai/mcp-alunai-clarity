"""
Learning engine component for AutoCode domain.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from clarity.shared.infrastructure import get_cache
from clarity.shared.exceptions import AutoCodeError
from ..interfaces import LearningEngine, AutoCodeComponent
from ..command_learner import CommandLearner


class LearningEngineImpl(LearningEngine, AutoCodeComponent):
    """Implementation of learning and suggestion engine"""
    
    def __init__(self, config: Dict[str, Any], persistence_domain, domain_manager=None):
        """Initialize learning engine"""
        self.config = config
        self.persistence_domain = persistence_domain
        self.domain_manager = domain_manager
        self.autocode_config = config.get("autocode", {})
        
        # Suggestion caching
        self.suggestion_cache = get_cache(
            "command_suggestions",
            max_size=1000,      # Cache 1000 suggestions
            max_memory_mb=50,   # 50MB for suggestion data
            default_ttl=900.0   # 15 minutes TTL
        )
        
        # Learning progression caching
        self.progression_cache = get_cache(
            "learning_progression",
            max_size=200,       # Cache 200 learning topics
            max_memory_mb=25,   # 25MB for progression data
            default_ttl=3600.0  # 1 hour TTL
        )
        
        # Initialize command learner
        self.command_learner = None
        self._initialized = False
        self._command_history = []
        self._learning_stats = {
            "commands_processed": 0,
            "successful_suggestions": 0,
            "failed_suggestions": 0,
            "learning_sessions": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the learning engine"""
        if self._initialized:
            return
            
        logger.info("Initializing Learning Engine")
        
        try:
            # Initialize command learner
            self.command_learner = CommandLearner(
                self.autocode_config, 
                self.persistence_domain
            )
            await self.command_learner.initialize()
            
            # Load existing learning data
            await self._load_learning_history()
            
            self._initialized = True
            logger.info("Learning Engine initialized successfully")
            
        except (ValueError, KeyError, ImportError, OSError) as e:
            logger.error(f"Failed to initialize Learning Engine: {e}")
            raise AutoCodeError("Learning engine initialization failed", cause=e)
    
    async def shutdown(self) -> None:
        """Shutdown the learning engine"""
        if self.command_learner:
            await self.command_learner.shutdown()
            
        # Save learning statistics
        await self._save_learning_stats()
        
        self._initialized = False
        logger.info("Learning Engine shutdown complete")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": "LearningEngine",
            "initialized": self._initialized,
            "learning_stats": self._learning_stats,
            "command_history_size": len(self._command_history),
            "suggestion_cache_info": self.suggestion_cache.get_info(),
            "progression_cache_info": self.progression_cache.get_info(),
            "command_learner_enabled": self.command_learner is not None
        }
    
    async def suggest_command(self, intent: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get intelligent command suggestions based on intent and context"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create cache key
            cache_key = f"suggest_{hash(intent)}_{hash(str(context))}"
            
            # Try cache first
            cached_suggestions = self.suggestion_cache.get(cache_key)
            if cached_suggestions is not None:
                return cached_suggestions
            
            # Use command learner to generate suggestions
            if self.command_learner:
                suggestions = await self.command_learner.suggest_command(intent, context or {})
            else:
                suggestions = await self._generate_basic_suggestions(intent, context)
            
            # Enhance suggestions with additional context
            enhanced_suggestions = []
            for suggestion in suggestions:
                enhanced = {
                    "command": suggestion.get("command", ""),
                    "confidence": suggestion.get("confidence", 0.0),
                    "reasoning": suggestion.get("reasoning", ""),
                    "context": suggestion.get("context", {}),
                    "success_rate": suggestion.get("success_rate", 0.0),
                    "last_used": suggestion.get("last_used"),
                    "usage_count": suggestion.get("usage_count", 0),
                    "platform_specific": suggestion.get("platform_specific", False),
                    "requires_confirmation": suggestion.get("requires_confirmation", False),
                    "related_commands": suggestion.get("related_commands", []),
                    "estimated_duration": suggestion.get("estimated_duration"),
                    "risk_level": self._assess_command_risk(suggestion.get("command", "")),
                    "learning_source": suggestion.get("learning_source", "command_learner")
                }
                enhanced_suggestions.append(enhanced)
            
            # Sort by confidence and success rate
            enhanced_suggestions.sort(
                key=lambda x: (x["confidence"] * 0.7 + x["success_rate"] * 0.3), 
                reverse=True
            )
            
            # Cache the suggestions
            self.suggestion_cache.set(cache_key, enhanced_suggestions)
            
            logger.debug(f"Generated {len(enhanced_suggestions)} suggestions for intent: {intent[:50]}...")
            return enhanced_suggestions
            
        except (MemoryOperationError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to generate command suggestions: {e}")
            raise AutoCodeError("Command suggestion generation failed", cause=e)
    
    async def suggest_workflow_optimizations(self, current_workflow: List[str], 
                                           session_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Suggest workflow optimizations based on historical data"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create cache key
            cache_key = f"workflow_{hash(str(current_workflow))}_{hash(str(session_context))}"
            
            # Try cache first
            cached_optimizations = self.suggestion_cache.get(cache_key)
            if cached_optimizations is not None:
                return cached_optimizations
            
            optimizations = []
            
            # Analyze current workflow for patterns
            workflow_analysis = await self._analyze_workflow_patterns(current_workflow)
            
            # Find redundant commands
            redundant_commands = self._find_redundant_commands(current_workflow)
            if redundant_commands:
                optimizations.append({
                    "type": "redundancy_removal",
                    "description": f"Remove redundant commands: {', '.join(redundant_commands)}",
                    "impact": "medium",
                    "time_savings_minutes": len(redundant_commands) * 0.5,
                    "commands_to_remove": redundant_commands
                })
            
            # Suggest command combinations
            combinable_commands = self._find_combinable_commands(current_workflow)
            if combinable_commands:
                optimizations.append({
                    "type": "command_combination",
                    "description": f"Combine commands using && or pipes",
                    "impact": "low",
                    "time_savings_minutes": len(combinable_commands) * 0.2,
                    "suggested_combinations": combinable_commands
                })
            
            # Check for missing best practices
            missing_practices = await self._check_missing_best_practices(current_workflow, session_context)
            for practice in missing_practices:
                optimizations.append({
                    "type": "best_practice",
                    "description": practice["description"],
                    "impact": practice["impact"],
                    "time_savings_minutes": practice.get("time_savings", 0),
                    "recommended_command": practice.get("command")
                })
            
            # Suggest parallelization opportunities
            parallel_opportunities = self._find_parallelization_opportunities(current_workflow)
            if parallel_opportunities:
                optimizations.append({
                    "type": "parallelization",
                    "description": "Run independent commands in parallel",
                    "impact": "high",
                    "time_savings_minutes": parallel_opportunities["estimated_savings"],
                    "parallel_groups": parallel_opportunities["groups"]
                })
            
            # Sort by impact and time savings
            impact_priority = {"high": 3, "medium": 2, "low": 1}
            optimizations.sort(
                key=lambda x: (impact_priority.get(x["impact"], 0), x.get("time_savings_minutes", 0)), 
                reverse=True
            )
            
            # Cache the optimizations
            self.suggestion_cache.set(cache_key, optimizations, ttl=1800.0)  # 30 minutes TTL
            
            logger.debug(f"Generated {len(optimizations)} workflow optimizations")
            return optimizations
            
        except (MemoryOperationError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to suggest workflow optimizations: {e}")
            raise AutoCodeError("Workflow optimization failed", cause=e)
    
    async def get_learning_progression(self, topic: str, time_range_days: int = 180) -> Dict[str, Any]:
        """Track learning progression on a specific topic"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create cache key
            cache_key = f"progression_{topic}_{time_range_days}"
            
            # Try cache first
            cached_progression = self.progression_cache.get(cache_key)
            if cached_progression is not None:
                return cached_progression
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=time_range_days)
            
            # Query for topic-related memories
            topic_memories = await self.persistence_domain.retrieve_memories(
                query=f"topic:{topic} learning command usage",
                types=["command_execution", "autocode_session", "learning_event"],
                limit=1000,
                min_similarity=0.6
            )
            
            # Filter by date range
            relevant_memories = []
            for memory in topic_memories:
                try:
                    created_at = datetime.fromisoformat(memory.get("created_at", ""))
                    if start_date <= created_at <= end_date:
                        relevant_memories.append(memory)
                except (ValueError, KeyError):
                    continue
            
            # Analyze progression
            progression = {
                "topic": topic,
                "time_range_days": time_range_days,
                "total_interactions": len(relevant_memories),
                "learning_milestones": [],
                "skill_development": {},
                "knowledge_gaps": [],
                "suggested_next_steps": [],
                "progression_score": 0.0,
                "weekly_activity": self._calculate_weekly_activity(relevant_memories, start_date),
                "command_mastery": {},
                "error_patterns": {}
            }
            
            # Calculate skill development
            if relevant_memories:
                progression["skill_development"] = await self._analyze_skill_development(
                    relevant_memories, topic
                )
                progression["command_mastery"] = self._analyze_command_mastery(relevant_memories)
                progression["error_patterns"] = self._analyze_error_patterns(relevant_memories)
                progression["learning_milestones"] = self._identify_learning_milestones(relevant_memories)
                progression["progression_score"] = self._calculate_progression_score(progression)
                progression["knowledge_gaps"] = await self._identify_knowledge_gaps(topic, progression)
                progression["suggested_next_steps"] = await self._suggest_next_learning_steps(topic, progression)
            
            # Cache the progression
            self.progression_cache.set(cache_key, progression)
            
            logger.debug(f"Calculated learning progression for topic: {topic}")
            return progression
            
        except (MemoryOperationError, ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to get learning progression: {e}")
            raise AutoCodeError("Learning progression calculation failed", cause=e)
    
    async def process_bash_execution(self, command: str, working_directory: str, 
                                   success: bool, output: str, project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process bash command execution for learning"""
        if not self._initialized:
            await self.initialize()
        
        try:
            execution_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "command": command,
                "working_directory": working_directory,
                "success": success,
                "output": output[:1000] if output else "",  # Limit output size
                "project_context": project_context or {},
                "command_category": self._categorize_command(command),
                "estimated_duration": self._estimate_command_duration(command),
                "risk_level": self._assess_command_risk(command)
            }
            
            # Add to command history
            self._command_history.append(execution_event)
            
            # Keep only recent history in memory
            if len(self._command_history) > 1000:
                self._command_history = self._command_history[-1000:]
            
            # Update learning statistics
            self._learning_stats["commands_processed"] += 1
            if success:
                self._learning_stats["successful_suggestions"] += 1
            else:
                self._learning_stats["failed_suggestions"] += 1
            
            # Use command learner to process execution
            if self.command_learner:
                await self.command_learner.track_execution(
                    command, working_directory, success, output, project_context
                )
            
            # Store execution for learning analysis
            await self.persistence_domain.store_memory(
                memory_type="command_execution",
                content=execution_event,
                importance=0.6 if success else 0.8,  # Failed commands are more important for learning
                metadata={
                    "command_category": execution_event["command_category"],
                    "success": success,
                    "risk_level": execution_event["risk_level"]
                }
            )
            
            logger.debug(f"Processed bash execution: {command[:50]}... (success: {success})")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to process bash execution: {e}")
    
    async def _load_learning_history(self) -> None:
        """Load existing learning history"""
        try:
            # Load recent command executions
            recent_commands = await self.persistence_domain.retrieve_memories(
                query="command execution bash",
                types=["command_execution"],
                limit=500
            )
            
            for memory in recent_commands[-100:]:  # Keep only last 100
                content = memory.get("content", {})
                if isinstance(content, dict) and "command" in content:
                    self._command_history.append(content)
            
            logger.info(f"Loaded {len(self._command_history)} command history entries")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to load learning history: {e}")
    
    async def _save_learning_stats(self) -> None:
        """Save learning statistics"""
        try:
            stats_data = {
                "stats": self._learning_stats,
                "timestamp": datetime.utcnow().isoformat(),
                "command_history_size": len(self._command_history)
            }
            
            await self.persistence_domain.store_memory(
                memory_type="learning_stats",
                content=stats_data,
                importance=0.5,
                metadata={"stats_type": "learning_engine"}
            )
            
            logger.debug("Saved learning statistics")
            
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to save learning stats: {e}")
    
    def _categorize_command(self, command: str) -> str:
        """Categorize a command by type"""
        command_lower = command.lower().strip()
        
        if any(cmd in command_lower for cmd in ['git']):
            return "version_control"
        elif any(cmd in command_lower for cmd in ['npm', 'yarn', 'pip', 'cargo', 'go mod']):
            return "package_management"
        elif any(cmd in command_lower for cmd in ['test', 'pytest', 'jest']):
            return "testing"
        elif any(cmd in command_lower for cmd in ['build', 'compile', 'make']):
            return "build"
        elif any(cmd in command_lower for cmd in ['docker', 'kubectl']):
            return "containerization"
        elif any(cmd in command_lower for cmd in ['ls', 'cd', 'mkdir', 'rm', 'mv', 'cp']):
            return "file_system"
        elif any(cmd in command_lower for cmd in ['grep', 'find', 'awk', 'sed']):
            return "text_processing"
        else:
            return "general"
    
    def _assess_command_risk(self, command: str) -> str:
        """Assess the risk level of a command"""
        command_lower = command.lower().strip()
        
        # High risk commands
        if any(dangerous in command_lower for dangerous in [
            'rm -rf', 'sudo rm', 'format', 'fdisk', 'dd if=', '> /dev/', 'chmod 777'
        ]):
            return "high"
        
        # Medium risk commands
        elif any(risky in command_lower for risky in [
            'sudo', 'rm ', 'mv ', 'chmod', 'chown', 'kill -9'
        ]):
            return "medium"
        
        # Low risk commands
        else:
            return "low"
    
    def _estimate_command_duration(self, command: str) -> str:
        """Estimate how long a command typically takes"""
        command_lower = command.lower().strip()
        
        if any(cmd in command_lower for cmd in ['build', 'compile', 'test', 'npm install']):
            return "long"  # > 30 seconds
        elif any(cmd in command_lower for cmd in ['git clone', 'download', 'upload']):
            return "medium"  # 5-30 seconds
        else:
            return "short"  # < 5 seconds
    
    # Additional helper methods would continue here...
    # For brevity, I'll include just a few more key methods
    
    async def _generate_basic_suggestions(self, intent: str, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate basic suggestions when command learner is not available"""
        suggestions = []
        
        intent_lower = intent.lower()
        
        # Basic command mapping
        if "test" in intent_lower:
            suggestions.append({
                "command": "pytest",
                "confidence": 0.7,
                "reasoning": "Standard Python testing framework"
            })
        elif "install" in intent_lower:
            suggestions.append({
                "command": "pip install -r requirements.txt",
                "confidence": 0.6,
                "reasoning": "Common Python dependency installation"
            })
        elif "status" in intent_lower or "check" in intent_lower:
            suggestions.append({
                "command": "git status",
                "confidence": 0.8,
                "reasoning": "Check version control status"
            })
        
        return suggestions
    
    def _find_redundant_commands(self, workflow: List[str]) -> List[str]:
        """Find redundant commands in workflow"""
        redundant = []
        seen_commands = set()
        
        for command in workflow:
            # Normalize command (remove arguments for comparison)
            base_command = command.split()[0] if command.split() else command
            
            if base_command in seen_commands:
                redundant.append(command)
            else:
                seen_commands.add(base_command)
        
        return redundant
    
    def _find_combinable_commands(self, workflow: List[str]) -> List[Dict[str, Any]]:
        """Find commands that can be combined"""
        combinations = []
        
        # Look for patterns like cd + command
        for i in range(len(workflow) - 1):
            current = workflow[i].strip()
            next_cmd = workflow[i + 1].strip()
            
            if current.startswith('cd ') and not next_cmd.startswith('cd '):
                combinations.append({
                    "commands": [current, next_cmd],
                    "combined": f"{current} && {next_cmd}",
                    "type": "sequential"
                })
        
        return combinations