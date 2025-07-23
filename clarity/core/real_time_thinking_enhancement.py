"""
Real-Time Thinking Enhancement with MCP Integration.

This module provides real-time suggestions, workflow pattern detection, and
live memory correlation during thinking processes. It integrates with hook
systems to provide contextual suggestions as the user progresses through
different thinking stages.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.mcp_memory_retrieval import get_mcp_memory_retriever, MCPContext, RetrievalStrategy
from ..mcp.cache_integration import get_mcp_cache_adapter
from ..core.unified_cache import cache_get, cache_put, CacheType

logger = logging.getLogger(__name__)


class ThinkingStage(str, Enum):
    """Thinking stages for real-time enhancement."""
    INITIALIZATION = "initialization"
    PROBLEM_DEFINITION = "problem_definition"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"


class SuggestionType(str, Enum):
    """Types of real-time suggestions."""
    WORKFLOW_PATTERN = "workflow_pattern"
    TOOL_RECOMMENDATION = "tool_recommendation"
    MEMORY_CORRELATION = "memory_correlation"
    SIMILAR_SOLUTION = "similar_solution"
    OPTIMIZATION_TIP = "optimization_tip"
    BEST_PRACTICE = "best_practice"
    WARNING = "warning"


class SuggestionPriority(str, Enum):
    """Priority levels for suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThinkingContext:
    """Current thinking context for real-time enhancement."""
    session_id: str
    current_stage: ThinkingStage
    previous_stages: List[ThinkingStage] = field(default_factory=list)
    current_tools: List[str] = field(default_factory=list)
    project_context: Dict[str, Any] = field(default_factory=dict)
    user_intent: str = ""
    current_problem: str = ""
    progress_metrics: Dict[str, float] = field(default_factory=dict)
    session_start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    accumulated_insights: List[Dict[str, Any]] = field(default_factory=list)
    workflow_patterns_detected: Set[str] = field(default_factory=set)


@dataclass
class RealTimeSuggestion:
    """Real-time suggestion with context and metadata."""
    suggestion_id: str
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    title: str
    content: str
    context: Dict[str, Any]
    confidence: float
    source_data: Dict[str, Any]
    thinking_stage: ThinkingStage
    tools_related: List[str] = field(default_factory=list)
    memory_references: List[str] = field(default_factory=list)
    workflow_pattern: Optional[str] = None
    expiry_time: Optional[float] = None
    action_suggestions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class RealTimeThinkingEnhancer:
    """Real-time thinking enhancement with MCP integration and live suggestions."""
    
    def __init__(self, domain_manager, mcp_memory_retriever=None):
        """Initialize real-time thinking enhancer."""
        self.domain_manager = domain_manager
        self.mcp_memory_retriever = mcp_memory_retriever or get_mcp_memory_retriever(domain_manager)
        self.mcp_cache_adapter = get_mcp_cache_adapter()
        
        # Active thinking contexts
        self.active_contexts: Dict[str, ThinkingContext] = {}
        
        # Suggestion generation settings
        self.suggestion_settings = {
            "max_suggestions_per_stage": 5,
            "suggestion_confidence_threshold": 0.6,
            "memory_correlation_threshold": 0.7,
            "workflow_detection_threshold": 0.5,  # Lowered for better detection
            "background_processing_enabled": True,
            "real_time_monitoring_enabled": True
        }
        
        # Hook registry for different thinking events
        self.hooks = {
            "stage_transition": [],
            "tool_usage": [],
            "problem_updated": [],
            "insight_added": [],
            "workflow_detected": [],
            "suggestion_generated": []
        }
        
        # Background processing
        self.background_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rt_thinking")
        self.background_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.enhancement_stats = {
            "total_suggestions": 0,
            "suggestions_by_type": {},
            "suggestions_by_priority": {},
            "average_generation_time": 0.0,
            "hook_execution_count": 0,
            "successful_correlations": 0,
            "workflow_patterns_detected": 0
        }
    
    async def start_thinking_session(
        self,
        session_id: str,
        initial_problem: str = "",
        project_context: Dict[str, Any] = None,
        user_intent: str = "",
        enable_real_time: bool = True
    ) -> ThinkingContext:
        """Start a new real-time enhanced thinking session."""
        
        thinking_context = ThinkingContext(
            session_id=session_id,
            current_stage=ThinkingStage.INITIALIZATION,
            current_problem=initial_problem,
            project_context=project_context or {},
            user_intent=user_intent
        )
        
        self.active_contexts[session_id] = thinking_context
        
        # Start background monitoring if enabled
        if enable_real_time and self.suggestion_settings["real_time_monitoring_enabled"]:
            self.background_tasks[session_id] = asyncio.create_task(
                self._background_session_monitoring(session_id)
            )
        
        # Generate initial suggestions
        initial_suggestions = await self.generate_stage_suggestions(
            session_id, ThinkingStage.INITIALIZATION
        )
        
        # Trigger hooks
        await self._trigger_hooks("stage_transition", {
            "session_id": session_id,
            "new_stage": ThinkingStage.INITIALIZATION,
            "previous_stage": None,
            "context": thinking_context,
            "suggestions": initial_suggestions
        })
        
        logger.info(f"Started real-time thinking session: {session_id}")
        return thinking_context
    
    async def transition_thinking_stage(
        self,
        session_id: str,
        new_stage: ThinkingStage,
        stage_insights: Dict[str, Any] = None
    ) -> List[RealTimeSuggestion]:
        """Transition to a new thinking stage with real-time suggestions."""
        
        if session_id not in self.active_contexts:
            raise ValueError(f"No active thinking context for session: {session_id}")
        
        context = self.active_contexts[session_id]
        previous_stage = context.current_stage
        
        # Update context
        context.previous_stages.append(previous_stage)
        context.current_stage = new_stage
        context.last_update_time = time.time()
        
        # Add stage insights
        if stage_insights:
            context.accumulated_insights.append({
                "stage": previous_stage.value,
                "insights": stage_insights,
                "timestamp": time.time()
            })
        
        # Generate stage-specific suggestions
        suggestions = await self.generate_stage_suggestions(session_id, new_stage)
        
        # Detect workflow patterns
        await self._detect_workflow_patterns(session_id)
        
        # Trigger hooks
        await self._trigger_hooks("stage_transition", {
            "session_id": session_id,
            "new_stage": new_stage,
            "previous_stage": previous_stage,
            "context": context,
            "suggestions": suggestions,
            "stage_insights": stage_insights
        })
        
        logger.info(f"Session {session_id} transitioned: {previous_stage.value} → {new_stage.value}")
        return suggestions
    
    async def update_tool_context(
        self,
        session_id: str,
        tools_used: List[str],
        tool_results: Dict[str, Any] = None
    ) -> List[RealTimeSuggestion]:
        """Update tool usage context and generate tool-related suggestions."""
        
        if session_id not in self.active_contexts:
            return []
        
        context = self.active_contexts[session_id]
        context.current_tools = tools_used
        context.last_update_time = time.time()
        
        # Generate tool correlation suggestions
        suggestions = await self._generate_tool_correlation_suggestions(session_id, tools_used, tool_results)
        
        # Trigger hooks
        await self._trigger_hooks("tool_usage", {
            "session_id": session_id,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "context": context,
            "suggestions": suggestions
        })
        
        return suggestions
    
    async def update_problem_context(
        self,
        session_id: str,
        updated_problem: str,
        problem_insights: Dict[str, Any] = None
    ) -> List[RealTimeSuggestion]:
        """Update problem context and generate related suggestions."""
        
        if session_id not in self.active_contexts:
            return []
        
        context = self.active_contexts[session_id]
        context.current_problem = updated_problem
        context.last_update_time = time.time()
        
        if problem_insights:
            context.accumulated_insights.append({
                "type": "problem_update",
                "insights": problem_insights,
                "timestamp": time.time()
            })
        
        # Generate problem-related suggestions
        suggestions = await self._generate_problem_related_suggestions(session_id, updated_problem)
        
        # Trigger hooks
        await self._trigger_hooks("problem_updated", {
            "session_id": session_id,
            "updated_problem": updated_problem,
            "problem_insights": problem_insights,
            "context": context,
            "suggestions": suggestions
        })
        
        return suggestions
    
    async def generate_stage_suggestions(
        self,
        session_id: str,
        stage: ThinkingStage
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions specific to a thinking stage."""
        
        if session_id not in self.active_contexts:
            return []
        
        context = self.active_contexts[session_id]
        suggestions = []
        
        try:
            # Create MCP context for retrieval
            mcp_context = MCPContext(
                current_tools=context.current_tools,
                thinking_stage=stage.value,
                project_context=context.project_context,
                session_context={
                    "session_id": session_id,
                    "progress": len(context.previous_stages) / len(ThinkingStage),
                    "session_duration": time.time() - context.session_start_time
                },
                user_intent=context.user_intent
            )
            
            # Stage-specific suggestion generation
            if stage == ThinkingStage.PROBLEM_DEFINITION:
                suggestions.extend(await self._generate_problem_definition_suggestions(context, mcp_context))
            elif stage == ThinkingStage.RESEARCH:
                suggestions.extend(await self._generate_research_suggestions(context, mcp_context))
            elif stage == ThinkingStage.ANALYSIS:
                suggestions.extend(await self._generate_analysis_suggestions(context, mcp_context))
            elif stage == ThinkingStage.SYNTHESIS:
                suggestions.extend(await self._generate_synthesis_suggestions(context, mcp_context))
            elif stage == ThinkingStage.CONCLUSION:
                suggestions.extend(await self._generate_conclusion_suggestions(context, mcp_context))
            elif stage == ThinkingStage.IMPLEMENTATION:
                suggestions.extend(await self._generate_implementation_suggestions(context, mcp_context))
            
            # Add general workflow pattern suggestions
            workflow_suggestions = await self._generate_workflow_pattern_suggestions(context, mcp_context)
            suggestions.extend(workflow_suggestions)
            
            # Sort by priority and confidence
            suggestions = sorted(suggestions, key=lambda s: (s.priority.value, -s.confidence))
            
            # Limit suggestions
            max_suggestions = self.suggestion_settings["max_suggestions_per_stage"]
            suggestions = suggestions[:max_suggestions]
            
            # Update statistics
            self.enhancement_stats["total_suggestions"] += len(suggestions)
            for suggestion in suggestions:
                suggestion_type = suggestion.suggestion_type.value
                priority = suggestion.priority.value
                self.enhancement_stats["suggestions_by_type"][suggestion_type] = \
                    self.enhancement_stats["suggestions_by_type"].get(suggestion_type, 0) + 1
                self.enhancement_stats["suggestions_by_priority"][priority] = \
                    self.enhancement_stats["suggestions_by_priority"].get(priority, 0) + 1
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate stage suggestions for {session_id}: {e}")
            return []
    
    async def _generate_problem_definition_suggestions(
        self,
        context: ThinkingContext,
        mcp_context: MCPContext
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions for problem definition stage."""
        suggestions = []
        
        # Search for similar problems
        if context.current_problem:
            similar_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
                f"problem definition: {context.current_problem}",
                mcp_context,
                limit=3,
                strategy=RetrievalStrategy.ADAPTIVE_INTELLIGENT
            )
            
            if similar_memories:
                suggestions.append(RealTimeSuggestion(
                    suggestion_id=f"similar_problems_{context.session_id}_{int(time.time())}",
                    suggestion_type=SuggestionType.SIMILAR_SOLUTION,
                    priority=SuggestionPriority.HIGH,
                    title="Similar Problems Found",
                    content=f"Found {len(similar_memories)} similar problems in memory that might provide insights",
                    context={"similar_count": len(similar_memories)},
                    confidence=0.8,
                    source_data={"memories": [m.memory_id for m in similar_memories]},
                    thinking_stage=ThinkingStage.PROBLEM_DEFINITION,
                    memory_references=[m.memory_id for m in similar_memories],
                    action_suggestions=[
                        "Review similar problem approaches",
                        "Identify common patterns in problem-solving",
                        "Consider validated solution strategies"
                    ]
                ))
        
        # Suggest problem breakdown techniques
        suggestions.append(RealTimeSuggestion(
            suggestion_id=f"problem_breakdown_{context.session_id}_{int(time.time())}",
            suggestion_type=SuggestionType.BEST_PRACTICE,
            priority=SuggestionPriority.MEDIUM,
            title="Problem Breakdown Techniques",
            content="Consider breaking down the problem using structured approaches like 5W1H, root cause analysis, or problem trees",
            context={"technique_types": ["5W1H", "root_cause", "problem_tree"]},
            confidence=0.9,
            source_data={"best_practice": "problem_decomposition"},
            thinking_stage=ThinkingStage.PROBLEM_DEFINITION,
            action_suggestions=[
                "Apply 5W1H framework (Who, What, When, Where, Why, How)",
                "Use root cause analysis techniques",
                "Create a problem tree or mind map"
            ]
        ))
        
        return suggestions
    
    async def _generate_research_suggestions(
        self,
        context: ThinkingContext,
        mcp_context: MCPContext
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions for research stage."""
        suggestions = []
        
        # Find relevant research patterns
        research_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            f"research methodology: {context.current_problem}",
            mcp_context,
            limit=3,
            strategy=RetrievalStrategy.MCP_ENHANCED
        )
        
        if research_memories:
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"research_methods_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.WORKFLOW_PATTERN,
                priority=SuggestionPriority.HIGH,
                title="Research Methodology Patterns",
                content=f"Found {len(research_memories)} research patterns that could guide your investigation",
                context={"research_patterns": len(research_memories)},
                confidence=0.85,
                source_data={"research_memories": [m.memory_id for m in research_memories]},
                thinking_stage=ThinkingStage.RESEARCH,
                memory_references=[m.memory_id for m in research_memories],
                action_suggestions=[
                    "Review successful research approaches",
                    "Adapt proven methodologies to current problem",
                    "Identify key information sources"
                ]
            ))
        
        # Suggest research tools if none are being used
        if not context.current_tools:
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"research_tools_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
                priority=SuggestionPriority.MEDIUM,
                title="Research Tool Recommendations",
                content="Consider using research tools to enhance your investigation",
                context={"stage": "research", "tools_needed": True},
                confidence=0.7,
                source_data={"tool_category": "research"},
                thinking_stage=ThinkingStage.RESEARCH,
                tools_related=["web_search", "documentation_reader", "database_query"],
                action_suggestions=[
                    "Use web search tools for background information",
                    "Access documentation and reference materials",
                    "Query databases for relevant data"
                ]
            ))
        
        return suggestions
    
    async def _generate_analysis_suggestions(
        self,
        context: ThinkingContext,
        mcp_context: MCPContext
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions for analysis stage."""
        suggestions = []
        
        # Find analysis patterns
        analysis_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            f"analysis techniques: {context.current_problem}",
            mcp_context,
            limit=3,
            strategy=RetrievalStrategy.CROSS_SYSTEM_CORRELATION
        )
        
        if analysis_memories:
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"analysis_patterns_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.WORKFLOW_PATTERN,
                priority=SuggestionPriority.HIGH,
                title="Analysis Pattern Recommendations",
                content=f"Discovered {len(analysis_memories)} analysis patterns that match your context",
                context={"analysis_patterns": len(analysis_memories)},
                confidence=0.88,
                source_data={"analysis_memories": [m.memory_id for m in analysis_memories]},
                thinking_stage=ThinkingStage.ANALYSIS,
                memory_references=[m.memory_id for m in analysis_memories],
                action_suggestions=[
                    "Apply proven analysis frameworks",
                    "Use structured analytical techniques",
                    "Consider multiple analytical perspectives"
                ]
            ))
        
        # Suggest analysis tools
        if "analyzer" not in str(context.current_tools):
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"analysis_tools_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
                priority=SuggestionPriority.MEDIUM,
                title="Analysis Tool Suggestions",
                content="Analysis tools could help structure and validate your analytical thinking",
                context={"stage": "analysis", "tools_available": True},
                confidence=0.75,
                source_data={"tool_category": "analysis"},
                thinking_stage=ThinkingStage.ANALYSIS,
                tools_related=["data_analyzer", "pattern_detector", "correlation_finder"],
                action_suggestions=[
                    "Use data analysis tools for pattern detection",
                    "Apply correlation analysis techniques",
                    "Validate findings with analytical tools"
                ]
            ))
        
        return suggestions
    
    async def _generate_synthesis_suggestions(
        self,
        context: ThinkingContext,
        mcp_context: MCPContext
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions for synthesis stage."""
        suggestions = []
        
        # Check if enough insights have been accumulated
        if len(context.accumulated_insights) < 2:
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"synthesis_warning_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.WARNING,
                priority=SuggestionPriority.HIGH,
                title="Limited Insights for Synthesis",
                content="Consider gathering more insights before synthesis. Current insights might be insufficient for comprehensive synthesis.",
                context={"insights_count": len(context.accumulated_insights), "recommended_minimum": 3},
                confidence=0.9,
                source_data={"current_insights": len(context.accumulated_insights)},
                thinking_stage=ThinkingStage.SYNTHESIS,
                action_suggestions=[
                    "Return to analysis or research stages",
                    "Gather additional insights and data points",
                    "Consider different perspectives and approaches"
                ]
            ))
        
        # Find synthesis patterns
        synthesis_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            "synthesis techniques solution integration",
            mcp_context,
            limit=2,
            strategy=RetrievalStrategy.WORKFLOW_PATTERN
        )
        
        if synthesis_memories:
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"synthesis_patterns_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.WORKFLOW_PATTERN,
                priority=SuggestionPriority.MEDIUM,
                title="Synthesis Methodology Patterns",
                content=f"Found {len(synthesis_memories)} synthesis approaches that could guide integration",
                context={"synthesis_patterns": len(synthesis_memories)},
                confidence=0.8,
                source_data={"synthesis_memories": [m.memory_id for m in synthesis_memories]},
                thinking_stage=ThinkingStage.SYNTHESIS,
                memory_references=[m.memory_id for m in synthesis_memories],
                action_suggestions=[
                    "Apply systematic synthesis methodologies",
                    "Integrate insights using proven frameworks",
                    "Consider multiple synthesis perspectives"
                ]
            ))
        
        return suggestions
    
    async def _generate_conclusion_suggestions(
        self,
        context: ThinkingContext,
        mcp_context: MCPContext
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions for conclusion stage."""
        suggestions = []
        
        # Check thinking process completeness
        required_stages = {ThinkingStage.PROBLEM_DEFINITION, ThinkingStage.RESEARCH, ThinkingStage.ANALYSIS}
        completed_stages = set(context.previous_stages + [context.current_stage])
        missing_stages = required_stages - completed_stages
        
        if missing_stages:
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"completeness_check_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.WARNING,
                priority=SuggestionPriority.CRITICAL,
                title="Incomplete Thinking Process",
                content=f"Missing critical thinking stages: {', '.join(s.value for s in missing_stages)}",
                context={"missing_stages": [s.value for s in missing_stages]},
                confidence=0.95,
                source_data={"completeness_check": True},
                thinking_stage=ThinkingStage.CONCLUSION,
                action_suggestions=[
                    f"Complete missing stages: {', '.join(s.value for s in missing_stages)}",
                    "Ensure thorough analysis before concluding",
                    "Review thinking process for gaps"
                ]
            ))
        
        # Suggest decision validation
        suggestions.append(RealTimeSuggestion(
            suggestion_id=f"decision_validation_{context.session_id}_{int(time.time())}",
            suggestion_type=SuggestionType.BEST_PRACTICE,
            priority=SuggestionPriority.HIGH,
            title="Decision Validation Recommendations",
            content="Consider validating your conclusions using structured decision-making frameworks",
            context={"validation_techniques": ["pros_cons", "decision_matrix", "scenario_analysis"]},
            confidence=0.85,
            source_data={"decision_validation": True},
            thinking_stage=ThinkingStage.CONCLUSION,
            action_suggestions=[
                "Create pros and cons analysis",
                "Use decision matrix for complex choices",
                "Consider scenario analysis for validation"
            ]
        ))
        
        return suggestions
    
    async def _generate_implementation_suggestions(
        self,
        context: ThinkingContext,
        mcp_context: MCPContext
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions for implementation stage."""
        suggestions = []
        
        # Find implementation patterns
        impl_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            f"implementation plan execution: {context.current_problem}",
            mcp_context,
            limit=3,
            strategy=RetrievalStrategy.WORKFLOW_PATTERN
        )
        
        if impl_memories:
            suggestions.append(RealTimeSuggestion(
                suggestion_id=f"implementation_patterns_{context.session_id}_{int(time.time())}",
                suggestion_type=SuggestionType.WORKFLOW_PATTERN,
                priority=SuggestionPriority.HIGH,
                title="Implementation Pattern Guidance",
                content=f"Found {len(impl_memories)} implementation patterns for similar contexts",
                context={"implementation_patterns": len(impl_memories)},
                confidence=0.82,
                source_data={"impl_memories": [m.memory_id for m in impl_memories]},
                thinking_stage=ThinkingStage.IMPLEMENTATION,
                memory_references=[m.memory_id for m in impl_memories],
                action_suggestions=[
                    "Follow proven implementation methodologies",
                    "Break down implementation into manageable steps",
                    "Plan for monitoring and adjustment"
                ]
            ))
        
        # Suggest implementation tools
        suggestions.append(RealTimeSuggestion(
            suggestion_id=f"implementation_tools_{context.session_id}_{int(time.time())}",
            suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
            priority=SuggestionPriority.MEDIUM,
            title="Implementation Tool Recommendations",
            content="Implementation tools can help execute your plan systematically",
            context={"stage": "implementation", "tools_suggested": True},
            confidence=0.7,
            source_data={"tool_category": "implementation"},
            thinking_stage=ThinkingStage.IMPLEMENTATION,
            tools_related=["project_manager", "task_tracker", "deployment_tool"],
            action_suggestions=[
                "Use project management tools for planning",
                "Track implementation progress",
                "Utilize deployment and execution tools"
            ]
        ))
        
        return suggestions
    
    async def _generate_workflow_pattern_suggestions(
        self,
        context: ThinkingContext,
        mcp_context: MCPContext
    ) -> List[RealTimeSuggestion]:
        """Generate workflow pattern suggestions based on current context."""
        suggestions = []
        
        # Search for workflow patterns
        workflow_query = f"{context.current_stage.value} workflow patterns"
        if context.current_tools:
            workflow_query += f" with tools: {' '.join(context.current_tools)}"
        
        workflow_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            workflow_query,
            mcp_context,
            limit=2,
            strategy=RetrievalStrategy.WORKFLOW_PATTERN
        )
        
        for memory in workflow_memories:
            # Use final_score instead of confidence for RetrievalResult objects
            confidence_score = getattr(memory, 'final_score', getattr(memory, 'confidence', 0.0))
            if confidence_score > self.suggestion_settings["workflow_detection_threshold"]:
                suggestions.append(RealTimeSuggestion(
                    suggestion_id=f"workflow_pattern_{memory.memory_id}_{int(time.time())}",
                    suggestion_type=SuggestionType.WORKFLOW_PATTERN,
                    priority=SuggestionPriority.MEDIUM,
                    title="Workflow Pattern Match",
                    content=f"Detected workflow pattern that matches your current context",
                    context={"pattern_confidence": confidence_score},
                    confidence=confidence_score,
                    source_data={"workflow_memory": memory.memory_id},
                    thinking_stage=context.current_stage,
                    workflow_pattern=memory.memory_id,
                    memory_references=[memory.memory_id],
                    action_suggestions=[
                        "Review workflow pattern details",
                        "Adapt pattern to current context",
                        "Follow proven workflow steps"
                    ]
                ))
        
        return suggestions
    
    async def _generate_tool_correlation_suggestions(
        self,
        session_id: str,
        tools_used: List[str],
        tool_results: Dict[str, Any] = None
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions based on tool usage correlation."""
        suggestions = []
        
        if session_id not in self.active_contexts:
            return suggestions
        
        context = self.active_contexts[session_id]
        
        # Find tool correlation memories
        tool_query = f"tool correlation: {' '.join(tools_used)}"
        mcp_context = MCPContext(
            current_tools=tools_used,
            thinking_stage=context.current_stage.value,
            project_context=context.project_context
        )
        
        correlation_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            tool_query,
            mcp_context,
            limit=3,
            strategy=RetrievalStrategy.CROSS_SYSTEM_CORRELATION
        )
        
        for memory in correlation_memories:
            if memory.memory_type == "mcp_tool_correlation":
                suggestions.append(RealTimeSuggestion(
                    suggestion_id=f"tool_correlation_{memory.memory_id}_{int(time.time())}",
                    suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
                    priority=SuggestionPriority.MEDIUM,
                    title="Tool Correlation Insight",
                    content=f"Tool usage pattern suggests additional tools or approaches",
                    context={"correlation_found": True},
                    confidence=getattr(memory, 'cross_system_confidence', getattr(memory, 'final_score', 0.7)),
                    source_data={"correlation_memory": memory.memory_id},
                    thinking_stage=context.current_stage,
                    tools_related=tools_used,
                    memory_references=[memory.memory_id],
                    action_suggestions=[
                        "Review correlated tool recommendations",
                        "Consider complementary tools",
                        "Apply tool usage best practices"
                    ]
                ))
        
        return suggestions
    
    async def _generate_problem_related_suggestions(
        self,
        session_id: str,
        problem: str
    ) -> List[RealTimeSuggestion]:
        """Generate suggestions related to problem updates."""
        suggestions = []
        
        if session_id not in self.active_contexts:
            return suggestions
        
        context = self.active_contexts[session_id]
        
        # Search for similar problems
        mcp_context = MCPContext(
            current_tools=context.current_tools,
            thinking_stage=context.current_stage.value,
            project_context=context.project_context,
            user_intent=context.user_intent
        )
        
        similar_problems = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            problem,
            mcp_context,
            limit=3,
            strategy=RetrievalStrategy.ADAPTIVE_INTELLIGENT
        )
        
        if similar_problems:
            high_confidence_problems = [p for p in similar_problems if getattr(p, 'final_score', getattr(p, 'confidence', 0.0)) > 0.8]
            
            if high_confidence_problems:
                suggestions.append(RealTimeSuggestion(
                    suggestion_id=f"similar_problem_{session_id}_{int(time.time())}",
                    suggestion_type=SuggestionType.SIMILAR_SOLUTION,
                    priority=SuggestionPriority.HIGH,
                    title="Highly Similar Problem Found",
                    content=f"Found {len(high_confidence_problems)} very similar problems with solutions",
                    context={"high_confidence_matches": len(high_confidence_problems)},
                    confidence=max(getattr(p, 'final_score', getattr(p, 'confidence', 0.8)) for p in high_confidence_problems),
                    source_data={"similar_problems": [p.memory_id for p in high_confidence_problems]},
                    thinking_stage=context.current_stage,
                    memory_references=[p.memory_id for p in high_confidence_problems],
                    action_suggestions=[
                        "Review similar problem solutions",
                        "Adapt successful approaches to current context",
                        "Avoid known pitfalls and challenges"
                    ]
                ))
        
        return suggestions
    
    async def _detect_workflow_patterns(self, session_id: str) -> None:
        """Detect workflow patterns based on thinking progression."""
        if session_id not in self.active_contexts:
            return
        
        context = self.active_contexts[session_id]
        
        # Analyze stage progression pattern
        stage_sequence = [s.value for s in context.previous_stages] + [context.current_stage.value]
        
        # Check for common workflow patterns
        workflow_patterns = {
            "linear_progression": ["initialization", "problem_definition", "research", "analysis", "synthesis", "conclusion"],
            "iterative_refinement": ["analysis", "research", "analysis"],
            "rapid_prototyping": ["problem_definition", "implementation", "analysis", "implementation"]
        }
        
        for pattern_name, pattern_sequence in workflow_patterns.items():
            if self._sequence_matches_pattern(stage_sequence, pattern_sequence):
                if pattern_name not in context.workflow_patterns_detected:
                    context.workflow_patterns_detected.add(pattern_name)
                    self.enhancement_stats["workflow_patterns_detected"] += 1
                    
                    # Trigger workflow detection hook
                    await self._trigger_hooks("workflow_detected", {
                        "session_id": session_id,
                        "pattern_name": pattern_name,
                        "pattern_sequence": pattern_sequence,
                        "current_sequence": stage_sequence,
                        "context": context
                    })
    
    def _sequence_matches_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if a sequence contains a pattern."""
        if len(pattern) > len(sequence):
            return False
        
        # Check if pattern appears as subsequence
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        
        return False
    
    async def _background_session_monitoring(self, session_id: str) -> None:
        """Background monitoring for a thinking session."""
        while session_id in self.active_contexts:
            try:
                context = self.active_contexts[session_id]
                
                # Check for stale sessions (no updates for too long)
                time_since_update = time.time() - context.last_update_time
                if time_since_update > 3600:  # 1 hour
                    logger.info(f"Ending stale thinking session: {session_id}")
                    break
                
                # Generate periodic suggestions if session is active
                if time_since_update < 300:  # Within last 5 minutes
                    periodic_suggestions = await self._generate_periodic_suggestions(session_id)
                    if periodic_suggestions:
                        await self._trigger_hooks("suggestion_generated", {
                            "session_id": session_id,
                            "suggestions": periodic_suggestions,
                            "trigger": "periodic_monitoring"
                        })
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitoring error for {session_id}: {e}")
                await asyncio.sleep(60)
    
    async def _generate_periodic_suggestions(self, session_id: str) -> List[RealTimeSuggestion]:
        """Generate periodic suggestions for active sessions."""
        # For now, return empty list - can be enhanced with time-based suggestions
        return []
    
    def register_hook(self, event_type: str, hook_function: Callable) -> None:
        """Register a hook function for thinking events."""
        if event_type in self.hooks:
            self.hooks[event_type].append(hook_function)
        else:
            logger.warning(f"Unknown hook event type: {event_type}")
    
    def unregister_hook(self, event_type: str, hook_function: Callable) -> bool:
        """Unregister a hook function."""
        if event_type in self.hooks and hook_function in self.hooks[event_type]:
            self.hooks[event_type].remove(hook_function)
            return True
        return False
    
    async def _trigger_hooks(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger all registered hooks for an event type."""
        if event_type in self.hooks:
            for hook_function in self.hooks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(hook_function):
                        await hook_function(event_data)
                    else:
                        hook_function(event_data)
                    self.enhancement_stats["hook_execution_count"] += 1
                except Exception as e:
                    logger.error(f"Hook execution failed for {event_type}: {e}")
    
    async def end_thinking_session(self, session_id: str) -> Dict[str, Any]:
        """End a thinking session and return session summary."""
        if session_id not in self.active_contexts:
            return {"error": "Session not found"}
        
        context = self.active_contexts[session_id]
        
        # Cancel background task
        if session_id in self.background_tasks:
            self.background_tasks[session_id].cancel()
            del self.background_tasks[session_id]
        
        # Generate session summary
        session_summary = {
            "session_id": session_id,
            "duration": time.time() - context.session_start_time,
            "stages_completed": [s.value for s in context.previous_stages] + [context.current_stage.value],
            "tools_used": list(set(context.current_tools)) if context.current_tools else [],
            "insights_count": len(context.accumulated_insights),
            "workflow_patterns_detected": list(context.workflow_patterns_detected),
            "final_problem": context.current_problem,
            "project_context": context.project_context
        }
        
        # Store session summary as memory
        if self.domain_manager:
            try:
                await self.domain_manager.store_enhanced_context(
                    context_type="thinking_session_summary",
                    primary_content=session_summary,
                    mcp_correlations={
                        "workflow_patterns": list(context.workflow_patterns_detected),
                        "tools_correlation": context.current_tools
                    },
                    temporal_context={
                        "session_duration": session_summary["duration"],
                        "stages_progression": session_summary["stages_completed"]
                    },
                    importance=0.8,
                    metadata={"session_type": "real_time_enhanced_thinking"}
                )
            except Exception as e:
                logger.error(f"Failed to store session summary: {e}")
        
        # Clean up
        del self.active_contexts[session_id]
        
        logger.info(f"Ended thinking session: {session_id}")
        return session_summary
    
    async def get_enhancement_analytics(self) -> Dict[str, Any]:
        """Get real-time thinking enhancement analytics."""
        return {
            "performance_metrics": self.enhancement_stats.copy(),
            "active_sessions": len(self.active_contexts),
            "background_tasks": len(self.background_tasks),
            "suggestion_settings": self.suggestion_settings.copy(),
            "hook_registrations": {event_type: len(hooks) for event_type, hooks in self.hooks.items()},
            "session_summaries": {
                session_id: {
                    "current_stage": context.current_stage.value,
                    "duration": time.time() - context.session_start_time,
                    "stages_completed": len(context.previous_stages),
                    "tools_used": len(context.current_tools),
                    "insights": len(context.accumulated_insights)
                }
                for session_id, context in self.active_contexts.items()
            }
        }


# Global real-time thinking enhancer instance
_real_time_thinking_enhancer = None


def get_real_time_thinking_enhancer(domain_manager, mcp_memory_retriever=None) -> RealTimeThinkingEnhancer:
    """Get or create global real-time thinking enhancer."""
    global _real_time_thinking_enhancer
    if _real_time_thinking_enhancer is None:
        _real_time_thinking_enhancer = RealTimeThinkingEnhancer(domain_manager, mcp_memory_retriever)
    return _real_time_thinking_enhancer