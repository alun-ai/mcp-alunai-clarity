"""
Utility classes for analyzing and managing structured thinking processes.

This module provides analysis tools, memory mapping utilities, and helper functions
for integrating structured thinking with alunai-clarity's memory system.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
import statistics

from .structured_thinking import (
    StructuredThought, 
    ThinkingSession, 
    ThinkingStage, 
    ThinkingSummary,
    ThoughtRelationship,
    ThinkingPattern
)


class ThinkingAnalyzer:
    """Comprehensive analysis tools for structured thinking processes"""
    
    @staticmethod
    def analyze_thought_relationships(thoughts: List[StructuredThought]) -> Dict[str, Any]:
        """Analyze relationships between thoughts in a session"""
        if not thoughts:
            return {
                "total_relationships": 0,
                "relationship_distribution": {},
                "average_relationships_per_thought": 0.0,
                "most_connected_thought": None,
                "relationship_strength_stats": {}
            }
        
        total_relationships = sum(len(thought.relationships) for thought in thoughts)
        relationship_types = Counter()
        strength_values = []
        connection_counts = {}
        
        for thought in thoughts:
            connection_counts[thought.id] = len(thought.relationships)
            
            for rel in thought.relationships:
                relationship_types[rel.relationship_type] += 1
                strength_values.append(rel.strength)
        
        # Find most connected thought
        most_connected_thought = None
        if connection_counts:
            most_connected_id = max(connection_counts, key=connection_counts.get)
            most_connected_thought = next(
                (t for t in thoughts if t.id == most_connected_id), None
            )
        
        # Calculate strength statistics
        strength_stats = {}
        if strength_values:
            strength_stats = {
                "mean": statistics.mean(strength_values),
                "median": statistics.median(strength_values),
                "std_dev": statistics.stdev(strength_values) if len(strength_values) > 1 else 0.0,
                "min": min(strength_values),
                "max": max(strength_values)
            }
        
        return {
            "total_relationships": total_relationships,
            "relationship_distribution": dict(relationship_types),
            "average_relationships_per_thought": total_relationships / len(thoughts),
            "most_connected_thought": {
                "id": most_connected_thought.id if most_connected_thought else None,
                "thought_number": most_connected_thought.thought_number if most_connected_thought else None,
                "connection_count": connection_counts.get(most_connected_thought.id, 0) if most_connected_thought else 0
            },
            "relationship_strength_stats": strength_stats,
            "connection_distribution": dict(Counter(connection_counts.values()))
        }
    
    @staticmethod
    def generate_stage_summary(thoughts: List[StructuredThought], stage: ThinkingStage) -> str:
        """Generate summary for a specific thinking stage"""
        stage_thoughts = [t for t in thoughts if t.stage == stage]
        
        if not stage_thoughts:
            return f"No thoughts recorded for {stage.value.replace('_', ' ').title()} stage"
        
        # Combine content from all thoughts in this stage
        combined_content = " ".join(t.content for t in stage_thoughts)
        word_count = len(combined_content.split())
        thought_count = len(stage_thoughts)
        
        # Analyze tags and axioms for this stage
        all_tags = [tag for thought in stage_thoughts for tag in thought.tags]
        all_axioms = [axiom for thought in stage_thoughts for axiom in thought.axioms]
        assumptions = [assumption for thought in stage_thoughts for assumption in thought.assumptions_challenged]
        
        tag_summary = ", ".join(list(set(all_tags))[:5]) if all_tags else "no tags"
        axiom_summary = ", ".join(list(set(all_axioms))[:3]) if all_axioms else "no axioms applied"
        
        stage_name = stage.value.replace('_', ' ').title()
        summary = f"{stage_name}: {thought_count} thoughts, {word_count} words of analysis"
        
        if all_tags:
            summary += f" | Key themes: {tag_summary}"
        if all_axioms:
            summary += f" | Applied axioms: {axiom_summary}"
        if assumptions:
            summary += f" | Challenged {len(assumptions)} assumptions"
        
        return summary
    
    @staticmethod
    def calculate_session_confidence(session: ThinkingSession) -> float:
        """Calculate comprehensive confidence score for thinking session"""
        if not session.thoughts:
            return 0.0
        
        factors = []
        
        # Factor 1: Stage completion (0.0-1.0)
        stages_completed = len(set(t.stage for t in session.thoughts))
        stage_completion_factor = stages_completed / 5.0  # 5 total stages
        factors.append(stage_completion_factor)
        
        # Factor 2: Thought depth (based on content length)
        avg_content_length = sum(len(t.content) for t in session.thoughts) / len(session.thoughts)
        depth_factor = min(avg_content_length / 200.0, 1.0)  # Normalize to 200 chars as ideal
        factors.append(depth_factor)
        
        # Factor 3: Relationship richness
        total_relationships = sum(len(t.relationships) for t in session.thoughts)
        relationship_factor = min(total_relationships / len(session.thoughts), 1.0)
        factors.append(relationship_factor)
        
        # Factor 4: Assumption challenging (indicates critical thinking)
        assumptions_challenged = sum(len(t.assumptions_challenged) for t in session.thoughts)
        assumption_factor = min(assumptions_challenged / (len(session.thoughts) * 0.5), 1.0)
        factors.append(assumption_factor)
        
        # Factor 5: Axiom application (indicates principled thinking)
        axioms_applied = sum(len(t.axioms) for t in session.thoughts)
        axiom_factor = min(axioms_applied / (len(session.thoughts) * 0.3), 1.0)
        factors.append(axiom_factor)
        
        # Factor 6: Tag consistency (indicates organized thinking)
        all_tags = [tag for thought in session.thoughts for tag in thought.tags]
        unique_tags = set(all_tags)
        tag_consistency = len(all_tags) / max(len(unique_tags), 1) if unique_tags else 0
        tag_factor = min(tag_consistency / 2.0, 1.0)  # Normalize expecting ~2 uses per tag
        factors.append(tag_factor)
        
        # Weighted average with emphasis on stage completion
        weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]  # Stage completion gets higher weight
        weighted_score = sum(factor * weight for factor, weight in zip(factors, weights))
        
        return min(weighted_score, 1.0)
    
    @staticmethod
    def identify_thinking_patterns(sessions: List[ThinkingSession]) -> List[ThinkingPattern]:
        """Identify recurring patterns across multiple thinking sessions"""
        patterns = []
        
        if not sessions:
            return patterns
        
        # Group sessions by similar characteristics
        stage_patterns = defaultdict(list)
        axiom_patterns = defaultdict(list)
        
        for session in sessions:
            # Group by stages completed
            stages_key = tuple(sorted(session.stages_completed, key=lambda x: x.value))
            stage_patterns[stages_key].append(session)
            
            # Group by common axioms
            all_axioms = set()
            for thought in session.thoughts:
                all_axioms.update(thought.axioms)
            
            if all_axioms:
                axiom_key = tuple(sorted(all_axioms)[:3])  # Top 3 axioms
                axiom_patterns[axiom_key].append(session)
        
        # Create patterns for frequently occurring stage sequences
        for stages, stage_sessions in stage_patterns.items():
            if len(stage_sessions) >= 3:  # Pattern must occur at least 3 times
                pattern = ThinkingPattern(
                    pattern_name=f"Stage Pattern: {' → '.join([s.value.title() for s in stages])}",
                    description=f"Common thinking sequence involving {len(stages)} stages",
                    common_stages=list(stages),
                    usage_count=len(stage_sessions),
                    tags=["stage_pattern", "sequence"]
                )
                patterns.append(pattern)
        
        # Create patterns for common axiom combinations
        for axioms, axiom_sessions in axiom_patterns.items():
            if len(axiom_sessions) >= 2:  # Pattern must occur at least 2 times
                pattern = ThinkingPattern(
                    pattern_name=f"Axiom Pattern: {axioms[0][:30]}..." if axioms else "Common Axioms",
                    description=f"Common axiom combination used in {len(axiom_sessions)} sessions",
                    typical_axioms=list(axioms),
                    usage_count=len(axiom_sessions),
                    tags=["axiom_pattern", "principles"]
                )
                patterns.append(pattern)
        
        return patterns
    
    @staticmethod
    def analyze_session_progression(session: ThinkingSession) -> Dict[str, Any]:
        """Analyze how a thinking session progresses through stages"""
        if not session.thoughts:
            return {"error": "No thoughts to analyze"}
        
        # Sort thoughts by number to analyze progression
        sorted_thoughts = sorted(session.thoughts, key=lambda t: t.thought_number)
        
        stage_transitions = []
        current_stage = None
        
        for thought in sorted_thoughts:
            if thought.stage != current_stage:
                stage_transitions.append({
                    "thought_number": thought.thought_number,
                    "stage": thought.stage.value,
                    "transition_type": "new_stage" if current_stage is None else "stage_change"
                })
                current_stage = thought.stage
        
        # Analyze stage duration (thoughts per stage)
        stage_durations = Counter(t.stage for t in sorted_thoughts)
        
        # Check for non-linear progression
        stage_order = [ThinkingStage.PROBLEM_DEFINITION, ThinkingStage.RESEARCH, 
                      ThinkingStage.ANALYSIS, ThinkingStage.SYNTHESIS, ThinkingStage.CONCLUSION]
        
        stage_indices = []
        for thought in sorted_thoughts:
            try:
                stage_indices.append(stage_order.index(thought.stage))
            except ValueError:
                continue
        
        # Check if progression is generally forward
        is_linear = all(stage_indices[i] >= stage_indices[i-1] for i in range(1, len(stage_indices)))
        
        return {
            "total_thoughts": len(sorted_thoughts),
            "stage_transitions": stage_transitions,
            "stage_durations": {stage.value: count for stage, count in stage_durations.items()},
            "is_linear_progression": is_linear,
            "progression_score": sum(1 for i in range(1, len(stage_indices)) 
                                   if stage_indices[i] >= stage_indices[i-1]) / max(len(stage_indices) - 1, 1),
            "stages_visited": len(set(t.stage for t in sorted_thoughts)),
            "average_thoughts_per_stage": len(sorted_thoughts) / len(set(t.stage for t in sorted_thoughts))
        }


class ThinkingMemoryMapper:
    """Maps structured thinking to alunai-clarity memory types and operations"""
    
    # Memory type mappings for different thinking stages
    STAGE_MEMORY_TYPE_MAP = {
        ThinkingStage.PROBLEM_DEFINITION: "problem_analysis",
        ThinkingStage.RESEARCH: "research_notes", 
        ThinkingStage.ANALYSIS: "analysis_result",
        ThinkingStage.SYNTHESIS: "solution_synthesis",
        ThinkingStage.CONCLUSION: "conclusion_summary"
    }
    
    @staticmethod
    def thought_to_memory_type(thought: StructuredThought) -> str:
        """Map thinking stage to appropriate memory type"""
        return ThinkingMemoryMapper.STAGE_MEMORY_TYPE_MAP.get(
            thought.stage, "structured_thinking"
        )
    
    @staticmethod
    def prepare_memory_metadata(thought: StructuredThought, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Prepare comprehensive metadata for memory storage"""
        metadata = {
            "thinking_stage": thought.stage.value,
            "thought_number": thought.thought_number,
            "total_expected": thought.total_expected,
            "tags": thought.tags,
            "axioms": thought.axioms,
            "assumptions_challenged": thought.assumptions_challenged,
            "relationship_count": len(thought.relationships),
            "structured_thinking": True,
            "memory_source": "structured_thinking_integration"
        }
        
        if session_id:
            metadata["thinking_session_id"] = session_id
        
        # Add thought-specific metadata
        metadata.update(thought.metadata)
        
        return metadata
    
    @staticmethod
    def prepare_session_metadata(session: ThinkingSession) -> Dict[str, Any]:
        """Prepare metadata for storing thinking session summaries"""
        return {
            "session_id": session.id,
            "session_title": session.title,
            "total_thoughts": session.total_thoughts,
            "stages_completed": [stage.value for stage in session.stages_completed],
            "current_stage": session.current_stage.value,
            "progress_percentage": session.progress_percentage,
            "is_complete": session.is_complete,
            "project_context": session.project_context,
            "session_tags": session.tags,
            "structured_thinking_session": True
        }
    
    @staticmethod
    def create_search_queries(thought: StructuredThought) -> List[str]:
        """Generate search queries for finding related memories"""
        queries = []
        
        # Stage-based query
        stage_name = thought.stage.value.replace('_', ' ')
        queries.append(f"{stage_name} {' '.join(thought.tags[:3])}")
        
        # Content-based query (first 50 words)
        content_words = thought.content.split()[:50]
        if len(content_words) >= 10:
            queries.append(' '.join(content_words[:20]))
        
        # Tag-based query
        if thought.tags:
            queries.append(' '.join(thought.tags))
        
        # Axiom-based query
        if thought.axioms:
            queries.append(' '.join(thought.axioms))
        
        return queries
    
    @staticmethod
    def extract_keywords(thought: StructuredThought) -> List[str]:
        """Extract keywords for enhanced searchability"""
        keywords = []
        
        # Add stage as keyword
        keywords.append(thought.stage.value)
        
        # Add tags (already processed)
        keywords.extend(thought.tags)
        
        # Add axioms as keywords
        keywords.extend(thought.axioms)
        
        # Extract key terms from content (simple approach)
        content_words = thought.content.lower().split()
        
        # Common technical and thinking keywords
        important_keywords = {
            'implement', 'design', 'architecture', 'solution', 'problem',
            'analysis', 'research', 'conclusion', 'synthesis', 'approach',
            'strategy', 'method', 'process', 'system', 'framework',
            'pattern', 'principle', 'assumption', 'hypothesis', 'evidence'
        }
        
        for word in content_words:
            cleaned_word = word.strip('.,!?;:"()[]{}')
            if cleaned_word in important_keywords and len(cleaned_word) > 3:
                keywords.append(cleaned_word)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))


class ThinkingSessionManager:
    """Manages thinking sessions and their lifecycle"""
    
    def __init__(self):
        self.active_sessions: Dict[str, ThinkingSession] = {}
        self.session_history: List[str] = []
    
    def create_session(
        self, 
        title: str, 
        description: Optional[str] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> ThinkingSession:
        """Create a new thinking session"""
        session = ThinkingSession(
            title=title,
            description=description,
            project_context=project_context or {}
        )
        
        self.active_sessions[session.id] = session
        self.session_history.append(session.id)
        
        return session
    
    def add_thought_to_session(
        self, 
        session_id: str, 
        thought: StructuredThought
    ) -> bool:
        """Add a thought to an existing session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.add_thought(thought)
        
        # Update session stage if this thought represents progression
        if thought.stage.value > session.current_stage.value:
            session.current_stage = thought.stage
        
        return True
    
    def complete_session(self, session_id: str) -> Optional[ThinkingSession]:
        """Mark a session as completed"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        session.completed_at = datetime.now(timezone.utc)
        
        # Remove from active sessions
        completed_session = self.active_sessions.pop(session_id)
        
        return completed_session
    
    def get_session_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Suggest relevant sessions based on context"""
        suggestions = []
        
        # Simple keyword-based matching
        context_keywords = set()
        for key, value in context.items():
            if isinstance(value, str):
                context_keywords.update(value.lower().split())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        context_keywords.update(item.lower().split())
        
        for session in self.active_sessions.values():
            # Check title overlap
            title_words = set(session.title.lower().split())
            if title_words & context_keywords:
                suggestions.append(session.id)
                continue
            
            # Check tag overlap
            session_tags = set(session.tags)
            if session_tags & context_keywords:
                suggestions.append(session.id)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed sessions"""
        active_count = len(self.active_sessions)
        total_thoughts = sum(len(session.thoughts) for session in self.active_sessions.values())
        
        stage_distribution = Counter()
        for session in self.active_sessions.values():
            stage_distribution[session.current_stage.value] += 1
        
        return {
            "active_sessions": active_count,
            "total_history": len(self.session_history),
            "total_active_thoughts": total_thoughts,
            "average_thoughts_per_session": total_thoughts / max(active_count, 1),
            "stage_distribution": dict(stage_distribution),
            "most_active_session": max(
                self.active_sessions.values(),
                key=lambda s: len(s.thoughts)
            ).id if self.active_sessions else None
        }