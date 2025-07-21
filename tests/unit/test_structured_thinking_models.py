"""
Unit tests for structured thinking models and utilities.

This module validates the core structured thinking functionality
including models, validation, relationships, and analysis utilities.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from uuid import UUID

from clarity.domains.structured_thinking import (
    StructuredThought, 
    ThinkingSession, 
    ThinkingStage, 
    ThoughtRelationship,
    ThinkingSummary,
    ThinkingPattern
)
from clarity.domains.structured_thinking_utils import (
    ThinkingAnalyzer,
    ThinkingMemoryMapper,
    ThinkingSessionManager
)


class TestStructuredThought:
    """Test StructuredThought model validation and functionality"""
    
    def test_basic_thought_creation(self):
        """Test creating a basic structured thought"""
        thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.PROBLEM_DEFINITION,
            content="This is a test problem definition with sufficient length"
        )
        
        assert thought.thought_number == 1
        assert thought.stage == ThinkingStage.PROBLEM_DEFINITION
        assert thought.content == "This is a test problem definition with sufficient length"
        assert len(thought.id) > 0
        assert isinstance(thought.created_at, datetime)
        assert thought.importance == 0.5  # Default value
        assert thought.tags == []
        assert thought.axioms == []
        assert thought.assumptions_challenged == []
        assert thought.relationships == []
    
    def test_thought_with_comprehensive_metadata(self):
        """Test thought creation with all metadata fields"""
        thought = StructuredThought(
            thought_number=2,
            total_expected=5,
            stage=ThinkingStage.RESEARCH,
            content="Research findings about authentication systems with detailed analysis",
            tags=["authentication", "security", "jwt"],
            axioms=["Security first", "Defense in depth"],
            assumptions_challenged=["All users logout properly", "Network is secure"],
            importance=0.8
        )
        
        assert thought.thought_number == 2
        assert thought.total_expected == 5
        assert thought.stage == ThinkingStage.RESEARCH
        assert thought.tags == ["authentication", "security", "jwt"]
        assert thought.axioms == ["Security first", "Defense in depth"]
        assert thought.assumptions_challenged == ["All users logout properly", "Network is secure"]
        assert thought.importance == 0.8
    
    def test_thought_validation_errors(self):
        """Test validation errors for invalid thought data"""
        # Test thought number must be > 0
        with pytest.raises(ValueError):
            StructuredThought(
                thought_number=0,
                stage=ThinkingStage.PROBLEM_DEFINITION,
                content="Valid content with sufficient length"
            )
        
        # Test negative thought number
        with pytest.raises(ValueError):
            StructuredThought(
                thought_number=-1,
                stage=ThinkingStage.PROBLEM_DEFINITION,
                content="Valid content with sufficient length"
            )
        
        # Test content too short
        with pytest.raises(ValueError):
            StructuredThought(
                thought_number=1,
                stage=ThinkingStage.PROBLEM_DEFINITION,
                content="Short"  # Less than 10 characters
            )
        
        # Test empty content
        with pytest.raises(ValueError):
            StructuredThought(
                thought_number=1,
                stage=ThinkingStage.PROBLEM_DEFINITION,
                content=""
            )
        
        # Test content too long
        with pytest.raises(ValueError):
            StructuredThought(
                thought_number=1,
                stage=ThinkingStage.PROBLEM_DEFINITION,
                content="x" * 2001  # Exceeds 2000 character limit
            )
    
    def test_tag_validation_and_cleanup(self):
        """Test tag validation cleans and normalizes tags"""
        thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.RESEARCH,
            content="Research content with sufficient length for validation",
            tags=["  API  ", "Database", "  ", "SECURITY", "web-services"]
        )
        
        # Tags should be cleaned, lowercased, and empty strings removed
        assert thought.tags == ["api", "database", "security", "web-services"]
    
    def test_axioms_and_assumptions_cleanup(self):
        """Test axioms and assumptions are properly cleaned"""
        thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.ANALYSIS,
            content="Analysis content with detailed examination of the problem",
            axioms=["  Security First  ", "", "KISS Principle"],
            assumptions_challenged=["Users are technical", "  ", "System is always online  "]
        )
        
        assert thought.axioms == ["Security First", "KISS Principle"]
        assert thought.assumptions_challenged == ["Users are technical", "System is always online"]
    
    def test_thought_relationships(self):
        """Test thought relationships are properly handled"""
        relationship = ThoughtRelationship(
            source_thought_id="thought_1",
            target_thought_id="thought_2",
            relationship_type="builds_on",
            strength=0.8,
            description="Research builds on problem definition"
        )
        
        thought = StructuredThought(
            thought_number=2,
            stage=ThinkingStage.RESEARCH,
            content="Research findings building on previous problem definition",
            relationships=[relationship]
        )
        
        assert len(thought.relationships) == 1
        assert thought.relationships[0].relationship_type == "builds_on"
        assert thought.relationships[0].strength == 0.8
        assert thought.relationships[0].description == "Research builds on problem definition"


class TestThoughtRelationship:
    """Test ThoughtRelationship model validation"""
    
    def test_valid_relationship_creation(self):
        """Test creating valid thought relationships"""
        relationship = ThoughtRelationship(
            source_thought_id="thought_123",
            target_thought_id="thought_124",
            relationship_type="builds_on",
            strength=0.8
        )
        
        assert relationship.source_thought_id == "thought_123"
        assert relationship.target_thought_id == "thought_124"
        assert relationship.relationship_type == "builds_on"
        assert relationship.strength == 0.8
        assert relationship.description is None
    
    def test_relationship_with_description(self):
        """Test relationship with description"""
        relationship = ThoughtRelationship(
            source_thought_id="thought_1",
            target_thought_id="thought_2",
            relationship_type="challenges",
            strength=0.6,
            description="Analysis challenges initial assumptions"
        )
        
        assert relationship.description == "Analysis challenges initial assumptions"
    
    def test_relationship_type_validation(self):
        """Test relationship type must be from allowed set"""
        valid_types = ["builds_on", "challenges", "supports", "contradicts", "extends"]
        
        for rel_type in valid_types:
            relationship = ThoughtRelationship(
                source_thought_id="thought_1",
                target_thought_id="thought_2",
                relationship_type=rel_type,
                strength=0.5
            )
            assert relationship.relationship_type == rel_type
        
        # Test invalid relationship type
        with pytest.raises(ValueError):
            ThoughtRelationship(
                source_thought_id="thought_1",
                target_thought_id="thought_2",
                relationship_type="invalid_type",
                strength=0.5
            )
    
    def test_strength_validation(self):
        """Test relationship strength must be between 0.0 and 1.0"""
        # Valid strengths
        for strength in [0.0, 0.5, 1.0]:
            relationship = ThoughtRelationship(
                source_thought_id="thought_1",
                target_thought_id="thought_2",
                relationship_type="builds_on",
                strength=strength
            )
            assert relationship.strength == strength
        
        # Invalid strengths
        for invalid_strength in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValueError):
                ThoughtRelationship(
                    source_thought_id="thought_1",
                    target_thought_id="thought_2",
                    relationship_type="builds_on",
                    strength=invalid_strength
                )


class TestThinkingSession:
    """Test ThinkingSession model and functionality"""
    
    def test_basic_session_creation(self):
        """Test creating a basic thinking session"""
        session = ThinkingSession(
            title="Test Authentication System",
            description="Testing session creation and validation"
        )
        
        assert session.title == "Test Authentication System"
        assert session.description == "Testing session creation and validation"
        assert len(session.id) > 0
        assert session.current_stage == ThinkingStage.PROBLEM_DEFINITION
        assert session.thoughts == []
        assert session.tags == []
        assert session.project_context == {}
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_updated, datetime)
        assert session.completed_at is None
    
    def test_session_properties(self):
        """Test session computed properties"""
        session = ThinkingSession(title="Test Session")
        
        # Initially empty
        assert session.is_complete is False
        assert session.progress_percentage == 20  # PROBLEM_DEFINITION
        assert session.total_thoughts == 0
        assert session.stages_completed == []
        
        # Add thoughts and test progress
        thought1 = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.PROBLEM_DEFINITION,
            content="Problem definition content"
        )
        session.add_thought(thought1)
        
        assert session.total_thoughts == 1
        assert session.progress_percentage == 20
        assert ThinkingStage.PROBLEM_DEFINITION in session.stages_completed
        
        # Progress to research stage
        thought2 = StructuredThought(
            thought_number=2,
            stage=ThinkingStage.RESEARCH,
            content="Research findings content"
        )
        session.add_thought(thought2)
        
        assert session.total_thoughts == 2
        assert session.progress_percentage == 40  # Updated to RESEARCH
        assert len(session.stages_completed) == 2
    
    def test_session_completion(self):
        """Test session completion logic"""
        session = ThinkingSession(title="Completion Test")
        
        # Move to conclusion stage
        session.current_stage = ThinkingStage.CONCLUSION
        assert not session.is_complete  # Not complete until completed_at is set
        
        # Complete the session
        session.completed_at = datetime.now(timezone.utc)
        assert session.is_complete
        assert session.progress_percentage == 100
    
    def test_get_thoughts_by_stage(self):
        """Test filtering thoughts by stage"""
        session = ThinkingSession(title="Stage Filter Test")
        
        # Add thoughts from different stages
        thoughts = [
            StructuredThought(thought_number=1, stage=ThinkingStage.PROBLEM_DEFINITION, content="Problem content"),
            StructuredThought(thought_number=2, stage=ThinkingStage.RESEARCH, content="Research content one"),
            StructuredThought(thought_number=3, stage=ThinkingStage.RESEARCH, content="Research content two"),
            StructuredThought(thought_number=4, stage=ThinkingStage.ANALYSIS, content="Analysis content")
        ]
        
        for thought in thoughts:
            session.add_thought(thought)
        
        # Test filtering
        problem_thoughts = session.get_thoughts_by_stage(ThinkingStage.PROBLEM_DEFINITION)
        assert len(problem_thoughts) == 1
        assert problem_thoughts[0].content == "Problem content"
        
        research_thoughts = session.get_thoughts_by_stage(ThinkingStage.RESEARCH)
        assert len(research_thoughts) == 2
        
        synthesis_thoughts = session.get_thoughts_by_stage(ThinkingStage.SYNTHESIS)
        assert len(synthesis_thoughts) == 0
    
    def test_session_validation(self):
        """Test session validation rules"""
        # Title too short
        with pytest.raises(ValueError):
            ThinkingSession(title="Hi")
        
        # Title too long
        with pytest.raises(ValueError):
            ThinkingSession(title="x" * 201)
        
        # Description too long
        with pytest.raises(ValueError):
            ThinkingSession(
                title="Valid Title",
                description="x" * 1001
            )
    
    def test_session_tag_validation(self):
        """Test session tag validation and cleanup"""
        session = ThinkingSession(
            title="Tag Test Session",
            tags=["  project_alpha  ", "AUTHENTICATION", "  ", "security"]
        )
        
        assert session.tags == ["project_alpha", "authentication", "security"]


class TestThinkingAnalyzer:
    """Test ThinkingAnalyzer utility functions"""
    
    def test_analyze_thought_relationships_empty(self):
        """Test relationship analysis with empty thought list"""
        result = ThinkingAnalyzer.analyze_thought_relationships([])
        
        assert result["total_relationships"] == 0
        assert result["relationship_distribution"] == {}
        assert result["average_relationships_per_thought"] == 0.0
        assert result["most_connected_thought"] is None
    
    def test_analyze_thought_relationships_with_data(self):
        """Test relationship analysis with actual thought data"""
        # Create thoughts with relationships
        thought1 = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.PROBLEM_DEFINITION,
            content="Problem definition content"
        )
        
        relationship = ThoughtRelationship(
            source_thought_id="thought_2",
            target_thought_id=thought1.id,
            relationship_type="builds_on",
            strength=0.8
        )
        
        thought2 = StructuredThought(
            thought_number=2,
            stage=ThinkingStage.RESEARCH,
            content="Research content that builds on problem",
            relationships=[relationship]
        )
        
        thoughts = [thought1, thought2]
        result = ThinkingAnalyzer.analyze_thought_relationships(thoughts)
        
        assert result["total_relationships"] == 1
        assert result["relationship_distribution"]["builds_on"] == 1
        assert result["average_relationships_per_thought"] == 0.5
        assert result["most_connected_thought"]["connection_count"] == 1
        assert "relationship_strength_stats" in result
    
    def test_generate_stage_summary_empty(self):
        """Test stage summary generation with no thoughts"""
        summary = ThinkingAnalyzer.generate_stage_summary([], ThinkingStage.RESEARCH)
        
        assert "No thoughts recorded" in summary
        assert "Research" in summary
    
    def test_generate_stage_summary_with_thoughts(self):
        """Test stage summary generation with actual thoughts"""
        thoughts = [
            StructuredThought(
                thought_number=1,
                stage=ThinkingStage.RESEARCH,
                content="Research about authentication methods and best practices",
                tags=["authentication", "security"],
                axioms=["Security first"]
            ),
            StructuredThought(
                thought_number=2,
                stage=ThinkingStage.RESEARCH,
                content="Additional research on JWT implementation details",
                tags=["jwt", "tokens"]
            )
        ]
        
        summary = ThinkingAnalyzer.generate_stage_summary(thoughts, ThinkingStage.RESEARCH)
        
        assert "Research: 2 thoughts" in summary
        assert "words of analysis" in summary
        assert "authentication" in summary or "jwt" in summary
    
    def test_calculate_session_confidence(self):
        """Test session confidence calculation"""
        # Empty session
        empty_session = ThinkingSession(title="Empty Session")
        assert ThinkingAnalyzer.calculate_session_confidence(empty_session) == 0.0
        
        # Session with comprehensive thoughts
        session = ThinkingSession(title="Comprehensive Session")
        
        comprehensive_thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.ANALYSIS,
            content="Detailed analysis of authentication requirements and implementation strategies for securing user access",
            tags=["authentication", "security", "analysis"],
            axioms=["Security first", "KISS principle"],
            assumptions_challenged=["Users always use strong passwords"],
            relationships=[ThoughtRelationship(
                source_thought_id="t1",
                target_thought_id="t2", 
                relationship_type="builds_on",
                strength=0.8
            )]
        )
        
        session.add_thought(comprehensive_thought)
        confidence = ThinkingAnalyzer.calculate_session_confidence(session)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0  # Should have some confidence with comprehensive thought


class TestThinkingMemoryMapper:
    """Test ThinkingMemoryMapper utility functions"""
    
    def test_thought_to_memory_type_mapping(self):
        """Test mapping thinking stages to memory types"""
        test_cases = [
            (ThinkingStage.PROBLEM_DEFINITION, "problem_analysis"),
            (ThinkingStage.RESEARCH, "research_notes"),
            (ThinkingStage.ANALYSIS, "analysis_result"),
            (ThinkingStage.SYNTHESIS, "solution_synthesis"),
            (ThinkingStage.CONCLUSION, "conclusion_summary")
        ]
        
        for stage, expected_memory_type in test_cases:
            thought = StructuredThought(
                thought_number=1,
                stage=stage,
                content=f"Content for {stage.value} stage"
            )
            
            memory_type = ThinkingMemoryMapper.thought_to_memory_type(thought)
            assert memory_type == expected_memory_type
    
    def test_prepare_memory_metadata(self):
        """Test preparation of memory metadata"""
        thought = StructuredThought(
            thought_number=3,
            total_expected=5,
            stage=ThinkingStage.SYNTHESIS,
            content="Synthesis of research findings into solution approach",
            tags=["synthesis", "solution"],
            axioms=["KISS principle"],
            assumptions_challenged=["Current system is scalable"],
            metadata={"custom_field": "custom_value"}
        )
        
        metadata = ThinkingMemoryMapper.prepare_memory_metadata(thought, "session_123")
        
        assert metadata["thinking_stage"] == "synthesis"
        assert metadata["thought_number"] == 3
        assert metadata["total_expected"] == 5
        assert metadata["tags"] == ["synthesis", "solution"]
        assert metadata["axioms"] == ["KISS principle"]
        assert metadata["assumptions_challenged"] == ["Current system is scalable"]
        assert metadata["thinking_session_id"] == "session_123"
        assert metadata["structured_thinking"] is True
        assert metadata["custom_field"] == "custom_value"
    
    def test_create_search_queries(self):
        """Test search query generation for thoughts"""
        thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.RESEARCH,
            content="Research about JWT authentication implementation best practices and security considerations",
            tags=["jwt", "authentication", "security"]
        )
        
        queries = ThinkingMemoryMapper.create_search_queries(thought)
        
        assert len(queries) > 0
        assert any("research" in query.lower() for query in queries)
        assert any("jwt" in query.lower() for query in queries)
    
    def test_extract_keywords(self):
        """Test keyword extraction from thoughts"""
        thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.ANALYSIS,
            content="Analysis of authentication system architecture and implementation patterns",
            tags=["authentication", "architecture"],
            axioms=["Security first principle"]
        )
        
        keywords = ThinkingMemoryMapper.extract_keywords(thought)
        
        assert "analysis" in keywords
        assert "authentication" in keywords
        assert "architecture" in keywords
        assert len(keywords) > 0


class TestThinkingSessionManager:
    """Test ThinkingSessionManager functionality"""
    
    def test_create_session(self):
        """Test session creation"""
        manager = ThinkingSessionManager()
        
        session = manager.create_session(
            title="Test Session",
            description="Test description",
            project_context={"language": "python"}
        )
        
        assert session.title == "Test Session"
        assert session.description == "Test description"
        assert session.project_context["language"] == "python"
        assert session.id in manager.active_sessions
        assert session.id in manager.session_history
    
    def test_add_thought_to_session(self):
        """Test adding thoughts to sessions"""
        manager = ThinkingSessionManager()
        session = manager.create_session("Test Session")
        
        thought = StructuredThought(
            thought_number=1,
            stage=ThinkingStage.PROBLEM_DEFINITION,
            content="Problem definition content"
        )
        
        success = manager.add_thought_to_session(session.id, thought)
        assert success
        
        retrieved_session = manager.active_sessions[session.id]
        assert len(retrieved_session.thoughts) == 1
        assert retrieved_session.current_stage == ThinkingStage.PROBLEM_DEFINITION
        
        # Test adding to non-existent session
        fake_id = "nonexistent_session"
        success = manager.add_thought_to_session(fake_id, thought)
        assert not success
    
    def test_complete_session(self):
        """Test session completion"""
        manager = ThinkingSessionManager()
        session = manager.create_session("Test Session")
        
        completed = manager.complete_session(session.id)
        
        assert completed is not None
        assert completed.completed_at is not None
        assert session.id not in manager.active_sessions
        assert session.id in manager.session_history
    
    def test_session_statistics(self):
        """Test session statistics generation"""
        manager = ThinkingSessionManager()
        
        # Create multiple sessions with thoughts
        session1 = manager.create_session("Session 1")
        session2 = manager.create_session("Session 2")
        
        # Add thoughts
        thought1 = StructuredThought(thought_number=1, stage=ThinkingStage.PROBLEM_DEFINITION, content="Problem definition content")
        thought2 = StructuredThought(thought_number=2, stage=ThinkingStage.RESEARCH, content="Research content findings")
        
        manager.add_thought_to_session(session1.id, thought1)
        manager.add_thought_to_session(session2.id, thought2)
        
        stats = manager.get_session_statistics()
        
        assert stats["active_sessions"] == 2
        assert stats["total_history"] == 2
        assert stats["total_active_thoughts"] == 2
        assert stats["average_thoughts_per_session"] == 1.0
        assert "stage_distribution" in stats
        assert stats["most_active_session"] in [session1.id, session2.id]


# Integration tests combining multiple components
class TestStructuredThinkingIntegration:
    """Integration tests for structured thinking components"""
    
    def test_complete_thinking_workflow(self):
        """Test a complete thinking workflow from start to finish"""
        # Create session manager and session
        manager = ThinkingSessionManager()
        session = manager.create_session(
            title="Complete Workflow Test",
            description="Testing complete structured thinking workflow"
        )
        
        # Create thoughts for each stage
        stages_content = [
            (ThinkingStage.PROBLEM_DEFINITION, "Define the authentication problem clearly"),
            (ThinkingStage.RESEARCH, "Research JWT and OAuth implementation patterns"),
            (ThinkingStage.ANALYSIS, "Analyze security requirements and constraints"),
            (ThinkingStage.SYNTHESIS, "Synthesize research into implementation plan"),
            (ThinkingStage.CONCLUSION, "Conclude with specific implementation decisions")
        ]
        
        thoughts = []
        for i, (stage, content) in enumerate(stages_content, 1):
            thought = StructuredThought(
                thought_number=i,
                total_expected=5,
                stage=stage,
                content=content,
                tags=[stage.value.lower(), "workflow_test"],
                importance=min(0.6 + (i * 0.08), 1.0)  # Increasing importance, capped at 1.0
            )
            
            # Add relationships for thoughts after the first
            if i > 1:
                thought.relationships.append(ThoughtRelationship(
                    source_thought_id=thought.id,
                    target_thought_id=thoughts[i-2].id,  # Reference previous thought
                    relationship_type="builds_on",
                    strength=0.7 + (i * 0.05)
                ))
            
            thoughts.append(thought)
            manager.add_thought_to_session(session.id, thought)
        
        # Complete the session
        completed_session = manager.complete_session(session.id)
        
        # Analyze the completed session
        confidence = ThinkingAnalyzer.calculate_session_confidence(completed_session)
        relationship_analysis = ThinkingAnalyzer.analyze_thought_relationships(completed_session.thoughts)
        progression_analysis = ThinkingAnalyzer.analyze_session_progression(completed_session)
        
        # Validate results
        assert completed_session.is_complete
        assert completed_session.total_thoughts == 5
        assert len(completed_session.stages_completed) == 5  # All stages
        assert confidence > 0.5  # Should have good confidence
        assert relationship_analysis["total_relationships"] == 4  # 4 relationships created
        assert progression_analysis["is_linear_progression"]  # Should be linear
        
        # Test memory mapping for all thoughts
        for thought in thoughts:
            memory_type = ThinkingMemoryMapper.thought_to_memory_type(thought)
            metadata = ThinkingMemoryMapper.prepare_memory_metadata(thought, session.id)
            queries = ThinkingMemoryMapper.create_search_queries(thought)
            
            assert memory_type in ["problem_analysis", "research_notes", "analysis_result", 
                                 "solution_synthesis", "conclusion_summary"]
            assert metadata["thinking_session_id"] == session.id
            assert len(queries) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])