"""
Structured thinking domain models for integrating sequential thinking capabilities.

This module provides comprehensive models for structured thinking processes,
integrating the best features from mcp-sequential-thinking with alunai-clarity's
high-performance memory system.
"""

from enum import Enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class ThinkingStage(Enum):
    """Five-stage structured thinking process from mcp-sequential-thinking"""
    PROBLEM_DEFINITION = "problem_definition"
    RESEARCH = "research"
    ANALYSIS = "analysis" 
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"


class ThoughtRelationship(BaseModel):
    """Represents relationship between thoughts with strength scoring"""
    source_thought_id: str
    target_thought_id: str
    relationship_type: str = Field(..., pattern="^(builds_on|challenges|supports|contradicts|extends)$")
    strength: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "source_thought_id": "thought_123",
                "target_thought_id": "thought_124",
                "relationship_type": "builds_on",
                "strength": 0.8,
                "description": "Research findings build upon problem definition"
            }
        }


class StructuredThought(BaseModel):
    """Enhanced thought model with comprehensive metadata from sequential thinking"""
    id: str = Field(default_factory=lambda: f"thought_{str(uuid4())}")
    thought_number: int = Field(gt=0, description="Sequential number in thinking process")
    total_expected: Optional[int] = Field(default=None, gt=0, description="Total expected thoughts")
    stage: ThinkingStage = Field(..., description="Current thinking stage")
    content: str = Field(min_length=10, max_length=2000, description="Thought content")
    
    # Enhanced metadata from sequential thinking
    tags: List[str] = Field(default_factory=list, max_items=10, description="Descriptive tags")
    axioms: List[str] = Field(default_factory=list, max_items=5, description="Applied axioms")
    assumptions_challenged: List[str] = Field(default_factory=list, max_items=5)
    
    # Relationship tracking
    relationships: List[ThoughtRelationship] = Field(default_factory=list, max_items=10)
    
    # Standard metadata for alunai-clarity integration
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        """Clean and normalize tags"""
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Ensure content is not empty"""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    @field_validator('axioms')
    @classmethod
    def validate_axioms(cls, v):
        """Clean axioms list"""
        return [axiom.strip() for axiom in v if axiom.strip()]
    
    @field_validator('assumptions_challenged')
    @classmethod
    def validate_assumptions(cls, v):
        """Clean assumptions list"""
        return [assumption.strip() for assumption in v if assumption.strip()]
    
    class Config:
        schema_extra = {
            "example": {
                "thought_number": 1,
                "stage": "problem_definition",
                "content": "Need to implement user authentication with JWT tokens",
                "tags": ["authentication", "jwt", "security"],
                "axioms": ["Security first", "Follow industry standards"],
                "assumptions_challenged": ["Users will always logout properly"],
                "importance": 0.9
            }
        }


class ThinkingSession(BaseModel):
    """Complete thinking session with all thoughts and progress tracking"""
    id: str = Field(default_factory=lambda: f"session_{str(uuid4())}")
    title: str = Field(..., min_length=5, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    
    thoughts: List[StructuredThought] = Field(default_factory=list)
    current_stage: ThinkingStage = Field(default=ThinkingStage.PROBLEM_DEFINITION)
    
    # Session metadata for alunai-clarity integration
    project_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list, max_items=15)
    
    # Progress tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if thinking session is complete"""
        return self.current_stage == ThinkingStage.CONCLUSION and self.completed_at is not None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage based on current stage"""
        stage_values = {
            ThinkingStage.PROBLEM_DEFINITION: 20,
            ThinkingStage.RESEARCH: 40,
            ThinkingStage.ANALYSIS: 60,
            ThinkingStage.SYNTHESIS: 80,
            ThinkingStage.CONCLUSION: 100
        }
        return stage_values.get(self.current_stage, 0)
    
    @property
    def total_thoughts(self) -> int:
        """Total number of thoughts in session"""
        return len(self.thoughts)
    
    @property
    def stages_completed(self) -> List[str]:
        """List of stages that have been completed"""
        return list(set(thought.stage.value for thought in self.thoughts))
    
    def get_thoughts_by_stage(self, stage: ThinkingStage) -> List[StructuredThought]:
        """Get all thoughts for a specific stage"""
        return [thought for thought in self.thoughts if thought.stage == stage]
    
    def add_thought(self, thought: StructuredThought) -> None:
        """Add a thought to the session"""
        self.thoughts.append(thought)
        self.current_stage = thought.stage
        self.last_updated = datetime.now(timezone.utc)
    
    @field_validator('tags')
    @classmethod
    def validate_session_tags(cls, v):
        """Clean and normalize session tags"""
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Implement User Authentication System",
                "description": "Systematic analysis of JWT-based authentication implementation",
                "tags": ["authentication", "security", "jwt"],
                "project_context": {
                    "language": "python",
                    "framework": "fastapi"
                }
            }
        }


class ThinkingSummary(BaseModel):
    """Comprehensive summary of thinking process for analysis and retrieval"""
    session_id: str
    title: str
    total_thoughts: int
    stages_completed: List[str]  # Changed from ThinkingStage to str for JSON serialization
    
    # Stage summaries
    problem_summary: Optional[str] = None
    research_summary: Optional[str] = None
    analysis_summary: Optional[str] = None
    synthesis_summary: Optional[str] = None
    conclusion_summary: Optional[str] = None
    
    # Relationship analysis
    key_relationships: List[ThoughtRelationship] = Field(default_factory=list)
    assumptions_challenged_count: int = Field(default=0)
    axioms_applied: List[str] = Field(default_factory=list)
    
    # Quality metrics
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @property
    def is_comprehensive(self) -> bool:
        """Check if summary covers all thinking stages"""
        return len(self.stages_completed) == 5
    
    @property
    def relationship_density(self) -> float:
        """Calculate density of relationships per thought"""
        if self.total_thoughts == 0:
            return 0.0
        return len(self.key_relationships) / self.total_thoughts
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "title": "Authentication System Analysis",
                "total_thoughts": 8,
                "stages_completed": ["problem_definition", "research", "analysis"],
                "confidence_score": 0.85,
                "assumptions_challenged_count": 3,
                "axioms_applied": ["Security first", "KISS principle"]
            }
        }


class ThinkingPattern(BaseModel):
    """Represents recurring patterns in thinking processes for learning"""
    pattern_id: str = Field(default_factory=lambda: f"pattern_{str(uuid4())}")
    pattern_name: str = Field(..., min_length=5, max_length=100)
    description: str = Field(..., max_length=500)
    
    # Pattern characteristics
    common_stages: List[ThinkingStage] = Field(default_factory=list)
    typical_axioms: List[str] = Field(default_factory=list, max_items=10)
    common_assumptions: List[str] = Field(default_factory=list, max_items=10)
    success_indicators: List[str] = Field(default_factory=list, max_items=5)
    
    # Usage statistics
    usage_count: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    average_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Metadata
    first_observed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list, max_items=10)
    
    def record_usage(self, success: bool, confidence: float) -> None:
        """Record usage of this pattern"""
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
        
        # Update success rate (simple moving average)
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
            self.average_confidence = confidence
        else:
            weight = 1.0 / self.usage_count
            self.success_rate = (1 - weight) * self.success_rate + weight * (1.0 if success else 0.0)
            self.average_confidence = (1 - weight) * self.average_confidence + weight * confidence
    
    class Config:
        schema_extra = {
            "example": {
                "pattern_name": "API Integration Analysis",
                "description": "Common pattern for analyzing API integration requirements",
                "common_stages": ["problem_definition", "research", "analysis"],
                "typical_axioms": ["APIs are external dependencies", "Plan for failure"],
                "success_rate": 0.85
            }
        }