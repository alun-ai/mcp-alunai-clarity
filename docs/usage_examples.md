# Usage Examples and API Reference

## Overview

This guide provides practical examples of using Alunai Clarity's structured thinking integration and enhanced AutoCode features. All examples demonstrate real-world usage patterns with complete request/response cycles.

## 🧠 Structured Thinking Examples

### Complete Problem-Solving Session

#### Step 1: Problem Definition
```python
# Start with a clear problem definition
response = await process_structured_thought(
    stage="problem_definition",
    content="Need to implement a scalable user authentication system for a multi-tenant SaaS application with OAuth2, JWT tokens, and role-based access control",
    thought_number=1,
    session_id="auth_implementation_20240121",
    tags=["authentication", "saas", "scalability", "security"],
    axioms=["Security by design", "Principle of least privilege"],
    importance=0.9
)
```

**Response:**
```json
{
  "thought_id": "thought_auth_001",
  "memory_id": "mem_structured_thinking_456",
  "stage": "problem_definition",
  "thought_number": 1,
  "session_insights": {
    "stage_completed": "problem_definition",
    "insights": [
      "Problem definition stage completed - foundation established for analysis",
      "Applied 2 guiding principles"
    ]
  },
  "next_suggested_stage": "research"
}
```

#### Step 2: Research Phase
```python
# Gather relevant information and patterns
response = await process_structured_thought(
    stage="research",
    content="Found OAuth2 implementation patterns in memory. Django REST framework has built-in JWT support. Multi-tenancy requires tenant-specific user isolation. RBAC can be implemented with Django permissions or custom middleware.",
    thought_number=2,
    session_id="auth_implementation_20240121",
    tags=["research", "oauth2", "django", "multi-tenant", "rbac"],
    relationships=[
        {
            "source_thought_id": "thought_auth_002",
            "target_thought_id": "thought_auth_001", 
            "relationship_type": "builds_on",
            "strength": 0.9,
            "description": "Research builds on the problem definition"
        }
    ],
    importance=0.8
)
```

#### Step 3: Analysis Phase
```python
# Break down into components and challenge assumptions
response = await process_structured_thought(
    stage="analysis",
    content="Core components: OAuth2 provider integration, JWT token management, tenant isolation middleware, RBAC permission system, user session management. Critical paths: token refresh, tenant switching, permission caching.",
    thought_number=3,
    session_id="auth_implementation_20240121",
    tags=["analysis", "components", "architecture"],
    axioms=["Separation of concerns", "Fail securely"],
    assumptions_challenged=[
        "All tenants have same permission structure",
        "OAuth2 provider will always be available",
        "JWT tokens are sufficient for all use cases"
    ],
    relationships=[
        {
            "source_thought_id": "thought_auth_003",
            "target_thought_id": "thought_auth_002",
            "relationship_type": "extends", 
            "strength": 0.8
        }
    ],
    importance=0.9
)
```

#### Step 4: Generate Summary and Action Plan
```python
# Get comprehensive session analysis
summary = await generate_thinking_summary(
    session_id="auth_implementation_20240121",
    include_relationships=True,
    include_stage_summaries=True
)

# Generate concrete action plan
action_plan = await generate_coding_action_plan(
    session_id="auth_implementation_20240121"
)
```

**Action Plan Response:**
```json
{
  "session_id": "auth_implementation_20240121",
  "problem_statement": "Need to implement a scalable user authentication system...",
  "research_foundation": [
    "OAuth2 implementation patterns identified",
    "Django REST framework JWT support confirmed",
    "Multi-tenancy isolation requirements documented"
  ],
  "implementation_components": [
    "oauth2", "jwt", "multi-tenant", "rbac", "session-management"
  ],
  "action_items": [
    "Set up Django OAuth2 provider integration",
    "Implement JWT token management system", 
    "Create tenant isolation middleware",
    "Build RBAC permission framework"
  ],
  "success_criteria": [
    "Secure multi-tenant authentication",
    "Scalable permission system", 
    "OAuth2 compliance",
    "Performance under load"
  ],
  "risk_mitigation": {
    "All tenants have same permission structure": "Design flexible permission schema per tenant",
    "OAuth2 provider will always be available": "Implement fallback authentication methods",
    "Technical complexity": "Phase implementation with incremental testing"
  }
}
```

## 🤖 Enhanced AutoCode Examples

### Complex Command Suggestions with Structured Thinking

```python
# Request complex implementation guidance
response = await suggest_command(
    intent="implement microservice event sourcing architecture with CQRS pattern for order management system",
    context={
        "project_type": "enterprise",
        "language": "python",
        "framework": "fastapi", 
        "services": ["order", "payment", "inventory", "notification"],
        "complexity": "high"
    },
    use_structured_thinking=True
)
```

**Enhanced Response:**
```json
{
  "intent": "implement microservice event sourcing architecture with CQRS pattern for order management system",
  "suggestions": [
    {
      "command": "mkdir -p services/{order,payment,inventory,notification}/src/{commands,queries,events}",
      "confidence": 0.95,
      "reasoning": "Creates structured microservice architecture with CQRS separation",
      "thinking_analysis": {
        "session_id": "microservice_analysis_20240121_150000",
        "confidence_boost": 0.1,
        "research_backing": 4,
        "component_analysis": {
          "components": [
            "API integration", 
            "Data persistence",
            "Performance optimization",
            "Testing strategy"
          ],
          "tags": ["microservices", "event-sourcing", "cqrs", "architecture"],
          "axioms": [
            "Single responsibility per service",
            "Event-driven communication",
            "Data consistency through events"
          ],
          "assumptions": [
            "Services can be independently deployed",
            "Event store provides reliable delivery"
          ]
        }
      }
    },
    {
      "command": "poetry add fastapi uvicorn sqlalchemy alembic pydantic-settings",
      "confidence": 0.9,
      "reasoning": "Essential FastAPI stack for microservice development",
      "thinking_analysis": {
        "session_id": "microservice_analysis_20240121_150000",
        "research_backing": 3,
        "component_analysis": {
          "framework_specific": true,
          "dependency_rationale": "FastAPI ecosystem best practices"
        }
      }
    }
  ],
  "structured_thinking_analysis": {
    "problem_definition": {
      "intent": "implement microservice event sourcing architecture with CQRS pattern for order management system",
      "context_factors": ["project_type", "language", "framework", "services", "complexity"],
      "suggestion_count": 2
    },
    "research_findings": {
      "similar_patterns": 4
    },
    "analysis": {
      "high_confidence_suggestions": 2,
      "similar_historical_patterns": 4,
      "risk_factors": [
        "Complex pipeline operation",
        "Multi-service coordination required"
      ]
    },
    "synthesis": {
      "recommended_approach": {
        "command": "mkdir -p services/{order,payment,inventory,notification}/src/{commands,queries,events}",
        "confidence": 0.95
      },
      "alternative_approaches": [
        {
          "command": "poetry add fastapi uvicorn sqlalchemy alembic pydantic-settings",
          "confidence": 0.9
        }
      ],
      "historical_success_rate": 0.85
    },
    "thinking_applied": true
  },
  "total_suggestions": 2
}
```

### Project Pattern Analysis with Structured Insights

```python
# Analyze project patterns with enhanced intelligence
response = await get_project_patterns(
    project_path="/workspace/ecommerce-platform"
)
```

**Enhanced Response:**
```json
{
  "framework": {
    "primary": "fastapi",
    "version": "0.104.1", 
    "confidence": 0.92,
    "supporting_evidence": [
      "main.py with FastAPI app instance",
      "requirements.txt includes fastapi",
      "API route patterns detected"
    ]
  },
  "language": {
    "primary": "python",
    "version": "3.11",
    "distribution": {
      "python": 0.85,
      "javascript": 0.10,
      "yaml": 0.05
    }
  },
  "architecture_patterns": {
    "microservices": {
      "detected": true,
      "confidence": 0.88,
      "services": ["order", "payment", "inventory", "notification"],
      "communication": "event-driven"
    },
    "database": {
      "primary": "postgresql",
      "orm": "sqlalchemy",
      "migrations": "alembic"
    },
    "testing": {
      "framework": "pytest",
      "coverage": 0.78,
      "test_types": ["unit", "integration", "e2e"]
    }
  },
  "structured_analysis": {
    "thinking_session_id": "pattern_analysis_20240121_150030", 
    "analysis_depth": "structured_thinking_applied",
    "pattern_confidence": 0.94,
    "research_insights": [
      "Found 3 similar FastAPI microservice projects in memory",
      "Event sourcing patterns match previous implementations",
      "Database schema follows established conventions"
    ],
    "component_analysis": {
      "complexity_score": 0.85,
      "maintainability_score": 0.78,
      "scalability_indicators": [
        "Microservice architecture",
        "Event-driven communication", 
        "Database per service pattern"
      ]
    },
    "recommendations": [
      "Consider implementing circuit breaker pattern",
      "Add distributed tracing for better observability",
      "Implement centralized configuration management"
    ]
  }
}
```

## 📝 Memory Integration Examples

### Structured Thinking Memory Storage

```python
# Store structured thinking session as memory
memory_id = await store_memory(
    memory_type="structured_thinking",
    content="Completed authentication system analysis with 5-stage thinking process. Identified OAuth2, JWT, multi-tenancy, and RBAC requirements. Generated actionable implementation plan.",
    importance=0.9,
    metadata={
        "session_id": "auth_implementation_20240121",
        "stages_completed": ["problem_definition", "research", "analysis", "synthesis", "conclusion"],
        "total_thoughts": 5,
        "assumptions_challenged": 3,
        "relationships_formed": 4,
        "structured_thinking_session": True,
        "implementation_domain": "authentication"
    }
)

# Retrieve similar thinking sessions
similar_sessions = await retrieve_memory(
    query="authentication system structured thinking",
    memory_types=["structured_thinking", "solution_synthesis"],
    limit=3,
    min_similarity=0.7
)
```

### Proactive Memory with Structured Thinking

```python
# Configure proactive memory for structured thinking
await configure_proactive_memory(
    enabled=True,
    file_access_triggers=True,
    tool_execution_triggers=True,
    context_change_triggers=True,
    min_similarity_threshold=0.65,
    max_memories_per_trigger=5,
    auto_present_memories=True
)

# Check for relevant structured thinking memories
relevant_memories = await check_relevant_memories(
    context={
        "file_path": "/project/authentication/views.py",
        "task": "implementing JWT token validation",
        "thinking_stage": "synthesis"
    },
    auto_execute=True,
    min_similarity=0.6
)
```

## 🔍 Advanced Analysis Examples

### Relationship Analysis

```python
# Analyze thought relationships in detail
relationship_analysis = await analyze_thought_relationships(
    session_id="auth_implementation_20240121",
    relationship_types=["builds_on", "extends", "challenges"]
)
```

**Response:**
```json
{
  "session_id": "auth_implementation_20240121",
  "total_relationships": 8,
  "relationship_patterns": {
    "builds_on": 4,
    "extends": 2, 
    "challenges": 1,
    "supports": 1
  },
  "stage_connections": {
    "problem_definition": 0,
    "research": 2,
    "analysis": 4,
    "synthesis": 2
  },
  "detailed_relationships": [
    {
      "source_thought": 2,
      "source_stage": "research", 
      "target_thought_id": "thought_auth_001",
      "relationship_type": "builds_on",
      "strength": 0.9,
      "description": "Research builds on the problem definition"
    }
  ],
  "analysis": {
    "most_common_relationship": ["builds_on", 4],
    "most_connected_stage": ["analysis", 4],
    "average_strength": 0.82
  }
}
```

### Session Continuation

```python
# Get guidance for continuing thinking process
continuation = await continue_thinking_process(
    session_id="auth_implementation_20240121",
    suggested_stage="synthesis",
    context_query="authentication patterns microservices"
)
```

**Response:**
```json
{
  "session_id": "auth_implementation_20240121",
  "session_title": "Authentication System Implementation",
  "current_state": {
    "total_thoughts": 3,
    "stages_completed": ["problem_definition", "research", "analysis"],
    "last_stage": "analysis"
  },
  "continuation_suggestions": {
    "next_stage": "synthesis", 
    "current_progress": "3/5 stages completed",
    "last_thought_number": 3,
    "suggested_focus": "Combine insights to formulate solutions and approaches",
    "related_thoughts": 2
  },
  "relevant_context": [
    {
      "memory_type": "solution_synthesis",
      "content": "JWT token management with refresh token pattern",
      "similarity": 0.78
    },
    {
      "memory_type": "project_pattern", 
      "content": "Multi-tenant authentication architecture",
      "similarity": 0.82
    }
  ],
  "session_progress": 60
}
```

## 🎯 Workflow Integration Examples

### End-to-End Development Workflow

```python
# 1. Start structured thinking for new feature
session_response = await process_structured_thought(
    stage="problem_definition",
    content="Implement real-time notifications for order status updates",
    thought_number=1,
    tags=["notifications", "real-time", "orders"]
)

session_id = session_response["session_id"]

# 2. Get enhanced command suggestions during implementation
command_suggestions = await suggest_command(
    intent="set up WebSocket connections for real-time notifications",
    context={
        "current_session": session_id,
        "framework": "fastapi",
        "complexity": "medium"
    },
    use_structured_thinking=True
)

# 3. Analyze current project patterns
patterns = await get_project_patterns(
    project_path="/workspace/notification-service"
)

# 4. Continue thinking process with new insights
await process_structured_thought(
    stage="synthesis", 
    content=f"Based on analysis, will use FastAPI WebSocket with Redis pub/sub. Patterns show {patterns['architecture_patterns']['communication']} is already established.",
    thought_number=4,
    session_id=session_id,
    relationships=[{
        "target_thought_id": session_response["thought_id"],
        "relationship_type": "extends",
        "strength": 0.8
    }]
)

# 5. Generate final action plan
action_plan = await generate_coding_action_plan(session_id=session_id)
```

### Performance Monitoring Integration

```python
# Monitor structured thinking performance
thinking_stats = await memory_stats()
performance_stats = await qdrant_performance_stats()

# Analyze structured thinking usage patterns
structured_memories = await list_memories(
    types=["structured_thinking", "thought_process", "thinking_relationship"],
    limit=100,
    include_content=False
)

# Optimize if needed
if performance_stats["performance_rating"] != "excellent":
    optimization_result = await optimize_qdrant_collection()
```

## 🔧 Configuration Examples

### Custom Structured Thinking Configuration

```json
{
  "autocode": {
    "enabled": true,
    "structured_thinking": {
      "enabled": true,
      "integration_level": "enhanced",
      "complexity_threshold": 2,
      "auto_generate_action_plans": true,
      "track_assumption_evolution": true
    },
    "session_analysis": {
      "enabled": true,
      "apply_structured_thinking": true
    }
  },
  "memory": {
    "structured_thinking_retention": {
      "core_sessions": 365,
      "thought_processes": 180,
      "relationships": 90
    }
  }
}
```

### Proactive Memory for Structured Thinking

```json
{
  "proactive_memory": {
    "enabled": true,
    "triggers": {
      "structured_thinking_sessions": true,
      "complex_commands": true,
      "project_analysis": true
    },
    "similarity_threshold": 0.65,
    "max_memories_per_context": 5,
    "prioritize_structured_thinking": true
  }
}
```

This comprehensive set of examples demonstrates how to leverage Alunai Clarity's structured thinking capabilities for systematic problem-solving, enhanced code intelligence, and improved AI workflows. The integration of structured thinking with memory management and AutoCode features creates a powerful system for tackling complex development challenges.