# MCP Tools Reference

## Overview

Alunai Clarity provides **26 comprehensive MCP tools** that enable advanced memory management, structured thinking, AutoCode intelligence, and proactive AI capabilities. This reference covers all available tools with detailed parameters, usage examples, and integration patterns.

## 🧠 Core Memory Tools

### `store_memory`
Store new information in the high-performance Qdrant vector database.

**Parameters:**
- `memory_type` (string): Type of memory (conversation, fact, document, etc.)
- `content` (string): The content to store
- `importance` (float, optional): Importance score 0.0-1.0 (default: 0.5)
- `metadata` (object, optional): Additional metadata
- `context` (object, optional): Context information

**Returns:** Memory ID

**Example:**
```json
{
  "memory_type": "fact",
  "content": "Python uses duck typing for dynamic type checking",
  "importance": 0.8,
  "metadata": {
    "category": "programming",
    "language": "python"
  }
}
```

### `retrieve_memory`
Retrieve relevant memories based on query with vector similarity search.

**Parameters:**
- `query` (string): Search query
- `limit` (integer, optional): Maximum results (default: 5)
- `types` (array, optional): Filter by memory types
- `min_similarity` (float, optional): Minimum similarity score (default: 0.6)
- `include_metadata` (boolean, optional): Include metadata in results

**Returns:** Array of relevant memories

**Example:**
```json
{
  "query": "python type checking",
  "limit": 3,
  "types": ["fact", "code"],
  "min_similarity": 0.7,
  "include_metadata": true
}
```

### `list_memories`
List memories with filtering and pagination options.

**Parameters:**
- `types` (array, optional): Filter by memory types
- `limit` (integer, optional): Maximum results (default: 20)
- `offset` (integer, optional): Pagination offset
- `tier` (string, optional): Filter by memory tier
- `include_content` (boolean, optional): Include full content

**Returns:** Array of memories with metadata

### `update_memory`
Update existing memory entries.

**Parameters:**
- `memory_id` (string): ID of memory to update
- `updates` (object): Fields to update

**Returns:** Success status

### `delete_memory`
Remove specific memories by ID.

**Parameters:**
- `memory_ids` (array): Array of memory IDs to delete

**Returns:** Success status

### `memory_stats`
Get comprehensive statistics about the memory store.

**Returns:** Memory statistics including counts, types, performance metrics

## 🧠 Structured Thinking Tools

### `process_structured_thought`
Record and analyze structured thoughts with comprehensive metadata.

**Parameters:**
- `stage` (string): Thinking stage (problem_definition, research, analysis, synthesis, conclusion)
- `content` (string): Thought content
- `thought_number` (integer): Sequential number in thinking process
- `session_id` (string, optional): Thinking session ID
- `total_expected` (integer, optional): Expected total thoughts
- `tags` (array, optional): Thought tags
- `axioms` (array, optional): Guiding principles
- `assumptions_challenged` (array, optional): Assumptions being challenged
- `relationships` (array, optional): Relationships to other thoughts

**Returns:** Thought ID, memory ID, session insights, next suggested stage

**Example:**
```json
{
  "stage": "analysis",
  "content": "The authentication system needs JWT and session support",
  "thought_number": 3,
  "session_id": "auth_session_123",
  "tags": ["authentication", "security"],
  "axioms": ["Security by design"],
  "assumptions_challenged": ["Users always use secure passwords"],
  "relationships": [
    {
      "target_thought_id": "thought_1",
      "relationship_type": "builds_on",
      "strength": 0.9
    }
  ]
}
```

### `generate_thinking_summary`
Generate comprehensive thinking process summary.

**Parameters:**
- `session_id` (string): Thinking session ID
- `include_relationships` (boolean, optional): Include relationship analysis
- `include_stage_summaries` (boolean, optional): Include stage-by-stage summaries

**Returns:** Complete session analysis with relationships, confidence scores, and insights

### `continue_thinking_process`
Get context and suggestions for continuing structured thinking.

**Parameters:**
- `session_id` (string): Thinking session ID
- `suggested_stage` (string, optional): Suggested next stage
- `context_query` (string, optional): Query for relevant context

**Returns:** Current state, continuation suggestions, relevant context

### `analyze_thought_relationships`
Analyze and visualize relationships between thoughts.

**Parameters:**
- `session_id` (string): Thinking session ID
- `relationship_types` (array, optional): Filter by relationship types

**Returns:** Relationship patterns, connections, and analysis

## 🤖 AutoCode Intelligence Tools

### `suggest_command`
Get intelligent command suggestions with optional structured thinking analysis.

**Parameters:**
- `intent` (string): What the user wants to accomplish
- `context` (object, optional): Current context
- `use_structured_thinking` (boolean, optional): Apply structured thinking analysis

**Returns:** Command suggestions with confidence scores and reasoning

**Example:**
```json
{
  "intent": "run tests for authentication module",
  "context": {
    "project_type": "django",
    "language": "python"
  },
  "use_structured_thinking": true
}
```

### `get_project_patterns`
Get detected patterns for a project with enhanced analysis.

**Parameters:**
- `project_path` (string): Path to the project
- `pattern_types` (array, optional): Filter by pattern types

**Returns:** Project patterns with structured thinking insights

### `find_similar_sessions`
Find coding sessions similar to current context.

**Parameters:**
- `query` (string): Search query
- `context` (object, optional): Current context
- `time_range_days` (integer, optional): Time range for search

**Returns:** Similar sessions with relevance scores

### `get_continuation_context`
Get relevant context for continuing work on a task.

**Parameters:**
- `current_task` (string): Description of current task
- `project_context` (object, optional): Project context

**Returns:** Relevant context for task continuation

### `suggest_workflow_optimizations`
Suggest workflow improvements based on historical data.

**Parameters:**
- `current_workflow` (array): Current workflow steps
- `session_context` (object, optional): Session context

**Returns:** Workflow optimization suggestions

### `get_learning_progression`
Track learning progression on a specific topic.

**Parameters:**
- `topic` (string): Topic to track
- `time_range_days` (integer, optional): Time range (default: 180)

**Returns:** Learning progression analysis

### `autocode_stats`
Get AutoCode domain statistics and insights.

**Returns:** Comprehensive AutoCode statistics

## ⚡ Performance & Monitoring Tools

### `qdrant_performance_stats`
Get detailed Qdrant performance statistics and optimization recommendations.

**Returns:** Performance metrics, indexing ratios, recommendations

**Example Response:**
```json
{
  "performance_stats": {
    "total_memories": 45230,
    "indexed_memories": 45230,
    "indexing_ratio_percent": 100.0,
    "performance_rating": "excellent",
    "disk_size_mb": 892.5,
    "estimated_search_time_ms": 1.2
  },
  "recommendations": [
    "Performance is optimal"
  ]
}
```

### `optimize_qdrant_collection`
Optimize the Qdrant collection for better performance.

**Returns:** Optimization results and updated statistics

## 🔮 Proactive Memory Tools

### `suggest_memory_queries`
Suggest memory queries that Claude should execute based on current context.

**Parameters:**
- `current_context` (object): Current context information
- `task_description` (string, optional): Description of current task
- `limit` (integer, optional): Maximum suggestions (default: 3)

**Returns:** Suggested memory queries with reasoning

**Example:**
```json
{
  "current_context": {
    "file_path": "/project/auth.py",
    "command": "pytest tests/test_auth.py"
  },
  "task_description": "Testing authentication module"
}
```

### `check_relevant_memories`
Automatically check for and return relevant memories based on context.

**Parameters:**
- `context` (object): Current context
- `auto_execute` (boolean, optional): Automatically execute queries (default: true)
- `min_similarity` (float, optional): Minimum similarity threshold

**Returns:** Relevant memories organized by query

### `configure_proactive_memory`
Configure proactive memory checking behavior.

**Parameters:**
- `enabled` (boolean, optional): Enable proactive memory (default: true)
- `file_access_triggers` (boolean, optional): Trigger on file access
- `tool_execution_triggers` (boolean, optional): Trigger on tool execution
- `context_change_triggers` (boolean, optional): Trigger on context changes
- `min_similarity_threshold` (float, optional): Similarity threshold
- `max_memories_per_trigger` (integer, optional): Maximum memories per trigger
- `auto_present_memories` (boolean, optional): Auto-present relevant memories

**Returns:** Configuration confirmation

### `get_proactive_memory_stats`
Get statistics about proactive memory usage and effectiveness.

**Returns:** Usage statistics, effectiveness metrics, activity summaries

## 📊 Memory Types

### Core Memory Types
- `conversation`: Chat conversations and interactions
- `fact`: Factual information and knowledge
- `document`: Document content and summaries
- `entity`: People, places, organizations
- `reflection`: Insights and learnings
- `code`: Code snippets and programming knowledge

### AutoCode Memory Types  
- `project_pattern`: Project architectural patterns
- `command_pattern`: Bash command patterns and success rates
- `session_summary`: Coding session summaries
- `bash_execution`: Individual command executions

### Structured Thinking Memory Types
- `structured_thinking`: Complete thinking sessions
- `thought_process`: Individual thoughts with stage metadata
- `thinking_relationship`: Connections between thoughts
- `problem_analysis`: Problem definition stage thoughts
- `research_notes`: Research findings and insights
- `analysis_result`: Analysis stage conclusions
- `solution_synthesis`: Solution approaches and synthesis
- `conclusion_summary`: Final decisions and action plans

## 🔧 Integration Patterns

### Basic Memory Operations
```python
# Store and retrieve pattern
memory_id = await store_memory(
    memory_type="fact",
    content="Important information",
    importance=0.8
)

memories = await retrieve_memory(
    query="important information",
    limit=5
)
```

### Structured Thinking Workflow
```python
# Complete structured thinking session
thought_id = await process_structured_thought(
    stage="problem_definition",
    content="Define the problem clearly",
    thought_number=1
)

# Continue through stages...
summary = await generate_thinking_summary(session_id=session_id)
```

### Enhanced AutoCode Usage
```python
# Get enhanced command suggestions
suggestions = await suggest_command(
    intent="complex implementation task",
    context={"complexity": "high"},
    use_structured_thinking=True
)

# Analyze project with structured thinking
patterns = await get_project_patterns(project_path="/path/to/project")
```

### Proactive Memory Integration
```python
# Configure proactive behavior
await configure_proactive_memory(
    enabled=True,
    file_access_triggers=True,
    min_similarity_threshold=0.7
)

# Check for relevant memories automatically
relevant = await check_relevant_memories(
    context={"file_path": "important_file.py"},
    auto_execute=True
)
```

## 🎯 Best Practices

### Memory Management
- Use appropriate importance scores (0.0-1.0)
- Include relevant metadata for better retrieval
- Use specific memory types for better organization
- Regular cleanup of outdated memories

### Structured Thinking
- Complete all 5 stages for complex problems
- Use descriptive content and appropriate tags
- Build relationships between related thoughts
- Challenge assumptions at each stage

### AutoCode Integration
- Provide rich context for better suggestions
- Use structured thinking for complex tasks
- Monitor learning progression over time
- Optimize workflows based on historical data

### Performance Optimization
- Monitor performance stats regularly
- Optimize collections when recommended
- Use appropriate similarity thresholds
- Configure proactive memory for your workflow

This comprehensive toolset enables sophisticated AI workflows with memory persistence, structured thinking, and intelligent code assistance, all backed by high-performance vector storage and proactive intelligence.