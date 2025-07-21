# 🛠️ Enhanced Structured Thinking Tools Reference

**MCP Tools for memory-level structured thinking intelligence**

## 🆕 New MCP Tools

### `auto_progress_thinking_stage`
**Automatically progress to next thinking stage with intelligent content generation**

```bash
auto_progress_thinking_stage --session-id session_123 --auto-execute true
```

**Parameters:**
- `session_id` (required): ID of existing thinking session
- `auto_execute` (optional, default: true): Whether to automatically generate and store content

**Returns:**
```json
{
  "status": "auto_progressed",
  "stage": "research", 
  "thought_number": 2,
  "content": "Generated research content...",
  "confidence": 0.85,
  "auto_generated": true,
  "next_stage_available": true
}
```

**Use Cases:**
- Speed up thinking process with AI-generated content
- Get unstuck when unsure how to continue a thinking stage
- Generate comprehensive analysis quickly for time-sensitive decisions

---

### `suggest_proactive_thinking`
**Get context-aware suggestions for when structured thinking would be beneficial**

```bash  
suggest_proactive_thinking --context {
  "current_task": "Debug distributed system performance issues",
  "files_accessed": ["/logs/errors.log", "/services/api.py"],
  "commands_executed": ["docker stats", "grep ERROR /logs/*.log"],
  "recent_activity": ["Investigating timeouts", "Checking resources"]
} --limit 5
```

**Parameters:**
- `context` (required): Current work context with task, files, commands, activity
- `limit` (optional, default: 3): Maximum number of suggestions to return

**Returns:**
```json
{
  "suggestions": [
    {
      "type": "debugging_strategy",
      "priority": "high", 
      "suggestion": "Systematic debugging approach for recurring errors",
      "thinking_stages": ["problem_definition", "research", "analysis"],
      "confidence": 0.92,
      "estimated_time": "15-20 minutes",
      "benefits": ["Root cause identification", "Prevention strategies"]
    }
  ],
  "total_found": 3,
  "context_analysis": {
    "task_complexity": 0.8,
    "file_complexity": 2,
    "decision_points": 0
  },
  "reasoning": ["Multiple error patterns detected - systematic debugging recommended"]
}
```

**Use Cases:**
- Get intelligent suggestions based on your current work context
- Discover when structured thinking would be most beneficial
- Receive time estimates and benefits for different thinking approaches

---

### `auto_trigger_thinking_from_context`
**Automatically start structured thinking based on context analysis**

```bash
auto_trigger_thinking_from_context --context {
  "current_task": "Migrate legacy system to microservices",
  "project_context": {
    "detected_frameworks": ["php", "mysql"],
    "complexity": 0.9
  }
} --threshold 0.8
```

**Parameters:**
- `context` (required): Rich context including task, project info, files, commands
- `threshold` (optional, default: 0.8): Minimum confidence required to auto-trigger

**Returns:**
```json
{
  "status": "auto_triggered",
  "session_id": "session_20240121_143052",
  "suggestion_type": "complex_task_analysis", 
  "confidence": 0.94,
  "estimated_time": "25-30 minutes",
  "benefits": ["Systematic analysis", "Risk identification", "Actionable plan"],
  "auto_generated": true
}
```

**Use Cases:**
- Automatically start structured thinking for complex problems
- Let AI decide when systematic analysis would be most valuable
- Get immediate structured thinking without manual setup

---

### `get_enhanced_thinking_suggestions`
**Get comprehensive thinking suggestions with full context integration**

```bash
get_enhanced_thinking_suggestions --context {
  "current_task": "Optimize database performance", 
  "project_context": {
    "detected_frameworks": ["postgresql", "redis", "python"],
    "project_complexity": 0.7
  },
  "files_accessed": ["/models/user.py", "/logs/slow_queries.log"],
  "commands_executed": ["psql -c 'EXPLAIN ANALYZE'", "redis-cli info"]
}
```

**Parameters:**
- `context` (required): Comprehensive context with task, project, files, commands, activity

**Returns:**
```json
{
  "suggestions": [
    {
      "type": "performance_analysis",
      "priority": "high",
      "suggestion": "Analyze database query optimization opportunities",
      "confidence": 0.89,
      "auto_executable": true,
      "hook_integration": true,
      "memory_integration": true,
      "adjusted_time_estimate": "18-23 minutes"
    }
  ],
  "active_sessions": {
    "count": 1,
    "sessions": [
      {
        "session_id": "session_123",
        "type": "performance_analysis", 
        "stage": "research",
        "auto_started": true
      }
    ]
  },
  "context_analysis": {
    "complexity_score": 0.75,
    "intelligence_level": "high",
    "auto_progression_recommended": true
  }
}
```

**Use Cases:**
- Get the most intelligent suggestions with full context awareness
- See active thinking sessions and their progress
- Understand how context complexity affects suggestions and timing

---

## 🔧 Enhanced Existing Tools

### `process_structured_thought` (Enhanced)
**Now includes automatic stage progression suggestions**

Original functionality plus:
- Auto-suggests next stage based on current thought content
- Provides confidence scoring for stage completion
- Integrates with memory system for research enhancement

### `generate_thinking_summary` (Enhanced)  
**Now includes automatic action plan generation**

Original functionality plus:
- Generates concrete coding action plans
- Provides implementation priorities and timelines
- Creates actionable next steps with success criteria

### `continue_thinking_process` (Enhanced)
**Now includes proactive progression recommendations**

Original functionality plus:
- Analyzes optimal next steps based on session progress
- Provides context-aware continuation guidance  
- Suggests when to switch between manual and auto modes

---

## 🎯 Usage Patterns

### Pattern 1: Fully Automated Analysis
```bash
# 1. Let system detect opportunity and auto-trigger
auto_trigger_thinking_from_context --context [RICH_CONTEXT] --threshold 0.8

# 2. Auto-progress through all stages  
auto_progress_thinking_stage --session-id [SESSION_ID] --auto-execute true
# (Repeat until completion)

# 3. Get final action plan automatically
# (Generated when thinking process completes)
```

### Pattern 2: Guided Analysis with Human Oversight
```bash
# 1. Get proactive suggestions first
suggest_proactive_thinking --context [CONTEXT] --limit 5

# 2. Review suggestions and choose approach
auto_trigger_thinking_from_context --threshold 0.7

# 3. Auto-progress with review at each stage
auto_progress_thinking_stage --auto-execute false  # Review before storing
# Manual review and edit, then continue
```

### Pattern 3: Context-Aware Development Planning
```bash
# 1. Get enhanced suggestions with full context
get_enhanced_thinking_suggestions --context [PROJECT_CONTEXT]

# 2. Use suggestions to inform development approach
# 3. Auto-trigger for complex features
auto_trigger_thinking_from_context --threshold 0.8

# 4. Let system guide implementation planning
auto_progress_thinking_stage --auto-execute true
```

---

## ⚡ Performance Characteristics

| Tool | Typical Response Time | Context Required | Intelligence Level |
|------|---------------------|------------------|-------------------|
| `suggest_proactive_thinking` | 5-10 seconds | Medium | High |
| `auto_trigger_thinking_from_context` | 8-15 seconds | High | Very High |
| `auto_progress_thinking_stage` | 10-20 seconds | Session-based | Very High |
| `get_enhanced_thinking_suggestions` | 6-12 seconds | Very High | Maximum |

**Optimization Tips:**
- Rich context = better suggestions and faster decisions
- Higher confidence thresholds = slower but higher quality
- Auto-execution = faster workflow, less control
- Manual review = slower workflow, more control

---

## 🛡️ Error Handling

### Common Error Responses

**Insufficient Context:**
```json
{
  "status": "below_threshold",
  "suggestions": [...],
  "message": "Suggestions available but below confidence threshold 0.8"
}
```
**Resolution:** Lower threshold or provide richer context

**Session Not Found:**
```json
{
  "error": "Session not found"  
}
```
**Resolution:** Check session ID or create new session

**Extension Unavailable:**
```json
{
  "error": "Structured thinking extension not available"
}
```  
**Resolution:** Ensure MCP server is properly configured with structured thinking

---

## 🔍 Debugging Commands

**Check System Status:**
```bash
# Verify structured thinking is working
process_structured_thought --stage problem_definition --content "Test problem"

# Check memory integration  
retrieve_memory --query "structured thinking" --types ["thinking_session"]

# Validate hook system
memory_stats  # Should show hook executions
```

**Context Testing:**
```bash
# Test minimal context
suggest_proactive_thinking --context {"current_task": "Simple test"}

# Test rich context  
suggest_proactive_thinking --context {
  "current_task": "Complex debugging",
  "files_accessed": ["/logs/error.log"],
  "commands_executed": ["grep ERROR"],
  "recent_activity": ["Debugging system"]
}
```

---

## 💡 Pro Tips

### Maximizing Intelligence
1. **Provide Rich Context**: Include project frameworks, recent files, commands
2. **Use Auto-Execute**: Let system generate content, then refine as needed  
3. **Trust High Confidence**: Suggestions >0.8 confidence are typically valuable
4. **Build Context Gradually**: System gets smarter as you work on related files

### Integration with Memory System
- Auto-generated research stages retrieve similar past problems
- Action plans reference previously successful implementation patterns
- Component detection improves based on past project analysis
- Context building leverages memory of your working patterns

### Workflow Optimization
- Start with `suggest_proactive_thinking` to understand opportunities
- Use `auto_trigger_thinking_from_context` for complex problems (confidence >0.8)
- Let `auto_progress_thinking_stage` handle routine progression
- Reserve manual thinking for creative or highly specialized problems

**The enhanced tools provide the same level of intelligence and automation as the memory system - use them to transform your problem-solving workflow! 🚀**