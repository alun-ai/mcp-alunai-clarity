# Proactive Memory System 🧠

The Proactive Memory System revolutionizes how Claude accesses and utilizes its memory by **automatically presenting relevant context** and **proactively surfacing memories** instead of requiring manual memory tool invocations.

## 🎯 **Core Problem Solved**

Previously, Claude would:
- ❌ Only access memories when explicitly asked with memory tools
- ❌ Miss relevant context from past work and conversations
- ❌ Require manual `check_relevant_memories` calls for context
- ❌ Not leverage historical patterns and insights automatically

Now Claude:
- ✅ **Automatically presents** relevant memories during workflows
- ✅ **Proactively surfaces** context based on file access and tool usage
- ✅ **Seamlessly integrates** memory consultation into decision-making
- ✅ **Provides continuous context** without interrupting natural conversation flow

## 🚀 **How It Works**

### **1. Automatic Memory Triggering**

The system automatically checks for relevant memories during key activities:

#### **📁 File Access Triggers**
```python
# When you read a file, relevant memories are automatically presented
User reads: "/project/authentication.py"

Claude automatically receives:
🧠 **Relevant Past Context** (triggered by file access: authentication.py)
Based on your file access, here are relevant memories:

1. **Code Pattern** (2025-01-15): "JWT authentication implementation with refresh tokens"
2. **Project Pattern** (2025-01-10): "FastAPI authentication architecture with OAuth2"
3. **Session Summary** (2025-01-08): "Debugging authentication middleware issues"

*This context was automatically retrieved to inform my response.*
```

#### **⚡ Tool Execution Triggers**
```python
# Before executing certain tools, relevant memories are surfaced
User: Edit a file with complex logic

Claude automatically receives:
🧠 **Relevant Past Context** (triggered by tool: Edit)
Based on upcoming file editing, here are relevant memories:

1. **Command Pattern** (2025-01-16): "Similar file editing patterns in this project"
2. **Best Practice** (2025-01-12): "Code style conventions for this codebase"

*This context was automatically retrieved to inform my response.*
```

#### **🔄 Context Change Triggers**
```python
# When conversation context shifts, relevant memories are presented
User shifts from "frontend work" to "database optimization"

Claude automatically receives:
🧠 **Relevant Past Context** (triggered by context change)
Based on the shift to database topics, here are relevant memories:

1. **Technical Insight** (2025-01-14): "Database indexing strategies that improved performance"
2. **Architectural Decision** (2025-01-11): "Choice of PostgreSQL over MongoDB for this project"

*This context was automatically retrieved to inform my response.*
```

### **2. Smart Memory Presentation**

#### **Contextual Formatting**
Memories are automatically formatted for natural integration:

```markdown
🧠 **Relevant Past Context** (triggered by file access: main.py)
Based on your file access, here are relevant memories:

1. **Code Pattern** (2025-01-15): "Error handling patterns used in this module"
   - Try-catch blocks with specific exception types
   - Logging strategy for debugging production issues

2. **Performance Insight** (2025-01-12): "Optimization that improved startup time by 40%"
   - Lazy loading of heavy dependencies
   - Caching configuration values

3. **Recent Change** (2025-01-10): "Recent refactoring to improve testability"
   - Dependency injection pattern implementation
   - Mock-friendly interface design

*This context was automatically retrieved to inform my response.*
```

#### **Relevance-Based Filtering**
- Only shows memories above configured similarity threshold (default: 0.6)
- Limits to most relevant memories (default: 3 per trigger)
- Prioritizes recent and high-importance memories

### **3. Comprehensive Context Checking**

#### **Multi-Query Analysis**
```python
# System generates multiple queries to find comprehensive context
Context: {"file_path": "/project/api/auth.py", "task": "adding 2FA"}

Generated queries:
1. "authentication auth.py two-factor 2FA security"
2. "api authentication endpoint security"  
3. "two factor authentication implementation"
4. "auth.py file modifications security"

# Combines results for comprehensive context coverage
```

#### **Cross-Session Memory Integration**
```python
# Automatically links related work across multiple sessions
Current: Working on user authentication
                           ↓
Past Session 1: "JWT implementation challenges"
Past Session 2: "Security best practices discussion"  
Past Session 3: "Authentication testing strategies"
                           ↓
Present: All relevant context automatically provided
```

## 📋 **Available MCP Tools**

The Proactive Memory System provides 2 new MCP tools for configuration and monitoring, plus manual memory management tools:

### 1. configure_proactive_memory
Configure proactive memory behavior and triggers.

**Parameters:**
- `enabled` (optional): Enable/disable proactive memory (default: true)
- `file_access_triggers` (optional): Enable memory checking on file access (default: true)
- `tool_execution_triggers` (optional): Enable memory checking before tool execution (default: true)
- `context_change_triggers` (optional): Enable memory checking on context changes (default: true)
- `min_similarity_threshold` (optional): Minimum similarity for memory relevance (default: 0.6)
- `max_memories_per_trigger` (optional): Maximum memories to present per trigger (default: 3)
- `auto_present_memories` (optional): Automatically present memories to Claude (default: true)

**Example:**
```json
{
  "enabled": true,
  "file_access_triggers": true,
  "tool_execution_triggers": false,
  "min_similarity_threshold": 0.7,
  "max_memories_per_trigger": 2,
  "auto_present_memories": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Proactive memory configuration updated successfully",
  "config": {
    "enabled": true,
    "triggers": {
      "file_access": true,
      "tool_execution": false,
      "context_change": true
    },
    "similarity_threshold": 0.7,
    "max_memories_per_trigger": 2,
    "auto_present": true
  }
}
```

### 2. get_proactive_memory_stats
Get statistics about proactive memory usage and effectiveness.

**No parameters required**

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_proactive_presentations": 145,
    "presentations_last_24h": 12,
    "analytics_entries": 48,
    "most_common_triggers": [
      {"trigger": "file_access", "count": 89},
      {"trigger": "tool_execution", "count": 34},
      {"trigger": "context_change", "count": 22}
    ],
    "memory_effectiveness": {
      "average_memories_per_presentation": 2.3,
      "most_common_memory_types": [
        {"type": "code_pattern", "count": 67},
        {"type": "session_summary", "count": 45},
        {"type": "project_pattern", "count": 33}
      ]
    },
    "recent_activity": {
      "last_presentation": "2025-01-20T14:30:00Z",
      "presentations_today": 8,
      "average_daily_presentations": 12.5
    }
  }
}
```

### 3. Manual Memory Management Tools

#### store_memory
Store new information in memory explicitly.

**Parameters:**
- `memory_type` (required): Type of memory (e.g., "coding_constraints", "project_pattern", "fact")
- `content` (required): The information to store
- `importance` (optional): Importance score 0.0-1.0 (default: 0.5)
- `metadata` (optional): Additional metadata object
- `context` (optional): Context information

**Trigger Phrases:**
- "Remember this:"
- "Store this in memory"
- "Save this information"

**Example:**
```
User: Remember this: Use async/await for all database operations
Assistant: [Automatically calls store_memory with memory_type="coding_rule"]
```

#### retrieve_memory
Manually search and retrieve relevant memories.

**Parameters:**
- `query` (required): Search query for memories
- `types` (optional): Filter by memory types array
- `limit` (optional): Maximum memories to return (default: 5)
- `min_similarity` (optional): Minimum similarity score (default: 0.6)
- `include_metadata` (optional): Include metadata in results (default: false)

**Example:**
```
User: What do we know about authentication patterns?
Assistant: [Uses retrieve_memory with query="authentication patterns"]
```

### 4. Enhanced check_relevant_memories
The existing `check_relevant_memories` tool is enhanced with automatic context analysis.

**Parameters:**
- `context` (required): Context to search memories against
- `auto_execute` (optional): Whether to automatically execute memory retrieval queries (default: true)
- `min_similarity` (optional): Minimum similarity score for memory matches (default: 0.6)

**Enhanced Features:**
- Automatically generates multiple contextual queries
- Provides comprehensive context analysis
- Returns analytics about memory presentation
- Tracks usage for system optimization

## ⚙️ **Configuration**

### **Default Configuration**
```json
{
  "proactive_memory": {
    "enabled": true,
    "triggers": {
      "file_access": true,
      "tool_execution": true,
      "context_change": true
    },
    "similarity_threshold": 0.6,
    "max_memories_per_trigger": 3,
    "auto_present": true
  }
}
```

### **Fine-Tuning Options**

#### **Trigger Sensitivity**
```json
{
  "proactive_memory": {
    "triggers": {
      "file_access": true,        // Every file read/write
      "tool_execution": true,     // Before Edit, Write, Bash, Read tools
      "context_change": false     // When conversation topic shifts
    }
  }
}
```

#### **Memory Quality Control**
```json
{
  "proactive_memory": {
    "similarity_threshold": 0.8,     // Higher = more relevant memories only
    "max_memories_per_trigger": 2,   // Fewer memories for less noise
    "auto_present": true             // Still present automatically
  }
}
```

#### **Performance Optimization**
```json
{
  "proactive_memory": {
    "enabled": true,
    "triggers": {
      "file_access": true,        // Keep most valuable trigger
      "tool_execution": false,    // Disable for performance
      "context_change": false     // Disable for performance
    },
    "similarity_threshold": 0.7,  // Reduce computation
    "max_memories_per_trigger": 1 // Minimal memory presentation
  }
}
```

### **Environment Variables**

For Docker deployments:

```bash
# Core proactive memory settings
PROACTIVE_MEMORY_ENABLED=true
PROACTIVE_MEMORY_FILE_ACCESS_TRIGGERS=true
PROACTIVE_MEMORY_TOOL_EXECUTION_TRIGGERS=true
PROACTIVE_MEMORY_CONTEXT_CHANGE_TRIGGERS=true

# Quality and performance settings
PROACTIVE_MEMORY_SIMILARITY_THRESHOLD=0.6
PROACTIVE_MEMORY_MAX_MEMORIES_PER_TRIGGER=3
PROACTIVE_MEMORY_AUTO_PRESENT=true
```

## 🔧 **Integration Details**

### **Hook System Architecture**

The proactive memory system integrates with the AutoCode hook system:

```python
# Automatic registration with existing hooks
class HookManager:
    def _on_file_access(self, context):
        # Triggers: file_access
        if self.proactive_config["triggers"]["file_access"]:
            await self._suggest_file_related_memories(file_path)
    
    def _on_tool_pre_execution(self, context):
        # Triggers: tool_execution  
        if self.proactive_config["triggers"]["tool_execution"]:
            await self._suggest_contextual_memories(tool_name, arguments)
    
    def _on_context_change(self, context):
        # Triggers: context_change
        if self.proactive_config["triggers"]["context_change"]:
            await self._auto_trigger_memory_check(context)
```

### **Memory Presentation Flow**

1. **Trigger Detection**: Hook system detects relevant events
2. **Context Analysis**: Extracts keywords and context from the event
3. **Memory Retrieval**: Searches for relevant memories using generated queries
4. **Relevance Filtering**: Applies similarity threshold and limits
5. **Automatic Presentation**: Formats and presents memories to Claude
6. **Analytics Tracking**: Records usage data for optimization

### **Memory Types Integrated**

The system works with all memory types but prioritizes:

- **code**: Patterns and implementations related to current work
- **project_pattern**: Architecture and structure insights
- **session_summary**: Past conversations and decisions
- **command_pattern**: Successful command and tool usage patterns
- **fact**: Technical knowledge and best practices

## 🎯 **Best Practices**

### **1. Optimal Configuration**

#### **For Active Development**
```json
{
  "proactive_memory": {
    "enabled": true,
    "triggers": {
      "file_access": true,      // Essential for code context
      "tool_execution": true,   // Helpful for tool patterns
      "context_change": true    // Good for topic transitions
    },
    "similarity_threshold": 0.6,
    "max_memories_per_trigger": 3
  }
}
```

#### **For High-Performance Environments**
```json
{
  "proactive_memory": {
    "enabled": true,
    "triggers": {
      "file_access": true,      // Keep most valuable
      "tool_execution": false,  // Disable for speed
      "context_change": false   // Disable for speed
    },
    "similarity_threshold": 0.8, // Only highly relevant
    "max_memories_per_trigger": 1
  }
}
```

#### **For Learning and Exploration**
```json
{
  "proactive_memory": {
    "enabled": true,
    "triggers": {
      "file_access": true,
      "tool_execution": true,
      "context_change": true
    },
    "similarity_threshold": 0.5, // Lower for more context
    "max_memories_per_trigger": 5  // More memories
  }
}
```

### **2. Memory Quality Optimization**

#### **Improve Memory Relevance**
- Use descriptive commit messages (stored in memories)
- Include context in conversations (helps with matching)
- Work on related projects to build pattern connections
- Ask Claude to explain decisions (creates richer memories)

#### **Build Valuable Memory Patterns**
- Document architectural decisions in conversations
- Discuss trade-offs and alternatives (creates decision context)
- Share learning insights and discoveries
- Review and reflect on past work periodically

### **3. Effective Usage Patterns**

#### **Manual Memory Storage**
```
✅ Use explicit phrases to trigger memory storage:
   "Remember this: [important information]"
   "Store this in memory: [coding pattern]"
   "Save this information: [project decision]"

✅ Structure important information for storage:
   "Remember this coding constraint: Always validate input parameters"
   "Store this pattern: Use dependency injection for testability"

❌ Don't assume information will be automatically stored
❌ Don't use vague phrases like "remember" without "this:"
```

#### **Manual Memory Retrieval**
```
✅ Ask specific questions to trigger retrieval:
   "What patterns do we have for authentication?"
   "What coding constraints should I follow?"
   "What decisions were made about database design?"

✅ Use retrieve_memory directly for complex searches:
   User: "Search memories for React component patterns"
   Assistant: [Uses retrieve_memory with query="React component patterns"]

❌ Don't expect Claude to automatically know all past context
❌ Don't assume memories are retrieved without asking
```

#### **Let the System Work**
```
❌ Don't manually call check_relevant_memories for every context
✅ Let proactive memory automatically surface relevant context

❌ Don't disable all triggers if experiencing noise
✅ Adjust similarity_threshold to reduce irrelevant memories

❌ Don't ignore presented memories
✅ Acknowledge and build upon automatically provided context
```

#### **Monitor and Adjust**
```python
# Regularly check system effectiveness
Use: get_proactive_memory_stats

# Look for optimization opportunities:
- High presentation count but low relevance → increase similarity_threshold
- Missing important context → decrease similarity_threshold  
- Too much noise → reduce max_memories_per_trigger
- Performance issues → disable less critical triggers
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Proactive Memory Not Working**
1. Check that `enabled` is set to `true`
2. Verify at least one trigger is enabled
3. Check logs for hook registration errors
4. Ensure memory system is properly initialized

**Solution:**
```bash
# Check configuration
curl -X POST http://localhost:8080/tools/get_proactive_memory_stats

# Verify triggers
curl -X POST http://localhost:8080/tools/configure_proactive_memory \
  -d '{"enabled": true, "file_access_triggers": true}'
```

#### **Too Many Irrelevant Memories**
1. Increase `similarity_threshold` (try 0.7 or 0.8)
2. Reduce `max_memories_per_trigger` 
3. Disable less useful triggers

**Solution:**
```json
{
  "similarity_threshold": 0.8,
  "max_memories_per_trigger": 2,
  "tool_execution_triggers": false
}
```

#### **Missing Important Context**
1. Decrease `similarity_threshold` (try 0.5 or 0.4)
2. Increase `max_memories_per_trigger`
3. Enable more trigger types

**Solution:**
```json
{
  "similarity_threshold": 0.5,
  "max_memories_per_trigger": 5,
  "context_change_triggers": true
}
```

#### **Performance Issues**
1. Disable non-essential triggers
2. Increase `similarity_threshold` to reduce computation
3. Reduce `max_memories_per_trigger`
4. Monitor with `get_proactive_memory_stats`

**Solution:**
```json
{
  "triggers": {
    "file_access": true,
    "tool_execution": false,
    "context_change": false
  },
  "similarity_threshold": 0.8,
  "max_memories_per_trigger": 1
}
```

### **Performance Optimization**

#### **Memory Usage**
- Proactive memory adds minimal overhead to existing memory operations
- Memory presentation uses efficient formatting and caching
- Analytics data is stored separately and cleaned periodically

#### **Response Time Impact**
- File access triggers: ~50-100ms additional latency
- Tool execution triggers: ~30-50ms additional latency  
- Context change triggers: ~100-200ms additional latency
- All triggers are asynchronous and non-blocking

#### **Optimization Strategies**
```python
# High-performance configuration
{
  "proactive_memory": {
    "enabled": true,
    "triggers": {
      "file_access": true,        # Keep highest-value trigger
      "tool_execution": false,    # Disable for speed
      "context_change": false     # Disable for speed
    },
    "similarity_threshold": 0.8,  # Reduce computation
    "max_memories_per_trigger": 1, # Minimal presentation
    "auto_present": true
  }
}
```

## 📊 **Analytics and Monitoring**

### **Usage Analytics**

The system automatically tracks:
- **Presentation Frequency**: How often memories are presented
- **Trigger Effectiveness**: Which triggers provide the most value
- **Memory Type Distribution**: What types of memories are most useful
- **Similarity Score Distribution**: How relevant presented memories are

### **Optimization Insights**

Use `get_proactive_memory_stats` to understand:
```json
{
  "optimization_recommendations": [
    "Consider increasing similarity_threshold to 0.7 (currently showing many low-relevance memories)",
    "file_access triggers are highly effective (89% of presentations)",
    "context_change triggers have low usage (consider disabling for performance)",
    "Average 2.3 memories per presentation is optimal"
  ]
}
```

### **Health Monitoring**

Monitor system health with:
- **Presentation Rate**: Should be steady, not constantly high
- **Memory Relevance**: Average similarity scores should be > 0.6
- **Error Rate**: Should be minimal in memory retrieval
- **Response Time**: Triggers should not significantly impact response time

## 🌟 **Advanced Usage**

### **Custom Integration**

#### **With IDE Extensions**
```python
# Automatically trigger memory consultation in IDE extensions
await mcp_client.call_tool("check_relevant_memories", {
    "context": {
        "file_path": current_file,
        "cursor_position": line_number,
        "task": "code_completion"
    },
    "auto_execute": true
})
```

#### **With CI/CD Pipelines**
```python
# Use proactive memory in automated workflows
await mcp_client.call_tool("check_relevant_memories", {
    "context": {
        "pipeline_stage": "deployment",
        "project_path": "/src",
        "environment": "production"
    }
})
```

### **Memory Strategy Optimization**

#### **Project-Specific Tuning**
```python
# Large codebase: Focus on file access
{
  "file_access_triggers": true,
  "tool_execution_triggers": false,
  "similarity_threshold": 0.8,
  "max_memories_per_trigger": 1
}

# Exploratory work: Broad context
{
  "file_access_triggers": true,
  "tool_execution_triggers": true,
  "context_change_triggers": true,
  "similarity_threshold": 0.4,
  "max_memories_per_trigger": 5
}

# Production environment: Minimal overhead
{
  "file_access_triggers": true,
  "tool_execution_triggers": false,
  "context_change_triggers": false,
  "similarity_threshold": 0.9,
  "max_memories_per_trigger": 1
}
```

## 🔗 **Integration with Other Systems**

### **AutoCode Integration**
Proactive memory integrates seamlessly with AutoCode features:
- **Command Learning**: Automatically surfaces relevant command patterns
- **Project Patterns**: Presents architectural insights during development
- **Session History**: Provides context from similar past sessions
- **Workflow Optimization**: Suggests improvements based on memory analysis

### **MCP Awareness Integration**
Works together with MCP Awareness for comprehensive assistance:
- **Tool Discovery**: Proactive memory remembers which MCP tools were effective
- **Usage Patterns**: Surfaces successful tool usage from past sessions
- **Configuration Memory**: Recalls optimal configurations for different scenarios

### **Claude Desktop Integration**
Designed for seamless Claude Desktop experience:
- **Non-Intrusive**: Memories appear naturally in conversation context
- **Performance Optimized**: Minimal impact on response times
- **Configuration Sync**: Settings persist across Claude Desktop sessions

## 📚 **API Reference**

For complete API documentation and additional technical details, see:
- [Main README](../README.md#proactive-memory-system)
- [AutoCode Guide](./autocode_guide.md#proactive-memory-integration)
- [MCP Awareness Documentation](./mcp_awareness.md#memory-integration)

## 🤝 **Support and Feedback**

The Proactive Memory System is designed to enhance your Claude experience through intelligent, automatic context provision. For issues, optimization tips, or feature requests:

1. Check the troubleshooting section above
2. Monitor system performance with `get_proactive_memory_stats`
3. Experiment with configuration options for your workflow
4. Submit feedback through the project repository

The system learns and improves with usage - the more context it has from your conversations and work patterns, the more valuable its automatic memory presentations become.