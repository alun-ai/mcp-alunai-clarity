# Alunai Clarity

**An MCP server that provides cognitive abilities to Claude and other MCP-aware systems.**

High-performance SQLite vector storage with intelligent memory management, procedural thinking, and real-time MCP server discovery.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)

## Technical Capabilities

Alunai Clarity provides the following core capabilities:

- **High-performance vector memory storage** using SQLite with sqlite-vec extension for sub-millisecond search
- **Procedural and sequential thinking** with 5-stage systematic analysis framework
- **Automatic memory capture** from "Remember this:" patterns without explicit tool calls
- **Real-time MCP server discovery** and proactive tool suggestions
- **Connection pooling with recovery** for reliable production deployments
- **Concurrent session support** for multiple Claude instances sharing memory
- **Intelligent embedding models** with fallback system for faster initialization
- **AutoCode intelligence** for project pattern recognition and workflow optimization
- **Docker containerization** with volume mounting for persistent storage
- **Claude Code hooks integration** for seamless user experience
- **Performance monitoring** with optimization recommendations
- **JSON migration tools** for upgrading from file-based storage

## Architecture

### Memory System
- **Vector Database**: SQLite with sqlite-vec extension for high-performance vector search
- **Embedding Models**: Configurable sentence transformers with intelligent fallbacks
- **Connection Pooling**: Shared client management with automatic recovery mechanisms
- **Memory Types**: 12 specialized types including structured thinking and project patterns
- **Tiered Storage**: Short-term, long-term, and archival memory tiers

### Thinking Framework
- **Native Sequential Thinking**: 5-stage analysis process (problem_analysis → context_exploration → solution_generation → evaluation_analysis → implementation_planning)
- **Procedural Thinking Triggers**: Automatic detection of complex task patterns
- **Claude Code Integration**: UserPromptSubmit and PermissionDecision hooks
- **Pattern Recognition**: Complexity scoring and automatic thinking activation

### Production Features
- **Connection Recovery**: Multi-tier fallback system with timeout protection
- **Stale Lock Detection**: Automatic cleanup of abandoned connection locks
- **Performance Optimization**: Fast initialization with optimized HNSW parameters
- **Error Handling**: Comprehensive error recovery with "instance is closed" detection
- **Health Monitoring**: Real-time connection health checks and statistics

## Quick Start

### Docker Compose Deployment (Recommended)

```bash
# Clone and start
git clone https://github.com/alun-ai/mcp-alunai-clarity.git
cd mcp-alunai-clarity
docker-compose up -d

# Test connection
curl http://localhost:8000/health
```

### MCP Client Configuration

Add to:
- Claude Desktop `claude_desktop_config.json`:
- Cursor/Claude Terminal: `./.mcp.json` or Global `~/.claude.json`

```json
{
  "mcpServers": {
    "alunai-clarity": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "./.claude/alunai-clarity:/app/data",
        "ghcr.io/alun-ai/mcp-alunai-clarity:latest"
      ],
      "type": "stdio"
    }
  }
}
```

Persistent Storage structure:
> Note: These will be created automatically in whatever directory you start Claude in.
- `./.claude/alunai-clarity/config.json` - Configuration
- `./.claude/alunai-clarity/sqlite/` - SQLite database files
- `./.claude/alunai-clarity/cache/` - Model cache
- `./.claude/alunai-clarity/hooks.json` - Claude Code hooks

### Enabling Auto-Capture

Add to your `CLAUDE.md`:

```markdown
You have persistent memory with automatic capture enabled.

Remember important information by saying "Remember this: [content]"
This will automatically store memories without explicit tool calls.
```

## Core Tools

### Memory Management
- `store_memory` - Store memories with vector indexing
- `retrieve_memories` - Semantic similarity search
- `list_memories` - Browse with filtering
- `update_memory` - Modify existing memories
- `delete_memory` - Remove memories
- `memory_stats` - System statistics

### Thinking Framework
- `sequential_thinking` - 5-stage systematic analysis
- `structured_thinking_domain` - Structured thought processing
- Automatic procedural thinking triggers for complex tasks

### MCP Awareness
- `discover_mcp_tools` - Auto-discover installed MCP servers
- `suggest_mcp_alternatives` - Recommend MCP tools over scripts
- `get_mcp_tool_info` - Tool information and schemas

### Performance & Monitoring
- `memory_stats` - Detailed performance metrics and database statistics
- `sqlite_performance_stats` - SQLite-specific performance monitoring
- `connection_health_check` - Connection status monitoring

### AutoCode Intelligence
- `suggest_command` - Context-aware command suggestions
- `get_project_patterns` - Framework and architecture detection
- `find_similar_sessions` - Historical session analysis
- `autocode_stats` - Intelligence system metrics

## Example Configuration
> Alunai Clarity will set these configurations up automatically.  These are just for reference.

### Basic Configuration
```json
{
  "sqlite": {
    "path": "/app/data/sqlite/memory.db",
    "wal_mode": true,
    "timeout": 30.0,
    "max_retries": 3
  },
  "embedding": {
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "fast_model": "paraphrase-MiniLM-L3-v2",
    "dimensions": 384,
    "cache_dir": "/app/data/cache"
  }
}
```

### Advanced Configuration
```json
{
  "sqlite": {
    "path": "/app/data/sqlite/memory.db",
    "wal_mode": true,
    "timeout": 30.0,
    "max_retries": 3,
    "pragma_settings": {
      "journal_mode": "WAL",
      "synchronous": "NORMAL",
      "cache_size": 10000,
      "temp_store": "MEMORY"
    }
  },
  "autocode": {
    "enabled": true,
    "command_learning": {
      "enabled": true,
      "min_confidence_threshold": 0.3
    },
    "pattern_detection": {
      "enabled": true,
      "supported_languages": ["typescript", "javascript", "python", "rust", "go"]
    }
  }
}
```

## Performance

### Memory Search Performance
| Memory Count | Vector Search | Improvement |
|--------------|---------------|-------------|
| 1K memories  | ~0.1ms       | 100x faster |
| 10K memories | ~1ms         | 100x faster |
| 100K memories| ~2ms         | 500x faster |

### Initialization Performance
- First-time startup: ~6s (model loading + initialization)
- Subsequent operations: ~10ms
- Fast model option: ~3s initialization with paraphrase-MiniLM-L3-v2
- Connection recovery: Sub-second automatic failover

## Usage Examples

### Automatic Memory Triggers

The system automatically captures memories by default using the built in hooks functionaly.  However, if you want to trigger manual captures you can use these patterns:

#### Memory Storage Phrases
```
"Remember this: [content]"
"Store this information: [content]"
"Keep track of: [content]"
"Note that: [content]"
"Save this: [content]"
"Don't forget: [content]"
"Remember that." # Used if something important like a summary was in the previous response.
```

#### Procedural Thinking Triggers
Complex tasks automatically activate 5-stage sequential thinking:
> These are just examples, the system will trigger with many similar forms of expression.
```
"How should I approach..."              # Planning requests
"What steps do I need to..."            # Multi-step processes
"Help me design/implement/architect..." # Complex design tasks
"I need to systematically..."           # Systematic approach requests
"Break down this problem..."            # Problem decomposition
"Analyze and solve..."                  # Analysis requests
```

#### Memory Retrieval Phrases
These phrases trigger automatic memory consultation:
> These are just examples, the system will trigger with many similar forms of expression.
```
"What do you remember about..."
"Have we worked on this before..."
"Recall my preferences for..."
"What did we learn about..."
"Show me similar projects..."
"What patterns have we used..."
```

#### MCP Tool Discovery Phrases
These trigger automatic MCP server discovery and tool suggestions:
> Beta - this is still being developed, but works better than native.
```
"I need to query the database..."       # → postgres MCP tools
"Let me browse this website..."         # → playwright MCP tools
"I should run some SQL..."              # → database MCP tools
"I need to test the API..."             # → fetch/curl MCP tools
"Let me check the filesystem..."        # → filesystem MCP tools
```

### Example Workflows

#### 1. Project Setup with Memory
```
User: "Remember this: I prefer using TypeScript with strict mode,
       ESLint with Airbnb config, and Jest for testing"

Claude: ✅ Stored memory: coding_preferences
        🧠 This will be automatically referenced in future projects
```

#### 2. Complex Task Analysis
```
User: "Let's plan out a secure authentication system for our API"

Claude: 🧠 **PROCEDURAL THINKING ACTIVATED**

        Using sequential_thinking for systematic analysis:

        **Stage 1: Problem Analysis**
        - Authentication requirements analysis
        - Security threat modeling
        - API endpoint protection needs

        **Stage 2: Context Exploration**
        - Current system architecture review
        - Available auth frameworks
        - Security best practices

        [Continues through all 5 stages...]
```

#### 3. Automatic Memory Consultation
```
User: "I'm setting up a new React project"

Claude: 🧠 **Relevant memories found:**
        - Your TypeScript + ESLint preferences (stored 2 days ago)
        - React project structure patterns (from 3 similar projects)
        - Testing setup with Jest configuration

        Based on your preferences, I'll set up TypeScript with strict mode...
```

#### 4. MCP Tool Suggestions
```
User: "I need to query the user table to check account status"

Claude: 💡 **MCP Tools Available:**
        - **postgres_query**: Execute SQL queries directly
        - **postgres_schema**: Get table schema information

        Instead of writing a script, let me use the postgres MCP tool:
        [Uses postgres_query tool directly]
```

### Automatic Patterns

#### Memory Auto-Capture Patterns
- **"Remember this:"** → Stores as `user_preference` or `important_fact`
- **Code explanations** → Stored as `code_pattern` or `solution_approach`
- **Project decisions** → Stored as `architectural_decision`
- **Error solutions** → Stored as `troubleshooting_solution`
- **Workflow improvements** → Stored as `workflow_optimization`

#### Thinking Pattern Recognition
- **Long, complex requests** → Triggers sequential thinking
- **Multiple "and" clauses** → Activates systematic breakdown
- **Planning keywords** → Enables structured analysis
- **Problem-solving context** → Engages 5-stage process

#### MCP Tool Pattern Matching
- **Database terms** → Suggests postgres/sqlite MCP tools
- **Web/browser terms** → Suggests playwright/puppeteer tools
- **File operations** → Suggests filesystem MCP tools
- **API testing** → Suggests fetch/http MCP tools
- **Git operations** → Suggests git MCP tools

Results:
- Automatic data integrity verification
- 100x+ performance improvement
- Batch processing for large datasets
- Rollback safety (original files unchanged)

## Troubleshooting

### Connection Issues
- **Hanging operations**: Fixed with timeout protection and automatic recovery
- **"Instance is closed" errors**: Automatic detection and client recreation
- **Stale locks**: Automatic cleanup based on process validation

### Performance Issues
- **Slow initialization**: Use `fast_model` configuration option
- **Memory usage**: Check `qdrant_performance_stats` for optimization
- **Search performance**: Run `optimize_qdrant_collection`

## Development

### Local Development
```bash
# Clone repository
git clone https://github.com/alun-ai/mcp-alunai-clarity.git
cd mcp-alunai-clarity

# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/

# Build Docker image
docker build -t mcp-alunai-clarity .
```

### Testing
```bash
# Run connection fix tests
python test_qdrant_connection_fixes.py

# Performance testing
python -c "
import asyncio
from clarity.domains.persistence import QdrantPersistenceDomain
# Test code here
"
```

## License

MIT License - see [LICENSE](LICENSE) file for details.