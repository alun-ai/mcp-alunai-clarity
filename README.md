# Alunai Clarity 🧠

**An MCP server that gives cognitive abilities to Claude and other MCP aware systems.**

Featuring **high-performance Qdrant vector database** storage, **proactive memory consultation**, **real MCP server discovery**, and intelligent **AutoCodeIndex** assistance for enhanced AI capabilities.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/github/v/release/alun-ai/mcp-alunai-clarity?color=blue)
![Performance](https://img.shields.io/badge/performance-10--100x_faster-brightgreen.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)

## Overview

Alunai Clarity transforms Claude and other MCP-aware systems by providing **cognitive abilities** through advanced memory management, **proactive intelligence**, and **real-time MCP awareness**. The system enables AI assistants to maintain context, learn from interactions, and intelligently leverage available tools.

## 🎯 Key Improvements in v1.0.0

### **🧠 Sequential Thinking Integration (New)**
- **5-Stage Thinking Process**: Problem Definition → Research → Analysis → Synthesis → Conclusion
- **Structured Problem Solving**: Systematic approach to complex coding challenges
- **Assumption Tracking**: Monitor and challenge assumptions throughout thinking process
- **Relationship Mapping**: Connect thoughts and build comprehensive understanding
- **Action Plan Generation**: Convert structured thinking into concrete coding steps

### **🚀 High-Performance Vector Database**
- **Qdrant Integration**: Replaced JSON storage with enterprise-grade vector database
- **10-100x Performance**: Sub-millisecond search vs. 100ms+ linear scans
- **Unlimited Scalability**: Handle millions of memories without performance degradation
- **Advanced Filtering**: Vector similarity + metadata filtering in single queries

### **🧠 Enhanced AutoCode Intelligence**
- **Structured Command Analysis**: Apply 5-stage thinking to complex command suggestions
- **Pattern Analysis Enhancement**: Structured thinking for project pattern detection
- **Session Analysis**: Apply systematic thinking to coding session analysis
- **Risk Assessment**: Identify and mitigate risks in coding decisions

### **🔧 Proactive Memory Consultation**
- **Automatic Memory Suggestions**: Claude proactively references relevant memories before actions
- **Context-Aware Queries**: Smart memory retrieval based on current files, commands, and tasks
- **Zero-Friction Integration**: Seamless memory consultation without interrupting workflow

### **🔍 Real MCP Server Discovery**
- **Automatic MCP Detection**: Discovers actual installed MCP servers from Claude Desktop config
- **Live Tool Discovery**: Connects to running servers to get real tool definitions and schemas
- **Smart Tool Inference**: Infers tools when live discovery fails, based on server patterns
- **Proactive Tool Suggestions**: Claude suggests MCP tools before using scripts or indirect methods
- **Multi-Source Discovery**: Configuration files, environment variables, live servers, and known tools

## Features

### **🔥 Core Memory System (Enhanced)**
- **High-Performance Vector Storage**: Qdrant-based for enterprise scalability
- **Proactive Memory Retrieval**: Automatic consultation before tool execution
- **Advanced Similarity Search**: Vector embeddings with semantic understanding
- **Smart Context Analysis**: Extracts keywords from files, commands, and conversations
- **Real-Time Updates**: Atomic operations with no file rewriting
- **Memory Access Tracking**: Usage patterns and optimization recommendations

### **🔍 MCP Awareness System (New)**
- **Real MCP Server Discovery**: Automatically discovers installed MCP servers from configuration
- **Live Tool Indexing**: Connects to running servers to catalog available tools with real schemas
- **Proactive Tool Suggestions**: Claude suggests MCP tools before indirect approaches
- **Smart Pattern Recognition**: Infers tools from server names when live discovery fails
- **Multi-Source Discovery**: Configuration files, environment variables, live servers, and fallbacks
- **Context-Aware Recommendations**: Tool suggestions based on current files, tasks, and errors

### **AutoCodeIndex Intelligence System**
- **Intelligent Command Suggestions**: Context-aware command recommendations with confidence scoring
- **Project Pattern Recognition**: Automatic detection of frameworks, architectures, and coding patterns
- **Session History & Context**: Advanced conversation analysis and intelligent session navigation
- **Learning Progression Tracking**: Monitor skill development and knowledge growth over time
- **Workflow Optimization**: Historical pattern-based suggestions for improving development workflows
- **Automatic File & Command Tracking**: Zero-friction learning from Claude's interactions

### **🎯 Advanced Capabilities**
- **26 MCP Tools**: Comprehensive toolset including structured thinking, proactive memory, and performance monitoring
- **12 Memory Types**: Including 8 new structured thinking types (structured_thinking, thought_process, thinking_relationship, problem_analysis, research_notes, analysis_result, solution_synthesis, conclusion_summary)
- **Structured Thinking Tools**: `process_structured_thought`, `generate_thinking_summary`, `continue_thinking_process`, `analyze_thought_relationships`
- **Enhanced AutoCode Tools**: `suggest_command` with structured thinking analysis, enhanced `get_project_patterns`
- **Proactive Memory Tools**: `suggest_memory_queries`, `check_relevant_memories`, `configure_proactive_memory`, `get_proactive_memory_stats`
- **Performance Tools**: `qdrant_performance_stats`, `optimize_qdrant_collection`
- **MCP Awareness Tools**: Real-time MCP server discovery and proactive tool suggestions
- **Automatic Hook System**: Seamless integration with Claude's normal operations
- **Multi-Language Support**: TypeScript, JavaScript, Python, Rust, Go, Java, and more
- **Framework Detection**: React, Vue, Angular, Django, Flask, FastAPI, and others
- **Platform Intelligence**: macOS, Linux, and Windows-specific optimizations

### **⚡ Performance & Monitoring**
- **Real-Time Performance Stats**: Memory usage, search times, indexing ratios
- **Automatic Optimization**: Collection optimization with usage-based recommendations
- **Scalability Metrics**: Handle 100K+ memories with sub-millisecond search
- **Memory Analytics**: Type distribution, access patterns, tier management

### **Integration & Compatibility**
- **Claude Integration**: Ready-to-use integration with Claude desktop application
- **MCP Protocol Support**: Fully compatible with the Model Context Protocol
- **Docker Support**: Easy deployment using Docker containers
- **Configuration-Driven**: Extensive customization options with sensible defaults

## Quick Start

### 🚀 High-Performance Memory + Structured Thinking (v1.0.0)

**New users:** Get instant high-performance memory with Qdrant vector database plus revolutionary structured thinking capabilities.
**Existing users:** [Migrate from JSON](#-migration-from-json-storage) for 10-100x performance boost and structured thinking integration.

#### **1. Docker Setup (Recommended) - Single Volume Mount**

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

**Storage Structure:**
- `./.claude/alunai-clarity/config.json` → Configuration file
- `./.claude/alunai-clarity/qdrant/` → **Shared** vector database storage
- `./.claude/alunai-clarity/cache/` → Embedding model cache
- `./.claude/alunai-clarity/backups/` → Backup files

> **Note:** All directories will be created automatically when you first run the container.

#### **Multiple Sessions (Concurrent Claude Instances)**

**✅ Now Supported!** You can run multiple Claude instances in the same project **sharing the same memory database** without conflicts.

All Claude instances will:
- Share the same memories and context
- Access the same vector database safely
- Collaborate with shared knowledge
- No setup required - works automatically

Simply run Claude in multiple terminals - they'll all share memories seamlessly.

#### **2. Enable Proactive Features with `CLAUDE.md`**
```markdown
You have high-performance persistent memory with proactive consultation and MCP awareness.

🔧 **MCP TOOLS AVAILABLE** - Use these FIRST before scripts or indirect methods:
- Your system will automatically discover MCP tools from your configuration
- Use postgres MCP tools instead of psql scripts
- Use playwright MCP tools instead of manual browsing
- Use memory MCP tools proactively for context

Before taking actions, automatically check relevant memories using:
- suggest_memory_queries: Get recommended memory searches
- check_relevant_memories: Auto-retrieve contextual memories

Automatically store: user preferences, project architecture,
command patterns, and solutions with vector search capabilities.
```

#### **3. Performance Monitoring**
```markdown
Monitor your memory system performance:
- qdrant_performance_stats: View detailed performance metrics
- optimize_qdrant_collection: Optimize for better performance
```

### 📈 **Performance Comparison**

| **Memory Count** | **JSON Search** | **Qdrant Search** | **Improvement** |
|------------------|-----------------|-------------------|-----------------|
| 1K memories | ~10ms | ~0.1ms | **100x faster** |
| 10K memories | ~100ms | ~1ms | **100x faster** |
| 100K memories | ~1000ms+ | ~2ms | **500x faster** |

### Option 1: Using Docker (Full Configuration)

Use the pre-built Docker image from GitHub Container Registry:

```json
{
  "mcpServers": {
    "alunai-clarity": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--user", "$(id -u):$(id -g)",
        "-v",
        "./.claude/alunai-clarity:/app/data",
        "-e",
        "MEMORY_FILE_PATH",
        "-e",
        "AUTOCODE_ENABLED",
        "-e",
        "AUTOCODE_COMMAND_LEARNING_ENABLED",
        "-e",
        "AUTOCODE_PATTERN_DETECTION_ENABLED",
        "-e",
        "AUTOCODE_SESSION_ANALYSIS_ENABLED",
        "-e",
        "AUTOCODE_HISTORY_NAVIGATION_ENABLED",
        "-e",
        "AUTOCODE_AUTO_SCAN_PROJECTS",
        "-e",
        "AUTOCODE_TRACK_BASH_COMMANDS",
        "-e",
        "AUTOCODE_GENERATE_SESSION_SUMMARIES",
        "-e",
        "AUTOCODE_MIN_CONFIDENCE_THRESHOLD",
        "-e",
        "AUTOCODE_SIMILARITY_THRESHOLD",
        "-e",
        "AUTOCODE_STRUCTURED_THINKING_ENABLED",
        "ghcr.io/alun-ai/mcp-alunai-clarity:latest"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/data/memory.json",
        "AUTOCODE_ENABLED": "true",
        "AUTOCODE_COMMAND_LEARNING_ENABLED": "true",
        "AUTOCODE_PATTERN_DETECTION_ENABLED": "true",
        "AUTOCODE_SESSION_ANALYSIS_ENABLED": "true",
        "AUTOCODE_HISTORY_NAVIGATION_ENABLED": "true",
        "AUTOCODE_AUTO_SCAN_PROJECTS": "true",
        "AUTOCODE_TRACK_BASH_COMMANDS": "true",
        "AUTOCODE_GENERATE_SESSION_SUMMARIES": "true",
        "AUTOCODE_MIN_CONFIDENCE_THRESHOLD": "0.2",
        "AUTOCODE_SIMILARITY_THRESHOLD": "0.5",
        "AUTOCODE_STRUCTURED_THINKING_ENABLED": "true"
      },
      "type": "stdio"
    }
  }
}
```

## Using Memory with Claude

MCP AlunAI Memory enables Claude to remember information across conversations without requiring explicit commands, while AutoCodeIndex provides intelligent code assistance.

### 🧠 **Automatic Memory Features**
Claude will automatically:
- Remember important details you share about projects and preferences
- Store user preferences, facts, and coding patterns
- Recall relevant information when needed
- Track file modifications and command usage
- Learn from successful workflows and approaches

### 🚀 **AutoCodeIndex Intelligence**
Claude gains powerful code intelligence capabilities:
- **Smart Command Suggestions**: Get context-aware command recommendations
- **Project Pattern Recognition**: Automatic detection of frameworks and architectures
- **Session Continuity**: Resume work with relevant context from previous sessions
- **Learning Tracking**: Monitor your progress on specific technologies and topics
- **Workflow Optimization**: Receive suggestions based on successful past patterns

### 💬 **Memory Recall**
To see what Claude remembers, simply ask:
- "What do you remember about me?"
- "What projects have we worked on?"
- "Show me my learning progression in React"
- "What command patterns work best for this project?"

### ✅ **Verifying AutoCodeIndex is Active**
Once enabled, you'll immediately notice:
- **Command suggestions with confidence scores**: "Based on your project, I suggest `npm test` (95% success rate)"
- **Automatic project analysis**: "I've detected a React + TypeScript project with Jest testing"
- **Session continuity**: "Continuing from our work 3 days ago on the authentication system..."
- **Learning insights**: "You've progressed from basic React to advanced patterns like custom hooks"

**Quick verification**: Ask "Show me AutoCodeIndex stats" to see system status and learning data.

### 🛠 **Available Tools**

#### **Core Memory Tools**
1. **store_memory** - Store memories with automatic vector indexing
2. **retrieve_memory** - High-performance vector similarity search
3. **list_memories** - Browse memories with advanced filtering
4. **update_memory** - Update existing memories with re-indexing
5. **delete_memory** - Remove memories from vector database
6. **memory_stats** - Get comprehensive memory statistics

#### **🧠 Proactive Memory Consultation (New)**
7. **suggest_memory_queries** - Get recommended memory searches based on current context
8. **check_relevant_memories** - Automatically retrieve contextually relevant memories
9. **configure_proactive_memory** - Configure automatic memory presentation behavior
10. **get_proactive_memory_stats** - Monitor proactive memory usage and effectiveness

#### **⚡ Performance & Optimization (New)**
9. **qdrant_performance_stats** - Detailed performance metrics and recommendations
10. **optimize_qdrant_collection** - Optimize vector database for better performance

#### **🔍 MCP Awareness Tools (New)**
11. **discover_mcp_tools** - Automatically discover installed MCP servers and tools
12. **suggest_mcp_alternatives** - Suggest MCP tools instead of indirect methods
13. **get_mcp_tool_info** - Get detailed information about available MCP tools

#### **🚀 AutoCodeIndex Intelligence**
14. **suggest_command** - Get smart command suggestions based on intent
15. **get_project_patterns** - Analyze and retrieve project patterns
16. **find_similar_sessions** - Find related past work sessions
17. **get_continuation_context** - Get context for continuing tasks
18. **suggest_workflow_optimizations** - Improve workflows based on history
19. **get_learning_progression** - Track learning progress on topics
20. **autocode_stats** - View AutoCodeIndex system statistics

### 📝 **System Prompt**
For optimal functionality with proactive memory consultation and MCP awareness, add this to your Claude system prompt:

```
This Claude instance has been enhanced with high-performance persistent memory,
proactive memory consultation, MCP awareness, and AutoCodeIndex capabilities.

🔧 **MCP TOOLS AVAILABLE** - Use these FIRST before scripts or indirect methods:
- Your system automatically discovers MCP tools from your configuration
- ALWAYS prefer MCP tools over scripts (postgres tools vs psql, playwright vs manual browsing)
- Check for MCP alternatives before using shell commands or writing code
- Use memory tools proactively to enhance responses with relevant context

PROACTIVE MEMORY USAGE:
- Before taking actions, automatically use suggest_memory_queries and check_relevant_memories
- Proactively surface relevant context from previous conversations and work
- Reference related memories when providing suggestions or solutions

AUTOMATIC CAPABILITIES:
- Remember important details across conversations with vector search
- Learn from coding patterns and command usage with performance tracking
- Discover and suggest MCP tools dynamically based on installed servers
- Provide intelligent assistance based on project context and historical data
- Monitor performance with qdrant_performance_stats and optimize when needed
```

See the [User Guide](docs/user_guide.md) for detailed usage instructions and examples.

## AutoCodeIndex Technical Details

### 🏗️ **System Architecture**

AutoCodeIndex consists of 5 integrated phases:

1. **Foundation & File Access Tracking**: Automatic monitoring of file operations and project context
2. **Command Intelligence System**: Learning from command execution patterns and providing smart suggestions
3. **Project Pattern Recognition**: Detecting frameworks, architectures, and coding patterns across languages
4. **Session History & Context**: Advanced conversation analysis and intelligent session navigation
5. **Integration & MCP Hooks**: Seamless integration with Claude's normal operations

### 🔧 **Memory Types**

AutoCodeIndex introduces 4 specialized memory types:

- **`project_pattern`**: Stores detected project patterns, frameworks, and architectural decisions
- **`command_pattern`**: Tracks command usage, success rates, and optimization opportunities
- **`session_summary`**: Rich analysis of conversation sessions with task tracking and insights
- **`bash_execution`**: Command execution history with context and learning data

### 🚀 **Key Capabilities**

**Intelligent Command Assistance:**
- Context-aware suggestions based on project type and platform
- Success rate tracking with confidence scoring
- Retry pattern detection and automatic improvements
- Platform-specific optimizations (macOS, Linux, Windows)

**Project Pattern Intelligence:**
- Automatic framework detection (React, Vue, Angular, Django, Flask, FastAPI, Rust frameworks)
- Architectural pattern recognition (MVC, component-based, layered, microservices)
- Naming convention analysis and consistency checking
- Dependency analysis and technology stack evolution

**Session Continuity:**
- Comprehensive conversation analysis with task extraction
- Semantic similarity search across historical sessions
- Context continuation for seamless task resumption
- Learning progression tracking across multiple sessions

**Workflow Optimization:**
- Historical pattern-based workflow suggestions
- Bottleneck identification and mitigation recommendations
- Efficiency pattern recognition and reuse
- Cross-session learning and continuous improvement

### ⚙️ **Configuration Options**

AutoCodeIndex provides extensive configuration options:

```json
{
  "autocode": {
    "enabled": true,
    "command_learning": {
      "enabled": true,
      "min_confidence_threshold": 0.3,
      "max_suggestions": 5
    },
    "pattern_detection": {
      "enabled": true,
      "supported_languages": ["typescript", "javascript", "python", "rust", "go", "java"],
      "max_scan_depth": 5
    },
    "session_analysis": {
      "enabled": true,
      "track_architectural_decisions": true,
      "extract_learning_patterns": true
    },
    "history_navigation": {
      "enabled": true,
      "similarity_threshold": 0.6,
      "context_window_days": 30
    }
  }
}
```

### 📊 **Implementation Statistics**

- **15,000+ lines of code** across 20+ specialized modules
- **20+ MCP tools** including high-performance vector search, proactive consultation, and MCP awareness
- **4 specialized memory types** for comprehensive pattern storage
- **Qdrant vector database** with sub-millisecond search capabilities
- **Real MCP server discovery system** with live tool indexing and smart inference
- **Proactive memory consultation system** with automatic context awareness
- **MCP awareness hooks** for intelligent tool suggestions and error resolution
- **Performance monitoring** with optimization recommendations
- **Automatic hook system** for zero-friction operation
- **Multi-language support** for major programming languages
- **Platform intelligence** for macOS, Linux, and Windows
- **10-100x performance improvement** over previous JSON storage

## 📚 Documentation

### **🚀 Getting Started**
- [Quick Start Guide](docs/quick_start.md) - *Get high-performance memory working in 5 minutes*
- [Migration Guide](QDRANT_MIGRATION.md) - **Migrate from JSON to Qdrant for 10-100x performance boost**
- [Proactive AutoCode Guide](docs/proactive_autocode.md) - *Enable intelligent proactive suggestions*

### **📖 User Guides**
- [User Guide](docs/user_guide.md) - *Complete usage instructions*
- [AutoCodeIndex Guide](docs/autocode_guide.md) - *Comprehensive AutoCodeIndex documentation*
- [Performance Optimization](docs/performance.md) - *Optimize your memory system*

### **🔧 Technical Documentation**
- [Docker Usage Guide](docs/docker_usage.md) - *Container deployment*
- [Compatibility Guide](docs/compatibility.md) - *System requirements*
- [Architecture](docs/architecture.md) - *Technical architecture*
- [Claude Integration Guide](docs/claude_integration.md) - *Claude Desktop integration*

### **🎯 New in v0.4.0**
- **[Real MCP Server Discovery](docs/mcp_awareness.md)** - Automatic MCP tool discovery and indexing
- **[MCP Awareness System](docs/mcp_awareness.md)** - Proactive tool suggestions and smart alternatives
- **[Qdrant Vector Database](QDRANT_MIGRATION.md)** - High-performance storage backend
- **[Proactive Memory Consultation](docs/proactive_memory.md)** - Automatic memory referencing
- **[Performance Monitoring](docs/performance_monitoring.md)** - Stats and optimization tools

## Examples

The `examples` directory contains scripts demonstrating how to interact with MCP AlunAI Memory:

- `store_memory_example.py`: Example of storing a memory
- `retrieve_memory_example.py`: Example of retrieving memories
- `mcp_awareness_example.py`: Example of MCP server discovery and tool suggestions

### **🔍 MCP Awareness Examples**

**Automatic Tool Discovery:**
```python
# Your Claude Desktop config.json has:
{
  "mcpServers": {
    "postgres": {"command": "npx", "args": ["@postgresql/mcp-server"]},
    "playwright": {"command": "npx", "args": ["@playwright/mcp-server"]}
  }
}

# Claude automatically discovers and suggests:
💡 **MCP Tools Available:**
- postgres_query: Execute SQL queries directly
- playwright_navigate: Automate web browser interactions
- alunai-memory_store: Store persistent memories
```

**Proactive Suggestions:**
```markdown
User: "I need to query the database for user information"

Claude: 💡 **Before writing SQL scripts, consider these MCP tools:**
- **postgres_query**: Execute SQL queries against PostgreSQL databases directly
- **postgres_schema**: Get database schema information
- Why relevant: Matches keywords: database, query
- MCP tools are often more reliable than shell commands!
```

**Context-Aware Recommendations:**
```markdown
# When working with .sql files:
Claude: 💡 **Context-Aware MCP Tools**: Based on your current context:
- **postgres_query**: Execute SQL queries against PostgreSQL databases

# When errors occur:
Claude: 🔧 **Error Resolution**: These MCP tools might help resolve the error:
- **postgres_query**: Execute SQL queries against PostgreSQL databases
```

## 🔄 **Migration from JSON Storage**

**⚠️ EXISTING USERS: Migrate for 10-100x Performance Boost**

If you're upgrading from a previous version with JSON storage, follow these steps:

### **Quick Migration**

#### **Option A: Docker Migration (Recommended)**

```bash
# 1. Pull latest Docker image
docker pull ghcr.io/alun-ai/mcp-alunai-clarity:v1.0.0

# 2. Find your JSON file (common locations)
# ~/.memory_mcp/data/memory.json
# ./.claude/alunai-memory/memory.json
# ./memory.json

# 3. Run migration using CLI in Docker container
docker run --entrypoint="python" \
           --user $(id -u):$(id -g) \
           -v /path/to/your/memory.json:/tmp/memory.json \
           -v ./.claude/alunai-clarity:/app/data \
           ghcr.io/alun-ai/mcp-alunai-clarity:latest \
           -m clarity.cli.import_json /tmp/memory.json
```

### **Migration Results**
```
✅ Import verification PASSED
   - Total memories in Qdrant: 1,247
   - Indexed memories: 1,247
   - Search functionality: Working
   - Performance improvement: 100x faster search
```
### **Migration Support**
- **Automatic verification** ensures 100% data integrity
- **Batch processing** handles large datasets efficiently
- **Progress tracking** shows real-time migration status
- **Error recovery** with detailed logging and retry options
- **Rollback safety** - original JSON files remain untouched

## Troubleshooting

### **🚀 Performance Issues**
1. **Slow search performance**: Run `optimize_qdrant_collection` tool
2. **High memory usage**: Check `qdrant_performance_stats` for optimization recommendations
3. **Large dataset imports**: Use larger `--batch-size` (500-1000) for JSON migration

### **🔄 Migration Issues**
1. **"No memories found"**: Verify JSON file structure with `--dry-run` flag
2. **Import failures**: Use smaller `--batch-size` (50-100) and check logs
3. **Permission errors**: Ensure write access to Qdrant data directory

### **⚙️ General Issues**
1. Check the [Compatibility Guide](docs/compatibility.md) for dependency requirements
2. Ensure your Python version is 3.8-3.12
3. For dependency conflicts: `pip install qdrant-client>=1.7.0`
4. Try using Docker for simplified deployment
5. **Migration help**: See comprehensive [Migration Guide](QDRANT_MIGRATION.md)

## Documentation

### Core Documentation
- **[Structured Thinking Guide](docs/structured_thinking.md)**: Complete guide to 5-stage structured thinking integration
- **[MCP Tools Reference](docs/mcp_tools_reference.md)**: All 26 MCP tools with parameters and examples
- **[AutoCode Enhancements](docs/autocode_enhancements.md)**: Enhanced intelligence capabilities
- **[Usage Examples](docs/usage_examples.md)**: Practical examples and API reference
- **[User Guide](docs/user_guide.md)**: Getting started and configuration
- **[Proactive Memory](docs/proactive_memory.md)**: Automatic memory consultation
- **[MCP Awareness](docs/mcp_awareness.md)**: Real-time MCP server discovery

### Advanced Topics
- **[Architecture](docs/architecture.md)**: System design and components
- **[AutoCode Guide](docs/autocode_guide.md)**: Code intelligence features
- **[Docker Usage](docs/docker_usage.md)**: Container deployment
- **[Claude Integration](docs/claude_integration.md)**: Claude Desktop setup

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.