# MCP Alun.ai Memory with AutoCodeIndex

An advanced MCP (Model Context Protocol) server implementation that provides advanced persistent memory capabilities and intelligent code project assistance for Large Language Models, specifically designed to integrate with the Claude desktop application.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project combines optimal memory techniques with intelligent code project analysis to create the **AutoCodeIndex** system - a comprehensive solution that helps Claude maintain context about coding patterns, project structures, command usage, and session history across conversations.

## Features

### **Core Memory System**
- **Tiered Memory Architecture**: Short-term, long-term, and archival memory tiers
- **Multiple Memory Types**: Support for conversations, knowledge, entities, reflections, and code patterns
- **Semantic Search**: Retrieve memories based on semantic similarity with vector embeddings
- **Automatic Memory Management**: Intelligent memory capture without explicit commands
- **Memory Consolidation**: Automatic consolidation of short-term memories into long-term memory
- **Memory Management**: Importance-based memory retention and forgetting

### **AutoCodeIndex Intelligence System**
- **Intelligent Command Suggestions**: Context-aware command recommendations with confidence scoring
- **Project Pattern Recognition**: Automatic detection of frameworks, architectures, and coding patterns
- **Session History & Context**: Advanced conversation analysis and intelligent session navigation
- **Learning Progression Tracking**: Monitor skill development and knowledge growth over time
- **Workflow Optimization**: Historical pattern-based suggestions for improving development workflows
- **Automatic File & Command Tracking**: Zero-friction learning from Claude's interactions

### **Advanced Capabilities**
- **7 New MCP Tools**: Comprehensive toolset for intelligent code assistance
- **4 Specialized Memory Types**: project_pattern, command_pattern, session_summary, bash_execution
- **Automatic Hook System**: Seamless integration with Claude's normal operations
- **Multi-Language Support**: TypeScript, JavaScript, Python, Rust, Go, Java, and more
- **Framework Detection**: React, Vue, Angular, Django, Flask, FastAPI, and others
- **Platform Intelligence**: macOS, Linux, and Windows-specific optimizations

### **Integration & Compatibility**
- **Claude Integration**: Ready-to-use integration with Claude desktop application
- **MCP Protocol Support**: Fully compatible with the Model Context Protocol
- **Docker Support**: Easy deployment using Docker containers
- **Configuration-Driven**: Extensive customization options with sensible defaults

## Quick Start

### Option 1: Using Docker (Recommended)

Use the pre-built Docker image from GitHub Container Registry:

```json
{
  "mcpServers": {
    "memory": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
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
        "ghcr.io/alun-ai/mcp-alunai-memory:latest"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/tmp/memory.json",
        "AUTOCODE_ENABLED": "true",
        "AUTOCODE_COMMAND_LEARNING_ENABLED": "true",
        "AUTOCODE_PATTERN_DETECTION_ENABLED": "true",
        "AUTOCODE_SESSION_ANALYSIS_ENABLED": "true",
        "AUTOCODE_HISTORY_NAVIGATION_ENABLED": "true"
      }
    }
  }
}
```

Alternatively, to build locally:

```bash
# Clone the repository
git clone https://github.com/alun-ai/mcp-alunai-memory.git
cd mcp-alunai-memory

# Build the Docker image
docker build -t mcp-alunai-memory .
```

### Option 2: Standard Installation

1. **Prerequisites**:
   - Python 3.8-3.12
   - pip package manager

2. **Installation**:
   ```bash
   # Clone the repository
   git clone https://github.com/alun-ai/mcp-alunai-memory.git
   cd mcp-alunai-memory

   # Install dependencies
   pip install -r requirements.txt

   # Run setup script
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Claude Desktop Integration**:

   Add the following to your Claude Desktop MCP configuration:

   ```json
   {
     "mcpServers": {
       "memory": {
         "command": "python",
         "args": ["-m", "memory_mcp"],
         "env": {
           "MEMORY_FILE_PATH": "./memory.json"
         }
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
AutoCodeIndex provides 7 intelligent tools:
1. **suggest_command** - Get smart command suggestions based on intent
2. **get_project_patterns** - Analyze and retrieve project patterns
3. **find_similar_sessions** - Find related past work sessions
4. **get_continuation_context** - Get context for continuing tasks
5. **suggest_workflow_optimizations** - Improve workflows based on history
6. **get_learning_progression** - Track learning progress on topics
7. **autocode_stats** - View AutoCodeIndex system statistics

### 📝 **System Prompt**
For optimal functionality, add this to your Claude system prompt:

```
This Claude instance has been enhanced with persistent memory and AutoCodeIndex
capabilities. Claude will automatically remember important details across
conversations, learn from coding patterns and command usage, and provide
intelligent assistance based on project context and historical data.
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

- **7,386 lines of code** across 10 specialized modules
- **7 new MCP tools** for intelligent code assistance
- **4 new memory types** for comprehensive pattern storage
- **Automatic hook system** for zero-friction operation
- **Multi-language support** for major programming languages
- **Platform intelligence** for macOS, Linux, and Windows

## Documentation

- [User Guide](docs/user_guide.md)
- [AutoCodeIndex Guide](docs/autocode_guide.md) - *Comprehensive AutoCodeIndex documentation*
- [Docker Usage Guide](docs/docker_usage.md)
- [Compatibility Guide](docs/compatibility.md)
- [Architecture](docs/architecture.md)
- [Claude Integration Guide](docs/claude_integration.md)

## Examples

The `examples` directory contains scripts demonstrating how to interact with MCP AlunAI Memory:

- `store_memory_example.py`: Example of storing a memory
- `retrieve_memory_example.py`: Example of retrieving memories

## Troubleshooting

If you encounter issues:

1. Check the [Compatibility Guide](docs/compatibility.md) for dependency requirements
2. Ensure your Python version is 3.8-3.12
3. For NumPy issues, use: `pip install "numpy>=1.20.0,<2.0.0"`
4. Try using Docker for simplified deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.