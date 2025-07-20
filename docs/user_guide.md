# User Guide: MCP AlunAI Memory with AutoCodeIndex

This guide explains how to set up and use the MCP AlunAI Memory server with Claude Desktop for persistent memory capabilities and intelligent code project assistance through the AutoCodeIndex system.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [How Memory Works](#how-memory-works)
4. [AutoCodeIndex Features](#autocodeix-features)
5. [Usage Examples](#usage-examples)
6. [AutoCodeIndex Tools](#autocodeix-tools)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Standard Installation

1. **Prerequisites**:
   - Python 3.8-3.12
   - pip package manager

2. **Clone the repository**:
   ```bash
   git clone https://github.com/alun-ai/claude-memory-mcp.git
   cd claude-memory-mcp
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### Option 2: Docker Installation (Recommended)

Use the pre-built Docker image with AutoCodeIndex features:

```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "./.claude/alunai-memory:/data",
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
        "ghcr.io/alun-ai/mcp-alunai-memory:latest"
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
        "AUTOCODE_SIMILARITY_THRESHOLD": "0.5"
      }
    }
  }
}
```

See the [Docker Usage Guide](docker_usage.md) for detailed instructions on running the server in a container.

## Configuration

### Claude Desktop Integration

To integrate with Claude Desktop, add the Memory MCP Server to your Claude configuration file:

**Location**:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Configuration**:
```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "python",
      "args": ["-m", "memory_mcp"],
      "env": {
        "MEMORY_FILE_PATH": "./.claude/alunai-memory/memory.json",
        "AUTOCODE_ENABLED": "true",
        "AUTOCODE_COMMAND_LEARNING_ENABLED": "true",
        "AUTOCODE_PATTERN_DETECTION_ENABLED": "true",
        "AUTOCODE_SESSION_ANALYSIS_ENABLED": "true",
        "AUTOCODE_HISTORY_NAVIGATION_ENABLED": "true",
        "AUTOCODE_AUTO_SCAN_PROJECTS": "true",
        "AUTOCODE_TRACK_BASH_COMMANDS": "true",
        "AUTOCODE_GENERATE_SESSION_SUMMARIES": "true",
        "AUTOCODE_MIN_CONFIDENCE_THRESHOLD": "0.2",
        "AUTOCODE_SIMILARITY_THRESHOLD": "0.5"
      }
    }
  }
}
```

### Automatic Memory Setup

#### Option A: Project-Specific Memory (Recommended)

Create a `CLAUDE.md` file in your project root:

```markdown
# Project Memory Configuration

You have persistent memory capabilities via the alunai-memory MCP server.

## Automatic Memory Instructions

Automatically store important information about:
- User preferences and coding patterns
- Project architecture and decisions  
- Successful workflows and command patterns
- File modifications and patterns
- Errors and their solutions

Store memories without being explicitly asked when you encounter:
- New user preferences
- Important project details
- Successful command patterns
- Learning moments or insights

## Memory Storage Guidelines

- Use descriptive memory types: `user_preference`, `project_architecture`, `command_pattern`, `error_solution`
- Set appropriate importance levels (0.1-1.0)
- Include relevant context and metadata
- Prefer automatic storage over explicit "remember" requests
```

#### Option B: Global System Prompt

For optimal memory and AutoCodeIndex usage, add these instructions to your Claude Desktop System Prompt:

```
You have persistent memory capabilities via the alunai-memory MCP server. Automatically store important information about user preferences, project architecture, successful workflows, command patterns, file modifications, and error solutions. Store memories without being explicitly asked when you encounter new user preferences, important project details, successful command patterns, or learning moments.

This Claude instance has been enhanced with persistent memory and AutoCodeIndex 
capabilities. Claude will automatically:
1. Remember important details about you across conversations
2. Store key facts and preferences you share
3. Recall relevant information when needed
4. Learn from coding patterns and command usage
5. Provide intelligent code assistance based on project context
6. Track your learning progression on technologies and topics
7. Suggest workflow optimizations based on historical patterns

You don't need to explicitly ask Claude to remember information or provide
code intelligence. Simply have natural conversations about coding projects,
and Claude will maintain memory of important details while providing smart
assistance based on your development patterns.

To explore these capabilities, try asking:
- "What do you remember about me?"
- "What projects have we worked on?"
- "Show me my learning progression in React"
- "Suggest commands for testing this project"
```

## How Memory Works

### Memory Types

The Memory MCP Server supports several types of memories:

1. **Entity Memories**: Information about people, places, things
   - User preferences and traits
   - Personal information

2. **Fact Memories**: Factual information
   - General knowledge
   - Specific facts shared by the user

3. **Conversation Memories**: Important parts of conversations
   - Significant exchanges
   - Key discussion points

4. **Reflection Memories**: Insights and patterns
   - Observations about the user
   - Recurring themes

5. **Project Pattern Memories**: AutoCodeIndex project analysis
   - Framework and architecture detection
   - Coding patterns and conventions
   - Technology stack information

6. **Command Pattern Memories**: AutoCodeIndex command intelligence
   - Command usage patterns and success rates
   - Platform-specific optimizations
   - Context-aware suggestions

7. **Session Summary Memories**: AutoCodeIndex session analysis
   - Task completion tracking
   - Learning moments and insights
   - Architectural decisions

8. **Bash Execution Memories**: AutoCodeIndex command tracking
   - Command execution history
   - Success/failure patterns
   - Performance metrics

### Memory Tiers

Memories are stored in three tiers:

1. **Short-term Memory**: Recently created or accessed memories
   - Higher importance (>0.3 by default)
   - Frequently accessed

2. **Long-term Memory**: Older, less frequently accessed memories
   - Lower importance (<0.3 by default)
   - Less frequently accessed

3. **Archived Memory**: Rarely accessed but potentially valuable memories
   - Used for long-term storage
   - Still searchable but less likely to be retrieved

## How to Verify AutoCodeIndex is Working

Once enabled, you should see several clear indicators that AutoCodeIndex is active and learning:

### 🔍 **Immediate Verification**

**Check system status:**
```
User: "Show me AutoCodeIndex stats"
Claude: [Uses autocode_stats tool and shows:]
- AutoCodeIndex Status: ✅ Active
- Total Projects Analyzed: 3
- Command Patterns Learned: 127
- Session Summaries Generated: 15
- Memory Types Active: 8 (including 4 AutoCodeIndex types)
```

**Test command suggestions:**
```
User: "How do I run tests in this project?"
Claude: "Based on your package.json and past usage patterns, I suggest: `npm test`
This command has a 95% success rate in similar React projects."
```

### 🚀 **Visible Effects During Usage**

**1. Contextual Command Intelligence:**
- Claude provides specific commands with confidence percentages
- Platform-specific suggestions (different commands for macOS vs Linux)
- Historical success rate information

**2. Project Pattern Recognition:**
- Automatic framework detection without being told
- Architecture analysis and naming convention insights
- Technology stack identification

**3. Session Continuity:**
- Claude references previous work sessions automatically
- Provides context about past decisions and approaches
- Tracks learning progression over time

**4. Workflow Optimization:**
- Proactive suggestions for command aliases and shortcuts
- Identification of repetitive patterns
- Performance improvement recommendations

### 📊 **What AutoCodeIndex Learns About You**

**Development Patterns:**
```
User: "What patterns have you learned about my coding style?"
Claude: "I've observed you prefer:
- TypeScript over JavaScript (used in 85% of projects)
- Jest for testing (consistent across 4 projects)
- Functional components in React (95% usage rate)
- camelCase naming (98% consistent)"
```

**Command Usage:**
```
User: "What commands do I use most often?"
Claude: "Your top commands:
1. `npm run dev` (used 47 times, 100% success rate)
2. `git status` (used 31 times)
3. `npm test` (used 23 times, 96% success rate)
Platform: macOS optimizations active"
```

### ⚡ **No AutoCodeIndex? Here's What You'd Miss**

Without AutoCodeIndex enabled, Claude would:
- ❌ Not provide command confidence scores or success rates
- ❌ Not automatically detect project frameworks or patterns
- ❌ Not reference previous session context automatically
- ❌ Not track learning progression or suggest optimizations
- ❌ Not provide platform-specific command variations

## AutoCodeIndex Features

AutoCodeIndex provides intelligent code project assistance through several key capabilities:

### 🧠 Intelligent Command Assistance
AutoCodeIndex learns from your command usage and provides smart suggestions:
- **Context-aware recommendations** based on project type and platform
- **Success rate tracking** with confidence scoring for reliability
- **Platform-specific optimizations** for macOS, Linux, and Windows
- **Retry pattern detection** with automatic improvement suggestions

### 🔍 Project Pattern Recognition
Automatic detection and analysis of project structures:
- **Framework detection**: React, Vue, Angular, Django, Flask, FastAPI, and more
- **Architecture analysis**: MVC, component-based, microservices patterns
- **Naming convention analysis** with consistency checking
- **Dependency mapping** and technology stack evolution tracking

### 📈 Session History & Context
Advanced conversation analysis for seamless workflow continuity:
- **Task tracking**: Automatically identifies completed, in-progress, and failed tasks
- **Learning progression**: Monitors skill development over time
- **Context continuation**: Resume work with relevant background from previous sessions
- **Architectural decisions**: Records design choices and reasoning

### ⚡ Workflow Optimization
Continuous improvement based on historical patterns:
- **Efficiency analysis**: Identifies bottlenecks and optimization opportunities
- **Pattern reuse**: Suggests approaches that worked well in similar situations
- **Cross-session learning**: Builds knowledge across multiple development sessions
- **Performance tracking**: Monitors and improves development velocity

## Usage Examples

### Scenario 1: Remembering User Preferences

**User**: "I really prefer to code in Python rather than JavaScript."

*Claude will automatically store this preference without any explicit command. In future conversations, Claude will remember this preference and tailor responses accordingly.*

**User**: "What programming language do I prefer?"

*Claude will automatically retrieve the memory:*

**Claude**: "You've mentioned that you prefer to code in Python rather than JavaScript."

### Scenario 2: Storing and Retrieving Personal Information

**User**: "My dog's name is Buddy, he's a golden retriever."

*Claude will automatically store this entity information.*

**User**: "What do you remember about my pet?"

**Claude**: "You mentioned that you have a golden retriever named Buddy."

### Scenario 3: Explicit Memory Operations (if needed)

While automatic memory is enabled by default, you can still use explicit commands:

**User**: "Please remember that my favorite color is blue."

**Claude**: "I'll remember that your favorite color is blue."

**User**: "What's my favorite color?"

**Claude**: "Your favorite color is blue."

### Scenario 4: AutoCodeIndex Command Intelligence

**User**: "I need to run tests for this React project."

*AutoCodeIndex analyzes the project structure and suggests the optimal command:*

**Claude**: "Based on your project's package.json and past usage patterns, I suggest: `npm test`. This command has a 98% success rate in similar React projects and will run your Jest test suite."

### Scenario 5: Project Pattern Recognition

**User**: "Can you analyze this project structure?"

*AutoCodeIndex automatically detects patterns:*

**Claude**: "I've analyzed your project and detected:
- React + TypeScript frontend with Vite build system
- Component-based architecture with 95% consistent camelCase naming
- Material-UI design system integration
- Jest + React Testing Library testing setup
- ESLint + Prettier code quality tools

Recommendation: Consider adding Storybook for component documentation based on your project's complexity."

### Scenario 6: Session Continuity

**User**: "Continue working on the authentication system."

*AutoCodeIndex provides relevant context from previous sessions:*

**Claude**: "Based on our session from 3 days ago, you were implementing JWT authentication with refresh tokens. I see you completed the login endpoint using bcrypt for password hashing and were working on the password reset flow with nodemailer. The email verification component was next on the list. Shall we continue with that implementation?"

## AutoCodeIndex Tools

AutoCodeIndex provides 7 intelligent tools for enhanced development assistance:

### 1. suggest_command
Get smart command suggestions based on your intent and project context.

**Example usage:**
```
User: "How do I start the development server?"
Claude uses suggest_command to analyze your project and suggests: `npm run dev`
```

### 2. get_project_patterns
Analyze and retrieve detected patterns for your current project.

**Example usage:**
```
User: "What patterns have you detected in this codebase?"
Claude uses get_project_patterns to show framework, architecture, and naming patterns.
```

### 3. find_similar_sessions
Find previous sessions similar to your current work for relevant context.

**Example usage:**
```
User: "Have we worked on user authentication before?"
Claude uses find_similar_sessions to locate related past conversations.
```

### 4. get_continuation_context
Get relevant context for continuing work on a specific task.

**Example usage:**
```
User: "Let's continue with the API implementation."
Claude uses get_continuation_context to provide relevant background.
```

### 5. suggest_workflow_optimizations
Receive suggestions for improving your development workflow based on usage patterns.

**Example usage:**
```
Claude suggests: "I notice you often run 'npm install && npm run dev'. 
Consider creating an alias: alias quickstart='npm install && npm run dev'"
```

### 6. get_learning_progression
Track your learning progress on specific technologies and topics.

**Example usage:**
```
User: "Show my React learning progression."
Claude uses get_learning_progression to display skill development over time.
```

### 7. autocode_stats
View AutoCodeIndex system statistics and health information.

**Example usage:**
```
User: "How much data has AutoCodeIndex collected?"
Claude uses autocode_stats to show memory usage and system metrics.
```

## Advanced Configuration

### Custom Configuration File

Create a custom configuration file at `~/.memory_mcp/config/config.json`:

```json
{
  "auto_memory": {
    "enabled": true,
    "threshold": 0.6,
    "store_assistant_messages": false,
    "entity_extraction_enabled": true
  },
  "alunai-memory": {
    "max_short_term_items": 200,
    "max_long_term_items": 2000,
    "consolidation_interval_hours": 48
  },
  "autocode": {
    "enabled": true,
    "auto_scan_projects": true,
    "track_bash_commands": true,
    "generate_session_summaries": true,
    "command_learning": {
      "enabled": true,
      "min_confidence_threshold": 0.3,
      "max_suggestions": 5
    },
    "pattern_detection": {
      "enabled": true,
      "supported_languages": ["typescript", "javascript", "python", "rust"],
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

### Auto-Memory Settings

- `enabled`: Enable/disable automatic memory (default: true)
- `threshold`: Minimum importance threshold for auto-storage (0.0-1.0)
- `store_assistant_messages`: Whether to store assistant messages (default: false)
- `entity_extraction_enabled`: Enable entity extraction from messages (default: true)

### AutoCodeIndex Settings

- `autocode.enabled`: Enable/disable AutoCodeIndex features (default: true)
- `auto_scan_projects`: Automatically scan project directories for patterns (default: true)
- `track_bash_commands`: Track command execution for learning (default: true)
- `command_learning.min_confidence_threshold`: Minimum confidence for command suggestions (0.0-1.0)
- `pattern_detection.max_scan_depth`: Maximum directory depth to scan for patterns
- `session_analysis.track_architectural_decisions`: Track design decisions in conversations
- `history_navigation.similarity_threshold`: Minimum similarity for session matching (0.0-1.0)

## Troubleshooting

### Memory Not Being Stored

1. **Check auto-memory settings**: Ensure auto_memory.enabled is true in config
2. **Check threshold**: Lower the auto_memory.threshold value (e.g., to 0.4)
3. **Use explicit commands**: You can always use explicit "please remember..." commands

### Memory Not Being Retrieved

1. **Check query relevance**: Ensure your query is related to stored memories
2. **Check memory existence**: Use the list_memories tool to see if the memory exists
3. **Try more specific queries**: Be more specific in your retrieval queries

### AutoCodeIndex Not Working

1. **Check AutoCodeIndex settings**: Ensure `autocode.enabled` is set to true in config
2. **Verify project structure**: Ensure project has recognizable files (package.json, requirements.txt, etc.)
3. **Check language support**: Verify your project's languages are in `supported_languages`
4. **Review scan depth**: Ensure `max_scan_depth` allows scanning your project structure

### Command Suggestions Not Helpful

1. **Lower confidence threshold**: Reduce `min_confidence_threshold` for more suggestions
2. **Build command history**: Use Claude for more development tasks to build pattern data
3. **Check platform detection**: Ensure platform-specific commands are being tracked

### Pattern Detection Issues

1. **Verify project files**: Ensure standard project files are present and properly formatted
2. **Check ignore patterns**: Review if important directories are being excluded
3. **Increase scan depth**: Consider increasing `max_scan_depth` for complex projects

### Server Not Starting

See the [Compatibility Guide](compatibility.md) for resolving dependency and compatibility issues.

### Additional Help

If you continue to experience issues, please:
1. Check the server logs for error messages
2. Refer to the [Compatibility Guide](compatibility.md)
3. Open an issue on GitHub with detailed information about your problem