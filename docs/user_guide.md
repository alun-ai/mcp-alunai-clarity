# User Guide: Claude Memory MCP Server

This guide explains how to set up and use the Memory MCP Server with Claude Desktop for persistent memory capabilities.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [How Memory Works](#how-memory-works)
4. [Usage Examples](#usage-examples)
5. [Advanced Configuration](#advanced-configuration)
6. [Troubleshooting](#troubleshooting)

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
    "memory": {
      "command": "python",
      "args": ["-m", "memory_mcp"],
      "env": {
        "MEMORY_FILE_PATH": "/path/to/your/memory.json"
      }
    }
  }
}
```

### Memory System Prompt

For optimal memory usage, add these instructions to your Claude Desktop System Prompt:

```
This Claude instance has been enhanced with persistent memory capabilities.
Claude will automatically:
1. Remember important details about you across conversations
2. Store key facts and preferences you share
3. Recall relevant information when needed

You don't need to explicitly ask Claude to remember or recall information.
Simply have natural conversations, and Claude will maintain memory of important details.

To see what Claude remembers about you, just ask "What do you remember about me?"
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
  "memory": {
    "max_short_term_items": 200,
    "max_long_term_items": 2000,
    "consolidation_interval_hours": 48
  }
}
```

### Auto-Memory Settings

- `enabled`: Enable/disable automatic memory (default: true)
- `threshold`: Minimum importance threshold for auto-storage (0.0-1.0)
- `store_assistant_messages`: Whether to store assistant messages (default: false)
- `entity_extraction_enabled`: Enable entity extraction from messages (default: true)

## Troubleshooting

### Memory Not Being Stored

1. **Check auto-memory settings**: Ensure auto_memory.enabled is true in config
2. **Check threshold**: Lower the auto_memory.threshold value (e.g., to 0.4)
3. **Use explicit commands**: You can always use explicit "please remember..." commands

### Memory Not Being Retrieved

1. **Check query relevance**: Ensure your query is related to stored memories
2. **Check memory existence**: Use the list_memories tool to see if the memory exists
3. **Try more specific queries**: Be more specific in your retrieval queries

### Server Not Starting

See the [Compatibility Guide](compatibility.md) for resolving dependency and compatibility issues.

### Additional Help

If you continue to experience issues, please:
1. Check the server logs for error messages
2. Refer to the [Compatibility Guide](compatibility.md)
3. Open an issue on GitHub with detailed information about your problem