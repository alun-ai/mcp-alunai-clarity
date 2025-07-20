# Claude Desktop Integration Guide

This guide explains how to integrate the Memory MCP Server with the Claude Desktop application for enhanced memory capabilities.

## Overview

The Memory MCP Server implements the Model Context Protocol (MCP) to provide Claude with persistent memory capabilities. After setting up the server, you can configure Claude Desktop to use it for remembering information across conversations.

## Prerequisites

- Claude Desktop application installed
- Memory MCP Server installed and configured

## Configuration

### 1. Locate Claude Desktop Configuration

The Claude Desktop configuration file is typically located at:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### 2. Add Memory MCP Server Configuration

Edit your `claude_desktop_config.json` file to include the Memory MCP Server:

```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "python",
      "args": ["-m", "memory_mcp"],
      "env": {
        "MEMORY_FILE_PATH": "./.claude/alunai-memory/memory.json"
      }
    }
  }
}
```

Replace `/path/to/your/memory.json` with your desired memory file location.

### 3. Optional: Configure MCP Server

You can customize the Memory MCP Server by creating a configuration file at `~/.memory_mcp/config/config.json`:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": false
  },
  "alunai-memory": {
    "max_short_term_items": 100,
    "max_long_term_items": 1000,
    "max_archival_items": 10000,
    "consolidation_interval_hours": 24,
    "short_term_threshold": 0.3,
    "file_path": "/path/to/your/memory.json"
  },
  "embedding": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "cache_dir": "~/.memory_mcp/cache"
  },
  "retrieval": {
    "default_top_k": 5,
    "semantic_threshold": 0.75,
    "recency_weight": 0.3,
    "importance_weight": 0.7
  }
}
```

### 4. Docker Container Option (Recommended)

Alternatively, you can run the Memory MCP Server as a Docker container using the pre-built image:

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

For persistent memory across Docker runs, you can mount a volume:

```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v", "~/.claude/global-memory:/data",
        "-e",
        "MEMORY_FILE_PATH=/data/memory.json",
        "ghcr.io/alun-ai/mcp-alunai-memory:latest"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/data/memory.json"
      }
    }
  }
}
```

The recommended approach uses project-local storage with proactive AutoCode features enabled for intelligent command suggestions and automatic pattern analysis.

## Using Memory Tools in Claude

Once configured, Claude Desktop will automatically connect to the Memory MCP Server. You can use the provided memory tools in your conversations with Claude:

### Store Memory

To explicitly store information in memory:

```
Could you remember that my favorite color is blue?
```

Claude will use the `store_memory` tool to save this information.

### Retrieve Memory

To recall information from memory:

```
What's my favorite color?
```

Claude will use the `retrieve_memory` tool to search for relevant memories.

### System Prompt

For optimal memory usage, consider adding these instructions to your Claude Desktop System Prompt:

```
Follow these steps for each interaction:

1. Memory Retrieval:
   - Always begin your chat by saying only "Remembering..." and retrieve all relevant information from your knowledge graph
   - Always refer to your knowledge graph as your "memory"

2. Memory Update:
   - While conversing with the user, be attentive to any new information about the user
   - If any new information was gathered during the interaction, update your memory
```

## Troubleshooting

### Memory Server Not Starting

If the Memory MCP Server fails to start:

1. Check your Python installation and ensure all dependencies are installed
2. Verify the configuration file paths are correct
3. Check if the memory file directory exists and is writable
4. Look for error messages in the Claude Desktop logs

### Memory Not Being Stored

If Claude is not storing memories:

1. Ensure the MCP server is running (check Claude Desktop logs)
2. Verify that your system prompt includes instructions to use memory
3. Make sure Claude has clear information to store (be explicit)

### Memory File Corruption

If the memory file becomes corrupted:

1. Stop Claude Desktop
2. Rename the corrupted file
3. The MCP server will create a new empty memory file on next start

## Advanced Configuration

### Custom Embedding Models

You can use different embedding models by changing the `embedding.model` configuration:

```json
"embedding": {
  "model": "sentence-transformers/paraphrase-MiniLM-L6-v2",
  "dimensions": 384
}
```

### Memory Consolidation Settings

Adjust memory consolidation behavior:

```json
"alunai-memory": {
  "consolidation_interval_hours": 12,
  "importance_decay_rate": 0.02
}
```

### Retrieval Fine-Tuning

Fine-tune memory retrieval by adjusting these parameters:

```json
"retrieval": {
  "recency_weight": 0.4,
  "importance_weight": 0.6
}
```

Increase `recency_weight` to prioritize recent memories, or increase `importance_weight` to prioritize important memories.
