# Claude Desktop Integration Guide

This guide explains how to set up and use the Memory MCP Server with the Claude Desktop application.

## Installation

First, ensure you have installed the Memory MCP Server by following the instructions in the [README.md](../README.md) file.

## Configuration

### 1. Configure Claude Desktop

To enable the Memory MCP Server in Claude Desktop, you need to add it to the Claude Desktop configuration file.

The configuration file is typically located at:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

#### Option A: Docker Configuration (Recommended)

Edit the file to add the following MCP server configuration:

```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "MEMORY_FILE_PATH",
        "memory-mcp"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/tmp/memory.json"
      }
    }
  }
}
```

#### Option B: Python Configuration

For local Python installation:

```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "python",
      "args": ["-m", "memory_mcp"],
      "env": {
        "MEMORY_FILE_PATH": "./memory.json"
      }
    }
  }
}
```

### 2. Configure Environment Variables (Optional)

You can customize the behavior of the Memory MCP Server by setting environment variables:

- `MCP_DATA_DIR`: Directory for memory data (default: `~/.memory_mcp`)
- `MCP_CONFIG_DIR`: Directory for configuration files (default: `~/.memory_mcp/config`)

### 3. Customize Memory File Location (Optional)

By default, the Memory MCP Server stores memory data in:
- `~/.memory_mcp/data/memory.json`

You can customize this location by setting the `MEMORY_FILE_PATH` environment variable in the Claude Desktop configuration.

## Using Memory Features in Claude

### 1. Starting Claude Desktop

After configuring the MCP server, start Claude Desktop. The Memory MCP Server will start automatically when Claude connects to it.

### 2. Available Memory Tools

Claude has access to the following memory-related tools:

#### store_memory
Store new information in memory.

```json
{
  "type": "conversation|fact|document|entity|reflection|code",
  "content": {
    // Type-specific content structure
  },
  "importance": 0.75, // Optional: 0.0-1.0 (higher is more important)
  "metadata": {}, // Optional: Additional metadata
  "context": {} // Optional: Contextual information
}
```

#### retrieve_memory
Retrieve relevant memories based on a query.

```json
{
  "query": "What is the capital of France?",
  "limit": 5, // Optional: Maximum number of results
  "types": ["fact", "document"], // Optional: Memory types to include
  "min_similarity": 0.6, // Optional: Minimum similarity score
  "include_metadata": true // Optional: Include metadata in results
}
```

#### list_memories
List available memories with filtering options.

```json
{
  "types": ["conversation", "fact"], // Optional: Memory types to include
  "limit": 20, // Optional: Maximum number of results
  "offset": 0, // Optional: Offset for pagination
  "tier": "short_term", // Optional: Memory tier to filter by
  "include_content": true // Optional: Include memory content in results
}
```

#### update_memory
Update existing memory entries.

```json
{
  "memory_id": "mem_1234567890",
  "updates": {
    "content": {}, // Optional: New content
    "importance": 0.8, // Optional: New importance score
    "metadata": {}, // Optional: Updates to metadata
    "context": {} // Optional: Updates to context
  }
}
```

#### delete_memory
Remove specific memories.

```json
{
  "memory_ids": ["mem_1234567890", "mem_0987654321"]
}
```

#### memory_stats
Get statistics about the memory store.

```json
{}
```

### 3. Example Usage

Claude can use these memory tools to store and retrieve information. Here are some example prompts:

#### Storing a Fact

```
Please remember that Paris is the capital of France.
```

Claude might use the `store_memory` tool to save this fact:

```json
{
  "type": "fact",
  "content": {
    "fact": "Paris is the capital of France",
    "confidence": 0.98,
    "domain": "geography"
  },
  "importance": 0.7
}
```

#### Retrieving Information

```
What important geographical facts do you remember?
```

Claude might use the `retrieve_memory` tool to find relevant facts:

```json
{
  "query": "important geographical facts",
  "types": ["fact"],
  "min_similarity": 0.6
}
```

#### Saving User Preferences

```
Please remember that I prefer to see code examples in Python, not JavaScript.
```

Claude might use the `store_memory` tool to save this preference:

```json
{
  "type": "entity",
  "content": {
    "name": "user",
    "entity_type": "person",
    "attributes": {
      "code_preference": "Python"
    }
  },
  "importance": 0.8
}
```

### 4. Memory Persistence

The Memory MCP Server maintains memory persistence across conversations. 

When Claude starts a new conversation, it can access memories from previous conversations. The memory system uses a tiered approach:

- **Short-term memory**: Recently created or accessed memories
- **Long-term memory**: Older, less frequently accessed memories
- **Archived memory**: Rarely accessed memories that may still be valuable

The system automatically manages the movement of memories between tiers based on access patterns, importance, and other factors.

## Advanced Configuration

### Memory Consolidation

The Memory MCP Server automatically consolidates memories based on the configured interval (default: 24 hours). 

You can customize this behavior by setting the `consolidation_interval_hours` parameter in the configuration file.

### Memory Tiers

The memory tiers have default size limits that you can adjust in the configuration:

```json
{
  "alunai-memory": {
    "max_short_term_items": 100,
    "max_long_term_items": 1000,
    "max_archival_items": 10000
  }
}
```

### Embedding Model

The Memory MCP Server uses an embedding model to convert text into vector representations for semantic search.

You can customize the embedding model in the configuration:

```json
{
  "embedding": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "cache_dir": "~/.memory_mcp/cache"
  }
}
```

## Troubleshooting

### Checking Server Status

The Memory MCP Server logs to standard error. In the Claude Desktop console output, you should see messages indicating the server is running.

### Common Issues

#### Server won't start

- Check if the path to the memory file is valid
- Verify that all dependencies are installed
- Check permissions for data directories

#### Memory not persisting

- Verify that the memory file path is correct
- Check if the memory file exists and is writable
- Ensure Claude has permission to execute the MCP server

#### Embedding model issues

- Check if the embedding model is installed
- Verify that the model name is correct
- Ensure you have sufficient disk space for model caching

## Security Considerations

The Memory MCP Server stores memories on your local file system. Consider these security aspects:

- **Data Privacy**: The memory file contains all stored memories, which may include sensitive information.
- **File Permissions**: Ensure the memory file has appropriate permissions to prevent unauthorized access.
- **Encryption**: Consider encrypting the memory file if it contains sensitive information.

## Further Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Claude Desktop Documentation](https://claude.ai/docs)
- [Memory MCP Server GitHub Repository](https://github.com/alun-ai/claude-memory-mcp)
