# Docker Deployment

This document explains how to run the Memory MCP Server using Docker with the recommended `docker run -i --rm` pattern for MCP server integration.

## Prerequisites

- Docker installed on your system
- Claude Desktop application

## Quick Start (Recommended)

1. **Use the pre-built image from GitHub Container Registry:**

   Add the following to your Claude Desktop MCP configuration file:

   ```json
   {
     "mcpServers": {
       "alunai-memory": {
         "command": "docker",
         "args": [
           "run",
           "-i",
           "--rm",
           "-e", "MEMORY_FILE_PATH",
           "-e", "AUTOCODE_ENABLED",
           "-e", "AUTOCODE_COMMAND_LEARNING_ENABLED",
           "-e", "AUTOCODE_PATTERN_DETECTION_ENABLED",
           "-e", "AUTOCODE_SESSION_ANALYSIS_ENABLED",
           "-e", "AUTOCODE_HISTORY_NAVIGATION_ENABLED",
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

2. **Alternative: Build locally (optional):**
   ```bash
   git clone https://github.com/alun-ai/mcp-alunai-memory.git
   cd mcp-alunai-memory
   docker build -t mcp-alunai-memory .
   ```

3. **Test the server:**
   ```bash
   echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}' | docker run -i --rm -e MEMORY_FILE_PATH=/tmp/memory.json ghcr.io/alun-ai/mcp-alunai-memory:latest
   ```

## How It Works

The Memory MCP Server now uses a stateless container approach:

- **No background containers**: Each MCP request spawns a fresh container that exits when done
- **Environment-based configuration**: All settings passed via environment variables
- **Ephemeral storage**: Memory stored in `/tmp/memory.json` (lives only during conversation)
- **Clean isolation**: Each container starts fresh with no shared state

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_FILE_PATH` | Path to store memory data | `/tmp/memory.json` |

### Custom Memory Persistence

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
        "-v",
        "/path/to/persistent/memory.json:/app/memory.json",
        "-e",
        "MEMORY_FILE_PATH=/app/memory.json",
        "ghcr.io/alun-ai/mcp-alunai-memory:latest"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/app/memory.json"
      }
    }
  }
}
```

### Debug Mode

To enable debug logging:

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
        "ghcr.io/alun-ai/mcp-alunai-memory:latest",
        "--debug"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/tmp/memory.json"
      }
    }
  }
}
```

## Advantages of This Approach

- **No container management**: No need to start/stop background containers
- **Clean state**: Each conversation starts with a clean container environment
- **Resource efficient**: Containers only run when needed
- **Easy deployment**: Works identical to other MCP servers like Jira integration
- **No networking issues**: Uses stdio protocol, no network ports required

## Troubleshooting

### Container fails to start
1. Ensure Docker is running
2. Check if the image is available: `docker images | grep mcp-alunai-memory`
3. Test manually: `docker run -i --rm ghcr.io/alun-ai/mcp-alunai-memory:latest --debug`

### Memory not persisting
- Default configuration uses ephemeral storage (`/tmp/memory.json`)
- For persistence, use volume mounts as shown in configuration options

### Claude can't connect
1. Verify MCP configuration syntax in Claude Desktop
2. Check Claude Desktop logs for connection errors
3. Test the container manually with the echo command shown above