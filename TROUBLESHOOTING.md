# MCP Persistent Memory Troubleshooting Guide

## Common Connection Issues

### Issue: "Connection closed" error in Claude Code

**Error Message:**
```
[DEBUG] MCP server "memory": Connection failed: McpError: MCP error -32000: Connection closed
[DEBUG] MCP server "memory": Error message: MCP error -32000: Connection closed
```

**Root Cause:** Missing dependencies or incorrect configuration path

### Solution 1: Local Development Setup

If you're running the server locally for development:

1. **Install dependencies in a virtual environment:**
   ```bash
   cd /path/to/mcp-persistent-memory
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Use correct Claude Desktop configuration:**
   ```json
   {
     "mcpServers": {
       "memory": {
         "command": "/path/to/mcp-persistent-memory/venv/bin/python",
         "args": ["-m", "memory_mcp"],
         "cwd": "/path/to/mcp-persistent-memory",
         "env": {
           "MCP_CONFIG_DIR": "/Users/youruser/.memory_mcp/config",
           "MEMORY_FILE_PATH": "/Users/youruser/.memory_mcp/data/memory.json",
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

### Solution 2: Docker Setup (Recommended)

For production use, use the pre-built Docker image:

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
        "ghcr.io/alun-ai/mcp-persistent-memory:latest"
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

## Verification Steps

### 1. Test Server Manually

Test if the server starts correctly:

```bash
# For local setup:
source venv/bin/activate
MCP_CONFIG_DIR=~/.memory_mcp/config MEMORY_FILE_PATH=/tmp/test_memory.json python -m memory_mcp --debug

# For Docker:
docker run -i --rm -e MEMORY_FILE_PATH=/tmp/memory.json ghcr.io/alun-ai/mcp-persistent-memory:latest --debug
```

**Expected output:**
```
2025-07-19 18:16:30 | INFO | memory_mcp.domains.manager:initialize:56 - Memory Domain Manager initialized
2025-07-19 18:16:30 | INFO | __main__:main:81 - Starting Memory MCP Server using stdio transport
```

### 2. Check AutoCodeIndex Status

Once connected to Claude, verify AutoCodeIndex is working:

```
User: "Show me AutoCodeIndex stats"
```

**Expected response:**
```
- AutoCodeIndex Status: ✅ Active
- Total Projects Analyzed: [number]
- Command Patterns Learned: [number]
- Session Summaries Generated: [number]
- Memory Types Active: 8 (including 4 AutoCodeIndex types)
```

### 3. Test Command Intelligence

```
User: "How do I run tests in this project?"
```

**Expected response with AutoCodeIndex:**
```
Based on your project's package.json and past usage patterns, I suggest: `npm test`
This command has a 95% success rate in similar React projects.
```

## Common Issues and Fixes

### Issue: Python ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'loguru'`

**Fix:** Install dependencies in virtual environment (see Solution 1 above)

### Issue: Permission denied or read-only filesystem

**Error:** `OSError: [Errno 30] Read-only file system: '/app'`

**Fix:** Set proper environment variables:
```bash
export MCP_CONFIG_DIR=~/.memory_mcp/config
export MEMORY_FILE_PATH=~/.memory_mcp/data/memory.json
```

### Issue: Docker image not found

**Error:** `Unable to find image 'ghcr.io/alun-ai/mcp-persistent-memory:latest'`

**Fix:** Pull the image manually:
```bash
docker pull ghcr.io/alun-ai/mcp-persistent-memory:latest
```

### Issue: AutoCodeIndex not responding

**Symptoms:** No command suggestions or pattern recognition

**Fix:** Verify AutoCodeIndex environment variables are set to "true":
```json
{
  "env": {
    "AUTOCODE_ENABLED": "true",
    "AUTOCODE_COMMAND_LEARNING_ENABLED": "true",
    "AUTOCODE_PATTERN_DETECTION_ENABLED": "true",
    "AUTOCODE_SESSION_ANALYSIS_ENABLED": "true",
    "AUTOCODE_HISTORY_NAVIGATION_ENABLED": "true"
  }
}
```

## Configuration Locations

### Claude Desktop Config File Locations:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

### Memory MCP Config and Data:

- **Config:** `~/.memory_mcp/config/config.json`
- **Data:** `~/.memory_mcp/data/memory.json`
- **Cache:** `~/.memory_mcp/cache/`

## Debug Mode

Enable detailed logging by adding `--debug` flag or setting debug in configuration:

```json
{
  "args": ["-m", "memory_mcp", "--debug"]
}
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_FILE_PATH` | `/tmp/memory.json` | Path to memory storage file |
| `MCP_CONFIG_DIR` | - | Configuration directory path |
| `AUTOCODE_ENABLED` | `true` | Enable/disable AutoCodeIndex |
| `AUTOCODE_COMMAND_LEARNING_ENABLED` | `true` | Enable command learning |
| `AUTOCODE_PATTERN_DETECTION_ENABLED` | `true` | Enable pattern detection |
| `AUTOCODE_SESSION_ANALYSIS_ENABLED` | `true` | Enable session analysis |
| `AUTOCODE_HISTORY_NAVIGATION_ENABLED` | `true` | Enable history navigation |

## Getting Help

If you continue to experience issues:

1. Enable debug mode and check logs
2. Verify all environment variables are correctly set
3. Test server startup manually before connecting to Claude
4. Check file permissions for config and data directories
5. Submit an issue with complete configuration and error logs

## Success Indicators

When everything is working correctly, you should see:

1. **Server startup logs** showing all domains initialized
2. **AutoCodeIndex activation** in Claude responses
3. **Command suggestions** with confidence scores
4. **Project pattern recognition** automatically working
5. **Session continuity** across conversations