# Local Development Environment

This directory contains Docker-based development tools for testing the alunai-clarity MCP server with real-time code mounting and hook execution debugging.

## Files

### Docker Configuration
- **`Dockerfile.dev`** - Development Docker image with live code mounting and debug logging
- **`docker-compose.dev.yml`** - Docker Compose setup with persistent volumes and development environment
- **`mcp-config-dev.json`** - Claude Desktop MCP configuration for connecting to the dev container

### Development Scripts
- **`dev-test.sh`** - Main development testing script with commands:
  - `./dev-test.sh start` - Start the development container
  - `./dev-test.sh stop` - Stop the development container
  - `./dev-test.sh logs -f` - Follow logs in real-time
  - `./dev-test.sh shell` - Open shell in container
  - `./dev-test.sh test hooks` - Test hook execution
  - `./dev-test.sh config` - Show MCP configuration for Claude
  - `./dev-test.sh clean` - Clean up containers and volumes

### Test Scripts
- **`test-mcp-connection.py`** - Basic MCP server connection and import testing
- **`test-hook-execution.py`** - Hook system initialization and method testing
- **`test-hook-final.py`** - Comprehensive hook execution test with proper configuration

## Quick Start

1. **Start Development Environment**
   ```bash
   cd local-dev
   ./dev-test.sh start
   ```

2. **Monitor Logs**
   ```bash
   ./dev-test.sh logs -f
   ```

3. **Configure Claude Desktop**
   Add the configuration from `mcp-config-dev.json` to your Claude Desktop settings.

4. **Test Hook Execution**
   Connect Claude to the container and run commands that should trigger hooks.

## Features

- **Live Code Mounting**: Changes to your local code are immediately reflected in the container
- **Debug Logging**: Full debug output to help identify hook execution issues
- **Real-time Testing**: Test hook functionality as you interact with Claude
- **Persistent Data**: Container data persists between restarts
- **Easy Debugging**: Shell access and comprehensive test scripts

## Troubleshooting

- If container fails to start, check logs with `./dev-test.sh logs`
- If hooks aren't working, run `./dev-test.sh test hooks` to identify issues
- For comprehensive testing, run the test scripts directly in the container

## Notes

This development environment is specifically designed to help debug hook execution issues that are difficult to test with unit/integration tests alone. The real-time logging and live code mounting make it easy to identify and fix problems as they occur.