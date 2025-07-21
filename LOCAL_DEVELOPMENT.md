# Local Development Setup

This guide helps you set up Alunai Clarity MCP Server for local development and testing.

## Quick Start

1. **Run the setup script:**
   ```bash
   ./start-local-dev.sh
   ```

   This script will:
   - Start Qdrant vector database in Docker
   - Set up Python virtual environment  
   - Install dependencies
   - Create necessary config files

2. **Test the server manually:**
   ```bash
   python -m clarity --config .claude/alunai-clarity/config.json --debug
   ```

3. **Use with Claude Desktop:**
   The `.mcp.json` file is already configured for local development. Restart Claude Desktop to use the local server.

## Manual Setup

### Prerequisites

- Python 3.8+
- Docker (for Qdrant)
- Git

### Step-by-step Setup

1. **Start Qdrant database:**
   ```bash
   docker run -d \
     --name alunai-clarity-qdrant \
     -p 6333:6333 \
     -p 6334:6334 \
     -v ./qdrant_data:/qdrant/storage \
     --rm \
     qdrant/qdrant:latest
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. **Configure for local development:**
   The configuration is already set up in `.claude/alunai-clarity/config.json`

4. **Test the server:**
   ```bash
   python -m clarity --config .claude/alunai-clarity/config.json --debug
   ```

## Configuration

### Local Development Config

The local config (`.claude/alunai-clarity/config.json`) includes:

- **Qdrant**: Local instance at `http://localhost:6333`
- **AutoCode**: All features enabled for testing
- **Logging**: Debug level with file logging
- **Audit**: Full audit trail enabled
- **Security**: Permissive settings for development

### MCP Configuration

The `.mcp.json` file is configured to:
- Run the Python module directly (no Docker)
- Use local config and memory files
- Enable debug mode
- Set proper environment variables

## Testing

### Manual Testing

1. **Test MCP server directly:**
   ```bash
   python -m clarity --config .claude/alunai-clarity/config.json --debug
   ```

2. **Test with MCP CLI (if available):**
   ```bash
   mcp-cli test stdio -- python -m clarity --config .claude/alunai-clarity/config.json
   ```

### Integration Testing with Claude Desktop

1. Ensure Qdrant is running
2. Restart Claude Desktop
3. Try using memory-related commands:
   - "Store this information: Python is great"
   - "What do you remember about Python?"
   - "Analyze this project structure"

## Development Features

### Enabled Components

- **Memory Management**: Full Qdrant-based storage
- **AutoCode Intelligence**: Project pattern detection
- **Command Learning**: Bash command tracking
- **Session Analysis**: Conversation summaries
- **History Navigation**: Similar session finding
- **Audit Trails**: Full operation logging
- **Configuration Validation**: Schema-based validation

### Log Files

- **Application logs**: `.claude/alunai-clarity/alunai-clarity.log`
- **Audit logs**: Integrated into main log with structured format
- **Qdrant data**: `./qdrant_data/` directory

## Troubleshooting

### Common Issues

1. **Qdrant connection failed:**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   
   # Restart Qdrant if needed
   docker stop alunai-clarity-qdrant
   docker run -d --name alunai-clarity-qdrant -p 6333:6333 -p 6334:6334 --rm qdrant/qdrant:latest
   ```

2. **Module import errors:**
   ```bash
   # Ensure you're in the right directory and venv is activated
   source venv/bin/activate  # or test_env/bin/activate
   pip install -e .
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```

3. **Configuration errors:**
   ```bash
   # Validate configuration
   python -c "from clarity.shared.config import validate_config_file; print(validate_config_file('.claude/alunai-clarity/config.json'))"
   ```

### Cleanup

```bash
# Stop Qdrant
docker stop alunai-clarity-qdrant

# Clean up data (optional)
rm -rf qdrant_data .claude/alunai-clarity/*.log
```

## Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Qdrant | Local Docker | External service |
| Logging | Debug level | Info level |
| Security | Permissive | Strict validation |
| Audit | File-based | Database/external |
| Config validation | Warnings only | Strict errors |

## Next Steps

1. Test all MCP functions work correctly
2. Try the AutoCode project analysis features
3. Test memory storage and retrieval
4. Verify audit logging is working
5. Test configuration hot-reload