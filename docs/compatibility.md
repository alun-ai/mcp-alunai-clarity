# Compatibility Guide

This guide helps you resolve compatibility issues with the Memory MCP Server.

## Supported Environments

The Memory MCP Server is compatible with:

- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Windows, macOS, Linux

## Key Dependencies

| Dependency | Supported Versions | Notes |
|------------|-------------------|-------|
| NumPy | 1.20.0 - 1.x.x | **Not compatible with NumPy 2.x** |
| Pydantic | 2.4.0 - 2.x.x | |
| sentence-transformers | 2.2.2 - 2.x.x | |
| MCP library | 1.2.0+ | Model Context Protocol Python SDK |

## Common Issues and Solutions

### NumPy 2.x Incompatibility

**Issue**: The error message mentions NumPy version incompatibility.

**Solution**:
```bash
pip uninstall numpy
pip install "numpy>=1.20.0,<2.0.0"
```

### Python Version Errors

**Issue**: You see an error about unsupported Python version.

**Solution**:
1. Check your Python version: `python --version`
2. Install a supported Python version (3.8-3.12)
3. Create a new virtual environment with the supported version:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### MCP Libraries Not Found

**Issue**: Error about missing MCP libraries.

**Solution**:
```bash
pip install mcp-cli>=0.1.0,<0.3.0 mcp-server>=0.1.0,<0.3.0
```

If you need a newer version of the MCP libraries, you can install them directly:
```bash
pip install git+https://github.com/anthropics/mcp-cli.git
pip install git+https://github.com/anthropics/mcp-server.git
```

### Other Dependency Issues

**Solution**:
1. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Docker Option

If you continue to have dependency issues, consider using Docker instead:

```bash
docker run -d \
  --name memory-mcp \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/data:/app/data" \
  whenmoon-afk/claude-memory-mcp
```

See the [Docker Usage Guide](docker_usage.md) for more details.

## Bypassing Compatibility Check

If you want to skip the compatibility check (not recommended):

```bash
python -m memory_mcp --skip-compatibility-check
```