# Alunai Clarity

**An MCP server that provides cognitive abilities to Claude and other MCP-aware systems.**

High-performance Qdrant vector database storage with intelligent memory management, procedural thinking, and real-time MCP server discovery.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)

## Technical Capabilities

Alunai Clarity provides the following core capabilities:

- **High-performance vector memory storage** using Qdrant database with sub-millisecond search
- **Procedural and sequential thinking** with 5-stage systematic analysis framework
- **Automatic memory capture** from "Remember this:" patterns without explicit tool calls
- **Real-time MCP server discovery** and proactive tool suggestions
- **Connection pooling with recovery** for reliable production deployments
- **Concurrent session support** for multiple Claude instances sharing memory
- **Intelligent embedding models** with fallback system for faster initialization
- **AutoCode intelligence** for project pattern recognition and workflow optimization
- **Docker containerization** with volume mounting for persistent storage
- **Claude Code hooks integration** for seamless user experience
- **Performance monitoring** with optimization recommendations
- **JSON migration tools** for upgrading from file-based storage

## Architecture

### Memory System
- **Vector Database**: Qdrant for semantic similarity search and scalable storage
- **Embedding Models**: Configurable sentence transformers with intelligent fallbacks
- **Connection Pooling**: Shared client management with automatic recovery mechanisms
- **Memory Types**: 12 specialized types including structured thinking and project patterns
- **Tiered Storage**: Short-term, long-term, and archival memory tiers

### Thinking Framework
- **Native Sequential Thinking**: 5-stage analysis process (problem_analysis → context_exploration → solution_generation → evaluation_analysis → implementation_planning)
- **Procedural Thinking Triggers**: Automatic detection of complex task patterns
- **Claude Code Integration**: UserPromptSubmit and PermissionDecision hooks
- **Pattern Recognition**: Complexity scoring and automatic thinking activation

### Production Features
- **Connection Recovery**: Multi-tier fallback system with timeout protection
- **Stale Lock Detection**: Automatic cleanup of abandoned connection locks
- **Performance Optimization**: Fast initialization with optimized HNSW parameters
- **Error Handling**: Comprehensive error recovery with "instance is closed" detection
- **Health Monitoring**: Real-time connection health checks and statistics

## Quick Start

### Docker Setup (Recommended)

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "alunai-clarity": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "./.claude/alunai-clarity:/app/data",
        "ghcr.io/alun-ai/mcp-alunai-clarity:latest"
      ],
      "type": "stdio"
    }
  }
}
```

Storage structure:
- `./.claude/alunai-clarity/config.json` - Configuration
- `./.claude/alunai-clarity/qdrant/` - Vector database 
- `./.claude/alunai-clarity/cache/` - Model cache
- `./.claude/alunai-clarity/hooks.json` - Claude Code hooks

### Enabling Auto-Capture

Add to your `CLAUDE.md`:

```markdown
You have persistent memory with automatic capture enabled.

Remember important information by saying "Remember this: [content]"
This will automatically store memories without explicit tool calls.
```

## Core Tools

### Memory Management
- `store_memory` - Store memories with vector indexing
- `retrieve_memories` - Semantic similarity search
- `list_memories` - Browse with filtering
- `update_memory` - Modify existing memories
- `delete_memory` - Remove memories
- `memory_stats` - System statistics

### Thinking Framework
- `sequential_thinking` - 5-stage systematic analysis
- `structured_thinking_domain` - Structured thought processing
- Automatic procedural thinking triggers for complex tasks

### MCP Awareness
- `discover_mcp_tools` - Auto-discover installed MCP servers
- `suggest_mcp_alternatives` - Recommend MCP tools over scripts
- `get_mcp_tool_info` - Tool information and schemas

### Performance & Monitoring
- `qdrant_performance_stats` - Detailed performance metrics
- `optimize_qdrant_collection` - Database optimization
- `connection_health_check` - Connection status monitoring

### AutoCode Intelligence
- `suggest_command` - Context-aware command suggestions
- `get_project_patterns` - Framework and architecture detection
- `find_similar_sessions` - Historical session analysis
- `autocode_stats` - Intelligence system metrics

## Configuration

### Basic Configuration
```json
{
  "qdrant": {
    "path": "/app/data/qdrant",
    "timeout": 30.0,
    "prefer_grpc": false
  },
  "embedding": {
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "fast_model": "paraphrase-MiniLM-L3-v2",
    "dimensions": 384,
    "cache_dir": "/app/data/cache"
  }
}
```

### Advanced Configuration
```json
{
  "qdrant": {
    "init_index_params": {
      "m": 8,
      "ef_construct": 64,
      "full_scan_threshold": 20
    },
    "index_params": {
      "m": 16,
      "ef_construct": 200,
      "full_scan_threshold": 50
    }
  },
  "autocode": {
    "enabled": true,
    "command_learning": {
      "enabled": true,
      "min_confidence_threshold": 0.3
    },
    "pattern_detection": {
      "enabled": true,
      "supported_languages": ["typescript", "javascript", "python", "rust", "go"]
    }
  }
}
```

## Performance

### Memory Search Performance
| Memory Count | Vector Search | Improvement |
|--------------|---------------|-------------|
| 1K memories  | ~0.1ms       | 100x faster |
| 10K memories | ~1ms         | 100x faster |
| 100K memories| ~2ms         | 500x faster |

### Initialization Performance
- First-time startup: ~6s (model loading + initialization)
- Subsequent operations: ~10ms
- Fast model option: ~3s initialization with paraphrase-MiniLM-L3-v2
- Connection recovery: Sub-second automatic failover

## Migration from JSON

Migrate existing JSON-based memory to high-performance Qdrant:

```bash
# Using Docker
docker run --entrypoint="python" \
           -v /path/to/memory.json:/tmp/memory.json \
           -v ./.claude/alunai-clarity:/app/data \
           ghcr.io/alun-ai/mcp-alunai-clarity:latest \
           -m clarity.cli.import_json /tmp/memory.json
```

Results:
- Automatic data integrity verification
- 100x+ performance improvement
- Batch processing for large datasets
- Rollback safety (original files unchanged)

## Troubleshooting

### Connection Issues
- **Hanging operations**: Fixed with timeout protection and automatic recovery
- **"Instance is closed" errors**: Automatic detection and client recreation
- **Stale locks**: Automatic cleanup based on process validation

### Performance Issues
- **Slow initialization**: Use `fast_model` configuration option
- **Memory usage**: Check `qdrant_performance_stats` for optimization
- **Search performance**: Run `optimize_qdrant_collection`

### Migration Issues
- **Import failures**: Use `--dry-run` to validate JSON structure
- **Large datasets**: Increase `--batch-size` for better throughput
- **Permission errors**: Ensure write access to data directory

## Development

### Local Development
```bash
# Clone repository
git clone https://github.com/alun-ai/mcp-alunai-clarity.git
cd mcp-alunai-clarity

# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/

# Build Docker image
docker build -t mcp-alunai-clarity .
```

### Testing
```bash
# Run connection fix tests
python test_qdrant_connection_fixes.py

# Performance testing
python -c "
import asyncio
from clarity.domains.persistence import QdrantPersistenceDomain
# Test code here
"
```

## License

MIT License - see [LICENSE](LICENSE) file for details.