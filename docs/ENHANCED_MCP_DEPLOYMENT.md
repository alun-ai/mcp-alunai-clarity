# Enhanced MCP Discovery System - Deployment Guide

## Overview

This guide covers the deployment of the Enhanced MCP Discovery System, which provides comprehensive MCP server discovery, learning capabilities, and integration with Claude Code's native functionality.

## System Requirements

### Dependencies
- Python 3.9+
- Claude Code CLI (optional, for native integration)
- MCP client libraries (optional, graceful fallback available)
- Asyncio-compatible environment

### Optional Dependencies
- `mcp` package for full slash command discovery
- `psutil` for memory usage monitoring

## Deployment Configuration

### 1. Core Component Activation

The enhanced system automatically initializes when the `MCPToolIndexer` is created:

```python
from clarity.mcp.tool_indexer import MCPToolIndexer

# Initialize with your domain manager
indexer = MCPToolIndexer(domain_manager)

# The enhanced components are automatically initialized:
# - Native Claude Code discovery
# - Hook-based learning
# - Resource reference monitoring  
# - Workflow memory patterns
# - Performance optimization
```

### 2. Claude Code Hook Integration

Configure Claude Code hooks to enable real-time learning:

#### Hook Configuration File: `~/.config/claude/hooks.json`

```json
{
  "hooks": {
    "PreToolUse": {
      "command": "/path/to/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py",
      "args": ["--pre-tool", "--tool", "{tool_name}", "--args", "{tool_args}"],
      "timeout": 5000,
      "enabled": true
    },
    "PostToolUse": {
      "command": "/path/to/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py", 
      "args": ["--post-tool", "--tool", "{tool_name}", "--result", "{tool_result}", "--success", "{success}"],
      "timeout": 5000,
      "enabled": true
    },
    "UserPromptSubmit": {
      "command": "/path/to/mcp-alunai-clarity/clarity/mcp/hook_analyzer.py",
      "args": ["--prompt-submit", "--prompt", "{prompt}"],
      "timeout": 5000,
      "enabled": true
    }
  }
}
```

### 3. Performance Configuration

#### Environment Variables

```bash
# Performance optimization settings
export MCP_DISCOVERY_CACHE_TTL=300          # 5 minutes
export MCP_DISCOVERY_RESPONSE_TARGET=500    # 500ms target
export MCP_DISCOVERY_MAX_PARALLEL=10        # Max parallel operations
export MCP_DISCOVERY_BATCH_SIZE=5           # Server discovery batch size

# Logging configuration
export MCP_DISCOVERY_LOG_LEVEL=INFO
export MCP_DISCOVERY_PERFORMANCE_TRACKING=true
```

#### Performance Optimization Settings

```python
from clarity.mcp.performance_optimization import PerformanceOptimizer

# Configure performance optimizer
optimizer = PerformanceOptimizer()
optimizer.target_response_time = 0.5  # 500ms target
optimizer.cache.default_ttl = 300     # 5 minute cache
optimizer.max_parallel_tasks = 10     # Parallel limit
```

### 4. Production Monitoring

#### Performance Monitoring

```python
# Get performance report
performance_report = optimizer.get_performance_report()

# Monitor key metrics:
# - Average response time
# - Cache hit rate  
# - Target compliance rate
# - Slowest operations
```

#### Memory Usage Monitoring

```python
# Get memory statistics
memory_stats = optimizer.get_memory_usage()

# Monitor:
# - RSS memory usage
# - Cache size
# - Metrics count
```

## Feature Configuration

### 1. Native Discovery Bridge

Automatically discovers servers from:
- Claude Code configuration files
- `claude mcp list` command output
- Environment variables

Configuration paths checked:
- `~/.config/claude/config.json`
- `~/.claude/config.json`
- `./claude_config.json`

### 2. Hook Integration Learning

Real-time learning from:
- Tool usage patterns
- Successful workflows
- Resource reference opportunities
- Command success/failure rates

### 3. Resource Reference Monitoring

Detects opportunities for `@server:protocol://` usage in:
- File operations
- Database queries
- Web requests
- Git operations
- API endpoints
- Documentation access

### 4. Slash Command Discovery

Discovers MCP-exposed slash commands:
- Connects to MCP servers
- Retrieves available prompts
- Categorizes commands
- Provides contextual suggestions

### 5. Workflow Memory Enhancement

Stores and suggests successful patterns:
- MCP workflow sequences
- Context-aware suggestions
- Pattern analytics
- Success rate tracking

## API Usage Examples

### Basic Discovery

```python
# Discover and index all MCP tools
tools = await indexer.discover_and_index_tools()
print(f"Discovered {len(tools)} MCP tools")
```

### Resource Suggestions

```python
# Get resource reference suggestions
suggestions = await indexer.get_resource_suggestions(
    "read the database configuration file"
)

for suggestion in suggestions:
    print(f"Suggested: {suggestion['reference']}")
    print(f"Confidence: {suggestion['confidence']}")
```

### Workflow Suggestions

```python
# Get workflow suggestions
suggestions = await indexer.get_workflow_suggestions(
    "setup database connection",
    context={'project_type': 'web_app'}
)

for suggestion in suggestions:
    print(f"Pattern: {suggestion['pattern']}")
    print(f"Tools: {suggestion['tools']}")
```

### Slash Command Suggestions

```python
# Get contextual slash command suggestions
suggestions = await indexer.get_slash_command_suggestions(
    "I need to process some data"
)

for suggestion in suggestions:
    print(f"Command: {suggestion['command']}")
    print(f"Description: {suggestion['description']}")
```

### Performance Analytics

```python
# Get comprehensive analytics
analytics = await indexer.get_comprehensive_analytics()

print(f"Total servers: {analytics['discovery_status']['total_servers']}")
print(f"Total tools: {analytics['discovery_status']['total_tools']}")
print(f"Resource patterns: {analytics['resource_monitoring']['patterns_learned']}")
```

## Integration with Existing Systems

### Domain Manager Integration

The system integrates with any domain manager that implements:

```python
class DomainManager:
    async def store_memory(self, memory_type: str, content: str, 
                          importance: float, metadata: dict = None) -> str:
        """Store memory and return memory ID"""
        pass
    
    async def retrieve_memories(self, query: str, types: list = None,
                               limit: int = 10, min_similarity: float = 0.5) -> list:
        """Retrieve relevant memories"""
        pass
```

### Custom Server Discovery

Extend server discovery with custom sources:

```python
class CustomToolIndexer(MCPToolIndexer):
    async def _discover_servers_from_custom_source(self):
        """Custom server discovery logic"""
        return {"custom-server": {"command": "python", "args": ["-m", "custom"]}}
```

## Production Deployment Checklist

### Pre-Deployment
- [ ] Verify Python dependencies installed
- [ ] Configure hook integration paths
- [ ] Set performance environment variables
- [ ] Test hook analyzer script permissions
- [ ] Validate domain manager integration

### Deployment
- [ ] Deploy enhanced MCP discovery components
- [ ] Configure Claude Code hooks
- [ ] Enable performance monitoring
- [ ] Verify native discovery functionality
- [ ] Test resource reference suggestions

### Post-Deployment
- [ ] Monitor performance metrics
- [ ] Verify learning functionality
- [ ] Check memory usage patterns
- [ ] Validate hook system integration
- [ ] Review discovery analytics

## Troubleshooting

### Common Issues

#### Hook Integration Not Working
1. Verify hook analyzer script is executable: `chmod +x clarity/mcp/hook_analyzer.py`
2. Check Claude Code hooks configuration file exists
3. Verify Python path in hook configuration
4. Check hook analyzer debug output: `--debug` flag

#### Performance Issues
1. Check target response time settings
2. Verify cache configuration
3. Monitor parallel execution limits
4. Review server discovery timeouts

#### Discovery Issues
1. Verify Claude Code configuration files exist
2. Check MCP client availability
3. Review server configuration formats
4. Monitor discovery cache invalidation

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export MCP_DISCOVERY_LOG_LEVEL=DEBUG
```

### Performance Debugging

```python
# Generate performance report
report = await indexer.performance_optimizer.get_performance_report()

# Check slowest operations
for op in report['slowest_operations']:
    print(f"{op[0]}: {op[1]:.1f}ms")
```

## Security Considerations

### Hook System Security
- Hook analyzer runs with limited permissions
- Timeouts prevent hanging processes
- Error handling prevents information leakage
- Input validation on all hook parameters

### Memory Storage Security
- No sensitive data stored in patterns
- Content truncation for large inputs
- Metadata sanitization
- Configurable memory retention

### Network Security
- Server discovery uses secure connections when available
- Timeout protection against slow responses
- Graceful handling of connection failures
- No credential storage in discovery cache

## Performance Tuning

### Cache Optimization
```python
# Adjust cache settings for workload
optimizer.cache.max_size = 2000           # Increase cache size
optimizer.cache.default_ttl = 600         # 10 minute TTL for stable servers
```

### Parallel Execution Tuning
```python
# Optimize for server count and response time
optimizer.max_parallel_tasks = 15         # Increase for many servers
optimizer.executor.timeout = 10.0         # Adjust for slow servers
```

### Memory Management
```python
# Configure memory usage limits
optimizer.max_metrics = 2000              # Increase metrics retention
workflow_enhancer.max_patterns = 500      # Limit pattern storage
```

## Monitoring and Metrics

### Key Performance Indicators
- **Response Time**: Average < 500ms for critical operations
- **Cache Hit Rate**: Target > 70% for repeated operations
- **Discovery Success Rate**: Target > 95% for available servers
- **Memory Usage**: Monitor growth trends
- **Learning Effectiveness**: Track suggestion acceptance rates

### Alerting Thresholds
- Response time > 1000ms for 5 consecutive operations
- Cache hit rate < 30% over 1 hour period
- Memory usage growth > 50MB/hour
- Discovery failure rate > 20% over 10 minutes

---

**Enhanced MCP Discovery System v3.0**  
*Production Deployment Guide*  
*Phase 5: Deployment Documentation*