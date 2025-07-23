# Infrastructure Transformation Guide

## 🏗️ **Complete Infrastructure Overview**

This document details the infrastructure transformation that enables seamless Claude Code integration with real-world testing.

### **Architecture Evolution**

#### **Before: File-Based (Problematic)**
```
┌─────────────────┐
│   Claude Code   │
│   (Instance 1)  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌──────────────────┐
│ Alunai Clarity  │───▶│ File-Based       │
│ Container       │    │ Qdrant Storage   │
└─────────────────┘    │ /app/data/qdrant │
                       └──────────────────┘
                               ▲
                               │ CONFLICT!
                               ▼
┌─────────────────┐    ┌──────────────────┐
│   Claude Code   │───▶│ Same File Path   │
│   (Instance 2)  │    │ /app/data/qdrant │
└─────────────────┘    └──────────────────┘

❌ Result: "Storage folder already accessed by another instance"
```

#### **After: Server-Based (Solution)**
```
┌─────────────────┐    ┌─────────────────┐
│   Claude Code   │───▶│ Alunai Clarity  │
│   (Instance 1)  │    │ Container       │
└─────────────────┘    └─────────┬───────┘
                                 │
┌─────────────────┐              │ HTTP API
│   Claude Code   │───▶┌─────────▼───────┐    ┌──────────────────┐
│   (Instance 2)  │    │ Alunai Clarity  │───▶│ Qdrant Server    │
└─────────────────┘    │ Container       │    │ (Standalone)     │
                       └─────────────────┘    │ Port 6333-6334   │
                                              └──────────────────┘

✅ Result: Multiple clients can connect simultaneously
```

### **Key Infrastructure Components**

#### **1. Qdrant Server Container**
```yaml
qdrant-server:
  image: qdrant/qdrant:v1.7.0
  container_name: qdrant-server-dev
  ports:
    - "6333:6333"  # HTTP API
    - "6334:6334"  # gRPC API
  volumes:
    - ~/.claude/alunai-clarity/qdrant:/qdrant/storage
  healthcheck:
    test: ["CMD-SHELL", "timeout 3 bash -c '</dev/tcp/localhost/6333' || exit 1"]
```

**Features:**
- Standalone vector database server
- HTTP and gRPC APIs for client connections
- Persistent storage in unified location
- Health checks for reliability

#### **2. Application Container**
```yaml
alunai-clarity-dev:
  build:
    context: ..
    dockerfile: local-dev/Dockerfile.dev
  container_name: alunai-clarity-mcp-dev
  depends_on:
    qdrant-server:
      condition: service_healthy
  volumes:
    - ..:/app                                          # Live code updates
    - ~/.claude/alunai-clarity/app-data:/app/data     # Persistent data
    - ~/.claude/alunai-clarity/pip-cache:/root/.cache/pip  # Package cache
  environment:
    - QDRANT_URL=http://qdrant-server:6333
```

**Features:**
- MCP server for Claude Code integration
- Live code mounting for development
- Server-based Qdrant configuration
- Unified persistent storage

#### **3. Unified Storage Structure**
```
~/.claude/alunai-clarity/
├── qdrant/          # Vector database storage
│   ├── collections/
│   ├── meta.json
│   └── storage/
├── app-data/        # Application configuration and cache
│   ├── config.json
│   └── cache/
└── pip-cache/       # Python package cache
    └── pip/
```

**Benefits:**
- Single location for all persistent data
- Easy backup and migration
- Clear separation of concerns
- Persistent across container rebuilds

### **Configuration Management**

#### **Server-Based Qdrant Configuration**
```json
{
  "qdrant": {
    "url": "http://qdrant-server:6333",
    "prefer_grpc": false,
    "index_params": {
      "m": 16,
      "ef_construct": 200,
      "full_scan_threshold": 10000
    }
  }
}
```

**Key Points:**
- Uses URL instead of file path
- Container-to-container communication
- Configurable performance parameters
- No file locking conflicts

#### **Environment Variable Override**
```bash
QDRANT_URL=http://qdrant-server:6333
MEMORY_CONFIG_PATH=/app/data/config.json
```

**Purpose:**
- Ensures server-based configuration
- Overrides any file-based defaults
- Provides configuration flexibility

### **Hook Integration System**

#### **Claude Code Hooks Configuration**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "bash|shell|exec|run_command",
        "hooks": [
          {
            "type": "command",
            "command": "python /app/clarity/mcp/hook_analyzer.py --pre-tool --tool={tool_name} --args={args}",
            "timeout_ms": 2000,
            "continue_on_error": true
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command", 
            "command": "python /app/clarity/mcp/hook_analyzer.py --post-tool --tool={tool_name} --result={result}",
            "timeout_ms": 3000,
            "continue_on_error": true
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python /app/clarity/mcp/hook_analyzer.py --prompt-submit --prompt={prompt}",
            "timeout_ms": 1500,
            "continue_on_error": true,
            "modify_prompt": true
          }
        ]
      }
    ]
  }
}
```

**Features:**
- Captures all Claude Code tool usage
- Analyzes commands before execution
- Learns from execution results
- Enhances prompts with context

### **Network Architecture**

#### **Container Communication**
```
Claude Code Host
       │
       │ MCP Protocol
       ▼
┌─────────────────┐ Port 8000
│ alunai-clarity- │
│ mcp-dev         │
└─────────┬───────┘
          │
          │ HTTP API
          ▼
┌─────────────────┐ Port 6333-6334
│ qdrant-server-  │
│ dev             │
└─────────────────┘
```

**Benefits:**
- Clean service separation
- Scalable architecture
- Standard protocols (HTTP, gRPC, MCP)
- Port isolation for security

### **Performance Optimizations**

#### **Connection Pooling**
```python
# Qdrant connection pool (min=2, max=10)
connection_pool = QdrantConnectionPool(
    min_connections=2,
    max_connections=10,
    url="http://qdrant-server:6333"
)
```

#### **Lazy Loading**
```python
# Embedding model loaded on first use
embedding_model = None  # Loaded lazily to save startup time

# Qdrant client initialized on first vector operation
client = None  # Initialized when needed
```

#### **Caching Strategy**
```python
# MCP memory caching for performance
cache_type = CacheType.MCP_WORKFLOW
ttl = 1800.0  # 30 minutes
```

### **Health Monitoring**

#### **Container Health Checks**
```yaml
# Qdrant health check
healthcheck:
  test: ["CMD-SHELL", "timeout 3 bash -c '</dev/tcp/localhost/6333' || exit 1"]
  interval: 10s
  timeout: 5s
  retries: 3

# Application health check  
healthcheck:
  test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
  interval: 30s
  timeout: 10s
  retries: 3
```

#### **Service Dependencies**
```yaml
depends_on:
  qdrant-server:
    condition: service_healthy
```

**Benefits:**
- Ensures proper startup order
- Prevents connection failures
- Automatic recovery mechanisms
- Reliable service orchestration

### **Development Workflow Integration**

#### **Live Code Updates**
```yaml
volumes:
  - ..:/app  # Mount project root for live updates
```

#### **Persistent Development Data**
```yaml
volumes:
  - ~/.claude/alunai-clarity/app-data:/app/data
  - ~/.claude/alunai-clarity/pip-cache:/root/.cache/pip
```

#### **Debug Access**
```bash
# Access running container
docker exec -it alunai-clarity-mcp-dev bash

# View real-time logs
docker logs -f alunai-clarity-mcp-dev
```

### **Security Considerations**

#### **Network Isolation**
- Services communicate via Docker network
- Ports exposed only where necessary
- Container-to-container communication preferred

#### **Data Persistence**
- User data isolated in home directory
- No sensitive data in container images
- Clear separation of code and data

#### **Hook Security**
- Hooks run in controlled container environment
- Timeout limits prevent hanging
- Error handling prevents disruption

### **Troubleshooting Infrastructure**

#### **Common Issues and Solutions**

1. **Container Health Failures**
   ```bash
   # Check container logs
   docker logs qdrant-server-dev
   docker logs alunai-clarity-mcp-dev
   
   # Verify network connectivity
   docker exec alunai-clarity-mcp-dev curl http://qdrant-server:6333/
   ```

2. **Memory Storage Failures**
   ```bash
   # Check Qdrant collections
   curl http://localhost:6333/collections
   
   # Verify configuration
   docker exec alunai-clarity-mcp-dev cat /app/data/config.json
   ```

3. **Hook Integration Issues**
   ```bash
   # Verify hook configuration
   docker exec alunai-clarity-mcp-dev cat /root/.config/claude-code/hooks.json
   
   # Test hook script
   docker exec alunai-clarity-mcp-dev python /app/clarity/mcp/hook_analyzer.py --help
   ```

### **Migration and Backup**

#### **Data Backup**
```bash
# Backup all persistent data
tar -czf alunai-clarity-backup.tar.gz ~/.claude/alunai-clarity/
```

#### **Data Migration**
```bash
# Restore from backup
tar -xzf alunai-clarity-backup.tar.gz -C ~/
```

#### **Container Recreation**
```bash
# Completely rebuild environment
docker-compose -f local-dev/docker-compose.dev.yml down --volumes
./local-dev/start-dev.sh
```

---

**🎯 This infrastructure enables reliable, scalable, and maintainable Claude Code integration for real-world development.**