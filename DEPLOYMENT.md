# 🚀 MCP Alunai Clarity - Deployment Guide

## Single Standardized Deployment Path

This guide provides the **single, official way** to deploy MCP Alunai Clarity using SQLite persistence.

## 📦 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/alun-ai/mcp-alunai-clarity.git
cd mcp-alunai-clarity

# Start the server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

### Option 2: Pre-built Image

```bash
# Create data directory
mkdir -p data/sqlite data/cache data/backups

# Run with Docker
docker run -d \
  --name mcp-alunai-clarity \
  -p 8000:8000 \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  ghcr.io/alun-ai/mcp-alunai-clarity:latest

# View logs
docker logs -f mcp-alunai-clarity
```

## 🔧 Configuration

The server uses SQLite for memory persistence with the following defaults:

- **Database**: `./data/sqlite/memory.db`
- **Configuration**: `./data/config.json` (auto-created)
- **Port**: `8000`
- **Health Check**: `http://localhost:8000/health`

## 🧪 Verification

### Health Check
```bash
curl http://localhost:8000/health
```

### Test Memory Storage
```bash
# Example MCP tool call (adjust for your MCP client)
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "store_memory",
      "arguments": {
        "content": "Test memory for deployment verification",
        "type": "episodic",
        "importance": 0.8
      }
    }
  }'
```

## 📊 Performance

**Expected Performance Metrics:**
- **Storage**: 2,500+ memories/second
- **Search**: <2ms average response time
- **Memory Usage**: 50-100MB
- **Startup**: 2-5 seconds

## 🗂️ Data Persistence

All data is persisted in the `./data/` directory:

```
data/
├── sqlite/
│   ├── memory.db       # Main SQLite database
│   ├── memory.db-shm   # Shared memory file (WAL mode)
│   └── memory.db-wal   # Write-ahead log
├── cache/              # Embedding model cache
├── backups/            # Database backups
└── config.json         # Runtime configuration
```

**Backup Strategy:**
```bash
# Backup the database
cp ./data/sqlite/memory.db ./backups/memory-$(date +%Y%m%d-%H%M%S).db

# Or backup everything
tar -czf backup-$(date +%Y%m%d-%H%M%S).tar.gz data/
```

## 🔨 Development

### Build Locally
```bash
# Build the image
docker-compose build

# Run with development settings
docker-compose up

# Access container for debugging
docker-compose exec mcp-alunai-clarity bash
```

### Run Tests
```bash
# Run the comprehensive test suite
docker-compose exec mcp-alunai-clarity python tests/unit/sqlite/test_suite_validation.py

# Run specific tests
docker-compose exec mcp-alunai-clarity python -m pytest tests/unit/sqlite/ -v
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

2. **Permission Errors**
   ```bash
   # Fix data directory permissions
   sudo chown -R $USER:$USER ./data
   ```

3. **Container Won't Start**
   ```bash
   # Check logs for errors
   docker-compose logs mcp-alunai-clarity
   
   # Restart with fresh build
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

4. **Database Locked**
   ```bash
   # Stop container and restart
   docker-compose restart mcp-alunai-clarity
   ```

## 🔧 Production Deployment

### Environment Variables
```bash
# Optional environment overrides
export MEMORY_CONFIG_PATH=/app/data/config.json
export SQLITE_DATA_PATH=/app/data/sqlite
export PYTHONPATH=/app
```

### Resource Limits
```yaml
# Add to docker-compose.yml service
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 512M
    reservations:
      cpus: '0.5'
      memory: 128M
```

### Monitoring
```bash
# Monitor container resources
docker stats mcp-alunai-clarity

# Check database size
ls -lh ./data/sqlite/memory.db

# Monitor memory usage
docker-compose exec mcp-alunai-clarity python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'CPU usage: {psutil.cpu_percent()}%')
"
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alun-ai/mcp-alunai-clarity/issues)
- **Performance**: See test results in `tests/unit/sqlite/SQLITE_TEST_SUITE_SUMMARY.md`
- **Architecture**: SQLite-based with 90% complexity reduction from previous Qdrant implementation

---

**This is the single, standardized deployment method.** All other deployment files have been removed to avoid confusion.