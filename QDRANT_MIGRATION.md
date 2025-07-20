# Qdrant Migration Guide

## 🚀 **High-Performance Vector Database Migration**

This guide helps you migrate from the legacy JSON storage to the new **Qdrant vector database** for **10-100x performance improvements**.

## 📊 **Performance Comparison**

| Metric | JSON Storage | Qdrant | Improvement |
|--------|--------------|--------|-------------|
| **Search Speed** | O(n) linear | O(log n) | **10-100x faster** |
| **Memory Usage** | Full file in RAM | Indexed access | **50-90% reduction** |
| **Scalability** | <10K memories | Millions | **100x+ capacity** |
| **Search Features** | Basic text | Vector + filters | **Advanced similarity** |
| **Concurrent Access** | File locks | Atomic operations | **Better reliability** |

## 🛠️ **Migration Process**

### **Step 1: Get Updated Version**

#### **Option A: Docker (Recommended)**
```bash
# Pull the latest Docker image with Qdrant support
docker pull ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1
```

#### **Option B: Python Package**
```bash
# Install the updated package with Qdrant support
pip install --upgrade mcp-alunai-memory

# Or install from source
pip install -e .
```

### **Step 2: Locate Your JSON Memory File**

Find your existing memory file (typically at):
```bash
# Common locations:
~/.memory_mcp/data/memory.json
./memory.json
/path/to/your/memory.json
```

### **Step 3: Run the Migration Command**

#### **Docker Migration (Recommended)**

```bash
# Basic migration using CLI in Docker container
docker run --entrypoint="python" \
           -v /path/to/your/memory.json:/tmp/memory.json \
           -v ./.claude/alunai-memory:/app/data \
           ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1 \
           -m memory_mcp.cli.import_json /tmp/memory.json

# Advanced migration options
docker run --entrypoint="python" \
           -v /path/to/your/memory.json:/tmp/memory.json \
           -v ./.claude/alunai-memory:/app/data \
           ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1 \
           -m memory_mcp.cli.import_json /tmp/memory.json --batch-size 500 --verbose

# Dry run to check what will be imported
docker run --entrypoint="python" \
           -v /path/to/your/memory.json:/tmp/memory.json \
           ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1 \
           -m memory_mcp.cli.import_json /tmp/memory.json --dry-run
```

#### **Direct Python Migration**

```bash
# Basic migration
python -m memory_mcp.cli.import_json /path/to/your/memory.json

# With custom configuration
python -m memory_mcp.cli.import_json /path/to/memory.json --config /path/to/config.json

# Large batch size for faster import
python -m memory_mcp.cli.import_json /path/to/your/memory.json --batch-size 500

# Dry run to check what will be imported
python -m memory_mcp.cli.import_json /path/to/your/memory.json --dry-run
```

### **Step 4: Verify Migration**

```bash
# The import command automatically verifies by default
# You can skip verification for faster import:
python -m memory_mcp.cli.import_json /path/to/memory.json --no-verify
```

### **Step 5: Update Configuration**

The system now uses Qdrant by default. Your new configuration should include:

```json
{
  "qdrant": {
    "path": "~/.memory_mcp/qdrant_data",
    "index_params": {
      "m": 16,
      "ef_construct": 200,
      "full_scan_threshold": 10000
    }
  },
  "embedding": {
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384
  }
}
```

## 🔧 **Migration Examples**

### **Example 1: Standard Migration**
```bash
$ python -m memory_mcp.cli.import_json ~/.memory_mcp/data/memory.json

2025-01-20 15:30:45 | INFO | Loading memories from: ~/.memory_mcp/data/memory.json
2025-01-20 15:30:46 | INFO | Extracted 1,247 memories from JSON data
2025-01-20 15:30:46 | INFO | Starting import of 1,247 memories in batches of 100
2025-01-20 15:30:48 | INFO | Import progress: 100.0% (1,247 successful, 0 failed)

============================================================
JSON MEMORY IMPORT SUMMARY
============================================================
Total memories processed: 1,247
Successful imports: 1,247
Failed imports: 0
Success rate: 100.0%
Import duration: 0:00:02.341

✅ Import verification PASSED
   - Total memories in Qdrant: 1,247
   - Indexed memories: 1,247
   - Search functionality: Working

🎉 JSON memory import completed successfully!
```

### **Example 2: Large Dataset Migration**
```bash
$ python -m memory_mcp.cli.import_json large_memory.json --batch-size 1000 --verbose

2025-01-20 15:35:12 | INFO | Loading memories from: large_memory.json
2025-01-20 15:35:15 | INFO | Extracted 50,000 memories from JSON data
2025-01-20 15:35:15 | INFO | Starting import of 50,000 memories in batches of 1000
2025-01-20 15:35:20 | INFO | Processing batch 1/50 (1,000 memories)
2025-01-20 15:35:25 | INFO | Import progress: 2.0% (1,000 successful, 0 failed)
...
2025-01-20 15:37:45 | INFO | Import progress: 100.0% (50,000 successful, 0 failed)

Total memories processed: 50,000
Successful imports: 50,000
Success rate: 100.0%
Import duration: 0:02:33.124
```

## 🎯 **Post-Migration Benefits**

### **Immediate Performance Improvements**

1. **Search Speed**: Memory queries now execute in 1-5ms instead of 100ms+
2. **Memory Efficiency**: Only relevant data loaded into RAM
3. **Advanced Filtering**: Search by memory type, importance, date ranges
4. **Concurrent Access**: Multiple processes can access memories safely

### **New MCP Tools Available**

After migration, you get access to new high-performance tools:

- `qdrant_performance_stats` - Get detailed performance metrics
- `optimize_qdrant_collection` - Optimize database for better performance
- Enhanced `retrieve_memory` with advanced filtering options

### **Performance Monitoring**

```python
# Check performance stats
{
  "total_memories": 50000,
  "indexed_memories": 50000,
  "indexing_ratio_percent": 100.0,
  "performance_rating": "excellent",
  "disk_size_mb": 245.7,
  "ram_size_mb": 89.3,
  "estimated_search_time_ms": 2.25
}
```

## 🐳 **Docker Migration**

### **Docker Compose Example**
```yaml
version: '3.8'
services:
  alunai-memory:
    image: ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1
    volumes:
      - ./qdrant_data:/app/data/qdrant
      - ./config:/app/config
    environment:
      - QDRANT_DATA_PATH=/app/data/qdrant
      - MEMORY_CONFIG_PATH=/app/config/memory_config.json
```

### **Docker Migration Commands**

#### **Single Command Migration**
```bash
# Replace /path/to/your/memory.json with your actual file path
docker run --entrypoint="python" \
           -v /path/to/your/memory.json:/tmp/memory.json \
           -v ./.claude/alunai-memory:/app/data \
           ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1 \
           -m memory_mcp.cli.import_json /tmp/memory.json
```

#### **Common File Locations**
```bash
# For JSON file in ~/.memory_mcp/data/memory.json
docker run --entrypoint="python" \
           -v ~/.memory_mcp/data/memory.json:/tmp/memory.json \
           -v ./.claude/alunai-memory:/app/data \
           ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1 \
           -m memory_mcp.cli.import_json /tmp/memory.json

# For JSON file in current directory
docker run --entrypoint="python" \
           -v ./memory.json:/tmp/memory.json \
           -v ./.claude/alunai-memory:/app/data \
           ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1 \
           -m memory_mcp.cli.import_json /tmp/memory.json

# For JSON file in Claude directory
docker run --entrypoint="python" \
           -v ./.claude/alunai-memory/memory.json:/tmp/memory.json \
           -v ./.claude/alunai-memory:/app/data \
           ghcr.io/alun-ai/mcp-alunai-memory:v0.3.1 \
           -m memory_mcp.cli.import_json /tmp/memory.json
```

## 🔍 **Troubleshooting**

### **Common Issues**

**1. "No memories found in JSON file"**
```bash
# Check file format - ensure it has the expected structure
python -m memory_mcp.cli.import_json memory.json --dry-run
```

**2. "Embedding model download slow"**
```bash
# Pre-download models to cache
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

**3. "Import fails with memory errors"**
```bash
# Use smaller batch sizes
python -m memory_mcp.cli.import_json memory.json --batch-size 50
```

### **Performance Optimization**

**For Large Datasets (>10K memories):**
```bash
# Optimize after import
python -c "
import asyncio
from memory_mcp.domains.persistence import QdrantPersistenceDomain
from memory_mcp.utils.config import load_config

async def optimize():
    config = load_config('config.json')
    persistence = QdrantPersistenceDomain(config)
    await persistence.initialize()
    await persistence.optimize_collection()

asyncio.run(optimize())
"
```

## 📈 **Expected Migration Times**

| Memory Count | Batch Size | Estimated Time | Peak RAM Usage |
|--------------|------------|----------------|----------------|
| 1K memories | 100 | 10-20 seconds | 500MB |
| 10K memories | 500 | 1-2 minutes | 1GB |
| 50K memories | 1000 | 5-8 minutes | 2GB |
| 100K memories | 1000 | 10-15 minutes | 4GB |

## ✅ **Migration Checklist**

- [ ] **Backup your JSON file** before migration
- [ ] **Install updated dependencies** (`qdrant-client>=1.7.0`)
- [ ] **Run migration command** with appropriate batch size
- [ ] **Verify import results** (should show 100% success rate)
- [ ] **Test search functionality** with sample queries
- [ ] **Update configuration** to use Qdrant settings
- [ ] **Restart MCP server** to use new storage backend
- [ ] **Monitor performance** with `qdrant_performance_stats`
- [ ] **Optimize collection** if needed with `optimize_qdrant_collection`
- [ ] **Archive old JSON file** (optional, after verification)

## 🎊 **Success Indicators**

After successful migration, you should see:

- ✅ **Sub-millisecond search times** (check with performance stats)
- ✅ **100% indexing ratio** (all memories properly indexed)
- ✅ **Working similarity search** (test with various queries)
- ✅ **Reduced memory usage** (only active data in RAM)
- ✅ **Better search relevance** (vector similarity vs text matching)

## 🚨 **Rollback Plan**

If you need to rollback:

1. **Keep your original JSON file** as backup
2. **Change configuration** back to JSON storage (not recommended)
3. **Or export from Qdrant** using backup tools (better approach)

The new Qdrant system provides **superior performance and features**, so rollback should only be considered for debugging purposes.

---

## 🎯 **Next Steps After Migration**

1. **Explore new search capabilities** with advanced filtering
2. **Monitor performance** using the new stats tools
3. **Consider optimizing** for your specific use case
4. **Update your applications** to leverage vector search features
5. **Set up regular backups** of the Qdrant data directory

**Welcome to high-performance memory storage!** 🚀