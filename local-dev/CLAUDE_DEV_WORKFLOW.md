# Claude Development Workflow Guide

## 🎯 **Quick Start for Claude Code Integration**

This guide provides the complete workflow for local development and real-world testing with Claude Code integration.

### **Prerequisites**
- Docker and Docker Compose installed
- Claude Code CLI with MCP support
- Project repository cloned locally

### **1. Environment Setup**

```bash
# Navigate to project root
cd /path/to/mcp-alunai-clarity

# Start the complete development environment
./local-dev/start-dev.sh

# Verify containers are running
docker ps --filter "name=alunai-clarity" --filter "name=qdrant"
```

**Expected Output:**
```
NAMES                    STATUS                    PORTS
alunai-clarity-mcp-dev   Up X minutes (healthy)    0.0.0.0:8000->8000/tcp
qdrant-server-dev        Up X minutes (healthy)    0.0.0.0:6333-6334->6333-6334/tcp
```

### **2. MCP Connection Validation**

```bash
# Check MCP server configuration
cat .mcp.json

# Verify Claude Code hook configuration (inside container)
docker exec alunai-clarity-mcp-dev cat /root/.config/claude-code/hooks.json
```

**Hook Configuration Should Include:**
- PreToolUse hooks for bash/shell commands
- PostToolUse hooks for all tools
- UserPromptSubmit hooks for prompt enhancement

### **3. Real-World Testing Workflow**

#### **Step 1: Restart Claude Code**
When starting a new development session:
1. Close current Claude Code session
2. Restart Claude Code to get fresh MCP connection
3. Verify MCP tools are available: `ListMcpResourcesTool`

#### **Step 2: Validate Infrastructure**
```python
# Test inside Claude Code - this should work immediately
docker exec alunai-clarity-mcp-dev python -c "
import asyncio
import sys
sys.path.append('/app')
from clarity.domains.manager import MemoryDomainManager

async def quick_test():
    config = {
        'qdrant': {'url': 'http://qdrant-server:6333'},
        'embedding': {'default_model': 'sentence-transformers/all-MiniLM-L6-v2'},
        'alunai-clarity': {'short_term_threshold': 0.3}
    }
    manager = MemoryDomainManager(config)
    await manager.initialize()
    
    memory_id = await manager.store_memory(
        memory_type='structured_thinking',
        content={'test': 'Claude development workflow validation'},
        importance=0.8
    )
    print(f'✅ Infrastructure working: {memory_id}')

asyncio.run(quick_test())
"
```

#### **Step 3: Test Comprehensive Features**
Use prompts from the comprehensive testing guide:
```python
# Test structured thinking
"We need comprehensive testing for all features"

# Test MCP tool discovery
"I need to read a configuration file and analyze its contents"

# Test hook integration - any bash command will trigger hooks
ls -la local-dev/
```

### **4. Development Patterns**

#### **Memory Testing Pattern**
```python
async def test_feature():
    manager = MemoryDomainManager(config)
    await manager.initialize()
    
    # Store test memory
    memory_id = await manager.store_memory(
        memory_type='your_type',
        content={'your': 'data'},
        importance=0.8
    )
    
    # Retrieve and validate
    memories = await manager.retrieve_memories('your query', limit=5)
    print(f'Found {len(memories)} memories')
    
    return len(memories) > 0
```

#### **Hook Testing Pattern**
```bash
# Any bash command will trigger hooks automatically
docker exec alunai-clarity-mcp-dev python your_test.py

# Check if hooks captured the execution
docker logs alunai-clarity-mcp-dev | grep "hook"
```

#### **MCP Enhanced Testing Pattern**
```python
# Test with MCP context for enhanced functionality
memories = await manager.retrieve_memories(
    query='your search',
    mcp_enhanced=True,
    mcp_context={
        'current_tools': ['memory_manager', 'testing_framework'],
        'thinking_stage': 'your_stage',
        'user_intent': 'your_intent'
    }
)
```

### **5. Debugging Workflow**

#### **Container Logs**
```bash
# View real-time logs
docker logs -f alunai-clarity-mcp-dev
docker logs -f qdrant-server-dev

# Check specific errors
docker logs alunai-clarity-mcp-dev 2>&1 | grep -i error
```

#### **Memory System Debug**
```bash
# Access container for debugging
docker exec -it alunai-clarity-mcp-dev bash

# Check memory statistics
python -c "
import asyncio
from clarity.domains.manager import MemoryDomainManager
async def stats():
    manager = MemoryDomainManager({'qdrant': {'url': 'http://qdrant-server:6333'}})
    await manager.initialize()
    stats = await manager.get_memory_stats()
    print(f'Total memories: {stats.get(\"total_memories\", 0)}')
asyncio.run(stats())
"
```

#### **Qdrant Database Debug**
```bash
# Check Qdrant health
curl http://localhost:6333/

# View collections
curl http://localhost:6333/collections

# Count points in collection
curl http://localhost:6333/collections/memories
```

### **6. Common Issues and Solutions**

#### **Issue: MCP Tools Not Available**
```bash
# Solution: Restart Claude Code and verify .mcp.json
# Check if containers are running
docker ps --filter "name=alunai-clarity"

# Restart containers if needed
docker-compose -f local-dev/docker-compose.dev.yml restart
```

#### **Issue: Memory Storage Failures**
```bash
# Solution: Check Qdrant connection
docker exec alunai-clarity-mcp-dev python -c "
import requests
try:
    r = requests.get('http://qdrant-server:6333')
    print(f'Qdrant status: {r.status_code}')
except Exception as e:
    print(f'Qdrant connection failed: {e}')
"
```

#### **Issue: Hook System Not Working**
```bash
# Solution: Verify hook configuration
docker exec alunai-clarity-mcp-dev cat /root/.config/claude-code/hooks.json

# Check hook script exists
docker exec alunai-clarity-mcp-dev ls -la /app/clarity/mcp/hook_analyzer.py
```

### **7. Development Best Practices**

#### **Before Starting Development**
1. ✅ Start development environment: `./local-dev/start-dev.sh`
2. ✅ Verify containers are healthy
3. ✅ Restart Claude Code for fresh MCP connection
4. ✅ Run infrastructure validation test

#### **During Development**
1. ✅ Use structured prompts from comprehensive testing guide
2. ✅ Test both standard and MCP-enhanced functionality
3. ✅ Verify hook integration captures your tool usage
4. ✅ Check logs for any errors or warnings

#### **After Development Session**
1. ✅ Run comprehensive feature test to validate everything works
2. ✅ Check memory statistics to confirm data persistence
3. ✅ Document any new features or issues discovered

### **8. Key Files and Locations**

#### **Configuration Files**
- **MCP Config**: `.mcp.json` (Claude Code server configuration)
- **Docker Compose**: `local-dev/docker-compose.dev.yml`
- **Unified Storage**: `~/.claude/alunai-clarity/` (all persistent data)

#### **Testing Files**
- **Comprehensive Guide**: `COMPREHENSIVE_TESTING_IMPLEMENTATION.md`
- **Test Scripts**: `tests/` directory
- **Development Scripts**: `local-dev/` directory

#### **Key Containers**
- **Application**: `alunai-clarity-mcp-dev` (port 8000)
- **Database**: `qdrant-server-dev` (ports 6333-6334)

### **9. Success Metrics**

Your development environment is working correctly when:
- ✅ Both containers show "healthy" status
- ✅ Memory storage and retrieval works (>80% similarity scores)
- ✅ Hook system captures tool usage automatically
- ✅ MCP-enhanced retrieval returns contextual results
- ✅ Memory statistics show growing data (80+ memories)

### **10. Next Session Checklist**

When you return to development:
1. 📋 Run `./local-dev/start-dev.sh`
2. 📋 Restart Claude Code
3. 📋 Run infrastructure validation test
4. 📋 Proceed with feature development using established patterns

---

**🎯 This workflow ensures smooth Claude Code integration for real-world testing every time!**