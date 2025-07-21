# 🧪 Alunai Clarity Testing Suite

**Comprehensive validation for all MCP tools, hooks, and automatic memory storage functionality**

## 📋 Testing Overview

This testing suite provides focused, manageable test documents for validating all aspects of the Alunai Clarity MCP server. Each document covers specific functionality areas and can be run independently or as part of a complete validation process.

## 🎯 Test Categories

### **Core Functionality**
- **[Core Memory Operations](core-memory-operations.md)** *(120 seconds)*
  - Memory storage, retrieval, and statistics
  - Hook validation for memory operations
  - Vector search functionality verification

- **[Hook System Validation](hook-system-validation.md)** *(120 seconds)*
  - Conversation lifecycle hooks (start/end detection)
  - Automatic session summary generation
  - Hook execution timing and performance

- **[Conversation Lifecycle Testing](conversation-lifecycle-hooks.md)** *(300 seconds)*
  - Detailed conversation start/end hook validation
  - Multi-topic session summary generation
  - End-to-end workflow verification

### **Advanced Features**
- **[Structured Thinking Integration](structured-thinking-integration.md)** *(240 seconds)*
  - 5-stage thinking process validation
  - Session relationship tracking  
  - Summary generation with confidence scoring
  - **🆕 Enhanced features:** Auto-progression, proactive suggestions, smart context integration

- **[Enhanced Structured Thinking Validation](enhanced-structured-thinking-validation.md)** *(420 seconds)*
  - **Real user scenarios** for enhanced structured thinking features
  - **Auto-stage progression** testing (like memory's auto-retrieval)
  - **Proactive thinking suggestions** validation (like memory's query suggestions)
  - **Enhanced component detection** with multi-dimensional analysis
  - **Smart context integration** with project intelligence

- **[AutoCode Intelligence](autocode-intelligence.md)** *(150 seconds)*
  - Project pattern detection
  - Intelligent command suggestions
  - Learning progression tracking

- **[Proactive Memory System](proactive-memory-system.md)** *(90 seconds)*
  - Automatic memory query suggestions
  - Context-aware memory retrieval
  - Proactive memory configuration

### **Performance & Monitoring**
- **[Performance Monitoring](performance-monitoring.md)** *(60 seconds)*
  - Qdrant performance analysis
  - Collection optimization
  - System health validation

## 🚀 Quick Start Testing

### **Essential Validation (5 minutes)**
Run these core tests to verify basic functionality:
1. [Core Memory Operations](core-memory-operations.md) - Memory storage & retrieval
2. [Hook System Validation](hook-system-validation.md) - Conversation end hooks

### **Complete Validation (20 minutes)**
Run all test suites for comprehensive validation:
1. **Core Functionality** (7 minutes) - Memory, hooks, conversation lifecycle
2. **Advanced Features** (7 minutes) - Structured thinking, AutoCode, proactive memory  
3. **Performance** (1 minute) - System monitoring and optimization
4. **Integration Testing** (5 minutes) - Cross-feature validation

## 📊 Success Criteria

### **Critical Requirements ✅**
- [ ] All 26 MCP tools execute successfully
- [ ] **Session summaries automatically generated on conversation end**
- [ ] Vector search returns results with similarity scores > 0.6
- [ ] Hook execution timing < 100ms per operation
- [ ] Memory storage and retrieval functional

### **Performance Requirements ✅**
- [ ] Search response time < 1 second
- [ ] Collection status: "green" 
- [ ] No memory leaks during extended testing
- [ ] Hook overhead < 10% of operation time

### **Feature Requirements ✅**
- [ ] Structured thinking 5-stage process complete
- [ ] AutoCode pattern detection functional
- [ ] Proactive memory suggestions generated
- [ ] Session relationship tracking working

## 🚨 Troubleshooting

### **Common Issues**

**Vector Search Issues:**
- If search returns empty results, wait 30 seconds for indexing
- Messages about "unreliable for small datasets" are NORMAL
- Always validate with actual queries, not just performance stats

**Hook System Issues:**
- Session summaries use memory type `test_session_summary` or similar
- Check memory stats for session summary count increases
- Manual conversation end triggers should show in audit logs

**Performance Issues:**
- `"search_functional": false` may appear due to HNSW optimization (normal)
- Focus on actual search results and `collection_status: "green"`
- Ignore indexed memory count warnings for small datasets

### **Failure Investigation**

If any test fails:

1. **Run Core Memory Operations first** - Validates basic functionality
2. **Check system health** - Run performance monitoring tests
3. **Validate hook execution** - Check hook system validation
4. **Review audit logs** - Look for error patterns in logs

## 📁 File Structure

```
tests/docs/
├── index.md                           # This overview document
├── core-memory-operations.md          # Memory CRUD and vector search
├── hook-system-validation.md          # Basic hook functionality  
├── conversation-lifecycle-hooks.md    # Detailed conversation hooks
├── structured-thinking-integration.md # 5-stage thinking process
├── autocode-intelligence.md           # Pattern detection & suggestions
├── proactive-memory-system.md         # Context-aware memory features
└── performance-monitoring.md          # System health and optimization
```

## 🎯 Testing Best Practices

### **Before Testing**
- Ensure MCP server is running and accessible
- Check that all dependencies are installed
- Verify Qdrant database is healthy and accessible
- Clear any existing test data if needed

### **During Testing**
- Run tests in suggested order (core → advanced → performance)
- Wait for vector indexing between memory operations (30 seconds)
- Monitor hook execution timing and success rates
- Document any failures with specific error messages

### **After Testing**  
- Review memory statistics for expected memory type distributions
- Verify session summaries were generated (check memory counts)
- Check that all hooks executed successfully
- Validate performance metrics meet success criteria

---

**💡 Pro Tip:** Start with the **Essential Validation** tests to quickly verify core functionality, then run specific feature tests based on your needs. The modular structure allows you to focus on areas of concern without running the entire suite.