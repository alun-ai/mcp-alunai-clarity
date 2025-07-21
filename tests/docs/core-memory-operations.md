# 🧠 Core Memory Operations Test Suite

**Testing memory storage, retrieval, and vector search functionality with hook validation**

⏱️ **Estimated Duration:** 120 seconds

## 🎯 Test Objectives

Validate the fundamental memory operations that form the foundation of Alunai Clarity's cognitive capabilities:

- Memory storage with semantic indexing
- Vector search with similarity scoring
- Memory retrieval and filtering
- Hook execution for all memory operations
- Performance metrics and system statistics

## 🧪 Test Cases

### Test 1.1: Memory Storage with Hook Validation

**Purpose:** Verify memory storage works with proper indexing and hook triggering

**Test Command:**
```
Store this important team knowledge: "Our authentication system uses JWT tokens with 24-hour expiry, refresh tokens with 7-day expiry, and requires 2FA for admin users. All tokens are stored in httpOnly cookies for security."
```

**Expected Results:**
- ✅ Memory stored with semantic indexing
- ✅ Hook triggered with execution timing
- ✅ Automatic session tracking initiated
- ✅ Memory ID returned for reference

**Functional Validation:**
```
What authentication patterns do we use in our system?
```

**Success Criteria:**
- Must retrieve: JWT tokens, refresh tokens, 2FA requirements
- Similarity score must be > 0.6
- Memory content should match stored information

**⚠️ Important Testing Note:** 
If initial retrieval returns empty results, the memory may need time to index. Wait 30 seconds and retry the query. This is normal behavior for vector indexing.

---

### Test 1.2: Advanced Memory Retrieval

**Purpose:** Test semantic search capabilities and similarity scoring

**Test Command:**
```
Search for any security-related patterns we've discussed, including authentication and authorization
```

**Expected Results:**
- ✅ Semantic search finds auth-related memories
- ✅ Similarity scoring above 0.7
- ✅ Hook execution logged with timing
- ✅ Context-aware results with metadata

**Functional Validation:**
```
List all memories related to security with similarity scores
```

**Success Criteria:**
- Must show: Memories with similarity scores > 0.7 for security-related content
- Verify actual similarity scores are displayed, not just retrieved memories
- Should find memories from previous test (Test 1.1)

---

### Test 1.3: Memory Statistics and Performance

**Purpose:** Validate system health and performance monitoring

**Test Command:**
```
Show me comprehensive memory statistics and performance metrics
```

**Expected Results:**
- ✅ Total memory count, types breakdown
- ✅ Vector database performance stats
- ✅ Search time metrics (sub-millisecond)
- ✅ Hook trigger confirmation
- ✅ Collection optimization status

**Success Criteria:**
- Collection status should be "green"
- Memory count should include newly stored memories
- Performance stats should show reasonable search times
- Memory types should show diverse distribution

---

## 🎯 Success Criteria Summary

### **Critical Validations ✅**
- [ ] Memory storage successful with valid memory ID
- [ ] Vector search returns relevant results with similarity > 0.6
- [ ] Hook execution timing recorded for all operations
- [ ] Memory statistics show healthy system state

### **Performance Validations ✅**
- [ ] Search response time < 1 second
- [ ] Memory indexing completes within 30 seconds
- [ ] Collection status shows "green"
- [ ] No errors in hook execution logs

### **Functional Validations ✅**
- [ ] Stored memory is retrievable by semantic search
- [ ] Similarity scoring provides meaningful relevance ranking
- [ ] Memory metadata preserved and accessible
- [ ] Multiple memory operations work in sequence

## 🚨 Troubleshooting

### **Common Issues**

**Empty Search Results:**
- **Cause:** Vector indexing delay
- **Solution:** Wait 30 seconds after storage, then retry search
- **Prevention:** Allow indexing time between storage and retrieval tests

**Performance Warnings:**
- **Normal:** "indexed_memories count removed - unreliable for small datasets"
- **Normal:** "search_functional": false (due to HNSW optimization)  
- **Focus on:** Actual search results and collection_status: "green"

**Hook Execution Issues:**
- Check that hooks are properly registered in the system
- Verify hook execution timing is recorded
- Look for hook errors in system logs

### **Failure Investigation**

If memory operations fail:

1. **Check Basic Connectivity:**
   ```
   Run memory statistics to verify system is accessible
   ```

2. **Test Simple Storage:**
   ```
   Store a minimal test memory: "Test memory validation"
   ```

3. **Validate Search Functionality:**
   ```
   Search for: "test memory validation"
   Expect: Similarity score > 0.8 for exact match
   ```

4. **Performance Baseline:**
   ```
   Check collection health and optimization status
   Ignore warnings about small dataset indexing
   ```

## 💡 Testing Tips

- **Sequential Testing:** Run tests in order (1.1 → 1.2 → 1.3) as later tests depend on earlier ones
- **Timing:** Allow 30-45 seconds between storage and retrieval for indexing
- **Validation:** Always verify both successful execution AND functional results
- **Debugging:** If tests fail, check memory statistics first to validate system health

**🎯 Expected Total Test Time:** 2-3 minutes including indexing delays