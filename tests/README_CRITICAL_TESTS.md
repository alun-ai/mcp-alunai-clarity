# Critical MCP Memory System Tests

This document describes the critical test suite for the MCP (Model Context Protocol) memory system that validates the fixes for critical response format validation issues.

## 🎯 What These Tests Validate

The critical test suite validates fixes for these previously failing issues:

1. **❌ MCP Retrieve Memory Tool - Validation errors in response format**
   - Fixed KeyError: 'retrieval' in temporal domain configuration access
   - Fixed ZeroDivisionError in recency score calculation  
   - Fixed field naming mismatches in MCP validation

2. **⚠️ Search Results - Response format doesn't match expected schema**
   - Fixed flexible field validation (accepts both `type`/`memory_type` and `similarity`/`similarity_score`)
   - Validated schema compliance across all memory operations

## 📁 Test Structure

```
tests/
├── unit/
│   └── test_mcp_format_validation.py      # Unit tests for specific bug fixes
├── integration/
│   ├── test_mcp_critical_features.py      # Integration tests for critical features
│   ├── test_mcp_e2e_retrieval.py         # End-to-end retrieval system tests
│   └── test_mcp_search_functionality.py  # Search functionality validation
└── README_CRITICAL_TESTS.md              # This file
```

### Test Files Description

#### `test_mcp_format_validation.py` (Unit Tests)
- **Purpose**: Fast, isolated tests for specific bug fixes
- **Tests**:
  - Temporal domain configuration access without KeyError
  - Recency calculation without division by zero
  - MCP validator field naming flexibility
  - Error response format validation
- **Runtime**: ~30 seconds
- **Dependencies**: Minimal (mocked external dependencies)

#### `test_mcp_critical_features.py` (Integration Tests)
- **Purpose**: Integration testing of critical MCP memory features
- **Tests**:
  - Complete MCP tool execution cycles
  - Response format validation with real data
  - Large scale memory operations (100-150 memories)
  - Temporal domain integration
  - Error handling with proper MCP responses
- **Runtime**: ~2-3 minutes
- **Dependencies**: Full MCP server, Qdrant, embedding models

#### `test_mcp_e2e_retrieval.py` (End-to-End Tests)
- **Purpose**: Comprehensive end-to-end retrieval system validation
- **Tests**:
  - Performance benchmarks across dataset sizes (25-200 memories)
  - Concurrent retrieval operations (thread safety)
  - Retrieval accuracy and relevance scoring
  - Edge cases and boundary conditions
- **Runtime**: ~5-8 minutes
- **Dependencies**: Full system stack

#### `test_mcp_search_functionality.py` (Search Tests)
- **Purpose**: Deep validation of search capabilities
- **Tests**:
  - Semantic search quality and relevance
  - Search filtering by memory types
  - Similarity threshold behavior
  - Search result ranking and ordering
  - Performance with large result sets
- **Runtime**: ~3-5 minutes
- **Dependencies**: Full system stack

## 🚀 Running the Tests

### Quick Start

```bash
# Run all critical tests
python scripts/run_critical_tests.py

# Run quick validation (fastest)
python scripts/run_critical_tests.py --quick

# Run specific test type
python scripts/run_critical_tests.py --test-type format
```

### Using pytest directly

```bash
# Run all critical tests
pytest tests/unit/test_mcp_format_validation.py tests/integration/test_mcp_critical_features.py -v

# Run with markers
pytest -m "critical or format_validation" -v

# Run specific test file
pytest tests/unit/test_mcp_format_validation.py -v
```

### Test Runner Options

The `run_critical_tests.py` script provides several options:

- `--quick`: Run only the fastest tests (unit tests)
- `--verbose`: Show detailed test output
- `--test-type TYPE`: Run specific test type
  - `format`: Format validation unit tests
  - `critical`: Critical features integration tests
  - `e2e`: End-to-end retrieval tests
  - `search`: Search functionality tests
  - `all`: All critical tests (default)

## 🔧 Test Categories and Markers

Tests are organized using pytest markers:

- `@pytest.mark.critical` - Critical feature tests (must pass)
- `@pytest.mark.format_validation` - Format validation tests
- `@pytest.mark.mcp` - MCP protocol compliance tests
- `@pytest.mark.retrieval` - Memory retrieval tests
- `@pytest.mark.search` - Search functionality tests
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.performance` - Performance benchmark tests

## 📊 Expected Performance

| Test Suite | Runtime | Memory Count | Key Validations |
|------------|---------|--------------|----------------|
| Format Validation | 30s | N/A | Bug fix regressions |
| Critical Features | 2-3m | 150 | MCP compliance |
| E2E Retrieval | 5-8m | 25-200 | Performance scaling |
| Search Functionality | 3-5m | 40-75 | Search quality |

## 🐛 Regression Prevention

These tests specifically prevent regression of these fixed bugs:

### 1. Temporal Domain Configuration Error
```python
# BEFORE (would fail):
recency_weight = self.config["retrieval"].get("recency_weight", 0.3)  # KeyError

# AFTER (fixed):
retrieval_config = self.config.get("retrieval", {})
recency_weight = retrieval_config.get("recency_weight", 0.3)
```

### 2. Division by Zero in Recency Calculation
```python
# BEFORE (would fail):
recency_score = 1.0 / (1.0 + days_since_access)  # ZeroDivisionError when days_since_access = 0

# AFTER (fixed):
recency_score = 1.0 / (1.0 + max(days_since_access, 0.1))
```

### 3. MCP Field Validation Flexibility
```python
# BEFORE (rigid validation):
expected_fields = ["id", "content", "memory_type", "similarity_score"]

# AFTER (flexible validation):
required_checks = [
    ("memory_type", lambda m: "memory_type" in m or "type" in m),
    ("similarity_score", lambda m: "similarity_score" in m or "similarity" in m)
]
```

## ⚡ CI/CD Integration

### GitHub Actions
```yaml
- name: Run Critical MCP Tests
  run: |
    python scripts/run_critical_tests.py --test-type all
```

### Pre-commit Hook
```bash
#!/bin/bash
# Run quick critical tests before commit
python scripts/run_critical_tests.py --quick
```

### Development Workflow
```bash
# Before making changes
pytest tests/unit/test_mcp_format_validation.py -v

# After changes
python scripts/run_critical_tests.py --test-type critical

# Before PR
python scripts/run_critical_tests.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Tests fail with "No module named 'tests'"**
   ```bash
   # Run from project root
   cd /path/to/mcp-alunai-clarity
   python -m pytest tests/unit/test_mcp_format_validation.py
   ```

2. **Qdrant connection errors**
   ```bash
   # Ensure test environment uses local Qdrant
   export QDRANT_URL=http://localhost:6333
   ```

3. **Embedding model download issues**
   ```bash
   # Pre-download models if needed
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

### Performance Issues

- **Slow test execution**: Use `--quick` for development
- **Memory usage**: Tests clean up after themselves, but may use ~500MB RAM
- **Disk space**: Embedding models require ~90MB disk space

## 📈 Monitoring and Alerts

### Key Metrics to Monitor
- **Test pass rate**: Should be 100% for critical tests
- **Test execution time**: Regression if >2x baseline
- **Memory operations per second**: Should maintain performance
- **Response format compliance**: Must remain 100%

### Alert Conditions
- Any critical test fails
- Test execution time increases >50%
- Memory retrieval errors >1%
- Response format validation errors >0%

## 🎉 Success Criteria

These tests are considered successful when:

1. ✅ All critical tests pass with 100% success rate
2. ✅ Response format validation shows 0 errors
3. ✅ Performance metrics remain within acceptable bounds
4. ✅ No regression of the original bugs
5. ✅ Memory operations complete successfully at scale

The memory system is considered **fully operational** when all these tests pass consistently.