# Alunai Clarity Test Suite

Comprehensive test suite for Alunai Clarity MCP server covering unit tests, integration tests, performance tests, and deployment validation.

## Test Structure

```
tests/
├── conftest.py                    # Test configuration and fixtures
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_memory_operations.py  # Core memory CRUD operations
│   ├── test_domain_managers.py    # Domain-specific functionality
│   ├── test_mcp_server.py         # MCP server and tools
│   ├── test_autocode.py           # AutoCode functionality
│   └── test_config.py             # Configuration and schema validation
├── integration/                   # Integration tests (end-to-end)
│   └── test_end_to_end.py         # Complete workflow testing
├── performance/                   # Performance and scalability tests
│   └── test_qdrant_performance.py # Database performance testing
├── deployment/                    # Deployment and infrastructure tests
│   └── test_docker.py             # Docker containerization testing
└── README.md                      # This file
```

## Test Categories

### 🔧 Unit Tests (`tests/unit/`)

Fast, isolated tests that validate individual components:

- **Memory Operations**: Store, retrieve, update, delete operations
- **Domain Managers**: Persistence, episodic, semantic, temporal, and AutoCode domains
- **MCP Server**: Tool registration, request handling, response formatting
- **AutoCode**: Pattern detection, command learning, session analysis
- **Configuration**: Config loading, validation, schema compliance

### 🔗 Integration Tests (`tests/integration/`)

End-to-end tests that validate complete workflows:

- **Memory Lifecycle**: Complete CRUD workflows with ranking and temporal adjustments
- **MCP Tool Workflows**: Full tool execution pipelines
- **AutoCode Workflows**: Pattern detection to storage workflows
- **Configuration Loading**: Real config file loading and validation

### ⚡ Performance Tests (`tests/performance/`)

Performance and scalability validation:

- **Qdrant Operations**: Storage, search, and concurrent operation performance
- **Embedding Generation**: Vector embedding performance testing
- **Memory Manager**: Mixed workload performance testing
- **Scalability**: Large-scale operation testing (10K+ memories)

### 🐳 Deployment Tests (`tests/deployment/`)

Infrastructure and deployment validation:

- **Docker**: Build, run, networking, and security testing
- **Container**: Startup time, memory usage, port exposure
- **Configuration**: Environment variables, volume mounting

## Running Tests

### Quick Test Run
```bash
# Run all unit tests (fast)
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_memory_operations.py -v

# Run tests with specific marker
pytest -m unit -v
```

### Test Categories
```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# Slow tests (performance + integration)
pytest -m slow

# All tests except slow ones
pytest -m "not slow"
```

### Coverage Testing
```bash
# Run with coverage report
pytest --cov=clarity --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html
```

### Docker Tests
```bash
# Run Docker-related tests (requires Docker)
pytest -m requires_docker

# Skip Docker tests
pytest -m "not requires_docker"
```

## Test Markers

- `unit`: Fast, isolated unit tests
- `integration`: Integration tests with external dependencies
- `performance`: Performance and benchmarking tests
- `slow`: Slow-running tests (performance + large integration tests)
- `requires_qdrant`: Tests requiring Qdrant vector database
- `requires_embedding`: Tests requiring embedding model downloads
- `requires_docker`: Tests requiring Docker to be available

## Test Configuration

### Environment Variables
```bash
# Set test environment
export PYTEST_CURRENT_TEST=1

# Skip slow tests by default
export PYTEST_DISABLE_SLOW=1

# Custom test data directory
export TEST_DATA_DIR=/tmp/clarity_test_data
```

### Fixtures

Key fixtures available in all tests:

- `test_config`: Test configuration dictionary
- `temp_config_file`: Temporary config file path
- `temp_data_dir`: Temporary data directory
- `mock_domain_manager`: Mocked memory domain manager
- `mock_persistence_domain`: Mocked persistence domain
- `sample_memory_data`: Sample memory objects for testing
- `performance_test_data`: Generator for performance test data

## Performance Benchmarks

Expected performance targets:

### Memory Operations
- **Storage**: >100 memories/second
- **Search**: >50 searches/second, <50ms average
- **Concurrent**: >200 operations/second mixed workload

### Embedding Operations
- **Generation**: >100 embeddings/second, <50ms average
- **Similarity**: >10,000 calculations/second, <100μs average

### Container Performance
- **Startup**: <30 seconds
- **Memory**: <1GB for basic operation
- **Build**: <10 minutes

## Writing New Tests

### Unit Test Example
```python
@pytest.mark.unit
class TestMyFeature:
    @pytest.mark.asyncio
    async def test_my_async_function(self, mock_domain_manager):
        # Arrange
        test_input = {"key": "value"}
        mock_domain_manager.my_method.return_value = "expected_result"
        
        # Act
        result = await my_async_function(test_input)
        
        # Assert
        assert result == "expected_result"
        mock_domain_manager.my_method.assert_called_once_with(test_input)
```

### Integration Test Example
```python
@pytest.mark.integration
class TestMyWorkflow:
    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_config, temp_data_dir):
        # Test complete end-to-end workflow
        manager = MemoryDomainManager(test_config)
        await manager.initialize()
        
        # Test workflow steps
        memory_id = await manager.store_memory(...)
        results = await manager.retrieve_memories(...)
        
        assert len(results) > 0
```

### Performance Test Example
```python
@pytest.mark.performance
class TestMyPerformance:
    @pytest.mark.asyncio
    async def test_operation_performance(self, performance_test_data):
        start_time = time.time()
        
        # Perform operations
        for data in performance_test_data(1000):
            await my_operation(data)
        
        elapsed = time.time() - start_time
        ops_per_second = 1000 / elapsed
        
        assert ops_per_second > 100, f"Too slow: {ops_per_second:.2f} ops/sec"
```

## Continuous Integration

### GitHub Actions
Tests are automatically run on:
- Pull requests
- Pushes to main branch
- Weekly scheduled runs

Test matrix includes:
- Python 3.8, 3.9, 3.10, 3.11, 3.12
- Ubuntu, macOS, Windows
- With and without optional dependencies

### Test Reports
- Coverage reports uploaded to Codecov
- Performance benchmarks tracked over time
- Docker image vulnerability scanning

## Troubleshooting

### Common Issues

**Tests fail with "QdrantClient not found"**
```bash
# Install with Qdrant support
pip install "qdrant-client>=1.7.0"
```

**Embedding tests fail with download errors**
```bash
# Skip embedding tests
pytest -m "not requires_embedding"
```

**Docker tests fail**
```bash
# Check Docker is running
docker version

# Skip Docker tests
pytest -m "not requires_docker"
```

### Debug Mode
```bash
# Run with verbose output and no capture
pytest -v -s --tb=long

# Run single test with debugging
pytest tests/unit/test_memory_operations.py::TestMemoryStorage::test_store_memory -v -s
```

### Performance Issues
```bash
# Run only fast tests
pytest -m "unit and not slow"

# Profile test execution
pytest --profile-svg

# Memory profiling
pytest --memprof
```

## Contributing

When adding new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Include all test types**: unit, integration, performance if applicable
3. **Use appropriate markers** to categorize tests
4. **Mock external dependencies** in unit tests
5. **Test error conditions** and edge cases
6. **Update documentation** including this README if needed

Ensure all tests pass before submitting PRs:
```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=clarity --cov-fail-under=80
```