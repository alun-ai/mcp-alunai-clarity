# SQLite Memory Persistence Tests

This directory contains tests for the SQLite-based memory persistence system.

## Overview

The SQLite persistence system provides a simple, reliable alternative to complex vector database infrastructure. It uses SQLite with optional sqlite-vec extension for vector operations.

## Test Files

- `sqlite_persistence_test.py` - Comprehensive validation of SQLite memory operations
- `sqlite_vec_simple.py` - SQLite persistence domain implementation  
- `test_data_generator.py` - Test data generation utilities

## Running Tests

```bash
# Run comprehensive SQLite validation
python tests/migration/sqlite_persistence_test.py
```

## Test Coverage

The test suite validates:

- ✅ **Memory Storage**: All memory types and tiers
- ✅ **Vector Search**: Semantic similarity with 384-dimensional embeddings
- ✅ **Filtering**: Memory type, tier, importance, and metadata filtering
- ✅ **CRUD Operations**: Create, read, update, delete operations
- ✅ **Performance**: Storage and search performance metrics
- ✅ **Statistics**: Database monitoring and reporting

## Memory Types Supported

- `structured_thinking` - Analysis and reasoning
- `episodic` - Event and experience memories
- `procedural` - Step-by-step processes
- `semantic` - Conceptual knowledge

## Memory Tiers Supported

- `short_term` - Recent, frequently accessed
- `long_term` - Important, persistent storage
- `archival` - Historical, infrequently accessed
- `system` - Configuration and operational data

## Benefits of SQLite Approach

- **No External Dependencies**: Single file database
- **ACID Transactions**: Built-in data integrity
- **Proven Reliability**: Battle-tested technology
- **Simple Deployment**: No server management required
- **Extensive Tooling**: Rich ecosystem of tools and utilities