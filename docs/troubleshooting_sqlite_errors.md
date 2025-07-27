# Troubleshooting SQLite Errors in Qdrant Local Storage

## Issue Description

You may encounter SQLite errors like:
```
<method 'commit' of 'sqlite3.Connection' objects> returned NULL without setting an exception
```

This error comes from **Qdrant's internal SQLite persistence layer**, not from the application code itself.

## Root Cause

Qdrant's local client uses SQLite internally for metadata persistence. This can fail due to:

1. **File permission issues** with the SQLite database
2. **Concurrent access conflicts** when multiple processes access the same database
3. **Database corruption** from improper shutdowns
4. **Disk space issues**
5. **Docker volume mounting issues**

## Solutions

### Solution 1: Switch to Remote Qdrant (Recommended)

Use a dedicated Qdrant instance instead of local file storage:

```json
{
  "qdrant": {
    "url": "http://localhost:6333",
    "api_key": null,
    "prefer_grpc": true
  }
}
```

Start Qdrant with Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Solution 2: Fix Local Storage Issues

If you must use local storage:

1. **Ensure proper permissions:**
```bash
sudo chown -R $(id -u):$(id -g) /app/data/qdrant
chmod -R 755 /app/data/qdrant
```

2. **Use dedicated volume for Docker:**
```bash
docker run -v qdrant_data:/app/data/qdrant your-image
```

3. **Clear corrupted database:**
```bash
rm -rf /app/data/qdrant/*
# Restart the container - database will be recreated
```

### Solution 3: Temporary Workaround

The error is often transient. Simply retry the operation:
- Wait 5-10 seconds
- Try the store_memory operation again
- The system usually recovers automatically

## Prevention

1. **Use remote Qdrant** for production deployments
2. **Ensure proper Docker volume management**
3. **Monitor disk space** in the container
4. **Use graceful shutdowns** to prevent database corruption

## Enhanced Error Handling

The latest version (v1.17.0+) includes enhanced error handling that:
- Detects Qdrant SQLite issues specifically
- Provides clear user guidance
- Automatically suggests retry timing
- Logs detailed diagnostics for troubleshooting

## Configuration Example

Complete configuration to avoid SQLite issues:

```json
{
  "qdrant": {
    "url": "http://localhost:6333",
    "timeout": 30.0,
    "prefer_grpc": true,
    "max_retries": 3
  },
  "embedding": {
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384
  },
  "health_monitoring": {
    "enabled": true,
    "interval": 60.0
  }
}
```

This configuration uses remote Qdrant with health monitoring to prevent connection issues.