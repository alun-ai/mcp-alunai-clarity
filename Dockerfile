FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    git \
    curl \
    gettext-base \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install sqlite-vec extension
RUN git clone https://github.com/asg017/sqlite-vec.git /tmp/sqlite-vec \
    && cd /tmp/sqlite-vec \
    && make loadable \
    && mkdir -p /usr/local/lib \
    && cp dist/vec0.so /usr/local/lib/ \
    && rm -rf /tmp/sqlite-vec

RUN pip install --user --no-warn-script-location -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/lib/vec0.so /usr/local/lib/
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# SQLite data directory
ENV SQLITE_DATA_PATH=/app/data/sqlite
ENV MEMORY_CONFIG_PATH=/app/data/config.json

# Create data directories
RUN mkdir -p /app/data/sqlite /app/data/backups /app/data/cache

# Create production configuration optimized for SQLite
RUN echo '{\
  "server": {\
    "host": "127.0.0.1",\
    "port": 8000,\
    "debug": false\
  },\
  "sqlite": {\
    "path": "/app/data/sqlite/memory.db",\
    "wal_mode": true,\
    "timeout": 30.0,\
    "max_retries": 3,\
    "retry_backoff": 1.0,\
    "pragma_settings": {\
      "journal_mode": "WAL",\
      "synchronous": "NORMAL",\
      "cache_size": 10000,\
      "temp_store": "MEMORY",\
      "mmap_size": 268435456\
    }\
  },\
  "embedding": {\
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",\
    "dimensions": 384,\
    "cache_dir": "/app/data/cache",\
    "fast_model": "paraphrase-MiniLM-L3-v2"\
  },\
  "alunai-clarity": {\
    "max_short_term_items": 1000,\
    "max_long_term_items": 10000,\
    "max_archival_items": 100000,\
    "consolidation_interval_hours": 24,\
    "short_term_threshold": 0.3,\
    "legacy_file_path": "/app/data/legacy_memory.json"\
  },\
  "health_monitoring": {\
    "enabled": true,\
    "interval": 60.0,\
    "connection_recovery_timeout": 10.0\
  },\
  "retrieval": {\
    "default_top_k": 5,\
    "semantic_threshold": 0.75,\
    "recency_weight": 0.3,\
    "importance_weight": 0.7\
  },\
  "autocode": {\
    "enabled": true,\
    "auto_scan_projects": true,\
    "track_bash_commands": true,\
    "generate_session_summaries": true,\
    "command_learning": {\
      "enabled": true,\
      "min_confidence_threshold": 0.3,\
      "max_suggestions": 5,\
      "track_failures": true\
    },\
    "pattern_detection": {\
      "enabled": true,\
      "supported_languages": ["typescript", "javascript", "python", "rust", "go", "java"],\
      "max_scan_depth": 5,\
      "ignore_patterns": ["node_modules", ".git", "__pycache__", "target", "build", "dist"]\
    },\
    "session_analysis": {\
      "enabled": true,\
      "min_session_length": 3,\
      "track_architectural_decisions": true,\
      "extract_learning_patterns": true,\
      "identify_workflow_improvements": true,\
      "confidence_threshold": 0.6\
    },\
    "history_navigation": {\
      "enabled": true,\
      "similarity_threshold": 0.6,\
      "max_results": 10,\
      "context_window_days": 30,\
      "prioritize_recent": true,\
      "include_incomplete_sessions": true\
    },\
    "history_retention": {\
      "session_summaries_days": 90,\
      "command_patterns_days": 30,\
      "project_patterns_days": 365\
    },\
    "mcp_awareness": {\
      "enabled": true,\
      "index_tools_on_startup": true,\
      "proactive_suggestions": true,\
      "suggest_alternatives": true,\
      "context_aware_suggestions": true,\
      "error_resolution_suggestions": true,\
      "max_recent_suggestions": 10\
    }\
  },\
  "performance": {\
    "memory_cache_size_mb": 100,\
    "embedding_cache_size": 1000,\
    "connection_pool_size": 5,\
    "query_timeout_ms": 30000,\
    "bulk_insert_batch_size": 100\
  },\
  "logging": {\
    "level": "INFO",\
    "enable_performance_metrics": true,\
    "log_sql_operations": false,\
    "log_embedding_operations": false\
  }\
}' > /app/data/config.json

# Set permissions
RUN chmod +x setup.sh 2>/dev/null || true

# Volume for persistent data
VOLUME ["/app/data"]

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# SQLite-based MCP server entrypoint\n\
# Fix permissions for mounted volume\n\
if [ -d "/app/data" ]; then\n\
    # Get the host user ID from volume ownership\n\
    HOST_UID=$(stat -c %u /app/data 2>/dev/null || echo 0)\n\
    HOST_GID=$(stat -c %g /app/data 2>/dev/null || echo 0)\n\
    \n\
    # Only change permissions if not already root-owned\n\
    if [ "$HOST_UID" != "0" ] && [ "$HOST_GID" != "0" ]; then\n\
        # Create user if needed\n\
        if ! id -u app-user >/dev/null 2>&1; then\n\
            groupadd -g $HOST_GID app-group 2>/dev/null || true\n\
            useradd -u $HOST_UID -g $HOST_GID -s /bin/bash app-user 2>/dev/null || true\n\
        fi\n\
        \n\
        # Ensure data directory permissions\n\
        chown -R $HOST_UID:$HOST_GID /app/data\n\
        \n\
        # Run as the host user\n\
        exec gosu app-user python -m clarity "$@"\n\
    fi\n\
fi\n\
\n\
# Default: run as root\n\
exec python -m clarity "$@"\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Install gosu for user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/app/entrypoint.sh"]