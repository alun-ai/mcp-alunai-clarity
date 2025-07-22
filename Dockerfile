FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --user --no-warn-script-location -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Qdrant data directory
ENV QDRANT_DATA_PATH=/app/data/qdrant
ENV MEMORY_CONFIG_PATH=/app/data/config.json

# Create data directories
RUN mkdir -p /app/data/qdrant /app/data/backups

# Create default configuration
RUN echo '{\
  "qdrant": {\
    "path": "/app/data/qdrant",\
    "index_params": {\
      "m": 16,\
      "ef_construct": 200,\
      "full_scan_threshold": 10000\
    }\
  },\
  "embedding": {\
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",\
    "dimensions": 384,\
    "cache_dir": "/app/data/cache"\
  },\
  "alunai-clarity": {\
    "max_short_term_items": 1000,\
    "max_long_term_items": 10000,\
    "max_archival_items": 100000\
  }\
}' > /app/data/default_config.json

# Set permissions
RUN chmod +x setup.sh 2>/dev/null || true

# Volume for persistent data
VOLUME ["/app/data"]

# Create entrypoint script to handle permissions
RUN echo '#!/bin/bash\n\
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
        exec su-exec app-user python -m clarity "$@"\n\
    fi\n\
fi\n\
\n\
# Default: run as root\n\
exec python -m clarity "$@"\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Install su-exec for user switching
RUN apt-get update && apt-get install -y su-exec && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/app/entrypoint.sh"]