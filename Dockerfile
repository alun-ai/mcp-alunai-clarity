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
ENV MEMORY_CONFIG_PATH=/app/config/memory_config.json

# Create data directories
RUN mkdir -p /app/data/qdrant /app/config /app/backups

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
  "alunai-memory": {\
    "max_short_term_items": 1000,\
    "max_long_term_items": 10000,\
    "max_archival_items": 100000\
  }\
}' > /app/config/memory_config.json

# Set permissions
RUN chmod +x setup.sh 2>/dev/null || true

# Volume for persistent data
VOLUME ["/app/data"]

ENTRYPOINT ["python", "-m", "memory_mcp"]