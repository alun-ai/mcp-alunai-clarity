#!/bin/bash

echo "Testing Alunai Clarity Alpha Image..."

# Create a test config file
cat > test-config.json << EOF
{
  "qdrant": {
    "path": "/app/data/qdrant",
    "index_params": {
      "m": 16,
      "ef_construct": 200,
      "full_scan_threshold": 10000
    }
  },
  "embedding": {
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "cache_dir": "/app/data/cache"
  },
  "alunai-clarity": {
    "max_short_term_items": 1000,
    "max_long_term_items": 10000,
    "max_archival_items": 100000,
    "short_term_threshold": 0.7,
    "long_term_threshold": 0.3
  },
  "autocode": {
    "enabled": true,
    "auto_scan_projects": true,
    "track_bash_commands": true
  },
  "retrieval": {
    "recency_weight": 0.3,
    "importance_weight": 0.7
  }
}
EOF

echo "Created test configuration"

# Run the container with the test config
echo "Starting container in test mode..."
docker run --rm -it \
  -v "$(pwd)/test-config.json:/app/config.json" \
  -v "$(pwd)/test-data:/app/data" \
  --name alunai-clarity-test \
  alunai-clarity:alpha \
  --config /app/config.json

echo "Test completed. Check output above for any errors."