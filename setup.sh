#!/bin/bash

# Claude Memory MCP Setup Script

echo "Setting up Claude Memory MCP Server..."

# Create configuration directory
CONFIG_DIR="$HOME/.memory_mcp/config"
DATA_DIR="$HOME/.memory_mcp/data"

mkdir -p $CONFIG_DIR
mkdir -p $DATA_DIR

# Generate default configuration if it doesn't exist
if [ ! -f "$CONFIG_DIR/config.json" ]; then
    echo "Creating default configuration..."
    cat > "$CONFIG_DIR/config.json" << EOF
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": false
  },
  "memory": {
    "max_short_term_items": 100,
    "max_long_term_items": 1000,
    "max_archival_items": 10000,
    "consolidation_interval_hours": 24,
    "file_path": "$DATA_DIR/memory.json"
  },
  "embedding": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "cache_dir": "$HOME/.memory_mcp/cache"
  }
}
EOF
    echo "Default configuration created at $CONFIG_DIR/config.json"
fi

# Create default memory file if it doesn't exist
if [ ! -f "$DATA_DIR/memory.json" ]; then
    echo "Creating empty memory file..."
    cat > "$DATA_DIR/memory.json" << EOF
{
  "metadata": {
    "version": "1.0",
    "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "updated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "memory_stats": {
      "total_memories": 0,
      "active_memories": 0,
      "archived_memories": 0
    }
  },
  "memory_index": {
    "index_type": "hnsw",
    "index_parameters": {
      "m": 16,
      "ef_construction": 200,
      "ef": 50
    },
    "entries": {}
  },
  "short_term_memory": [],
  "long_term_memory": [],
  "archived_memory": [],
  "memory_schema": {
    "conversation": {
      "required_fields": ["role", "message"],
      "optional_fields": ["summary", "entities", "sentiment", "intent"]
    },
    "fact": {
      "required_fields": ["fact", "confidence"],
      "optional_fields": ["domain", "entities", "references"]
    },
    "document": {
      "required_fields": ["title", "text"],
      "optional_fields": ["summary", "chunks", "metadata"]
    },
    "code": {
      "required_fields": ["language", "code"],
      "optional_fields": ["description", "purpose", "dependencies"]
    }
  },
  "config": {
    "memory_management": {
      "max_short_term_memories": 100,
      "max_long_term_memories": 10000,
      "archival_threshold_days": 30,
      "deletion_threshold_days": 365,
      "importance_decay_rate": 0.01,
      "minimum_importance_threshold": 0.2
    },
    "retrieval": {
      "default_top_k": 5,
      "semantic_threshold": 0.75,
      "recency_weight": 0.3,
      "importance_weight": 0.7
    },
    "embedding": {
      "default_model": "sentence-transformers/all-MiniLM-L6-v2",
      "dimensions": 384,
      "batch_size": 8
    }
  }
}
EOF
    echo "Empty memory file created at $DATA_DIR/memory.json"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! You can now start the memory MCP server with: python -m memory_mcp"
