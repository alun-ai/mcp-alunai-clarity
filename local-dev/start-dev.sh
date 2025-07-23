#!/bin/bash

# Start Development Environment for Claude Code Integration
# This script sets up the complete infrastructure for real-world testing

set -e

echo "🚀 Starting Alunai Clarity Development Environment..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "📁 Project root: $PROJECT_ROOT"

# Create unified storage directory
STORAGE_DIR="$HOME/.claude/alunai-clarity"
echo "📦 Creating unified storage at: $STORAGE_DIR"
mkdir -p "$STORAGE_DIR"/{qdrant,app-data,pip-cache}

# Create default config if it doesn't exist
CONFIG_FILE="$STORAGE_DIR/app-data/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚙️ Creating default configuration..."
    cat > "$CONFIG_FILE" << 'EOF'
{
  "qdrant": {
    "url": "http://qdrant-server:6333",
    "prefer_grpc": false,
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
    "short_term_threshold": 0.3
  }
}
EOF
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f local-dev/docker-compose.dev.yml down --remove-orphans >/dev/null 2>&1 || true

# Build and start containers
echo "🔨 Building and starting containers..."
docker-compose -f local-dev/docker-compose.dev.yml up -d --build

# Wait for containers to be healthy
echo "⏳ Waiting for containers to be healthy..."
max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    qdrant_health=$(docker inspect qdrant-server-dev --format='{{.State.Health.Status}}' 2>/dev/null || echo "starting")
    app_health=$(docker inspect alunai-clarity-mcp-dev --format='{{.State.Health.Status}}' 2>/dev/null || echo "starting")
    
    if [ "$qdrant_health" = "healthy" ] && [ "$app_health" = "healthy" ]; then
        echo "✅ All containers are healthy!"
        break
    fi
    
    echo "   Attempt $((attempt + 1))/$max_attempts - Qdrant: $qdrant_health, App: $app_health"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Containers failed to become healthy within timeout"
    echo "📋 Container status:"
    docker-compose -f local-dev/docker-compose.dev.yml ps
    echo "📋 Recent logs:"
    docker-compose -f local-dev/docker-compose.dev.yml logs --tail=20
    exit 1
fi

# Test Qdrant connection
echo "🔍 Testing Qdrant connection..."
if curl -s http://localhost:6333/ >/dev/null; then
    echo "✅ Qdrant server is responding"
else
    echo "❌ Qdrant server is not responding"
    exit 1
fi

# Test application health
echo "🔍 Testing application health..."
app_test=$(docker exec alunai-clarity-mcp-dev python -c "
import sys
sys.path.append('/app')
try:
    from clarity.domains.manager import MemoryDomainManager
    print('✅ Application imports working')
except Exception as e:
    print(f'❌ Application import failed: {e}')
    sys.exit(1)
" 2>&1)

echo "$app_test"

if echo "$app_test" | grep -q "❌"; then
    echo "❌ Application health check failed"
    exit 1
fi

# Display status
echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📊 Container Status:"
docker ps --filter "name=alunai-clarity" --filter "name=qdrant" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🔗 Service URLs:"
echo "   Qdrant Web UI: http://localhost:6333/dashboard"
echo "   Application Debug: http://localhost:8000"

echo ""
echo "📋 Next Steps:"
echo "   1. Restart Claude Code for fresh MCP connection"
echo "   2. Verify MCP tools with: ListMcpResourcesTool"
echo "   3. Run infrastructure test from CLAUDE_DEV_WORKFLOW.md"
echo "   4. Start development using comprehensive testing patterns"

echo ""
echo "🛠️ Common Commands:"
echo "   View logs: docker logs -f alunai-clarity-mcp-dev"
echo "   Access container: docker exec -it alunai-clarity-mcp-dev bash"
echo "   Stop environment: docker-compose -f local-dev/docker-compose.dev.yml down"

echo ""
echo "📁 Unified Storage Location: $STORAGE_DIR"
echo "   Qdrant data: $STORAGE_DIR/qdrant"
echo "   App config: $STORAGE_DIR/app-data"
echo "   Python cache: $STORAGE_DIR/pip-cache"

echo ""
echo "✨ Ready for Claude Code integration and real-world testing!"