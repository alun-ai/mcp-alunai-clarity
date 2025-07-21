#!/bin/bash
# Local development startup script for Alunai Clarity MCP Server

set -e

echo "🚀 Starting Alunai Clarity Local Development Environment"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed. Please install Docker first."
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is required but not installed. Please install Python 3.8+ first."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "📦 Starting Qdrant vector database..."

# Start Qdrant in Docker with unified storage
docker run -d \
    --name alunai-clarity-qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v ./.claude/alunai-clarity/qdrant:/qdrant/storage \
    --rm \
    qdrant/qdrant:latest

echo "⏳ Waiting for Qdrant to start..."
sleep 5

# Check if Qdrant is accessible
if ! curl -s http://localhost:6333/health &> /dev/null; then
    echo "❌ Qdrant failed to start or is not accessible"
    docker stop alunai-clarity-qdrant 2>/dev/null || true
    exit 1
fi

echo "✅ Qdrant is running at http://localhost:6333"

# Create necessary directories (unified structure)
mkdir -p .claude/alunai-clarity/qdrant
mkdir -p .claude/alunai-clarity/cache
mkdir -p .claude/alunai-clarity/backups

# Check if virtual environment exists and dependencies are installed
if [ ! -d "venv" ] && [ ! -d "test_env" ]; then
    echo "📦 Setting up Python virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment (check which one exists)
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d "test_env" ]; then
    echo "🔧 Activating virtual environment (test_env)..."
    source test_env/bin/activate
fi

# Install/update dependencies
echo "📦 Installing/updating Python dependencies..."
pip install -e .

echo ""
echo "✅ Local development environment is ready!"
echo ""
echo "📝 Configuration (Unified Storage):"
echo "  - Qdrant: http://localhost:6333"
echo "  - Config file: .claude/alunai-clarity/config.json"
echo "  - Vector DB: .claude/alunai-clarity/qdrant/"
echo "  - Cache: .claude/alunai-clarity/cache/"
echo "  - Backups: .claude/alunai-clarity/backups/"
echo ""
echo "🎯 To test the MCP server manually:"
echo "  python -m clarity --config .claude/alunai-clarity/config.json --debug"
echo ""
echo "🛑 To stop Qdrant:"
echo "  docker stop alunai-clarity-qdrant"
echo ""
echo "💡 The .mcp.json file is configured for local development."
echo "   Claude Desktop should automatically use the local server now."