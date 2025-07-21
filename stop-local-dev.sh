#!/bin/bash
# Stop local development environment for Alunai Clarity MCP Server

echo "🛑 Stopping Alunai Clarity Local Development Environment"

# Stop Qdrant container
if docker ps | grep -q "alunai-clarity-qdrant"; then
    echo "📦 Stopping Qdrant database..."
    docker stop alunai-clarity-qdrant
    echo "✅ Qdrant stopped"
else
    echo "ℹ️  Qdrant container not running"
fi

# Optional: Clean up data (uncomment if you want to reset everything)
# echo "🧹 Cleaning up data..."
# rm -rf .claude/alunai-clarity/qdrant
# rm -rf .claude/alunai-clarity/cache
# rm -rf .claude/alunai-clarity/backups
# rm -f .claude/alunai-clarity/*.log
# echo "✅ Data cleaned up"

echo ""
echo "✅ Local development environment stopped"
echo ""
echo "💡 To start again, run: ./start-local-dev.sh"