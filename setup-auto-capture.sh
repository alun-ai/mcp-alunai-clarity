#!/bin/bash
# Automatic hook setup for Claude Code auto-capture
# This script automatically creates hooks.json in the correct location for Claude Code

set -e

echo "🔧 Setting up Claude Code auto-capture hooks..."

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Create .claude directory structure  
CLAUDE_DIR="$PROJECT_DIR/.claude/alunai-clarity"
mkdir -p "$CLAUDE_DIR"

# Detect container name for hooks
CONTAINER_NAME=""
if docker ps --format "{{.Names}}" | grep -q "alunai-clarity-mcp-dev"; then
    CONTAINER_NAME="alunai-clarity-mcp-dev"
elif docker ps --format "{{.Names}}" | grep -q "alunai-clarity-mcp"; then
    CONTAINER_NAME="alunai-clarity-mcp"
elif docker ps --format "{{.Names}}" | grep -E "(alunai|clarity|mcp)" | head -1; then
    CONTAINER_NAME=$(docker ps --format "{{.Names}}" | grep -E "(alunai|clarity|mcp)" | head -1)
else
    echo "❌ No MCP container found running. Please start your MCP server first."
    exit 1
fi

echo "✅ Detected MCP container: $CONTAINER_NAME"

# Create hooks.json configuration
HOOKS_FILE="$CLAUDE_DIR/hooks.json"
cat > "$HOOKS_FILE" << EOF
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "docker exec $CONTAINER_NAME python /app/clarity/mcp/hook_analyzer.py --prompt-submit --prompt={prompt}",
            "timeout_ms": 2000,
            "continue_on_error": true,
            "modify_prompt": true
          }
        ]
      }
    ]
  },
  "metadata": {
    "created_by": "mcp-alunai-clarity",
    "version": "2.0.0",
    "description": "MCP auto-capture hooks using Docker container execution",
    "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")",
    "container_name": "$CONTAINER_NAME"
  }
}
EOF

echo "✅ Created hooks configuration at: $HOOKS_FILE"

# Test that the hook works
echo "🧪 Testing hook execution..."
if docker exec "$CONTAINER_NAME" python /app/clarity/mcp/hook_analyzer.py --prompt-submit --prompt="Remember this: Auto-capture setup test" > /dev/null 2>&1; then
    echo "✅ Hook execution test successful"
else
    echo "⚠️  Hook execution test failed, but hooks are configured"
fi

echo ""
echo "🎉 Auto-capture setup complete!"
echo ""
echo "📋 What this enables:"
echo "   • Type 'Remember this: [content]' in Claude Code"
echo "   • Memory will be automatically stored without explicit tool calls"
echo "   • Works immediately - no restart required"
echo ""
echo "🔍 Hook file created at: $HOOKS_FILE"
echo "🐳 Using container: $CONTAINER_NAME"