#!/bin/bash
# Sync hooks.json from mounted MCP container data directory to current project

# Source: mounted container data directory (where hooks.json is created)
SOURCE_HOOKS="/Users/chadupton/Documents/Github/alun-ai/alun.ai/.claude/alunai-clarity/hooks.json"

# Destination: current project's .claude directory (where Claude Code looks)
DEST_DIR="./.claude/alunai-clarity"
DEST_HOOKS="$DEST_DIR/hooks.json"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Check if source hooks.json exists
if [ -f "$SOURCE_HOOKS" ]; then
    # Copy hooks.json to current project
    cp "$SOURCE_HOOKS" "$DEST_HOOKS"
    echo "✅ Synced hooks.json to current project"
    echo "Source: $SOURCE_HOOKS"
    echo "Destination: $DEST_HOOKS"
    
    # Show the container ID from the hooks
    if command -v jq >/dev/null 2>&1; then
        CONTAINER_ID=$(jq -r '.metadata.container_name' "$DEST_HOOKS")
        echo "Container: $CONTAINER_ID"
    fi
else
    echo "❌ Source hooks.json not found at: $SOURCE_HOOKS"
    echo "Make sure your MCP container is running and has created hooks.json"
    exit 1
fi