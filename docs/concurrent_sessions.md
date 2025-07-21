# Concurrent Claude Sessions Guide

This guide explains how multiple Claude instances can now run in the same project **sharing the same memory database** without conflicts.

## Problem (Solved!)

Previously, when multiple Claude instances tried to access the same local Qdrant database, they encountered file locking conflicts:

```
Storage folder ./qdrant_data is already accessed by another instance of Qdrant client. 
If you require concurrent access, use Qdrant server instead.
```

## Solution: Shared Database Access

Alunai Clarity now uses a **shared Qdrant client approach** that allows multiple Claude instances to safely access the same database concurrently.

## How It Works

The system now uses intelligent coordination to allow multiple Claude instances to share the same Qdrant database:

1. **Shared Client Management**: A singleton client manager coordinates access
2. **File-Based Coordination**: Uses lightweight file coordination to prevent conflicts  
3. **Automatic Sharing**: All instances automatically share memories and context
4. **Zero Configuration**: No environment variables or special setup needed

## Usage

Simply run Claude in multiple terminals - they'll all share the same memory database:

**Terminal 1:**
```bash
# Just run Claude normally
claude-desktop
```

**Terminal 2:**  
```bash
# Run Claude in another terminal
claude-desktop  
```

**Terminal 3:**
```bash
# Run Claude in yet another terminal  
claude-desktop
```

All instances share the same memories and can collaborate!

### Storage Structure

```
.claude/alunai-clarity/
├── config.json                 # Shared configuration
├── qdrant/                     # **Shared** vector database
│   └── .qdrant_coordination/   # Coordination files (auto-managed)
├── cache/                      # Shared embedding cache
└── backups/                    # Shared backups
```

## Benefits

- **Zero Setup Complexity**: No configuration changes needed
- **Shared Memories**: All instances share the same context and knowledge
- **No Lock Conflicts**: Intelligent coordination prevents database conflicts
- **True Collaboration**: Multiple Claude instances work with shared state
- **Seamless Experience**: Works exactly like single-instance usage

## Configuration

No special configuration needed! Use the standard Docker setup:

```json
{
  "mcpServers": {
    "alunai-clarity": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "./.claude/alunai-clarity:/app/data",
        "ghcr.io/alun-ai/mcp-alunai-clarity:latest"
      ],
      "type": "stdio"
    }
  }
}
```

## When to Use

- **Collaborative Work**: Multiple developers working on the same project
- **Multi-Modal Tasks**: Run different types of work simultaneously (coding + documentation)
- **Context Switching**: Switch between terminals without losing memory context
- **Parallel Development**: Work on different aspects of the same project simultaneously
- **Team Workflows**: Share knowledge and context across multiple sessions

## Technical Implementation

The shared client approach uses:

1. **Singleton Pattern**: Single Qdrant client instance per storage path
2. **File Coordination**: Lightweight coordination files prevent race conditions
3. **Process Safety**: Safe for multiple processes accessing same storage
4. **Graceful Fallback**: Falls back to wait-and-retry if coordination fails
5. **Automatic Cleanup**: Coordination files cleaned up automatically