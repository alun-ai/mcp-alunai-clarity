# Quick Start: Automatic Memory Setup

This guide shows you how to set up MCP AlunAI Memory for automatic, seamless memory storage without having to say "remember" every time.

## 1. Configure Docker Integration

Add this to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "./.claude/alunai-memory:/data",
        "-e",
        "MEMORY_FILE_PATH=/data/memory.json",
        "-e",
        "AUTOCODE_AUTO_SCAN_PROJECTS=true",
        "-e",
        "AUTOCODE_TRACK_BASH_COMMANDS=true", 
        "-e",
        "AUTOCODE_GENERATE_SESSION_SUMMARIES=true",
        "-e",
        "AUTOCODE_MIN_CONFIDENCE_THRESHOLD=0.2",
        "-e",
        "AUTOCODE_SIMILARITY_THRESHOLD=0.5",
        "ghcr.io/alun-ai/mcp-alunai-memory:latest"
      ]
    }
  }
}
```

## 2. Enable Automatic Memory (Recommended Method)

### Option A: Using CLAUDE.md (Project-Specific)

Create a `CLAUDE.md` file in your project root:

```markdown
# Project Memory Configuration

You have persistent memory capabilities via the alunai-memory MCP server. 

## Automatic Memory Instructions

Automatically store important information about:
- User preferences and coding patterns
- Project architecture and decisions  
- Successful workflows and command patterns
- File modifications and patterns
- Errors and their solutions

Store memories without being explicitly asked when you encounter:
- New user preferences
- Important project details
- Successful command patterns
- Learning moments or insights

## Memory Storage Guidelines

- Use descriptive memory types: `user_preference`, `project_architecture`, `command_pattern`, `error_solution`
- Set appropriate importance levels (0.1-1.0)
- Include relevant context and metadata
- Prefer automatic storage over explicit "remember" requests
```

### Option B: Claude Desktop System Prompt (Global)

Add this to your Claude Desktop system prompt:

```
You have persistent memory capabilities via the alunai-memory MCP server. Automatically store important information about user preferences, project architecture, successful workflows, command patterns, file modifications, and error solutions. Store memories without being explicitly asked when you encounter new user preferences, important project details, successful command patterns, or learning moments.
```

## 3. Test Automatic Memory

1. **Start a conversation** in your project directory
2. **Mention a preference**: "I prefer TypeScript over JavaScript for this project"
3. **Check memory storage**: The information should be automatically stored
4. **Verify persistence**: Ask "What do you remember about my preferences?"

## 4. View Memory Activity

To see what's being stored automatically:

```
Show me my memory stats
```

Or check the memory file directly:
```bash
cat .claude/alunai-memory/memory.json | jq
```

## 5. Advanced: Memory Types

The system automatically uses these memory types:

- `user_preference` - Personal coding preferences and settings
- `project_architecture` - Technical decisions and project structure
- `command_pattern` - Successful command sequences and workflows  
- `error_solution` - Problems encountered and their solutions
- `session_summary` - Important conversation insights
- `bash_execution` - Command execution patterns and results

## Benefits

✅ **No manual "remember" commands needed**  
✅ **Project-isolated memory** (each project gets its own memory)  
✅ **Persistent across Claude sessions**  
✅ **Automatic learning** from your patterns  
✅ **Intelligent context** for future conversations  

## Troubleshooting

**Memory not storing automatically?**
- Ensure your system prompt includes automatic memory instructions
- Check that the MCP server is running: `docker ps | grep alunai-memory`
- Verify the memory file exists: `ls -la .claude/alunai-memory/`

**Want to disable automatic memory temporarily?**
- Remove the automatic memory instructions from your system prompt
- Memories will only be stored when explicitly requested

## Next Steps

- See [User Guide](user_guide.md) for advanced memory management
- Check [AutoCodeIndex Guide](autocode_guide.md) for intelligent code assistance
- Review [Claude Integration Guide](claude_integration.md) for detailed configuration