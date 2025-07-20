# MCP AlunAI Memory Project Configuration

You have persistent memory capabilities and proactive AutoCode intelligence via the alunai-memory MCP server.

## Automatic Memory + Proactive Behavior Instructions

**Automatically store** important information about:
- User preferences and coding patterns
- Project architecture and decisions  
- Successful workflows and command patterns
- File modifications and patterns
- Errors and their solutions

**Be proactive** with:
- Command suggestions with confidence scores when I mention tasks
- Pattern analysis when I work with files
- Architecture insights based on detected frameworks
- Session summaries with key learnings
- Workflow optimizations based on successful patterns

Store memories and provide proactive suggestions without being explicitly asked.

## Memory Storage Guidelines

- Use descriptive memory types: `user_preference`, `project_architecture`, `command_pattern`, `error_solution`
- Set appropriate importance levels (0.1-1.0)
- Include relevant context and metadata
- Prefer automatic storage over explicit "remember" requests

## Project Context

This is the **MCP AlunAI Memory** project - an advanced MCP server that provides persistent memory capabilities and intelligent code assistance through the AutoCodeIndex system.

### Key Technologies
- Python 3.8-3.12
- Docker containers
- Model Context Protocol (MCP)
- Vector embeddings with sentence-transformers
- FastAPI for API endpoints

### Development Patterns
- Use Docker for consistent deployment
- Store memory locally in `.claude/alunai-memory/` for project isolation
- Test with both local Python and Docker configurations
- Follow semantic versioning for releases

### Common Commands
```bash
# Docker build and test
docker build -t mcp-alunai-memory .
docker run -i --rm -v ./.claude/alunai-memory:/data -e MEMORY_FILE_PATH=/data/memory.json mcp-alunai-memory

# Local development
python -m memory_mcp --debug
pytest tests/

# Check memory file
cat .claude/alunai-memory/memory.json | jq
```

## AutoCodeIndex Features

Remember that this project includes:
- Intelligent command suggestions
- Project pattern recognition  
- Session history analysis
- Learning progression tracking
- Workflow optimization suggestions
- Automatic file and command tracking