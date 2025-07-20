# Proactive AutoCode Configuration

This guide shows how to configure MCP AlunAI Memory for maximum proactive behavior, where AutoCode features automatically provide suggestions and insights without being explicitly asked.

## 🚀 Proactive Configuration

### Docker Configuration (Recommended)

Use this enhanced configuration for maximum proactive behavior:

```json
{
  "mcpServers": {
    "alunai-memory": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm", "-v", "./.claude/alunai-memory:/data",
        "-e", "MEMORY_FILE_PATH",
        "-e", "AUTOCODE_ENABLED",
        "-e", "AUTOCODE_COMMAND_LEARNING_ENABLED",
        "-e", "AUTOCODE_PATTERN_DETECTION_ENABLED", 
        "-e", "AUTOCODE_SESSION_ANALYSIS_ENABLED",
        "-e", "AUTOCODE_HISTORY_NAVIGATION_ENABLED",
        "-e", "AUTOCODE_AUTO_SCAN_PROJECTS",
        "-e", "AUTOCODE_TRACK_BASH_COMMANDS",
        "-e", "AUTOCODE_GENERATE_SESSION_SUMMARIES",
        "-e", "AUTOCODE_MIN_CONFIDENCE_THRESHOLD",
        "-e", "AUTOCODE_SIMILARITY_THRESHOLD",
        "ghcr.io/alun-ai/mcp-alunai-memory:latest"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/data/memory.json",
        "AUTOCODE_ENABLED": "true",
        "AUTOCODE_COMMAND_LEARNING_ENABLED": "true",
        "AUTOCODE_PATTERN_DETECTION_ENABLED": "true",
        "AUTOCODE_SESSION_ANALYSIS_ENABLED": "true", 
        "AUTOCODE_HISTORY_NAVIGATION_ENABLED": "true",
        "AUTOCODE_AUTO_SCAN_PROJECTS": "true",
        "AUTOCODE_TRACK_BASH_COMMANDS": "true",
        "AUTOCODE_GENERATE_SESSION_SUMMARIES": "true",
        "AUTOCODE_MIN_CONFIDENCE_THRESHOLD": "0.2",
        "AUTOCODE_SIMILARITY_THRESHOLD": "0.5"
      }
    }
  }
}
```

## ⚙️ Key Proactive Settings Explained

### Core Proactive Features
- **`AUTOCODE_AUTO_SCAN_PROJECTS: "true"`** - Automatically scans and analyzes project structure when Claude accesses files
- **`AUTOCODE_TRACK_BASH_COMMANDS: "true"`** - Automatically learns from every bash command you run
- **`AUTOCODE_GENERATE_SESSION_SUMMARIES: "true"`** - Automatically creates session summaries with insights

### Sensitivity Controls
- **`AUTOCODE_MIN_CONFIDENCE_THRESHOLD: "0.2"`** - Lower threshold (default: 0.3) means more suggestions
- **`AUTOCODE_SIMILARITY_THRESHOLD: "0.5"`** - Lower threshold (default: 0.6) means more pattern matches

## 🎯 What Proactive Behavior Looks Like

With proactive configuration enabled, you should see:

### Automatic Command Suggestions
When you mention wanting to do something, Claude might proactively suggest:
```
Based on this React project, I suggest: `npm test` (85% success rate)
For building Docker containers here, try: `docker build -t project-name .`
```

### Project Analysis
When you first work in a project directory:
```
I've detected a TypeScript React project with Jest testing and Docker deployment.
Key patterns: Component-based architecture, custom hooks, Material-UI styling.
```

### Pattern Recognition
When working with files:
```
I notice you're following a hexagonal architecture pattern here.
Previous similar components used the useCustomHook pattern.
```

### Session Insights
At conversation end:
```
Session Summary: Debugged authentication flow, updated 3 components.
Key learning: JWT refresh token implementation works best with axios interceptors.
```

## 🔧 Advanced Proactive Configuration

### Maximum Proactivity Settings
For even more aggressive suggestions, use these lower thresholds:

```json
{
  "env": {
    "AUTOCODE_MIN_CONFIDENCE_THRESHOLD": "0.1",
    "AUTOCODE_SIMILARITY_THRESHOLD": "0.3",
    "AUTOCODE_MAX_SUGGESTIONS": "8",
    "AUTOCODE_CONTEXT_WINDOW_DAYS": "60"
  }
}
```

### Selective Proactivity
Enable only specific proactive features:

```json
{
  "env": {
    "AUTOCODE_AUTO_SCAN_PROJECTS": "true",
    "AUTOCODE_TRACK_BASH_COMMANDS": "false",
    "AUTOCODE_GENERATE_SESSION_SUMMARIES": "true"
  }
}
```

## 🎭 CLAUDE.md for Proactive Behavior

Add this to your `CLAUDE.md` for maximum proactivity:

```markdown
# Proactive AutoCode Configuration

You have advanced AutoCode intelligence enabled. Be proactive in:

## Command Suggestions
- Suggest optimal commands when I mention tasks
- Include confidence scores and success rates
- Recommend alternatives when commands might fail

## Pattern Analysis  
- Automatically analyze project structure and patterns
- Point out architectural decisions and consistency
- Suggest improvements based on detected patterns

## Learning Insights
- Share insights about successful approaches
- Mention when current work relates to previous sessions
- Suggest workflow optimizations based on history

## Session Intelligence
- Summarize key decisions and learnings at conversation end
- Track progress on ongoing tasks across sessions
- Identify and suggest next steps for incomplete work

Use these capabilities proactively without being asked.
```

## 🧪 Testing Proactive Features

To verify proactive behavior is working:

1. **Start a new conversation** in a project directory
2. **Mention a task**: "I need to test this application"
3. **Expect proactive response**: Claude should suggest specific commands with confidence scores
4. **Work with files**: Claude should automatically analyze patterns and architecture
5. **End session**: Expect automatic session summary with insights

## 🔍 Monitoring Proactive Activity

Check what AutoCode is learning:

```bash
# View memory file to see automatic pattern storage
cat .claude/alunai-memory/memory.json | jq '.memories[] | select(.memory_type | contains("command_pattern", "project_pattern"))'

# Check Docker logs for AutoCode activity
docker logs $(docker ps | grep alunai-memory | awk '{print $1}') | grep AutoCode
```

## ⚡ Performance Considerations

- Lower confidence thresholds may increase processing time
- More proactive features mean more memory usage
- Consider project size when enabling auto-scanning

## 🎛️ Tuning Proactivity

If suggestions are too frequent:
- Increase `AUTOCODE_MIN_CONFIDENCE_THRESHOLD` to 0.4-0.5
- Reduce `AUTOCODE_MAX_SUGGESTIONS` to 3
- Set `AUTOCODE_AUTO_SCAN_PROJECTS` to false for large projects

If suggestions are too rare:
- Decrease `AUTOCODE_MIN_CONFIDENCE_THRESHOLD` to 0.1-0.15
- Increase `AUTOCODE_SIMILARITY_THRESHOLD` to 0.3-0.4
- Enable all proactive features