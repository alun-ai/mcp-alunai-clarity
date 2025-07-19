# AutoCodeIndex Guide

The AutoCodeIndex system is an intelligent code project assistance platform integrated with MCP Persistent Memory. It provides Claude with sophisticated understanding of code patterns, project structures, command usage, and session history to enhance development workflows.

## Overview

AutoCodeIndex transforms Claude into an intelligent coding assistant that learns from your development patterns and provides contextual assistance. It operates completely automatically through MCP hooks, requiring no manual intervention while building a comprehensive understanding of your coding practices.

## Features

### 🧠 Intelligent Command Assistance

AutoCodeIndex learns from your command usage patterns and provides smart suggestions:

- **Context-Aware Suggestions**: Commands recommended based on project type, platform, and current context
- **Success Rate Tracking**: Monitors which commands work best in different situations
- **Retry Pattern Detection**: Automatically identifies and suggests fixes for failed command patterns
- **Platform Intelligence**: Provides platform-specific optimizations for macOS, Linux, and Windows

**Example Usage:**
```
You: "I need to delete this directory"
Claude: Based on your platform and past usage, I suggest: `rm -rf directory_name`
This command has a 95% success rate in similar contexts.
```

### 🔍 Project Pattern Recognition

Automatically detects and indexes project patterns across multiple languages and frameworks:

**Supported Frameworks:**
- **Frontend**: React, Vue.js, Angular, Svelte
- **Backend**: Django, Flask, FastAPI, Express.js, Rails
- **Languages**: TypeScript, JavaScript, Python, Rust, Go, Java, PHP, Ruby
- **Build Tools**: Webpack, Vite, Parcel, Rollup, esbuild

**Pattern Detection:**
- **Architectural Patterns**: MVC, component-based, layered, microservices, hexagonal
- **Naming Conventions**: camelCase, snake_case, kebab-case consistency analysis
- **Component Relationships**: Dependency mapping and import/export analysis
- **Testing Patterns**: Jest, Pytest, Cargo test, and framework-specific testing

**Example:**
```
You: "Analyze this project structure"
Claude: I've detected a React + TypeScript project with:
- Component-based architecture
- camelCase naming convention (98% consistent)
- Material-UI design system
- Jest testing setup
- Recommended next steps: Add Storybook for component documentation
```

### 📈 Session History & Context

Advanced conversation analysis provides intelligent session navigation:

**Session Analysis:**
- **Task Tracking**: Automatically identifies completed, in-progress, and failed tasks
- **Learning Moments**: Captures insights, discoveries, and knowledge gaps
- **Architectural Decisions**: Records design choices and their reasoning
- **File Modification Tracking**: Monitors code changes and their context

**Context Continuation:**
- **Seamless Resumption**: Pick up exactly where you left off in previous sessions
- **Relevant Context**: Automatically provides background from similar past work
- **Pattern Reuse**: Suggests approaches that worked well in similar situations

**Example:**
```
You: "Continue working on the user authentication system"
Claude: Based on our previous session 3 days ago, you were implementing JWT tokens 
with refresh logic. I see you completed the login endpoint and were working on 
the password reset flow. The approach using bcrypt and nodemailer was successful.
Shall we continue with the email verification component?
```

### 📊 Learning Progression Tracking

Monitor your development skills and knowledge growth over time:

**Tracked Metrics:**
- **Technology Mastery**: Progress on specific languages, frameworks, and tools
- **Problem-Solving Patterns**: Evolution of your debugging and implementation approaches  
- **Knowledge Gaps**: Areas where you consistently need assistance
- **Skill Development**: Improvement trends across different technologies

**Example:**
```
You: "Show my React learning progression"
Claude: Over the past 3 months, I've tracked significant growth:
- Advanced from functional components to custom hooks (mastered useCallback, useMemo)
- Performance optimization skills improved 40% (React.memo, code splitting)
- State management evolved from useState to Context API to Zustand
- Current focus areas: React Query, testing with React Testing Library
- Suggestion: Ready to explore advanced patterns like render props and compound components
```

### ⚡ Workflow Optimization

Continuous improvement suggestions based on historical patterns:

**Optimization Areas:**
- **Command Efficiency**: Streamlined command sequences for common tasks
- **Development Workflows**: Improved development, testing, and deployment processes
- **Tool Usage**: Better utilization of development tools and shortcuts
- **Architecture Decisions**: Patterns that scale well based on past project success

**Example:**
```
Claude: I've noticed you often run `npm install` followed by `npm run dev`. 
Based on your project patterns, I suggest creating this alias:
`alias quickstart="npm install && npm run dev"`
This could save you 15-20 seconds per startup based on your usage patterns.
```

## Available Tools

AutoCodeIndex provides 7 MCP tools for intelligent assistance:

### 1. suggest_command
Get intelligent command suggestions based on intent and context.

**Parameters:**
- `intent` (required): What you want to accomplish
- `context` (optional): Current project context

**Example:**
```json
{
  "intent": "run tests for React components",
  "context": {
    "project_type": "react",
    "testing_framework": "jest"
  }
}
```

### 2. get_project_patterns
Analyze and retrieve detected patterns for a project.

**Parameters:**
- `project_path` (required): Path to the project
- `pattern_types` (optional): Specific pattern types to retrieve

**Example:**
```json
{
  "project_path": "/path/to/project",
  "pattern_types": ["architectural", "naming", "testing"]
}
```

### 3. find_similar_sessions
Find previous sessions similar to your current work.

**Parameters:**
- `query` (required): Description of current task or context
- `context` (optional): Additional context
- `time_range_days` (optional): Limit search to recent days

**Example:**
```json
{
  "query": "implementing user authentication with JWT",
  "context": {
    "technologies": ["react", "node.js", "mongodb"]
  },
  "time_range_days": 30
}
```

### 4. get_continuation_context
Get relevant context for continuing work on a task.

**Parameters:**
- `current_task` (required): Description of current task
- `project_context` (optional): Current project information

**Example:**
```json
{
  "current_task": "add password reset functionality",
  "project_context": {
    "project_type": "web_app",
    "backend": "express"
  }
}
```

### 5. suggest_workflow_optimizations
Get suggestions for improving your development workflow.

**Parameters:**
- `current_workflow` (required): List of current workflow steps
- `session_context` (optional): Current session context

**Example:**
```json
{
  "current_workflow": [
    "git pull origin main",
    "npm install", 
    "npm run dev",
    "open browser",
    "start coding"
  ]
}
```

### 6. get_learning_progression
Track your learning progress on specific topics.

**Parameters:**
- `topic` (required): Technology or concept to track
- `time_range_days` (optional): Time range for analysis (default: 180)

**Example:**
```json
{
  "topic": "typescript",
  "time_range_days": 90
}
```

### 7. autocode_stats
View AutoCodeIndex system statistics and health information.

**No parameters required**

## Configuration

AutoCodeIndex can be extensively customized through configuration:

### Full Configuration Example

```json
{
  "autocode": {
    "enabled": true,
    "auto_scan_projects": true,
    "track_bash_commands": true,
    "generate_session_summaries": true,
    
    "command_learning": {
      "enabled": true,
      "min_confidence_threshold": 0.3,
      "max_suggestions": 5,
      "track_failures": true,
      "platform_specific": true
    },
    
    "pattern_detection": {
      "enabled": true,
      "supported_languages": [
        "typescript", "javascript", "python", "rust", "go", "java", "php", "ruby"
      ],
      "max_scan_depth": 5,
      "ignore_patterns": [
        "node_modules", ".git", "__pycache__", "target", "build", "dist"
      ],
      "framework_detection": true,
      "architecture_analysis": true,
      "naming_convention_analysis": true
    },
    
    "session_analysis": {
      "enabled": true,
      "min_session_length": 3,
      "track_architectural_decisions": true,
      "extract_learning_patterns": true,
      "identify_workflow_improvements": true,
      "confidence_threshold": 0.6
    },
    
    "history_navigation": {
      "enabled": true,
      "similarity_threshold": 0.6,
      "max_results": 10,
      "context_window_days": 30,
      "prioritize_recent": true,
      "include_incomplete_sessions": true
    },
    
    "history_retention": {
      "session_summaries_days": 90,
      "command_patterns_days": 30,
      "project_patterns_days": 365
    }
  }
}
```

### Environment Variables

For Docker deployments, use environment variables:

```bash
# Core AutoCode settings
AUTOCODE_ENABLED=true
AUTOCODE_AUTO_SCAN_PROJECTS=true
AUTOCODE_TRACK_BASH_COMMANDS=true

# Command learning
AUTOCODE_COMMAND_LEARNING_ENABLED=true
AUTOCODE_COMMAND_MIN_CONFIDENCE=0.3

# Pattern detection  
AUTOCODE_PATTERN_DETECTION_ENABLED=true
AUTOCODE_PATTERN_MAX_SCAN_DEPTH=5

# Session analysis
AUTOCODE_SESSION_ANALYSIS_ENABLED=true
AUTOCODE_TRACK_ARCHITECTURAL_DECISIONS=true

# History navigation
AUTOCODE_HISTORY_NAVIGATION_ENABLED=true
AUTOCODE_SIMILARITY_THRESHOLD=0.6
AUTOCODE_CONTEXT_WINDOW_DAYS=30
```

## Memory Types

AutoCodeIndex introduces 4 specialized memory types:

### project_pattern
Stores comprehensive project analysis including:
- Framework and language detection results
- Architectural patterns and design decisions
- Naming conventions and coding standards
- Component relationships and dependencies
- Build tools and development environment setup

### command_pattern  
Tracks command usage and optimization:
- Command execution history with success/failure rates
- Platform-specific command variations
- Context-aware command suggestions
- Retry patterns and failure analysis
- Performance metrics and optimization opportunities

### session_summary
Rich analysis of conversation sessions:
- Task completion tracking and outcomes
- Learning moments and knowledge acquisition
- Architectural decisions and their reasoning
- File modifications and code changes
- Workflow patterns and efficiency metrics

### bash_execution
Detailed command execution tracking:
- Full command line with arguments and flags
- Execution context (working directory, environment)
- Output and error messages
- Execution time and performance data
- Success/failure patterns for learning

## Best Practices

### 1. Optimize AutoCodeIndex Performance

**Project Organization:**
- Use consistent project structure across similar projects
- Follow established naming conventions for better pattern recognition
- Keep configuration files (package.json, requirements.txt) up to date

**Command Usage:**
- Use descriptive commit messages for better session analysis
- Group related commands together for workflow optimization
- Document complex command sequences with comments

### 2. Maximize Learning Benefits

**Explicit Learning:**
- Ask Claude to explain decisions and provide learning insights
- Request comparisons between different approaches
- Seek recommendations for skill development areas

**Pattern Recognition:**
- Work on similar projects to build strong pattern recognition
- Experiment with different approaches to the same problem
- Ask for architecture reviews and pattern analysis

### 3. Effective Session Management

**Session Continuity:**
- Start sessions with clear context about what you want to accomplish
- Reference previous work when continuing multi-session projects
- End sessions with explicit summary of what was accomplished

**Context Building:**
- Provide project context when switching between different codebases
- Mention relevant constraints, requirements, and goals
- Update Claude on any external changes or decisions

## Troubleshooting

### Common Issues

**AutoCodeIndex Not Working:**
1. Check that `autocode.enabled` is set to `true` in configuration
2. Verify memory system is properly initialized
3. Check logs for any initialization errors

**Pattern Detection Not Accurate:**
1. Ensure project has recognizable structure (package.json, etc.)
2. Check `supported_languages` includes your project's languages
3. Verify `max_scan_depth` allows scanning your project structure

**Command Suggestions Not Helpful:**
1. Increase `min_confidence_threshold` for higher quality suggestions
2. Check that command execution tracking is enabled
3. Build more command history by using Claude for development tasks

**Session Analysis Missing Data:**
1. Ensure `min_session_length` threshold is appropriate
2. Check that `track_architectural_decisions` is enabled
3. Verify conversations include enough technical detail for analysis

### Performance Optimization

**Large Projects:**
- Adjust `max_scan_depth` to balance thoroughness with performance
- Use `ignore_patterns` to exclude unnecessary directories
- Consider enabling only specific AutoCodeIndex features for very large codebases

**Memory Usage:**
- Adjust `history_retention` settings to manage storage
- Periodically clean old session summaries and command patterns
- Monitor memory usage through `autocode_stats` tool

**Analysis Speed:**
- Reduce `context_window_days` for faster similarity searches
- Lower `similarity_threshold` to reduce computation time
- Disable unused features to improve overall performance

## Integration Examples

### With VS Code
AutoCodeIndex works seamlessly with VS Code through Claude Desktop:

1. Configure MCP server in Claude Desktop settings
2. Use Claude sidebar for intelligent assistance
3. AutoCodeIndex automatically tracks file changes and commands

### With Terminal Workflows
AutoCodeIndex learns from your terminal usage:

1. Command execution is automatically tracked
2. Success/failure patterns are analyzed
3. Platform-specific optimizations are suggested

### With Git Workflows
AutoCodeIndex understands Git patterns:

1. Commit message analysis for project evolution
2. Branch naming convention detection
3. Merge conflict resolution pattern learning

## API Reference

See the main [API documentation](../README.md#available-tools) for detailed information about each tool's parameters and responses.

## Advanced Usage

### Custom Pattern Detection
Extend AutoCodeIndex with custom pattern detection rules:

```json
{
  "pattern_detection": {
    "custom_patterns": {
      "my_framework": {
        "files": ["my-config.json"],
        "content_patterns": ["customFrameworkInit"]
      }
    }
  }
}
```

### Advanced Command Learning
Configure sophisticated command learning behaviors:

```json
{
  "command_learning": {
    "intent_mapping": {
      "test": ["npm test", "pytest", "cargo test"],
      "build": ["npm run build", "cargo build", "make"]
    },
    "success_threshold": 0.8,
    "retry_analysis": true
  }
}
```

### Session Analysis Customization
Tailor session analysis to your workflow:

```json
{
  "session_analysis": {
    "custom_indicators": {
      "completion": ["done", "finished", "completed", "shipped"],
      "learning": ["learned", "discovered", "realized", "understood"],
      "issues": ["problem", "bug", "error", "failed"]
    }
  }
}
```

## Support and Feedback

For issues, feature requests, or questions about AutoCodeIndex:

1. Check this documentation and the main README
2. Review configuration options and troubleshooting steps  
3. Submit issues to the project repository
4. Join community discussions for tips and best practices

AutoCodeIndex is designed to enhance your development workflow through intelligent assistance and continuous learning. The more you use it, the more valuable its insights and suggestions become.