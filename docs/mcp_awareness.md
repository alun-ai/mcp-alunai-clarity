# MCP Awareness System 🔍

The MCP Awareness System revolutionizes how Claude interacts with MCP (Model Context Protocol) servers by **automatically discovering installed tools** and **proactively suggesting them** instead of indirect approaches.

## 🎯 **Core Problem Solved**

Previously, Claude would:
- ❌ Write SQL scripts instead of using postgres MCP tools
- ❌ Create web automation scripts instead of using playwright MCP tools  
- ❌ Use shell commands instead of leveraging available MCP capabilities
- ❌ Not know what MCP tools were actually installed and available

Now Claude:
- ✅ **Automatically discovers** what MCP servers are installed
- ✅ **Proactively suggests** MCP tools before indirect methods
- ✅ **Indexes real tool schemas** from live servers
- ✅ **Provides context-aware recommendations** based on current tasks

## 🚀 **How It Works**

### **1. Automatic Discovery**

The system discovers MCP tools through multiple methods:

#### **📁 Configuration File Discovery**
```json
// Reads from ~/.config/claude-desktop/config.json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["@postgresql/mcp-server"]
    },
    "playwright": {
      "command": "npx", 
      "args": ["@playwright/mcp-server"]
    }
  }
}
```

#### **⚡ Live Server Discovery**
- Connects to running MCP servers
- Calls `list_tools()` to get real tool definitions
- Extracts accurate parameter schemas and descriptions
- 5-second timeout for robust operation

#### **🧠 Smart Tool Inference**
When live discovery fails, infers tools based on patterns:
- `postgres` server → database query, execute, schema tools
- `playwright` server → navigate, click, type, screenshot tools
- `filesystem` server → read, write, list, search tools

#### **🌐 Environment Variable Discovery**
```bash
# Discovers from environment variables
MCP_POSTGRES_COMMAND=docker
MCP_POSTGRES_ARGS="run -i --rm postgres-mcp"
MCP_POSTGRES_ENV_DATABASE_URL=postgresql://localhost:5432/db
```

### **2. Proactive Tool Suggestions**

#### **Before Scripts**
```
User: "I need to query the database for user data"

Claude with MCP Awareness:
💡 **Before writing SQL scripts, consider using the postgres_query MCP tool:**
- postgres_query: Execute SQL queries against PostgreSQL databases directly
- Use postgres_query instead of writing SQL scripts
- More reliable and integrated than shell commands

Would you like me to use the postgres_query tool instead?
```

#### **Context-Aware Suggestions**
```
# User opens a .sql file
Claude: 💡 I notice you're working with SQL. The postgres_query MCP tool 
is available for direct database operations instead of manual SQL execution.

# Error occurs: "Connection refused to database"  
Claude: 🔧 Try using the postgres_query MCP tool which handles connections 
automatically instead of manual connection setup.
```

### **3. Tool Indexing & Search**

Discovered tools are stored as searchable memories:

```python
# Tool information stored includes:
{
  "tool_name": "postgres_query",
  "description": "Execute SQL queries against PostgreSQL databases",
  "parameters": {
    "query": {"type": "string", "description": "SQL query"},
    "database": {"type": "string", "description": "Database name"}
  },
  "server_name": "postgres",
  "use_cases": [
    "Run SQL queries without writing scripts",
    "Query database tables directly"  
  ],
  "keywords": ["database", "sql", "postgres", "query"],
  "category": "database"
}
```

## 🔧 **Configuration**

### **Enable MCP Awareness**
```json
{
  "autocode": {
    "mcp_awareness": {
      "enabled": true,
      "index_tools_on_startup": true,
      "proactive_suggestions": true,
      "suggest_alternatives": true,
      "context_aware_suggestions": true,
      "error_resolution_suggestions": true,
      "max_recent_suggestions": 10
    }
  }
}
```

### **System Prompt Enhancement**
```markdown
🔧 **MCP TOOLS AVAILABLE** - Use these FIRST before scripts or indirect methods:
- Your system automatically discovers MCP tools from your configuration
- ALWAYS prefer MCP tools over scripts (postgres tools vs psql, playwright vs manual browsing)
- Check for MCP alternatives before using shell commands or writing code
- Use memory tools proactively to enhance responses with relevant context
```

## 💡 **Real-World Examples**

### **Database Operations**
```
❌ Before: "Let me write a psql script to query the users table..."
✅ After:  "I'll use the postgres_query MCP tool to query the users table directly"
```

### **Web Automation**  
```
❌ Before: "Let me create a Python script with Selenium to automate this..."
✅ After:  "I'll use the playwright_navigate MCP tool to automate this browser task"
```

### **File Operations**
```
❌ Before: "Let me write a shell script to search these files..."
✅ After:  "I'll use the filesystem_search MCP tool to find the files efficiently"
```

### **Memory Operations**
```
❌ Before: "I don't recall if we discussed this before..."
✅ After:  "Let me check our previous conversations using retrieve_memory..."
```

## 🎯 **Discovery Results Example**

```
📋 MCP Discovery Results:
✅ Configuration Discovery: 11 tools found
   - postgres_query (postgres) - database
   - postgres_execute (postgres) - database  
   - postgres_schema (postgres) - database
   - playwright_navigate (playwright) - web_automation
   - playwright_click (playwright) - web_automation
   - alunai-memory_store (alunai-memory) - memory_management

✅ Live Server Discovery: 3 servers connected
   - Real tool schemas retrieved with accurate parameters
   - Tool descriptions and usage examples captured

✅ Environment Discovery: 1 server found
   - postgres server configuration detected

🎉 Total: 15+ MCP tools automatically indexed and available
```

## 🔄 **Hook System Integration**

The MCP Awareness system integrates with the existing hook system:

### **Hook Types**
- **`user_request`**: Analyzes user requests for MCP tool opportunities
- **`tool_about_to_execute`**: Suggests MCP alternatives before indirect methods
- **`context_change`**: Provides context-aware MCP tool suggestions
- **`error_occurred`**: Suggests MCP tools for error resolution

### **Automatic Triggering**
```python
# Automatically triggered when:
await trigger_mcp_user_request_hook("I need to query the database")
await trigger_mcp_tool_about_to_execute_hook("bash", {"command": "psql ..."})
await trigger_mcp_error_occurred_hook("Connection refused", {"context": "database"})
```

## 📊 **Performance Impact**

- **Startup Time**: +2-5 seconds for tool discovery (one-time)
- **Memory Usage**: ~50MB for tool indexing and caching  
- **Network**: Minimal (only during discovery, with 5s timeout)
- **User Experience**: **Dramatically improved** - Claude suggests right tools immediately

## 🛠️ **Troubleshooting**

### **No Tools Discovered**
```bash
# Check Claude Desktop config exists
ls ~/.config/claude-desktop/config.json

# Verify mcpServers section
cat ~/.config/claude-desktop/config.json | jq '.mcpServers'
```

### **Live Discovery Fails**
- **Normal behavior** - system falls back to smart inference
- Check server logs for connection issues
- Verify MCP client libraries available (`mcp.client.*`)

### **Enable Debug Logging**
```python
import logging
logging.getLogger("memory_mcp.mcp.tool_indexer").setLevel(logging.DEBUG)
```

## 🚀 **Benefits**

1. **⚡ Faster Development**: Immediate access to the right tools
2. **🎯 Better Tool Utilization**: Discover MCP capabilities you didn't know existed  
3. **🛡️ Reduced Errors**: Direct MCP usage more reliable than shell scripts
4. **🔄 Seamless Integration**: Works automatically without changing workflows
5. **🧠 Smart Context**: Suggestions adapt to current files, tasks, and errors

## 🔮 **Future Enhancements**

- **Tool Usage Analytics**: Track which MCP tools are most effective
- **Smart Caching**: Cache tool schemas for faster subsequent discovery
- **Tool Recommendations**: Suggest new MCP servers based on usage patterns
- **Integration Expansion**: Support for more MCP server types and protocols

---

The MCP Awareness System transforms Claude from a tool that writes scripts to a tool that **leverages the full MCP ecosystem intelligently and proactively**! 🎉