# Memory MCP Server Architecture

This document outlines the architecture for the Claude Memory MCP server, which provides persistent memory capabilities for Large Language Models.

## Architectural Overview

The server follows a functional domain-based architecture rather than a traditional layered approach. This design is inspired by cognitive memory processing patterns and focuses on four primary domains:

```
┌─────────────────────────────────────────────────────────┐
│                   Claude Desktop                        │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                     MCP Interface                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐ │
│  │ Tool Definitions│  │ Request Handler │  │ Security │ │
│  └─────────────────┘  └─────────────────┘  └──────────┘ │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                Memory Domain Manager                    │
├─────────────────┬─────────────────┬────────────────────┤
│  Episodic Domain│  Semantic Domain│  Temporal Domain   │
├─────────────────┴─────────────────┴────────────────────┤
│                  Persistence Domain                    │
└─────────────────────────────────────────────────────────┘
```

## Domain Details

### 1. Episodic Domain

The Episodic Domain is responsible for managing memory of specific events, conversations, and experiences.

**Responsibilities:**
- Recording and retrieving conversation histories
- Managing session-based interactions
- Contextualizing memories with temporal and situational details
- Narrative memory construction
- Recording agent reflections and observations

**Key Components:**
- `EpisodicProcessor`: Processes new episodic memories
- `EpisodicRetrieval`: Retrieves relevant episodic memories based on context
- `ConversationManager`: Manages conversation history and context
- `ReflectionEngine`: Creates reflection memories from experiences

### 2. Semantic Domain

The Semantic Domain handles factual knowledge, conceptual understanding, and relational knowledge.

**Responsibilities:**
- Managing factual information and knowledge
- Organizing categorical and conceptual information
- Handling entity relationships and attributes
- Knowledge consolidation and organization
- Abstract concept representation

**Key Components:**
- `KnowledgeProcessor`: Processes factual information
- `EntityManager`: Manages entity information and relationships
- `ConceptualOrganizer`: Organizes related concepts
- `KnowledgeRetrieval`: Retrieves relevant knowledge based on context

### 3. Temporal Domain

The Temporal Domain provides time-aware processing across all memory operations.

**Responsibilities:**
- Managing memory decay and importance over time
- Temporal indexing and sequencing
- Chronological relationship tracking
- Time-based memory consolidation
- Recency effects in retrieval

**Key Components:**
- `TemporalProcessor`: Adds temporal context to memories
- `DecayManager`: Implements importance decay algorithms
- `ChronologicalIndexer`: Maintains temporal indices
- `TimeAwareRetrieval`: Incorporates time in retrieval relevance

### 4. Persistence Domain

The Persistence Domain handles the storage, retrieval, and management of memories across sessions.

**Responsibilities:**
- File system operations
- Vector embedding generation and storage
- Index management
- Memory file structure 
- Backup and recovery
- Efficient storage formats

**Key Components:**
- `FileManager`: Handles reading and writing to the memory file
- `EmbeddingManager`: Generates and stores vector embeddings
- `IndexManager`: Manages semantic and other indices
- `BackupManager`: Handles backup and recovery operations

## Communication Patterns

The domains communicate through the Memory Domain Manager, which orchestrates interactions between domains. This design allows for:

1. **Parallel Processing**: Domains can process information concurrently
2. **Domain Specialization**: Each domain focuses on specific memory tasks
3. **Cross-Domain Integration**: Information flows naturally between domains
4. **Cohesive Memory Operations**: Complex operations utilize multiple domains

## MCP Interface

The MCP Interface layer handles communication with Claude Desktop through the Model Context Protocol:

### Tool Definitions

The server exposes these memory-related tools:

- `store_memory`: Store new information in memory
- `retrieve_memory`: Retrieve relevant memories based on query
- `list_memories`: List available memories with filtering options
- `update_memory`: Update existing memory entries
- `delete_memory`: Remove specific memories
- `memory_stats`: Get statistics about the memory store

### Request Handler

The Request Handler processes incoming MCP requests:
- Validates request parameters
- Routes requests to appropriate domain operations
- Formats responses according to MCP protocol
- Handles errors and exceptions

### Security

The Security component provides:
- Input validation and sanitization
- Access control for file operations
- Rate limiting to prevent abuse
- Secure error handling

## Memory File Structure

The memory system uses a JSON-based file structure with the following components:

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "ISO-8601 timestamp",
    "updated_at": "ISO-8601 timestamp"
  },
  "memory_index": {
    // Vector index for fast semantic search
  },
  "short_term_memory": [
    // Recent and frequently accessed memories
  ],
  "long_term_memory": [
    // Older or less frequently accessed memories
  ],
  "archived_memory": [
    // Rarely accessed but potentially valuable memories
  ],
  "memory_schema": {
    // Schema definitions for memory entries
  },
  "config": {
    // Configuration settings for memory management
  }
}
```

## Memory Management Operations

The system performs these key memory operations:

### 1. Addition
New memories are added with:
- Unique ID generation
- Embedding vector creation
- Metadata population
- Index updating

### 2. Retrieval
Memories are retrieved through:
- Semantic search via the index
- Filtering by metadata (type, time, tags)
- Contextual relationships
- Hybrid ranking (combining semantic similarity, recency, importance)

### 3. Consolidation
Periodic consolidation processes:
- Move memories between tiers based on access patterns
- Update importance scores based on usage
- Merge related memories when appropriate
- Summarize lengthy content for efficiency

### 4. Forgetting
Controlled memory management:
- Archive less important memories
- Prune redundant information
- Delete obsolete or low-value memories
- Maintain a forgetting curve inspired by human memory

## Research Foundation

This architecture is based on comprehensive research of current LLM persistent memory techniques:

- **Tiered Memory Architecture**: Inspired by MemGPT's OS-inspired approach
- **Context-Sensitivity**: Drawn from academic research on episodic memory for LLMs
- **Embedding-Based Retrieval**: Building on vector database approaches
- **Self-Reflection**: Incorporating ideas from LLM memory consolidation research

The functional domain-based architecture represents an evolution beyond simple component decomposition, organizing the system around cognitive memory functions rather than technical boundaries.
