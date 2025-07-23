"""
Memory Domain Manager that orchestrates all memory operations.
"""

import uuid
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from clarity.domains.episodic import EpisodicDomain
from clarity.domains.semantic import SemanticDomain
from clarity.domains.temporal import TemporalDomain
from clarity.domains.persistence import PersistenceDomain
from clarity.autocode.domain import AutoCodeDomain
from clarity.shared.monitoring import performance_monitor, get_metrics_collector
from clarity.utils.schema import get_mcp_memory_types, is_mcp_integrated_type, get_memory_cache_config
from clarity.core.unified_cache import cache_put, cache_get, CacheType
from clarity.mcp.cache_integration import get_mcp_cache_adapter
from clarity.core.mcp_memory_retrieval import (
    get_mcp_memory_retriever, MCPContext, RetrievalStrategy, retrieve_with_mcp_context
)


class MemoryDomainManager:
    """
    Orchestrates operations across all memory domains.
    
    This class coordinates interactions between the different functional domains
    of the memory system. It provides a unified interface for memory operations
    while delegating specific tasks to the appropriate domain.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the memory domain manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_collector = get_metrics_collector()
        
        # Initialize domains
        self.persistence_domain = PersistenceDomain(config)
        self.episodic_domain = EpisodicDomain(config, self.persistence_domain)
        self.semantic_domain = SemanticDomain(config, self.persistence_domain)
        self.temporal_domain = TemporalDomain(config, self.persistence_domain)
        self.autocode_domain = AutoCodeDomain(config, self.persistence_domain)
        
        # Initialize MCP integration
        self.mcp_cache_adapter = get_mcp_cache_adapter()
        self.mcp_memory_types = get_mcp_memory_types()
        self.mcp_memory_retriever = None  # Initialized lazily
    
    async def initialize(self) -> None:
        """Initialize all domains."""
        logger.info("Initializing Memory Domain Manager")
        
        # Initialize domains in order (persistence first)
        await self.persistence_domain.initialize()
        await self.episodic_domain.initialize()
        await self.semantic_domain.initialize()
        await self.temporal_domain.initialize()
        await self.autocode_domain.initialize()
        
        # Initialize AutoCode command learner with domain manager reference
        await self.autocode_domain.set_command_learner(self)
        
        logger.info("Memory Domain Manager initialized")
    
    @performance_monitor.measure("memory.store", tags={"component": "memory"})
    async def store_memory(
        self,
        memory_type: str,
        content: Dict[str, Any],
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a new memory.
        
        Args:
            memory_type: Type of memory (conversation, fact, document, entity, reflection, code)
            content: Memory content (type-specific structure)
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
            context: Contextual information
            
        Returns:
            Memory ID
        """
        # Generate a unique ID for the memory
        memory_id = f"mem_{str(uuid.uuid4())}"
        
        # Create memory object
        memory = {
            "id": memory_id,
            "type": memory_type,
            "content": content,
            "importance": importance,
            "metadata": metadata or {},
            "context": context or {}
        }
        
        # Add temporal information
        memory = await self.temporal_domain.process_new_memory(memory)
        
        # Process based on memory type - MCP-enhanced processing
        if memory_type in ["conversation", "reflection"]:
            memory = await self.episodic_domain.process_memory(memory)
        elif memory_type in ["fact", "document", "entity"]:
            memory = await self.semantic_domain.process_memory(memory)
        elif memory_type == "code" or memory_type == "enhanced_context":
            # Code and enhanced context memories get processed by both domains
            memory = await self.episodic_domain.process_memory(memory)
            memory = await self.semantic_domain.process_memory(memory)
        elif memory_type in ["project_pattern", "command_pattern", "session_summary", "bash_execution"]:
            # Legacy AutoCode memories (deprecated but still supported)
            memory = await self.autocode_domain.process_memory(memory) if hasattr(self.autocode_domain, 'process_memory') else memory
        elif memory_type in ["mcp_thinking_workflow", "thinking_mcp_integration", "structured_thinking", "thought_process"]:
            # MCP-enhanced thinking memories get processed by episodic domain with MCP context
            memory = await self._process_mcp_thinking_memory(memory)
        elif memory_type in ["mcp_resource_pattern", "mcp_workflow_pattern", "mcp_tool_correlation"]:
            # MCP workflow and correlation memories get processed by semantic domain with MCP context
            memory = await self._process_mcp_workflow_memory(memory)
        elif memory_type == "thinking_relationship":
            # Relationship memories are processed minimally (just embeddings if needed)
            memory = await self._process_relationship_memory(memory)
        elif is_mcp_integrated_type(memory_type):
            # Any other MCP-integrated type gets enhanced processing
            memory = await self._process_mcp_enhanced_memory(memory)
        
        # Determine memory tier based on importance and recency
        tier = "short_term"
        if importance < self.config["alunai-clarity"].get("short_term_threshold", 0.3):
            tier = "long_term"
        
        # Store the memory
        await self.persistence_domain.store_memory(memory, tier)
        
        logger.info(f"Stored {memory_type} memory with ID {memory_id} in {tier} tier")
        
        # Cache memory if MCP integration is enabled for this type
        if get_memory_cache_config(memory_type):
            await self._cache_mcp_memory(memory_id, memory, memory_type)
        
        return memory_id
    
    @performance_monitor.measure("memory.retrieve", tags={"component": "memory"})
    async def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_similarity: float = 0.6,
        include_metadata: bool = False,
        mcp_enhanced: bool = False,
        mcp_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a query with optional MCP enhancement.
        
        Args:
            query: Query string
            limit: Maximum number of memories to retrieve
            memory_types: Types of memories to include (None for all types)
            min_similarity: Minimum similarity score for results
            include_metadata: Whether to include metadata in the results
            mcp_enhanced: Whether to use MCP-enhanced retrieval
            mcp_context: MCP context for enhanced retrieval
            
        Returns:
            List of relevant memories
        """
        if mcp_enhanced and mcp_context:
            return await self._retrieve_memories_mcp_enhanced(query, limit, memory_types, min_similarity, include_metadata, mcp_context)
        else:
            return await self._retrieve_memories_standard(query, limit, memory_types, min_similarity, include_metadata)
    
    async def _retrieve_memories_standard(
        self,
        query: str,
        limit: int,
        memory_types: Optional[List[str]],
        min_similarity: float,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Standard memory retrieval without MCP enhancement."""
        # Generate query embedding
        embedding = await self.persistence_domain.generate_embedding(query)
        
        # Retrieve memories using semantic search
        memories = await self.persistence_domain.search_memories(
            embedding=embedding,
            limit=limit,
            types=memory_types,
            min_similarity=min_similarity
        )
        
        # Apply temporal adjustments to relevance
        memories = await self.temporal_domain.adjust_memory_relevance(memories, query)
        
        # Format results
        result_memories = []
        for memory in memories:
            # Debug: check what fields are available
            logger.debug(f"Memory fields: {list(memory.keys())}")
            result_memory = {
                "id": memory.get("id", memory.get("memory_id")),
                "type": memory.get("type", memory.get("memory_type")),
                "content": memory.get("content"),
                "similarity": memory.get("similarity", memory.get("similarity_score", 0.0))
            }
            
            # Include metadata if requested
            if include_metadata:
                result_memory["metadata"] = memory.get("metadata", {})
                result_memory["created_at"] = memory.get("created_at")
                result_memory["last_accessed"] = memory.get("last_accessed")
                result_memory["importance"] = memory.get("importance", 0.5)
                result_memory["mcp_enhanced"] = False
            
            result_memories.append(result_memory)
        
        # Update access time for retrieved memories
        for memory in memories:
            memory_id = memory.get("id", memory.get("memory_id"))
            if memory_id:
                await self.temporal_domain.update_memory_access(memory_id)
        
        return result_memories
    
    async def _retrieve_memories_mcp_enhanced(
        self,
        query: str,
        limit: int,
        memory_types: Optional[List[str]],
        min_similarity: float,
        include_metadata: bool,
        mcp_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """MCP-enhanced memory retrieval with context awareness."""
        # Initialize MCP retriever if needed
        if self.mcp_memory_retriever is None:
            self.mcp_memory_retriever = get_mcp_memory_retriever(self)
        
        # Create MCP context
        context = MCPContext(
            current_tools=mcp_context.get("current_tools", []),
            thinking_stage=mcp_context.get("thinking_stage"),
            workflow_pattern=mcp_context.get("workflow_pattern"),
            project_context=mcp_context.get("project_context", {}),
            session_context=mcp_context.get("session_context", {}),
            user_intent=mcp_context.get("user_intent"),
            performance_requirements=mcp_context.get("performance_requirements", {})
        )
        
        # Determine strategy based on context
        strategy = RetrievalStrategy.ADAPTIVE_INTELLIGENT
        if mcp_context.get("retrieval_strategy"):
            try:
                strategy = RetrievalStrategy(mcp_context["retrieval_strategy"])
            except ValueError:
                logger.warning(f"Unknown retrieval strategy: {mcp_context['retrieval_strategy']}")
        
        # Perform MCP-enhanced retrieval
        enhanced_results = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            query, context, limit, strategy, min_similarity, include_metadata
        )
        
        # Convert enhanced results to standard format
        result_memories = []
        for result in enhanced_results:
            result_memory = {
                "id": result.memory_id,
                "type": result.memory_type,
                "content": result.content,
                "similarity": result.final_score,  # Use enhanced score
                "base_similarity": result.base_similarity
            }
            
            if include_metadata:
                result_memory["metadata"] = result.content.get("metadata", {})
                result_memory["mcp_enhanced"] = True
                result_memory["mcp_relevance_boost"] = result.mcp_relevance_boost
                result_memory["workflow_matches"] = result.workflow_matches
                result_memory["tool_correlations"] = result.tool_correlations
                result_memory["cross_system_confidence"] = result.cross_system_confidence
                result_memory["retrieval_context"] = result.retrieval_context
            
            result_memories.append(result_memory)
        
        # Update access times
        for result in enhanced_results:
            if result.memory_id:
                await self.temporal_domain.update_memory_access(result.memory_id)
        
        logger.info(f"MCP-enhanced retrieval returned {len(result_memories)} results for query: {query[:50]}...")
        
        return result_memories
    
    async def list_memories(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
        tier: Optional[str] = None,
        include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List available memories with filtering options.
        
        Args:
            memory_types: Types of memories to include (None for all types)
            limit: Maximum number of memories to retrieve
            offset: Offset for pagination
            tier: Memory tier to retrieve from (None for all tiers)
            include_content: Whether to include memory content in the results
            
        Returns:
            List of memories
        """
        # Retrieve memories from persistence domain
        memories = await self.persistence_domain.list_memories(
            types=memory_types,
            limit=limit,
            offset=offset,
            tier=tier,
            include_content=include_content  # CRITICAL FIX: Pass include_content parameter
        )
        
        # Format results
        result_memories = []
        for memory in memories:
            result_memory = {
                "id": memory["id"],
                "type": memory["type"],
                "created_at": memory.get("created_at"),
                "last_accessed": memory.get("last_accessed"),
                "importance": memory.get("importance", 0.5),
                "tier": memory.get("tier", "short_term")
            }
            
            # Include content if requested
            if include_content:
                result_memory["content"] = memory.get("content")
            
            result_memories.append(result_memory)
        
        return result_memories
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            updates: Updates to apply to the memory
            
        Returns:
            Success flag
        """
        # Retrieve the memory
        memory = await self.persistence_domain.get_memory(memory_id)
        if not memory:
            logger.error(f"Memory {memory_id} not found")
            return False
        
        # Apply updates
        if "content" in updates:
            memory["content"] = updates["content"]
            
            # Re-process embedding if content changes
            if memory["type"] in ["conversation", "reflection"]:
                memory = await self.episodic_domain.process_memory(memory)
            elif memory["type"] in ["fact", "document", "entity"]:
                memory = await self.semantic_domain.process_memory(memory)
            elif memory["type"] == "code":
                memory = await self.episodic_domain.process_memory(memory)
                memory = await self.semantic_domain.process_memory(memory)
        
        if "importance" in updates:
            memory["importance"] = updates["importance"]
        
        if "metadata" in updates:
            memory["metadata"].update(updates["metadata"])
        
        if "context" in updates:
            memory["context"].update(updates["context"])
        
        # Update last_modified timestamp
        memory = await self.temporal_domain.update_memory_modification(memory)
        
        # Determine if memory tier should change based on updates
        current_tier = await self.persistence_domain.get_memory_tier(memory_id)
        new_tier = current_tier
        
        if "importance" in updates:
            if updates["importance"] >= self.config["alunai-clarity"].get("short_term_threshold", 0.3) and current_tier != "short_term":
                new_tier = "short_term"
            elif updates["importance"] < self.config["alunai-clarity"].get("short_term_threshold", 0.3) and current_tier == "short_term":
                new_tier = "long_term"
        
        # Store the updated memory
        await self.persistence_domain.update_memory(memory_id, updates)
        
        logger.info(f"Updated memory {memory_id}")
        
        return True
    
    async def _process_mcp_thinking_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP-enhanced thinking memories."""
        # Add MCP context integration
        if "mcp_context" not in memory:
            memory["mcp_context"] = {}
        
        # Extract workflow patterns from content
        content = memory.get("content", {})
        if "workflow_pattern" in content:
            workflow_pattern = content["workflow_pattern"]
            # Cache the workflow pattern for future use
            await self.mcp_cache_adapter.cache_workflow_pattern(
                workflow_pattern, 
                memory["mcp_context"]
            )
        
        # Process with episodic domain for temporal relationships
        memory = await self.episodic_domain.process_memory(memory)
        
        # Add MCP correlation data
        memory["mcp_context"]["processed_by"] = "episodic_domain"
        memory["mcp_context"]["thinking_enhanced"] = True
        
        return memory
    
    async def _process_mcp_workflow_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP workflow and correlation memories."""
        # Add MCP context integration
        if "mcp_context" not in memory:
            memory["mcp_context"] = {}
        
        # Extract and cache workflow patterns
        content = memory.get("content", {})
        if memory["type"] == "mcp_workflow_pattern":
            pattern_data = {
                "trigger_context": content.get("trigger_context", ""),
                "tool_sequence": content.get("tool_sequence", []),
                "success_metrics": content.get("success_metrics", {}),
                "effectiveness_score": content.get("success_metrics", {}).get("completion_rate", 0.0)
            }
            await self.mcp_cache_adapter.cache_workflow_pattern(
                pattern_data,
                memory["mcp_context"]
            )
        
        # Process with semantic domain for concept relationships
        memory = await self.semantic_domain.process_memory(memory)
        
        # Add MCP correlation data
        memory["mcp_context"]["processed_by"] = "semantic_domain"
        memory["mcp_context"]["workflow_enhanced"] = True
        
        return memory
    
    async def _process_relationship_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process thinking relationship memories (minimal processing)."""
        # Add MCP context but minimal processing
        if "mcp_context" not in memory:
            memory["mcp_context"] = {}
        
        # Relationships don't need embeddings, just store the structure
        memory["mcp_context"]["processed_by"] = "minimal_processing"
        memory["mcp_context"]["relationship_type"] = memory.get("content", {}).get("relationship_type", "unknown")
        
        return memory
    
    async def _process_mcp_enhanced_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process any MCP-enhanced memory type."""
        # Add MCP context integration
        if "mcp_context" not in memory:
            memory["mcp_context"] = {}
        
        # Determine best processing domain based on memory type
        memory_type = memory["type"]
        
        if "thinking" in memory_type or "workflow" in memory_type:
            # Process with episodic domain for temporal context
            memory = await self.episodic_domain.process_memory(memory)
            memory["mcp_context"]["processed_by"] = "episodic_domain"
        elif "resource" in memory_type or "pattern" in memory_type:
            # Process with semantic domain for concept relationships
            memory = await self.semantic_domain.process_memory(memory)
            memory["mcp_context"]["processed_by"] = "semantic_domain"
        else:
            # Default to both domains for enhanced context
            memory = await self.episodic_domain.process_memory(memory)
            memory = await self.semantic_domain.process_memory(memory)
            memory["mcp_context"]["processed_by"] = "both_domains"
        
        memory["mcp_context"]["mcp_enhanced"] = True
        
        return memory
    
    async def _cache_mcp_memory(self, memory_id: str, memory: Dict[str, Any], memory_type: str) -> None:
        """Cache MCP-integrated memory for performance optimization."""
        try:
            # Determine cache type based on memory type
            if "workflow" in memory_type or "pattern" in memory_type:
                cache_type = CacheType.MCP_WORKFLOW
            else:
                cache_type = CacheType.MEMORY_PATTERN
            
            # Cache the memory with MCP context
            await cache_put(
                key=f"mcp_memory_{memory_id}",
                value={
                    "memory": memory,
                    "cache_timestamp": memory.get("created_at"),
                    "mcp_integration": memory.get("mcp_context", {})
                },
                cache_type=cache_type,
                ttl=1800.0,  # 30 minutes
                metadata={
                    "memory_type": memory_type,
                    "memory_id": memory_id,
                    "mcp_enhanced": True
                }
            )
            
            logger.debug(f"Cached MCP memory {memory_id} of type {memory_type}")
            
        except Exception as e:
            logger.warning(f"Failed to cache MCP memory {memory_id}: {e}")
    
    async def delete_memories(
        self,
        memory_ids: List[str]
    ) -> bool:
        """
        Delete memories.
        
        Args:
            memory_ids: IDs of memories to delete
            
        Returns:
            Success flag
        """
        deleted_ids = await self.persistence_domain.delete_memories(memory_ids)
        success = len(deleted_ids) == len(memory_ids)
        
        if success:
            logger.info(f"Deleted {len(deleted_ids)} memories")
        else:
            logger.error(f"Failed to delete some memories. Deleted {len(deleted_ids)}/{len(memory_ids)}")
        
        return success
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Memory statistics
        """
        # Get basic stats from persistence domain
        stats = await self.persistence_domain.get_memory_stats()
        
        # Enrich with domain-specific stats
        episodic_stats = await self.episodic_domain.get_stats()
        semantic_stats = await self.semantic_domain.get_stats()
        temporal_stats = await self.temporal_domain.get_stats()
        
        stats.update({
            "episodic_domain": episodic_stats,
            "semantic_domain": semantic_stats,
            "temporal_domain": temporal_stats,
            "autocode_domain": await self.autocode_domain.get_stats(),
            "mcp_integration": await self._get_mcp_integration_stats(),
            "mcp_retrieval": await self._get_mcp_retrieval_stats()
        })
        
        return stats
    
    async def _get_mcp_integration_stats(self) -> Dict[str, Any]:
        """Get MCP integration statistics."""
        try:
            mcp_analytics = await self.mcp_cache_adapter.get_mcp_cache_analytics()
            
            # Count MCP memory types
            mcp_type_counts = {}
            for memory_type, config in self.mcp_memory_types.items():
                if config.get("mcp_integration", "none") != "none":
                    # This would require a query to count memories - simplified for now
                    mcp_type_counts[memory_type] = 0  # Placeholder
            
            return {
                "mcp_memory_types_configured": len([t for t, c in self.mcp_memory_types.items() if c.get("mcp_integration", "none") != "none"]),
                "mcp_cache_performance": mcp_analytics.get("cache_performance", {}),
                "mcp_memory_type_counts": mcp_type_counts,
                "mcp_cache_enabled_types": len([t for t, c in self.mcp_memory_types.items() if c.get("cache_enabled", False)])
            }
        except Exception as e:
            logger.warning(f"Failed to get MCP integration stats: {e}")
            return {"error": str(e)}
    
    async def _get_mcp_retrieval_stats(self) -> Dict[str, Any]:
        """Get MCP retrieval statistics."""
        try:
            if self.mcp_memory_retriever:
                return await self.mcp_memory_retriever.get_retrieval_analytics()
            else:
                return {
                    "mcp_retriever_initialized": False,
                    "note": "MCP retriever will be initialized on first enhanced retrieval"
                }
        except Exception as e:
            logger.warning(f"Failed to get MCP retrieval stats: {e}")
            return {"error": str(e)}
    
    # MCP-Enhanced Memory Retrieval Methods
    async def retrieve_with_workflow_context(
        self,
        query: str,
        current_tools: List[str] = None,
        thinking_stage: str = None,
        workflow_pattern: str = None,
        limit: int = 10,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with workflow context awareness."""
        mcp_context = {
            "current_tools": current_tools or [],
            "thinking_stage": thinking_stage,
            "workflow_pattern": workflow_pattern,
            "retrieval_strategy": "workflow_pattern"
        }
        
        return await self.retrieve_memories(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            mcp_enhanced=True,
            mcp_context=mcp_context,
            include_metadata=True
        )
    
    async def retrieve_with_tool_context(
        self,
        query: str,
        current_tools: List[str],
        project_context: Dict[str, Any] = None,
        limit: int = 10,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with tool correlation awareness."""
        mcp_context = {
            "current_tools": current_tools,
            "project_context": project_context or {},
            "retrieval_strategy": "cross_system_correlation"
        }
        
        return await self.retrieve_memories(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            mcp_enhanced=True,
            mcp_context=mcp_context,
            include_metadata=True
        )
    
    async def retrieve_intelligent(
        self,
        query: str,
        mcp_context: Dict[str, Any] = None,
        limit: int = 10,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Intelligent adaptive retrieval using all available MCP context."""
        if not mcp_context:
            mcp_context = {}
        
        mcp_context["retrieval_strategy"] = "adaptive_intelligent"
        
        return await self.retrieve_memories(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            mcp_enhanced=True,
            mcp_context=mcp_context,
            include_metadata=True
        )
    
    async def search_workflow_patterns(
        self,
        trigger_context: str,
        tools_used: List[str] = None,
        project_type: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for workflow patterns matching specific criteria."""
        # Build search query
        query_parts = [trigger_context]
        if tools_used:
            query_parts.extend(tools_used)
        if project_type:
            query_parts.append(project_type)
        
        query = " ".join(query_parts)
        
        return await self.retrieve_memories(
            query=query,
            memory_types=["mcp_workflow_pattern", "mcp_thinking_workflow"],
            limit=limit,
            min_similarity=0.5,
            mcp_enhanced=True,
            mcp_context={
                "current_tools": tools_used or [],
                "project_context": {"project_type": project_type} if project_type else {},
                "retrieval_strategy": "workflow_pattern"
            },
            include_metadata=True
        )
    
    async def find_similar_solutions(
        self,
        problem_description: str,
        context: Dict[str, Any] = None,
        limit: int = 8
    ) -> List[Dict[str, Any]]:
        """Find similar solutions based on problem description and context."""
        mcp_context = context or {}
        mcp_context["retrieval_strategy"] = "adaptive_intelligent"
        mcp_context["user_intent"] = "find_similar_solutions"
        
        return await self.retrieve_memories(
            query=problem_description,
            memory_types=[
                "mcp_thinking_workflow", "mcp_workflow_pattern", 
                "enhanced_context", "structured_thinking"
            ],
            limit=limit,
            min_similarity=0.5,
            mcp_enhanced=True,
            mcp_context=mcp_context,
            include_metadata=True
        )
    
    # MCP-Enhanced Memory Storage Methods
    async def store_mcp_thinking_workflow(
        self,
        session_id: str,
        thinking_stage: str,
        workflow_pattern: Dict[str, Any],
        tool_context: Dict[str, Any],
        importance: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an MCP-enhanced thinking workflow."""
        return await self.store_memory(
            memory_type="mcp_thinking_workflow",
            content={
                "session_id": session_id,
                "thinking_stage": thinking_stage,
                "workflow_pattern": workflow_pattern,
                "tool_context": tool_context
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_mcp_workflow_pattern(
        self,
        pattern_id: str,
        trigger_context: str,
        tool_sequence: List[str],
        success_metrics: Dict[str, Any],
        context_conditions: Dict[str, Any],
        importance: float = 0.9,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an MCP workflow pattern."""
        return await self.store_memory(
            memory_type="mcp_workflow_pattern",
            content={
                "pattern_id": pattern_id,
                "trigger_context": trigger_context,
                "tool_sequence": tool_sequence,
                "success_metrics": success_metrics,
                "context_conditions": context_conditions
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_mcp_resource_pattern(
        self,
        resource_reference: str,
        access_pattern: Dict[str, Any],
        success_metrics: Dict[str, Any],
        mcp_server_context: Dict[str, Any],
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an MCP resource pattern."""
        return await self.store_memory(
            memory_type="mcp_resource_pattern",
            content={
                "resource_reference": resource_reference,
                "access_pattern": access_pattern,
                "success_metrics": success_metrics,
                "mcp_server_context": mcp_server_context
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_enhanced_context(
        self,
        context_type: str,
        primary_content: Dict[str, Any],
        mcp_correlations: Dict[str, Any],
        temporal_context: Dict[str, Any],
        importance: float = 0.6,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store enhanced contextual memory."""
        return await self.store_memory(
            memory_type="enhanced_context",
            content={
                "context_type": context_type,
                "primary_content": primary_content,
                "mcp_correlations": mcp_correlations,
                "temporal_context": temporal_context
            },
            importance=importance,
            metadata=metadata
        )
    
    # Legacy AutoCode-specific methods (deprecated but maintained for backwards compatibility)
    async def store_project_pattern(
        self,
        pattern_type: str,
        framework: str,
        language: str,
        structure: Dict[str, Any],
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a project pattern."""
        return await self.store_memory(
            memory_type="project_pattern",
            content={
                "pattern_type": pattern_type,
                "framework": framework,
                "language": language,
                "structure": structure
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_command_pattern(
        self,
        command: str,
        context: Dict[str, Any],
        success_rate: float,
        platform: str,
        importance: float = 0.6,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a command pattern."""
        return await self.store_memory(
            memory_type="command_pattern",
            content={
                "command": command,
                "context": context,
                "success_rate": success_rate,
                "platform": platform
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_session_summary(
        self,
        session_id: str,
        tasks_completed: List[Dict],
        patterns_used: List[str],
        files_modified: List[str],
        importance: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a session summary."""
        return await self.store_memory(
            memory_type="session_summary",
            content={
                "session_id": session_id,
                "tasks_completed": tasks_completed,
                "patterns_used": patterns_used,
                "files_modified": files_modified
            },
            importance=importance,
            metadata=metadata
        )
    
    async def store_bash_execution(
        self,
        command: str,
        exit_code: int,
        output: str,
        context: Dict[str, Any],
        importance: float = 0.4,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a bash execution record (deprecated - use enhanced_context instead)."""
        logger.warning("store_bash_execution is deprecated. Consider using store_enhanced_context with context_type='bash_execution'")
        return await self.store_memory(
            memory_type="bash_execution",
            content={
                "command": command,
                "exit_code": exit_code,
                "output": output[:1000] if output else "",  # Truncate long output
                "timestamp": context.get("timestamp", ""),
                "context": context
            },
            importance=importance,
            metadata=metadata
        )
