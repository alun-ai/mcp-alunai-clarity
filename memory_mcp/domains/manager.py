"""
Memory Domain Manager that orchestrates all memory operations.
"""

import uuid
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from memory_mcp.domains.episodic import EpisodicDomain
from memory_mcp.domains.semantic import SemanticDomain
from memory_mcp.domains.temporal import TemporalDomain
from memory_mcp.domains.persistence import PersistenceDomain


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
        
        # Initialize domains
        self.persistence_domain = PersistenceDomain(config)
        self.episodic_domain = EpisodicDomain(config, self.persistence_domain)
        self.semantic_domain = SemanticDomain(config, self.persistence_domain)
        self.temporal_domain = TemporalDomain(config, self.persistence_domain)
    
    async def initialize(self) -> None:
        """Initialize all domains."""
        logger.info("Initializing Memory Domain Manager")
        
        # Initialize domains in order (persistence first)
        await self.persistence_domain.initialize()
        await self.episodic_domain.initialize()
        await self.semantic_domain.initialize()
        await self.temporal_domain.initialize()
        
        logger.info("Memory Domain Manager initialized")
    
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
        
        # Process based on memory type
        if memory_type in ["conversation", "reflection"]:
            memory = await self.episodic_domain.process_memory(memory)
        elif memory_type in ["fact", "document", "entity"]:
            memory = await self.semantic_domain.process_memory(memory)
        elif memory_type == "code":
            # Code memories get processed by both domains
            memory = await self.episodic_domain.process_memory(memory)
            memory = await self.semantic_domain.process_memory(memory)
        
        # Determine memory tier based on importance and recency
        tier = "short_term"
        if importance < self.config["memory"].get("short_term_threshold", 0.3):
            tier = "long_term"
        
        # Store the memory
        await self.persistence_domain.store_memory(memory, tier)
        
        logger.info(f"Stored {memory_type} memory with ID {memory_id} in {tier} tier")
        
        return memory_id
    
    async def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_similarity: float = 0.6,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a query.
        
        Args:
            query: Query string
            limit: Maximum number of memories to retrieve
            memory_types: Types of memories to include (None for all types)
            min_similarity: Minimum similarity score for results
            include_metadata: Whether to include metadata in the results
            
        Returns:
            List of relevant memories
        """
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
            result_memory = {
                "id": memory["id"],
                "type": memory["type"],
                "content": memory["content"],
                "similarity": memory.get("similarity", 0.0)
            }
            
            # Include metadata if requested
            if include_metadata:
                result_memory["metadata"] = memory.get("metadata", {})
                result_memory["created_at"] = memory.get("created_at")
                result_memory["last_accessed"] = memory.get("last_accessed")
                result_memory["importance"] = memory.get("importance", 0.5)
            
            result_memories.append(result_memory)
        
        # Update access time for retrieved memories
        for memory in memories:
            await self.temporal_domain.update_memory_access(memory["id"])
        
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
            tier=tier
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
                result_memory["content"] = memory["content"]
            
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
            if updates["importance"] >= self.config["memory"].get("short_term_threshold", 0.3) and current_tier != "short_term":
                new_tier = "short_term"
            elif updates["importance"] < self.config["memory"].get("short_term_threshold", 0.3) and current_tier == "short_term":
                new_tier = "long_term"
        
        # Store the updated memory
        await self.persistence_domain.update_memory(memory, new_tier)
        
        logger.info(f"Updated memory {memory_id}")
        
        return True
    
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
        success = await self.persistence_domain.delete_memories(memory_ids)
        
        if success:
            logger.info(f"Deleted {len(memory_ids)} memories")
        else:
            logger.error(f"Failed to delete memories")
        
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
            "temporal_domain": temporal_stats
        })
        
        return stats
