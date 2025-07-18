"""
Temporal Domain for time-aware memory processing.

The Temporal Domain is responsible for:
- Managing memory decay and importance over time
- Temporal indexing and sequencing
- Chronological relationship tracking
- Time-based memory consolidation
- Recency effects in retrieval
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

from loguru import logger

from memory_mcp.domains.persistence import PersistenceDomain


class TemporalDomain:
    """
    Manages time-aware memory processing.
    
    This domain handles temporal aspects of memory, including
    decay over time, recency-based relevance, and time-based
    consolidation of memories.
    """
    
    def __init__(self, config: Dict[str, Any], persistence_domain: PersistenceDomain) -> None:
        """
        Initialize the temporal domain.
        
        Args:
            config: Configuration dictionary
            persistence_domain: Reference to the persistence domain
        """
        self.config = config
        self.persistence_domain = persistence_domain
        self.last_consolidation = datetime.now()
    
    async def initialize(self) -> None:
        """Initialize the temporal domain."""
        logger.info("Initializing Temporal Domain")
        
        # Schedule initial consolidation if needed
        consolidation_interval = self.config["memory"].get("consolidation_interval_hours", 24)
        self.consolidation_interval = timedelta(hours=consolidation_interval)
        
        # Get last consolidation time from persistence
        last_consolidation = await self.persistence_domain.get_metadata("last_consolidation")
        if last_consolidation:
            try:
                self.last_consolidation = datetime.fromisoformat(last_consolidation)
            except ValueError:
                logger.warning(f"Invalid last_consolidation timestamp: {last_consolidation}")
                self.last_consolidation = datetime.now()
        
        # Check if consolidation is due
        if datetime.now() - self.last_consolidation > self.consolidation_interval:
            logger.info("Consolidation is due. Will run after initialization.")
            # Note: We don't run consolidation here to avoid slow startup
            # It will run on the next memory operation
    
    async def process_new_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new memory with temporal information.
        
        Args:
            memory: The memory to process
            
        Returns:
            Processed memory with temporal information
        """
        # Add timestamps
        now = datetime.now().isoformat()
        memory["created_at"] = now
        memory["last_accessed"] = now
        memory["last_modified"] = now
        memory["access_count"] = 0
        
        return memory
    
    async def update_memory_access(self, memory_id: str) -> None:
        """
        Update the access time for a memory.
        
        Args:
            memory_id: ID of the memory to update
        """
        # Get the memory
        memory = await self.persistence_domain.get_memory(memory_id)
        if not memory:
            logger.warning(f"Memory {memory_id} not found for access update")
            return
        
        # Update access time and count
        memory["last_accessed"] = datetime.now().isoformat()
        memory["access_count"] = memory.get("access_count", 0) + 1
        
        # Save the updated memory
        current_tier = await self.persistence_domain.get_memory_tier(memory_id)
        await self.persistence_domain.update_memory(memory, current_tier)
        
        # Check if consolidation is due
        await self._check_consolidation()
    
    async def update_memory_modification(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the modification time for a memory.
        
        Args:
            memory: The memory to update
            
        Returns:
            Updated memory
        """
        memory["last_modified"] = datetime.now().isoformat()
        return memory
    
    async def adjust_memory_relevance(
        self,
        memories: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Adjust memory relevance based on temporal factors.
        
        Args:
            memories: List of memories to adjust
            query: The query string
            
        Returns:
            Adjusted memories
        """
        # Weight configuration
        recency_weight = self.config["memory"].get("retrieval", {}).get("recency_weight", 0.3)
        importance_weight = self.config["memory"].get("retrieval", {}).get("importance_weight", 0.7)
        
        now = datetime.now()
        adjusted_memories = []
        
        for memory in memories:
            # Calculate recency score
            last_accessed_str = memory.get("last_accessed", memory.get("created_at"))
            try:
                last_accessed = datetime.fromisoformat(last_accessed_str)
                days_since_access = (now - last_accessed).days
                # Recency score: 1.0 for just accessed, decreasing with time
                recency_score = 1.0 / (1.0 + days_since_access)
            except (ValueError, TypeError):
                recency_score = 0.5  # Default if timestamp is invalid
            
            # Get importance score
            importance_score = memory.get("importance", 0.5)
            
            # Get similarity score (from semantic search)
            similarity_score = memory.get("similarity", 0.5)
            
            # Combine scores
            combined_score = (
                similarity_score * (1.0 - recency_weight - importance_weight) +
                recency_score * recency_weight +
                importance_score * importance_weight
            )
            
            # Update memory with combined score
            memory["adjusted_score"] = combined_score
            memory["recency_score"] = recency_score
            
            adjusted_memories.append(memory)
        
        # Sort by combined score
        adjusted_memories.sort(key=lambda m: m["adjusted_score"], reverse=True)
        
        return adjusted_memories
    
    async def _check_consolidation(self) -> None:
        """Check if memory consolidation is due and run if needed."""
        now = datetime.now()
        
        # Check if enough time has passed since last consolidation
        if now - self.last_consolidation > self.consolidation_interval:
            logger.info("Running memory consolidation")
            await self._consolidate_memories()
            
            # Update last consolidation time
            self.last_consolidation = now
            await self.persistence_domain.set_metadata("last_consolidation", now.isoformat())
    
    async def _consolidate_memories(self) -> None:
        """
        Consolidate memories based on temporal patterns.
        
        This includes:
        - Moving old short-term memories to long-term
        - Archiving rarely accessed long-term memories
        - Adjusting importance scores based on access patterns
        """
        # Placeholder for consolidation logic
        logger.info("Memory consolidation not yet implemented")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the temporal domain.
        
        Returns:
            Temporal domain statistics
        """
        return {
            "last_consolidation": self.last_consolidation.isoformat(),
            "next_consolidation": (self.last_consolidation + self.consolidation_interval).isoformat(),
            "status": "initialized"
        }
