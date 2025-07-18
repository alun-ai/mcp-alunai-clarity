"""
Episodic Domain for managing episodic memories.

The Episodic Domain is responsible for:
- Recording and retrieving conversation histories
- Managing session-based interactions
- Contextualizing memories with temporal and situational details
- Narrative memory construction
- Recording agent reflections and observations
"""

from typing import Any, Dict, List

from loguru import logger

from memory_mcp.domains.persistence import PersistenceDomain


class EpisodicDomain:
    """
    Manages episodic memories (conversations, experiences, reflections).
    
    This domain handles memories that are experiential in nature,
    including conversation histories, reflections, and interactions.
    """
    
    def __init__(self, config: Dict[str, Any], persistence_domain: PersistenceDomain) -> None:
        """
        Initialize the episodic domain.
        
        Args:
            config: Configuration dictionary
            persistence_domain: Reference to the persistence domain
        """
        self.config = config
        self.persistence_domain = persistence_domain
    
    async def initialize(self) -> None:
        """Initialize the episodic domain."""
        logger.info("Initializing Episodic Domain")
        # Initialization logic will be implemented here
    
    async def process_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an episodic memory.
        
        This includes extracting key information, generating embeddings,
        and enriching the memory with additional metadata.
        
        Args:
            memory: The memory to process
            
        Returns:
            Processed memory
        """
        logger.debug(f"Processing episodic memory: {memory['id']}")
        
        # Extract text representation for embedding
        text_content = self._extract_text_content(memory)
        
        # Generate embedding
        embedding = await self.persistence_domain.generate_embedding(text_content)
        memory["embedding"] = embedding
        
        # Additional processing will be implemented here
        
        return memory
    
    def _extract_text_content(self, memory: Dict[str, Any]) -> str:
        """
        Extract text content from a memory for embedding generation.
        
        Args:
            memory: The memory to extract text from
            
        Returns:
            Text representation of the memory
        """
        if memory["type"] == "conversation":
            # For conversation memories, extract from the message content
            if "role" in memory["content"] and "message" in memory["content"]:
                return f"{memory['content']['role']}: {memory['content']['message']}"
                
            # Handle conversation arrays
            if "messages" in memory["content"]:
                messages = memory["content"]["messages"]
                if isinstance(messages, list):
                    return "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
        
        elif memory["type"] == "reflection":
            # For reflection memories, combine subject and reflection
            if "subject" in memory["content"] and "reflection" in memory["content"]:
                return f"{memory['content']['subject']}: {memory['content']['reflection']}"
        
        # Fallback: try to convert content to string
        try:
            return str(memory["content"])
        except:
            return f"Memory {memory['id']} of type {memory['type']}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the episodic domain.
        
        Returns:
            Episodic domain statistics
        """
        return {
            "memory_types": {
                "conversation": 0,
                "reflection": 0
            },
            "status": "initialized"
        }
