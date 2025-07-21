"""
Adapter pattern implementations to decouple domain dependencies.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from loguru import logger

from clarity.shared.exceptions import AutoCodeError, MemoryOperationError
from .interfaces import (
    MemoryStorageInterface,
    EmbeddingInterface, 
    PersistenceDomainInterface,
    EpisodicDomainInterface,
    SemanticDomainInterface,
    TemporalDomainInterface,
    AutoCodeDomainInterface
)


class MemoryStorageAdapter(MemoryStorageInterface):
    """Adapter for memory storage operations that delegates to actual persistence domain"""
    
    def __init__(self, persistence_domain):
        self._persistence_domain = persistence_domain
    
    async def store_memory(self, 
                          memory_type: str, 
                          content: Union[str, Dict[str, Any]], 
                          importance: float = 0.5,
                          metadata: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Store memory using the persistence domain"""
        try:
            return await self._persistence_domain.store_memory(
                memory_type, content, importance, metadata, context
            )
        except (MemoryOperationError, QdrantConnectionError, ConnectionError, ValueError) as e:
            logger.error(f"Storage adapter error: {e}")
            raise MemoryOperationError("Failed to store memory", cause=e)
    
    async def retrieve_memories(self,
                               query: str,
                               limit: int = 5,
                               types: Optional[List[str]] = None,
                               min_similarity: float = 0.6,
                               include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Retrieve memories using the persistence domain"""
        try:
            return await self._persistence_domain.retrieve_memories(
                query, limit, types, min_similarity, include_metadata
            )
        except (MemoryOperationError, QdrantConnectionError, ConnectionError, ValueError) as e:
            logger.error(f"Retrieval adapter error: {e}")
            raise MemoryOperationError("Failed to retrieve memories", cause=e)
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get specific memory using the persistence domain"""
        try:
            return await self._persistence_domain.get_memory(memory_id)
        except (MemoryOperationError, QdrantConnectionError, ConnectionError, KeyError) as e:
            logger.error(f"Get memory adapter error: {e}")
            raise MemoryOperationError("Failed to get memory", cause=e)
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory using the persistence domain"""
        try:
            return await self._persistence_domain.update_memory(memory_id, updates)
        except (MemoryOperationError, QdrantConnectionError, ConnectionError, ValueError, KeyError) as e:
            logger.error(f"Update memory adapter error: {e}")
            raise MemoryOperationError("Failed to update memory", cause=e)
    
    async def delete_memories(self, memory_ids: List[str]) -> List[str]:
        """Delete memories using the persistence domain"""
        try:
            return await self._persistence_domain.delete_memories(memory_ids)
        except (MemoryOperationError, QdrantConnectionError, ConnectionError, KeyError) as e:
            logger.error(f"Delete memories adapter error: {e}")
            raise MemoryOperationError("Failed to delete memories", cause=e)


class EmbeddingAdapter(EmbeddingInterface):
    """Adapter for embedding generation that delegates to persistence domain"""
    
    def __init__(self, persistence_domain):
        self._persistence_domain = persistence_domain
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the persistence domain"""
        try:
            return await self._persistence_domain.generate_embedding(text)
        except (RuntimeError, OSError, ValueError, ImportError) as e:
            logger.error(f"Embedding adapter error: {e}")
            raise MemoryOperationError("Failed to generate embedding", cause=e)
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            # If persistence domain has batch method, use it
            if hasattr(self._persistence_domain, 'generate_embeddings_batch'):
                return await self._persistence_domain.generate_embeddings_batch(texts)
            else:
                # Fall back to individual generation
                embeddings = []
                for text in texts:
                    embedding = await self.generate_embedding(text)
                    embeddings.append(embedding)
                return embeddings
        except (RuntimeError, OSError, ValueError, ImportError, MemoryError) as e:
            logger.error(f"Batch embedding adapter error: {e}")
            raise MemoryOperationError("Failed to generate batch embeddings", cause=e)


class PersistenceDomainAdapter(PersistenceDomainInterface):
    """Adapter that provides the full persistence interface"""
    
    def __init__(self, persistence_domain):
        self._domain = persistence_domain
        self._storage_adapter = MemoryStorageAdapter(persistence_domain)
        self._embedding_adapter = EmbeddingAdapter(persistence_domain)
    
    async def initialize(self) -> None:
        """Initialize the persistence domain"""
        await self._domain.initialize()
    
    async def shutdown(self) -> None:
        """Shutdown the persistence domain"""
        if hasattr(self._domain, 'shutdown'):
            await self._domain.shutdown()
    
    def get_domain_info(self) -> Dict[str, Any]:
        """Get domain information"""
        return {
            "name": "PersistenceDomainAdapter",
            "wrapped_type": type(self._domain).__name__,
            "initialized": getattr(self._domain, '_initialized', False)
        }
    
    # Delegate storage operations
    async def store_memory(self, *args, **kwargs) -> str:
        return await self._storage_adapter.store_memory(*args, **kwargs)
    
    async def retrieve_memories(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return await self._storage_adapter.retrieve_memories(*args, **kwargs)
    
    async def get_memory(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        return await self._storage_adapter.get_memory(*args, **kwargs)
    
    async def update_memory(self, *args, **kwargs) -> bool:
        return await self._storage_adapter.update_memory(*args, **kwargs)
    
    async def delete_memories(self, *args, **kwargs) -> List[str]:
        return await self._storage_adapter.delete_memories(*args, **kwargs)
    
    # Delegate embedding operations
    async def generate_embedding(self, *args, **kwargs) -> List[float]:
        return await self._embedding_adapter.generate_embedding(*args, **kwargs)
    
    async def generate_embeddings_batch(self, *args, **kwargs) -> List[List[float]]:
        return await self._embedding_adapter.generate_embeddings_batch(*args, **kwargs)
    
    # Persistence-specific operations
    async def search_memories(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return await self._domain.search_memories(*args, **kwargs)
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        return await self._domain.get_memory_stats()
    
    async def optimize_collection(self) -> bool:
        return await self._domain.optimize_collection()


class EpisodicDomainAdapter(EpisodicDomainInterface):
    """Adapter for episodic domain with interface compliance"""
    
    def __init__(self, episodic_domain):
        self._domain = episodic_domain
    
    async def initialize(self) -> None:
        await self._domain.initialize()
    
    async def shutdown(self) -> None:
        if hasattr(self._domain, 'shutdown'):
            await self._domain.shutdown()
    
    def get_domain_info(self) -> Dict[str, Any]:
        return {
            "name": "EpisodicDomainAdapter",
            "wrapped_type": type(self._domain).__name__,
            "initialized": getattr(self._domain, '_initialized', False)
        }
    
    async def process_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory for episodic storage"""
        try:
            return await self._domain.process_memory(memory)
        except (MemoryOperationError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Episodic domain adapter error: {e}")
            raise MemoryOperationError("Failed to process episodic memory", cause=e)
    
    async def get_episodic_context(self, current_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Get episodic context for current memory"""
        try:
            # If the domain has this method, use it
            if hasattr(self._domain, 'get_episodic_context'):
                return await self._domain.get_episodic_context(current_memory)
            else:
                # Provide basic episodic context
                return {
                    "recent_memories": [],
                    "related_episodes": [],
                    "temporal_context": datetime.utcnow().isoformat()
                }
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Episodic context adapter error: {e}")
            return {"error": str(e)}


class SemanticDomainAdapter(SemanticDomainInterface):
    """Adapter for semantic domain with interface compliance"""
    
    def __init__(self, semantic_domain):
        self._domain = semantic_domain
    
    async def initialize(self) -> None:
        await self._domain.initialize()
    
    async def shutdown(self) -> None:
        if hasattr(self._domain, 'shutdown'):
            await self._domain.shutdown()
    
    def get_domain_info(self) -> Dict[str, Any]:
        return {
            "name": "SemanticDomainAdapter",
            "wrapped_type": type(self._domain).__name__,
            "initialized": getattr(self._domain, '_initialized', False)
        }
    
    async def process_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory for semantic storage"""
        try:
            return await self._domain.process_memory(memory)
        except (MemoryOperationError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Semantic domain adapter error: {e}")
            raise MemoryOperationError("Failed to process semantic memory", cause=e)
    
    async def extract_concepts(self, memory: Dict[str, Any]) -> List[str]:
        """Extract semantic concepts from memory"""
        try:
            if hasattr(self._domain, 'extract_concepts'):
                return await self._domain.extract_concepts(memory)
            else:
                # Basic concept extraction from content
                content = memory.get('content', '')
                if isinstance(content, dict):
                    content = str(content)
                
                # Simple keyword extraction (would be enhanced in real implementation)
                words = content.lower().split()
                concepts = [word for word in words if len(word) > 4][:5]
                return concepts
        except (AttributeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Concept extraction adapter error: {e}")
            return []
    
    async def find_related_concepts(self, concept: str) -> List[Dict[str, Any]]:
        """Find memories related to a concept"""
        try:
            if hasattr(self._domain, 'find_related_concepts'):
                return await self._domain.find_related_concepts(concept)
            else:
                # Basic concept search using the persistence domain
                return []
        except (AttributeError, ValueError, MemoryOperationError) as e:
            logger.error(f"Related concepts adapter error: {e}")
            return []


class TemporalDomainAdapter(TemporalDomainInterface):
    """Adapter for temporal domain with interface compliance"""
    
    def __init__(self, temporal_domain):
        self._domain = temporal_domain
    
    async def initialize(self) -> None:
        await self._domain.initialize()
    
    async def shutdown(self) -> None:
        if hasattr(self._domain, 'shutdown'):
            await self._domain.shutdown()
    
    def get_domain_info(self) -> Dict[str, Any]:
        return {
            "name": "TemporalDomainAdapter",
            "wrapped_type": type(self._domain).__name__,
            "initialized": getattr(self._domain, '_initialized', False)
        }
    
    async def process_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory for temporal storage"""
        try:
            return await self._domain.process_memory(memory)
        except (MemoryOperationError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Temporal domain adapter error: {e}")
            raise MemoryOperationError("Failed to process temporal memory", cause=e)
    
    async def get_temporal_context(self, 
                                  timestamp: datetime,
                                  window_hours: int = 24) -> Dict[str, Any]:
        """Get temporal context around a timestamp"""
        try:
            if hasattr(self._domain, 'get_temporal_context'):
                return await self._domain.get_temporal_context(timestamp, window_hours)
            else:
                # Provide basic temporal context
                return {
                    "timestamp": timestamp.isoformat(),
                    "window_hours": window_hours,
                    "related_memories": []
                }
        except (AttributeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Temporal context adapter error: {e}")
            return {"error": str(e)}
    
    async def get_memory_timeline(self, 
                                 start_time: datetime,
                                 end_time: datetime) -> List[Dict[str, Any]]:
        """Get memories within a time range"""
        try:
            if hasattr(self._domain, 'get_memory_timeline'):
                return await self._domain.get_memory_timeline(start_time, end_time)
            else:
                # Basic timeline implementation
                return []
        except (AttributeError, ValueError, MemoryOperationError) as e:
            logger.error(f"Memory timeline adapter error: {e}")
            return []


class AutoCodeDomainAdapter(AutoCodeDomainInterface):
    """Adapter for AutoCode domain with interface compliance"""
    
    def __init__(self, autocode_domain):
        self._domain = autocode_domain
    
    async def initialize(self) -> None:
        await self._domain.initialize()
    
    async def shutdown(self) -> None:
        if hasattr(self._domain, 'shutdown'):
            await self._domain.shutdown()
    
    def get_domain_info(self) -> Dict[str, Any]:
        return {
            "name": "AutoCodeDomainAdapter",
            "wrapped_type": type(self._domain).__name__,
            "initialized": getattr(self._domain, '_initialized', False)
        }
    
    async def process_file_access(self, 
                                 file_path: str, 
                                 access_type: str,
                                 project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process file access events"""
        try:
            await self._domain.process_file_access(file_path, access_type, project_context)
        except (AttributeError, OSError, ValueError) as e:
            logger.warning(f"File access adapter error: {e}")
    
    async def process_bash_execution(self,
                                    command: str,
                                    working_directory: str,
                                    success: bool,
                                    output: str,
                                    project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process bash command execution"""
        try:
            await self._domain.process_bash_execution(
                command, working_directory, success, output, project_context
            )
        except (AttributeError, ValueError, OSError) as e:
            logger.warning(f"Bash execution adapter error: {e}")
    
    async def suggest_command(self, 
                             intent: str, 
                             context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Suggest commands based on intent"""
        try:
            return await self._domain.suggest_command(intent, context)
        except (AutoCodeError, AttributeError, ValueError) as e:
            logger.error(f"Command suggestion adapter error: {e}")
            raise AutoCodeError("Failed to suggest command", cause=e)
    
    async def get_project_patterns(self, 
                                  project_path: str,
                                  pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get patterns for a project"""
        try:
            return await self._domain.get_project_patterns(project_path, pattern_types)
        except (AutoCodeError, AttributeError, OSError, ValueError) as e:
            logger.error(f"Project patterns adapter error: {e}")
            raise AutoCodeError("Failed to get project patterns", cause=e)
    
    async def generate_session_summary(self, conversation_log: List[Dict]) -> str:
        """Generate session summary"""
        try:
            return await self._domain.generate_session_summary(conversation_log)
        except (AutoCodeError, AttributeError, ValueError, KeyError) as e:
            logger.error(f"Session summary adapter error: {e}")
            raise AutoCodeError("Failed to generate session summary", cause=e)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get AutoCode statistics"""
        try:
            return await self._domain.get_stats()
        except (AttributeError, ValueError, MemoryOperationError) as e:
            logger.error(f"AutoCode stats adapter error: {e}")
            return {"error": str(e)}


# Factory functions for creating adapters
def create_persistence_adapter(persistence_domain) -> PersistenceDomainAdapter:
    """Create a persistence domain adapter"""
    return PersistenceDomainAdapter(persistence_domain)


def create_episodic_adapter(episodic_domain) -> EpisodicDomainAdapter:
    """Create an episodic domain adapter"""
    return EpisodicDomainAdapter(episodic_domain)


def create_semantic_adapter(semantic_domain) -> SemanticDomainAdapter:
    """Create a semantic domain adapter"""
    return SemanticDomainAdapter(semantic_domain)


def create_temporal_adapter(temporal_domain) -> TemporalDomainAdapter:
    """Create a temporal domain adapter"""
    return TemporalDomainAdapter(temporal_domain)


def create_autocode_adapter(autocode_domain) -> AutoCodeDomainAdapter:
    """Create an AutoCode domain adapter"""
    return AutoCodeDomainAdapter(autocode_domain)