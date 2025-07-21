"""
Domain interfaces for reducing coupling between domains.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncContextManager
from datetime import datetime


class MemoryStorageInterface(ABC):
    """Interface for memory storage operations"""
    
    @abstractmethod
    async def store_memory(self, 
                          memory_type: str, 
                          content: Union[str, Dict[str, Any]], 
                          importance: float = 0.5,
                          metadata: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory and return its ID"""
        pass
    
    @abstractmethod
    async def retrieve_memories(self,
                               query: str,
                               limit: int = 5,
                               types: Optional[List[str]] = None,
                               min_similarity: float = 0.6,
                               include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Retrieve memories based on query"""
        pass
    
    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        pass
    
    @abstractmethod
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory"""
        pass
    
    @abstractmethod
    async def delete_memories(self, memory_ids: List[str]) -> List[str]:
        """Delete memories by IDs"""
        pass


class EmbeddingInterface(ABC):
    """Interface for embedding generation"""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass


class MemoryDomainInterface(ABC):
    """Base interface for memory domains"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the domain"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the domain"""
        pass
    
    @abstractmethod
    def get_domain_info(self) -> Dict[str, Any]:
        """Get domain information"""
        pass


class EpisodicDomainInterface(MemoryDomainInterface):
    """Interface for episodic memory domain"""
    
    @abstractmethod
    async def process_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory for episodic storage"""
        pass
    
    @abstractmethod
    async def get_episodic_context(self, current_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Get episodic context for current memory"""
        pass


class SemanticDomainInterface(MemoryDomainInterface):
    """Interface for semantic memory domain"""
    
    @abstractmethod
    async def process_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory for semantic storage"""
        pass
    
    @abstractmethod
    async def extract_concepts(self, memory: Dict[str, Any]) -> List[str]:
        """Extract semantic concepts from memory"""
        pass
    
    @abstractmethod
    async def find_related_concepts(self, concept: str) -> List[Dict[str, Any]]:
        """Find memories related to a concept"""
        pass


class TemporalDomainInterface(MemoryDomainInterface):
    """Interface for temporal memory domain"""
    
    @abstractmethod
    async def process_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory for temporal storage"""
        pass
    
    @abstractmethod
    async def get_temporal_context(self, 
                                  timestamp: datetime,
                                  window_hours: int = 24) -> Dict[str, Any]:
        """Get temporal context around a timestamp"""
        pass
    
    @abstractmethod
    async def get_memory_timeline(self, 
                                 start_time: datetime,
                                 end_time: datetime) -> List[Dict[str, Any]]:
        """Get memories within a time range"""
        pass


class PersistenceDomainInterface(MemoryDomainInterface, MemoryStorageInterface, EmbeddingInterface):
    """Interface for persistence domain"""
    
    @abstractmethod
    async def search_memories(self,
                             embedding: Optional[List[float]] = None,
                             limit: int = 10,
                             types: Optional[List[str]] = None,
                             min_similarity: float = 0.0,
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search memories using embeddings"""
        pass
    
    @abstractmethod
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics"""
        pass
    
    @abstractmethod
    async def optimize_collection(self) -> bool:
        """Optimize the underlying storage"""
        pass


class AutoCodeDomainInterface(MemoryDomainInterface):
    """Interface for AutoCode domain"""
    
    @abstractmethod
    async def process_file_access(self, 
                                 file_path: str, 
                                 access_type: str,
                                 project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process file access events"""
        pass
    
    @abstractmethod
    async def process_bash_execution(self,
                                    command: str,
                                    working_directory: str,
                                    success: bool,
                                    output: str,
                                    project_context: Optional[Dict[str, Any]] = None) -> None:
        """Process bash command execution"""
        pass
    
    @abstractmethod
    async def suggest_command(self, 
                             intent: str, 
                             context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Suggest commands based on intent"""
        pass
    
    @abstractmethod
    async def get_project_patterns(self, 
                                  project_path: str,
                                  pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get patterns for a project"""
        pass
    
    @abstractmethod
    async def generate_session_summary(self, conversation_log: List[Dict]) -> str:
        """Generate session summary"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get AutoCode statistics"""
        pass


class MemoryManagerInterface(ABC):
    """Interface for the main memory domain manager"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize all domains"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown all domains"""
        pass
    
    @abstractmethod
    async def store_memory(self,
                          memory_type: str,
                          content: Union[str, Dict[str, Any]],
                          importance: float = 0.5,
                          metadata: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Store memory across appropriate domains"""
        pass
    
    @abstractmethod
    async def retrieve_memories(self,
                               query: str,
                               limit: int = 5,
                               types: Optional[List[str]] = None,
                               min_similarity: float = 0.6,
                               include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Retrieve memories across domains"""
        pass
    
    @abstractmethod
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory across domains"""
        pass
    
    @abstractmethod
    async def delete_memories(self, memory_ids: List[str]) -> List[str]:
        """Delete memories across domains"""
        pass
    
    @abstractmethod
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        pass


class DomainRegistry(ABC):
    """Registry for domain management with dependency injection"""
    
    @abstractmethod
    def register_domain(self, name: str, domain: MemoryDomainInterface) -> None:
        """Register a domain"""
        pass
    
    @abstractmethod
    def get_domain(self, name: str) -> Optional[MemoryDomainInterface]:
        """Get a registered domain"""
        pass
    
    @abstractmethod
    def get_all_domains(self) -> Dict[str, MemoryDomainInterface]:
        """Get all registered domains"""
        pass
    
    @abstractmethod
    def remove_domain(self, name: str) -> bool:
        """Remove a domain from registry"""
        pass


class ServiceInterface(ABC):
    """Interface for service layer components"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service"""
        pass
    
    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        pass


class CacheInterface(ABC):
    """Interface for caching operations"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class ConnectionInterface(ABC):
    """Interface for database connections"""
    
    @abstractmethod
    async def acquire(self) -> AsyncContextManager[Any]:
        """Acquire a connection from the pool"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the connection"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        pass


class EventInterface(ABC):
    """Interface for event handling"""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: callable) -> str:
        """Subscribe to an event type"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        pass


class ConfigurationInterface(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        pass
    
    @abstractmethod
    def reload(self) -> None:
        """Reload configuration from source"""
        pass
    
    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        pass


class MetricsInterface(ABC):
    """Interface for metrics collection"""
    
    @abstractmethod
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        pass
    
    @abstractmethod
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        pass