"""
Domain registry implementation for dependency injection and service discovery.
"""

import threading
from typing import Dict, Optional, Any, List, Type, TypeVar, Generic
from loguru import logger

from clarity.shared.exceptions import ConfigurationError
from .interfaces import (
    DomainRegistry, 
    MemoryDomainInterface, 
    ServiceInterface,
    MemoryManagerInterface
)

T = TypeVar('T', bound=MemoryDomainInterface)


class DomainRegistryImpl(DomainRegistry):
    """Implementation of domain registry with dependency injection"""
    
    def __init__(self):
        self._domains: Dict[str, MemoryDomainInterface] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._initialized_domains: set = set()
        self._lock = threading.RLock()
        self._services: Dict[str, ServiceInterface] = {}
    
    def register_domain(self, name: str, domain: MemoryDomainInterface) -> None:
        """Register a domain with the registry"""
        with self._lock:
            if name in self._domains:
                logger.warning(f"Domain '{name}' is already registered, replacing")
            
            self._domains[name] = domain
            logger.debug(f"Registered domain: {name}")
    
    def register_service(self, name: str, service: ServiceInterface) -> None:
        """Register a service with the registry"""
        with self._lock:
            self._services[name] = service
            logger.debug(f"Registered service: {name}")
    
    def get_domain(self, name: str) -> Optional[MemoryDomainInterface]:
        """Get a registered domain"""
        with self._lock:
            return self._domains.get(name)
    
    def get_service(self, name: str) -> Optional[ServiceInterface]:
        """Get a registered service"""
        with self._lock:
            return self._services.get(name)
    
    def get_domain_typed(self, name: str, domain_type: Type[T]) -> Optional[T]:
        """Get a domain with type checking"""
        domain = self.get_domain(name)
        if domain and isinstance(domain, domain_type):
            return domain
        return None
    
    def get_all_domains(self) -> Dict[str, MemoryDomainInterface]:
        """Get all registered domains"""
        with self._lock:
            return self._domains.copy()
    
    def get_all_services(self) -> Dict[str, ServiceInterface]:
        """Get all registered services"""
        with self._lock:
            return self._services.copy()
    
    def remove_domain(self, name: str) -> bool:
        """Remove a domain from registry"""
        with self._lock:
            if name in self._domains:
                del self._domains[name]
                self._initialized_domains.discard(name)
                logger.debug(f"Removed domain: {name}")
                return True
            return False
    
    def remove_service(self, name: str) -> bool:
        """Remove a service from registry"""
        with self._lock:
            if name in self._services:
                del self._services[name]
                logger.debug(f"Removed service: {name}")
                return True
            return False
    
    def register_dependency(self, domain_name: str, depends_on: List[str]) -> None:
        """Register dependencies between domains"""
        with self._lock:
            self._dependencies[domain_name] = depends_on
            logger.debug(f"Registered dependencies for {domain_name}: {depends_on}")
    
    def get_initialization_order(self) -> List[str]:
        """Get domains in dependency initialization order"""
        with self._lock:
            # Topological sort for dependency order
            visited = set()
            temp_visited = set()
            result = []
            
            def visit(domain: str):
                if domain in temp_visited:
                    raise ConfigurationError(f"Circular dependency detected involving {domain}")
                if domain in visited:
                    return
                
                temp_visited.add(domain)
                
                # Visit dependencies first
                for dependency in self._dependencies.get(domain, []):
                    if dependency in self._domains:
                        visit(dependency)
                
                temp_visited.remove(domain)
                visited.add(domain)
                result.append(domain)
            
            # Visit all domains
            for domain_name in self._domains.keys():
                if domain_name not in visited:
                    visit(domain_name)
            
            return result
    
    async def initialize_all(self) -> None:
        """Initialize all domains in dependency order"""
        with self._lock:
            initialization_order = self.get_initialization_order()
        
        logger.info(f"Initializing domains in order: {initialization_order}")
        
        for domain_name in initialization_order:
            if domain_name not in self._initialized_domains:
                domain = self._domains.get(domain_name)
                if domain:
                    try:
                        logger.debug(f"Initializing domain: {domain_name}")
                        await domain.initialize()
                        self._initialized_domains.add(domain_name)
                        logger.debug(f"Domain {domain_name} initialized successfully")
                    except (RuntimeError, AttributeError, ImportError, ConfigurationError) as e:
                        logger.error(f"Failed to initialize domain {domain_name}: {e}")
                        raise ConfigurationError(f"Domain initialization failed: {domain_name}", cause=e)
        
        # Initialize services
        for service_name, service in self._services.items():
            try:
                logger.debug(f"Initializing service: {service_name}")
                await service.initialize()
                logger.debug(f"Service {service_name} initialized successfully")
            except (RuntimeError, AttributeError, ImportError, ConfigurationError) as e:
                logger.error(f"Failed to initialize service {service_name}: {e}")
                raise ConfigurationError(f"Service initialization failed: {service_name}", cause=e)
    
    async def shutdown_all(self) -> None:
        """Shutdown all domains in reverse dependency order"""
        with self._lock:
            initialization_order = self.get_initialization_order()
        
        # Shutdown in reverse order
        shutdown_order = list(reversed(initialization_order))
        logger.info(f"Shutting down domains in order: {shutdown_order}")
        
        # Shutdown services first
        for service_name, service in self._services.items():
            try:
                logger.debug(f"Shutting down service: {service_name}")
                await service.shutdown()
            except (RuntimeError, AttributeError, ValueError) as e:
                logger.warning(f"Error shutting down service {service_name}: {e}")
        
        # Then shutdown domains
        for domain_name in shutdown_order:
            if domain_name in self._initialized_domains:
                domain = self._domains.get(domain_name)
                if domain:
                    try:
                        logger.debug(f"Shutting down domain: {domain_name}")
                        await domain.shutdown()
                        self._initialized_domains.remove(domain_name)
                    except (RuntimeError, AttributeError, ValueError) as e:
                        logger.warning(f"Error shutting down domain {domain_name}: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            return {
                "total_domains": len(self._domains),
                "initialized_domains": len(self._initialized_domains),
                "total_services": len(self._services),
                "registered_domains": list(self._domains.keys()),
                "registered_services": list(self._services.keys()),
                "dependencies": self._dependencies.copy(),
                "initialization_order": self.get_initialization_order()
            }
    
    def validate_dependencies(self) -> List[str]:
        """Validate all dependencies are satisfied"""
        issues = []
        
        with self._lock:
            for domain_name, dependencies in self._dependencies.items():
                if domain_name not in self._domains:
                    issues.append(f"Domain {domain_name} has dependencies but is not registered")
                    continue
                
                for dep in dependencies:
                    if dep not in self._domains:
                        issues.append(f"Domain {domain_name} depends on {dep}, but {dep} is not registered")
        
        return issues
    
    def is_domain_initialized(self, name: str) -> bool:
        """Check if a domain is initialized"""
        with self._lock:
            return name in self._initialized_domains
    
    def get_domain_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all domains"""
        health_status = {}
        
        with self._lock:
            domains = self._domains.copy()
            services = self._services.copy()
        
        # Check domain health
        for name, domain in domains.items():
            try:
                health_info = {
                    "registered": True,
                    "initialized": name in self._initialized_domains,
                    "type": type(domain).__name__,
                    "healthy": True
                }
                
                # Get domain info if available
                if hasattr(domain, 'get_domain_info'):
                    health_info["info"] = domain.get_domain_info()
                
                # Check if domain has health check method
                if hasattr(domain, 'is_healthy'):
                    health_info["healthy"] = domain.is_healthy()
                
                health_status[name] = health_info
                
            except (RuntimeError, AttributeError, ValueError, TypeError) as e:
                health_status[name] = {
                    "registered": True,
                    "initialized": False,
                    "healthy": False,
                    "error": str(e)
                }
        
        # Check service health
        for name, service in services.items():
            service_name = f"service_{name}"
            try:
                health_info = {
                    "registered": True,
                    "type": type(service).__name__,
                    "healthy": service.is_healthy() if hasattr(service, 'is_healthy') else True
                }
                
                if hasattr(service, 'get_service_info'):
                    health_info["info"] = service.get_service_info()
                
                health_status[service_name] = health_info
                
            except (RuntimeError, AttributeError, ValueError, TypeError) as e:
                health_status[service_name] = {
                    "registered": True,
                    "healthy": False,
                    "error": str(e)
                }
        
        return health_status


class DependencyInjector:
    """Dependency injection container for managing object creation and dependencies"""
    
    def __init__(self):
        self._instances: Dict[Type, Any] = {}
        self._factories: Dict[Type, callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.Lock()
    
    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """Register a singleton instance"""
        with self._lock:
            self._singletons[interface] = instance
    
    def register_factory(self, interface: Type[T], factory: callable) -> None:
        """Register a factory function for creating instances"""
        with self._lock:
            self._factories[interface] = factory
    
    def get(self, interface: Type[T]) -> T:
        """Get an instance of the requested interface"""
        with self._lock:
            # Check singletons first
            if interface in self._singletons:
                return self._singletons[interface]
            
            # Check if we have a cached instance
            if interface in self._instances:
                return self._instances[interface]
            
            # Use factory to create instance
            if interface in self._factories:
                instance = self._factories[interface]()
                self._instances[interface] = instance
                return instance
            
            raise ConfigurationError(f"No registration found for interface: {interface}")
    
    def create_new(self, interface: Type[T]) -> T:
        """Create a new instance (ignoring cache)"""
        with self._lock:
            if interface in self._factories:
                return self._factories[interface]()
            
            raise ConfigurationError(f"No factory found for interface: {interface}")
    
    def clear(self) -> None:
        """Clear all registrations"""
        with self._lock:
            self._instances.clear()
            self._factories.clear()
            self._singletons.clear()


# Global registry instance
domain_registry = DomainRegistryImpl()
dependency_injector = DependencyInjector()