"""
Comprehensive audit trail system for tracking operations and security events.

This module provides:
- Audit event tracking with structured data
- Security event monitoring
- Performance audit trails
- Data access audit logging
- Compliance and observability features
"""

import uuid
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from loguru import logger

from .utils.logging_utils import get_logger
from .async_utils import AsyncBatcher, async_timed


class AuditEventType(Enum):
    """Types of audit events"""
    # Security events
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION = "permission_escalation"
    
    # Data operations
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # System operations
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGE = "configuration_change"
    
    # Business operations
    MEMORY_OPERATION = "memory_operation"
    COMMAND_EXECUTION = "command_execution"
    FILE_ACCESS = "file_access"
    PROJECT_ANALYSIS = "project_analysis"
    
    # Performance events
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"
    RESOURCE_LIMIT_REACHED = "resource_limit_reached"
    
    # Error events
    ERROR_OCCURRED = "error_occurred"
    EXCEPTION_RAISED = "exception_raised"
    RECOVERY_ACTION = "recovery_action"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Structured audit event data"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    actor: str  # Who performed the action
    resource: str  # What was affected
    action: str  # What was done
    outcome: str  # Success, failure, etc.
    details: Dict[str, Any]
    context: Dict[str, Any]
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditTrailStorage:
    """Storage backend for audit events"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize audit storage
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._events_buffer: List[AuditEvent] = []
        self._buffer_size = config.get('buffer_size', 100)
        self._flush_interval = config.get('flush_interval_seconds', 30)
        self._storage_backend = config.get('storage_backend', 'memory')
        self._last_flush = time.time()
        
        # Initialize storage backend
        if self._storage_backend == 'file':
            self._init_file_storage()
        elif self._storage_backend == 'database':
            self._init_database_storage()
    
    def _init_file_storage(self) -> None:
        """Initialize file-based storage"""
        import os
        from pathlib import Path
        
        self._audit_file_path = self.config.get('audit_file_path', 'logs/audit_trail.jsonl')
        audit_dir = os.path.dirname(self._audit_file_path)
        if audit_dir:
            Path(audit_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized file audit storage: {self._audit_file_path}")
    
    def _init_database_storage(self) -> None:
        """Initialize database storage (placeholder for future implementation)"""
        self.logger.info("Database audit storage not yet implemented, falling back to file storage")
        self._storage_backend = 'file'
        self._init_file_storage()
    
    async def store_event(self, event: AuditEvent) -> None:
        """Store an audit event
        
        Args:
            event: Audit event to store
        """
        try:
            # Add to buffer
            self._events_buffer.append(event)
            
            # Check if we need to flush
            current_time = time.time()
            if (len(self._events_buffer) >= self._buffer_size or 
                current_time - self._last_flush >= self._flush_interval):
                await self._flush_buffer()
                
        except Exception as e:
            self.logger.error(f"Failed to store audit event: {e}", context={
                'event_id': event.event_id,
                'event_type': event.event_type.value
            })
    
    async def _flush_buffer(self) -> None:
        """Flush buffered events to storage"""
        if not self._events_buffer:
            return
        
        try:
            if self._storage_backend == 'file':
                await self._flush_to_file()
            elif self._storage_backend == 'database':
                await self._flush_to_database()
            
            self.logger.debug(f"Flushed {len(self._events_buffer)} audit events to {self._storage_backend}")
            self._events_buffer.clear()
            self._last_flush = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to flush audit buffer: {e}")
    
    async def _flush_to_file(self) -> None:
        """Flush events to file storage"""
        try:
            with open(self._audit_file_path, 'a', encoding='utf-8') as f:
                for event in self._events_buffer:
                    event_dict = asdict(event)
                    # Convert datetime and enum to serializable formats
                    event_dict['timestamp'] = event.timestamp.isoformat()
                    event_dict['event_type'] = event.event_type.value
                    event_dict['severity'] = event.severity.value
                    
                    f.write(json.dumps(event_dict, ensure_ascii=False) + '\n')
                    
        except Exception as e:
            self.logger.error(f"Failed to write audit events to file: {e}")
            raise
    
    async def _flush_to_database(self) -> None:
        """Flush events to database (placeholder)"""
        # Future implementation for database storage
        pass
    
    async def query_events(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          event_types: Optional[List[AuditEventType]] = None,
                          severity: Optional[AuditSeverity] = None,
                          actor: Optional[str] = None,
                          limit: int = 1000) -> List[AuditEvent]:
        """Query audit events with filters
        
        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            severity: Filter by severity level
            actor: Filter by actor
            limit: Maximum number of events to return
            
        Returns:
            List of matching audit events
        """
        # For file storage, this would read and parse the file
        # For database storage, this would query the database
        # For now, return empty list as placeholder
        self.logger.warning("Audit event querying not yet implemented")
        return []


class AuditTrailManager:
    """Main audit trail management system"""
    
    _instance = None
    
    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, '_initialized'):
            return
        
        self.config = config or {
            'enabled': True,
            'buffer_size': 100,
            'flush_interval_seconds': 30,
            'storage_backend': 'file',
            'audit_file_path': 'logs/audit_trail.jsonl',
            'include_context': True,
            'performance_threshold_ms': 1000.0
        }
        
        self.logger = get_logger(__name__)
        self.storage = AuditTrailStorage(self.config)
        self._session_contexts: Dict[str, Dict[str, Any]] = {}
        self._performance_monitors: Dict[str, float] = {}
        self._initialized = True
        
        self.logger.info("Audit trail manager initialized", context={
            'storage_backend': self.config['storage_backend'],
            'enabled': self.config['enabled']
        })
    
    def is_enabled(self) -> bool:
        """Check if audit trail is enabled"""
        return self.config.get('enabled', True)
    
    async def log_event(self,
                       event_type: AuditEventType,
                       actor: str,
                       resource: str,
                       action: str,
                       outcome: str,
                       severity: AuditSeverity = AuditSeverity.LOW,
                       details: Optional[Dict[str, Any]] = None,
                       context: Optional[Dict[str, Any]] = None,
                       session_id: Optional[str] = None,
                       duration_ms: Optional[float] = None) -> str:
        """Log an audit event
        
        Args:
            event_type: Type of the audit event
            actor: Who performed the action
            resource: What was affected
            action: What was done
            outcome: Result of the action
            severity: Severity level
            details: Additional event details
            context: Event context information
            session_id: Associated session ID
            duration_ms: Operation duration in milliseconds
            
        Returns:
            Event ID
        """
        if not self.is_enabled():
            return ""
        
        try:
            event_id = str(uuid.uuid4())
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                severity=severity,
                actor=actor,
                resource=resource,
                action=action,
                outcome=outcome,
                details=details or {},
                context=context or {},
                session_id=session_id,
                duration_ms=duration_ms
            )
            
            await self.storage.store_event(event)
            
            # Also log to standard logger for immediate visibility
            log_level = self._severity_to_log_level(severity)
            self.logger.log(log_level, f"AUDIT: {action} on {resource}", context={
                'audit_event_id': event_id,
                'event_type': event_type.value,
                'actor': actor,
                'outcome': outcome,
                'duration_ms': duration_ms
            })
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            return ""
    
    def _severity_to_log_level(self, severity: AuditSeverity) -> int:
        """Convert audit severity to log level"""
        import logging
        mapping = {
            AuditSeverity.LOW: logging.DEBUG,
            AuditSeverity.MEDIUM: logging.INFO,
            AuditSeverity.HIGH: logging.WARNING,
            AuditSeverity.CRITICAL: logging.ERROR
        }
        return mapping.get(severity, logging.INFO)
    
    @asynccontextmanager
    async def audit_operation(self,
                             event_type: AuditEventType,
                             actor: str,
                             resource: str,
                             action: str,
                             session_id: Optional[str] = None,
                             severity: AuditSeverity = AuditSeverity.LOW):
        """Context manager for auditing operations
        
        Usage:
            async with audit_manager.audit_operation(
                AuditEventType.DATA_READ,
                "user123",
                "memory_collection",
                "retrieve_memories"
            ):
                # Perform operation
                result = await some_operation()
        """
        start_time = time.time()
        event_id = None
        
        try:
            yield
            # Success case
            duration_ms = (time.time() - start_time) * 1000
            event_id = await self.log_event(
                event_type=event_type,
                actor=actor,
                resource=resource,
                action=action,
                outcome="success",
                severity=severity,
                session_id=session_id,
                duration_ms=duration_ms
            )
            
            # Check performance threshold
            threshold = self.config.get('performance_threshold_ms', 1000.0)
            if duration_ms > threshold:
                await self.log_event(
                    event_type=AuditEventType.PERFORMANCE_THRESHOLD_EXCEEDED,
                    actor=actor,
                    resource=resource,
                    action=action,
                    outcome="threshold_exceeded",
                    severity=AuditSeverity.MEDIUM,
                    details={'duration_ms': duration_ms, 'threshold_ms': threshold},
                    session_id=session_id
                )
            
        except Exception as e:
            # Failure case
            duration_ms = (time.time() - start_time) * 1000
            event_id = await self.log_event(
                event_type=event_type,
                actor=actor,
                resource=resource,
                action=action,
                outcome="failure",
                severity=AuditSeverity.HIGH,
                details={'error': str(e), 'error_type': type(e).__name__},
                session_id=session_id,
                duration_ms=duration_ms
            )
            raise
    
    async def log_security_event(self,
                                event_type: AuditEventType,
                                actor: str,
                                resource: str,
                                action: str,
                                outcome: str,
                                details: Optional[Dict[str, Any]] = None,
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None) -> str:
        """Log a security-related audit event
        
        Args:
            event_type: Security event type
            actor: Actor performing the action
            resource: Resource being accessed
            action: Action being performed
            outcome: Result of the action
            details: Additional security details
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Event ID
        """
        return await self.log_event(
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            outcome=outcome,
            severity=AuditSeverity.HIGH,
            details=details,
            context={'ip_address': ip_address, 'user_agent': user_agent}
        )
    
    async def log_data_access(self,
                             operation: str,
                             actor: str,
                             resource_type: str,
                             resource_id: str,
                             outcome: str,
                             details: Optional[Dict[str, Any]] = None,
                             session_id: Optional[str] = None) -> str:
        """Log data access event
        
        Args:
            operation: Type of operation (read, write, delete, etc.)
            actor: Who performed the operation
            resource_type: Type of resource accessed
            resource_id: ID of the specific resource
            outcome: Result of the operation
            details: Additional operation details
            session_id: Associated session ID
            
        Returns:
            Event ID
        """
        event_type_mapping = {
            'read': AuditEventType.DATA_READ,
            'write': AuditEventType.DATA_WRITE,
            'delete': AuditEventType.DATA_DELETE,
            'export': AuditEventType.DATA_EXPORT,
            'import': AuditEventType.DATA_IMPORT
        }
        
        event_type = event_type_mapping.get(operation.lower(), AuditEventType.DATA_READ)
        
        return await self.log_event(
            event_type=event_type,
            actor=actor,
            resource=f"{resource_type}:{resource_id}",
            action=operation,
            outcome=outcome,
            severity=AuditSeverity.MEDIUM,
            details=details,
            session_id=session_id
        )
    
    def create_audit_decorator(self,
                              event_type: AuditEventType,
                              resource_name: Optional[str] = None,
                              actor_param: str = 'actor',
                              severity: AuditSeverity = AuditSeverity.LOW):
        """Create decorator for automatic audit logging
        
        Args:
            event_type: Type of audit event
            resource_name: Name of the resource (if not dynamic)
            actor_param: Parameter name containing actor info
            severity: Event severity level
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            async def async_wrapper(*args, **kwargs):
                # Extract actor from parameters
                actor = kwargs.get(actor_param, 'system')
                resource = resource_name or func.__name__
                action = func.__name__
                
                async with self.audit_operation(
                    event_type=event_type,
                    actor=actor,
                    resource=resource,
                    action=action,
                    severity=severity
                ):
                    return await func(*args, **kwargs)
            
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we'll log without the context manager
                import asyncio
                
                actor = kwargs.get(actor_param, 'system')
                resource = resource_name or func.__name__
                action = func.__name__
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log success asynchronously
                    asyncio.create_task(self.log_event(
                        event_type=event_type,
                        actor=actor,
                        resource=resource,
                        action=action,
                        outcome="success",
                        severity=severity,
                        duration_ms=duration_ms
                    ))
                    
                    return result
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log failure asynchronously
                    asyncio.create_task(self.log_event(
                        event_type=event_type,
                        actor=actor,
                        resource=resource,
                        action=action,
                        outcome="failure",
                        severity=AuditSeverity.HIGH,
                        details={'error': str(e)},
                        duration_ms=duration_ms
                    ))
                    
                    raise
            
            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


# Global audit trail manager instance
_audit_manager_instance = None

def get_audit_manager(config: Optional[Dict[str, Any]] = None) -> AuditTrailManager:
    """Get global audit trail manager instance"""
    global _audit_manager_instance
    if _audit_manager_instance is None:
        _audit_manager_instance = AuditTrailManager(config)
    return _audit_manager_instance


# Convenience functions
async def audit_event(event_type: AuditEventType,
                     actor: str,
                     resource: str,
                     action: str,
                     outcome: str,
                     **kwargs) -> str:
    """Log audit event using global manager"""
    manager = get_audit_manager()
    return await manager.log_event(event_type, actor, resource, action, outcome, **kwargs)


def audit_operation(event_type: AuditEventType,
                   actor: str,
                   resource: str,
                   action: str,
                   **kwargs):
    """Audit operation context manager using global manager"""
    manager = get_audit_manager()
    return manager.audit_operation(event_type, actor, resource, action, **kwargs)


def audit_decorator(event_type: AuditEventType, **kwargs):
    """Audit decorator using global manager"""
    manager = get_audit_manager()
    return manager.create_audit_decorator(event_type, **kwargs)