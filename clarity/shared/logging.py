"""
Comprehensive logging integration that combines enhanced logging, audit trails, and observability.

This module provides a unified interface for:
- Enhanced structured logging
- Audit trail integration
- Performance monitoring
- Distributed tracing
- Health monitoring
"""

import asyncio
import functools
from typing import Any, Dict, Optional, Callable, Union
from contextlib import asynccontextmanager

from .utils.logging_utils import get_logger as get_base_logger, ContextLogger
from .audit_trail import (
    AuditEventType, AuditSeverity, get_audit_manager,
    audit_operation, audit_decorator
)
from .observability import (
    get_observability_manager, trace_operation, 
    time_operation, observe_operation
)


class EnhancedLogger:
    """Enhanced logger with integrated audit, tracing, and metrics"""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.name = name
        self.base_logger = get_base_logger(name, context)
        self.audit_manager = get_audit_manager()
        self.observability = get_observability_manager()
        self.default_context = context or {}
    
    def debug(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Enhanced debug logging"""
        self.base_logger.debug(message, *args, context=context, **kwargs)
    
    def info(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Enhanced info logging"""
        self.base_logger.info(message, *args, context=context, **kwargs)
    
    def warning(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Enhanced warning logging"""
        self.base_logger.warning(message, *args, context=context, **kwargs)
    
    def error(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Enhanced error logging"""
        self.base_logger.error(message, *args, context=context, **kwargs)
    
    def critical(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Enhanced critical logging"""
        self.base_logger.critical(message, *args, context=context, **kwargs)
    
    def exception(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Enhanced exception logging"""
        self.base_logger.exception(message, *args, context=context, **kwargs)
    
    async def audit_info(self, message: str,
                        event_type: AuditEventType,
                        actor: str,
                        resource: str,
                        action: str,
                        outcome: str = "success",
                        severity: AuditSeverity = AuditSeverity.LOW,
                        context: Optional[Dict[str, Any]] = None,
                        **kwargs):
        """Log info message with audit trail"""
        self.info(message, context=context, **kwargs)
        await self.audit_manager.log_event(
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            outcome=outcome,
            severity=severity,
            details={'log_message': message},
            context=context
        )
    
    async def audit_error(self, message: str,
                         event_type: AuditEventType,
                         actor: str,
                         resource: str,
                         action: str,
                         error: Exception,
                         context: Optional[Dict[str, Any]] = None,
                         **kwargs):
        """Log error message with audit trail"""
        self.error(message, context=context, **kwargs)
        await self.audit_manager.log_event(
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            outcome="error",
            severity=AuditSeverity.HIGH,
            details={
                'log_message': message,
                'error': str(error),
                'error_type': type(error).__name__
            },
            context=context
        )
    
    def with_context(self, **context) -> 'EnhancedLogger':
        """Create logger with additional context"""
        new_context = self.default_context.copy()
        new_context.update(context)
        return EnhancedLogger(self.name, new_context)
    
    @asynccontextmanager
    async def operation_context(self, 
                               operation_name: str,
                               actor: str = "system",
                               resource: Optional[str] = None,
                               audit_event_type: Optional[AuditEventType] = None,
                               tags: Optional[Dict[str, Any]] = None):
        """Context manager for comprehensive operation logging"""
        resource = resource or operation_name
        
        # Log operation start
        self.info(f"Starting operation: {operation_name}", context={
            'operation': operation_name,
            'actor': actor,
            'resource': resource
        })
        
        # Use observability manager for comprehensive monitoring
        async with self.observability.observe_operation(
            operation_name=operation_name,
            actor=actor,
            tags=tags,
            audit_event_type=audit_event_type
        ) as span_id:
            try:
                yield span_id
                
                # Log successful completion
                self.info(f"Completed operation: {operation_name}", context={
                    'operation': operation_name,
                    'actor': actor,
                    'resource': resource,
                    'outcome': 'success',
                    'span_id': span_id
                })
                
            except Exception as e:
                # Log operation failure
                self.error(f"Operation failed: {operation_name} - {str(e)}", context={
                    'operation': operation_name,
                    'actor': actor,
                    'resource': resource,
                    'outcome': 'error',
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'span_id': span_id
                })
                raise
    
    def create_operation_decorator(self,
                                  operation_name: Optional[str] = None,
                                  actor: str = "system",
                                  audit_event_type: Optional[AuditEventType] = None,
                                  include_args: bool = False,
                                  include_result: bool = False):
        """Create decorator for automatic operation logging"""
        
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or f"{self.name}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    # Extract context from kwargs if present
                    tags = {}
                    if include_args:
                        tags['args_count'] = len(args)
                        tags['kwargs_keys'] = list(kwargs.keys())
                    
                    async with self.operation_context(
                        operation_name=op_name,
                        actor=actor,
                        audit_event_type=audit_event_type,
                        tags=tags
                    ):
                        result = await func(*args, **kwargs)
                        
                        if include_result and result is not None:
                            self.debug(f"Operation {op_name} returned result", context={
                                'operation': op_name,
                                'result_type': type(result).__name__,
                                'result_length': len(result) if hasattr(result, '__len__') else None
                            })
                        
                        return result
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    # For sync functions, we'll use a different approach
                    import time
                    start_time = time.time()
                    
                    self.info(f"Starting operation: {op_name}", context={
                        'operation': op_name,
                        'actor': actor
                    })
                    
                    try:
                        result = func(*args, **kwargs)
                        duration_ms = (time.time() - start_time) * 1000
                        
                        self.info(f"Completed operation: {op_name}", context={
                            'operation': op_name,
                            'actor': actor,
                            'duration_ms': duration_ms,
                            'outcome': 'success'
                        })
                        
                        # Record metrics
                        self.observability.metrics.record_timer(f"operation.{op_name}", duration_ms)
                        self.observability.metrics.increment_counter(f"operations.{op_name}.success")
                        
                        # Audit if requested
                        if audit_event_type:
                            asyncio.create_task(self.audit_manager.log_event(
                                event_type=audit_event_type,
                                actor=actor,
                                resource=op_name,
                                action=func.__name__,
                                outcome="success",
                                duration_ms=duration_ms
                            ))
                        
                        return result
                        
                    except Exception as e:
                        duration_ms = (time.time() - start_time) * 1000
                        
                        self.error(f"Operation failed: {op_name} - {str(e)}", context={
                            'operation': op_name,
                            'actor': actor,
                            'duration_ms': duration_ms,
                            'outcome': 'error',
                            'error': str(e),
                            'error_type': type(e).__name__
                        })
                        
                        # Record metrics
                        self.observability.metrics.record_timer(f"operation.{op_name}", duration_ms)
                        self.observability.metrics.increment_counter(f"operations.{op_name}.error")
                        
                        # Audit if requested
                        if audit_event_type:
                            asyncio.create_task(self.audit_manager.log_event(
                                event_type=audit_event_type,
                                actor=actor,
                                resource=op_name,
                                action=func.__name__,
                                outcome="error",
                                details={'error': str(e)},
                                duration_ms=duration_ms,
                                severity=AuditSeverity.HIGH
                            ))
                        
                        raise
                
                return sync_wrapper
        
        return decorator


# Enhanced logger factory
def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> EnhancedLogger:
    """Get enhanced logger instance
    
    Args:
        name: Logger name (usually __name__)
        context: Optional context dictionary
        
    Returns:
        EnhancedLogger instance with audit and observability integration
    """
    return EnhancedLogger(name, context)


# Convenience decorators
def log_operation(operation_name: Optional[str] = None,
                 actor: str = "system",
                 audit_event_type: Optional[AuditEventType] = None,
                 logger_name: Optional[str] = None):
    """Decorator for automatic operation logging
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        actor: Actor performing the operation
        audit_event_type: Type of audit event to log
        logger_name: Logger name (defaults to function's module)
        
    Example:
        @log_operation("user_login", "user123", AuditEventType.AUTHENTICATION)
        async def login_user(username: str, password: str):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Get logger name from function's module if not provided
        func_logger_name = logger_name or func.__module__
        logger = get_logger(func_logger_name)
        
        return logger.create_operation_decorator(
            operation_name=operation_name,
            actor=actor,
            audit_event_type=audit_event_type
        )(func)
    
    return decorator


def audit_operation(event_type: AuditEventType,
                   actor: str = "system",
                   resource: Optional[str] = None,
                   severity: AuditSeverity = AuditSeverity.LOW):
    """Decorator for audit-only operation logging"""
    return audit_decorator(event_type, resource_name=resource, actor_param='actor', severity=severity)


# Context managers for common patterns
@asynccontextmanager
async def logged_operation(operation_name: str,
                          logger_name: str = __name__,
                          actor: str = "system",
                          audit_event_type: Optional[AuditEventType] = None,
                          **kwargs):
    """Context manager for logged operations"""
    logger = get_logger(logger_name)
    async with logger.operation_context(
        operation_name=operation_name,
        actor=actor,
        audit_event_type=audit_event_type,
        **kwargs
    ) as span_id:
        yield logger, span_id


# Initialize logging configuration on module import
def configure_comprehensive_logging(config: Optional[Dict[str, Any]] = None):
    """Configure comprehensive logging with all features
    
    Args:
        config: Configuration dictionary with sections for:
            - logging: Base logging configuration
            - audit: Audit trail configuration  
            - observability: Observability configuration
    """
    from .utils.logging_utils import configure_logging
    
    if not config:
        config = {}
    
    # Configure base logging
    logging_config = config.get('logging', {})
    configure_logging(
        level=logging_config.get('level', 'INFO'),
        format_type=logging_config.get('format', 'enhanced'),
        use_color=logging_config.get('use_color', True),
        include_context=logging_config.get('include_context', True),
        log_to_file=logging_config.get('log_to_file', False),
        log_file_path=logging_config.get('log_file_path'),
        max_file_size=logging_config.get('max_file_size', 10 * 1024 * 1024),
        backup_count=logging_config.get('backup_count', 5)
    )
    
    # Initialize audit manager
    audit_config = config.get('audit', {})
    get_audit_manager(audit_config)
    
    # Initialize observability manager
    observability_config = config.get('observability', {})
    get_observability_manager(observability_config)
    
    # Log initialization
    logger = get_logger(__name__)
    logger.info("Comprehensive logging system initialized", context={
        'logging_enabled': True,
        'audit_enabled': audit_config.get('enabled', True),
        'observability_enabled': observability_config.get('enabled', True),
        'tracing_enabled': observability_config.get('tracing', {}).get('enabled', True),
        'metrics_enabled': observability_config.get('metrics', {}).get('enabled', True),
        'health_monitoring_enabled': observability_config.get('health', {}).get('enabled', True)
    })


# Auto-configure with defaults if not already configured
try:
    configure_comprehensive_logging()
except Exception as e:
    # Fall back to basic logging if comprehensive setup fails
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"Failed to initialize comprehensive logging, using basic setup: {e}")