from functools import wraps
from typing import Callable, Type, Tuple, Union, List
from clarity.shared.exceptions import ClarityException
from clarity.shared.simple_logging import get_logger

def handle_errors(
    *exception_mappings: Union[Type[Exception], Tuple[Type[Exception], Type[ClarityException]]]
):
    """
    Decorator for consistent error handling with exception mapping
    
    Usage:
        @handle_errors(ValueError, (ConnectionError, QdrantConnectionError))
        async def some_function():
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(func.__module__)
                
                # Map exception to appropriate Clarity exception
                clarity_exception = _map_exception(e, exception_mappings)
                
                # Log the error with context
                context = clarity_exception.to_dict()
                context["error_message"] = context.pop("message")  # Rename to avoid LogRecord conflict
                logger.error(
                    f"Error in {func.__name__}: {clarity_exception.message}",
                    extra=context
                )
                
                raise clarity_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(func.__module__)
                clarity_exception = _map_exception(e, exception_mappings)
                
                context = clarity_exception.to_dict()
                context["error_message"] = context.pop("message")  # Rename to avoid LogRecord conflict
                logger.error(
                    f"Error in {func.__name__}: {clarity_exception.message}",
                    extra=context
                )
                
                raise clarity_exception
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def _map_exception(
    original: Exception, 
    mappings: Tuple
) -> ClarityException:
    """Map original exception to appropriate Clarity exception"""
    for mapping in mappings:
        if isinstance(mapping, tuple):
            original_type, clarity_type = mapping
            if isinstance(original, original_type):
                return clarity_type(
                    message=str(original),
                    context={"original_type": original_type.__name__},
                    cause=original
                )
        elif isinstance(original, mapping):
            return ClarityException(
                message=str(original),
                context={"original_type": mapping.__name__},
                cause=original
            )
    
    # Default mapping
    return ClarityException(
        message=str(original),
        context={"original_type": type(original).__name__},
        cause=original
    )