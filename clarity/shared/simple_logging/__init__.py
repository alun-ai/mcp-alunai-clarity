import logging
import sys
from typing import Optional, Dict

class SimpleLogger:
    """Simple logging utility for initial implementation"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def initialize(cls, log_level: str = "INFO"):
        """Initialize simple logging configuration"""
        if cls._initialized:
            return
            
        # Configure basic logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            stream=sys.stderr
        )
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create logger instance"""
        if not cls._initialized:
            cls.initialize()
            
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            
        return cls._loggers[name]

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get logger"""
    return SimpleLogger.get_logger(name)

def setup_logging(log_level: str = "INFO"):
    """Setup logging for the application"""
    SimpleLogger.initialize(log_level)

def log_operation(operation_name: str, actor: str = "system", audit_event_type=None, **kwargs):
    """Decorator to log operations with context"""
    def decorator(func):
        def wrapper(*args, **kwargs_inner):
            logger = get_logger(func.__module__)
            logger.info(f"Operation: {operation_name} - actor: {actor} - event_type: {audit_event_type}")
            return func(*args, **kwargs_inner)
        return wrapper
    return decorator

def logged_operation(operation_name: str):
    """Decorator to log operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info(f"Operation: {operation_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator