import logging
import sys
import os
from typing import Dict, Optional, Any, Union
from datetime import datetime
from pathlib import Path


class EnhancedFormatter(logging.Formatter):
    """Enhanced logging formatter with color support and context"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_color: bool = True, include_context: bool = True):
        self.use_color = use_color and sys.stderr.isatty()
        self.include_context = include_context
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Base format
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        name = record.name
        message = record.getMessage()
        
        # Add color if enabled
        if self.use_color:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            level = f"{color}{level}{reset}"
        
        # Build base log line
        log_line = f"{timestamp} | {level:8} | {name} | {message}"
        
        # Add context if available and enabled
        if self.include_context and hasattr(record, 'context'):
            context = record.context
            if isinstance(context, dict):
                context_str = ' | '.join(f"{k}={v}" for k, v in context.items())
                log_line += f" | {context_str}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += '\n' + self.formatException(record.exc_info)
        
        return log_line


class LoggingManager:
    """Enhanced logging manager with configuration support"""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _handlers: Dict[str, logging.Handler] = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._default_config = {
                'level': 'INFO',
                'format': 'enhanced',
                'use_color': True,
                'include_context': True,
                'log_to_file': False,
                'log_file_path': None,
                'max_file_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            }
            self._initialized = True
    
    def configure_logging(self, 
                         level: str = 'INFO',
                         format_type: str = 'enhanced',
                         use_color: bool = True,
                         include_context: bool = True,
                         log_to_file: bool = False,
                         log_file_path: Optional[str] = None,
                         max_file_size: int = 10 * 1024 * 1024,
                         backup_count: int = 5) -> None:
        """Configure global logging settings
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: Format type ('enhanced', 'simple', 'json')
            use_color: Enable colored output for console
            include_context: Include context information in logs
            log_to_file: Enable file logging
            log_file_path: Path for log file
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        
        # Choose formatter
        if format_type == 'enhanced':
            formatter = EnhancedFormatter(use_color=use_color, include_context=include_context)
        elif format_type == 'simple':
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
        elif format_type == 'json':
            formatter = JSONFormatter()
        else:
            formatter = EnhancedFormatter(use_color=use_color, include_context=include_context)
        
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        self._handlers['console'] = console_handler
        
        # Configure file handler if requested
        if log_to_file and log_file_path:
            self._configure_file_logging(log_file_path, log_level, formatter, max_file_size, backup_count)
        
        # Store configuration
        self._current_config = {
            'level': level,
            'format_type': format_type,
            'use_color': use_color,
            'include_context': include_context,
            'log_to_file': log_to_file,
            'log_file_path': log_file_path,
            'max_file_size': max_file_size,
            'backup_count': backup_count
        }
    
    def _configure_file_logging(self, 
                               log_file_path: str,
                               log_level: int,
                               formatter: logging.Formatter,
                               max_file_size: int,
                               backup_count: int) -> None:
        """Configure file logging with rotation"""
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        self._handlers['file'] = file_handler
    
    def get_logger(self, name: str, context: Optional[Dict[str, Any]] = None) -> 'ContextLogger':
        """Get or create a logger with optional context
        
        Args:
            name: Logger name (usually __name__)
            context: Optional context dictionary
            
        Returns:
            ContextLogger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        return ContextLogger(self._loggers[name], context)
    
    def set_level(self, level: str, logger_name: Optional[str] = None) -> None:
        """Set logging level for specific logger or all loggers
        
        Args:
            level: Logging level string
            logger_name: Specific logger name (None for all)
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        if logger_name:
            if logger_name in self._loggers:
                self._loggers[logger_name].setLevel(log_level)
        else:
            logging.getLogger().setLevel(log_level)
            for handler in self._handlers.values():
                handler.setLevel(log_level)


class ContextLogger:
    """Logger wrapper that supports context information"""
    
    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        self._logger = logger
        self._context = context or {}
    
    def _log_with_context(self, level: int, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log message with context information"""
        if self._logger.isEnabledFor(level):
            # Merge context
            merged_context = self._context.copy()
            if context:
                merged_context.update(context)
            
            # Create log record
            record = self._logger.makeRecord(
                self._logger.name, level, '', 0, message, args, None, None, None
            )
            
            # Add context to record
            if merged_context:
                record.context = merged_context
            
            # Handle the record
            self._logger.handle(record)
    
    def debug(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message with context"""
        self._log_with_context(logging.DEBUG, message, *args, context=context, **kwargs)
    
    def info(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with context"""
        self._log_with_context(logging.INFO, message, *args, context=context, **kwargs)
    
    def warning(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message with context"""
        self._log_with_context(logging.WARNING, message, *args, context=context, **kwargs)
    
    def error(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message with context"""
        self._log_with_context(logging.ERROR, message, *args, context=context, **kwargs)
    
    def critical(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message with context"""
        self._log_with_context(logging.CRITICAL, message, *args, context=context, **kwargs)
    
    def exception(self, message: str, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log exception with context"""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, *args, context=context, **kwargs)
    
    def with_context(self, **context) -> 'ContextLogger':
        """Create new logger with additional context"""
        new_context = self._context.copy()
        new_context.update(context)
        return ContextLogger(self._logger, new_context)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


# Global logging manager instance
logging_manager = LoggingManager()

# Convenience functions
def configure_logging(**kwargs) -> None:
    """Configure global logging settings"""
    logging_manager.configure_logging(**kwargs)

def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> ContextLogger:
    """Get logger with optional context"""
    return logging_manager.get_logger(name, context)

def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """Set logging level"""
    logging_manager.set_level(level, logger_name)