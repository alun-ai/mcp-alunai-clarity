from typing import Dict, Any, Optional
import traceback
from datetime import datetime

class ClarityException(Exception):
    """Base exception for all Clarity errors with enhanced context"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = None,
        context: Dict[str, Any] = None,
        cause: Exception = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow().isoformat()
        self.stack_trace = traceback.format_exc() if cause else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace
        }

class MemoryOperationError(ClarityException):
    """Memory storage/retrieval operation errors"""
    pass

class QdrantConnectionError(ClarityException):
    """Qdrant database connection and operation errors"""
    pass

class MCPProtocolError(ClarityException):
    """MCP protocol violations and communication errors"""
    pass

class ValidationError(ClarityException):
    """Input validation and data format errors"""
    pass

class ConfigurationError(ClarityException):
    """Configuration loading and validation errors"""
    pass

class AutoCodeError(ClarityException):
    """AutoCode domain operation errors"""
    pass