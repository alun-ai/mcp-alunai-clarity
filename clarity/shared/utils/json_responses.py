import json
from typing import Any, Dict, Optional, List


class MCPResponseBuilder:
    """Utility class for building standardized MCP tool responses"""
    
    @staticmethod
    def success(data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Create a successful JSON response
        
        Args:
            data: Optional dictionary of response data
            **kwargs: Additional fields to include in response
            
        Returns:
            JSON string representing successful response
        """
        response = {"success": True}
        
        if data:
            response.update(data)
        if kwargs:
            response.update(kwargs)
            
        return json.dumps(response)
    
    @staticmethod
    def error(error: str, error_code: Optional[str] = None, **kwargs) -> str:
        """Create an error JSON response
        
        Args:
            error: Error message
            error_code: Optional error code for categorization
            **kwargs: Additional fields to include in response
            
        Returns:
            JSON string representing error response
        """
        response = {
            "success": False,
            "error": error
        }
        
        if error_code:
            response["error_code"] = error_code
        if kwargs:
            response.update(kwargs)
            
        return json.dumps(response)
    
    @staticmethod
    def data(data: Dict[str, Any], success: bool = True) -> str:
        """Create a data response with success flag
        
        Args:
            data: Response data dictionary
            success: Whether operation was successful
            
        Returns:
            JSON string representing data response
        """
        response = {"success": success}
        response.update(data)
        return json.dumps(response)
    
    @staticmethod
    def memory_stored(memory_id: str) -> str:
        """Standard response for successful memory storage"""
        return MCPResponseBuilder.success(memory_id=memory_id)
    
    @staticmethod
    def memories_retrieved(memories: List[Dict[str, Any]], count: Optional[int] = None) -> str:
        """Standard response for memory retrieval"""
        response_data = {"memories": memories}
        if count is not None:
            response_data["count"] = count
        return MCPResponseBuilder.success(response_data)
    
    @staticmethod
    def operation_completed(operation: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Standard response for completed operations"""
        response_data = {"operation": operation, "completed": True}
        if details:
            response_data.update(details)
        return MCPResponseBuilder.success(response_data)


class SafeJSONHandler:
    """Utility class for safe JSON operations with error handling"""
    
    @staticmethod
    def load_file(file_path: str, default: Any = None) -> Any:
        """Safely load JSON from file with error handling
        
        Args:
            file_path: Path to JSON file
            default: Default value to return on error
            
        Returns:
            Loaded JSON data or default value
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            return default
    
    @staticmethod
    def save_file(data: Any, file_path: str, indent: int = 2) -> bool:
        """Safely save data to JSON file
        
        Args:
            data: Data to save
            file_path: Target file path
            indent: JSON indentation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except (IOError, TypeError):
            return False
    
    @staticmethod
    def parse_string(json_str: str, default: Any = None) -> Any:
        """Safely parse JSON string
        
        Args:
            json_str: JSON string to parse
            default: Default value to return on error
            
        Returns:
            Parsed JSON data or default value
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return default
    
    @staticmethod
    def to_string(data: Any, default: str = "{}") -> str:
        """Safely convert data to JSON string
        
        Args:
            data: Data to convert
            default: Default string to return on error
            
        Returns:
            JSON string or default value
        """
        try:
            return json.dumps(data, ensure_ascii=False)
        except TypeError:
            return default