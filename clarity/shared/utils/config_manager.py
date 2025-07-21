import os
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from clarity.shared.exceptions import ConfigurationError
from clarity.shared.simple_logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Enhanced configuration management with validation and type safety"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config_cache: Dict[str, Any] = {}
        self._config_schema: Dict[str, Any] = {}
    
    def load_config(self, 
                   config_path: Optional[str] = None, 
                   use_cache: bool = True,
                   create_if_missing: bool = True) -> Dict[str, Any]:
        """Load configuration with enhanced error handling and validation
        
        Args:
            config_path: Path to config file (uses instance path if None)
            use_cache: Whether to use cached configuration
            create_if_missing: Create default config if file doesn't exist
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If config loading fails
        """
        path = config_path or self.config_path
        if not path:
            raise ConfigurationError("No configuration path provided")
        
        # Check cache first
        if use_cache and path in self._config_cache:
            return self._config_cache[path]
        
        try:
            if not os.path.exists(path):
                if create_if_missing:
                    logger.info(f"Creating default configuration at {path}")
                    config = self._create_default_config(path)
                else:
                    raise ConfigurationError(f"Configuration file not found: {path}")
            else:
                config = self._load_config_file(path)
            
            # Validate against schema if available
            if self._config_schema:
                config = self._validate_config(config)
            
            # Cache the configuration
            self._config_cache[path] = config
            
            return config
            
        except (OSError, ValueError, KeyError, PermissionError) as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load configuration from {path}: {str(e)}")
    
    def save_config(self, 
                   config: Dict[str, Any], 
                   config_path: Optional[str] = None,
                   backup: bool = True) -> bool:
        """Save configuration with backup support
        
        Args:
            config: Configuration dictionary to save
            config_path: Target path (uses instance path if None)
            backup: Create backup of existing config
            
        Returns:
            True if successful, False otherwise
        """
        path = config_path or self.config_path
        if not path:
            logger.error("No configuration path provided for saving")
            return False
        
        try:
            # Create backup if requested and file exists
            if backup and os.path.exists(path):
                backup_path = f"{path}.backup"
                import shutil
                shutil.copy2(path, backup_path)
                logger.debug(f"Created config backup: {backup_path}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save configuration
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self._config_cache[path] = config.copy()
            
            logger.info(f"Configuration saved to {path}")
            return True
            
        except (OSError, ValueError, PermissionError, TypeError) as e:
            logger.error(f"Failed to save configuration to {path}: {str(e)}")
            return False
    
    def get_config_value(self, 
                        key: str, 
                        default: Any = None,
                        config_path: Optional[str] = None) -> Any:
        """Get a specific configuration value using dot notation
        
        Args:
            key: Configuration key (supports dot notation like 'database.host')
            default: Default value if key not found
            config_path: Config file path
            
        Returns:
            Configuration value or default
        """
        try:
            config = self.load_config(config_path)
            return self._get_nested_value(config, key, default)
        except (OSError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to get config value '{key}': {str(e)}")
            return default
    
    def set_config_value(self, 
                        key: str, 
                        value: Any,
                        config_path: Optional[str] = None,
                        save: bool = True) -> bool:
        """Set a specific configuration value using dot notation
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            config_path: Config file path
            save: Whether to save config after setting
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.load_config(config_path)
            self._set_nested_value(config, key, value)
            
            if save:
                return self.save_config(config, config_path)
            else:
                # Update cache
                path = config_path or self.config_path
                if path:
                    self._config_cache[path] = config
                return True
                
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to set config value '{key}': {str(e)}")
            return False
    
    def validate_config_schema(self, schema: Dict[str, Any]) -> None:
        """Set configuration schema for validation
        
        Args:
            schema: JSON schema dictionary for validation
        """
        self._config_schema = schema
    
    def clear_cache(self) -> None:
        """Clear the configuration cache"""
        self._config_cache.clear()
    
    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from file with error handling"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.debug(f"Loaded configuration from {path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file {path}: {str(e)}")
        except IOError as e:
            raise ConfigurationError(f"Cannot read config file {path}: {str(e)}")
    
    def _create_default_config(self, path: str) -> Dict[str, Any]:
        """Create a default configuration file"""
        default_config = {
            "version": "1.0.0",
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "alunai_clarity_memories",
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_size": 384
            },
            "memory": {
                "default_tier": "working",
                "max_context_length": 8000,
                "cleanup_interval_hours": 24
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save default config
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default configuration at {path}")
        return default_config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema"""
        # Basic validation - could be enhanced with jsonschema library
        required_sections = ['qdrant', 'memory', 'logging']
        
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing required config section: {section}")
                config[section] = {}
        
        return config
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested configuration value using dot notation"""
        keys = key.split('.')
        target = config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the final value
        target[keys[-1]] = value


# Global configuration manager instance
config_manager = ConfigManager()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load configuration"""
    return config_manager.load_config(config_path)

def get_config_value(key: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return config_manager.get_config_value(key, default)

def set_config_value(key: str, value: Any, save: bool = True) -> bool:
    """Convenience function to set configuration value"""
    return config_manager.set_config_value(key, value, save=save)