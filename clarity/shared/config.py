"""
Unified configuration management system for Alunai Clarity.

This module provides a complete configuration solution with:
- Schema validation and type checking
- Environment-specific configurations
- Runtime monitoring and hot-reload
- Security validation
- Migration support
- Audit trails for configuration changes

Usage:
    from clarity.shared.config import configure_alunai_clarity, get_config
    
    # Initialize configuration system
    await configure_alunai_clarity('config.json', environment='production')
    
    # Get configuration values
    qdrant_url = get_config('qdrant.url')
    
    # Set runtime overrides
    await set_config('logging.level', 'DEBUG')
"""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass

from .config_validation import (
    SecureConfigManager, ConfigValidator, ConfigSchema, 
    ConfigEnvironment, ValidationResult, ValidationRule
)
from .config_runtime import (
    RuntimeConfigMonitor, ConfigMigrator, ConfigChange, 
    get_runtime_monitor, initialize_config_monitoring
)
from .exceptions import ConfigurationError
try:
    # Try importing from the comprehensive logging system
    from .logging import get_logger, log_operation
    from .audit_trail import AuditEventType
except ImportError:
    # Fallback to simple logging for standalone testing
    from .logging import get_logger
    def log_operation(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    class AuditEventType:
        CONFIGURATION_CHANGE = "configuration_change"
        SYSTEM_START = "system_start"


@dataclass
class ConfigurationSetup:
    """Configuration setup result"""
    config_path: str
    environment: ConfigEnvironment
    validation_result: ValidationResult
    runtime_monitor: RuntimeConfigMonitor
    config_data: Dict[str, Any]


class AlunaiClarityConfig:
    """Main configuration management class for Alunai Clarity"""
    
    def __init__(self):
        """Initialize Alunai Clarity configuration manager"""
        self.logger = get_logger(__name__)
        self._initialized = False
        self._config_setup: Optional[ConfigurationSetup] = None
        self._config_migrator = ConfigMigrator()
        self._change_listeners: List[Callable[[str, Any, Any], None]] = []
        
        # Register default migrations
        self._register_default_migrations()
    
    @log_operation(
        operation_name="configure_alunai_clarity",
        actor="system",
        audit_event_type=AuditEventType.SYSTEM_START
    )
    async def configure(self, 
                       config_path: str,
                       environment: Union[str, ConfigEnvironment] = ConfigEnvironment.DEVELOPMENT,
                       auto_reload: bool = True,
                       create_default: bool = True,
                       strict_validation: bool = None,
                       migrate_if_needed: bool = True) -> ConfigurationSetup:
        """Configure Alunai Clarity with comprehensive settings
        
        Args:
            config_path: Path to configuration file
            environment: Configuration environment
            auto_reload: Enable automatic configuration reloading
            create_default: Create default configuration if file doesn't exist
            strict_validation: Enable strict validation (defaults based on environment)
            migrate_if_needed: Automatically migrate configuration if needed
            
        Returns:
            Configuration setup result
            
        Raises:
            ConfigurationError: If configuration setup fails
        """
        # Convert environment to enum if needed
        if isinstance(environment, str):
            environment = ConfigEnvironment(environment.lower())
        
        # Default strict validation for production
        if strict_validation is None:
            strict_validation = environment in [ConfigEnvironment.PRODUCTION, ConfigEnvironment.STAGING]
        
        self.logger.info("Configuring Alunai Clarity", context={
            'config_path': config_path,
            'environment': environment.value,
            'auto_reload': auto_reload,
            'strict_validation': strict_validation
        })
        
        try:
            # Ensure configuration file exists
            if not os.path.exists(config_path):
                if create_default:
                    await self._create_default_configuration(config_path, environment)
                else:
                    raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            # Initialize secure configuration manager
            secure_manager = SecureConfigManager(environment)
            
            # Load and validate configuration
            config_data, validation_result = secure_manager.load_and_validate_config(
                config_path, strict_validation
            )
            
            # Migrate configuration if needed
            if migrate_if_needed:
                config_data = await self._migrate_configuration_if_needed(
                    config_data, config_path, secure_manager
                )
            
            # Initialize runtime monitoring
            runtime_monitor = await initialize_config_monitoring(
                config_path, environment, auto_reload
            )
            
            # Register change callback
            runtime_monitor.register_change_callback(self._on_configuration_change)
            
            # Create setup result
            self._config_setup = ConfigurationSetup(
                config_path=config_path,
                environment=environment,
                validation_result=validation_result,
                runtime_monitor=runtime_monitor,
                config_data=config_data
            )
            
            self._initialized = True
            
            # Log successful configuration
            self.logger.info("Alunai Clarity configuration completed successfully", context={
                'validation_errors': len(validation_result.errors),
                'validation_warnings': len(validation_result.warnings),
                'security_issues': len(validation_result.security_issues),
                'config_sections': list(config_data.keys())
            })
            
            return self._config_setup
            
        except Exception as e:
            self.logger.error(f"Failed to configure Alunai Clarity: {str(e)}")
            raise ConfigurationError(f"Configuration setup failed: {str(e)}")
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """Get configuration value with runtime overrides
        
        Args:
            path: Dot-notation configuration path (e.g., 'qdrant.url')
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if not self._initialized:
            raise ConfigurationError("Configuration not initialized. Call configure() first.")
        
        return self._config_setup.runtime_monitor.get_config_value(path, default)
    
    async def set_config(self, path: str, value: Any, temporary: bool = True) -> None:
        """Set runtime configuration override
        
        Args:
            path: Dot-notation configuration path
            value: New value
            temporary: If True, override is lost on restart
        """
        if not self._initialized:
            raise ConfigurationError("Configuration not initialized. Call configure() first.")
        
        old_value = self.get_config(path)
        await self._config_setup.runtime_monitor.set_runtime_override(path, value, temporary)
        
        # Notify listeners
        for listener in self._change_listeners:
            try:
                listener(path, old_value, value)
            except Exception as e:
                self.logger.error(f"Configuration change listener failed: {str(e)}")
    
    def register_change_listener(self, 
                                callback: Callable[[str, Any, Any], None]) -> None:
        """Register callback for configuration changes
        
        Args:
            callback: Function called with (path, old_value, new_value)
        """
        self._change_listeners.append(callback)
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get comprehensive configuration information"""
        if not self._initialized:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "config_path": self._config_setup.config_path,
            "environment": self._config_setup.environment.value,
            "validation_summary": {
                "valid": self._config_setup.validation_result.valid,
                "errors_count": len(self._config_setup.validation_result.errors),
                "warnings_count": len(self._config_setup.validation_result.warnings),
                "security_issues_count": len(self._config_setup.validation_result.security_issues)
            },
            "runtime_info": self._config_setup.runtime_monitor.get_configuration_summary(),
            "available_sections": list(self._config_setup.config_data.keys())
        }
    
    def validate_current_config(self) -> ValidationResult:
        """Validate current configuration
        
        Returns:
            Validation result
        """
        if not self._initialized:
            raise ConfigurationError("Configuration not initialized.")
        
        # Reload to get latest validation
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._config_setup.runtime_monitor.reload_configuration()
        )
    
    async def reload_config(self) -> ValidationResult:
        """Manually reload configuration from file
        
        Returns:
            Validation result
        """
        if not self._initialized:
            raise ConfigurationError("Configuration not initialized.")
        
        return await self._config_setup.runtime_monitor.reload_configuration(force=True)
    
    def export_config(self, include_overrides: bool = True, 
                     include_sensitive: bool = False) -> Dict[str, Any]:
        """Export current configuration
        
        Args:
            include_overrides: Include runtime overrides
            include_sensitive: Include sensitive values
            
        Returns:
            Configuration dictionary
        """
        if not self._initialized:
            raise ConfigurationError("Configuration not initialized.")
        
        config = self._config_setup.config_data.copy()
        
        # Add runtime overrides if requested
        if include_overrides:
            for path, value in self._config_setup.runtime_monitor._runtime_overrides.items():
                self._set_nested_value(config, path, value)
        
        # Remove sensitive values if not requested
        if not include_sensitive:
            config = self._sanitize_sensitive_data(config)
        
        return config
    
    async def _create_default_configuration(self, config_path: str, 
                                          environment: ConfigEnvironment) -> None:
        """Create default configuration file"""
        self.logger.info(f"Creating default configuration for {environment.value} environment")
        
        secure_manager = SecureConfigManager(environment)
        secure_manager.create_validated_default_config(config_path)
    
    async def _migrate_configuration_if_needed(self, 
                                             config_data: Dict[str, Any],
                                             config_path: str,
                                             secure_manager: SecureConfigManager) -> Dict[str, Any]:
        """Migrate configuration if version differs"""
        current_version = config_data.get('version', '1.0.0')
        target_version = '1.0.0'  # Current version
        
        if current_version != target_version:
            self.logger.info(f"Migrating configuration from {current_version} to {target_version}")
            
            migrated_config = self._config_migrator.migrate_config(config_data, target_version)
            
            # Save migrated configuration
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(migrated_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Configuration migration completed successfully")
            return migrated_config
        
        return config_data
    
    def _register_default_migrations(self) -> None:
        """Register default configuration migrations"""
        
        # Example migration from 1.0.0 to 1.1.0
        def migrate_1_0_to_1_1(config: Dict[str, Any]) -> Dict[str, Any]:
            """Example migration: add new audit configuration section"""
            if 'audit' not in config:
                config['audit'] = {
                    'enabled': True,
                    'storage_backend': 'file',
                    'buffer_size': 100
                }
            return config
        
        self._config_migrator.register_migration('1.0.0', '1.1.0', migrate_1_0_to_1_1)
    
    def _on_configuration_change(self, change: ConfigChange) -> None:
        """Handle configuration change events"""
        self.logger.info("Configuration changed", context={
            'change_type': change.change_type.value,
            'path': change.path,
            'source': change.source,
            'requires_restart': change.requires_restart
        })
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        target = config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def _sanitize_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive configuration data"""
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        
        def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            sanitized = {}
            for key, value in d.items():
                key_lower = key.lower()
                if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                    sanitized[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    sanitized[key] = sanitize_dict(value)
                else:
                    sanitized[key] = value
            return sanitized
        
        return sanitize_dict(config)


# Global configuration manager instance
_alunai_config = AlunaiClarityConfig()


# Convenience functions for easy access
async def configure_alunai_clarity(config_path: str, **kwargs) -> ConfigurationSetup:
    """Configure Alunai Clarity system
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional configuration options
        
    Returns:
        Configuration setup result
    """
    return await _alunai_config.configure(config_path, **kwargs)


def get_config(path: str, default: Any = None) -> Any:
    """Get configuration value
    
    Args:
        path: Dot-notation configuration path
        default: Default value if not found
        
    Returns:
        Configuration value
    """
    return _alunai_config.get_config(path, default)


async def set_config(path: str, value: Any, temporary: bool = True) -> None:
    """Set configuration value
    
    Args:
        path: Dot-notation configuration path
        value: New value
        temporary: If True, override is lost on restart
    """
    await _alunai_config.set_config(path, value, temporary)


def register_config_change_listener(callback: Callable[[str, Any, Any], None]) -> None:
    """Register configuration change listener
    
    Args:
        callback: Function called with (path, old_value, new_value)
    """
    _alunai_config.register_change_listener(callback)


def get_config_info() -> Dict[str, Any]:
    """Get configuration system information"""
    return _alunai_config.get_configuration_info()


async def reload_config() -> ValidationResult:
    """Reload configuration from file"""
    return await _alunai_config.reload_config()


def export_config(**kwargs) -> Dict[str, Any]:
    """Export current configuration"""
    return _alunai_config.export_config(**kwargs)


# Configuration validation shortcuts
def validate_config_file(config_path: str, 
                        environment: str = 'development') -> ValidationResult:
    """Validate configuration file without loading"""
    from .config_validation import validate_config_file
    env = ConfigEnvironment(environment.lower())
    return validate_config_file(config_path, env)


def create_default_config(output_path: str, 
                         environment: str = 'development') -> Dict[str, Any]:
    """Create default configuration file"""
    from .config_validation import create_default_config
    env = ConfigEnvironment(environment.lower())
    return create_default_config(output_path, env)


# Environment helpers
def is_production() -> bool:
    """Check if running in production environment"""
    return get_config('environment', 'development').lower() == 'production'


def is_development() -> bool:
    """Check if running in development environment"""
    return get_config('environment', 'development').lower() == 'development'


def get_environment() -> str:
    """Get current environment name"""
    return get_config('environment', 'development').lower()


# Configuration-based feature flags
def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled via configuration
    
    Args:
        feature_name: Name of the feature to check
        
    Returns:
        True if feature is enabled
    """
    return get_config(f'features.{feature_name}.enabled', False)


def get_feature_config(feature_name: str) -> Dict[str, Any]:
    """Get configuration for a specific feature
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Feature configuration dictionary
    """
    return get_config(f'features.{feature_name}', {})