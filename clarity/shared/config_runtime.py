"""
Runtime configuration monitoring, migration, and hot-reload capabilities.

This module provides:
- Configuration hot-reloading without restart
- Configuration migration between versions
- Runtime configuration monitoring and alerts
- Environment variable integration
- Configuration templating and interpolation
"""

import os
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config_validation import SecureConfigManager, ConfigEnvironment, ValidationResult
from .exceptions import ConfigurationError, ValidationError
try:
    # Try importing from the comprehensive logging system
    from .logging import get_logger, log_operation
    from .audit_trail import AuditEventType, AuditSeverity
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
    
    class AuditSeverity:
        INFO = "info"


class ConfigChangeType(Enum):
    """Types of configuration changes"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RELOADED = "reloaded"
    MIGRATED = "migrated"


@dataclass
class ConfigChange:
    """Configuration change event"""
    change_type: ConfigChangeType
    path: str
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    requires_restart: bool = False


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, config_monitor: 'RuntimeConfigMonitor'):
        """Initialize config file watcher
        
        Args:
            config_monitor: Runtime config monitor instance
        """
        super().__init__()
        self.config_monitor = config_monitor
        self.logger = get_logger(__name__)
        self._last_modified = {}
        self._debounce_delay = 1.0  # seconds
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        # Check if it's a config file
        file_path = event.src_path
        if not any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml']):
            return
        
        # Debounce file system events
        current_time = time.time()
        if file_path in self._last_modified:
            if current_time - self._last_modified[file_path] < self._debounce_delay:
                return
        
        self._last_modified[file_path] = current_time
        
        # Trigger reload asynchronously
        asyncio.create_task(self.config_monitor.handle_file_change(file_path))
    
    def on_created(self, event):
        """Handle file creation events"""
        self.on_modified(event)
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        asyncio.create_task(self.config_monitor.handle_file_deletion(file_path))


class RuntimeConfigMonitor:
    """Runtime configuration monitoring and hot-reload system"""
    
    def __init__(self, 
                 environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT,
                 auto_reload: bool = True,
                 validation_on_change: bool = True):
        """Initialize runtime configuration monitor
        
        Args:
            environment: Configuration environment
            auto_reload: Enable automatic configuration reloading
            validation_on_change: Validate configuration on changes
        """
        self.environment = environment
        self.auto_reload = auto_reload
        self.validation_on_change = validation_on_change
        
        self.logger = get_logger(__name__, context={
            'component': 'config_monitor',
            'environment': environment.value
        })
        
        self.secure_manager = SecureConfigManager(environment)
        self._current_config: Dict[str, Any] = {}
        self._config_file_path: Optional[str] = None
        self._change_callbacks: List[Callable[[ConfigChange], None]] = []
        self._observer: Optional[Observer] = None
        self._running = False
        self._config_history: List[ConfigChange] = []
        self._max_history = 100
        
        # Runtime configuration cache
        self._runtime_overrides: Dict[str, Any] = {}
        self._environment_variables: Dict[str, str] = {}
    
    @log_operation(
        operation_name="initialize_config_monitoring",
        actor="system",
        audit_event_type=AuditEventType.SYSTEM_START
    )
    async def initialize(self, config_file_path: str) -> None:
        """Initialize configuration monitoring
        
        Args:
            config_file_path: Path to the main configuration file
        """
        self._config_file_path = config_file_path
        
        # Load initial configuration
        await self.reload_configuration()
        
        # Load environment variables
        self._load_environment_variables()
        
        # Start file watching if auto-reload is enabled
        if self.auto_reload:
            self._start_file_watching()
        
        self.logger.info("Runtime configuration monitoring initialized", context={
            'config_file': config_file_path,
            'auto_reload': self.auto_reload,
            'validation_enabled': self.validation_on_change
        })
    
    async def shutdown(self) -> None:
        """Shutdown configuration monitoring"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        self._running = False
        self.logger.info("Configuration monitoring shutdown complete")
    
    @log_operation(
        operation_name="reload_configuration",
        actor="system",
        audit_event_type=AuditEventType.CONFIGURATION_CHANGE
    )
    async def reload_configuration(self, force: bool = False) -> ValidationResult:
        """Reload configuration from file
        
        Args:
            force: Force reload even if file hasn't changed
            
        Returns:
            Validation result
        """
        if not self._config_file_path:
            raise ConfigurationError("No configuration file path set")
        
        try:
            # Load and validate configuration
            new_config, validation_result = self.secure_manager.load_and_validate_config(
                self._config_file_path,
                strict_mode=(self.environment == ConfigEnvironment.PRODUCTION)
            )
            
            # Apply environment variable overrides
            self._apply_environment_overrides(new_config)
            
            # Apply runtime overrides
            self._apply_runtime_overrides(new_config)
            
            # Check for changes
            if self._current_config != new_config or force:
                old_config = self._current_config.copy()
                self._current_config = new_config
                
                # Record change
                change = ConfigChange(
                    change_type=ConfigChangeType.RELOADED,
                    path=self._config_file_path,
                    old_value=old_config,
                    new_value=new_config,
                    source="file_reload"
                )
                
                await self._record_change(change)
                await self._notify_change_callbacks(change)
                
                self.logger.info("Configuration reloaded successfully", context={
                    'validation_errors': len(validation_result.errors),
                    'validation_warnings': len(validation_result.warnings),
                    'security_issues': len(validation_result.security_issues)
                })
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {str(e)}")
            raise
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get configuration value with runtime overrides
        
        Args:
            path: Dot-notation configuration path
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        # Check runtime overrides first
        if path in self._runtime_overrides:
            return self._runtime_overrides[path]
        
        # Check environment variables
        env_key = self._path_to_env_var(path)
        if env_key in self._environment_variables:
            return self._coerce_env_value(self._environment_variables[env_key], path)
        
        # Get from main configuration
        return self._get_nested_value(self._current_config, path, default)
    
    @log_operation(
        operation_name="set_runtime_override",
        actor="system",
        audit_event_type=AuditEventType.CONFIGURATION_CHANGE
    )
    async def set_runtime_override(self, path: str, value: Any, 
                                  temporary: bool = True) -> None:
        """Set runtime configuration override
        
        Args:
            path: Dot-notation configuration path
            value: Override value
            temporary: If True, override is lost on restart
        """
        old_value = self.get_config_value(path)
        self._runtime_overrides[path] = value
        
        # Record change
        change = ConfigChange(
            change_type=ConfigChangeType.MODIFIED,
            path=path,
            old_value=old_value,
            new_value=value,
            source="runtime_override",
            requires_restart=not temporary
        )
        
        await self._record_change(change)
        await self._notify_change_callbacks(change)
        
        self.logger.info(f"Set runtime override for '{path}'", context={
            'old_value': old_value,
            'new_value': value,
            'temporary': temporary
        })
    
    def remove_runtime_override(self, path: str) -> bool:
        """Remove runtime configuration override
        
        Args:
            path: Dot-notation configuration path
            
        Returns:
            True if override was removed, False if it didn't exist
        """
        if path in self._runtime_overrides:
            del self._runtime_overrides[path]
            self.logger.debug(f"Removed runtime override for '{path}'")
            return True
        return False
    
    def register_change_callback(self, callback: Callable[[ConfigChange], None]) -> None:
        """Register callback for configuration changes
        
        Args:
            callback: Function to call when configuration changes
        """
        self._change_callbacks.append(callback)
        self.logger.debug("Registered configuration change callback")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        return {
            'environment': self.environment.value,
            'config_file_path': self._config_file_path,
            'auto_reload_enabled': self.auto_reload,
            'validation_enabled': self.validation_on_change,
            'runtime_overrides_count': len(self._runtime_overrides),
            'environment_variables_count': len(self._environment_variables),
            'change_history_count': len(self._config_history),
            'last_reload': self._config_history[-1].timestamp.isoformat() if self._config_history else None,
            'current_config_keys': list(self._current_config.keys()),
            'monitoring_active': self._running
        }
    
    def get_change_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent configuration change history
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            List of change events
        """
        recent_changes = self._config_history[-limit:]
        return [
            {
                'change_type': change.change_type.value,
                'path': change.path,
                'timestamp': change.timestamp.isoformat(),
                'source': change.source,
                'requires_restart': change.requires_restart,
                'has_old_value': change.old_value is not None,
                'has_new_value': change.new_value is not None
            }
            for change in recent_changes
        ]
    
    async def handle_file_change(self, file_path: str) -> None:
        """Handle file system change event"""
        if file_path == self._config_file_path:
            try:
                await self.reload_configuration()
            except Exception as e:
                self.logger.error(f"Failed to reload configuration after file change: {str(e)}")
    
    async def handle_file_deletion(self, file_path: str) -> None:
        """Handle file deletion event"""
        if file_path == self._config_file_path:
            self.logger.critical(f"Configuration file deleted: {file_path}")
            
            change = ConfigChange(
                change_type=ConfigChangeType.DELETED,
                path=file_path,
                source="file_system"
            )
            
            await self._record_change(change)
            await self._notify_change_callbacks(change)
    
    def _start_file_watching(self) -> None:
        """Start file system watching"""
        if not self._config_file_path:
            return
        
        config_dir = os.path.dirname(os.path.abspath(self._config_file_path))
        
        self._observer = Observer()
        event_handler = ConfigFileWatcher(self)
        self._observer.schedule(event_handler, config_dir, recursive=False)
        self._observer.start()
        self._running = True
        
        self.logger.debug(f"Started file watching for directory: {config_dir}")
    
    def _load_environment_variables(self) -> None:
        """Load environment variables that override configuration"""
        env_prefix = "ALUNAI_CLARITY_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_path = key[len(env_prefix):].lower().replace('_', '.')
                self._environment_variables[config_path] = value
        
        self.logger.debug(f"Loaded {len(self._environment_variables)} environment variable overrides")
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> None:
        """Apply environment variable overrides to configuration"""
        for path, value in self._environment_variables.items():
            coerced_value = self._coerce_env_value(value, path)
            self._set_nested_value(config, path, coerced_value)
    
    def _apply_runtime_overrides(self, config: Dict[str, Any]) -> None:
        """Apply runtime overrides to configuration"""
        for path, value in self._runtime_overrides.items():
            self._set_nested_value(config, path, value)
    
    def _path_to_env_var(self, path: str) -> str:
        """Convert configuration path to environment variable name"""
        return f"ALUNAI_CLARITY_{path.upper().replace('.', '_')}"
    
    def _coerce_env_value(self, value: str, path: str) -> Any:
        """Coerce environment variable string to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Return as string
        return value
    
    def _get_nested_value(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation"""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        target = config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    async def _record_change(self, change: ConfigChange) -> None:
        """Record configuration change in history"""
        self._config_history.append(change)
        
        # Limit history size
        if len(self._config_history) > self._max_history:
            self._config_history = self._config_history[-self._max_history:]
    
    async def _notify_change_callbacks(self, change: ConfigChange) -> None:
        """Notify all registered change callbacks"""
        for callback in self._change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(change)
                else:
                    callback(change)
            except Exception as e:
                self.logger.error(f"Configuration change callback failed: {str(e)}")


class ConfigMigrator:
    """Configuration migration system for version upgrades"""
    
    def __init__(self):
        """Initialize configuration migrator"""
        self.logger = get_logger(__name__)
        self._migration_handlers: Dict[str, Callable] = {}
    
    def register_migration(self, from_version: str, to_version: str, 
                          handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Register migration handler
        
        Args:
            from_version: Source version
            to_version: Target version
            handler: Migration function
        """
        key = f"{from_version}->{to_version}"
        self._migration_handlers[key] = handler
        self.logger.debug(f"Registered migration: {key}")
    
    @log_operation(
        operation_name="migrate_configuration",
        actor="system",
        audit_event_type=AuditEventType.CONFIGURATION_CHANGE
    )
    def migrate_config(self, config: Dict[str, Any], target_version: str) -> Dict[str, Any]:
        """Migrate configuration to target version
        
        Args:
            config: Configuration to migrate
            target_version: Target version
            
        Returns:
            Migrated configuration
            
        Raises:
            ConfigurationError: If migration fails
        """
        current_version = config.get('version', '1.0.0')
        
        if current_version == target_version:
            return config
        
        # Find migration path
        migration_path = self._find_migration_path(current_version, target_version)
        if not migration_path:
            raise ConfigurationError(
                f"No migration path found from {current_version} to {target_version}"
            )
        
        # Apply migrations in sequence
        migrated_config = config.copy()
        
        for from_ver, to_ver in migration_path:
            key = f"{from_ver}->{to_ver}"
            if key in self._migration_handlers:
                try:
                    migrated_config = self._migration_handlers[key](migrated_config)
                    migrated_config['version'] = to_ver
                    self.logger.info(f"Applied migration: {key}")
                except Exception as e:
                    raise ConfigurationError(f"Migration {key} failed: {str(e)}")
            else:
                raise ConfigurationError(f"Missing migration handler for {key}")
        
        return migrated_config
    
    def _find_migration_path(self, from_version: str, to_version: str) -> List[Tuple[str, str]]:
        """Find migration path between versions"""
        # Simple implementation - assumes linear versioning
        # In practice, this would use a more sophisticated graph algorithm
        
        available_migrations = []
        for key in self._migration_handlers.keys():
            from_ver, to_ver = key.split('->')
            available_migrations.append((from_ver, to_ver))
        
        # For now, just check if direct migration exists
        if (from_version, to_version) in available_migrations:
            return [(from_version, to_version)]
        
        return []


# Global instances
_runtime_monitor = None

def get_runtime_monitor(environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT) -> RuntimeConfigMonitor:
    """Get global runtime configuration monitor"""
    global _runtime_monitor
    if _runtime_monitor is None:
        _runtime_monitor = RuntimeConfigMonitor(environment)
    return _runtime_monitor


# Convenience functions
async def initialize_config_monitoring(config_path: str, 
                                     environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT,
                                     auto_reload: bool = True) -> RuntimeConfigMonitor:
    """Initialize configuration monitoring system"""
    monitor = get_runtime_monitor(environment)
    await monitor.initialize(config_path)
    return monitor


def get_runtime_config_value(path: str, default: Any = None) -> Any:
    """Get configuration value with runtime overrides"""
    if _runtime_monitor:
        return _runtime_monitor.get_config_value(path, default)
    return default


async def set_runtime_config(path: str, value: Any, temporary: bool = True) -> None:
    """Set runtime configuration override"""
    if _runtime_monitor:
        await _runtime_monitor.set_runtime_override(path, value, temporary)