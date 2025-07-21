"""
Comprehensive configuration validation and schema enforcement system.

This module provides:
- JSON Schema validation with custom validators
- Environment-specific configuration validation
- Type checking and coercion
- Secure configuration management
- Configuration migration support
- Runtime configuration validation
"""

import os
import re
import ipaddress
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

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
    
    class AuditSeverity:
        INFO = "info"


class ValidationType(Enum):
    """Types of validation"""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    PATTERN_CHECK = "pattern_check"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"
    SECURITY = "security"


class ConfigEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ValidationRule:
    """Configuration validation rule"""
    name: str
    validation_type: ValidationType
    path: str  # Dot-notation path in config
    description: str
    required: bool = False
    default_value: Any = None
    allowed_types: List[type] = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    dependencies: List[str] = field(default_factory=list)
    environments: List[ConfigEnvironment] = field(default_factory=list)
    security_sensitive: bool = False


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coerced_values: Dict[str, Any] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)


class ConfigValidator:
    """Comprehensive configuration validator"""
    
    def __init__(self, environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        """Initialize configuration validator
        
        Args:
            environment: Current environment for environment-specific validation
        """
        self.environment = environment
        self.logger = get_logger(__name__)
        if hasattr(self.logger, 'context'):
            # Enhanced logger with context support
            self.logger = get_logger(__name__, context={
                'component': 'config_validator',
                'environment': environment.value if hasattr(environment, 'value') else str(environment)
            })
        else:
            # Simple logger fallback
            self.logger = get_logger(__name__)
        self.validation_rules: List[ValidationRule] = []
        self._custom_validators: Dict[str, Callable] = {}
        self._type_coercers: Dict[type, Callable] = {
            int: self._coerce_int,
            float: self._coerce_float,
            bool: self._coerce_bool,
            str: self._coerce_string,
            list: self._coerce_list,
            dict: self._coerce_dict
        }
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule
        
        Args:
            rule: Validation rule to add
        """
        # Check if rule applies to current environment
        if rule.environments and self.environment not in rule.environments:
            return
        
        self.validation_rules.append(rule)
        self.logger.debug(f"Added validation rule: {rule.name}")
    
    def register_custom_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """Register a custom validator function
        
        Args:
            name: Validator name
            validator: Validator function that returns True if valid
        """
        self._custom_validators[name] = validator
        self.logger.debug(f"Registered custom validator: {name}")
    
    @log_operation(
        operation_name="validate_configuration",
        actor="system",
        audit_event_type=AuditEventType.CONFIGURATION_CHANGE
    )
    def validate_config(self, config: Dict[str, Any], 
                       strict_mode: bool = False) -> ValidationResult:
        """Validate configuration against all rules
        
        Args:
            config: Configuration to validate
            strict_mode: If True, warnings become errors
            
        Returns:
            ValidationResult with details of validation
        """
        result = ValidationResult(valid=True)
        
        # Create a copy for potential coercion
        validated_config = self._deep_copy(config)
        
        for rule in self.validation_rules:
            try:
                rule_result = self._validate_rule(rule, validated_config, strict_mode)
                
                # Merge results
                result.errors.extend(rule_result.errors)
                result.warnings.extend(rule_result.warnings)
                result.coerced_values.update(rule_result.coerced_values)
                result.missing_required.extend(rule_result.missing_required)
                result.security_issues.extend(rule_result.security_issues)
                
            except Exception as e:
                error_msg = f"Validation rule '{rule.name}' failed: {str(e)}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Check for dependency violations
        dependency_errors = self._validate_dependencies(validated_config)
        result.errors.extend(dependency_errors)
        
        # Final validity check
        result.valid = len(result.errors) == 0 and (not strict_mode or len(result.warnings) == 0)
        
        # Log validation summary
        if hasattr(self.logger, 'audit_info'):
            self.logger.info("Configuration validation completed", context={
                'valid': result.valid,
                'errors_count': len(result.errors),
                'warnings_count': len(result.warnings),
                'security_issues_count': len(result.security_issues),
                'coerced_values_count': len(result.coerced_values)
            })
        else:
            self.logger.info(f"Configuration validation completed: Valid={result.valid}, "
                           f"Errors={len(result.errors)}, Warnings={len(result.warnings)}, "
                           f"Security={len(result.security_issues)}, Coerced={len(result.coerced_values)}")
        
        return result
    
    def _validate_rule(self, rule: ValidationRule, config: Dict[str, Any], 
                      strict_mode: bool) -> ValidationResult:
        """Validate a single rule"""
        result = ValidationResult(valid=True)
        
        # Get the value from config
        value = self._get_nested_value(config, rule.path)
        
        # Check if value exists
        if value is None:
            if rule.required:
                result.missing_required.append(rule.path)
                result.errors.append(f"Required configuration '{rule.path}' is missing")
            elif rule.default_value is not None:
                # Apply default value
                self._set_nested_value(config, rule.path, rule.default_value)
                result.coerced_values[rule.path] = rule.default_value
            return result
        
        # Type checking and coercion
        if rule.allowed_types:
            coerced_value = self._check_and_coerce_type(value, rule.allowed_types, rule.path)
            if coerced_value != value:
                result.coerced_values[rule.path] = coerced_value
                self._set_nested_value(config, rule.path, coerced_value)
                value = coerced_value
        
        # Validation type specific checks
        if rule.validation_type == ValidationType.RANGE_CHECK:
            if not self._validate_range(value, rule.min_value, rule.max_value):
                result.errors.append(
                    f"Configuration '{rule.path}' value {value} is out of range "
                    f"[{rule.min_value}, {rule.max_value}]"
                )
        
        elif rule.validation_type == ValidationType.PATTERN_CHECK:
            if rule.pattern and not re.match(rule.pattern, str(value)):
                result.errors.append(
                    f"Configuration '{rule.path}' value '{value}' does not match pattern '{rule.pattern}'"
                )
        
        elif rule.validation_type == ValidationType.CUSTOM:
            if rule.custom_validator and not rule.custom_validator(value):
                result.errors.append(
                    f"Configuration '{rule.path}' value '{value}' failed custom validation"
                )
        
        # Check allowed values
        if rule.allowed_values is not None and value not in rule.allowed_values:
            result.errors.append(
                f"Configuration '{rule.path}' value '{value}' not in allowed values: {rule.allowed_values}"
            )
        
        # Security checks
        if rule.security_sensitive:
            security_issues = self._check_security(rule.path, value)
            result.security_issues.extend(security_issues)
        
        return result
    
    def _validate_dependencies(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration dependencies"""
        errors = []
        
        # Create dependency graph
        dependencies = {}
        for rule in self.validation_rules:
            if rule.dependencies:
                dependencies[rule.path] = rule.dependencies
        
        # Check dependencies
        for config_path, deps in dependencies.items():
            if self._get_nested_value(config, config_path) is not None:
                for dep in deps:
                    if self._get_nested_value(config, dep) is None:
                        errors.append(
                            f"Configuration '{config_path}' requires '{dep}' to be set"
                        )
        
        return errors
    
    def _check_security(self, path: str, value: Any) -> List[str]:
        """Check for security issues in configuration values"""
        issues = []
        
        str_value = str(value).lower()
        
        # Check for common security anti-patterns
        if 'password' in path.lower() or 'secret' in path.lower() or 'key' in path.lower():
            # Check for weak passwords/keys
            if len(str(value)) < 8:
                issues.append(f"Security: '{path}' appears to be a weak credential (too short)")
            
            # Check for common weak passwords
            weak_patterns = ['password', '123456', 'admin', 'root', 'default']
            if any(weak in str_value for weak in weak_patterns):
                issues.append(f"Security: '{path}' appears to use a weak or default credential")
        
        # Check for insecure protocols
        if isinstance(value, str):
            if value.startswith('http://') and 'localhost' not in value and '127.0.0.1' not in value:
                issues.append(f"Security: '{path}' uses insecure HTTP protocol")
            
            if 'disable' in str_value and ('ssl' in str_value or 'tls' in str_value):
                issues.append(f"Security: '{path}' appears to disable SSL/TLS")
        
        return issues
    
    def _check_and_coerce_type(self, value: Any, allowed_types: List[type], path: str) -> Any:
        """Check type and attempt coercion if necessary"""
        if any(isinstance(value, t) for t in allowed_types):
            return value
        
        # Attempt type coercion
        for target_type in allowed_types:
            if target_type in self._type_coercers:
                try:
                    coerced = self._type_coercers[target_type](value)
                    self.logger.debug(f"Coerced '{path}' from {type(value).__name__} to {target_type.__name__}")
                    return coerced
                except (ValueError, TypeError):
                    continue
        
        # If no coercion worked, raise error
        raise ValidationError(
            f"Configuration '{path}' has type {type(value).__name__}, "
            f"expected one of: {[t.__name__ for t in allowed_types]}"
        )
    
    def _validate_range(self, value: Any, min_val: Optional[Union[int, float]], 
                       max_val: Optional[Union[int, float]]) -> bool:
        """Validate that value is within specified range"""
        if not isinstance(value, (int, float)):
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True
    
    def _coerce_int(self, value: Any) -> int:
        """Coerce value to integer"""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return int(value)  # Let it raise ValueError if can't convert
    
    def _coerce_float(self, value: Any) -> float:
        """Coerce value to float"""
        if isinstance(value, (int, bool)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        return float(value)
    
    def _coerce_bool(self, value: Any) -> bool:
        """Coerce value to boolean"""
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on', 'enabled')
        return bool(value)
    
    def _coerce_string(self, value: Any) -> str:
        """Coerce value to string"""
        return str(value)
    
    def _coerce_list(self, value: Any) -> List[Any]:
        """Coerce value to list"""
        if isinstance(value, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # Split by comma
                return [item.strip() for item in value.split(',')]
        if not isinstance(value, list):
            return [value]
        return value
    
    def _coerce_dict(self, value: Any) -> Dict[str, Any]:
        """Coerce value to dictionary"""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Cannot parse string as dictionary")
        if isinstance(value, dict):
            return value
        raise ValueError("Cannot coerce to dictionary")
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value using dot notation"""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested value using dot notation"""
        keys = path.split('.')
        target = config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy configuration object"""
        if isinstance(obj, dict):
            return {key: self._deep_copy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj


class ConfigSchema:
    """Configuration schema definition and management"""
    
    def __init__(self):
        """Initialize configuration schema"""
        self.logger = get_logger(__name__)
        self._schemas: Dict[str, List[ValidationRule]] = {}
    
    def define_alunai_clarity_schema(self) -> List[ValidationRule]:
        """Define the complete Alunai Clarity configuration schema"""
        rules = [
            # Core system configuration
            ValidationRule(
                name="version",
                validation_type=ValidationType.PATTERN_CHECK,
                path="version",
                description="Configuration version for compatibility checking",
                required=True,
                pattern=r"^\d+\.\d+\.\d+$",
                allowed_types=[str]
            ),
            
            # Qdrant configuration
            ValidationRule(
                name="qdrant_url",
                validation_type=ValidationType.PATTERN_CHECK,
                path="qdrant.url",
                description="Qdrant database connection URL",
                required=True,
                pattern=r"^https?://[\w\.-]+(:\d+)?$",
                allowed_types=[str],
                security_sensitive=True
            ),
            
            ValidationRule(
                name="qdrant_collection",
                validation_type=ValidationType.TYPE_CHECK,
                path="qdrant.collection_name",
                description="Qdrant collection name for storing memories",
                required=True,
                allowed_types=[str],
                pattern=r"^[a-zA-Z0-9_-]+$"
            ),
            
            ValidationRule(
                name="qdrant_vector_size",
                validation_type=ValidationType.RANGE_CHECK,
                path="qdrant.vector_size",
                description="Vector embedding size",
                required=True,
                allowed_types=[int],
                min_value=100,
                max_value=4096,
                default_value=384
            ),
            
            ValidationRule(
                name="qdrant_embedding_model",
                validation_type=ValidationType.TYPE_CHECK,
                path="qdrant.embedding_model",
                description="Sentence transformer model for embeddings",
                required=False,
                allowed_types=[str],
                default_value="all-MiniLM-L6-v2"
            ),
            
            # Memory management configuration
            ValidationRule(
                name="memory_default_tier",
                validation_type=ValidationType.TYPE_CHECK,
                path="memory.default_tier",
                description="Default memory tier for new memories",
                required=False,
                allowed_types=[str],
                allowed_values=["working", "episodic", "semantic", "procedural"],
                default_value="working"
            ),
            
            ValidationRule(
                name="memory_max_context",
                validation_type=ValidationType.RANGE_CHECK,
                path="memory.max_context_length",
                description="Maximum context length for memory operations",
                required=False,
                allowed_types=[int],
                min_value=1000,
                max_value=100000,
                default_value=8000
            ),
            
            ValidationRule(
                name="memory_cleanup_interval",
                validation_type=ValidationType.RANGE_CHECK,
                path="memory.cleanup_interval_hours",
                description="Interval for memory cleanup operations",
                required=False,
                allowed_types=[int],
                min_value=1,
                max_value=168,  # 1 week
                default_value=24
            ),
            
            # AutoCode configuration
            ValidationRule(
                name="autocode_enabled",
                validation_type=ValidationType.TYPE_CHECK,
                path="autocode.enabled",
                description="Enable AutoCode domain functionality",
                required=False,
                allowed_types=[bool],
                default_value=True
            ),
            
            ValidationRule(
                name="autocode_scan_projects",
                validation_type=ValidationType.TYPE_CHECK,
                path="autocode.auto_scan_projects",
                description="Automatically scan projects for patterns",
                required=False,
                allowed_types=[bool],
                default_value=True,
                dependencies=["autocode.enabled"]
            ),
            
            ValidationRule(
                name="autocode_track_bash",
                validation_type=ValidationType.TYPE_CHECK,
                path="autocode.track_bash_commands",
                description="Track bash command executions",
                required=False,
                allowed_types=[bool],
                default_value=True,
                dependencies=["autocode.enabled"]
            ),
            
            # Logging configuration
            ValidationRule(
                name="logging_level",
                validation_type=ValidationType.TYPE_CHECK,
                path="logging.level",
                description="Logging level",
                required=False,
                allowed_types=[str],
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                default_value="INFO"
            ),
            
            ValidationRule(
                name="logging_format",
                validation_type=ValidationType.TYPE_CHECK,
                path="logging.format",
                description="Logging format type",
                required=False,
                allowed_types=[str],
                allowed_values=["enhanced", "simple", "json"],
                default_value="enhanced"
            ),
            
            ValidationRule(
                name="logging_file_enabled",
                validation_type=ValidationType.TYPE_CHECK,
                path="logging.log_to_file",
                description="Enable file logging",
                required=False,
                allowed_types=[bool],
                default_value=False
            ),
            
            ValidationRule(
                name="logging_file_path",
                validation_type=ValidationType.CUSTOM,
                path="logging.log_file_path",
                description="Path for log files",
                required=False,
                allowed_types=[str],
                custom_validator=lambda x: x is None or Path(x).parent.exists() or x.startswith('/tmp'),
                dependencies=["logging.log_to_file"]
            ),
            
            # Audit trail configuration
            ValidationRule(
                name="audit_enabled",
                validation_type=ValidationType.TYPE_CHECK,
                path="audit.enabled",
                description="Enable audit trail system",
                required=False,
                allowed_types=[bool],
                default_value=True
            ),
            
            ValidationRule(
                name="audit_storage_backend",
                validation_type=ValidationType.TYPE_CHECK,
                path="audit.storage_backend",
                description="Audit storage backend type",
                required=False,
                allowed_types=[str],
                allowed_values=["file", "database", "memory"],
                default_value="file",
                dependencies=["audit.enabled"]
            ),
            
            ValidationRule(
                name="audit_buffer_size",
                validation_type=ValidationType.RANGE_CHECK,
                path="audit.buffer_size",
                description="Audit event buffer size",
                required=False,
                allowed_types=[int],
                min_value=10,
                max_value=1000,
                default_value=100,
                dependencies=["audit.enabled"]
            ),
            
            # Observability configuration
            ValidationRule(
                name="observability_tracing_enabled",
                validation_type=ValidationType.TYPE_CHECK,
                path="observability.tracing.enabled",
                description="Enable distributed tracing",
                required=False,
                allowed_types=[bool],
                default_value=True
            ),
            
            ValidationRule(
                name="observability_tracing_sampling",
                validation_type=ValidationType.RANGE_CHECK,
                path="observability.tracing.sampling_rate",
                description="Tracing sampling rate (0.0 to 1.0)",
                required=False,
                allowed_types=[float],
                min_value=0.0,
                max_value=1.0,
                default_value=1.0,
                dependencies=["observability.tracing.enabled"]
            ),
            
            ValidationRule(
                name="observability_metrics_enabled",
                validation_type=ValidationType.TYPE_CHECK,
                path="observability.metrics.enabled",
                description="Enable metrics collection",
                required=False,
                allowed_types=[bool],
                default_value=True
            ),
            
            ValidationRule(
                name="observability_health_enabled",
                validation_type=ValidationType.TYPE_CHECK,
                path="observability.health.enabled",
                description="Enable health monitoring",
                required=False,
                allowed_types=[bool],
                default_value=True
            ),
            
            # Security configuration
            ValidationRule(
                name="security_max_file_size",
                validation_type=ValidationType.RANGE_CHECK,
                path="security.max_file_size_mb",
                description="Maximum file size for processing (MB)",
                required=False,
                allowed_types=[int],
                min_value=1,
                max_value=1024,
                default_value=100
            ),
            
            ValidationRule(
                name="security_allowed_extensions",
                validation_type=ValidationType.TYPE_CHECK,
                path="security.allowed_file_extensions",
                description="Allowed file extensions for processing",
                required=False,
                allowed_types=[list],
                default_value=[".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml"]
            ),
            
            # Performance configuration
            ValidationRule(
                name="performance_max_concurrent",
                validation_type=ValidationType.RANGE_CHECK,
                path="performance.max_concurrent_operations",
                description="Maximum concurrent operations",
                required=False,
                allowed_types=[int],
                min_value=1,
                max_value=100,
                default_value=10
            ),
            
            ValidationRule(
                name="performance_timeout",
                validation_type=ValidationType.RANGE_CHECK,
                path="performance.operation_timeout_seconds",
                description="Operation timeout in seconds",
                required=False,
                allowed_types=[int],
                min_value=1,
                max_value=3600,
                default_value=300
            )
        ]
        
        return rules
    
    def get_environment_specific_rules(self, 
                                     environment: ConfigEnvironment) -> List[ValidationRule]:
        """Get validation rules specific to an environment"""
        base_rules = self.define_alunai_clarity_schema()
        
        # Add environment-specific rules
        if environment == ConfigEnvironment.PRODUCTION:
            # Production-specific rules
            prod_rules = [
                ValidationRule(
                    name="prod_https_required",
                    validation_type=ValidationType.CUSTOM,
                    path="qdrant.url",
                    description="HTTPS required in production",
                    required=True,
                    allowed_types=[str],
                    custom_validator=lambda x: x.startswith('https://') or 'localhost' in x,
                    environments=[ConfigEnvironment.PRODUCTION],
                    security_sensitive=True
                ),
                
                ValidationRule(
                    name="prod_debug_disabled",
                    validation_type=ValidationType.TYPE_CHECK,
                    path="logging.level",
                    description="Debug logging should be disabled in production",
                    required=False,
                    allowed_types=[str],
                    allowed_values=["INFO", "WARNING", "ERROR", "CRITICAL"],
                    environments=[ConfigEnvironment.PRODUCTION]
                ),
                
                ValidationRule(
                    name="prod_audit_required",
                    validation_type=ValidationType.TYPE_CHECK,
                    path="audit.enabled",
                    description="Audit trail required in production",
                    required=True,
                    allowed_types=[bool],
                    allowed_values=[True],
                    environments=[ConfigEnvironment.PRODUCTION]
                )
            ]
            base_rules.extend(prod_rules)
        
        elif environment == ConfigEnvironment.DEVELOPMENT:
            # Development-specific rules (more permissive)
            dev_rules = [
                ValidationRule(
                    name="dev_debug_allowed",
                    validation_type=ValidationType.TYPE_CHECK,
                    path="logging.level",
                    description="Debug logging allowed in development",
                    required=False,
                    allowed_types=[str],
                    allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    environments=[ConfigEnvironment.DEVELOPMENT]
                )
            ]
            base_rules.extend(dev_rules)
        
        return base_rules


class SecureConfigManager:
    """Secure configuration management with validation and encryption"""
    
    def __init__(self, environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        """Initialize secure configuration manager
        
        Args:
            environment: Configuration environment
        """
        self.environment = environment
        self.logger = get_logger(__name__)
        self.validator = ConfigValidator(environment)
        self.schema = ConfigSchema()
        
        # Load environment-specific validation rules
        rules = self.schema.get_environment_specific_rules(environment)
        for rule in rules:
            self.validator.add_validation_rule(rule)
        
        # Register custom validators
        self._register_custom_validators()
    
    def _register_custom_validators(self) -> None:
        """Register custom validation functions"""
        
        def validate_ip_address(value: str) -> bool:
            """Validate IP address"""
            try:
                ipaddress.ip_address(value)
                return True
            except ValueError:
                return False
        
        def validate_url_format(value: str) -> bool:
            """Validate URL format"""
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
                r'localhost|'  # localhost
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            return bool(url_pattern.match(value))
        
        def validate_file_path(value: str) -> bool:
            """Validate file path is safe"""
            # Check for path traversal attempts
            if '..' in value or value.startswith('/'):
                return False
            # Check if parent directory exists or is creatable
            try:
                Path(value).parent.mkdir(parents=True, exist_ok=True)
                return True
            except (OSError, PermissionError):
                return False
        
        self.validator.register_custom_validator('ip_address', validate_ip_address)
        self.validator.register_custom_validator('url_format', validate_url_format)
        self.validator.register_custom_validator('file_path', validate_file_path)
    
    @log_operation(
        operation_name="validate_and_load_config",
        actor="system",
        audit_event_type=AuditEventType.CONFIGURATION_CHANGE
    )
    def load_and_validate_config(self, config_path: str,
                                strict_mode: bool = None) -> Tuple[Dict[str, Any], ValidationResult]:
        """Load and validate configuration from file
        
        Args:
            config_path: Path to configuration file
            strict_mode: Enable strict validation (defaults based on environment)
            
        Returns:
            Tuple of (validated_config, validation_result)
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Default strict mode based on environment
        if strict_mode is None:
            strict_mode = self.environment in [ConfigEnvironment.PRODUCTION, ConfigEnvironment.STAGING]
        
        # Load configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")
        
        # Validate configuration
        result = self.validator.validate_config(config, strict_mode)
        
        # Apply coerced values
        for path, value in result.coerced_values.items():
            self.logger.debug(f"Applied coerced value for {path}: {value}")
        
        # Log validation results
        if result.errors:
            error_msg = f"Configuration validation failed with {len(result.errors)} errors"
            if hasattr(self.logger, 'audit_error'):
                self.logger.error(error_msg, context={
                    'config_path': config_path,
                    'errors': result.errors[:5],  # First 5 errors
                    'total_errors': len(result.errors)
                })
            else:
                self.logger.error(f"{error_msg}: {'; '.join(result.errors[:3])}")
            
            if strict_mode or self.environment == ConfigEnvironment.PRODUCTION:
                raise ConfigurationError(f"{error_msg}: {'; '.join(result.errors)}")
        
        if result.warnings:
            if hasattr(self.logger, 'audit_warning'):
                self.logger.warning(f"Configuration has {len(result.warnings)} warnings", context={
                    'config_path': config_path,
                    'warnings': result.warnings[:3],  # First 3 warnings
                    'total_warnings': len(result.warnings)
                })
            else:
                self.logger.warning(f"Configuration has {len(result.warnings)} warnings: {'; '.join(result.warnings[:3])}")
        
        if result.security_issues:
            if hasattr(self.logger, 'audit_critical'):
                self.logger.critical(f"Configuration has {len(result.security_issues)} security issues", context={
                    'config_path': config_path,
                    'security_issues': result.security_issues
                })
            else:
                self.logger.critical(f"Configuration has {len(result.security_issues)} security issues: {'; '.join(result.security_issues)}")
        
        return config, result
    
    def create_validated_default_config(self, output_path: str) -> Dict[str, Any]:
        """Create a default configuration file with all validated defaults
        
        Args:
            output_path: Path where to save the default configuration
            
        Returns:
            Default configuration dictionary
        """
        config = {}
        
        # Apply all default values from validation rules
        for rule in self.validator.validation_rules:
            if rule.default_value is not None:
                self.validator._set_nested_value(config, rule.path, rule.default_value)
        
        # Validate the default configuration
        result = self.validator.validate_config(config, strict_mode=False)
        
        if not result.valid:
            raise ConfigurationError(f"Default configuration is invalid: {'; '.join(result.errors)}")
        
        # Save to file
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Created validated default configuration at {output_path}")
            
        except (OSError, TypeError) as e:
            raise ConfigurationError(f"Failed to save default configuration: {str(e)}")
        
        return config


# Global instances
_secure_config_manager = None

def get_secure_config_manager(environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT) -> SecureConfigManager:
    """Get global secure configuration manager instance"""
    global _secure_config_manager
    if _secure_config_manager is None:
        _secure_config_manager = SecureConfigManager(environment)
    return _secure_config_manager


# Convenience functions
def validate_config_file(config_path: str, 
                        environment: Union[str, ConfigEnvironment] = ConfigEnvironment.DEVELOPMENT,
                        strict_mode: bool = None) -> ValidationResult:
    """Convenience function to validate a configuration file"""
    # Convert string to enum if needed
    if isinstance(environment, str):
        environment = ConfigEnvironment(environment.lower())
    
    manager = get_secure_config_manager(environment)
    _, result = manager.load_and_validate_config(config_path, strict_mode)
    return result


def create_default_config(output_path: str,
                         environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT) -> Dict[str, Any]:
    """Convenience function to create default configuration"""
    manager = get_secure_config_manager(environment)
    return manager.create_validated_default_config(output_path)