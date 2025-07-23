"""
Ultrathink Configuration Management for MCP Hook Integration.

This module provides configuration management for the automatic ultrathink
enablement feature when structured thinking patterns are detected.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UltrathinkConfig:
    """Configuration manager for ultrathink auto-enablement feature."""
    
    DEFAULT_CONFIG = {
        'enabled': True,
        'minimum_confidence': 0.7,
        'ultrathink_directive': "\n\n🧠 **Enhanced Thinking Mode Enabled**: Please engage ultrathink mode for this complex reasoning task. Take time to think through each step carefully, consider multiple perspectives, and provide detailed analysis.",
        'custom_patterns': [],
        'pattern_weights': {
            'high_confidence': ['structured', 'step.by.step', 'systematic', 'comprehensive'],
            'medium_confidence': ['complex', 'multiple', 'thorough', 'detailed'],
            'low_confidence': ['approach', 'analyze', 'consider', 'evaluate']
        },
        'confidence_scores': {
            'high_confidence': 0.3,
            'medium_confidence': 0.2,
            'low_confidence': 0.1
        },
        'exclusion_patterns': [
            r'\bquick\s+(question|answer)\b',
            r'\bsimple\s+(question|answer|task)\b',
            r'\bjust\s+(need|want|checking)\b',
            r'\bbrief\s+(summary|overview)\b'
        ]
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ultrathink configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Try to use the same directory structure as the main MCP config
        config_dirs = [
            os.path.expanduser('~/.claude/alunai-clarity'),
            os.path.expanduser('~/.config/alunai-clarity'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'config')
        ]
        
        for config_dir in config_dirs:
            if os.path.exists(config_dir):
                return os.path.join(config_dir, 'ultrathink_config.json')
        
        # Default to the first option (create if needed)
        default_dir = config_dirs[0]
        os.makedirs(default_dir, exist_ok=True)
        return os.path.join(default_dir, 'ultrathink_config.json')
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge with default config
                self.config.update(user_config)
                logger.info(f"Loaded ultrathink config from {self.config_path}")
            else:
                # Create default config file
                self._save_config()
                logger.info(f"Created default ultrathink config at {self.config_path}")
                
        except Exception as e:
            logger.warning(f"Failed to load ultrathink config from {self.config_path}: {e}")
            logger.info("Using default ultrathink configuration")
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved ultrathink config to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save ultrathink config to {self.config_path}: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
        self._save_config()
        logger.info(f"Updated ultrathink configuration: {updates}")
    
    def is_enabled(self) -> bool:
        """Check if ultrathink auto-enablement is enabled."""
        return self.config.get('enabled', True)
    
    def get_minimum_confidence(self) -> float:
        """Get the minimum confidence threshold."""
        return self.config.get('minimum_confidence', 0.7)
    
    def get_ultrathink_directive(self) -> str:
        """Get the ultrathink directive text."""
        return self.config.get('ultrathink_directive', self.DEFAULT_CONFIG['ultrathink_directive'])
    
    def get_custom_patterns(self) -> List[str]:
        """Get user-defined custom patterns."""
        return self.config.get('custom_patterns', [])
    
    def get_pattern_weights(self) -> Dict[str, List[str]]:
        """Get pattern weight categories."""
        return self.config.get('pattern_weights', self.DEFAULT_CONFIG['pattern_weights'])
    
    def get_confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for different pattern categories."""
        return self.config.get('confidence_scores', self.DEFAULT_CONFIG['confidence_scores'])
    
    def get_exclusion_patterns(self) -> List[str]:
        """Get patterns that should exclude ultrathink enablement."""
        return self.config.get('exclusion_patterns', [])
    
    def add_custom_pattern(self, pattern: str, weight_category: str = 'medium_confidence') -> None:
        """
        Add a custom pattern for detecting structured thinking.
        
        Args:
            pattern: Regular expression pattern
            weight_category: Weight category (high_confidence, medium_confidence, low_confidence)
        """
        if 'custom_patterns' not in self.config:
            self.config['custom_patterns'] = []
        
        custom_pattern_entry = {
            'pattern': pattern,
            'weight_category': weight_category,
            'added_by': 'user'
        }
        
        self.config['custom_patterns'].append(custom_pattern_entry)
        self._save_config()
        logger.info(f"Added custom ultrathink pattern: {pattern} ({weight_category})")
    
    def remove_custom_pattern(self, pattern: str) -> bool:
        """
        Remove a custom pattern.
        
        Args:
            pattern: Pattern to remove
            
        Returns:
            True if pattern was found and removed, False otherwise
        """
        if 'custom_patterns' not in self.config:
            return False
        
        original_count = len(self.config['custom_patterns'])
        self.config['custom_patterns'] = [
            p for p in self.config['custom_patterns'] 
            if p.get('pattern') != pattern
        ]
        
        if len(self.config['custom_patterns']) < original_count:
            self._save_config()
            logger.info(f"Removed custom ultrathink pattern: {pattern}")
            return True
        
        return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()
        self._save_config()
        logger.info("Reset ultrathink configuration to defaults")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            'config_path': self.config_path,
            'enabled': self.is_enabled(),
            'minimum_confidence': self.get_minimum_confidence(),
            'custom_patterns_count': len(self.get_custom_patterns()),
            'exclusion_patterns_count': len(self.get_exclusion_patterns()),
            'directive_length': len(self.get_ultrathink_directive()),
            'last_modified': os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else None
        }


def load_ultrathink_config(config_path: Optional[str] = None) -> UltrathinkConfig:
    """
    Load ultrathink configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        UltrathinkConfig instance
    """
    return UltrathinkConfig(config_path)


def create_sample_config(output_path: str) -> None:
    """
    Create a sample configuration file with documentation.
    
    Args:
        output_path: Path where to create the sample config
    """
    sample_config = {
        "_comment": "Ultrathink Auto-Enablement Configuration",
        "_description": "This configuration controls when ultrathink mode is automatically enabled based on prompt patterns",
        
        "enabled": True,
        "_enabled_comment": "Set to false to disable automatic ultrathink enablement",
        
        "minimum_confidence": 0.7,
        "_minimum_confidence_comment": "Confidence threshold (0.0-1.0) for enabling ultrathink. Higher values are more selective.",
        
        "ultrathink_directive": "\n\n🧠 **Enhanced Thinking Mode Enabled**: Please engage ultrathink mode for this complex reasoning task. Take time to think through each step carefully, consider multiple perspectives, and provide detailed analysis.",
        "_ultrathink_directive_comment": "Text appended to prompts when ultrathink is enabled",
        
        "custom_patterns": [
            {
                "pattern": r"\barchitectural\s+design\b",
                "weight_category": "high_confidence",
                "added_by": "user",
                "_comment": "Example custom pattern for architectural design discussions"
            }
        ],
        "_custom_patterns_comment": "User-defined patterns for detecting structured thinking needs",
        
        "pattern_weights": {
            "high_confidence": ["structured", "step.by.step", "systematic", "comprehensive"],
            "medium_confidence": ["complex", "multiple", "thorough", "detailed"],
            "low_confidence": ["approach", "analyze", "consider", "evaluate"]
        },
        "_pattern_weights_comment": "Keywords that influence confidence scoring",
        
        "confidence_scores": {
            "high_confidence": 0.3,
            "medium_confidence": 0.2,
            "low_confidence": 0.1
        },
        "_confidence_scores_comment": "Point values added to confidence score for each category match",
        
        "exclusion_patterns": [
            r"\bquick\s+(question|answer)\b",
            r"\bsimple\s+(question|answer|task)\b",
            r"\bjust\s+(need|want|checking)\b",
            r"\bbrief\s+(summary|overview)\b"
        ],
        "_exclusion_patterns_comment": "Patterns that prevent ultrathink enablement even if other patterns match"
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample ultrathink configuration created at: {output_path}")


if __name__ == "__main__":
    # CLI for managing ultrathink configuration
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultrathink Configuration Manager")
    parser.add_argument('--create-sample', type=str, help="Create sample config file at specified path")
    parser.add_argument('--show-config', action='store_true', help="Show current configuration")
    parser.add_argument('--reset', action='store_true', help="Reset to default configuration")
    parser.add_argument('--config-path', type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config(args.create_sample)
    elif args.show_config:
        config = load_ultrathink_config(args.config_path)
        print("Current Ultrathink Configuration:")
        print(json.dumps(config.get_config(), indent=2))
    elif args.reset:
        config = load_ultrathink_config(args.config_path)
        config.reset_to_defaults()
        print("Configuration reset to defaults")
    else:
        parser.print_help()