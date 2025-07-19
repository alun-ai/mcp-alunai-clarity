"""
AutoCode domain for code project intelligence and command learning.

This module provides intelligent code project awareness, command learning,
and session history tracking for the MCP persistent memory system.
"""

__version__ = "0.1.0"

from .domain import AutoCodeDomain
from .command_learner import CommandLearner
from .pattern_detector import PatternDetector
from .session_analyzer import SessionAnalyzer
from .history_navigator import HistoryNavigator
from .hooks import AutoCodeHooks
from .server import AutoCodeServerExtension
from .hook_manager import HookManager, HookRegistry

__all__ = [
    "AutoCodeDomain", 
    "CommandLearner", 
    "PatternDetector",
    "SessionAnalyzer",
    "HistoryNavigator",
    "AutoCodeHooks", 
    "AutoCodeServerExtension",
    "HookManager",
    "HookRegistry"
]