"""
Automatic memory management module.

This module provides automatic memory capture and retrieval capabilities
to make memory functionality more intuitive and seamless.
"""

from .system_prompt import get_memory_system_prompt, get_memory_integration_template
from .auto_capture import should_store_memory, extract_memory_content
