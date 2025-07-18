"""
Automatic memory capture utilities.

This module provides functions for automatically determining
when to store memories and extracting content from messages.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def should_store_memory(message: str, threshold: float = 0.6) -> bool:
    """
    Determine if a message contains information worth storing in memory.
    
    Uses simple heuristics to decide if the message likely contains personal
    information, preferences, or important facts.
    
    Args:
        message: The message to analyze
        threshold: Threshold for importance (0.0-1.0)
        
    Returns:
        True if the message should be stored, False otherwise
    """
    # Check for personal preference indicators
    preference_patterns = [
        r"I (?:like|love|enjoy|prefer|favorite|hate|dislike)",
        r"my favorite",
        r"I am (?:a|an)",
        r"I'm (?:a|an)",
        r"my name is",
        r"call me",
        r"I work",
        r"I live",
        r"my (?:husband|wife|partner|spouse|child|son|daughter|pet)",
        r"I have (?:a|an|\\d+)",
        r"I often",
        r"I usually",
        r"I always",
        r"I never",
    ]
    
    # Check for factual information
    fact_patterns = [
        r"(?:is|are|was|were) (?:born|founded|created|established|started) (?:in|on|by)",
        r"(?:is|are|was|were) (?:the|a|an) (?:capital|largest|smallest|best|worst|most|least)",
        r"(?:is|are|was|were) (?:located|situated|found|discovered)",
        r"(?:is|are|was|were) (?:invented|designed|developed)",
    ]
    
    # Calculate message complexity (proxy for information richness)
    words = message.split()
    complexity = min(1.0, len(words) / 50.0)  # Normalize to 0.0-1.0
    
    # Check for presence of preference indicators
    preference_score = 0.0
    for pattern in preference_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            preference_score = 0.8
            break
    
    # Check for presence of fact indicators
    fact_score = 0.0
    for pattern in fact_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            fact_score = 0.6
            break
    
    # Question sentences typically don't contain storable information
    question_ratio = len(re.findall(r"\?", message)) / max(1, len(re.findall(r"[.!?]", message)))
    
    # Combined score
    combined_score = max(preference_score, fact_score) * (1.0 - question_ratio) * complexity
    
    return combined_score >= threshold


def extract_memory_content(message: str) -> Tuple[str, Dict[str, Any], float]:
    """
    Extract memory content, type, and importance from a message.
    
    Args:
        message: The message to extract from
        
    Returns:
        Tuple of (memory_type, content_dict, importance)
    """
    # Check if it's likely about the user (preferences, personal info)
    user_patterns = [
        r"I (?:like|love|enjoy|prefer|favorite|hate|dislike)",
        r"my favorite", 
        r"I am (?:a|an)",
        r"I'm (?:a|an)",
        r"my name is",
        r"call me",
        r"I work",
        r"I live",
    ]
    
    # Check for fact patterns
    fact_patterns = [
        r"(?:is|are|was|were) (?:born|founded|created|established|started) (?:in|on|by)",
        r"(?:is|are|was|were) (?:the|a|an) (?:capital|largest|smallest|best|worst|most|least)",
        r"(?:is|are|was|were) (?:located|situated|found|discovered)",
        r"(?:is|are|was|were) (?:invented|designed|developed)",
    ]
    
    # Default values
    memory_type = "conversation"
    content = {"role": "user", "message": message}
    importance = 0.5
    
    # Check for user preferences or traits (entity memory)
    for pattern in user_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            memory_type = "entity"
            # Basic extraction of attribute
            attribute_match = re.search(r"I (?:like|love|enjoy|prefer|hate|dislike) (.+?)(?:\.|$|,)", message, re.IGNORECASE)
            if attribute_match:
                attribute_value = attribute_match.group(1).strip()
                content = {
                    "name": "user",
                    "entity_type": "person",
                    "attributes": {
                        "preference": attribute_value
                    }
                }
                importance = 0.7
                return memory_type, content, importance
                
            # Check for "I am" statements
            trait_match = re.search(r"I (?:am|'m) (?:a|an) (.+?)(?:\.|$|,)", message, re.IGNORECASE)
            if trait_match:
                trait_value = trait_match.group(1).strip()
                content = {
                    "name": "user",
                    "entity_type": "person",
                    "attributes": {
                        "trait": trait_value
                    }
                }
                importance = 0.7
                return memory_type, content, importance
                
            # Default entity if specific extraction fails
            content = {
                "name": "user",
                "entity_type": "person",
                "attributes": {
                    "statement": message
                }
            }
            importance = 0.6
            return memory_type, content, importance
    
    # Check for factual information
    for pattern in fact_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            memory_type = "fact"
            content = {
                "fact": message,
                "confidence": 0.8,
                "domain": "general"
            }
            importance = 0.6
            return memory_type, content, importance
    
    # Default as conversation memory with moderate importance
    return memory_type, content, importance