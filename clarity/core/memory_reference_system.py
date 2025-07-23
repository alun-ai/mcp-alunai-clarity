"""
Memory Reference System with @memory:pattern:// URI Support.

This module provides a sophisticated memory reference system that allows users
to create, resolve, and manage memory references using URI-like patterns.
It supports complex pattern matching, caching, and cross-system integration.
"""

import re
import time
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, parse_qs
import asyncio

from ..core.mcp_memory_retrieval import get_mcp_memory_retriever, MCPContext, RetrievalStrategy
from ..core.unified_cache import cache_get, cache_put, CacheType
from ..mcp.cache_integration import get_mcp_cache_adapter

logger = logging.getLogger(__name__)


class ReferenceType(str, Enum):
    """Types of memory references."""
    DIRECT = "direct"              # @memory:direct://memory_id
    PATTERN = "pattern"            # @memory:pattern://category/subcategory
    QUERY = "query"               # @memory:query://search_terms
    WORKFLOW = "workflow"         # @memory:workflow://workflow_pattern_id
    CORRELATION = "correlation"   # @memory:correlation://context_hash
    TEMPORAL = "temporal"         # @memory:temporal://timeframe/category
    SEMANTIC = "semantic"         # @memory:semantic://concept/relationship


class ReferenceScope(str, Enum):
    """Scope of memory reference resolution."""
    LOCAL = "local"               # Current session/context only
    SESSION = "session"           # Current session and related
    PROJECT = "project"           # Project-specific memories
    GLOBAL = "global"             # All available memories
    MCP_ENHANCED = "mcp_enhanced" # MCP-integrated memories only


@dataclass
class MemoryReferenceSpec:
    """Specification for a memory reference."""
    reference_uri: str
    reference_type: ReferenceType
    scope: ReferenceScope
    pattern_components: Dict[str, str] = field(default_factory=dict)
    query_parameters: Dict[str, List[str]] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiry_time: Optional[float] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ResolvedReference:
    """Result of memory reference resolution."""
    reference_spec: MemoryReferenceSpec
    matched_memories: List[Dict[str, Any]]
    resolution_confidence: float
    resolution_time: float
    cache_hit: bool
    resolution_context: Dict[str, Any]
    error_message: Optional[str] = None


class MemoryReferenceParser:
    """Parser for memory reference URIs."""
    
    def __init__(self):
        """Initialize the parser with reference patterns."""
        self.reference_patterns = {
            ReferenceType.DIRECT: r"^@memory:direct://([a-zA-Z0-9_-]+)(?:\?(.+))?$",
            ReferenceType.PATTERN: r"^@memory:pattern://([a-zA-Z0-9_/\-]+)(?:\?(.+))?$",
            ReferenceType.QUERY: r"^@memory:query://([^?]+)(?:\?(.+))?$",
            ReferenceType.WORKFLOW: r"^@memory:workflow://([a-zA-Z0-9_-]+)(?:\?(.+))?$",
            ReferenceType.CORRELATION: r"^@memory:correlation://([a-zA-Z0-9_-]+)(?:\?(.+))?$",
            ReferenceType.TEMPORAL: r"^@memory:temporal://([^?]+)(?:\?(.+))?$",
            ReferenceType.SEMANTIC: r"^@memory:semantic://([^?]+)(?:\?(.+))?$"
        }
    
    def parse_reference(self, reference_uri: str) -> Optional[MemoryReferenceSpec]:
        """Parse a memory reference URI into a specification."""
        if not reference_uri.startswith("@memory:"):
            return None
        
        # Try each reference type pattern
        for ref_type, pattern in self.reference_patterns.items():
            match = re.match(pattern, reference_uri)
            if match:
                return self._create_reference_spec(reference_uri, ref_type, match)
        
        return None
    
    def _create_reference_spec(self, uri: str, ref_type: ReferenceType, match: re.Match) -> MemoryReferenceSpec:
        """Create a reference specification from parsed components."""
        path_component = match.group(1)
        query_string = match.group(2) if len(match.groups()) > 1 else None
        
        # Parse path component based on reference type
        pattern_components = self._parse_path_component(ref_type, path_component)
        
        # Parse query parameters
        query_parameters = {}
        filters = {}
        metadata = {}
        scope = ReferenceScope.GLOBAL  # Default scope
        
        if query_string:
            params = parse_qs(query_string, keep_blank_values=True)
            for key, values in params.items():
                if key == "scope":
                    try:
                        scope = ReferenceScope(values[0])
                    except ValueError:
                        scope = ReferenceScope.GLOBAL
                elif key.startswith("filter_"):
                    filter_key = key[7:]  # Remove "filter_" prefix
                    filters[filter_key] = values[0] if len(values) == 1 else values
                elif key.startswith("meta_"):
                    meta_key = key[5:]  # Remove "meta_" prefix
                    metadata[meta_key] = values[0] if len(values) == 1 else values
                else:
                    query_parameters[key] = values
        
        # Handle expiry time if specified
        expiry_time = None
        if "expires" in query_parameters:
            try:
                expiry_seconds = float(query_parameters["expires"][0])
                expiry_time = time.time() + expiry_seconds
            except (ValueError, IndexError):
                pass
        
        return MemoryReferenceSpec(
            reference_uri=uri,
            reference_type=ref_type,
            scope=scope,
            pattern_components=pattern_components,
            query_parameters=query_parameters,
            filters=filters,
            metadata=metadata,
            expiry_time=expiry_time
        )
    
    def _parse_path_component(self, ref_type: ReferenceType, path: str) -> Dict[str, str]:
        """Parse path component based on reference type."""
        components = {}
        
        if ref_type == ReferenceType.DIRECT:
            components["memory_id"] = path
        
        elif ref_type == ReferenceType.PATTERN:
            path_parts = path.split("/")
            components["category"] = path_parts[0] if len(path_parts) > 0 else ""
            components["subcategory"] = path_parts[1] if len(path_parts) > 1 else ""
            components["specific"] = "/".join(path_parts[2:]) if len(path_parts) > 2 else ""
        
        elif ref_type == ReferenceType.QUERY:
            components["search_terms"] = path.replace("+", " ").replace("%20", " ")
        
        elif ref_type == ReferenceType.WORKFLOW:
            components["workflow_id"] = path
        
        elif ref_type == ReferenceType.CORRELATION:
            components["context_hash"] = path
        
        elif ref_type == ReferenceType.TEMPORAL:
            path_parts = path.split("/")
            components["timeframe"] = path_parts[0] if len(path_parts) > 0 else ""
            components["category"] = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
        
        elif ref_type == ReferenceType.SEMANTIC:
            path_parts = path.split("/")
            components["concept"] = path_parts[0] if len(path_parts) > 0 else ""
            components["relationship"] = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
        
        return components
    
    def validate_reference(self, reference_spec: MemoryReferenceSpec) -> Tuple[bool, Optional[str]]:
        """Validate a memory reference specification."""
        # Check if reference has expired
        if reference_spec.expiry_time and time.time() > reference_spec.expiry_time:
            return False, "Reference has expired"
        
        # Validate reference type specific requirements
        if reference_spec.reference_type == ReferenceType.DIRECT:
            if not reference_spec.pattern_components.get("memory_id"):
                return False, "Direct reference missing memory_id"
        
        elif reference_spec.reference_type == ReferenceType.PATTERN:
            if not reference_spec.pattern_components.get("category"):
                return False, "Pattern reference missing category"
        
        elif reference_spec.reference_type == ReferenceType.QUERY:
            if not reference_spec.pattern_components.get("search_terms"):
                return False, "Query reference missing search terms"
        
        # Validate scope
        if reference_spec.scope not in [s.value for s in ReferenceScope]:
            return False, f"Invalid scope: {reference_spec.scope}"
        
        return True, None


class MemoryReferenceResolver:
    """Resolver for memory references with pattern matching and caching."""
    
    def __init__(self, domain_manager, mcp_memory_retriever=None):
        """Initialize the resolver."""
        self.domain_manager = domain_manager
        self.mcp_memory_retriever = mcp_memory_retriever or get_mcp_memory_retriever(domain_manager)
        self.mcp_cache_adapter = get_mcp_cache_adapter()
        self.parser = MemoryReferenceParser()
        
        # Resolution settings
        self.resolution_settings = {
            "default_limit": 10,
            "min_similarity": 0.6,
            "cache_ttl": 1800.0,  # 30 minutes
            "max_correlation_results": 5,
            "semantic_expansion_enabled": True,
            "temporal_window_hours": 24
        }
        
        # Resolution statistics
        self.resolution_stats = {
            "total_resolutions": 0,
            "cache_hits": 0,
            "resolutions_by_type": {},
            "average_resolution_time": 0.0,
            "successful_resolutions": 0,
            "failed_resolutions": 0
        }
    
    async def resolve_reference(
        self,
        reference_uri: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> ResolvedReference:
        """Resolve a memory reference URI to actual memories."""
        start_time = time.time()
        self.resolution_stats["total_resolutions"] += 1
        
        try:
            # Parse the reference
            reference_spec = self.parser.parse_reference(reference_uri)
            if not reference_spec:
                return self._create_error_result(reference_uri, "Invalid reference URI format", start_time)
            
            # Validate the reference
            is_valid, error_message = self.parser.validate_reference(reference_spec)
            if not is_valid:
                return self._create_error_result(reference_uri, error_message, start_time)
            
            # Check cache first
            if use_cache:
                cached_result = await self._get_cached_resolution(reference_spec)
                if cached_result:
                    self.resolution_stats["cache_hits"] += 1
                    return cached_result
            
            # Resolve based on reference type
            resolved = await self._resolve_by_type(reference_spec, context or {})
            
            # Cache the result
            if use_cache and resolved.matched_memories:
                await self._cache_resolution(reference_spec, resolved)
            
            # Update statistics
            self._update_resolution_stats(reference_spec.reference_type, start_time, True)
            
            return resolved
            
        except Exception as e:
            logger.error(f"Failed to resolve reference {reference_uri}: {e}")
            self._update_resolution_stats(None, start_time, False)
            return self._create_error_result(reference_uri, str(e), start_time)
    
    async def _resolve_by_type(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve reference based on its type."""
        
        if reference_spec.reference_type == ReferenceType.DIRECT:
            return await self._resolve_direct_reference(reference_spec, context)
        
        elif reference_spec.reference_type == ReferenceType.PATTERN:
            return await self._resolve_pattern_reference(reference_spec, context)
        
        elif reference_spec.reference_type == ReferenceType.QUERY:
            return await self._resolve_query_reference(reference_spec, context)
        
        elif reference_spec.reference_type == ReferenceType.WORKFLOW:
            return await self._resolve_workflow_reference(reference_spec, context)
        
        elif reference_spec.reference_type == ReferenceType.CORRELATION:
            return await self._resolve_correlation_reference(reference_spec, context)
        
        elif reference_spec.reference_type == ReferenceType.TEMPORAL:
            return await self._resolve_temporal_reference(reference_spec, context)
        
        elif reference_spec.reference_type == ReferenceType.SEMANTIC:
            return await self._resolve_semantic_reference(reference_spec, context)
        
        else:
            raise ValueError(f"Unsupported reference type: {reference_spec.reference_type}")
    
    async def _resolve_direct_reference(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve a direct memory reference by ID."""
        memory_id = reference_spec.pattern_components["memory_id"]
        
        # Get memory directly
        memory = await self.domain_manager.persistence_domain.get_memory(memory_id)
        
        matched_memories = [memory] if memory else []
        confidence = 1.0 if memory else 0.0
        
        return ResolvedReference(
            reference_spec=reference_spec,
            matched_memories=matched_memories,
            resolution_confidence=confidence,
            resolution_time=time.time(),
            cache_hit=False,
            resolution_context={"method": "direct_lookup", "memory_id": memory_id}
        )
    
    async def _resolve_pattern_reference(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve a pattern-based memory reference."""
        category = reference_spec.pattern_components.get("category", "")
        subcategory = reference_spec.pattern_components.get("subcategory", "")
        specific = reference_spec.pattern_components.get("specific", "")
        
        # Build search query from pattern components
        query_parts = [category, subcategory, specific]
        query = " ".join(part for part in query_parts if part)
        
        # Determine memory types based on category
        memory_types = self._get_memory_types_for_category(category)
        
        # Apply scope filtering
        if reference_spec.scope == ReferenceScope.MCP_ENHANCED:
            mcp_context = MCPContext(
                project_context=context.get("project_context", {}),
                session_context=context.get("session_context", {}),
                user_intent=f"pattern_resolution_{category}"
            )
            
            results = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
                query, mcp_context, 
                limit=self.resolution_settings["default_limit"],
                strategy=RetrievalStrategy.ADAPTIVE_INTELLIGENT
            )
            
            # Convert RetrievalResult objects to dict format
            matched_memories = []
            for result in results:
                matched_memories.append({
                    "id": result.memory_id,
                    "type": result.memory_type,
                    "content": result.content,
                    "similarity": result.final_score,
                    "mcp_enhanced": True
                })
            
            confidence = sum(r.final_score for r in results) / len(results) if results else 0.0
            
        else:
            # Use standard retrieval
            matched_memories = await self.domain_manager.retrieve_memories(
                query,
                limit=self.resolution_settings["default_limit"],
                memory_types=memory_types,
                min_similarity=self.resolution_settings["min_similarity"]
            )
            
            confidence = sum(m.get("similarity", 0.0) for m in matched_memories) / len(matched_memories) if matched_memories else 0.0
        
        # Apply filters
        if reference_spec.filters:
            matched_memories = self._apply_filters(matched_memories, reference_spec.filters)
        
        return ResolvedReference(
            reference_spec=reference_spec,
            matched_memories=matched_memories,
            resolution_confidence=confidence,
            resolution_time=time.time(),
            cache_hit=False,
            resolution_context={
                "method": "pattern_matching",
                "query": query,
                "category": category,
                "memory_types": memory_types
            }
        )
    
    async def _resolve_query_reference(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve a query-based memory reference."""
        search_terms = reference_spec.pattern_components["search_terms"]
        
        # Determine retrieval strategy based on scope
        if reference_spec.scope == ReferenceScope.MCP_ENHANCED:
            mcp_context = MCPContext(
                project_context=context.get("project_context", {}),
                session_context=context.get("session_context", {}),
                user_intent="query_resolution"
            )
            
            results = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
                search_terms, mcp_context,
                limit=self.resolution_settings["default_limit"],
                strategy=RetrievalStrategy.ADAPTIVE_INTELLIGENT
            )
            
            # Convert to dict format
            matched_memories = []
            for result in results:
                matched_memories.append({
                    "id": result.memory_id,
                    "type": result.memory_type,
                    "content": result.content,
                    "similarity": result.final_score,
                    "mcp_enhanced": True
                })
            
            confidence = sum(r.final_score for r in results) / len(results) if results else 0.0
            
        else:
            # Standard semantic search
            matched_memories = await self.domain_manager.retrieve_memories(
                search_terms,
                limit=self.resolution_settings["default_limit"],
                min_similarity=self.resolution_settings["min_similarity"]
            )
            
            confidence = sum(m.get("similarity", 0.0) for m in matched_memories) / len(matched_memories) if matched_memories else 0.0
        
        # Apply filters
        if reference_spec.filters:
            matched_memories = self._apply_filters(matched_memories, reference_spec.filters)
        
        return ResolvedReference(
            reference_spec=reference_spec,
            matched_memories=matched_memories,
            resolution_confidence=confidence,
            resolution_time=time.time(),
            cache_hit=False,
            resolution_context={"method": "query_search", "search_terms": search_terms}
        )
    
    async def _resolve_workflow_reference(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve a workflow-specific memory reference."""
        workflow_id = reference_spec.pattern_components["workflow_id"]
        
        # Search for workflow patterns
        workflow_memories = await self.domain_manager.retrieve_memories(
            f"workflow {workflow_id}",
            memory_types=["mcp_workflow_pattern", "mcp_thinking_workflow"],
            limit=self.resolution_settings["default_limit"],
            min_similarity=0.5
        )
        
        confidence = sum(m.get("similarity", 0.0) for m in workflow_memories) / len(workflow_memories) if workflow_memories else 0.0
        
        return ResolvedReference(
            reference_spec=reference_spec,
            matched_memories=workflow_memories,
            resolution_confidence=confidence,
            resolution_time=time.time(),
            cache_hit=False,
            resolution_context={"method": "workflow_lookup", "workflow_id": workflow_id}
        )
    
    async def _resolve_correlation_reference(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve a correlation-based memory reference."""
        context_hash = reference_spec.pattern_components["context_hash"]
        
        # Use MCP correlation capabilities
        mcp_context = MCPContext(
            project_context=context.get("project_context", {}),
            session_context=context.get("session_context", {})
        )
        
        # Search for correlated memories
        correlation_memories = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            f"correlation {context_hash}",
            mcp_context,
            limit=self.resolution_settings["max_correlation_results"],
            strategy=RetrievalStrategy.CROSS_SYSTEM_CORRELATION
        )
        
        # Convert to dict format
        matched_memories = []
        for result in correlation_memories:
            matched_memories.append({
                "id": result.memory_id,
                "type": result.memory_type,
                "content": result.content,
                "similarity": result.final_score,
                "correlation_confidence": result.cross_system_confidence
            })
        
        confidence = sum(r.cross_system_confidence for r in correlation_memories) / len(correlation_memories) if correlation_memories else 0.0
        
        return ResolvedReference(
            reference_spec=reference_spec,
            matched_memories=matched_memories,
            resolution_confidence=confidence,
            resolution_time=time.time(),
            cache_hit=False,
            resolution_context={"method": "correlation_analysis", "context_hash": context_hash}
        )
    
    async def _resolve_temporal_reference(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve a temporal-based memory reference."""
        timeframe = reference_spec.pattern_components.get("timeframe", "")
        category = reference_spec.pattern_components.get("category", "")
        
        # Parse timeframe (e.g., "last_week", "2023-12", "recent")
        time_filter = self._parse_timeframe(timeframe)
        
        # Build query with temporal component
        query = f"temporal {category}" if category else "temporal"
        
        # Get memories with temporal filtering
        memories = await self.domain_manager.retrieve_memories(
            query,
            limit=self.resolution_settings["default_limit"],
            min_similarity=0.5
        )
        
        # Apply temporal filtering (would need temporal domain integration)
        # For now, return all memories
        matched_memories = memories
        confidence = 0.7  # Default confidence for temporal queries
        
        return ResolvedReference(
            reference_spec=reference_spec,
            matched_memories=matched_memories,
            resolution_confidence=confidence,
            resolution_time=time.time(),
            cache_hit=False,
            resolution_context={"method": "temporal_search", "timeframe": timeframe, "category": category}
        )
    
    async def _resolve_semantic_reference(
        self,
        reference_spec: MemoryReferenceSpec,
        context: Dict[str, Any]
    ) -> ResolvedReference:
        """Resolve a semantic-based memory reference."""
        concept = reference_spec.pattern_components.get("concept", "")
        relationship = reference_spec.pattern_components.get("relationship", "")
        
        # Build semantic query
        query_parts = [concept, relationship] if relationship else [concept]
        query = " ".join(query_parts)
        
        # Use semantic search with MCP enhancement
        mcp_context = MCPContext(
            project_context=context.get("project_context", {}),
            session_context=context.get("session_context", {}),
            user_intent="semantic_resolution"
        )
        
        results = await self.mcp_memory_retriever.retrieve_memories_with_mcp_context(
            query, mcp_context,
            limit=self.resolution_settings["default_limit"],
            strategy=RetrievalStrategy.ADAPTIVE_INTELLIGENT
        )
        
        # Convert to dict format
        matched_memories = []
        for result in results:
            matched_memories.append({
                "id": result.memory_id,
                "type": result.memory_type,
                "content": result.content,
                "similarity": result.final_score,
                "semantic_relevance": True
            })
        
        confidence = sum(r.final_score for r in results) / len(results) if results else 0.0
        
        return ResolvedReference(
            reference_spec=reference_spec,
            matched_memories=matched_memories,
            resolution_confidence=confidence,
            resolution_time=time.time(),
            cache_hit=False,
            resolution_context={"method": "semantic_analysis", "concept": concept, "relationship": relationship}
        )
    
    def _get_memory_types_for_category(self, category: str) -> Optional[List[str]]:
        """Get appropriate memory types for a category."""
        category_mapping = {
            "workflow": ["mcp_workflow_pattern", "mcp_thinking_workflow"],
            "thinking": ["structured_thinking", "thought_process", "mcp_thinking_workflow"],
            "tool": ["mcp_tool_correlation"],
            "resource": ["mcp_resource_pattern"],
            "context": ["enhanced_context"],
            "correlation": ["mcp_tool_correlation", "thinking_mcp_integration"],
            "pattern": ["mcp_workflow_pattern", "mcp_resource_pattern"]
        }
        
        return category_mapping.get(category.lower())
    
    def _parse_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """Parse timeframe string into time filter parameters."""
        # Simplified timeframe parsing
        timeframe_mapping = {
            "recent": {"hours": 24},
            "today": {"hours": 24},
            "yesterday": {"hours": 48, "offset": 24},
            "last_week": {"days": 7},
            "last_month": {"days": 30}
        }
        
        return timeframe_mapping.get(timeframe.lower(), {"days": 1})
    
    def _apply_filters(self, memories: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to memory results."""
        filtered_memories = []
        
        for memory in memories:
            include_memory = True
            
            for filter_key, filter_value in filters.items():
                if filter_key == "type" and memory.get("type") != filter_value:
                    include_memory = False
                    break
                elif filter_key == "min_similarity":
                    try:
                        min_sim = float(filter_value)
                        if memory.get("similarity", 0.0) < min_sim:
                            include_memory = False
                            break
                    except ValueError:
                        pass
                elif filter_key == "importance":
                    try:
                        min_importance = float(filter_value)
                        if memory.get("importance", 0.0) < min_importance:
                            include_memory = False
                            break
                    except ValueError:
                        pass
            
            if include_memory:
                filtered_memories.append(memory)
        
        return filtered_memories
    
    async def _get_cached_resolution(self, reference_spec: MemoryReferenceSpec) -> Optional[ResolvedReference]:
        """Get cached resolution result."""
        try:
            cache_key = self._generate_cache_key(reference_spec)
            cached_data = await cache_get(cache_key, CacheType.MEMORY_PATTERN)
            
            if cached_data:
                # Reconstruct ResolvedReference from cached data
                return ResolvedReference(
                    reference_spec=reference_spec,
                    matched_memories=cached_data["matched_memories"],
                    resolution_confidence=cached_data["resolution_confidence"],
                    resolution_time=cached_data["resolution_time"],
                    cache_hit=True,
                    resolution_context=cached_data["resolution_context"]
                )
        except Exception as e:
            logger.debug(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_resolution(self, reference_spec: MemoryReferenceSpec, result: ResolvedReference) -> None:
        """Cache resolution result."""
        try:
            cache_key = self._generate_cache_key(reference_spec)
            cache_data = {
                "matched_memories": result.matched_memories,
                "resolution_confidence": result.resolution_confidence,
                "resolution_time": result.resolution_time,
                "resolution_context": result.resolution_context,
                "cached_at": time.time()
            }
            
            await cache_put(
                cache_key,
                cache_data,
                CacheType.MEMORY_PATTERN,
                ttl=self.resolution_settings["cache_ttl"],
                metadata={
                    "reference_type": reference_spec.reference_type.value,
                    "scope": reference_spec.scope.value,
                    "result_count": len(result.matched_memories)
                }
            )
        except Exception as e:
            logger.debug(f"Cache storage failed: {e}")
    
    def _generate_cache_key(self, reference_spec: MemoryReferenceSpec) -> str:
        """Generate cache key for reference resolution."""
        key_components = [
            reference_spec.reference_uri,
            reference_spec.scope.value,
            json.dumps(reference_spec.filters, sort_keys=True),
            json.dumps(reference_spec.query_parameters, sort_keys=True)
        ]
        key_string = "|".join(key_components)
        return f"memory_ref_{hashlib.md5(key_string.encode()).hexdigest()[:16]}"
    
    def _create_error_result(self, reference_uri: str, error_message: str, start_time: float) -> ResolvedReference:
        """Create an error result for failed resolution."""
        return ResolvedReference(
            reference_spec=MemoryReferenceSpec(
                reference_uri=reference_uri,
                reference_type=ReferenceType.DIRECT,  # Default
                scope=ReferenceScope.GLOBAL
            ),
            matched_memories=[],
            resolution_confidence=0.0,
            resolution_time=time.time() - start_time,
            cache_hit=False,
            resolution_context={"error": True},
            error_message=error_message
        )
    
    def _update_resolution_stats(self, reference_type: Optional[ReferenceType], start_time: float, success: bool) -> None:
        """Update resolution statistics."""
        resolution_time = time.time() - start_time
        
        if success:
            self.resolution_stats["successful_resolutions"] += 1
        else:
            self.resolution_stats["failed_resolutions"] += 1
        
        if reference_type:
            type_key = reference_type.value
            self.resolution_stats["resolutions_by_type"][type_key] = \
                self.resolution_stats["resolutions_by_type"].get(type_key, 0) + 1
        
        # Update average resolution time
        total_resolutions = self.resolution_stats["total_resolutions"]
        current_avg = self.resolution_stats["average_resolution_time"]
        new_avg = ((current_avg * (total_resolutions - 1)) + resolution_time) / total_resolutions
        self.resolution_stats["average_resolution_time"] = new_avg
    
    async def get_resolution_analytics(self) -> Dict[str, Any]:
        """Get memory reference resolution analytics."""
        return {
            "performance_metrics": self.resolution_stats.copy(),
            "resolution_settings": self.resolution_settings.copy(),
            "supported_reference_types": [rt.value for rt in ReferenceType],
            "supported_scopes": [sc.value for sc in ReferenceScope],
            "cache_effectiveness": {
                "hit_rate": self.resolution_stats["cache_hits"] / max(self.resolution_stats["total_resolutions"], 1),
                "total_resolutions": self.resolution_stats["total_resolutions"],
                "cache_hits": self.resolution_stats["cache_hits"]
            }
        }


class MemoryReferenceManager:
    """High-level manager for memory reference system."""
    
    def __init__(self, domain_manager, mcp_memory_retriever=None):
        """Initialize memory reference manager."""
        self.domain_manager = domain_manager
        self.resolver = MemoryReferenceResolver(domain_manager, mcp_memory_retriever)
        self.parser = MemoryReferenceParser()
        
        # Reference registry for tracking active references
        self.reference_registry: Dict[str, MemoryReferenceSpec] = {}
    
    async def resolve(
        self,
        reference_uri: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ResolvedReference:
        """Resolve a memory reference URI."""
        return await self.resolver.resolve_reference(reference_uri, context)
    
    def parse(self, reference_uri: str) -> Optional[MemoryReferenceSpec]:
        """Parse a memory reference URI."""
        return self.parser.parse_reference(reference_uri)
    
    def validate(self, reference_uri: str) -> Tuple[bool, Optional[str]]:
        """Validate a memory reference URI."""
        spec = self.parse(reference_uri)
        if not spec:
            return False, "Invalid reference URI format"
        
        return self.parser.validate_reference(spec)
    
    async def create_pattern_reference(
        self,
        category: str,
        subcategory: str = "",
        specific: str = "",
        scope: ReferenceScope = ReferenceScope.GLOBAL,
        filters: Optional[Dict[str, Any]] = None,
        expires_in: Optional[float] = None
    ) -> str:
        """Create a pattern-based memory reference URI."""
        path_parts = [category, subcategory, specific]
        path = "/".join(part for part in path_parts if part)
        
        query_params = []
        if scope != ReferenceScope.GLOBAL:
            query_params.append(f"scope={scope.value}")
        
        if filters:
            for key, value in filters.items():
                query_params.append(f"filter_{key}={value}")
        
        if expires_in:
            query_params.append(f"expires={expires_in}")
        
        query_string = "&".join(query_params) if query_params else ""
        reference_uri = f"@memory:pattern://{path}"
        
        if query_string:
            reference_uri += f"?{query_string}"
        
        return reference_uri
    
    async def create_query_reference(
        self,
        search_terms: str,
        scope: ReferenceScope = ReferenceScope.GLOBAL,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a query-based memory reference URI."""
        encoded_terms = search_terms.replace(" ", "+")
        
        query_params = []
        if scope != ReferenceScope.GLOBAL:
            query_params.append(f"scope={scope.value}")
        
        if filters:
            for key, value in filters.items():
                query_params.append(f"filter_{key}={value}")
        
        query_string = "&".join(query_params) if query_params else ""
        reference_uri = f"@memory:query://{encoded_terms}"
        
        if query_string:
            reference_uri += f"?{query_string}"
        
        return reference_uri
    
    def register_reference(self, reference_uri: str, alias: str = None) -> bool:
        """Register a reference for easy lookup."""
        spec = self.parse(reference_uri)
        if not spec:
            return False
        
        key = alias or reference_uri
        self.reference_registry[key] = spec
        return True
    
    def get_registered_reference(self, key: str) -> Optional[MemoryReferenceSpec]:
        """Get a registered reference by key or alias."""
        return self.reference_registry.get(key)
    
    def list_registered_references(self) -> Dict[str, str]:
        """List all registered references."""
        return {key: spec.reference_uri for key, spec in self.reference_registry.items()}
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the reference system."""
        resolver_analytics = await self.resolver.get_resolution_analytics()
        
        return {
            "resolver_analytics": resolver_analytics,
            "registered_references": len(self.reference_registry),
            "reference_types_in_use": list(set(spec.reference_type.value for spec in self.reference_registry.values())),
            "scopes_in_use": list(set(spec.scope.value for spec in self.reference_registry.values()))
        }


# Global memory reference manager instance
_memory_reference_manager = None


def get_memory_reference_manager(domain_manager, mcp_memory_retriever=None) -> MemoryReferenceManager:
    """Get or create global memory reference manager."""
    global _memory_reference_manager
    if _memory_reference_manager is None:
        _memory_reference_manager = MemoryReferenceManager(domain_manager, mcp_memory_retriever)
    return _memory_reference_manager


# Convenience functions
async def resolve_memory_reference(
    domain_manager,
    reference_uri: str,
    context: Optional[Dict[str, Any]] = None
) -> ResolvedReference:
    """Convenience function to resolve a memory reference."""
    manager = get_memory_reference_manager(domain_manager)
    return await manager.resolve(reference_uri, context)