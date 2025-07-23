"""
MCP-Enhanced Memory Retrieval System.

This module provides context-aware memory retrieval capabilities that leverage
MCP workflow patterns and cross-system intelligence for superior memory search
and relevance scoring.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

from ..mcp.cache_integration import get_mcp_cache_adapter
from ..core.unified_cache import cache_get, cache_put, CacheType, cache_get_mcp_enhanced
from ..utils.schema import get_mcp_memory_types, is_mcp_integrated_type

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Memory retrieval strategies."""
    SEMANTIC_ONLY = "semantic_only"
    MCP_ENHANCED = "mcp_enhanced"
    WORKFLOW_PATTERN = "workflow_pattern"
    CROSS_SYSTEM_CORRELATION = "cross_system_correlation"
    ADAPTIVE_INTELLIGENT = "adaptive_intelligent"


@dataclass
class MCPContext:
    """MCP context for memory retrieval."""
    current_tools: List[str] = None
    thinking_stage: str = None
    workflow_pattern: str = None
    project_context: Dict[str, Any] = None
    session_context: Dict[str, Any] = None
    user_intent: str = None
    performance_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.current_tools is None:
            self.current_tools = []
        if self.project_context is None:
            self.project_context = {}
        if self.session_context is None:
            self.session_context = {}
        if self.performance_requirements is None:
            self.performance_requirements = {}


@dataclass
class RetrievalResult:
    """Enhanced memory retrieval result with MCP context."""
    memory_id: str
    memory_type: str
    content: Dict[str, Any]
    base_similarity: float
    mcp_relevance_boost: float
    final_score: float
    retrieval_context: Dict[str, Any]
    workflow_matches: List[str]
    tool_correlations: List[Dict[str, Any]]
    cross_system_confidence: float


class MCPMemoryRetriever:
    """Enhanced memory retrieval with MCP pattern matching and context awareness."""
    
    def __init__(self, domain_manager, unified_cache_manager=None):
        """Initialize MCP memory retriever."""
        self.domain_manager = domain_manager
        self.unified_cache = unified_cache_manager
        self.mcp_cache_adapter = get_mcp_cache_adapter()
        self.mcp_memory_types = get_mcp_memory_types()
        
        # Retrieval optimization settings
        self.relevance_weights = {
            "semantic_similarity": 0.4,
            "workflow_pattern_match": 0.25,
            "tool_correlation": 0.15,
            "temporal_relevance": 0.1,
            "usage_frequency": 0.1
        }
        
        # Performance metrics
        self.retrieval_stats = {
            "total_retrievals": 0,
            "mcp_enhanced_retrievals": 0,
            "cache_hits": 0,
            "average_response_time": 0.0,
            "cross_system_correlations_found": 0
        }
    
    async def retrieve_memories_with_mcp_context(
        self,
        query: str,
        mcp_context: MCPContext,
        limit: int = 10,
        strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE_INTELLIGENT,
        min_similarity: float = 0.6,
        include_metadata: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve memories with MCP context awareness and intelligent ranking.
        
        Args:
            query: Search query
            mcp_context: MCP context information
            limit: Maximum number of results
            strategy: Retrieval strategy to use
            min_similarity: Minimum similarity threshold
            include_metadata: Include metadata in results
            
        Returns:
            List of enhanced retrieval results
        """
        start_time = time.time()
        self.retrieval_stats["total_retrievals"] += 1
        
        try:
            # Generate cache key for this specific retrieval
            cache_key = self._generate_retrieval_cache_key(query, mcp_context, strategy)
            
            # Check cache first
            cached_results = await self._get_cached_retrieval_results(cache_key, mcp_context)
            if cached_results:
                logger.debug(f"Cache hit for retrieval: {cache_key}")
                self.retrieval_stats["cache_hits"] += 1
                return cached_results[:limit]
            
            # Perform MCP-enhanced retrieval
            if strategy == RetrievalStrategy.ADAPTIVE_INTELLIGENT:
                results = await self._adaptive_intelligent_retrieval(query, mcp_context, limit, min_similarity)
            elif strategy == RetrievalStrategy.MCP_ENHANCED:
                results = await self._mcp_enhanced_retrieval(query, mcp_context, limit, min_similarity)
            elif strategy == RetrievalStrategy.WORKFLOW_PATTERN:
                results = await self._workflow_pattern_retrieval(query, mcp_context, limit, min_similarity)
            elif strategy == RetrievalStrategy.CROSS_SYSTEM_CORRELATION:
                results = await self._cross_system_correlation_retrieval(query, mcp_context, limit, min_similarity)
            else:
                # Fallback to semantic only
                results = await self._semantic_only_retrieval(query, mcp_context, limit, min_similarity)
            
            # Cache the results
            await self._cache_retrieval_results(cache_key, results, mcp_context)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, len(results))
            
            self.retrieval_stats["mcp_enhanced_retrievals"] += 1
            logger.info(f"MCP-enhanced retrieval completed: {len(results)} results in {response_time:.3f}s")
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"MCP memory retrieval failed: {e}")
            # Fallback to basic retrieval
            return await self._fallback_retrieval(query, limit, min_similarity)
    
    async def _adaptive_intelligent_retrieval(
        self,
        query: str,
        mcp_context: MCPContext,
        limit: int,
        min_similarity: float
    ) -> List[RetrievalResult]:
        """Adaptive intelligent retrieval that combines multiple strategies."""
        
        # Step 1: Get base semantic results
        semantic_results = await self._get_base_semantic_results(query, limit * 2, min_similarity)
        
        # Step 2: Enhance results with MCP context
        enhanced_results = []
        
        for memory in semantic_results:
            # Calculate MCP relevance boost
            mcp_boost = await self._calculate_mcp_relevance_boost(memory, mcp_context)
            
            # Find workflow pattern matches
            workflow_matches = await self._find_workflow_pattern_matches(memory, mcp_context)
            
            # Calculate tool correlations
            tool_correlations = await self._calculate_tool_correlations(memory, mcp_context)
            
            # Calculate cross-system confidence
            cross_system_confidence = await self._calculate_cross_system_confidence(memory, mcp_context)
            
            # Calculate final score
            base_similarity = memory.get("similarity", 0.0)
            final_score = self._calculate_final_relevance_score(
                base_similarity, mcp_boost, workflow_matches, tool_correlations, cross_system_confidence
            )
            
            # Create enhanced result
            if final_score >= min_similarity:
                result = RetrievalResult(
                    memory_id=memory.get("id", memory.get("memory_id")),
                    memory_type=memory.get("type", memory.get("memory_type")),
                    content=memory.get("content", {}),
                    base_similarity=base_similarity,
                    mcp_relevance_boost=mcp_boost,
                    final_score=final_score,
                    retrieval_context={
                        "strategy": "adaptive_intelligent",
                        "mcp_context_used": True,
                        "enhancement_factors": {
                            "workflow_patterns": len(workflow_matches),
                            "tool_correlations": len(tool_correlations),
                            "cross_system_confidence": cross_system_confidence
                        }
                    },
                    workflow_matches=workflow_matches,
                    tool_correlations=tool_correlations,
                    cross_system_confidence=cross_system_confidence
                )
                enhanced_results.append(result)
        
        # Step 3: Sort by final score and return top results
        enhanced_results.sort(key=lambda r: r.final_score, reverse=True)
        
        logger.debug(f"Adaptive retrieval found {len(enhanced_results)} enhanced results")
        return enhanced_results
    
    async def _mcp_enhanced_retrieval(
        self,
        query: str,
        mcp_context: MCPContext,
        limit: int,
        min_similarity: float
    ) -> List[RetrievalResult]:
        """MCP-enhanced retrieval focusing on workflow patterns."""
        
        # Get similar workflow patterns first
        similar_patterns = await self.mcp_cache_adapter.get_similar_workflow_patterns(
            query, {
                "thinking_stage": mcp_context.thinking_stage,
                "tools_used": mcp_context.current_tools,
                "project_context": mcp_context.project_context
            }
        )
        
        # Expand search with workflow context
        expanded_query = self._expand_query_with_workflow_patterns(query, similar_patterns)
        
        # Get base results with expanded query
        base_results = await self._get_base_semantic_results(expanded_query, limit * 3, min_similarity * 0.8)
        
        # Filter and enhance with MCP context
        enhanced_results = []
        for memory in base_results:
            if is_mcp_integrated_type(memory.get("type", "")):
                mcp_boost = await self._calculate_mcp_relevance_boost(memory, mcp_context)
                workflow_matches = await self._find_workflow_pattern_matches(memory, mcp_context)
                
                if mcp_boost > 0.1 or workflow_matches:  # Only include if MCP-relevant
                    result = await self._create_enhanced_result(memory, mcp_context, "mcp_enhanced")
                    enhanced_results.append(result)
        
        enhanced_results.sort(key=lambda r: r.final_score, reverse=True)
        return enhanced_results
    
    async def _workflow_pattern_retrieval(
        self,
        query: str,
        mcp_context: MCPContext,
        limit: int,
        min_similarity: float
    ) -> List[RetrievalResult]:
        """Retrieval focused specifically on workflow patterns."""
        
        # Search for workflow pattern memories specifically
        workflow_memories = await self.domain_manager.retrieve_memories(
            query,
            limit=limit * 2,
            memory_types=["mcp_workflow_pattern", "mcp_thinking_workflow"],
            min_similarity=min_similarity * 0.7
        )
        
        enhanced_results = []
        for memory in workflow_memories:
            result = await self._create_enhanced_result(memory, mcp_context, "workflow_pattern")
            enhanced_results.append(result)
        
        enhanced_results.sort(key=lambda r: r.final_score, reverse=True)
        return enhanced_results
    
    async def _cross_system_correlation_retrieval(
        self,
        query: str,
        mcp_context: MCPContext,
        limit: int,
        min_similarity: float
    ) -> List[RetrievalResult]:
        """Retrieval using cross-system correlations."""
        
        # Find correlated tools and patterns
        tool_correlations = await self._find_tool_correlations_for_query(query, mcp_context)
        
        # Expand search based on correlations
        correlation_queries = self._generate_correlation_queries(query, tool_correlations)
        
        all_results = []
        for correlation_query in correlation_queries:
            results = await self._get_base_semantic_results(correlation_query, limit, min_similarity * 0.8)
            all_results.extend(results)
        
        # Remove duplicates and enhance
        unique_results = self._deduplicate_results(all_results)
        enhanced_results = []
        
        for memory in unique_results:
            result = await self._create_enhanced_result(memory, mcp_context, "cross_system_correlation")
            enhanced_results.append(result)
        
        enhanced_results.sort(key=lambda r: r.final_score, reverse=True)
        self.retrieval_stats["cross_system_correlations_found"] += len(enhanced_results)
        
        return enhanced_results
    
    async def _semantic_only_retrieval(
        self,
        query: str,
        mcp_context: MCPContext,
        limit: int,
        min_similarity: float
    ) -> List[RetrievalResult]:
        """Standard semantic retrieval without MCP enhancement."""
        
        memories = await self.domain_manager.retrieve_memories(
            query, limit=limit, min_similarity=min_similarity
        )
        
        results = []
        for memory in memories:
            result = RetrievalResult(
                memory_id=memory.get("id", memory.get("memory_id")),
                memory_type=memory.get("type", memory.get("memory_type")),
                content=memory.get("content", {}),
                base_similarity=memory.get("similarity", 0.0),
                mcp_relevance_boost=0.0,
                final_score=memory.get("similarity", 0.0),
                retrieval_context={"strategy": "semantic_only", "mcp_context_used": False},
                workflow_matches=[],
                tool_correlations=[],
                cross_system_confidence=0.0
            )
            results.append(result)
        
        return results
    
    async def _get_base_semantic_results(self, query: str, limit: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Get base semantic search results."""
        return await self.domain_manager.retrieve_memories(
            query, limit=limit, min_similarity=min_similarity, include_metadata=True
        )
    
    async def _calculate_mcp_relevance_boost(self, memory: Dict[str, Any], mcp_context: MCPContext) -> float:
        """Calculate MCP relevance boost for a memory."""
        boost = 0.0
        
        memory_type = memory.get("type", "")
        content = memory.get("content", {})
        
        # Boost for MCP-integrated memory types
        if is_mcp_integrated_type(memory_type):
            boost += 0.2
        
        # Boost for tool context matches
        if mcp_context.current_tools:
            memory_tools = content.get("tool_context", {}).get("tools_used", [])
            tool_overlap = len(set(mcp_context.current_tools) & set(memory_tools))
            if tool_overlap > 0:
                boost += tool_overlap * 0.1
        
        # Boost for thinking stage matches
        if mcp_context.thinking_stage:
            memory_stage = content.get("thinking_stage", "")
            if mcp_context.thinking_stage == memory_stage:
                boost += 0.15
        
        # Boost for workflow pattern matches
        if mcp_context.workflow_pattern:
            memory_workflow = content.get("workflow_pattern", {}).get("trigger_context", "")
            if mcp_context.workflow_pattern in memory_workflow:
                boost += 0.2
        
        return min(boost, 0.5)  # Cap boost at 0.5
    
    async def _find_workflow_pattern_matches(self, memory: Dict[str, Any], mcp_context: MCPContext) -> List[str]:
        """Find workflow pattern matches for a memory."""
        matches = []
        
        content = memory.get("content", {})
        
        # Check for direct workflow pattern references
        if "workflow_pattern" in content:
            pattern = content["workflow_pattern"]
            pattern_id = pattern.get("pattern_id", pattern.get("trigger_context", ""))
            if pattern_id:
                matches.append(pattern_id)
        
        # Check for tool sequence matches
        if mcp_context.current_tools and "tool_sequence" in content:
            memory_tools = content["tool_sequence"]
            if any(tool in memory_tools for tool in mcp_context.current_tools):
                matches.append("tool_sequence_match")
        
        return matches
    
    async def _calculate_tool_correlations(self, memory: Dict[str, Any], mcp_context: MCPContext) -> List[Dict[str, Any]]:
        """Calculate tool correlations for a memory."""
        correlations = []
        
        if not mcp_context.current_tools:
            return correlations
        
        content = memory.get("content", {})
        
        # Check for tool correlation data in memory
        if memory.get("type") == "mcp_tool_correlation":
            tool_name = content.get("tool_name", "")
            if tool_name in mcp_context.current_tools:
                correlation_patterns = content.get("correlation_patterns", [])
                for pattern in correlation_patterns:
                    correlations.append({
                        "tool": tool_name,
                        "pattern": pattern.get("pattern", ""),
                        "strength": pattern.get("strength", 0.0)
                    })
        
        return correlations
    
    async def _calculate_cross_system_confidence(self, memory: Dict[str, Any], mcp_context: MCPContext) -> float:
        """Calculate cross-system confidence score."""
        confidence = 0.0
        
        # Base confidence from memory type
        if is_mcp_integrated_type(memory.get("type", "")):
            confidence += 0.3
        
        # Confidence from MCP context fields
        mcp_context_data = memory.get("mcp_context", {})
        if mcp_context_data:
            confidence += 0.2
            
            # Boost for cross-system processing
            if mcp_context_data.get("mcp_enhanced", False):
                confidence += 0.2
            
            # Boost for correlation data
            if "correlation_data" in memory.get("content", {}):
                correlation_strength = memory["content"]["correlation_data"].get("correlation_strength", 0.0)
                confidence += correlation_strength * 0.3
        
        return min(confidence, 1.0)
    
    def _calculate_final_relevance_score(
        self,
        base_similarity: float,
        mcp_boost: float,
        workflow_matches: List[str],
        tool_correlations: List[Dict[str, Any]],
        cross_system_confidence: float
    ) -> float:
        """Calculate final relevance score using weighted factors."""
        
        # Apply weights
        semantic_score = base_similarity * self.relevance_weights["semantic_similarity"]
        workflow_score = (len(workflow_matches) * 0.2) * self.relevance_weights["workflow_pattern_match"]
        tool_score = (len(tool_correlations) * 0.15) * self.relevance_weights["tool_correlation"]
        cross_system_score = cross_system_confidence * 0.1
        
        # Combine scores
        final_score = semantic_score + workflow_score + tool_score + cross_system_score + mcp_boost
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    async def _create_enhanced_result(
        self,
        memory: Dict[str, Any],
        mcp_context: MCPContext,
        strategy: str
    ) -> RetrievalResult:
        """Create an enhanced retrieval result."""
        
        mcp_boost = await self._calculate_mcp_relevance_boost(memory, mcp_context)
        workflow_matches = await self._find_workflow_pattern_matches(memory, mcp_context)
        tool_correlations = await self._calculate_tool_correlations(memory, mcp_context)
        cross_system_confidence = await self._calculate_cross_system_confidence(memory, mcp_context)
        
        base_similarity = memory.get("similarity", 0.0)
        final_score = self._calculate_final_relevance_score(
            base_similarity, mcp_boost, workflow_matches, tool_correlations, cross_system_confidence
        )
        
        return RetrievalResult(
            memory_id=memory.get("id", memory.get("memory_id")),
            memory_type=memory.get("type", memory.get("memory_type")),
            content=memory.get("content", {}),
            base_similarity=base_similarity,
            mcp_relevance_boost=mcp_boost,
            final_score=final_score,
            retrieval_context={
                "strategy": strategy,
                "mcp_context_used": True,
                "enhancement_factors": {
                    "workflow_patterns": len(workflow_matches),
                    "tool_correlations": len(tool_correlations),
                    "cross_system_confidence": cross_system_confidence
                }
            },
            workflow_matches=workflow_matches,
            tool_correlations=tool_correlations,
            cross_system_confidence=cross_system_confidence
        )
    
    def _generate_retrieval_cache_key(self, query: str, mcp_context: MCPContext, strategy: RetrievalStrategy) -> str:
        """Generate cache key for retrieval results."""
        key_components = [
            query,
            str(mcp_context.current_tools),
            mcp_context.thinking_stage or "",
            mcp_context.workflow_pattern or "",
            strategy.value
        ]
        key_string = "|".join(key_components)
        return f"mcp_retrieval_{hashlib.md5(key_string.encode()).hexdigest()[:12]}"
    
    async def _get_cached_retrieval_results(self, cache_key: str, mcp_context: MCPContext) -> Optional[List[RetrievalResult]]:
        """Get cached retrieval results."""
        try:
            cached_data = await cache_get(cache_key, CacheType.MCP_WORKFLOW, context=mcp_context.__dict__)
            if cached_data and isinstance(cached_data, list):
                # Convert dict data back to RetrievalResult objects
                results = []
                for item in cached_data:
                    if isinstance(item, dict):
                        result = RetrievalResult(**item)
                        results.append(result)
                return results
        except Exception as e:
            logger.debug(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_retrieval_results(self, cache_key: str, results: List[RetrievalResult], mcp_context: MCPContext) -> None:
        """Cache retrieval results."""
        try:
            # Convert RetrievalResult objects to dicts for caching
            cache_data = [result.__dict__ for result in results]
            await cache_put(
                cache_key,
                cache_data,
                CacheType.MCP_WORKFLOW,
                ttl=900.0,  # 15 minutes
                metadata={
                    "result_count": len(results),
                    "mcp_context": mcp_context.__dict__,
                    "cached_at": time.time()
                }
            )
        except Exception as e:
            logger.debug(f"Cache storage failed: {e}")
    
    async def _fallback_retrieval(self, query: str, limit: int, min_similarity: float) -> List[RetrievalResult]:
        """Fallback retrieval method."""
        try:
            memories = await self.domain_manager.retrieve_memories(query, limit=limit, min_similarity=min_similarity)
            results = []
            for memory in memories:
                result = RetrievalResult(
                    memory_id=memory.get("id", memory.get("memory_id")),
                    memory_type=memory.get("type", memory.get("memory_type")),
                    content=memory.get("content", {}),
                    base_similarity=memory.get("similarity", 0.0),
                    mcp_relevance_boost=0.0,
                    final_score=memory.get("similarity", 0.0),
                    retrieval_context={"strategy": "fallback", "mcp_context_used": False},
                    workflow_matches=[],
                    tool_correlations=[],
                    cross_system_confidence=0.0
                )
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return []
    
    def _expand_query_with_workflow_patterns(self, query: str, patterns: List[Dict[str, Any]]) -> str:
        """Expand query with workflow pattern context."""
        if not patterns:
            return query
        
        pattern_terms = []
        for pattern in patterns:
            if isinstance(pattern, dict):
                trigger_context = pattern.get("trigger_context", "")
                if trigger_context:
                    pattern_terms.append(trigger_context)
        
        if pattern_terms:
            expanded_query = f"{query} {' '.join(pattern_terms[:3])}"  # Limit to 3 patterns
            return expanded_query
        
        return query
    
    async def _find_tool_correlations_for_query(self, query: str, mcp_context: MCPContext) -> List[Dict[str, Any]]:
        """Find tool correlations relevant to the query."""
        correlations = []
        
        # Search for tool correlation memories
        correlation_memories = await self.domain_manager.retrieve_memories(
            query,
            memory_types=["mcp_tool_correlation"],
            limit=10,
            min_similarity=0.5
        )
        
        for memory in correlation_memories:
            content = memory.get("content", {})
            tool_name = content.get("tool_name", "")
            correlation_patterns = content.get("correlation_patterns", [])
            
            for pattern in correlation_patterns:
                correlations.append({
                    "tool": tool_name,
                    "pattern": pattern.get("pattern", ""),
                    "strength": pattern.get("strength", 0.0),
                    "query_relevance": memory.get("similarity", 0.0)
                })
        
        return correlations
    
    def _generate_correlation_queries(self, base_query: str, correlations: List[Dict[str, Any]]) -> List[str]:
        """Generate additional queries based on tool correlations."""
        queries = [base_query]
        
        # Add queries based on strong correlations
        strong_correlations = [c for c in correlations if c.get("strength", 0) > 0.7]
        
        for correlation in strong_correlations[:3]:  # Limit to top 3
            pattern = correlation.get("pattern", "")
            if pattern and pattern not in base_query:
                expanded_query = f"{base_query} {pattern}"
                queries.append(expanded_query)
        
        return queries
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on memory ID."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            memory_id = result.get("id", result.get("memory_id"))
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_results.append(result)
        
        return unique_results
    
    def _update_performance_metrics(self, response_time: float, result_count: int):
        """Update performance metrics."""
        # Update average response time
        current_avg = self.retrieval_stats["average_response_time"]
        total_retrievals = self.retrieval_stats["total_retrievals"]
        
        if total_retrievals > 1:
            new_avg = ((current_avg * (total_retrievals - 1)) + response_time) / total_retrievals
            self.retrieval_stats["average_response_time"] = new_avg
        else:
            self.retrieval_stats["average_response_time"] = response_time
    
    async def get_retrieval_analytics(self) -> Dict[str, Any]:
        """Get retrieval analytics and performance metrics."""
        return {
            "performance_metrics": self.retrieval_stats.copy(),
            "relevance_weights": self.relevance_weights.copy(),
            "cache_performance": await self.mcp_cache_adapter.get_mcp_cache_analytics(),
            "mcp_memory_type_support": len([t for t, c in self.mcp_memory_types.items() if c.get("mcp_integration", "none") != "none"])
        }


# Global MCP memory retriever instance
_mcp_memory_retriever = None


def get_mcp_memory_retriever(domain_manager, unified_cache_manager=None) -> MCPMemoryRetriever:
    """Get or create global MCP memory retriever."""
    global _mcp_memory_retriever
    if _mcp_memory_retriever is None:
        _mcp_memory_retriever = MCPMemoryRetriever(domain_manager, unified_cache_manager)
    return _mcp_memory_retriever


# Convenience functions
async def retrieve_with_mcp_context(
    domain_manager,
    query: str,
    current_tools: List[str] = None,
    thinking_stage: str = None,
    workflow_pattern: str = None,
    limit: int = 10,
    strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE_INTELLIGENT
) -> List[RetrievalResult]:
    """Convenience function for MCP-enhanced memory retrieval."""
    
    mcp_context = MCPContext(
        current_tools=current_tools or [],
        thinking_stage=thinking_stage,
        workflow_pattern=workflow_pattern
    )
    
    retriever = get_mcp_memory_retriever(domain_manager)
    return await retriever.retrieve_memories_with_mcp_context(query, mcp_context, limit, strategy)