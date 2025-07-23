"""
Fallback Mechanisms for Enhanced MCP Discovery System.

This module provides graceful degradation strategies when Claude Code native
features are unavailable or when enhanced discovery methods fail, ensuring
the system continues to function with existing capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .tool_indexer import MCPToolIndexer, MCPToolInfo


class FallbackLevel(Enum):
    """Levels of fallback degradation."""
    FULL_NATIVE = "full_native"  # All native features available
    PARTIAL_NATIVE = "partial_native"  # Some native features available
    INFERENCE_ONLY = "inference_only"  # Only inference-based discovery
    MINIMAL = "minimal"  # Basic known tools only


class FeatureStatus(Enum):
    """Status of individual features."""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class SystemCapability:
    """Represents a system capability and its status."""
    name: str
    feature_type: str  # "native_config", "hook_system", "resource_refs", "slash_commands"
    status: FeatureStatus
    last_check: datetime
    error_count: int = 0
    success_count: int = 0
    fallback_method: Optional[Callable] = None


@dataclass 
class FallbackStrategy:
    """A fallback strategy for when features are unavailable."""
    name: str
    description: str
    trigger_conditions: List[str]
    fallback_method: Callable
    expected_quality: float  # 0.0 to 1.0, quality compared to full feature
    performance_impact: float  # 0.0 to 1.0, performance impact
    dependencies: List[str]  # Other features this fallback depends on


class AdaptiveFallbackManager:
    """
    Manages adaptive fallback strategies for enhanced MCP discovery.
    
    This system monitors the availability of various enhanced features and
    gracefully degrades functionality while maintaining core capabilities.
    """
    
    def __init__(self, tool_indexer: MCPToolIndexer):
        """
        Initialize the adaptive fallback manager.
        
        Args:
            tool_indexer: The enhanced MCP tool indexer
        """
        self.tool_indexer = tool_indexer
        self.capabilities: Dict[str, SystemCapability] = {}
        self.fallback_strategies: Dict[str, FallbackStrategy] = {}
        self.current_fallback_level = FallbackLevel.FULL_NATIVE
        self.last_capability_check = datetime.now() - timedelta(hours=1)
        
        # Performance metrics
        self.fallback_usage_stats: Dict[str, int] = {}
        self.capability_check_interval = timedelta(minutes=5)
        
        self._initialize_capabilities()
        self._initialize_fallback_strategies()
    
    async def initialize(self) -> None:
        """Initialize the fallback manager."""
        logger.info("Initializing adaptive fallback manager...")
        
        try:
            # Check initial capability status
            await self._check_all_capabilities()
            
            # Determine initial fallback level
            await self._determine_fallback_level()
            
            logger.info(f"Fallback manager initialized at level: {self.current_fallback_level.value}")
        
        except Exception as e:
            logger.error(f"Failed to initialize fallback manager: {e}")
            self.current_fallback_level = FallbackLevel.MINIMAL
    
    async def get_available_discovery_methods(self) -> List[str]:
        """
        Get list of currently available discovery methods.
        
        Returns:
            List of available discovery method names
        """
        available_methods = []
        
        # Check each capability
        if self.capabilities.get("native_config", {}).status == FeatureStatus.AVAILABLE:
            available_methods.append("claude_code_native")
        
        if self.capabilities.get("hook_system", {}).status == FeatureStatus.AVAILABLE:
            available_methods.append("hook_learning")
        
        if self.capabilities.get("resource_refs", {}).status == FeatureStatus.AVAILABLE:
            available_methods.append("resource_references")
        
        if self.capabilities.get("slash_commands", {}).status == FeatureStatus.AVAILABLE:
            available_methods.append("slash_commands")
        
        # Always available methods
        available_methods.extend(["environment_config", "known_tools", "inference"])
        
        return available_methods
    
    async def discover_tools_with_fallback(self) -> List[MCPToolInfo]:
        """
        Discover tools using adaptive fallback strategy.
        
        Returns:
            List of discovered MCP tools using best available methods
        """
        try:
            # Check if capabilities need refreshing
            await self._check_capabilities_if_needed()
            
            # Use the best available discovery methods
            discovery_methods = await self._get_prioritized_discovery_methods()
            
            all_tools = []
            for method_name, method_func in discovery_methods:
                try:
                    logger.debug(f"Attempting discovery with method: {method_name}")
                    tools = await method_func()
                    all_tools.extend(tools)
                    
                    # Update success statistics
                    self._update_capability_success(method_name)
                    
                except Exception as e:
                    logger.warning(f"Discovery method {method_name} failed: {e}")
                    self._update_capability_failure(method_name)
                    
                    # Try fallback if available
                    fallback = self._get_fallback_for_method(method_name)
                    if fallback:
                        try:
                            logger.debug(f"Using fallback strategy: {fallback.name}")
                            fallback_tools = await fallback.fallback_method()
                            all_tools.extend(fallback_tools)
                            
                            # Track fallback usage
                            self.fallback_usage_stats[fallback.name] = \
                                self.fallback_usage_stats.get(fallback.name, 0) + 1
                        
                        except Exception as fallback_error:
                            logger.warning(f"Fallback {fallback.name} also failed: {fallback_error}")
            
            # Remove duplicates
            unique_tools = self._deduplicate_tools(all_tools)
            
            logger.info(f"Discovered {len(unique_tools)} unique tools using fallback-aware methods")
            return unique_tools
        
        except Exception as e:
            logger.error(f"All discovery methods failed, using minimal fallback: {e}")
            return await self._minimal_discovery_fallback()
    
    async def suggest_with_fallback(
        self, 
        user_input: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate suggestions using adaptive fallback methods.
        
        Args:
            user_input: User's input
            context: Current context
            
        Returns:
            List of suggestions using best available methods
        """
        suggestions = []
        
        try:
            # Try enhanced suggestion methods based on available capabilities
            if self.current_fallback_level in [FallbackLevel.FULL_NATIVE, FallbackLevel.PARTIAL_NATIVE]:
                suggestions.extend(await self._enhanced_suggestions(user_input, context))
            
            # Always add basic suggestions
            suggestions.extend(await self._basic_suggestions(user_input, context))
            
            # Remove duplicates and sort by quality
            return self._prioritize_suggestions(suggestions)
        
        except Exception as e:
            logger.warning(f"Enhanced suggestions failed, using basic fallback: {e}")
            return await self._basic_suggestions(user_input, context)
    
    async def _check_all_capabilities(self) -> None:
        """Check the status of all system capabilities."""
        for capability_name in self.capabilities:
            await self._check_capability(capability_name)
    
    async def _check_capability(self, capability_name: str) -> FeatureStatus:
        """Check the status of a specific capability."""
        capability = self.capabilities.get(capability_name)
        if not capability:
            return FeatureStatus.UNKNOWN
        
        try:
            # Perform capability-specific checks
            if capability_name == "native_config":
                status = await self._check_native_config_capability()
            elif capability_name == "hook_system":
                status = await self._check_hook_system_capability()
            elif capability_name == "resource_refs":
                status = await self._check_resource_refs_capability()
            elif capability_name == "slash_commands":
                status = await self._check_slash_commands_capability()
            else:
                status = FeatureStatus.UNKNOWN
            
            # Update capability status
            capability.status = status
            capability.last_check = datetime.now()
            
            if status == FeatureStatus.AVAILABLE:
                capability.success_count += 1
            else:
                capability.error_count += 1
            
            return status
        
        except Exception as e:
            logger.warning(f"Failed to check capability {capability_name}: {e}")
            capability.status = FeatureStatus.UNAVAILABLE
            capability.error_count += 1
            return FeatureStatus.UNAVAILABLE
    
    async def _check_native_config_capability(self) -> FeatureStatus:
        """Check if Claude Code native configuration is available."""
        try:
            # Try to execute claude mcp list
            import subprocess
            result = subprocess.run(
                ["claude", "mcp", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return FeatureStatus.AVAILABLE
            else:
                return FeatureStatus.UNAVAILABLE
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return FeatureStatus.UNAVAILABLE
        except Exception:
            return FeatureStatus.DEGRADED
    
    async def _check_hook_system_capability(self) -> FeatureStatus:
        """Check if hook system integration is available."""
        try:
            # Check if hook configuration directory exists and is writable
            import os
            from pathlib import Path
            
            hook_config_dir = Path("~/.claude-code/hooks").expanduser()
            
            if hook_config_dir.exists() and os.access(hook_config_dir, os.W_OK):
                return FeatureStatus.AVAILABLE
            elif hook_config_dir.exists():
                return FeatureStatus.DEGRADED
            else:
                return FeatureStatus.UNAVAILABLE
        
        except Exception:
            return FeatureStatus.UNAVAILABLE
    
    async def _check_resource_refs_capability(self) -> FeatureStatus:
        """Check if resource reference monitoring is available."""
        # Resource reference monitoring is primarily software-based
        # so it should generally be available unless there are memory issues
        try:
            # Simple capability check - can we access the domain manager?
            if hasattr(self.tool_indexer, 'domain_manager'):
                return FeatureStatus.AVAILABLE
            else:
                return FeatureStatus.DEGRADED
        except Exception:
            return FeatureStatus.UNAVAILABLE
    
    async def _check_slash_commands_capability(self) -> FeatureStatus:
        """Check if slash command discovery is available."""
        # Similar to resource refs, this is primarily software-based
        try:
            # Check if we can access MCP server information
            if hasattr(self.tool_indexer, 'domain_manager'):
                return FeatureStatus.AVAILABLE
            else:
                return FeatureStatus.DEGRADED
        except Exception:
            return FeatureStatus.UNAVAILABLE
    
    async def _determine_fallback_level(self) -> FallbackLevel:
        """Determine the appropriate fallback level based on capability status."""
        available_count = sum(1 for cap in self.capabilities.values() 
                            if cap.status == FeatureStatus.AVAILABLE)
        total_count = len(self.capabilities)
        
        if available_count == total_count:
            self.current_fallback_level = FallbackLevel.FULL_NATIVE
        elif available_count >= total_count // 2:
            self.current_fallback_level = FallbackLevel.PARTIAL_NATIVE
        elif available_count > 0:
            self.current_fallback_level = FallbackLevel.INFERENCE_ONLY
        else:
            self.current_fallback_level = FallbackLevel.MINIMAL
        
        return self.current_fallback_level
    
    async def _get_prioritized_discovery_methods(self) -> List[tuple]:
        """Get discovery methods prioritized by availability and quality."""
        methods = []
        
        # Native methods (highest priority if available)
        if self.capabilities.get("native_config", {}).status == FeatureStatus.AVAILABLE:
            methods.append(("native_config", self.tool_indexer._discover_from_claude_code_native))
        
        # Standard methods
        methods.append(("environment_config", self.tool_indexer._discover_from_configuration))
        methods.append(("known_tools", self.tool_indexer._discover_known_tools))
        
        # MCP server methods (if available)
        if hasattr(self.tool_indexer, '_discover_from_mcp_servers'):
            methods.append(("mcp_servers", self.tool_indexer._discover_from_mcp_servers))
        
        return methods
    
    def _get_fallback_for_method(self, method_name: str) -> Optional[FallbackStrategy]:
        """Get fallback strategy for a failed method."""
        fallback_mapping = {
            "native_config": "inference_based_discovery",
            "hook_system": "pattern_based_suggestions", 
            "resource_refs": "static_reference_patterns",
            "slash_commands": "server_based_command_inference"
        }
        
        fallback_name = fallback_mapping.get(method_name)
        return self.fallback_strategies.get(fallback_name) if fallback_name else None
    
    async def _enhanced_suggestions(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions using enhanced methods when available."""
        suggestions = []
        
        # This would integrate with the enhanced suggestion systems
        # when they're available (hook learning, resource monitoring, etc.)
        
        return suggestions
    
    async def _basic_suggestions(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic suggestions using always-available methods."""
        suggestions = []
        
        # Basic keyword-based suggestions
        keywords = user_input.lower().split()
        
        if any(keyword in ["database", "sql", "query"] for keyword in keywords):
            suggestions.append({
                "type": "mcp_tool",
                "tool": "postgres_query",
                "confidence": 0.7,
                "reasoning": "Database operations detected"
            })
        
        if any(keyword in ["web", "browser", "page"] for keyword in keywords):
            suggestions.append({
                "type": "mcp_tool",
                "tool": "playwright_navigate",
                "confidence": 0.7,
                "reasoning": "Web automation operations detected"
            })
        
        return suggestions
    
    async def _minimal_discovery_fallback(self) -> List[MCPToolInfo]:
        """Minimal discovery fallback when all methods fail."""
        try:
            return await self.tool_indexer._discover_known_tools()
        except Exception as e:
            logger.error(f"Even minimal discovery failed: {e}")
            return []
    
    def _deduplicate_tools(self, tools: List[MCPToolInfo]) -> List[MCPToolInfo]:
        """Remove duplicate tools from the list."""
        seen_tools = set()
        unique_tools = []
        
        for tool in tools:
            tool_key = (tool.name, tool.server_name)
            if tool_key not in seen_tools:
                seen_tools.add(tool_key)
                unique_tools.append(tool)
        
        return unique_tools
    
    def _prioritize_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize suggestions by confidence and remove duplicates."""
        # Remove duplicates
        seen = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            suggestion_key = (suggestion.get("type"), suggestion.get("tool"))
            if suggestion_key not in seen:
                seen.add(suggestion_key)
                unique_suggestions.append(suggestion)
        
        # Sort by confidence
        unique_suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return unique_suggestions
    
    def _update_capability_success(self, capability_name: str) -> None:
        """Update success count for a capability."""
        if capability_name in self.capabilities:
            self.capabilities[capability_name].success_count += 1
    
    def _update_capability_failure(self, capability_name: str) -> None:
        """Update failure count for a capability."""
        if capability_name in self.capabilities:
            self.capabilities[capability_name].error_count += 1
    
    async def _check_capabilities_if_needed(self) -> None:
        """Check capabilities if enough time has passed."""
        if datetime.now() - self.last_capability_check > self.capability_check_interval:
            await self._check_all_capabilities()
            await self._determine_fallback_level()
            self.last_capability_check = datetime.now()
    
    def _initialize_capabilities(self) -> None:
        """Initialize system capability tracking."""
        self.capabilities = {
            "native_config": SystemCapability(
                name="Claude Code Native Configuration",
                feature_type="native_config",
                status=FeatureStatus.UNKNOWN,
                last_check=datetime.now() - timedelta(hours=1)
            ),
            "hook_system": SystemCapability(
                name="Hook System Integration",
                feature_type="hook_system", 
                status=FeatureStatus.UNKNOWN,
                last_check=datetime.now() - timedelta(hours=1)
            ),
            "resource_refs": SystemCapability(
                name="Resource Reference Monitoring",
                feature_type="resource_refs",
                status=FeatureStatus.UNKNOWN,
                last_check=datetime.now() - timedelta(hours=1)
            ),
            "slash_commands": SystemCapability(
                name="Slash Command Discovery",
                feature_type="slash_commands",
                status=FeatureStatus.UNKNOWN,
                last_check=datetime.now() - timedelta(hours=1)
            )
        }
    
    def _initialize_fallback_strategies(self) -> None:
        """Initialize fallback strategies for each feature."""
        self.fallback_strategies = {
            "inference_based_discovery": FallbackStrategy(
                name="Inference-Based Discovery",
                description="Infer tools from server names and patterns when native config unavailable",
                trigger_conditions=["native_config_unavailable"],
                fallback_method=self._inference_discovery_fallback,
                expected_quality=0.6,
                performance_impact=0.2,
                dependencies=[]
            ),
            "pattern_based_suggestions": FallbackStrategy(
                name="Pattern-Based Suggestions",
                description="Use static patterns for suggestions when hook learning unavailable",
                trigger_conditions=["hook_system_unavailable"],
                fallback_method=self._pattern_suggestions_fallback,
                expected_quality=0.5,
                performance_impact=0.1,
                dependencies=[]
            ),
            "static_reference_patterns": FallbackStrategy(
                name="Static Reference Patterns",
                description="Use predefined resource reference patterns",
                trigger_conditions=["resource_refs_unavailable"],
                fallback_method=self._static_references_fallback,
                expected_quality=0.4,
                performance_impact=0.1,
                dependencies=[]
            ),
            "server_based_command_inference": FallbackStrategy(
                name="Server-Based Command Inference",
                description="Infer slash commands from known server types",
                trigger_conditions=["slash_commands_unavailable"],
                fallback_method=self._server_command_inference_fallback,
                expected_quality=0.5,
                performance_impact=0.1,
                dependencies=[]
            )
        }
    
    async def _inference_discovery_fallback(self) -> List[MCPToolInfo]:
        """Fallback discovery using inference."""
        try:
            # This would use the existing inference methods in the tool indexer
            return await self.tool_indexer._discover_known_tools()
        except Exception as e:
            logger.warning(f"Inference discovery fallback failed: {e}")
            return []
    
    async def _pattern_suggestions_fallback(self) -> List[Dict[str, Any]]:
        """Fallback suggestions using static patterns."""
        return [
            {"type": "fallback_pattern", "pattern": "database_query", "confidence": 0.5},
            {"type": "fallback_pattern", "pattern": "web_automation", "confidence": 0.5},
            {"type": "fallback_pattern", "pattern": "file_operations", "confidence": 0.5}
        ]
    
    async def _static_references_fallback(self) -> List[str]:
        """Fallback resource references using static patterns."""
        return [
            "@filesystem:file://",
            "@postgres:database://",
            "@web:http://"
        ]
    
    async def _server_command_inference_fallback(self) -> List[str]:
        """Fallback slash commands using server inference."""
        return [
            "/mcp__postgres__query",
            "/mcp__playwright__navigate",
            "/mcp__filesystem__read"
        ]
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate a system health report."""
        total_capabilities = len(self.capabilities)
        available_capabilities = sum(1 for cap in self.capabilities.values() 
                                   if cap.status == FeatureStatus.AVAILABLE)
        
        return {
            "fallback_level": self.current_fallback_level.value,
            "capability_health": {
                "total": total_capabilities,
                "available": available_capabilities,
                "degraded": sum(1 for cap in self.capabilities.values() 
                              if cap.status == FeatureStatus.DEGRADED),
                "unavailable": sum(1 for cap in self.capabilities.values() 
                                 if cap.status == FeatureStatus.UNAVAILABLE)
            },
            "fallback_usage": self.fallback_usage_stats.copy(),
            "capability_details": {
                name: {
                    "status": cap.status.value,
                    "success_count": cap.success_count,
                    "error_count": cap.error_count,
                    "last_check": cap.last_check.isoformat()
                }
                for name, cap in self.capabilities.items()
            }
        }