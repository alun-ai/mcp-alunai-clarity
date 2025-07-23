"""Resource Reference Learning and Monitoring System.

This module monitors and suggests @server:protocol:// resource references
for more efficient MCP interactions.
"""

import re
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Pattern
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResourceReference:
    """Represents a resource reference pattern."""
    server: str
    protocol: str
    resource_path: str
    full_reference: str
    context: str
    usage_count: int
    success_rate: float
    average_response_time: float
    last_used: str
    
    @classmethod
    def parse_reference(cls, reference: str, context: str = "") -> Optional['ResourceReference']:
        """Parse a resource reference string."""
        # Match pattern: @server:protocol://resource
        match = re.match(r'@([^:]+):([^:]+)://(.+)', reference)
        if not match:
            return None
        
        server, protocol, resource_path = match.groups()
        return cls(
            server=server,
            protocol=protocol,
            resource_path=resource_path,
            full_reference=reference,
            context=context,
            usage_count=0,
            success_rate=1.0,
            average_response_time=0.0,
            last_used=datetime.now(timezone.utc).isoformat()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceReference':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ResourceOpportunity:
    """Represents a detected opportunity for resource reference usage."""
    opportunity_type: str
    suggested_reference: str
    current_approach: str
    confidence: float
    reason: str
    potential_benefits: List[str]
    context_match: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ResourceReferenceMonitor:
    """Monitors and suggests @server:protocol:// resource references."""
    
    def __init__(self):
        """Initialize the resource reference monitor."""
        self.reference_patterns = {}
        self.usage_history = []
        self.opportunity_patterns = self._initialize_opportunity_patterns()
        self.learned_mappings = {}
        self.server_capabilities = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_opportunities_detected': 0,
            'successful_suggestions': 0,
            'reference_usage_tracked': 0
        }
    
    def _initialize_opportunity_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for detecting resource reference opportunities."""
        return {
            'file_operations': {
                'patterns': [
                    re.compile(r'\bread\s+.*?file', re.I),
                    re.compile(r'\bwrite\s+.*?file', re.I),
                    re.compile(r'\bopen\s+.*?file', re.I),
                    re.compile(r'\bpath\s*[:=]\s*[\'"]([^\'"]+)[\'"]', re.I),
                    re.compile(r'\b(?:cat|head|tail|less)\s+([^\s]+)', re.I),
                    re.compile(r'\bls\s+([^\s]+)', re.I),
                    re.compile(r'\bfind\s+([^\s]+)', re.I)
                ],
                'server_types': ['filesystem', 'file', 'files'],
                'protocol': 'file',
                'confidence_base': 0.8,
                'benefits': [
                    'Direct file access without tool overhead',
                    'Better error handling and metadata',
                    'Automatic encoding detection',
                    'Streaming for large files'
                ]
            },
            'database_queries': {
                'patterns': [
                    re.compile(r'\b(?:SELECT|INSERT|UPDATE|DELETE)\b.*?FROM', re.I | re.DOTALL),
                    re.compile(r'\bquery.*?database', re.I),
                    re.compile(r'\bdatabase.*?query', re.I),
                    re.compile(r'\bexecute.*?query', re.I),
                    re.compile(r'\bget.*?(?:users?|data|records?)', re.I),
                    re.compile(r'\bpsql\s+.*?-c\s+[\'"]([^\'"]+)[\'"]', re.I),
                    re.compile(r'\bmysql\s+.*?-e\s+[\'"]([^\'"]+)[\'"]', re.I),
                    re.compile(r'\bsqlite3\s+.*?[\'"]([^\'"]+)[\'"]', re.I)
                ],
                'server_types': ['postgres', 'mysql', 'sqlite', 'database', 'db'],
                'protocol': 'query',
                'confidence_base': 0.9,
                'benefits': [
                    'Native SQL execution with proper types',
                    'Connection pooling and management',
                    'Transaction support',
                    'Better error reporting'
                ]
            },
            'web_requests': {
                'patterns': [
                    re.compile(r'\bcurl\s+.*?https?://([^\s]+)', re.I),
                    re.compile(r'\bwget\s+.*?https?://([^\s]+)', re.I),
                    re.compile(r'\bhttp\s+(?:GET|POST|PUT|DELETE)\s+([^\s]+)', re.I),
                    re.compile(r'\bfetch\s+.*?https?://([^\s]+)', re.I),
                    re.compile(r'\brequest.*?https?://([^\s]+)', re.I),
                    re.compile(r'\bmake.*?api\s+call', re.I),
                    re.compile(r'\bapi\s+(?:request|call)', re.I)
                ],
                'server_types': ['web', 'http', 'api', 'fetch'],
                'protocol': 'request',
                'confidence_base': 0.85,
                'benefits': [
                    'Automatic header management',
                    'Response parsing and validation',
                    'Rate limiting and retries',
                    'Authentication handling'
                ]
            },
            'git_operations': {
                'patterns': [
                    re.compile(r'\bgit\s+(clone|pull|push|commit|log|status)', re.I),
                    re.compile(r'\bgh\s+(repo|pr|issue)', re.I),
                    re.compile(r'\brepository\s+.*?https?://([^\s]+)', re.I),
                    re.compile(r'\bgithub\.com/([^/\s]+/[^/\s]+)', re.I)
                ],
                'server_types': ['git', 'github', 'gitlab'],
                'protocol': 'repo',
                'confidence_base': 0.8,
                'benefits': [
                    'Repository-aware operations',
                    'Automatic authentication',
                    'Branch and commit tracking',
                    'Integration with hosting platforms'
                ]
            },
            'api_endpoints': {
                'patterns': [
                    re.compile(r'\bapi\s+endpoint\s*[:=]\s*[\'"]([^\'"]+)[\'"]', re.I),
                    re.compile(r'\b(?:GET|POST|PUT|DELETE)\s+/api/([^\s]+)', re.I),
                    re.compile(r'\brest\s+api', re.I),
                    re.compile(r'\braphql', re.I)
                ],
                'server_types': ['api', 'rest', 'graphql'],
                'protocol': 'endpoint',
                'confidence_base': 0.75,
                'benefits': [
                    'Schema-aware requests',
                    'Automatic documentation',
                    'Response validation',
                    'Mock data generation'
                ]
            },
            'documentation_access': {
                'patterns': [
                    re.compile(r'\bdocs?\s+.*?https?://([^\s]+)', re.I),
                    re.compile(r'\bdocumentation\s+for\s+([^\s]+)', re.I),
                    re.compile(r'\breadme\s+file', re.I),
                    re.compile(r'\bmanual\s+page', re.I)
                ],
                'server_types': ['docs', 'documentation', 'readme'],
                'protocol': 'doc',
                'confidence_base': 0.7,
                'benefits': [
                    'Structured documentation access',
                    'Search within documentation',
                    'Version-aware content',
                    'Related content suggestions'
                ]
            }
        }
    
    def detect_resource_opportunities(self, prompt: str, context: Dict[str, Any] = None) -> List[ResourceOpportunity]:
        """Detect when resource references could be beneficial."""
        opportunities = []
        context = context or {}
        
        self.performance_stats['total_opportunities_detected'] += 1
        
        for opportunity_type, pattern_info in self.opportunity_patterns.items():
            for pattern in pattern_info['patterns']:
                matches = pattern.finditer(prompt)
                
                for match in matches:
                    # Extract potential resource path
                    resource_path = match.group(1) if match.groups() else match.group(0)
                    
                    # Check if we have available servers for this type
                    available_servers = self._find_compatible_servers(pattern_info['server_types'], context)
                    
                    if available_servers:
                        for server_name in available_servers[:2]:  # Limit to top 2 suggestions
                            suggested_reference = f"@{server_name}:{pattern_info['protocol']}://{resource_path}"
                            
                            # Calculate confidence based on context match
                            context_match = self._calculate_context_match(prompt, opportunity_type)
                            confidence = pattern_info['confidence_base'] * context_match
                            
                            opportunity = ResourceOpportunity(
                                opportunity_type=opportunity_type,
                                suggested_reference=suggested_reference,
                                current_approach=match.group(0),
                                confidence=confidence,
                                reason=self._generate_opportunity_reason(opportunity_type, resource_path),
                                potential_benefits=pattern_info['benefits'],
                                context_match=context_match
                            )
                            
                            opportunities.append(opportunity)
        
        # Remove duplicates and sort by confidence
        unique_opportunities = self._deduplicate_opportunities(opportunities)
        unique_opportunities.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_opportunities[:5]  # Return top 5
    
    async def learn_resource_pattern(self, reference: str, context: Dict[str, Any], success: bool = True, response_time: float = 0.0):
        """Learn successful resource reference patterns."""
        try:
            parsed_ref = ResourceReference.parse_reference(reference, context.get('prompt', ''))
            if not parsed_ref:
                logger.debug(f"Could not parse resource reference: {reference}")
                return
            
            pattern_key = f"{parsed_ref.server}:{parsed_ref.protocol}"
            
            if pattern_key not in self.reference_patterns:
                self.reference_patterns[pattern_key] = {
                    'usage_count': 0,
                    'success_count': 0,
                    'total_response_time': 0.0,
                    'contexts': [],
                    'resource_examples': []
                }
            
            pattern = self.reference_patterns[pattern_key]
            pattern['usage_count'] += 1
            
            if success:
                pattern['success_count'] += 1
                self.performance_stats['successful_suggestions'] += 1
            
            pattern['total_response_time'] += response_time
            pattern['contexts'].append(context.get('intent', 'unknown'))
            
            # Store example resources (keep only recent ones)
            pattern['resource_examples'].append(parsed_ref.resource_path)
            if len(pattern['resource_examples']) > 20:
                pattern['resource_examples'] = pattern['resource_examples'][-10:]
            
            # Update parsed reference
            parsed_ref.usage_count = pattern['usage_count']
            parsed_ref.success_rate = pattern['success_count'] / pattern['usage_count']
            parsed_ref.average_response_time = pattern['total_response_time'] / pattern['usage_count']
            
            # Store in usage history
            usage_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'reference': reference,
                'success': success,
                'response_time': response_time,
                'context': context.get('prompt', '')[:100],  # Truncate
                'server': parsed_ref.server,
                'protocol': parsed_ref.protocol
            }
            
            self.usage_history.append(usage_record)
            
            # Keep history manageable
            if len(self.usage_history) > 1000:
                self.usage_history = self.usage_history[-500:]
            
            self.performance_stats['reference_usage_tracked'] += 1
            
            # Store as structured memory if we have domain manager access
            await self._store_reference_pattern(parsed_ref, context)
            
        except Exception as e:
            logger.error(f"Failed to learn resource pattern: {e}")
    
    async def get_reference_suggestions(self, prompt: str, available_servers: List[str] = None) -> List[Dict[str, Any]]:
        """Get contextual resource reference suggestions."""
        suggestions = []
        
        # Detect opportunities
        context = {'available_servers': available_servers or []}
        opportunities = self.detect_resource_opportunities(prompt, context)
        
        for opportunity in opportunities:
            # Check if we have historical data about this pattern
            pattern_key = self._extract_pattern_key(opportunity.suggested_reference)
            historical_data = self.reference_patterns.get(pattern_key, {})
            
            suggestion = {
                'type': 'resource_reference',
                'reference': opportunity.suggested_reference,
                'confidence': opportunity.confidence,
                'reason': opportunity.reason,
                'benefits': opportunity.potential_benefits,
                'historical_success_rate': historical_data.get('success_count', 0) / max(1, historical_data.get('usage_count', 1)),
                'historical_usage': historical_data.get('usage_count', 0),
                'average_response_time': historical_data.get('total_response_time', 0) / max(1, historical_data.get('usage_count', 1)),
                'opportunity_type': opportunity.opportunity_type,
                'replacement_for': opportunity.current_approach
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    async def analyze_reference_usage_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns of resource references."""
        analysis = {
            'most_successful_patterns': [],
            'server_performance': {},
            'protocol_usage': {},
            'context_effectiveness': {},
            'recent_trends': {},
            'recommendations': []
        }
        
        # Analyze patterns by success rate
        pattern_performance = []
        for pattern_key, pattern_data in self.reference_patterns.items():
            if pattern_data['usage_count'] > 0:
                success_rate = pattern_data['success_count'] / pattern_data['usage_count']
                avg_response_time = pattern_data['total_response_time'] / pattern_data['usage_count']
                
                pattern_performance.append({
                    'pattern': pattern_key,
                    'success_rate': success_rate,
                    'usage_count': pattern_data['usage_count'],
                    'avg_response_time': avg_response_time,
                    'contexts': list(set(pattern_data['contexts']))
                })
        
        # Sort by success rate and usage count
        pattern_performance.sort(key=lambda x: (x['success_rate'], x['usage_count']), reverse=True)
        analysis['most_successful_patterns'] = pattern_performance[:10]
        
        # Analyze by server
        server_stats = {}
        protocol_stats = {}
        
        for record in self.usage_history:
            server = record['server']
            protocol = record['protocol']
            success = record['success']
            
            # Server stats
            if server not in server_stats:
                server_stats[server] = {'total': 0, 'success': 0, 'total_time': 0.0}
            server_stats[server]['total'] += 1
            if success:
                server_stats[server]['success'] += 1
            server_stats[server]['total_time'] += record.get('response_time', 0)
            
            # Protocol stats
            if protocol not in protocol_stats:
                protocol_stats[protocol] = {'total': 0, 'success': 0}
            protocol_stats[protocol]['total'] += 1
            if success:
                protocol_stats[protocol]['success'] += 1
        
        # Calculate server performance
        for server, stats in server_stats.items():
            analysis['server_performance'][server] = {
                'success_rate': stats['success'] / stats['total'],
                'usage_count': stats['total'],
                'avg_response_time': stats['total_time'] / stats['total']
            }
        
        # Calculate protocol usage
        for protocol, stats in protocol_stats.items():
            analysis['protocol_usage'][protocol] = {
                'success_rate': stats['success'] / stats['total'],
                'usage_count': stats['total']
            }
        
        # Recent trends (last 24 hours)
        recent_time = datetime.now(timezone.utc).timestamp() - 86400
        recent_records = []
        
        for record in self.usage_history:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')).timestamp()
                if timestamp > recent_time:
                    recent_records.append(record)
            except (ValueError, KeyError):
                continue
        
        if recent_records:
            recent_success = sum(1 for r in recent_records if r['success'])
            analysis['recent_trends'] = {
                'total_usage': len(recent_records),
                'success_rate': recent_success / len(recent_records),
                'top_servers': self._get_top_items([r['server'] for r in recent_records]),
                'top_protocols': self._get_top_items([r['protocol'] for r in recent_records])
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_usage_recommendations(analysis)
        
        return analysis
    
    def _find_compatible_servers(self, server_types: List[str], context: Dict[str, Any]) -> List[str]:
        """Find compatible servers for given server types."""
        available_servers = context.get('available_servers', [])
        
        if not available_servers:
            # Return potential server names if no context provided
            return server_types[:1]
        
        compatible = []
        for server_type in server_types:
            for server in available_servers:
                if server_type.lower() in server.lower():
                    compatible.append(server)
        
        # Also add exact matches
        for server in available_servers:
            if server.lower() in [s.lower() for s in server_types]:
                compatible.append(server)
        
        return list(set(compatible))  # Remove duplicates
    
    def _calculate_context_match(self, prompt: str, opportunity_type: str) -> float:
        """Calculate how well the prompt matches the opportunity type context."""
        prompt_lower = prompt.lower()
        
        # Context keywords for each opportunity type
        context_keywords = {
            'file_operations': ['file', 'directory', 'folder', 'path', 'read', 'write', 'open'],
            'database_queries': ['database', 'query', 'sql', 'table', 'select', 'insert', 'update'],
            'web_requests': ['http', 'api', 'request', 'endpoint', 'url', 'fetch', 'curl'],
            'git_operations': ['git', 'repository', 'repo', 'commit', 'branch', 'github'],
            'api_endpoints': ['api', 'endpoint', 'rest', 'graphql', 'service'],
            'documentation_access': ['docs', 'documentation', 'readme', 'manual', 'help']
        }
        
        keywords = context_keywords.get(opportunity_type, [])
        if not keywords:
            return 0.5  # Default match
        
        matches = sum(1 for keyword in keywords if keyword in prompt_lower)
        return min(1.0, (matches / len(keywords)) + 0.6)  # Base 0.6 + keyword matches
    
    def _generate_opportunity_reason(self, opportunity_type: str, resource_path: str) -> str:
        """Generate a reason for the resource reference opportunity."""
        reasons = {
            'file_operations': f"Direct file access to '{resource_path}' through MCP filesystem server",
            'database_queries': f"Execute query through MCP database server instead of shell commands",
            'web_requests': f"Use MCP web server for reliable HTTP requests to '{resource_path}'",
            'git_operations': f"Use MCP git server for repository operations on '{resource_path}'",
            'api_endpoints': f"Access API endpoint '{resource_path}' through dedicated MCP server",
            'documentation_access': f"Access documentation '{resource_path}' through structured MCP server"
        }
        
        return reasons.get(opportunity_type, f"Use MCP server for '{resource_path}' operations")
    
    def _deduplicate_opportunities(self, opportunities: List[ResourceOpportunity]) -> List[ResourceOpportunity]:
        """Remove duplicate opportunities based on suggested reference."""
        seen_references = set()
        unique_opportunities = []
        
        for opportunity in opportunities:
            if opportunity.suggested_reference not in seen_references:
                seen_references.add(opportunity.suggested_reference)
                unique_opportunities.append(opportunity)
        
        return unique_opportunities
    
    def _extract_pattern_key(self, reference: str) -> str:
        """Extract pattern key from resource reference."""
        parsed = ResourceReference.parse_reference(reference)
        if parsed:
            return f"{parsed.server}:{parsed.protocol}"
        return reference
    
    async def _store_reference_pattern(self, reference: ResourceReference, context: Dict[str, Any]):
        """Store reference pattern as structured memory."""
        try:
            # This would integrate with the domain manager if available
            # For now, just log the pattern
            logger.debug(f"Storing reference pattern: {reference.full_reference}")
            
            # TODO: Integrate with domain manager to store as memory
            # memory_content = {
            #     'reference': reference.to_dict(),
            #     'context': context,
            #     'learned_at': datetime.now(timezone.utc).isoformat()
            # }
            # await domain_manager.store_memory(
            #     memory_type="resource_reference_pattern",
            #     content=json.dumps(memory_content),
            #     importance=0.7
            # )
            
        except Exception as e:
            logger.debug(f"Could not store reference pattern: {e}")
    
    def _get_top_items(self, items: List[str], limit: int = 5) -> Dict[str, int]:
        """Get top items by frequency."""
        from collections import Counter
        counter = Counter(items)
        return dict(counter.most_common(limit))
    
    def _generate_usage_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on usage analysis."""
        recommendations = []
        
        # Check for underperforming servers
        for server, stats in analysis['server_performance'].items():
            if stats['success_rate'] < 0.5 and stats['usage_count'] > 5:
                recommendations.append(f"Consider reviewing {server} server configuration - low success rate ({stats['success_rate']:.1%})")
        
        # Check for popular but slow servers
        for server, stats in analysis['server_performance'].items():
            if stats['avg_response_time'] > 5000 and stats['usage_count'] > 10:  # >5 seconds
                recommendations.append(f"Consider optimizing {server} server - high response time ({stats['avg_response_time']:.0f}ms)")
        
        # Suggest popular patterns to users
        if analysis['most_successful_patterns']:
            top_pattern = analysis['most_successful_patterns'][0]
            if top_pattern['success_rate'] > 0.8 and top_pattern['usage_count'] > 5:
                recommendations.append(f"Pattern '{top_pattern['pattern']}' is highly successful ({top_pattern['success_rate']:.1%}) - consider using it more often")
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the monitor."""
        stats = self.performance_stats.copy()
        
        stats.update({
            'total_patterns_learned': len(self.reference_patterns),
            'total_usage_records': len(self.usage_history),
            'suggestion_success_rate': (
                stats['successful_suggestions'] / max(1, stats['total_opportunities_detected'])
            ),
            'patterns_by_server': {},
            'recent_activity': 0
        })
        
        # Count patterns by server
        for pattern_key in self.reference_patterns.keys():
            server = pattern_key.split(':')[0]
            stats['patterns_by_server'][server] = stats['patterns_by_server'].get(server, 0) + 1
        
        # Count recent activity (last hour)
        current_time = datetime.now(timezone.utc).timestamp()
        recent_cutoff = current_time - 3600
        
        for record in self.usage_history:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')).timestamp()
                if timestamp > recent_cutoff:
                    stats['recent_activity'] += 1
            except (ValueError, KeyError):
                continue
        
        return stats