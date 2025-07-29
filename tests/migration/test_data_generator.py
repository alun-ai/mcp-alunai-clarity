#!/usr/bin/env python3
"""
Test Data Generation Framework for SQLite Migration Testing

Generates realistic test data that covers all current Qdrant functionality:
- All memory types used in the system
- Realistic content patterns
- Complex metadata structures
- Embedding-friendly text content
- Edge cases and stress test data
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import string
import numpy as np


class MemoryTestDataGenerator:
    """
    Generates comprehensive test data for memory system testing.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with reproducible random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Content templates for different memory types
        self.content_templates = {
            "structured_thinking": [
                "Problem analysis: {problem}. Approach: {approach}. Key insights: {insights}",
                "Thinking through {topic}: Stage {stage} - {content}. Next steps: {next_steps}",
                "Systematic evaluation of {subject}. Criteria: {criteria}. Conclusion: {conclusion}"
            ],
            "thinking_relationship": [
                "Thought {source} builds upon {target} by {relationship_type}",
                "Connection discovered: {source} contradicts {target} because {reason}",
                "Dependency identified: {source} requires {target} for {purpose}"
            ],
            "procedural_thinking_trigger": [
                "Triggered procedural analysis for: {task}. Context: {context}. Complexity: {complexity}",
                "Step-by-step approach needed for {process}. Current state: {state}",
                "Procedural pattern detected in {domain}: {pattern_description}"
            ],
            "mcp_workflow_pattern": [
                "MCP workflow for {tool}: {workflow_steps}. Integration points: {integrations}",
                "Tool usage pattern: {pattern}. Efficiency improvements: {improvements}",
                "Cross-system workflow: {systems} -> {process} -> {outcome}"
            ],
            "mcp_thinking_workflow": [
                "Enhanced thinking with MCP tools: {tools_used}. Insights: {insights}",
                "Integrated workflow: {thinking_stage} + {mcp_capabilities} = {result}",
                "Multi-system reasoning: {reasoning_chain}"
            ],
            "thinking_mcp_integration": [
                "Cross-system intelligence: {source_system} insights enhance {target_system}",
                "MCP-enhanced decision making: {decision_context} -> {enhanced_outcome}",
                "Integrated reasoning chain: {chain_description}"
            ],
            "thinking_session_summary": [
                "Session completed: {title}. Thoughts: {thought_count}. Key outcomes: {outcomes}",
                "Structured thinking session: {duration} minutes, {stages} stages, {insights} insights",
                "Problem-solving session: {problem} -> {solution_approach} -> {recommendations}"
            ],
            "episodic": [
                "Event: {event_description}. Time: {timestamp}. Context: {situational_context}",
                "Experience with {subject}: {experience_details}. Learning: {lesson_learned}",
                "Interaction record: {participants} discussed {topic}. Outcome: {result}"
            ],
            "semantic": [
                "Concept: {concept}. Definition: {definition}. Applications: {applications}",
                "Knowledge about {domain}: {facts}. Relationships: {connections}",
                "Semantic relationship: {entity1} is {relationship} {entity2}"
            ],
            "procedural": [
                "How to {task}: Step 1: {step1}. Step 2: {step2}. Step 3: {step3}",
                "Process for {process_name}: {process_description}. Best practices: {practices}",
                "Procedural knowledge: {skill} requires {requirements} and produces {outputs}"
            ],
            "metadata": [
                "System metadata: {key} = {value}. Category: {category}",
                "Configuration setting: {setting_name} configured to {setting_value}",
                "System state: {component} status is {status} as of {timestamp}"
            ],
            "system": [
                "System event: {event_type} occurred in {subsystem}. Impact: {impact}",
                "Performance metric: {metric_name} = {metric_value} at {measurement_time}",
                "System configuration: {config_key} updated to {config_value}"
            ]
        }
        
        # Sample data for template filling
        self.sample_data = {
            "problems": [
                "authentication system design",
                "database performance optimization", 
                "API rate limiting strategy",
                "microservices communication",
                "data consistency patterns",
                "security vulnerability assessment",
                "user experience optimization",
                "scalability architecture planning"
            ],
            "approaches": [
                "systematic analysis",
                "iterative refinement",
                "comparative evaluation",
                "risk-based assessment",
                "stakeholder consultation",
                "prototype validation",
                "performance benchmarking",
                "security-first design"
            ],
            "tools": [
                "code_analyzer", "database_profiler", "performance_monitor",
                "security_scanner", "api_tester", "log_analyzer",
                "metrics_collector", "deployment_manager"
            ],
            "domains": [
                "web development", "data science", "machine learning",
                "cybersecurity", "devops", "mobile development",
                "cloud architecture", "system administration"
            ],
            "concepts": [
                "OAuth2 authentication", "JWT tokens", "database indexing",
                "load balancing", "caching strategies", "API design patterns",
                "microservices architecture", "event-driven systems"
            ]
        }
    
    def generate_memory(self, 
                       memory_type: str = None,
                       tier: str = None,
                       content_length: int = None,
                       importance: float = None,
                       include_metadata: bool = True) -> Dict[str, Any]:
        """
        Generate a realistic memory entry.
        
        Args:
            memory_type: Type of memory to generate
            tier: Memory tier (short_term, long_term, archival, system)
            content_length: Approximate content length
            importance: Importance score (0.0 to 1.0)
            include_metadata: Whether to include metadata
            
        Returns:
            Generated memory dictionary
        """
        # Select memory type
        if memory_type is None:
            memory_types = [
                "structured_thinking", "thinking_relationship", "procedural_thinking_trigger",
                "mcp_workflow_pattern", "mcp_thinking_workflow", "thinking_mcp_integration",
                "thinking_session_summary", "episodic", "semantic", "procedural", "metadata", "system"
            ]
            memory_type = random.choice(memory_types)
        
        # Select tier
        if tier is None:
            tiers = ["short_term", "long_term", "archival", "system"]
            # Weight tiers based on typical usage
            tier_weights = [0.5, 0.3, 0.15, 0.05]
            tier = random.choices(tiers, weights=tier_weights)[0]
        
        # Generate importance
        if importance is None:
            # Higher importance for certain types
            if memory_type in ["thinking_session_summary", "mcp_workflow_pattern"]:
                importance = random.uniform(0.7, 1.0)
            elif memory_type in ["metadata", "system"]:
                importance = random.uniform(0.3, 0.7)
            else:
                importance = random.uniform(0.1, 0.9)
        
        # Generate content
        content = self._generate_content(memory_type, content_length)
        
        # Generate base memory
        memory = {
            "id": str(uuid.uuid4()),
            "type": memory_type,
            "content": content,
            "importance": importance,
            "tier": tier,
            "created_at": self._random_timestamp().isoformat(),
            "updated_at": self._random_timestamp().isoformat()
        }
        
        # Add metadata if requested
        if include_metadata:
            memory["metadata"] = self._generate_metadata(memory_type, tier)
        
        return memory
    
    def _generate_content(self, memory_type: str, target_length: int = None) -> Dict[str, Any]:
        """Generate realistic content based on memory type."""
        if target_length is None:
            target_length = random.randint(100, 800)
        
        # Get template for this memory type
        templates = self.content_templates.get(memory_type, [
            "Generic content for {topic}: {description}. Details: {details}"
        ])
        template = random.choice(templates)
        
        # Fill template with sample data
        content_text = self._fill_template(template, target_length)
        
        # Create structured content based on memory type
        if memory_type == "structured_thinking":
            return {
                "thinking_stage": random.choice(["problem_definition", "analysis", "synthesis", "evaluation"]),
                "thought_number": random.randint(1, 10),
                "main_content": content_text,
                "insights": [self._generate_insight() for _ in range(random.randint(1, 3))],
                "assumptions": [self._generate_assumption() for _ in range(random.randint(0, 2))]
            }
        elif memory_type == "thinking_relationship":
            return {
                "relationship_type": random.choice(["builds_upon", "contradicts", "supports", "requires"]),
                "source_thought_id": str(uuid.uuid4()),
                "target_thought_id": str(uuid.uuid4()),
                "description": content_text,
                "strength": random.uniform(0.1, 1.0)
            }
        elif memory_type in ["mcp_workflow_pattern", "mcp_thinking_workflow"]:
            return {
                "workflow_name": f"workflow_{random.randint(1000, 9999)}",
                "tools_involved": random.sample(self.sample_data["tools"], random.randint(1, 4)),
                "description": content_text,
                "efficiency_score": random.uniform(0.5, 1.0),
                "usage_count": random.randint(1, 100)
            }
        else:
            # Generic structured content
            return {
                "title": self._generate_title(memory_type),
                "description": content_text,
                "tags": [self._generate_tag() for _ in range(random.randint(0, 3))],
                "context": self._generate_context_info()
            }
    
    def _fill_template(self, template: str, target_length: int) -> str:
        """Fill a template with realistic data to approximately match target length."""
        # Extract placeholders
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Generate values for placeholders
        values = {}
        for placeholder in placeholders:
            values[placeholder] = self._generate_placeholder_value(placeholder, target_length // len(placeholders))
        
        # Fill template
        filled = template.format(**values)
        
        # Extend if too short
        while len(filled) < target_length:
            extension = random.choice([
                f" Additional context: {self._generate_sentence()}.",
                f" Furthermore, {self._generate_sentence()}.",
                f" It's worth noting that {self._generate_sentence()}.",
                f" This relates to {random.choice(self.sample_data['concepts'])}."
            ])
            filled += extension
        
        return filled[:target_length + 50]  # Allow some overflow
    
    def _generate_placeholder_value(self, placeholder: str, target_length: int) -> str:
        """Generate a value for a template placeholder."""
        # Map placeholder to appropriate sample data
        placeholder_mappings = {
            "problem": self.sample_data["problems"],
            "approach": self.sample_data["approaches"],
            "tool": self.sample_data["tools"],
            "domain": self.sample_data["domains"],
            "concept": self.sample_data["concepts"],
            "topic": self.sample_data["concepts"] + self.sample_data["domains"],
        }
        
        if placeholder in placeholder_mappings:
            return random.choice(placeholder_mappings[placeholder])
        
        # Generate contextual content based on placeholder name
        if "step" in placeholder.lower():
            return f"execute {random.choice(self.sample_data['tools'])} with parameters"
        elif "time" in placeholder.lower() or "timestamp" in placeholder.lower():
            return self._random_timestamp().strftime("%Y-%m-%d %H:%M:%S")
        elif "count" in placeholder.lower():
            return str(random.randint(1, 20))
        elif "score" in placeholder.lower():
            return f"{random.uniform(0.1, 1.0):.2f}"
        else:
            # Generate generic content
            return self._generate_sentence()
    
    def _generate_sentence(self) -> str:
        """Generate a realistic sentence."""
        subjects = ["the system", "this approach", "the analysis", "our evaluation", "the process"]
        verbs = ["demonstrates", "indicates", "reveals", "suggests", "shows", "confirms"]
        objects = ["improved performance", "better outcomes", "significant insights", "key patterns", "important relationships"]
        
        return f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"
    
    def _generate_insight(self) -> str:
        """Generate a realistic insight."""
        insights = [
            "Performance bottleneck identified in database layer",
            "User behavior pattern suggests optimization opportunity",
            "Security vulnerability requires immediate attention",
            "Integration point shows high failure rate",
            "Caching strategy could improve response times",
            "Data consistency issues detected in distributed system"
        ]
        return random.choice(insights)
    
    def _generate_assumption(self) -> str:
        """Generate a realistic assumption."""
        assumptions = [
            "Database can handle increased load",
            "Users will adopt new interface patterns",
            "Network latency remains consistent",
            "Security protocols are properly implemented",
            "Error handling covers all edge cases",
            "Performance requirements are achievable"
        ]
        return random.choice(assumptions)
    
    def _generate_title(self, memory_type: str) -> str:
        """Generate a title appropriate for the memory type."""
        titles = {
            "structured_thinking": ["Analysis of {}", "Thinking through {}", "Evaluation of {}"],
            "episodic": ["Experience with {}", "Event: {}", "Interaction regarding {}"],
            "semantic": ["Knowledge: {}", "Concept: {}", "Understanding of {}"],
            "procedural": ["How to {}", "Process for {}", "Procedure: {}"],
            "system": ["System: {}", "Configuration: {}", "Status: {}"]
        }
        
        title_templates = titles.get(memory_type, ["Entry: {}"])
        template = random.choice(title_templates)
        topic = random.choice(self.sample_data["concepts"] + self.sample_data["domains"])
        
        return template.format(topic)
    
    def _generate_tag(self) -> str:
        """Generate a realistic tag."""
        tags = [
            "high-priority", "needs-review", "optimization", "security",
            "performance", "user-experience", "integration", "architecture",
            "bug-fix", "enhancement", "research", "documentation"
        ]
        return random.choice(tags)
    
    def _generate_context_info(self) -> Dict[str, Any]:
        """Generate realistic context information."""
        return {
            "session_id": f"session_{random.randint(1000, 9999)}",
            "user_id": f"user_{random.randint(100, 999)}",
            "environment": random.choice(["development", "staging", "production"]),
            "version": f"{random.randint(1, 5)}.{random.randint(0, 10)}.{random.randint(0, 20)}"
        }
    
    def _generate_metadata(self, memory_type: str, tier: str) -> Dict[str, Any]:
        """Generate realistic metadata based on memory type and tier."""
        base_metadata = {
            "memory_type": memory_type,
            "tier": tier,
            "version": "1.0",
            "source": "test_generator"
        }
        
        # Add type-specific metadata
        if memory_type == "structured_thinking":
            base_metadata.update({
                "thinking_session_id": f"session_{random.randint(1000, 9999)}",
                "thought_number": random.randint(1, 20),
                "total_expected": random.randint(5, 25),
                "thinking_stage": random.choice(["problem_definition", "analysis", "synthesis", "evaluation"]),
                "session_title": f"Analysis Session {random.randint(1, 100)}"
            })
        elif memory_type == "thinking_relationship":
            base_metadata.update({
                "relationship_type": random.choice(["builds_upon", "contradicts", "supports", "requires"]),
                "source_thought_id": str(uuid.uuid4()),
                "target_thought_id": str(uuid.uuid4()),
                "strength": random.uniform(0.1, 1.0)
            })
        elif memory_type in ["mcp_workflow_pattern", "mcp_thinking_workflow"]:
            base_metadata.update({
                "workflow_id": f"workflow_{random.randint(1000, 9999)}",
                "tools_used": random.sample(self.sample_data["tools"], random.randint(1, 3)),
                "efficiency_score": random.uniform(0.5, 1.0),
                "usage_frequency": random.choice(["high", "medium", "low"])
            })
        elif tier == "system":
            base_metadata.update({
                "system_component": random.choice(["database", "cache", "api", "auth", "monitoring"]),
                "criticality": random.choice(["high", "medium", "low"]),
                "auto_generated": True
            })
        
        # Add common metadata fields
        base_metadata.update({
            "created_by": f"user_{random.randint(100, 999)}",
            "environment": random.choice(["development", "staging", "production"]),
            "tags": [self._generate_tag() for _ in range(random.randint(0, 2))],
            "category": random.choice(["analysis", "implementation", "optimization", "research"]),
            "confidence_score": random.uniform(0.5, 1.0)
        })
        
        return base_metadata
    
    def _random_timestamp(self, days_back: int = 30) -> datetime:
        """Generate a random timestamp within the last N days."""
        now = datetime.now()
        random_days = random.uniform(0, days_back)
        return now - timedelta(days=random_days)
    
    def generate_test_queries(self, count: int = 50) -> List[str]:
        """Generate realistic test queries for retrieval testing."""
        query_templates = [
            # Direct concept queries
            "What do we know about {}?",
            "How does {} work?",
            "Explain {} implementation",
            "Show me {} patterns",
            
            # Problem-solving queries
            "How to solve {} problem?",
            "What causes {} issues?",
            "Best practices for {}",
            "Troubleshooting {} errors",
            
            # System queries
            "System status of {}",
            "Performance metrics for {}",
            "Configuration of {}",
            "Monitoring {} components",
            
            # Analytical queries
            "Analysis of {} performance",
            "Evaluation of {} approach",
            "Comparison of {} methods",
            "Assessment of {} risks"
        ]
        
        queries = []
        for _ in range(count):
            template = random.choice(query_templates)
            topic = random.choice(self.sample_data["concepts"] + self.sample_data["domains"])
            query = template.format(topic)
            queries.append(query)
        
        # Add some edge case queries
        edge_queries = [
            "",  # Empty query
            "a",  # Single character
            "the quick brown fox jumps over the lazy dog" * 10,  # Very long query
            "special chars: @#$%^&*(){}[]|\\:;\"'<>?,./",  # Special characters
            "unicode: café naïve résumé 🚀 🔥 💯",  # Unicode
            "SQL injection: ' OR 1=1 --",  # SQL injection attempt
        ]
        
        queries.extend(edge_queries)
        
        return queries
    
    def generate_performance_dataset(self, size: int) -> List[Dict[str, Any]]:
        """Generate a dataset optimized for performance testing."""
        dataset = []
        
        # Generate varied content lengths for performance testing
        for i in range(size):
            if i < size * 0.7:  # 70% normal size
                content_length = random.randint(100, 500)
            elif i < size * 0.9:  # 20% large content
                content_length = random.randint(500, 2000)
            else:  # 10% very large content
                content_length = random.randint(2000, 5000)
            
            memory = self.generate_memory(content_length=content_length)
            dataset.append(memory)
        
        return dataset
    
    def generate_concurrent_test_data(self, num_threads: int, operations_per_thread: int) -> List[List[Dict[str, Any]]]:
        """Generate test data for concurrent access testing."""
        thread_data = []
        
        for thread_id in range(num_threads):
            thread_memories = []
            for op_id in range(operations_per_thread):
                memory = self.generate_memory()
                memory["metadata"]["thread_id"] = thread_id
                memory["metadata"]["operation_id"] = op_id
                thread_memories.append(memory)
            
            thread_data.append(thread_memories)
        
        return thread_data


class QdrantTestMemoryGenerator:
    """
    Specialized generator for Qdrant-specific test scenarios.
    """
    
    def __init__(self):
        self.base_generator = MemoryTestDataGenerator()
    
    def generate_qdrant_stress_test_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate data to stress test Qdrant-specific features."""
        memories = []
        
        for i in range(size):
            memory = self.base_generator.generate_memory()
            
            # Add Qdrant-specific stress patterns
            if i % 100 == 0:
                # Large vector dimension tests
                memory["content"]["large_text"] = "A" * 10000
            
            if i % 50 == 0:
                # Complex metadata for filtering tests
                memory["metadata"]["complex_filter"] = {
                    "nested": {
                        "level1": {"level2": {"value": random.randint(1, 100)}},
                        "array": [random.randint(1, 10) for _ in range(5)]
                    }
                }
            
            memories.append(memory)
        
        return memories
    
    def generate_indexing_test_data(self) -> List[Dict[str, Any]]:
        """Generate data specifically for testing HNSW indexing behavior."""
        memories = []
        
        # Generate clusters of similar content to test indexing
        clusters = [
            "database performance optimization",
            "user authentication systems", 
            "API rate limiting strategies",
            "microservices communication patterns",
            "security vulnerability assessment"
        ]
        
        for cluster_topic in clusters:
            for i in range(20):  # 20 memories per cluster
                memory = self.base_generator.generate_memory()
                memory["content"]["description"] = f"{cluster_topic} - variant {i}: " + memory["content"]["description"]
                memories.append(memory)
        
        return memories


class SQLiteTestMemoryGenerator:
    """
    Specialized generator for SQLite-specific test scenarios.
    """
    
    def __init__(self):
        self.base_generator = MemoryTestDataGenerator()
    
    def generate_sqlite_optimization_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate data to test SQLite optimization features."""
        memories = []
        
        for i in range(size):
            memory = self.base_generator.generate_memory()
            
            # Add SQLite-specific optimization patterns
            if i % 10 == 0:
                # Test FTS (Full Text Search) capabilities
                memory["content"]["searchable_text"] = f"searchable content {i} with keywords: optimization, performance, database"
            
            if i % 25 == 0:
                # Test JSON field indexing
                memory["metadata"]["json_data"] = {
                    "indexed_field": f"value_{i % 100}",
                    "searchable_array": [f"item_{j}" for j in range(i % 5 + 1)]
                }
            
            memories.append(memory)
        
        return memories
    
    def generate_transaction_test_data(self) -> List[List[Dict[str, Any]]]:
        """Generate data for transaction testing."""
        # Generate batches that should be processed atomically
        batches = []
        
        for batch_id in range(10):
            batch = []
            for i in range(5):
                memory = self.base_generator.generate_memory()
                memory["metadata"]["batch_id"] = batch_id
                memory["metadata"]["batch_position"] = i
                batch.append(memory)
            batches.append(batch)
        
        return batches


class PerformanceTestDataset:
    """
    Specialized dataset for performance benchmarking.
    """
    
    def __init__(self):
        self.generator = MemoryTestDataGenerator()
    
    def create_scalability_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create datasets of various sizes for scalability testing."""
        sizes = [100, 500, 1000, 2000, 5000]
        datasets = {}
        
        for size in sizes:
            datasets[f"dataset_{size}"] = self.generator.generate_performance_dataset(size)
        
        return datasets
    
    def create_vector_similarity_benchmarks(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Create data specifically for vector similarity benchmarking."""
        # Generate memories with known similarity relationships
        base_topics = [
            "authentication security systems",
            "database optimization techniques", 
            "API design patterns",
            "machine learning algorithms",
            "cloud infrastructure management"
        ]
        
        memories = []
        queries = []
        
        for topic in base_topics:
            # Generate 10 variations of each topic
            for i in range(10):
                memory = self.generator.generate_memory()
                memory["content"]["description"] = f"{topic} implementation approach {i}: " + memory["content"]["description"]
                memories.append(memory)
            
            # Generate queries that should match these memories
            queries.extend([
                f"How to implement {topic}?",
                f"Best practices for {topic}",
                f"Examples of {topic}",
                f"Common issues with {topic}"
            ])
        
        return memories, queries


if __name__ == "__main__":
    # Demo the test data generator
    generator = MemoryTestDataGenerator()
    
    print("🧪 Memory Test Data Generator Demo")
    print("=" * 50)
    
    # Generate sample memories
    sample_memories = [generator.generate_memory() for _ in range(5)]
    
    for i, memory in enumerate(sample_memories):
        print(f"\n📝 Sample Memory {i+1}:")
        print(f"  Type: {memory['type']}")
        print(f"  Tier: {memory['tier']}")
        print(f"  Importance: {memory['importance']:.2f}")
        print(f"  Content: {str(memory['content'])[:100]}...")
    
    # Generate sample queries
    queries = generator.generate_test_queries(5)
    print(f"\n🔍 Sample Queries:")
    for i, query in enumerate(queries):
        print(f"  {i+1}. {query}")
    
    # Performance dataset
    perf_dataset = PerformanceTestDataset()
    datasets = perf_dataset.create_scalability_dataset()
    
    print(f"\n📊 Performance Datasets Created:")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} memories")
    
    print(f"\n✅ Test data generation framework ready!")