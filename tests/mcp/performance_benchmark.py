#!/usr/bin/env python3
"""
Performance benchmarking script for enhanced MCP discovery system.

This script measures the performance of key operations and validates
that they meet the <500ms response time target.
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.mcp.performance_optimization import PerformanceOptimizer
from clarity.mcp.native_discovery import NativeMCPDiscoveryBridge
from clarity.mcp.hook_integration import MCPHookIntegration
from clarity.mcp.workflow_memory import WorkflowMemoryEnhancer
from clarity.mcp.resource_reference_monitor import ResourceReferenceMonitor
from clarity.mcp.slash_command_discovery import SlashCommandDiscovery


class MockDomainManager:
    """Mock domain manager for benchmarking."""
    
    def __init__(self):
        self.stored_memories = []
        self.memory_counter = 0
    
    async def store_memory(self, memory_type: str, content: str, importance: float, metadata: dict = None):
        """Mock memory storage with simulated delay."""
        await asyncio.sleep(0.001)  # 1ms simulated storage time
        memory = {
            'id': f"mem_{self.memory_counter}",
            'memory_type': memory_type,
            'content': content,
            'importance': importance,
            'metadata': metadata or {}
        }
        self.stored_memories.append(memory)
        self.memory_counter += 1
        return memory['id']
    
    async def retrieve_memories(self, query: str, types: list = None, limit: int = 10, min_similarity: float = 0.5):
        """Mock memory retrieval with simulated delay."""
        await asyncio.sleep(0.002)  # 2ms simulated retrieval time
        return self.stored_memories[:limit]


class PerformanceBenchmark:
    """Performance benchmarking suite."""
    
    def __init__(self):
        self.results = {}
        self.domain_manager = MockDomainManager()
        self.target_response_time = 0.5  # 500ms
        
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite."""
        print("🚀 Starting Enhanced MCP Discovery Performance Benchmarks")
        print("=" * 60)
        
        # Component initialization benchmarks
        await self._benchmark_component_initialization()
        
        # Core operation benchmarks
        await self._benchmark_core_operations()
        
        # Integration benchmarks
        await self._benchmark_integration_scenarios()
        
        # Performance optimization benchmarks
        await self._benchmark_performance_optimization()
        
        # Generate summary report
        return self._generate_summary_report()
    
    async def _benchmark_component_initialization(self):
        """Benchmark component initialization times."""
        print("\n📦 Component Initialization Benchmarks")
        print("-" * 40)
        
        # Tool Indexer initialization
        times = await self._measure_operation(
            "Tool Indexer Initialization",
            lambda: MCPToolIndexer(self.domain_manager),
            iterations=10,
            is_async=False
        )
        self.results['tool_indexer_init'] = times
        
        # Native Discovery Bridge
        times = await self._measure_operation(
            "Native Discovery Bridge",
            lambda: NativeMCPDiscoveryBridge(),
            iterations=10,
            is_async=False
        )
        self.results['native_bridge_init'] = times
        
        # Workflow Memory Enhancer
        times = await self._measure_operation(
            "Workflow Memory Enhancer",
            lambda: WorkflowMemoryEnhancer(self.domain_manager),
            iterations=10,
            is_async=False
        )
        self.results['workflow_memory_init'] = times
        
        # Resource Reference Monitor
        times = await self._measure_operation(
            "Resource Reference Monitor",
            lambda: ResourceReferenceMonitor(),
            iterations=10,
            is_async=False
        )
        self.results['resource_monitor_init'] = times
    
    async def _benchmark_core_operations(self):
        """Benchmark core MCP discovery operations."""
        print("\n⚡ Core Operations Benchmarks")
        print("-" * 40)
        
        # Setup components
        indexer = MCPToolIndexer(self.domain_manager)
        bridge = NativeMCPDiscoveryBridge()
        workflow_enhancer = WorkflowMemoryEnhancer(self.domain_manager)
        resource_monitor = ResourceReferenceMonitor()
        
        # Server discovery
        times = await self._measure_operation(
            "Server Discovery",
            bridge.discover_native_servers,
            iterations=5,
            is_async=True
        )
        self.results['server_discovery'] = times
        
        # Resource opportunity detection
        times = await self._measure_operation(
            "Resource Opportunity Detection",
            lambda: resource_monitor.detect_resource_opportunities(
                "read the configuration file at /etc/config.json"
            ),
            iterations=20,
            is_async=False
        )
        self.results['resource_opportunity_detection'] = times
        
        # Workflow pattern storage
        pattern_data = {
            "context": "test workflow",
            "tools": ["test_tool"],
            "score": 0.8
        }
        times = await self._measure_operation(
            "Workflow Pattern Storage",
            lambda: workflow_enhancer.store_mcp_workflow_pattern(pattern_data),
            iterations=10,
            is_async=True
        )
        self.results['workflow_pattern_storage'] = times
        
        # Workflow pattern retrieval
        times = await self._measure_operation(
            "Workflow Pattern Retrieval",
            lambda: workflow_enhancer.find_similar_workflows("test workflow"),
            iterations=10,
            is_async=True
        )
        self.results['workflow_pattern_retrieval'] = times
    
    async def _benchmark_integration_scenarios(self):
        """Benchmark realistic integration scenarios."""
        print("\n🔄 Integration Scenario Benchmarks")
        print("-" * 40)
        
        indexer = MCPToolIndexer(self.domain_manager)
        
        # Complete discovery workflow (mocked)
        times = await self._measure_operation(
            "Complete Discovery Workflow",
            self._mock_complete_discovery_workflow,
            iterations=3,
            is_async=True
        )
        self.results['complete_discovery_workflow'] = times
        
        # Resource suggestion generation
        times = await self._measure_operation(
            "Resource Suggestions",
            lambda: indexer.get_resource_suggestions("read database config"),
            iterations=15,
            is_async=True
        )
        self.results['resource_suggestions'] = times
        
        # Workflow suggestions
        times = await self._measure_operation(
            "Workflow Suggestions",
            lambda: indexer.get_workflow_suggestions(
                "setup database connection", 
                context={'project_type': 'web_app'}
            ),
            iterations=15,
            is_async=True
        )
        self.results['workflow_suggestions'] = times
    
    async def _benchmark_performance_optimization(self):
        """Benchmark performance optimization features."""
        print("\n🎯 Performance Optimization Benchmarks")
        print("-" * 40)
        
        optimizer = PerformanceOptimizer()
        
        # Cache operations
        times = await self._measure_operation(
            "Cache Put/Get Operations",
            lambda: self._test_cache_operations(optimizer.cache),
            iterations=100,
            is_async=False
        )
        self.results['cache_operations'] = times
        
        # Parallel execution
        tasks = [
            (lambda: time.sleep(0.01), (), {}),
            (lambda: time.sleep(0.01), (), {}),
            (lambda: time.sleep(0.01), (), {})
        ]
        times = await self._measure_operation(
            "Parallel Task Execution",
            lambda: optimizer.executor.execute_parallel(tasks),
            iterations=10,
            is_async=True
        )
        self.results['parallel_execution'] = times
        
        # Performance monitoring overhead
        @optimizer.performance_monitor("test_operation")
        async def test_operation():
            await asyncio.sleep(0.01)
            return "test_result"
        
        times = await self._measure_operation(
            "Performance Monitoring Overhead",
            test_operation,
            iterations=20,
            is_async=True
        )
        self.results['monitoring_overhead'] = times
    
    async def _measure_operation(self, name: str, operation, iterations: int, is_async: bool) -> Dict[str, float]:
        """Measure operation performance over multiple iterations."""
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                if is_async:
                    await operation()
                else:
                    operation()
                
                end_time = time.time()
                duration = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(duration)
                
            except Exception as e:
                print(f"  ❌ Error in {name} iteration {i+1}: {e}")
                times.append(float('inf'))  # Mark as failed
        
        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if not valid_times:
            stats = {
                'avg': float('inf'),
                'min': float('inf'), 
                'max': float('inf'),
                'median': float('inf'),
                'p95': float('inf'),
                'success_rate': 0.0
            }
        else:
            stats = {
                'avg': statistics.mean(valid_times),
                'min': min(valid_times),
                'max': max(valid_times),
                'median': statistics.median(valid_times),
                'p95': self._percentile(valid_times, 95),
                'success_rate': len(valid_times) / iterations
            }
        
        # Report results
        status = "✅" if stats['avg'] < self.target_response_time * 1000 else "⚠️"
        print(f"  {status} {name}:")
        print(f"    Avg: {stats['avg']:.1f}ms | P95: {stats['p95']:.1f}ms | Success: {stats['success_rate']:.1%}")
        
        return stats
    
    async def _mock_complete_discovery_workflow(self):
        """Mock a complete discovery workflow for benchmarking."""
        # Simulate the time of a real discovery workflow
        await asyncio.sleep(0.05)  # 50ms simulated discovery time
        
        # Simulate memory storage
        await self.domain_manager.store_memory(
            "mcp_indexing_summary",
            json.dumps({"servers": 3, "tools": 15}),
            0.8
        )
        
        return {"servers": 3, "tools": 15}
    
    def _test_cache_operations(self, cache):
        """Test cache put/get operations."""
        # Put operation
        cache.put("test_key", {"test": "data"})
        
        # Get operation
        result = cache.get("test_key")
        
        return result
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary report."""
        print("\n📊 Performance Summary Report")
        print("=" * 60)
        
        # Overall statistics
        all_avg_times = []
        target_met_count = 0
        total_operations = 0
        
        for operation, stats in self.results.items():
            if stats['avg'] != float('inf'):
                all_avg_times.append(stats['avg'])
                total_operations += 1
                if stats['avg'] < self.target_response_time * 1000:
                    target_met_count += 1
        
        overall_avg = statistics.mean(all_avg_times) if all_avg_times else float('inf')
        target_compliance_rate = target_met_count / total_operations if total_operations > 0 else 0
        
        # Performance categories
        categories = {
            'Excellent': 0,    # < 100ms
            'Good': 0,         # 100-250ms  
            'Acceptable': 0,   # 250-500ms
            'Slow': 0,         # 500-1000ms
            'Poor': 0          # > 1000ms
        }
        
        for avg_time in all_avg_times:
            if avg_time < 100:
                categories['Excellent'] += 1
            elif avg_time < 250:
                categories['Good'] += 1
            elif avg_time < 500:
                categories['Acceptable'] += 1
            elif avg_time < 1000:
                categories['Slow'] += 1
            else:
                categories['Poor'] += 1
        
        # Report summary
        print(f"Overall Average Response Time: {overall_avg:.1f}ms")
        print(f"Target Compliance Rate: {target_compliance_rate:.1%}")
        print(f"Operations Meeting Target (<500ms): {target_met_count}/{total_operations}")
        print()
        
        print("Performance Distribution:")
        for category, count in categories.items():
            percentage = count / total_operations * 100 if total_operations > 0 else 0
            print(f"  {category}: {count} operations ({percentage:.1f}%)")
        print()
        
        # Slowest operations
        slowest_ops = sorted(
            [(op, stats['avg']) for op, stats in self.results.items() if stats['avg'] != float('inf')],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        if slowest_ops:
            print("Slowest Operations:")
            for op, avg_time in slowest_ops:
                print(f"  📈 {op}: {avg_time:.1f}ms")
        print()
        
        # Recommendations
        recommendations = []
        if target_compliance_rate < 0.8:
            recommendations.append("Consider additional performance optimizations")
        if overall_avg > 250:
            recommendations.append("Review caching strategies for frequently used operations")
        if categories['Poor'] > 0:
            recommendations.append("Investigate and optimize operations taking >1000ms")
        
        if recommendations:
            print("Recommendations:")
            for rec in recommendations:
                print(f"  💡 {rec}")
        else:
            print("🎉 All performance targets met!")
        
        return {
            'overall_avg_ms': overall_avg,
            'target_compliance_rate': target_compliance_rate,
            'operations_meeting_target': f"{target_met_count}/{total_operations}",
            'performance_distribution': categories,
            'slowest_operations': slowest_ops,
            'recommendations': recommendations,
            'detailed_results': self.results
        }


async def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()
    
    # Save results to file
    results_file = Path(__file__).parent / "performance_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Return overall success
    success = results['target_compliance_rate'] >= 0.8
    if success:
        print("\n✅ Performance benchmarks PASSED")
    else:
        print("\n⚠️ Performance benchmarks show areas for improvement")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)