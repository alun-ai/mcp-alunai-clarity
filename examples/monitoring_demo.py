#!/usr/bin/env python3
"""
Demonstration of Alunai Clarity performance monitoring and metrics collection.

This script shows how to use the monitoring system to track performance,
collect metrics, and generate health reports.
"""

import asyncio
import time
import random
from pathlib import Path

# Add clarity to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from clarity.shared.monitoring import (
    get_metrics_collector, 
    performance_monitor,
    get_telemetry_reporter
)
from clarity.shared.monitoring.performance import PerformanceProfiler
from clarity.shared.monitoring.telemetry import HealthChecker


def simulate_operations():
    """Simulate various system operations with different performance characteristics."""
    print("🔄 Simulating system operations...")
    
    metrics = get_metrics_collector()
    
    # Simulate counter metrics
    for i in range(100):
        metrics.increment_counter("requests_total", tags={"endpoint": "store_memory"})
        metrics.increment_counter("requests_total", tags={"endpoint": "retrieve_memories"})
        
        if random.random() < 0.05:  # 5% error rate
            metrics.increment_counter("errors_total", tags={"type": "memory_error"})
    
    # Simulate gauge metrics
    metrics.set_gauge("active_connections", random.randint(5, 20))
    metrics.set_gauge("memory_usage_mb", random.uniform(100, 500))
    metrics.set_gauge("cpu_usage_percent", random.uniform(10, 80))
    
    # Simulate histogram metrics (response times)
    for _ in range(200):
        # Normal operations (fast)
        duration = random.gauss(0.1, 0.02)  # 100ms ± 20ms
        metrics.record_histogram("response_time", max(0.001, duration), tags={"operation": "fast_operation"})
        
        # Some slow operations
        if random.random() < 0.1:  # 10% slow operations
            duration = random.gauss(2.0, 0.5)  # 2s ± 500ms
            metrics.record_histogram("response_time", max(0.001, duration), tags={"operation": "slow_operation"})
    
    print("✅ Simulated 300+ operations with mixed performance characteristics")


@performance_monitor.measure("demo.database_query")
def simulate_database_query(query_type: str = "simple"):
    """Simulate a database query with performance monitoring."""
    # Simulate query processing time
    if query_type == "simple":
        time.sleep(random.uniform(0.01, 0.05))  # 10-50ms
    elif query_type == "complex":
        time.sleep(random.uniform(0.1, 0.3))    # 100-300ms
    else:
        time.sleep(random.uniform(0.5, 1.0))    # 500ms-1s
    
    # Simulate occasional failures
    if random.random() < 0.02:  # 2% failure rate
        raise Exception(f"Database error in {query_type} query")
    
    return f"Query result for {query_type}"


@performance_monitor.measure("demo.memory_operation")
async def simulate_memory_operation(operation: str):
    """Simulate async memory operations."""
    # Simulate async processing
    await asyncio.sleep(random.uniform(0.05, 0.15))
    
    if operation == "store":
        return {"memory_id": f"mem_{random.randint(1000, 9999)}"}
    elif operation == "retrieve":
        return [{"id": f"mem_{i}", "content": f"Content {i}"} for i in range(5)]
    else:
        return {"status": "completed"}


def demonstrate_performance_profiling():
    """Demonstrate advanced performance profiling."""
    print("\n🔍 Demonstrating performance profiling...")
    
    profiler = PerformanceProfiler()
    
    # Start profiling a complex operation
    profiler.start_profiling("complex_workflow")
    
    # Simulate workflow steps
    time.sleep(0.1)  # Step 1: Setup
    profiler.checkpoint("complex_workflow", "setup_complete")
    
    time.sleep(0.2)  # Step 2: Data processing
    profiler.checkpoint("complex_workflow", "data_processing_complete")
    
    time.sleep(0.15) # Step 3: Finalization
    profiler.checkpoint("complex_workflow", "finalization_complete")
    
    # End profiling and get results
    results = profiler.end_profiling("complex_workflow")
    
    print(f"   Profile: {results['profile_name']}")
    print(f"   Total duration: {results['total_duration']:.3f}s")
    print(f"   Checkpoints: {results['checkpoint_count']}")
    
    for checkpoint in results['checkpoints']:
        print(f"     • {checkpoint['name']}: {checkpoint['duration']:.3f}s")
    
    print("✅ Performance profiling completed")


def demonstrate_telemetry_reporting():
    """Demonstrate telemetry and health reporting."""
    print("\n📊 Generating telemetry reports...")
    
    reporter = get_telemetry_reporter()
    
    # Generate health report
    health_report = reporter.generate_health_report()
    
    print(f"   System Status: {health_report['health_status']}")
    print(f"   Uptime: {health_report['system_info']['uptime_human']}")
    print(f"   Components: {len(health_report['component_health'])}")
    print(f"   Alerts: {len(health_report['alerts'])}")
    
    if health_report['alerts']:
        print("   ⚠️  Active alerts:")
        for alert in health_report['alerts'][:3]:  # Show first 3
            print(f"     • {alert['severity']}: {alert['message']}")
    
    # Export reports
    metrics_file = reporter.export_metrics_json()
    health_file = reporter.export_health_report()
    
    print(f"   📄 Metrics exported: {metrics_file.name}")
    print(f"   📄 Health report exported: {health_file.name}")
    
    # Performance insights
    insights = reporter.get_performance_insights()
    
    print(f"   💡 Performance recommendations: {len(insights['recommendations'])}")
    print(f"   ⚡ Optimization opportunities: {len(insights['optimization_opportunities'])}")
    
    if insights['recommendations']:
        print("   Top recommendations:")
        for rec in insights['recommendations'][:2]:
            print(f"     • {rec['type']}: {rec['recommendation']}")
    
    print("✅ Telemetry reporting completed")


def demonstrate_health_checking():
    """Demonstrate system health checking."""
    print("\n🏥 Performing health checks...")
    
    health_checker = HealthChecker()
    
    # Perform health check
    health_status = health_checker.check_system_health()
    
    print(f"   Overall Status: {health_status['overall_status']}")
    print(f"   Health Checks: {len(health_status['checks'])}")
    print(f"   Alerts: {len(health_status['alerts'])}")
    
    # Show check results
    passed_checks = sum(1 for status in health_status['checks'].values() if status == "passed")
    failed_checks = len(health_status['checks']) - passed_checks
    
    print(f"   ✅ Passed: {passed_checks}")
    print(f"   ❌ Failed: {failed_checks}")
    
    # Alert summary
    alert_summary = health_checker.get_alert_summary()
    print(f"   🚨 Critical: {alert_summary['critical_count']}")
    print(f"   ⚠️  Warnings: {alert_summary['warning_count']}")
    
    print("✅ Health checking completed")


async def run_async_operations():
    """Run async operations to generate realistic metrics."""
    print("\n⚡ Running async operations...")
    
    # Run multiple async operations concurrently
    tasks = []
    
    operations = ['store', 'retrieve', 'update', 'delete']
    for i in range(50):
        operation = random.choice(operations)
        task = simulate_memory_operation(operation)
        tasks.append(task)
    
    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    print(f"   ✅ Successful operations: {successful}")
    print(f"   ❌ Failed operations: {failed}")


def main():
    """Main demonstration function."""
    print("🚀 Alunai Clarity Monitoring System Demo")
    print("=" * 50)
    
    # Step 1: Simulate basic operations
    simulate_operations()
    
    # Step 2: Run monitored synchronous operations
    print("\n🔧 Running monitored operations...")
    for i in range(20):
        try:
            query_type = random.choice(['simple', 'complex', 'slow'])
            result = simulate_database_query(query_type)
        except Exception:
            pass  # Expected for demonstration
    
    # Step 3: Run async operations
    asyncio.run(run_async_operations())
    
    # Step 4: Performance profiling
    demonstrate_performance_profiling()
    
    # Step 5: Show current metrics
    print("\n📈 Current metrics summary:")
    metrics = get_metrics_collector()
    current_metrics = metrics.get_metrics()
    
    print(f"   Counters: {len(current_metrics['counters'])}")
    print(f"   Gauges: {len(current_metrics['gauges'])}")  
    print(f"   Histograms: {len(current_metrics['histograms'])}")
    print(f"   Operations tracked: {len(current_metrics['operations'])}")
    
    # Show top operations
    top_ops = metrics.get_top_operations(3, "count")
    if top_ops:
        print("   Top operations by count:")
        for op in top_ops:
            print(f"     • {op['operation']}: {op['count']} ops, {op['success_rate']:.1%} success")
    
    # Step 6: Telemetry reporting
    demonstrate_telemetry_reporting()
    
    # Step 7: Health checking
    demonstrate_health_checking()
    
    # Step 8: Export Prometheus metrics
    print("\n🔄 Exporting Prometheus metrics...")
    prometheus_data = metrics.export_prometheus()
    
    prometheus_file = Path("./data/telemetry/metrics.prom")
    prometheus_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(prometheus_file, 'w') as f:
        f.write(prometheus_data)
    
    print(f"   📄 Prometheus metrics exported: {prometheus_file}")
    print(f"   📏 Metrics size: {len(prometheus_data)} characters")
    
    print("\n" + "=" * 50)
    print("🎉 Monitoring system demonstration completed!")
    print("\nKey features demonstrated:")
    print("  • Automatic performance monitoring via decorators")
    print("  • Metrics collection (counters, gauges, histograms)")
    print("  • Performance profiling with checkpoints")
    print("  • Health checking and alerting")
    print("  • Telemetry reporting and export")
    print("  • Prometheus metrics integration")
    
    print(f"\nGenerated reports in: ./data/telemetry/")


if __name__ == "__main__":
    main()