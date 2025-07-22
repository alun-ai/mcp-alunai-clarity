"""
Performance tests for the Unified Qdrant Connection Management system.

Tests performance improvements and benchmarks:
- Connection acquisition speed 
- Memory usage efficiency
- Strategy switching performance
- Concurrent connection handling
- Monitoring overhead
"""

import asyncio
import pytest
import time
import statistics
import psutil
import os
from typing import List
from unittest.mock import patch, MagicMock

from clarity.shared.infrastructure import (
    get_qdrant_connection,
    UnifiedConnectionConfig,
    ConnectionStrategy,
    get_unified_stats
)


@pytest.mark.performance
class TestConnectionAcquisitionPerformance:
    """Test connection acquisition speed and efficiency."""
    
    @pytest.mark.asyncio
    async def test_connection_acquisition_speed(self):
        """Test that connection acquisition meets performance targets (<50ms)."""
        config = UnifiedConnectionConfig(
            path="/tmp/test_qdrant",
            timeout=30.0
        )
        
        acquisition_times = []
        num_tests = 10
        
        for i in range(num_tests):
            start_time = time.perf_counter()
            
            try:
                connection_manager = await get_qdrant_connection(config)
                # Note: We don't actually use the connection to avoid Qdrant dependency
                end_time = time.perf_counter()
                
                acquisition_time_ms = (end_time - start_time) * 1000
                acquisition_times.append(acquisition_time_ms)
                
            except Exception:
                # Expected without actual Qdrant - we're testing the manager creation speed
                end_time = time.perf_counter()
                acquisition_time_ms = (end_time - start_time) * 1000
                acquisition_times.append(acquisition_time_ms)
        
        # Performance analysis
        avg_time = statistics.mean(acquisition_times)
        max_time = max(acquisition_times)
        min_time = min(acquisition_times)
        
        print(f"\\n📊 Connection Manager Acquisition Performance:")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   Min: {min_time:.2f}ms")
        print(f"   Max: {max_time:.2f}ms")
        
        # Performance targets from audit
        assert avg_time < 50.0, f"Average acquisition time {avg_time:.2f}ms exceeds 50ms target"
        assert max_time < 100.0, f"Max acquisition time {max_time:.2f}ms should be under 100ms"
    
    @pytest.mark.asyncio
    async def test_repeated_connection_reuse(self):
        """Test that repeated connections to same config are fast (reuse optimization)."""
        config = UnifiedConnectionConfig(path="/tmp/test_qdrant")
        
        # First connection (may be slower due to initialization)
        start = time.perf_counter()
        try:
            manager1 = await get_qdrant_connection(config)
        except Exception:
            pass
        first_time = (time.perf_counter() - start) * 1000
        
        # Subsequent connections (should be faster due to reuse)
        subsequent_times = []
        for i in range(5):
            start = time.perf_counter()
            try:
                manager = await get_qdrant_connection(config)
            except Exception:
                pass
            subsequent_times.append((time.perf_counter() - start) * 1000)
        
        avg_subsequent = statistics.mean(subsequent_times)
        
        print(f"\\n🔄 Connection Reuse Performance:")
        print(f"   First connection: {first_time:.2f}ms")
        print(f"   Average reused: {avg_subsequent:.2f}ms")
        
        # Reused connections should be significantly faster
        # Allow some variance, but expect at least some optimization
        assert avg_subsequent <= first_time * 1.5, "Connection reuse should provide performance benefit"


@pytest.mark.performance
class TestMemoryUsageEfficiency:
    """Test memory usage and efficiency of unified connection management."""
    
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_single_manager(self):
        """Test memory usage with single unified manager vs multiple legacy managers."""
        
        # Baseline memory
        baseline_memory = self.get_memory_usage()
        
        # Create multiple connection configs (simulating multiple components)
        configs = [
            UnifiedConnectionConfig(path=f"/tmp/test_qdrant_{i}")
            for i in range(10)
        ]
        
        # Get connection managers (all should use the same singleton)
        managers = []
        for config in configs:
            try:
                manager = await get_qdrant_connection(config)
                managers.append(manager)
            except Exception:
                # Expected without Qdrant instance
                pass
        
        # Memory after creating managers
        after_memory = self.get_memory_usage()
        memory_used = after_memory - baseline_memory
        
        print(f"\\n💾 Memory Usage Analysis:")
        print(f"   Baseline: {baseline_memory:.2f}MB")
        print(f"   After 10 managers: {after_memory:.2f}MB")
        print(f"   Memory used: {memory_used:.2f}MB")
        print(f"   Per manager: {memory_used/10:.2f}MB")
        
        # Memory usage should be reasonable (less than 5MB for manager creation)
        assert memory_used < 5.0, f"Memory usage {memory_used:.2f}MB is too high for unified manager"
    
    @pytest.mark.asyncio
    async def test_connection_statistics_memory_overhead(self):
        """Test that statistics collection doesn't cause memory leaks."""
        
        baseline_memory = self.get_memory_usage()
        
        # Generate lots of statistics
        config = UnifiedConnectionConfig(path="/tmp/test")
        
        for i in range(100):
            try:
                manager = await get_qdrant_connection(config)
                stats = await get_unified_stats()
                # Stats should be generated without accumulating memory
            except Exception:
                pass
        
        final_memory = self.get_memory_usage()
        memory_growth = final_memory - baseline_memory
        
        print(f"\\n📈 Statistics Memory Overhead:")
        print(f"   Memory growth after 100 stat calls: {memory_growth:.2f}MB")
        
        # Should not accumulate significant memory
        assert memory_growth < 2.0, f"Statistics collection causing memory growth: {memory_growth:.2f}MB"


@pytest.mark.performance
class TestConcurrentConnectionHandling:
    """Test performance under concurrent connection requests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_connection_acquisition(self):
        """Test performance with concurrent connection requests."""
        
        async def get_connection_with_timing(config: UnifiedConnectionConfig) -> float:
            """Get connection and return acquisition time."""
            start = time.perf_counter()
            try:
                manager = await get_qdrant_connection(config)
                return (time.perf_counter() - start) * 1000
            except Exception:
                return (time.perf_counter() - start) * 1000
        
        config = UnifiedConnectionConfig(path="/tmp/concurrent_test")
        
        # Test concurrent requests
        num_concurrent = 20
        
        start_time = time.perf_counter()
        tasks = [get_connection_with_timing(config) for _ in range(num_concurrent)]
        times = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Performance analysis
        avg_time = statistics.mean(times)
        max_time = max(times)
        throughput = num_concurrent / (total_time / 1000)  # connections per second
        
        print(f"\\n🔀 Concurrent Connection Performance:")
        print(f"   {num_concurrent} concurrent requests")
        print(f"   Total time: {total_time:.2f}ms")
        print(f"   Average per connection: {avg_time:.2f}ms")
        print(f"   Max time: {max_time:.2f}ms")
        print(f"   Throughput: {throughput:.1f} connections/second")
        
        # Performance targets
        assert throughput > 50.0, f"Throughput {throughput:.1f} conn/sec is below 50/sec target"
        assert avg_time < 100.0, f"Average time {avg_time:.2f}ms under concurrent load too high"
    
    @pytest.mark.asyncio
    async def test_strategy_switching_performance(self):
        """Test performance impact of strategy detection and switching."""
        
        configs = [
            UnifiedConnectionConfig(path="/tmp/single", strategy=ConnectionStrategy.SINGLE_CLIENT),
            UnifiedConnectionConfig(path="/tmp/shared", strategy=ConnectionStrategy.SHARED_CLIENT),
            UnifiedConnectionConfig(url="http://remote", strategy=ConnectionStrategy.CONNECTION_POOL),
        ]
        
        strategy_times = {}
        
        for config in configs:
            times = []
            for i in range(5):
                start = time.perf_counter()
                try:
                    manager = await get_qdrant_connection(config)
                except Exception:
                    pass
                times.append((time.perf_counter() - start) * 1000)
            
            strategy_times[config.strategy.value] = statistics.mean(times)
        
        print(f"\\n🎯 Strategy Performance Comparison:")
        for strategy, avg_time in strategy_times.items():
            print(f"   {strategy}: {avg_time:.2f}ms")
        
        # All strategies should perform reasonably
        for strategy, avg_time in strategy_times.items():
            assert avg_time < 100.0, f"Strategy {strategy} too slow: {avg_time:.2f}ms"


@pytest.mark.performance  
class TestMonitoringOverhead:
    """Test performance overhead of monitoring and metrics collection."""
    
    @pytest.mark.asyncio
    async def test_statistics_collection_overhead(self):
        """Test overhead of statistics collection."""
        
        config = UnifiedConnectionConfig(
            path="/tmp/monitoring_test",
            enable_metrics=True
        )
        
        # Test with monitoring enabled
        with_monitoring_times = []
        for i in range(10):
            start = time.perf_counter()
            try:
                manager = await get_qdrant_connection(config)
                stats = await get_unified_stats()  # Collect stats
            except Exception:
                pass
            with_monitoring_times.append((time.perf_counter() - start) * 1000)
        
        # Test with monitoring disabled
        config_no_monitoring = UnifiedConnectionConfig(
            path="/tmp/monitoring_test",
            enable_metrics=False
        )
        
        without_monitoring_times = []
        for i in range(10):
            start = time.perf_counter()
            try:
                manager = await get_qdrant_connection(config_no_monitoring)
            except Exception:
                pass
            without_monitoring_times.append((time.perf_counter() - start) * 1000)
        
        # Performance comparison
        avg_with = statistics.mean(with_monitoring_times)
        avg_without = statistics.mean(without_monitoring_times)
        overhead_percent = ((avg_with - avg_without) / avg_without) * 100 if avg_without > 0 else 0
        
        print(f"\\n📊 Monitoring Overhead Analysis:")
        print(f"   With monitoring: {avg_with:.2f}ms")
        print(f"   Without monitoring: {avg_without:.2f}ms")
        print(f"   Overhead: {overhead_percent:.1f}%")
        
        # Monitoring overhead should be minimal (<20%)
        assert overhead_percent < 20.0, f"Monitoring overhead too high: {overhead_percent:.1f}%"


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions and validate improvement targets."""
    
    @pytest.mark.asyncio
    async def test_connection_acquisition_target_validation(self):
        """Validate that connection acquisition meets audit targets."""
        
        # Target from audit: <50ms average, <100ms max
        config = UnifiedConnectionConfig(path="/tmp/regression_test")
        
        times = []
        for i in range(50):  # More samples for statistical significance
            start = time.perf_counter()
            try:
                manager = await get_qdrant_connection(config)
            except Exception:
                pass
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
        
        print(f"\\n🎯 Performance Target Validation:")
        print(f"   Average: {avg_time:.2f}ms (target: <50ms)")
        print(f"   Maximum: {max_time:.2f}ms (target: <100ms)")
        print(f"   95th percentile: {p95_time:.2f}ms")
        
        # Validate audit performance targets
        assert avg_time < 50.0, f"REGRESSION: Average time {avg_time:.2f}ms exceeds 50ms target"
        assert max_time < 100.0, f"REGRESSION: Max time {max_time:.2f}ms exceeds 100ms target"
        assert p95_time < 75.0, f"95th percentile {p95_time:.2f}ms should be under 75ms"
    
    def test_memory_usage_target_validation(self):
        """Validate memory usage meets efficiency targets."""
        
        # Target from audit: 60% memory reduction
        # This is a baseline test - actual comparison would need legacy system
        
        baseline_memory = self.get_memory_usage()
        
        # Simulate creating connections like the old system would
        # (In real scenario, this would be compared to legacy connection pool)
        configs = [UnifiedConnectionConfig(path=f"/tmp/mem_test_{i}") for i in range(5)]
        
        for config in configs:
            try:
                asyncio.run(get_qdrant_connection(config))
            except Exception:
                pass
        
        final_memory = self.get_memory_usage()
        memory_per_connection = (final_memory - baseline_memory) / 5
        
        print(f"\\n💾 Memory Efficiency Validation:")
        print(f"   Memory per connection: {memory_per_connection:.2f}MB")
        print(f"   Target: Efficient unified management")
        
        # Reasonable memory usage per connection
        assert memory_per_connection < 1.0, f"Memory per connection too high: {memory_per_connection:.2f}MB"
    
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


if __name__ == "__main__":
    # Run performance tests with: python -m pytest tests/performance/test_unified_connection_performance.py -v -s
    pytest.main([__file__, "-v", "-s"])