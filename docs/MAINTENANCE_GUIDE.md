# Enhanced MCP Discovery System - Maintenance & Monitoring Guide

## Overview

This guide provides comprehensive instructions for maintaining, monitoring, and troubleshooting the Enhanced MCP Discovery System in production environments.

## Monitoring Dashboard

### Key Performance Indicators (KPIs)

#### Response Time Metrics
- **Target**: < 500ms for all discovery operations
- **Warning**: > 500ms average over 5 minutes
- **Critical**: > 1000ms average over 5 minutes

```python
# Monitor response times
analytics = await indexer.get_comprehensive_analytics()
avg_response_time = analytics['performance']['avg_response_time_ms']

if avg_response_time > 1000:
    print("🚨 CRITICAL: High response time detected")
elif avg_response_time > 500:
    print("⚠️ WARNING: Response time above target")
```

#### Cache Performance
- **Target**: > 70% cache hit rate
- **Warning**: < 30% cache hit rate over 1 hour
- **Critical**: < 10% cache hit rate over 1 hour

```python
# Monitor cache performance
cache_stats = indexer.performance_optimizer.cache.get_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
print(f"Cache size: {cache_stats['size']} entries")
```

#### Discovery Success Rate
- **Target**: > 95% successful discoveries
- **Warning**: < 80% success rate over 10 minutes
- **Critical**: < 50% success rate over 10 minutes

```python
# Monitor discovery success
discovery_status = analytics['discovery_status']
success_rate = discovery_status.get('success_rate', 0)
print(f"Discovery success rate: {success_rate:.1%}")
```

#### Memory Usage
- **Target**: < 200MB RSS memory
- **Warning**: > 500MB RSS memory
- **Critical**: > 1GB RSS memory

```python
# Monitor memory usage
memory_stats = indexer.performance_optimizer.get_memory_usage()
print(f"Memory usage: {memory_stats.get('rss_mb', 0):.1f} MB")
```

## Automated Monitoring Setup

### Production Monitor Integration

```python
from clarity.mcp.production_monitoring import get_production_monitor, console_alert_handler

# Initialize monitoring
monitor = get_production_monitor({
    'metrics_collection_interval': 60,    # 1 minute
    'alert_evaluation_interval': 30,      # 30 seconds
    'performance_report_interval': 300,   # 5 minutes
    'metrics_export_enabled': True,
    'metrics_export_path': '/var/log/mcp_discovery_metrics.jsonl',
    'alert_log_path': '/var/log/mcp_discovery_alerts.log'
})

# Add alert handlers
monitor.add_alert_handler(console_alert_handler)
monitor.add_alert_handler(email_alert_handler)  # Custom implementation
```

### Custom Alert Handlers

```python
def email_alert_handler(alert):
    \"\"\"Send email alerts for critical issues.\"\"\"
    if alert.severity == 'critical':
        send_email(
            to='ops@company.com',
            subject=f'MCP Discovery CRITICAL: {alert.metric_name}',
            body=f'Alert: {alert.message}\\nValue: {alert.current_value}\\nThreshold: {alert.threshold}'
        )

def slack_alert_handler(alert):
    \"\"\"Send Slack notifications for alerts.\"\"\"
    webhook_url = 'https://hooks.slack.com/...'
    payload = {
        'text': f'🚨 {alert.severity.upper()}: {alert.message}',
        'channel': '#ops-alerts'
    }
    requests.post(webhook_url, json=payload)

# Register handlers
monitor.add_alert_handler(email_alert_handler)
monitor.add_alert_handler(slack_alert_handler)
```

## Daily Maintenance Tasks

### 1. Health Check

```bash
#!/bin/bash
# daily_health_check.sh

echo "🔍 Daily MCP Discovery Health Check - $(date)"
echo "================================================"

# Run health validation
python3 -c "
import asyncio
from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.domain_manager import DomainManager

async def health_check():
    dm = DomainManager()
    await dm.initialize()
    indexer = MCPToolIndexer(dm)
    
    # Test discovery
    tools = await indexer.discover_and_index_tools()
    print(f'✅ Discovery working: {len(tools)} tools')
    
    # Test performance
    analytics = await indexer.get_comprehensive_analytics()
    response_time = analytics.get('performance', {}).get('avg_response_time_ms', 0)
    print(f'⚡ Average response time: {response_time:.1f}ms')
    
    # Test memory
    memory_stats = indexer.performance_optimizer.get_memory_usage()
    memory_mb = memory_stats.get('rss_mb', 0)
    print(f'💾 Memory usage: {memory_mb:.1f} MB')
    
    return response_time < 500 and memory_mb < 500

result = asyncio.run(health_check())
exit(0 if result else 1)
"

if [ $? -eq 0 ]; then
    echo "✅ Health check PASSED"
else
    echo "❌ Health check FAILED - investigate immediately"
    exit 1
fi
```

### 2. Cache Maintenance

```python
#!/usr/bin/env python3
# cache_maintenance.py

import asyncio
from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.domain_manager import DomainManager

async def daily_cache_maintenance():
    """Perform daily cache maintenance."""
    print("🧹 Daily Cache Maintenance")
    
    dm = DomainManager()
    await dm.initialize()
    indexer = MCPToolIndexer(dm)
    
    # Get cache stats before cleanup
    cache_stats_before = indexer.performance_optimizer.cache.get_stats()
    print(f"Cache before: {cache_stats_before['size']} entries, {cache_stats_before['hit_rate']:.1%} hit rate")
    
    # Clear stale cache entries (optional - cache has TTL)
    if cache_stats_before['hit_rate'] < 0.3:  # Low hit rate indicates stale data
        print("🗑️ Clearing cache due to low hit rate")
        indexer.performance_optimizer.clear_caches()
        await indexer.invalidate_discovery_cache()
    
    # Optimize performance settings based on usage
    recent_analytics = await indexer.get_comprehensive_analytics()
    avg_response_time = recent_analytics.get('performance', {}).get('avg_response_time_ms', 0)
    
    if avg_response_time > 750:  # Slow performance
        print("⚡ Adjusting performance settings for better speed")
        indexer.performance_optimizer.cache.default_ttl = 600  # Increase cache time
        indexer.performance_optimizer.max_parallel_tasks = 15   # More parallelism
    
    print("✅ Cache maintenance completed")

if __name__ == "__main__":
    asyncio.run(daily_cache_maintenance())
```

### 3. Log Rotation

```bash
#!/bin/bash
# log_rotation.sh

LOG_DIR="/var/log/mcp_discovery"
RETENTION_DAYS=30

echo "🗂️ Log Rotation - $(date)"

# Rotate metrics logs
if [ -f "$LOG_DIR/metrics.jsonl" ]; then
    mv "$LOG_DIR/metrics.jsonl" "$LOG_DIR/metrics.$(date +%Y%m%d).jsonl"
    touch "$LOG_DIR/metrics.jsonl"
    echo "📊 Rotated metrics log"
fi

# Rotate alert logs
if [ -f "$LOG_DIR/alerts.log" ]; then
    mv "$LOG_DIR/alerts.log" "$LOG_DIR/alerts.$(date +%Y%m%d).log"
    touch "$LOG_DIR/alerts.log"
    echo "🚨 Rotated alerts log"
fi

# Clean old logs
find "$LOG_DIR" -name "*.$(date -d "$RETENTION_DAYS days ago" +%Y%m%d)*" -delete
echo "🗑️ Cleaned logs older than $RETENTION_DAYS days"
```

## Weekly Maintenance Tasks

### 1. Performance Analysis

```python
#!/usr/bin/env python3
# weekly_performance_analysis.py

import asyncio
import json
from datetime import datetime, timedelta
from clarity.mcp.tool_indexer import MCPToolIndexer
from clarity.domain_manager import DomainManager

async def weekly_performance_analysis():
    """Generate weekly performance analysis report."""
    print("📊 Weekly Performance Analysis")
    print("=" * 50)
    
    dm = DomainManager()
    await dm.initialize()
    indexer = MCPToolIndexer(dm)
    
    # Get comprehensive analytics
    analytics = await indexer.get_comprehensive_analytics()
    
    # Performance summary
    print("\\n⚡ Performance Summary:")
    perf = analytics.get('performance', {})
    print(f"  Average Response Time: {perf.get('avg_response_time_ms', 0):.1f}ms")
    print(f"  Cache Hit Rate: {perf.get('cache_hit_rate', 0):.1%}")
    print(f"  Memory Usage: {perf.get('memory_usage_mb', 0):.1f} MB")
    
    # Discovery summary
    print("\\n🔍 Discovery Summary:")
    discovery = analytics.get('discovery_status', {})
    print(f"  Total Servers: {discovery.get('total_servers', 0)}")
    print(f"  Total Tools: {discovery.get('total_tools', 0)}")
    print(f"  Success Rate: {discovery.get('success_rate', 0):.1%}")
    
    # Learning summary
    print("\\n🧠 Learning Summary:")
    workflow = analytics.get('workflow_patterns', {})
    resource = analytics.get('resource_monitoring', {})
    print(f"  Workflow Patterns: {workflow.get('total_patterns', 0)}")
    print(f"  Resource Patterns: {resource.get('patterns_learned', 0)}")
    print(f"  Hook Integrations: {workflow.get('hook_interactions', 0)}")
    
    # Recommendations
    print("\\n💡 Recommendations:")
    if perf.get('avg_response_time_ms', 0) > 500:
        print("  - Consider optimizing server discovery timeouts")
        print("  - Review parallel processing settings")
    
    if perf.get('cache_hit_rate', 0) < 0.5:
        print("  - Increase cache TTL for stable servers")
        print("  - Review cache key strategies")
    
    if discovery.get('success_rate', 0) < 0.9:
        print("  - Review server configurations")
        print("  - Check network connectivity")
    
    # Save report
    report_file = f"/var/log/mcp_discovery/weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_file, 'w') as f:
        json.dump(analytics, f, indent=2)
    
    print(f"\\n📄 Report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(weekly_performance_analysis())
```

### 2. Pattern Learning Review

```python
#!/usr/bin/env python3
# pattern_learning_review.py

async def weekly_pattern_review():
    """Review and optimize learning patterns."""
    print("🧠 Weekly Pattern Learning Review")
    
    dm = DomainManager()
    await dm.initialize()
    indexer = MCPToolIndexer(dm)
    
    # Review workflow patterns
    workflow_analytics = await indexer.workflow_enhancer.get_pattern_analytics()
    print(f"\\n📋 Workflow Patterns: {workflow_analytics['total_patterns']}")
    
    if 'top_patterns' in workflow_analytics:
        print("  Top patterns by usage:")
        for pattern in workflow_analytics['top_patterns'][:5]:
            print(f"    - {pattern['pattern_type']}: {pattern['usage_count']} uses")
    
    # Review resource patterns
    resource_analytics = await indexer.resource_monitor.analyze_reference_usage_patterns()
    print(f"\\n📋 Resource Reference Patterns:")
    
    if 'most_successful_patterns' in resource_analytics:
        print("  Most successful patterns:")
        for pattern in resource_analytics['most_successful_patterns'][:5]:
            print(f"    - {pattern['pattern']}: {pattern['success_rate']:.1%} success")
    
    # Optimization recommendations
    recommendations = resource_analytics.get('recommendations', [])
    if recommendations:
        print("\\n💡 Pattern Optimization Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")

if __name__ == "__main__":
    asyncio.run(weekly_pattern_review())
```

## Monthly Maintenance Tasks

### 1. Dependency Updates

```bash
#!/bin/bash
# monthly_dependency_check.sh

echo "📦 Monthly Dependency Check - $(date)"
echo "====================================="

# Check Python dependencies
echo "🐍 Checking Python dependencies..."
pip list --outdated

# Check for security vulnerabilities
echo "🔒 Security vulnerability check..."
pip-audit

# Update dependencies (with caution)
echo "⬆️ Consider updating these packages:"
echo "   - Review changelogs before updating"
echo "   - Test in development environment first"
echo "   - Schedule maintenance window for updates"
```

### 2. Database Cleanup

```python
#!/usr/bin/env python3
# monthly_database_cleanup.py

async def monthly_database_cleanup():
    """Clean up old data and optimize database."""
    print("🗃️ Monthly Database Cleanup")
    
    dm = DomainManager()
    await dm.initialize()
    
    # Clean old memories (keep last 6 months)
    cutoff_date = datetime.now() - timedelta(days=180)
    
    # Example cleanup (adjust based on your domain manager implementation)
    old_memories = await dm.retrieve_memories(
        query="",
        created_before=cutoff_date,
        limit=1000
    )
    
    print(f"🗑️ Found {len(old_memories)} old memories to clean")
    
    # Archive or delete old memories
    for memory in old_memories:
        # Archive instead of delete for audit trail
        await dm.archive_memory(memory['id'])
    
    print("✅ Database cleanup completed")

if __name__ == "__main__":
    asyncio.run(monthly_database_cleanup())
```

## Troubleshooting Guide

### Common Issues and Solutions

#### High Response Times

**Symptoms**: Average response time > 1000ms

**Investigation**:
```python
# Check performance bottlenecks
report = indexer.performance_optimizer.get_performance_report()
slowest_ops = report['operation_stats']

for op, stats in slowest_ops.items():
    if stats['avg_duration_ms'] > 1000:
        print(f"🐌 Slow operation: {op} ({stats['avg_duration_ms']:.1f}ms)")
```

**Solutions**:
1. Increase parallel processing: `indexer.performance_optimizer.max_parallel_tasks = 15`
2. Reduce server discovery timeout: Configure shorter timeouts for slow servers
3. Enable more aggressive caching: `indexer.performance_optimizer.cache.default_ttl = 900`

#### Memory Leaks

**Symptoms**: Continuously increasing memory usage

**Investigation**:
```python
# Monitor memory growth
memory_stats = indexer.performance_optimizer.get_memory_usage()
print(f"Cache size: {memory_stats['cache_size']} entries")
print(f"Metrics count: {memory_stats['metrics_count']}")
```

**Solutions**:
1. Clear caches: `indexer.performance_optimizer.clear_caches()`
2. Reduce metrics retention: `indexer.performance_optimizer.max_metrics = 500`
3. Restart service if memory usage > 1GB

#### Discovery Failures

**Symptoms**: High discovery failure rate > 20%

**Investigation**:
```python
# Check server status
servers = await indexer.get_discovered_servers()
for name, config in servers.items():
    try:
        # Test server connectivity
        result = await indexer._test_server_connection(name, config)
        print(f"Server {name}: {'✅' if result else '❌'}")
    except Exception as e:
        print(f"Server {name}: ❌ {e}")
```

**Solutions**:
1. Update server configurations
2. Check network connectivity
3. Increase connection timeouts
4. Remove unreachable servers from configuration

#### Hook Integration Issues

**Symptoms**: Hook analyzer not working or slow

**Investigation**:
```bash
# Test hook analyzer directly
python clarity/mcp/hook_analyzer.py --help

# Check hook configuration
cat ~/.config/claude/hooks.json

# Test hook execution
python clarity/mcp/hook_analyzer.py --post-tool --tool Test --result "test" --success true --debug
```

**Solutions**:
1. Verify hook script permissions: `chmod +x clarity/mcp/hook_analyzer.py`
2. Update hook configuration paths
3. Check Python environment in hook configuration

### Emergency Procedures

#### System Unresponsive

1. **Immediate Action**: Restart the service
2. **Investigation**: Check logs for errors
3. **Recovery**: Clear all caches and restart with minimal configuration
4. **Prevention**: Implement health checks and auto-restart

#### High CPU Usage

1. **Immediate Action**: Reduce parallel processing
2. **Investigation**: Profile performance bottlenecks
3. **Recovery**: Optimize slow operations
4. **Prevention**: Set CPU usage alerts

#### Disk Space Issues

1. **Immediate Action**: Clean old logs and metrics
2. **Investigation**: Check log file sizes
3. **Recovery**: Implement log rotation
4. **Prevention**: Monitor disk usage

## Performance Optimization

### Production Tuning

```python
# High-performance production configuration
def configure_production_performance(indexer):
    \"\"\"Configure system for high-performance production use.\"\"\"
    
    # Optimize cache settings
    indexer.performance_optimizer.cache.max_size = 2000
    indexer.performance_optimizer.cache.default_ttl = 600  # 10 minutes
    
    # Increase parallel processing
    indexer.performance_optimizer.max_parallel_tasks = 20
    indexer.performance_optimizer.executor.timeout = 15.0
    
    # Optimize batch sizes
    indexer.performance_optimizer.batch_sizes = {
        'servers': 10,
        'tools': 50,
        'commands': 20,
        'patterns': 30
    }
    
    # Enable performance tracking
    indexer.performance_optimizer.cache_warmup_enabled = True
    
    print("⚡ Production performance configuration applied")
```

### Scaling Considerations

#### Horizontal Scaling
- Deploy multiple instances with shared domain manager
- Use load balancer for request distribution
- Implement cache synchronization between instances

#### Vertical Scaling
- Increase memory allocation for larger caches
- Use more CPU cores for parallel processing
- Optimize database performance

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup_mcp_data.sh

BACKUP_DIR="/backup/mcp_discovery/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

echo "💾 Backing up MCP Discovery data..."

# Backup configuration
cp -r ~/.config/claude "$BACKUP_DIR/claude_config"

# Backup logs
cp -r /var/log/mcp_discovery "$BACKUP_DIR/logs"

# Backup domain manager data (adjust based on implementation)
# pg_dump mcp_clarity > "$BACKUP_DIR/database.sql"

echo "✅ Backup completed: $BACKUP_DIR"
```

### Recovery Procedures

1. **Configuration Recovery**: Restore from backup and verify settings
2. **Data Recovery**: Restore domain manager data and verify integrity
3. **Cache Recovery**: Clear all caches and rebuild from configuration
4. **Validation**: Run full system validation after recovery

---

**Enhanced MCP Discovery System v3.0**  
*Maintenance & Monitoring Guide*  
*For production operations and system administration*