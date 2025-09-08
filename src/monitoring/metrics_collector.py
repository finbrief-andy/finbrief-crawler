#!/usr/bin/env python3
"""
Metrics Collection Service

Comprehensive metrics collection system for monitoring application performance,
business metrics, and system health indicators.
"""

import time
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import Lock
import json
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]
    timestamp: datetime


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    articles_processed: int
    strategies_generated: int
    api_requests: int
    database_queries: int
    cache_hits: int
    cache_misses: int
    active_users: int
    errors_count: int
    response_time_avg: float
    timestamp: datetime


class MetricsCollector:
    """Comprehensive metrics collection and storage"""
    
    def __init__(self, retention_hours: int = 24, collection_interval: int = 60):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to keep metrics in memory
            collection_interval: How often to collect metrics (seconds)
        """
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe storage
        self._lock = Lock()
        self._metrics_storage: Dict[str, deque] = defaultdict(lambda: deque())
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Prometheus integration
        self.prometheus_enabled = PROMETHEUS_AVAILABLE
        if self.prometheus_enabled:
            self._init_prometheus_metrics()
        
        # Collection state
        self._collecting = False
        self._collection_task = None
        
        # Performance tracking
        self._start_time = datetime.now()
        self._last_network_stats = None
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            # System metrics
            self.prom_cpu_usage = Gauge('finbrief_cpu_usage_percent', 'CPU usage percentage')
            self.prom_memory_usage = Gauge('finbrief_memory_usage_percent', 'Memory usage percentage')
            self.prom_disk_usage = Gauge('finbrief_disk_usage_percent', 'Disk usage percentage')
            
            # Application metrics
            self.prom_articles_total = Counter('finbrief_articles_processed_total', 'Total articles processed')
            self.prom_api_requests = Counter('finbrief_api_requests_total', 'Total API requests', ['method', 'endpoint'])
            self.prom_response_time = Histogram('finbrief_response_time_seconds', 'Response time in seconds', ['endpoint'])
            self.prom_errors = Counter('finbrief_errors_total', 'Total errors', ['type'])
            self.prom_active_users = Gauge('finbrief_active_users', 'Number of active users')
            
            # Database metrics
            self.prom_db_queries = Counter('finbrief_db_queries_total', 'Total database queries')
            self.prom_db_response_time = Histogram('finbrief_db_response_time_seconds', 'Database response time')
            
            # Cache metrics
            self.prom_cache_hits = Counter('finbrief_cache_hits_total', 'Cache hits')
            self.prom_cache_misses = Counter('finbrief_cache_misses_total', 'Cache misses')
            
            self.logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Prometheus metrics: {e}")
            self.prometheus_enabled = False
    
    def start_collection(self, prometheus_port: int = 8000):
        """Start metrics collection"""
        if self._collecting:
            self.logger.warning("Metrics collection already started")
            return
        
        self._collecting = True
        
        # Start Prometheus server if enabled
        if self.prometheus_enabled and prometheus_port:
            try:
                start_http_server(prometheus_port)
                self.logger.info(f"Prometheus metrics server started on port {prometheus_port}")
            except Exception as e:
                self.logger.error(f"Failed to start Prometheus server: {e}")
        
        # Start collection loop
        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self._collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        try:
            while self._collecting:
                await self._collect_system_metrics()
                await self._cleanup_old_metrics()
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Metrics collection loop cancelled")
        except Exception as e:
            self.logger.error(f"Metrics collection loop error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge('system.cpu.usage_percent', cpu_percent)
            
            if self.prometheus_enabled:
                self.prom_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_gauge('system.memory.usage_percent', memory.percent)
            self.record_gauge('system.memory.used_mb', memory.used / 1024 / 1024)
            
            if self.prometheus_enabled:
                self.prom_memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_gauge('system.disk.usage_percent', disk_percent)
            
            if self.prometheus_enabled:
                self.prom_disk_usage.set(disk_percent)
            
            # Network stats
            network = psutil.net_io_counters()
            if self._last_network_stats:
                bytes_sent_delta = network.bytes_sent - self._last_network_stats.bytes_sent
                bytes_recv_delta = network.bytes_recv - self._last_network_stats.bytes_recv
                
                self.record_gauge('system.network.bytes_sent_rate', bytes_sent_delta)
                self.record_gauge('system.network.bytes_recv_rate', bytes_recv_delta)
            
            self._last_network_stats = network
            
            # Process count
            process_count = len(psutil.pids())
            self.record_gauge('system.process.count', process_count)
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self.record_gauge('system.load.avg_1m', load_avg[0])
                self.record_gauge('system.load.avg_5m', load_avg[1])
                self.record_gauge('system.load.avg_15m', load_avg[2])
            except (AttributeError, OSError):
                # Not available on all platforms
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter metric"""
        with self._lock:
            self._counters[name] += value
            
            # Store time series data
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            )
            self._metrics_storage[name].append(point)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric"""
        with self._lock:
            self._gauges[name] = value
            
            # Store time series data
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            )
            self._metrics_storage[name].append(point)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric"""
        with self._lock:
            self._histograms[name].append(value)
            
            # Store time series data
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            )
            self._metrics_storage[name].append(point)
    
    def record_timer(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, labels)
    
    # Application-specific metric methods
    
    def record_article_processed(self, source: str = None):
        """Record article processing"""
        labels = {'source': source} if source else {}
        self.record_counter('app.articles.processed', 1.0, labels)
        
        if self.prometheus_enabled:
            self.prom_articles_total.inc()
    
    def record_api_request(self, method: str, endpoint: str, response_time: float = None):
        """Record API request"""
        labels = {'method': method, 'endpoint': endpoint}
        self.record_counter('app.api.requests', 1.0, labels)
        
        if response_time is not None:
            self.record_histogram('app.api.response_time', response_time, labels)
        
        if self.prometheus_enabled:
            self.prom_api_requests.labels(method=method, endpoint=endpoint).inc()
            if response_time is not None:
                self.prom_response_time.labels(endpoint=endpoint).observe(response_time)
    
    def record_error(self, error_type: str, details: str = None):
        """Record application error"""
        labels = {'type': error_type}
        if details:
            labels['details'] = details[:100]  # Limit label size
        
        self.record_counter('app.errors', 1.0, labels)
        
        if self.prometheus_enabled:
            self.prom_errors.labels(type=error_type).inc()
    
    def record_database_query(self, operation: str, duration: float = None):
        """Record database query"""
        labels = {'operation': operation}
        self.record_counter('app.database.queries', 1.0, labels)
        
        if duration is not None:
            self.record_histogram('app.database.response_time', duration, labels)
        
        if self.prometheus_enabled:
            self.prom_db_queries.inc()
            if duration is not None:
                self.prom_db_response_time.observe(duration)
    
    def record_cache_hit(self, cache_type: str = 'default'):
        """Record cache hit"""
        labels = {'type': cache_type}
        self.record_counter('app.cache.hits', 1.0, labels)
        
        if self.prometheus_enabled:
            self.prom_cache_hits.inc()
    
    def record_cache_miss(self, cache_type: str = 'default'):
        """Record cache miss"""
        labels = {'type': cache_type}
        self.record_counter('app.cache.misses', 1.0, labels)
        
        if self.prometheus_enabled:
            self.prom_cache_misses.inc()
    
    def set_active_users(self, count: int):
        """Set active users count"""
        self.record_gauge('app.users.active', count)
        
        if self.prometheus_enabled:
            self.prom_active_users.set(count)
    
    # Query methods
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {
                    name: {
                        'count': len(values),
                        'sum': sum(values),
                        'avg': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0
                    }
                    for name, values in self._histograms.items()
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metric history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            if name not in self._metrics_storage:
                return []
            
            return [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'labels': point.labels,
                    'metadata': point.metadata
                }
                for point in self._metrics_storage[name]
                if point.timestamp >= cutoff_time
            ]
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """Get current performance metrics summary"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Load average with fallback
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]
            
            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=len(psutil.pids()),
                load_average=load_avg,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return PerformanceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                load_average=[0.0, 0.0, 0.0],
                timestamp=datetime.now()
            )
    
    def get_application_summary(self) -> ApplicationMetrics:
        """Get current application metrics summary"""
        with self._lock:
            return ApplicationMetrics(
                articles_processed=int(self._counters.get('app.articles.processed', 0)),
                strategies_generated=int(self._counters.get('app.strategies.generated', 0)),
                api_requests=int(self._counters.get('app.api.requests', 0)),
                database_queries=int(self._counters.get('app.database.queries', 0)),
                cache_hits=int(self._counters.get('app.cache.hits', 0)),
                cache_misses=int(self._counters.get('app.cache.misses', 0)),
                active_users=int(self._gauges.get('app.users.active', 0)),
                errors_count=int(self._counters.get('app.errors', 0)),
                response_time_avg=sum(self._histograms.get('app.api.response_time', [0])) / 
                                len(self._histograms.get('app.api.response_time', [1])),
                timestamp=datetime.now()
            )
    
    async def _cleanup_old_metrics(self):
        """Clean up old metric data points"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            for name, points in self._metrics_storage.items():
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                
                # Clean up histogram data
                if name in self._histograms:
                    # Keep only recent histogram values
                    recent_count = max(1000, len(points))  # Keep at least 1000 values
                    if len(self._histograms[name]) > recent_count:
                        self._histograms[name] = self._histograms[name][-recent_count:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        perf = self.get_performance_summary()
        app = self.get_application_summary()
        
        # Determine health based on thresholds
        health_issues = []
        
        if perf.cpu_percent > 80:
            health_issues.append(f"High CPU usage: {perf.cpu_percent:.1f}%")
        
        if perf.memory_percent > 85:
            health_issues.append(f"High memory usage: {perf.memory_percent:.1f}%")
        
        if perf.disk_usage_percent > 90:
            health_issues.append(f"High disk usage: {perf.disk_usage_percent:.1f}%")
        
        if app.errors_count > 10:  # Last collection period
            health_issues.append(f"High error rate: {app.errors_count} errors")
        
        # Overall status
        if not health_issues:
            status = "healthy"
        elif len(health_issues) < 2:
            status = "warning"
        else:
            status = "critical"
        
        uptime = datetime.now() - self._start_time
        
        return {
            'status': status,
            'uptime_seconds': int(uptime.total_seconds()),
            'issues': health_issues,
            'performance': perf.__dict__,
            'application': app.__dict__,
            'prometheus_enabled': self.prometheus_enabled,
            'collection_active': self._collecting,
            'timestamp': datetime.now().isoformat()
        }


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_histogram(self.name, duration, self.labels)


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Convenience functions for easy access

def record_counter(name: str, value: float = 1.0, labels: Dict[str, str] = None):
    """Record a counter metric"""
    metrics_collector.record_counter(name, value, labels)


def record_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Record a gauge metric"""
    metrics_collector.record_gauge(name, value, labels)


def record_histogram(name: str, value: float, labels: Dict[str, str] = None):
    """Record a histogram metric"""
    metrics_collector.record_histogram(name, value, labels)


def record_timer(name: str, labels: Dict[str, str] = None):
    """Context manager for timing operations"""
    return metrics_collector.record_timer(name, labels)


# Application-specific convenience functions

def record_article_processed(source: str = None):
    """Record article processing"""
    metrics_collector.record_article_processed(source)


def record_api_request(method: str, endpoint: str, response_time: float = None):
    """Record API request"""
    metrics_collector.record_api_request(method, endpoint, response_time)


def record_error(error_type: str, details: str = None):
    """Record application error"""
    metrics_collector.record_error(error_type, details)


def record_database_query(operation: str, duration: float = None):
    """Record database query"""
    metrics_collector.record_database_query(operation, duration)


def record_cache_hit(cache_type: str = 'default'):
    """Record cache hit"""
    metrics_collector.record_cache_hit(cache_type)


def record_cache_miss(cache_type: str = 'default'):
    """Record cache miss"""
    metrics_collector.record_cache_miss(cache_type)


def set_active_users(count: int):
    """Set active users count"""
    metrics_collector.set_active_users(count)