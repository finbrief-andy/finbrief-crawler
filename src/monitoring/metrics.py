"""
Comprehensive metrics collection system for FinBrief application.
Tracks performance, usage, and system health metrics.
"""
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
import os


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: str  # counter, gauge, histogram, timer


class MetricsCollector:
    """
    Comprehensive metrics collection system with in-memory storage and persistence.
    """
    
    def __init__(self, retention_hours: int = 24, flush_interval: int = 60):
        self.retention_hours = retention_hours
        self.flush_interval = flush_interval
        
        # In-memory metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Aggregated statistics
        self.statistics: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background flush thread
        self.flush_thread = None
        self.running = False
        
        # Metrics storage directory
        self.metrics_dir = Path("logs/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_background_flush()
    
    def start_background_flush(self):
        """Start background thread for periodic metrics flushing"""
        if self.flush_thread is not None:
            return
        
        self.running = True
        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()
    
    def stop(self):
        """Stop the metrics collector and flush remaining data"""
        self.running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5)
        self._flush_metrics()
    
    def _background_flush(self):
        """Background thread function for periodic flushing"""
        while self.running:
            time.sleep(self.flush_interval)
            self._flush_metrics()
    
    def increment_counter(self, name: str, value: int = 1, **labels):
        """Increment a counter metric"""
        with self.lock:
            key = self._get_metric_key(name, labels)
            self.counters[key] += value
            self._add_metric_point(name, value, "counter", labels)
    
    def set_gauge(self, name: str, value: float, **labels):
        """Set a gauge metric value"""
        with self.lock:
            key = self._get_metric_key(name, labels)
            self.gauges[key] = value
            self._add_metric_point(name, value, "gauge", labels)
    
    def record_timer(self, name: str, duration: float, **labels):
        """Record a timer metric (duration in seconds)"""
        with self.lock:
            key = self._get_metric_key(name, labels)
            self.timers[key].append(duration)
            # Keep only recent timer values
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
            self._add_metric_point(name, duration, "timer", labels)
    
    def record_histogram(self, name: str, value: float, **labels):
        """Record a histogram metric"""
        with self.lock:
            key = self._get_metric_key(name, labels)
            self.histograms[key].append(value)
            # Keep only recent histogram values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            self._add_metric_point(name, value, "histogram", labels)
    
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate a unique key for a metric with labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _add_metric_point(self, name: str, value: Union[int, float], metric_type: str, labels: Dict[str, str]):
        """Add a metric point to the time series"""
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            metric_type=metric_type
        )
        
        key = self._get_metric_key(name, labels)
        self.metrics[key].append(metric_point)
    
    def get_counter_value(self, name: str, **labels) -> int:
        """Get current counter value"""
        key = self._get_metric_key(name, labels)
        return self.counters.get(key, 0)
    
    def get_gauge_value(self, name: str, **labels) -> float:
        """Get current gauge value"""
        key = self._get_metric_key(name, labels)
        return self.gauges.get(key, 0.0)
    
    def get_timer_stats(self, name: str, **labels) -> Dict[str, float]:
        """Get timer statistics (min, max, avg, p95, p99)"""
        key = self._get_metric_key(name, labels)
        values = self.timers.get(key, [])
        
        if not values:
            return {"count": 0}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_histogram_stats(self, name: str, **labels) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._get_metric_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return {"count": 0}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self.lock:
            result = {
                "timestamp": datetime.now().isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {},
                "histograms": {}
            }
            
            # Add timer statistics
            for key in self.timers:
                result["timers"][key] = self.get_timer_stats("", **{})
            
            # Add histogram statistics  
            for key in self.histograms:
                result["histograms"][key] = self.get_histogram_stats("", **{})
            
            return result
    
    def get_recent_metrics(self, hours: int = 1) -> List[MetricPoint]:
        """Get metrics from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = []
        
        with self.lock:
            for metric_deque in self.metrics.values():
                for metric_point in metric_deque:
                    if metric_point.timestamp >= cutoff_time:
                        recent_metrics.append(metric_point)
        
        return sorted(recent_metrics, key=lambda x: x.timestamp)
    
    def _flush_metrics(self):
        """Flush metrics to persistent storage"""
        try:
            timestamp = datetime.now()
            filename = f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.metrics_dir / filename
            
            metrics_data = {
                "timestamp": timestamp.isoformat(),
                "metrics": self.get_all_metrics(),
                "recent_points": [asdict(point) for point in self.get_recent_metrics(hours=1)]
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            # Clean up old metric files
            self._cleanup_old_files()
            
        except Exception as e:
            print(f"Failed to flush metrics: {e}")
    
    def _cleanup_old_files(self):
        """Remove old metric files beyond retention period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            
            for file_path in self.metrics_dir.glob("metrics_*.json"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()
                    
        except Exception as e:
            print(f"Failed to cleanup old metric files: {e}")


# Application-specific metrics
class FinBriefMetrics:
    """
    Application-specific metrics for FinBrief system.
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
    
    # News Crawler Metrics
    def record_news_crawled(self, source: str, count: int):
        """Record number of news articles crawled"""
        self.collector.increment_counter("news_articles_crawled_total", count, source=source)
    
    def record_crawl_duration(self, source: str, duration: float):
        """Record crawler execution time"""
        self.collector.record_timer("crawler_duration_seconds", duration, source=source)
    
    def record_crawl_error(self, source: str, error_type: str):
        """Record crawler errors"""
        self.collector.increment_counter("crawler_errors_total", 1, source=source, error_type=error_type)
    
    # API Metrics
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.collector.increment_counter("api_requests_total", 1, endpoint=endpoint, method=method, status=str(status_code))
        self.collector.record_timer("api_request_duration_seconds", duration, endpoint=endpoint, method=method)
    
    def record_api_error(self, endpoint: str, error_type: str):
        """Record API errors"""
        self.collector.increment_counter("api_errors_total", 1, endpoint=endpoint, error_type=error_type)
    
    # Database Metrics
    def record_database_operation(self, operation: str, table: str, duration: float, count: int = 1):
        """Record database operation metrics"""
        self.collector.increment_counter("database_operations_total", count, operation=operation, table=table)
        self.collector.record_timer("database_operation_duration_seconds", duration, operation=operation, table=table)
    
    def record_database_connection_pool(self, active: int, idle: int, total: int):
        """Record database connection pool metrics"""
        self.collector.set_gauge("database_connections_active", active)
        self.collector.set_gauge("database_connections_idle", idle)
        self.collector.set_gauge("database_connections_total", total)
    
    # NLP Processing Metrics
    def record_nlp_processing(self, processor: str, operation: str, duration: float, text_length: int):
        """Record NLP processing metrics"""
        self.collector.record_timer("nlp_processing_duration_seconds", duration, processor=processor, operation=operation)
        self.collector.record_histogram("nlp_text_length_characters", text_length, processor=processor, operation=operation)
    
    def record_nlp_model_load(self, model_name: str, duration: float):
        """Record NLP model loading time"""
        self.collector.record_timer("nlp_model_load_duration_seconds", duration, model=model_name)
    
    # Strategy Generation Metrics
    def record_strategy_generation(self, model: str, duration: float, confidence: float):
        """Record strategy generation metrics"""
        self.collector.record_timer("strategy_generation_duration_seconds", duration, model=model)
        self.collector.record_histogram("strategy_confidence_score", confidence, model=model)
    
    def record_strategy_error(self, model: str, error_type: str):
        """Record strategy generation errors"""
        self.collector.increment_counter("strategy_generation_errors_total", 1, model=model, error_type=error_type)
    
    # System Metrics
    def record_memory_usage(self, component: str, usage_mb: float):
        """Record memory usage"""
        self.collector.set_gauge("memory_usage_mb", usage_mb, component=component)
    
    def record_system_health(self, component: str, healthy: bool):
        """Record system health status"""
        self.collector.set_gauge("system_health_status", 1.0 if healthy else 0.0, component=component)
    
    # Vector Search Metrics
    def record_vector_search(self, operation: str, duration: float, result_count: int):
        """Record vector search metrics"""
        self.collector.record_timer("vector_search_duration_seconds", duration, operation=operation)
        self.collector.record_histogram("vector_search_results_count", result_count, operation=operation)
    
    def record_embedding_generation(self, model: str, batch_size: int, duration: float):
        """Record embedding generation metrics"""
        self.collector.record_timer("embedding_generation_duration_seconds", duration, model=model)
        self.collector.record_histogram("embedding_batch_size", batch_size, model=model)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all application metrics"""
        return {
            "crawler": {
                "total_articles_crawled": sum(
                    self.collector.get_counter_value("news_articles_crawled_total", source=source) 
                    for source in ["bloomberg", "reuters", "marketwatch", "vietstock"]
                ),
                "crawler_performance": {
                    source: self.collector.get_timer_stats("crawler_duration_seconds", source=source)
                    for source in ["bloomberg", "reuters", "marketwatch", "vietstock"]
                }
            },
            "api": {
                "total_requests": sum(
                    self.collector.get_counter_value("api_requests_total", endpoint=endpoint, method=method, status=status)
                    for endpoint in ["/api/news", "/api/analysis", "/api/strategies"]
                    for method in ["GET", "POST"]
                    for status in ["200", "400", "500"]
                ),
                "average_response_time": self.collector.get_timer_stats("api_request_duration_seconds")
            },
            "database": {
                "operations": sum(
                    self.collector.get_counter_value("database_operations_total", operation=op, table=table)
                    for op in ["SELECT", "INSERT", "UPDATE", "DELETE"]
                    for table in ["news", "analysis", "strategies"]
                ),
                "connection_pool": {
                    "active": self.collector.get_gauge_value("database_connections_active"),
                    "idle": self.collector.get_gauge_value("database_connections_idle"),
                    "total": self.collector.get_gauge_value("database_connections_total")
                }
            },
            "nlp": {
                "processing_performance": self.collector.get_timer_stats("nlp_processing_duration_seconds"),
                "text_length_distribution": self.collector.get_histogram_stats("nlp_text_length_characters")
            },
            "system": {
                "health_status": {
                    component: bool(self.collector.get_gauge_value("system_health_status", component=component))
                    for component in ["crawler", "api", "database", "nlp", "strategy"]
                }
            }
        }


# Global metrics instance
_global_metrics = None

def get_metrics() -> FinBriefMetrics:
    """Get or create global metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = FinBriefMetrics()
    return _global_metrics


if __name__ == "__main__":
    # Test metrics system
    metrics = get_metrics()
    
    # Test crawler metrics
    metrics.record_news_crawled("bloomberg", 25)
    metrics.record_crawl_duration("bloomberg", 2.5)
    
    # Test API metrics
    metrics.record_api_request("/api/news", "GET", 200, 0.15)
    metrics.record_api_request("/api/news", "GET", 200, 0.12)
    
    # Test database metrics
    metrics.record_database_operation("SELECT", "news", 0.05, 25)
    metrics.record_database_connection_pool(5, 10, 15)
    
    # Test NLP metrics
    metrics.record_nlp_processing("enhanced", "sentiment", 0.8, 500)
    
    # Test strategy metrics
    metrics.record_strategy_generation("gpt-4", 1.2, 0.85)
    
    # Test system metrics
    metrics.record_memory_usage("crawler", 256.5)
    metrics.record_system_health("api", True)
    
    # Get metrics summary
    summary = metrics.get_metrics_summary()
    print(json.dumps(summary, indent=2))
    
    # Stop metrics collection
    metrics.collector.stop()
    
    print("Metrics test completed. Check logs/metrics/ directory for output files.")