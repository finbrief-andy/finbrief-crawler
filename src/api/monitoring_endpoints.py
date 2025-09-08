#!/usr/bin/env python3
"""
Monitoring and Observability API Endpoints

REST API endpoints for system monitoring, health checks, metrics,
and operational insights.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.security import HTTPBearer
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from ..monitoring.metrics_collector import (
    metrics_collector,
    record_counter,
    record_gauge,
    record_histogram,
    record_timer
)
from ..monitoring.health_checks import (
    get_health_checker,
    check_system_health,
    quick_health_check,
    HealthStatus
)
from ..monitoring.logger import get_logger, get_logger_system
from ..auth.jwt_handler import get_current_user
from ..database.models import User

router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])
security = HTTPBearer()
logger = get_logger(__name__)

# Pydantic models for request/response validation

class MetricRequest(BaseModel):
    name: str
    value: float
    labels: Optional[Dict[str, str]] = None

class HealthCheckResponse(BaseModel):
    status: str
    uptime: float
    checks_passed: int
    checks_total: int
    timestamp: str

class SystemHealthResponse(BaseModel):
    status: str
    uptime: float
    uptime_formatted: str
    checks: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str

class MetricsResponse(BaseModel):
    counters: Dict[str, float]
    gauges: Dict[str, float]
    histograms: Dict[str, Dict[str, float]]
    timestamp: str

class LogStatsResponse(BaseModel):
    config: Dict[str, Any]
    loggers_count: int
    handlers: List[Dict[str, Any]]
    log_files: List[Dict[str, Any]]

# Health Check Endpoints

@router.get("/health", response_model=HealthCheckResponse)
async def get_health_status():
    """
    Get quick health status
    
    Returns a simple health check suitable for load balancers
    and monitoring systems that need fast responses.
    """
    try:
        health_summary = await quick_health_check()
        return HealthCheckResponse(**health_summary)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check unavailable")

@router.get("/health/detailed", response_model=SystemHealthResponse)
async def get_detailed_health_status(current_user: User = Depends(get_current_user)):
    """
    Get comprehensive health status
    
    Provides detailed information about all system components,
    resource usage, and potential issues.
    """
    try:
        health = await check_system_health()
        
        checks_data = []
        for check in health.checks:
            checks_data.append({
                "name": check.name,
                "status": check.status.value,
                "message": check.message,
                "response_time": check.response_time,
                "timestamp": check.timestamp.isoformat(),
                "metadata": check.metadata,
                "error": check.error
            })
        
        return SystemHealthResponse(
            status=health.status.value,
            uptime=health.uptime,
            uptime_formatted=health.summary["uptime_formatted"],
            checks=checks_data,
            summary=health.summary,
            timestamp=health.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/health/checks/{check_name}")
async def get_specific_health_check(
    check_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get results for a specific health check
    
    Allows monitoring of individual system components
    in detail.
    """
    try:
        checker = get_health_checker()
        result = await checker.run_check(check_name)
        
        return {
            "name": result.name,
            "status": result.status.value,
            "message": result.message,
            "response_time": result.response_time,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata,
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Specific health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Metrics Endpoints

@router.get("/metrics", response_model=MetricsResponse)
async def get_current_metrics(current_user: User = Depends(get_current_user)):
    """
    Get current application metrics
    
    Returns all collected metrics including counters, gauges,
    and histogram summaries.
    """
    try:
        metrics_data = metrics_collector.get_current_metrics()
        return MetricsResponse(**metrics_data)
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics unavailable: {str(e)}")

@router.get("/metrics/prometheus")
async def get_prometheus_metrics(current_user: User = Depends(get_current_user)):
    """
    Get metrics in Prometheus format
    
    Returns metrics in a format compatible with Prometheus
    monitoring system.
    """
    try:
        # This would typically be handled by the prometheus_client library
        # For now, return a simple text representation
        metrics_data = metrics_collector.get_current_metrics()
        
        prometheus_output = []
        
        # Counters
        for name, value in metrics_data["counters"].items():
            prometheus_output.append(f"# TYPE finbrief_{name.replace('.', '_')} counter")
            prometheus_output.append(f"finbrief_{name.replace('.', '_')} {value}")
        
        # Gauges
        for name, value in metrics_data["gauges"].items():
            prometheus_output.append(f"# TYPE finbrief_{name.replace('.', '_')} gauge")
            prometheus_output.append(f"finbrief_{name.replace('.', '_')} {value}")
        
        # Histograms (simplified)
        for name, hist_data in metrics_data["histograms"].items():
            base_name = f"finbrief_{name.replace('.', '_')}"
            prometheus_output.append(f"# TYPE {base_name} histogram")
            prometheus_output.append(f"{base_name}_count {hist_data['count']}")
            prometheus_output.append(f"{base_name}_sum {hist_data['sum']}")
        
        return Response(
            content="\n".join(prometheus_output),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Prometheus metrics unavailable: {str(e)}")

@router.get("/metrics/history/{metric_name}")
async def get_metric_history(
    metric_name: str,
    hours: int = Query(default=1, ge=1, le=24),
    current_user: User = Depends(get_current_user)
):
    """
    Get historical data for a specific metric
    
    Returns time series data for the specified metric
    over the requested time period.
    """
    try:
        history = metrics_collector.get_metric_history(metric_name, hours)
        
        return {
            "metric_name": metric_name,
            "time_range_hours": hours,
            "data_points": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Failed to get metric history: {e}")
        raise HTTPException(status_code=500, detail=f"Metric history unavailable: {str(e)}")

@router.post("/metrics/record")
async def record_custom_metric(
    metric_request: MetricRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Record a custom metric value
    
    Allows applications to submit custom metrics
    for monitoring and analysis.
    """
    try:
        # Record as gauge by default, could be extended to support different types
        record_gauge(metric_request.name, metric_request.value, metric_request.labels)
        
        return {
            "message": "Metric recorded successfully",
            "metric_name": metric_request.name,
            "value": metric_request.value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to record metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

# Performance Monitoring Endpoints

@router.get("/performance/summary")
async def get_performance_summary(current_user: User = Depends(get_current_user)):
    """
    Get system performance summary
    
    Returns key performance indicators including
    CPU, memory, disk usage, and application metrics.
    """
    try:
        perf_metrics = metrics_collector.get_performance_summary()
        app_metrics = metrics_collector.get_application_summary()
        
        return {
            "system": {
                "cpu_percent": perf_metrics.cpu_percent,
                "memory_percent": perf_metrics.memory_percent,
                "memory_used_mb": perf_metrics.memory_used_mb,
                "disk_usage_percent": perf_metrics.disk_usage_percent,
                "load_average": perf_metrics.load_average,
                "process_count": perf_metrics.process_count
            },
            "application": {
                "articles_processed": app_metrics.articles_processed,
                "api_requests": app_metrics.api_requests,
                "database_queries": app_metrics.database_queries,
                "cache_hit_rate": app_metrics.cache_hits / max(app_metrics.cache_hits + app_metrics.cache_misses, 1),
                "active_users": app_metrics.active_users,
                "errors_count": app_metrics.errors_count,
                "average_response_time": app_metrics.response_time_avg
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=f"Performance data unavailable: {str(e)}")

@router.get("/performance/alerts")
async def get_performance_alerts(current_user: User = Depends(get_current_user)):
    """
    Get performance-based alerts
    
    Returns current performance issues that require attention.
    """
    try:
        alerts = []
        
        # Get performance metrics
        perf_metrics = metrics_collector.get_performance_summary()
        app_metrics = metrics_collector.get_application_summary()
        
        # Check for performance issues
        if perf_metrics.cpu_percent > 80:
            alerts.append({
                "type": "performance",
                "severity": "warning" if perf_metrics.cpu_percent < 90 else "critical",
                "message": f"High CPU usage: {perf_metrics.cpu_percent:.1f}%",
                "metric": "cpu_percent",
                "value": perf_metrics.cpu_percent,
                "threshold": 80
            })
        
        if perf_metrics.memory_percent > 85:
            alerts.append({
                "type": "performance", 
                "severity": "warning" if perf_metrics.memory_percent < 95 else "critical",
                "message": f"High memory usage: {perf_metrics.memory_percent:.1f}%",
                "metric": "memory_percent",
                "value": perf_metrics.memory_percent,
                "threshold": 85
            })
        
        if perf_metrics.disk_usage_percent > 90:
            alerts.append({
                "type": "performance",
                "severity": "critical",
                "message": f"Low disk space: {perf_metrics.disk_usage_percent:.1f}% used",
                "metric": "disk_usage_percent", 
                "value": perf_metrics.disk_usage_percent,
                "threshold": 90
            })
        
        if app_metrics.errors_count > 10:  # More than 10 errors in collection period
            alerts.append({
                "type": "application",
                "severity": "warning",
                "message": f"High error rate: {app_metrics.errors_count} errors",
                "metric": "errors_count",
                "value": app_metrics.errors_count,
                "threshold": 10
            })
        
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Performance alerts unavailable: {str(e)}")

# Logging Endpoints

@router.get("/logs/stats", response_model=LogStatsResponse)
async def get_log_statistics(current_user: User = Depends(get_current_user)):
    """
    Get logging system statistics
    
    Returns information about log configuration, active loggers,
    and log file status.
    """
    try:
        logger_system = get_logger_system()
        stats = logger_system.get_log_stats()
        return LogStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get log statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Log statistics unavailable: {str(e)}")

@router.get("/logs/recent")
async def get_recent_log_entries(
    level: str = Query(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    lines: int = Query(default=100, ge=10, le=1000),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent log entries
    
    Returns the most recent log entries filtered by level.
    """
    try:
        from pathlib import Path
        import json
        
        log_file = Path("logs/finbrief.log")
        if not log_file.exists():
            return {
                "message": "Log file not found",
                "entries": [],
                "count": 0
            }
        
        entries = []
        with open(log_file, 'r') as f:
            # Read last N lines
            file_lines = f.readlines()
            recent_lines = file_lines[-lines:] if len(file_lines) > lines else file_lines
            
            for line in recent_lines:
                try:
                    log_entry = json.loads(line.strip())
                    if log_entry.get('level', '') >= level:
                        entries.append(log_entry)
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue
        
        return {
            "entries": entries,
            "count": len(entries),
            "level_filter": level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent log entries: {e}")
        raise HTTPException(status_code=500, detail=f"Recent logs unavailable: {str(e)}")

# System Information Endpoints

@router.get("/system/info")
async def get_system_information(current_user: User = Depends(get_current_user)):
    """
    Get comprehensive system information
    
    Returns detailed information about the system,
    application version, and configuration.
    """
    try:
        import platform
        import sys
        import psutil
        from pathlib import Path
        
        # System information
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version
            },
            "resources": {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
                "boot_time": boot_time.isoformat(),
                "uptime_hours": (datetime.now() - boot_time).total_seconds() / 3600
            },
            "application": {
                "name": "FinBrief",
                "version": "1.0.0",  # Could be read from version file
                "python_path": sys.executable,
                "working_directory": str(Path.cwd()),
                "process_id": os.getpid(),
                "environment": os.getenv("ENVIRONMENT", "development")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"Failed to get system information: {e}")
        raise HTTPException(status_code=500, detail=f"System information unavailable: {str(e)}")

@router.get("/system/processes")
async def get_process_information(current_user: User = Depends(get_current_user)):
    """
    Get process information
    
    Returns information about running processes and
    resource usage.
    """
    try:
        import psutil
        
        current_process = psutil.Process()
        
        # Get child processes
        children = []
        try:
            for child in current_process.children(recursive=True):
                try:
                    children.append({
                        "pid": child.pid,
                        "name": child.name(),
                        "status": child.status(),
                        "cpu_percent": child.cpu_percent(),
                        "memory_mb": child.memory_info().rss / (1024 * 1024),
                        "create_time": datetime.fromtimestamp(child.create_time()).isoformat()
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        process_info = {
            "main_process": {
                "pid": current_process.pid,
                "name": current_process.name(),
                "status": current_process.status(),
                "cpu_percent": current_process.cpu_percent(),
                "memory_mb": current_process.memory_info().rss / (1024 * 1024),
                "thread_count": current_process.num_threads(),
                "create_time": datetime.fromtimestamp(current_process.create_time()).isoformat(),
                "cwd": current_process.cwd()
            },
            "child_processes": children,
            "system_stats": {
                "total_processes": len(psutil.pids()),
                "cpu_count": psutil.cpu_count(),
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return process_info
        
    except Exception as e:
        logger.error(f"Failed to get process information: {e}")
        raise HTTPException(status_code=500, detail=f"Process information unavailable: {str(e)}")

# Operational Endpoints

@router.post("/metrics/start")
async def start_metrics_collection(
    prometheus_port: int = Query(default=8000),
    current_user: User = Depends(get_current_user)
):
    """
    Start metrics collection
    
    Starts the metrics collection system and optionally
    the Prometheus metrics server.
    """
    try:
        metrics_collector.start_collection(prometheus_port)
        
        return {
            "message": "Metrics collection started",
            "prometheus_port": prometheus_port,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start metrics collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start metrics: {str(e)}")

@router.post("/metrics/stop")
async def stop_metrics_collection(current_user: User = Depends(get_current_user)):
    """
    Stop metrics collection
    
    Stops the metrics collection system.
    """
    try:
        await metrics_collector.stop_collection()
        
        return {
            "message": "Metrics collection stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop metrics collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop metrics: {str(e)}")

# Dashboard Data Endpoint

@router.get("/dashboard")
async def get_dashboard_data(current_user: User = Depends(get_current_user)):
    """
    Get comprehensive dashboard data
    
    Returns all monitoring data needed for a
    comprehensive monitoring dashboard.
    """
    try:
        # Get health status
        health_summary = await quick_health_check()
        
        # Get performance metrics
        perf_metrics = metrics_collector.get_performance_summary()
        app_metrics = metrics_collector.get_application_summary()
        
        # Get recent alerts
        alerts_response = await get_performance_alerts(current_user)
        alerts = alerts_response["alerts"]
        
        dashboard_data = {
            "health": {
                "status": health_summary["status"],
                "uptime": health_summary["uptime"],
                "checks_passed": health_summary["checks_passed"],
                "checks_total": health_summary["checks_total"]
            },
            "performance": {
                "cpu_percent": perf_metrics.cpu_percent,
                "memory_percent": perf_metrics.memory_percent,
                "disk_usage_percent": perf_metrics.disk_usage_percent,
                "load_average": perf_metrics.load_average[0] if perf_metrics.load_average else 0
            },
            "application": {
                "articles_processed": app_metrics.articles_processed,
                "api_requests": app_metrics.api_requests,
                "active_users": app_metrics.active_users,
                "errors_count": app_metrics.errors_count,
                "cache_hit_rate": app_metrics.cache_hits / max(app_metrics.cache_hits + app_metrics.cache_misses, 1)
            },
            "alerts": {
                "count": len(alerts),
                "critical": len([a for a in alerts if a["severity"] == "critical"]),
                "warning": len([a for a in alerts if a["severity"] == "warning"]),
                "recent": alerts[:5]  # Top 5 alerts
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data unavailable: {str(e)}")

import os