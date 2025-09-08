"""
Monitoring API endpoints for FinBrief application.
Provides health checks, metrics, and observability endpoints.
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from scripts.main import get_current_user
from src.database.models_migration import User
from src.monitoring.health import get_health_monitor, check_system_health, check_all_health
from src.monitoring.metrics import get_metrics
from src.monitoring.logger import get_logger


# Pydantic models for API responses
class HealthCheckResponse(BaseModel):
    name: str
    status: str
    message: str
    duration_ms: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class SystemHealthResponse(BaseModel):
    status: str
    message: str
    uptime_seconds: float
    timestamp: datetime
    checks: Dict[str, Any]
    details: Dict[str, Any]


class MetricsSummaryResponse(BaseModel):
    timestamp: datetime
    crawler: Dict[str, Any]
    api: Dict[str, Any]
    database: Dict[str, Any]
    nlp: Dict[str, Any]
    system: Dict[str, Any]


class LogsFilterRequest(BaseModel):
    level: Optional[str] = Field(None, description="Log level filter")
    component: Optional[str] = Field(None, description="Component filter")
    start_time: Optional[datetime] = Field(None, description="Start time for log range")
    end_time: Optional[datetime] = Field(None, description="End time for log range")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of log entries")


# Create router
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Initialize logger for this module
logger = get_logger("monitoring.api")


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """
    Get overall system health status.
    Quick health check that doesn't run individual component checks.
    """
    try:
        health_data = await check_system_health()
        return SystemHealthResponse(**health_data)
    except Exception as e:
        logger.error("Failed to get system health", exception=e)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health/detailed")
async def get_detailed_health(
    run_checks: bool = Query(True, description="Whether to run fresh health checks")
):
    """
    Get detailed health status with individual component checks.
    """
    try:
        if run_checks:
            health_data = await check_all_health()
        else:
            monitor = get_health_monitor()
            health_data = {
                "summary": monitor.get_system_health_summary(),
                "individual_checks": {
                    name: result.__dict__ for name, result in monitor.last_results.items()
                }
            }
        
        return health_data
    except Exception as e:
        logger.error("Failed to get detailed health", exception=e)
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {str(e)}")


@router.get("/health/{component}")
async def get_component_health(component: str):
    """
    Get health status for a specific component.
    """
    try:
        monitor = get_health_monitor()
        result = await monitor.run_single_check(component)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Health check '{component}' not found")
        
        return HealthCheckResponse(
            name=result.name,
            status=result.status.value,
            message=result.message,
            duration_ms=result.duration_ms,
            timestamp=result.timestamp,
            details=result.details
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get component health", exception=e, component=component)
        raise HTTPException(status_code=500, detail=f"Component health check failed: {str(e)}")


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary(
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive metrics summary.
    Requires authentication.
    """
    try:
        metrics = get_metrics()
        summary = metrics.get_metrics_summary()
        
        return MetricsSummaryResponse(
            timestamp=datetime.now(),
            **summary
        )
    except Exception as e:
        logger.error("Failed to get metrics summary", exception=e, user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Metrics summary failed: {str(e)}")


@router.get("/metrics/raw")
async def get_raw_metrics(
    hours: int = Query(1, ge=1, le=24, description="Hours of metrics to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """
    Get raw metrics data for the specified time period.
    Admin-only endpoint.
    """
    if current_user.role.value not in ["admin", "system"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        metrics = get_metrics()
        all_metrics = metrics.collector.get_all_metrics()
        recent_points = metrics.collector.get_recent_metrics(hours=hours)
        
        return {
            "current_metrics": all_metrics,
            "recent_points": [
                {
                    "name": point.name,
                    "value": point.value,
                    "timestamp": point.timestamp.isoformat(),
                    "labels": point.labels,
                    "metric_type": point.metric_type
                }
                for point in recent_points
            ],
            "time_range_hours": hours,
            "total_points": len(recent_points)
        }
    except Exception as e:
        logger.error("Failed to get raw metrics", exception=e, user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Raw metrics retrieval failed: {str(e)}")


@router.get("/metrics/performance")
async def get_performance_metrics(
    component: Optional[str] = Query(None, description="Filter by component"),
    current_user: User = Depends(get_current_user)
):
    """
    Get performance-specific metrics (timers, response times, etc.).
    """
    try:
        metrics = get_metrics()
        
        # Get timer statistics for various operations
        performance_data = {
            "crawler_performance": {
                "bloomberg": metrics.collector.get_timer_stats("crawler_duration_seconds", source="bloomberg"),
                "reuters": metrics.collector.get_timer_stats("crawler_duration_seconds", source="reuters"),
                "marketwatch": metrics.collector.get_timer_stats("crawler_duration_seconds", source="marketwatch"),
                "vietstock": metrics.collector.get_timer_stats("crawler_duration_seconds", source="vietstock")
            },
            "api_performance": {
                "average_response_time": metrics.collector.get_timer_stats("api_request_duration_seconds"),
                "endpoint_performance": {
                    "/api/news": metrics.collector.get_timer_stats("api_request_duration_seconds", endpoint="/api/news"),
                    "/api/analysis": metrics.collector.get_timer_stats("api_request_duration_seconds", endpoint="/api/analysis"),
                    "/api/strategies": metrics.collector.get_timer_stats("api_request_duration_seconds", endpoint="/api/strategies")
                }
            },
            "database_performance": {
                "select_operations": metrics.collector.get_timer_stats("database_operation_duration_seconds", operation="SELECT"),
                "insert_operations": metrics.collector.get_timer_stats("database_operation_duration_seconds", operation="INSERT"),
                "update_operations": metrics.collector.get_timer_stats("database_operation_duration_seconds", operation="UPDATE")
            },
            "nlp_performance": {
                "processing_time": metrics.collector.get_timer_stats("nlp_processing_duration_seconds"),
                "model_load_time": metrics.collector.get_timer_stats("nlp_model_load_duration_seconds")
            },
            "strategy_performance": {
                "generation_time": metrics.collector.get_timer_stats("strategy_generation_duration_seconds")
            }
        }
        
        # Filter by component if specified
        if component and component in performance_data:
            return {component: performance_data[component]}
        
        return performance_data
        
    except Exception as e:
        logger.error("Failed to get performance metrics", exception=e, user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")


@router.post("/metrics/record")
async def record_custom_metric(
    metric_name: str,
    metric_type: str = Query(..., regex="^(counter|gauge|timer|histogram)$"),
    value: float = Query(...),
    labels: Optional[Dict[str, str]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Record a custom metric.
    Admin-only endpoint for testing and manual metric recording.
    """
    if current_user.role.value not in ["admin", "system"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        metrics = get_metrics()
        labels = labels or {}
        
        if metric_type == "counter":
            metrics.collector.increment_counter(metric_name, int(value), **labels)
        elif metric_type == "gauge":
            metrics.collector.set_gauge(metric_name, value, **labels)
        elif metric_type == "timer":
            metrics.collector.record_timer(metric_name, value, **labels)
        elif metric_type == "histogram":
            metrics.collector.record_histogram(metric_name, value, **labels)
        
        logger.info("Custom metric recorded", 
                   metric_name=metric_name, 
                   metric_type=metric_type, 
                   value=value,
                   user_id=current_user.id)
        
        return {
            "status": "success",
            "message": f"Metric '{metric_name}' recorded",
            "metric_name": metric_name,
            "metric_type": metric_type,
            "value": value,
            "labels": labels
        }
        
    except Exception as e:
        logger.error("Failed to record custom metric", exception=e, user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Metric recording failed: {str(e)}")


@router.get("/logs")
async def get_application_logs(
    level: Optional[str] = Query(None, description="Log level filter (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    component: Optional[str] = Query(None, description="Component filter"),
    hours: int = Query(1, ge=1, le=24, description="Hours of logs to retrieve"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries"),
    current_user: User = Depends(get_current_user)
):
    """
    Get application logs with filtering options.
    Admin-only endpoint.
    """
    if current_user.role.value not in ["admin", "system"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # This is a basic implementation - in production you'd want to read from log files
        # or use a proper log aggregation system
        
        from pathlib import Path
        import json
        
        logs = []
        log_dir = Path("logs")
        
        if log_dir.exists():
            # Read from structured log file
            log_file = log_dir / "finbrief.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Parse JSON log entries and apply filters
                    for line in reversed(lines[-limit:]):  # Get recent entries
                        try:
                            log_entry = json.loads(line.strip())
                            
                            # Apply level filter
                            if level and log_entry.get("level") != level.upper():
                                continue
                            
                            # Apply component filter
                            if component and component not in log_entry.get("logger", ""):
                                continue
                            
                            # Apply time filter
                            log_time = datetime.fromisoformat(log_entry.get("timestamp", ""))
                            cutoff_time = datetime.now() - timedelta(hours=hours)
                            if log_time < cutoff_time:
                                continue
                            
                            logs.append(log_entry)
                            
                        except (json.JSONDecodeError, ValueError):
                            continue  # Skip malformed log entries
        
        return {
            "logs": logs[:limit],
            "total_entries": len(logs),
            "filters": {
                "level": level,
                "component": component,
                "hours": hours,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error("Failed to get application logs", exception=e, user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Log retrieval failed: {str(e)}")


@router.get("/status")
async def get_monitoring_status():
    """
    Get monitoring system status (public endpoint).
    """
    try:
        from pathlib import Path
        
        # Check if monitoring components are working
        logs_dir = Path("logs")
        metrics_dir = logs_dir / "metrics"
        
        return {
            "monitoring_active": True,
            "logs_directory_exists": logs_dir.exists(),
            "metrics_directory_exists": metrics_dir.exists(),
            "health_checks_available": len(get_health_monitor().health_checkers),
            "metrics_collectors_active": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get monitoring status", exception=e)
        raise HTTPException(status_code=500, detail=f"Monitoring status check failed: {str(e)}")


@router.post("/test")
async def test_monitoring_system(
    current_user: User = Depends(get_current_user)
):
    """
    Test the monitoring system by generating sample metrics and logs.
    Admin-only endpoint for testing.
    """
    if current_user.role.value not in ["admin", "system"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Record test metrics
        metrics = get_metrics()
        metrics.record_news_crawled("test_source", 10)
        metrics.record_api_request("/api/test", "GET", 200, 0.15)
        metrics.record_database_operation("SELECT", "test_table", 0.05, 5)
        
        # Generate test logs
        logger.info("Monitoring test initiated", user_id=current_user.id, test_type="manual")
        logger.warning("Test warning message", component="test")
        logger.debug("Test debug message", details={"test": True})
        
        # Run a quick health check
        monitor = get_health_monitor()
        health_result = await monitor.run_single_check("system_resources")
        
        return {
            "status": "success",
            "message": "Monitoring system test completed",
            "test_results": {
                "metrics_recorded": True,
                "logs_generated": True,
                "health_check": {
                    "status": health_result.status.value,
                    "duration_ms": health_result.duration_ms
                } if health_result else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Monitoring test failed", exception=e, user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Monitoring test failed: {str(e)}")


# Health check endpoint for load balancers (no auth required)
@router.get("/ping")
async def ping():
    """Simple health check endpoint for load balancers and uptime monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "finbrief-monitoring"
    }