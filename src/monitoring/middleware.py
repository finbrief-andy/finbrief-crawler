"""
Monitoring middleware for automatic metrics collection and logging.
Tracks API requests, database operations, and system performance.
"""
import time
import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from sqlalchemy import event
from sqlalchemy.engine import Engine

from src.monitoring.metrics import get_metrics
from src.monitoring.logger import get_api_logger, get_database_logger


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic monitoring of HTTP requests.
    """
    
    def __init__(self, app, track_body: bool = False, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.track_body = track_body
        self.exclude_paths = exclude_paths or ["/docs", "/openapi.json", "/favicon.ico"]
        self.metrics = get_metrics()
        self.logger = get_api_logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process HTTP request and track metrics"""
        
        # Skip monitoring for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        self.logger.info(
            f"Request started: {method} {path}",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error = e
            # Record error metrics
            self.metrics.record_api_error(path, type(e).__name__)
            self.logger.error(
                f"Request failed: {method} {path}",
                exception=e,
                request_id=request_id,
                method=method,
                path=path
            )
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            if response:
                status_code = response.status_code
                
                # Record metrics
                self.metrics.record_api_request(path, method, status_code, duration)
                
                # Log response
                self.logger.info(
                    f"Request completed: {method} {path} -> {status_code}",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=duration * 1000,
                    response_size=response.headers.get("content-length", "unknown")
                )
                
                # Add monitoring headers to response
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration:.4f}"
            
            else:
                # Log error completion
                self.logger.error(
                    f"Request errored: {method} {path}",
                    request_id=request_id,
                    method=method,
                    path=path,
                    duration_ms=duration * 1000,
                    error_type=type(error).__name__ if error else "unknown"
                )
        
        return response


class DatabaseMonitoringMixin:
    """
    Mixin for automatic database operation monitoring.
    Add this to your database session or connection class.
    """
    
    def __init__(self):
        self.metrics = get_metrics()
        self.logger = get_database_logger()
        self._setup_sqlalchemy_monitoring()
    
    def _setup_sqlalchemy_monitoring(self):
        """Set up SQLAlchemy event listeners for database monitoring"""
        
        @event.listens_for(Engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Track when database queries start"""
            context._query_start_time = time.time()
            context._query_statement = statement[:200]  # Truncate long statements
        
        @event.listens_for(Engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Track when database queries complete"""
            if hasattr(context, '_query_start_time'):
                duration = time.time() - context._query_start_time
                
                # Determine operation type
                operation = statement.strip().split()[0].upper() if statement else "UNKNOWN"
                
                # Try to extract table name
                table = "unknown"
                if "FROM" in statement.upper():
                    parts = statement.upper().split("FROM")[1].strip().split()
                    if parts:
                        table = parts[0].split('.')[0]  # Remove schema prefix
                elif "INTO" in statement.upper():
                    parts = statement.upper().split("INTO")[1].strip().split()
                    if parts:
                        table = parts[0].split('.')[0]
                elif "UPDATE" in statement.upper():
                    parts = statement.upper().split("UPDATE")[1].strip().split()
                    if parts:
                        table = parts[0].split('.')[0]
                
                # Record metrics
                self.metrics.record_database_operation(operation, table, duration)
                
                # Log slow queries
                if duration > 1.0:  # Log queries taking more than 1 second
                    self.logger.warning(
                        f"Slow database query detected",
                        operation=operation,
                        table=table,
                        duration_seconds=duration,
                        statement=context._query_statement
                    )
                else:
                    self.logger.debug(
                        f"Database query executed",
                        operation=operation,
                        table=table,
                        duration_seconds=duration
                    )


@asynccontextmanager
async def monitor_operation(operation_name: str, component: str = "unknown", **context):
    """
    Context manager for monitoring arbitrary operations.
    
    Usage:
        async with monitor_operation("news_crawling", component="crawler", source="bloomberg"):
            # Your operation here
            pass
    """
    metrics = get_metrics()
    logger = get_api_logger()
    
    start_time = time.time()
    
    logger.info(f"Operation started: {operation_name}", component=component, **context)
    
    try:
        yield
        
        # Operation completed successfully
        duration = time.time() - start_time
        
        # Record as timer metric
        metrics.collector.record_timer(f"{component}_operation_duration_seconds", duration, operation=operation_name)
        
        logger.info(
            f"Operation completed: {operation_name}",
            component=component,
            duration_seconds=duration,
            status="success",
            **context
        )
        
    except Exception as e:
        # Operation failed
        duration = time.time() - start_time
        
        # Record error metric
        metrics.collector.increment_counter(f"{component}_operation_errors_total", 1, operation=operation_name, error_type=type(e).__name__)
        
        logger.error(
            f"Operation failed: {operation_name}",
            exception=e,
            component=component,
            duration_seconds=duration,
            status="error",
            **context
        )
        
        raise


class PerformanceTracker:
    """
    Utility class for tracking performance of specific operations.
    """
    
    def __init__(self):
        self.metrics = get_metrics()
        self.active_operations = {}
    
    def start_operation(self, operation_id: str, operation_name: str, **context):
        """Start tracking an operation"""
        self.active_operations[operation_id] = {
            "name": operation_name,
            "start_time": time.time(),
            "context": context
        }
    
    def end_operation(self, operation_id: str, success: bool = True, **additional_context):
        """End tracking an operation"""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations.pop(operation_id)
        duration = time.time() - operation["start_time"]
        
        # Merge contexts
        context = {**operation["context"], **additional_context}
        
        # Record metrics
        component = context.get("component", "unknown")
        self.metrics.collector.record_timer(
            f"{component}_operation_duration_seconds",
            duration,
            operation=operation["name"],
            status="success" if success else "error"
        )
        
        if not success:
            self.metrics.collector.increment_counter(
                f"{component}_operation_errors_total",
                1,
                operation=operation["name"]
            )
    
    def get_active_operations(self) -> dict:
        """Get currently active operations"""
        current_time = time.time()
        return {
            op_id: {
                "name": op_data["name"],
                "duration_so_far": current_time - op_data["start_time"],
                "context": op_data["context"]
            }
            for op_id, op_data in self.active_operations.items()
        }


# Global performance tracker instance
_performance_tracker = None

def get_performance_tracker() -> PerformanceTracker:
    """Get or create global performance tracker"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker


# Decorator for automatic function monitoring
def monitor_function(operation_name: str = None, component: str = "unknown"):
    """
    Decorator for automatic function performance monitoring.
    
    Usage:
        @monitor_function("news_processing", component="crawler")
        def process_news_article(article):
            # Your function here
            pass
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            async with monitor_operation(op_name, component=component):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            import asyncio
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # For sync functions, we need to handle the async context manager differently
            metrics = get_metrics()
            logger = get_api_logger()
            
            start_time = time.time()
            logger.info(f"Operation started: {op_name}", component=component)
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                metrics.collector.record_timer(f"{component}_operation_duration_seconds", duration, operation=op_name)
                logger.info(f"Operation completed: {op_name}", component=component, duration_seconds=duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.collector.increment_counter(f"{component}_operation_errors_total", 1, operation=op_name, error_type=type(e).__name__)
                logger.error(f"Operation failed: {op_name}", exception=e, component=component, duration_seconds=duration)
                raise
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


if __name__ == "__main__":
    # Test monitoring middleware components
    import asyncio
    
    async def test_monitoring():
        print("Testing monitoring components...")
        
        # Test operation monitoring
        async with monitor_operation("test_operation", component="test", test_param="value"):
            await asyncio.sleep(0.1)  # Simulate work
            print("Operation completed successfully")
        
        # Test performance tracker
        tracker = get_performance_tracker()
        tracker.start_operation("op1", "test_operation", component="test")
        await asyncio.sleep(0.05)
        tracker.end_operation("op1", success=True)
        
        print(f"Active operations: {tracker.get_active_operations()}")
        
        # Test function decorator
        @monitor_function("test_decorated_function", component="test")
        async def test_func():
            await asyncio.sleep(0.02)
            return "test result"
        
        result = await test_func()
        print(f"Decorated function result: {result}")
        
        print("Monitoring components test completed.")
    
    # Run the test
    asyncio.run(test_monitoring())