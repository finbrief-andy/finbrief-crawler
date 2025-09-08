"""
Advanced logging system for FinBrief application.
Provides structured logging, performance tracking, and error monitoring.
"""
import logging
import logging.handlers
import json
import time
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from functools import wraps

import os


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs for better parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra context if provided
        if hasattr(record, 'extra_context'):
            log_entry["context"] = record.extra_context
        
        # Add performance metrics if present
        if hasattr(record, 'performance'):
            log_entry["performance"] = record.performance
        
        # Add request ID for tracing if present
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        # Add component information
        if hasattr(record, 'component'):
            log_entry["component"] = record.component
        
        return json.dumps(log_entry, ensure_ascii=False)


class FinBriefLogger:
    """
    Advanced logging system for FinBrief application with performance monitoring.
    """
    
    def __init__(self, name: str = "finbrief", log_level: str = "INFO", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Performance tracking
        self.performance_metrics = {}
    
    def _setup_handlers(self):
        """Set up logging handlers for console and file output"""
        
        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # File handler with structured JSON output
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "finbrief.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        
        # Performance metrics handler
        metrics_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "metrics.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        metrics_handler.setFormatter(StructuredFormatter())
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.addFilter(lambda record: hasattr(record, 'performance'))
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(metrics_handler)
    
    def info(self, message: str, **context):
        """Log info message with optional context"""
        self._log_with_context(logging.INFO, message, context)
    
    def debug(self, message: str, **context):
        """Log debug message with optional context"""
        self._log_with_context(logging.DEBUG, message, context)
    
    def warning(self, message: str, **context):
        """Log warning message with optional context"""
        self._log_with_context(logging.WARNING, message, context)
    
    def error(self, message: str, exception: Optional[Exception] = None, **context):
        """Log error message with optional exception and context"""
        if exception:
            context['exception_type'] = type(exception).__name__
            context['exception_message'] = str(exception)
        self._log_with_context(logging.ERROR, message, context, exc_info=exception is not None)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **context):
        """Log critical message with optional exception and context"""
        if exception:
            context['exception_type'] = type(exception).__name__
            context['exception_message'] = str(exception)
        self._log_with_context(logging.CRITICAL, message, context, exc_info=exception is not None)
    
    def _log_with_context(self, level: int, message: str, context: Dict[str, Any], exc_info: bool = False):
        """Internal method to log with context"""
        extra = {}
        if context:
            extra['extra_context'] = context
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics for an operation"""
        performance_data = {
            "operation": operation,
            "duration_seconds": round(duration, 4),
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        extra = {'performance': performance_data}
        self.logger.info(f"Performance: {operation} completed in {duration:.4f}s", extra=extra)
    
    @contextmanager
    def performance_timer(self, operation: str, **context):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            self.log_performance(operation, duration, status="error", error=str(e))
            self.error(f"Operation '{operation}' failed", exception=e, **context)
            raise
        else:
            duration = time.time() - start_time
            self.log_performance(operation, duration, status="success")
            self.info(f"Operation '{operation}' completed successfully", duration=duration, **context)
    
    def performance_decorator(self, operation_name: str = None):
        """Decorator for automatic performance monitoring"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                with self.performance_timer(op_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# Global logger instance
_global_logger = None

def get_logger(name: str = "finbrief", **kwargs) -> FinBriefLogger:
    """Get or create a FinBrief logger instance"""
    global _global_logger
    if _global_logger is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_dir = os.getenv("LOG_DIR", "logs")
        _global_logger = FinBriefLogger(name, log_level, log_dir)
    return _global_logger


# Component-specific loggers
def get_crawler_logger() -> FinBriefLogger:
    """Get logger for crawler components"""
    logger = get_logger("finbrief.crawler")
    return logger

def get_api_logger() -> FinBriefLogger:
    """Get logger for API components"""
    logger = get_logger("finbrief.api")
    return logger

def get_database_logger() -> FinBriefLogger:
    """Get logger for database operations"""
    logger = get_logger("finbrief.database")
    return logger

def get_nlp_logger() -> FinBriefLogger:
    """Get logger for NLP processing"""
    logger = get_logger("finbrief.nlp")
    return logger

def get_strategy_logger() -> FinBriefLogger:
    """Get logger for strategy generation"""
    logger = get_logger("finbrief.strategy")
    return logger


# Convenience functions for quick logging
def log_crawler_activity(message: str, source: str = None, count: int = None, **context):
    """Log crawler activity with standard context"""
    logger = get_crawler_logger()
    context.update({
        "component": "crawler",
        "source": source,
        "count": count
    })
    logger.info(message, **{k: v for k, v in context.items() if v is not None})

def log_api_request(message: str, method: str = None, endpoint: str = None, status_code: int = None, **context):
    """Log API request with standard context"""
    logger = get_api_logger()
    context.update({
        "component": "api",
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code
    })
    logger.info(message, **{k: v for k, v in context.items() if v is not None})

def log_database_operation(message: str, operation: str = None, table: str = None, count: int = None, **context):
    """Log database operation with standard context"""
    logger = get_database_logger()
    context.update({
        "component": "database",
        "operation": operation,
        "table": table,
        "count": count
    })
    logger.info(message, **{k: v for k, v in context.items() if v is not None})

def log_nlp_processing(message: str, processor: str = None, text_length: int = None, **context):
    """Log NLP processing with standard context"""
    logger = get_nlp_logger()
    context.update({
        "component": "nlp",
        "processor": processor,
        "text_length": text_length
    })
    logger.info(message, **{k: v for k, v in context.items() if v is not None})

def log_strategy_generation(message: str, model: str = None, confidence: float = None, **context):
    """Log strategy generation with standard context"""
    logger = get_strategy_logger()
    context.update({
        "component": "strategy",
        "model": model,
        "confidence": confidence
    })
    logger.info(message, **{k: v for k, v in context.items() if v is not None})


if __name__ == "__main__":
    # Test logging system
    logger = get_logger()
    
    # Test different log levels
    logger.info("Application starting", version="1.0.0", environment="development")
    logger.debug("Debug information", user_id=123, action="test")
    logger.warning("Warning message", threshold_exceeded=True, value=95)
    
    # Test performance monitoring
    with logger.performance_timer("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Test component-specific logging
    log_crawler_activity("Crawled news articles", source="bloomberg", count=25)
    log_api_request("API request processed", method="GET", endpoint="/api/news", status_code=200)
    log_database_operation("Inserted records", operation="INSERT", table="news", count=10)
    
    # Test error logging
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        logger.error("Test error occurred", exception=e, context="testing")
    
    print("Logging test completed. Check logs/ directory for output files.")