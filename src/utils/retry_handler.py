"""
Retry and error handling utilities for FinBrief pipeline.
"""
import time
import logging
import functools
from typing import Callable, Any, Type, Union, List
from datetime import datetime, timedelta


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        retry_on_exceptions: Union[Type[Exception], List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions or [Exception]
        
        if not isinstance(self.retry_on_exceptions, list):
            self.retry_on_exceptions = [self.retry_on_exceptions]


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on exception type and attempt count"""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception type is in the retry list
        for retry_exception in self.config.retry_on_exceptions:
            if isinstance(exception, retry_exception):
                return True
        
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt"""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.config.base_delay
        
        # Cap the delay at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.logger.debug(f"Attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    self.logger.error(f"Not retrying {func.__name__} after attempt {attempt}: {e}")
                    break
                
                delay = self.calculate_delay(attempt)
                self.logger.warning(f"Attempt {attempt}/{self.config.max_attempts} failed for {func.__name__}: {e}")
                
                if attempt < self.config.max_attempts:
                    self.logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds...")
                    time.sleep(delay)
        
        # If we get here, all attempts failed
        self.logger.error(f"All {self.config.max_attempts} attempts failed for {func.__name__}")
        raise last_exception


def with_retry(config: RetryConfig = None):
    """Decorator to add retry logic to functions"""
    def decorator(func):
        handler = RetryHandler(config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return handler.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.info(f"Circuit breaker HALF_OPEN for {func.__name__}")
            else:
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.info("Circuit breaker CLOSED")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


# Specific retry configurations for different operations
NEWS_FETCH_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    retry_on_exceptions=[ConnectionError, TimeoutError, Exception]
)

DATABASE_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    retry_on_exceptions=[Exception]  # Retry on any database exception
)

STRATEGY_GENERATION_RETRY = RetryConfig(
    max_attempts=2,
    base_delay=1.0,
    max_delay=5.0,
    retry_on_exceptions=[Exception]
)