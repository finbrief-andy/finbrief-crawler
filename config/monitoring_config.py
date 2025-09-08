"""
Monitoring system configuration for FinBrief application.
Centralized configuration for logging, metrics, and health checks.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Configuration for logging system"""
    log_level: str = "INFO"
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    structured_format: bool = True
    console_output: bool = True
    
    # Component-specific log levels
    component_levels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.component_levels is None:
            self.component_levels = {
                "crawler": "INFO",
                "api": "INFO", 
                "database": "WARNING",
                "nlp": "INFO",
                "strategy": "INFO",
                "monitoring": "DEBUG"
            }


@dataclass
class MetricsConfig:
    """Configuration for metrics system"""
    retention_hours: int = 24
    flush_interval_seconds: int = 60
    max_memory_points: int = 10000
    
    # Metric collection settings
    collect_system_metrics: bool = True
    collect_api_metrics: bool = True
    collect_database_metrics: bool = True
    collect_performance_metrics: bool = True
    
    # Alert thresholds
    api_response_time_threshold_ms: float = 1000.0
    database_query_time_threshold_ms: float = 500.0
    memory_usage_threshold_percent: float = 85.0
    cpu_usage_threshold_percent: float = 80.0
    disk_usage_threshold_percent: float = 90.0


@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    check_interval_seconds: int = 30
    timeout_seconds: float = 5.0
    
    # Health check settings
    enabled_checks: List[str] = None
    disabled_checks: List[str] = None
    
    # Database health check
    db_connection_timeout: float = 3.0
    db_query_timeout: float = 2.0
    
    # External service timeouts
    api_timeout: float = 5.0
    
    def __post_init__(self):
        if self.enabled_checks is None:
            self.enabled_checks = [
                "system_resources",
                "database", 
                "disk_space",
                "memory",
                "cpu",
                "vector_store",
                "nlp_models",
                "external_apis"
            ]
        
        if self.disabled_checks is None:
            self.disabled_checks = []


@dataclass
class MonitoringConfig:
    """Main monitoring configuration"""
    environment: str = "development"
    service_name: str = "finbrief-crawler"
    
    # Sub-configurations
    logging: LoggingConfig = None
    metrics: MetricsConfig = None
    health: HealthCheckConfig = None
    
    # API settings
    enable_monitoring_api: bool = True
    monitoring_api_auth_required: bool = True
    exclude_monitoring_paths: List[str] = None
    
    # Alerting (future feature)
    enable_alerting: bool = False
    alert_webhook_url: Optional[str] = None
    
    def __post_init__(self):
        if self.logging is None:
            self.logging = LoggingConfig()
        
        if self.metrics is None:
            self.metrics = MetricsConfig()
        
        if self.health is None:
            self.health = HealthCheckConfig()
        
        if self.exclude_monitoring_paths is None:
            self.exclude_monitoring_paths = [
                "/docs",
                "/openapi.json", 
                "/favicon.ico",
                "/monitoring/ping"  # Don't monitor the ping endpoint
            ]


def load_config_from_env() -> MonitoringConfig:
    """Load monitoring configuration from environment variables"""
    
    # Main configuration
    config = MonitoringConfig(
        environment=os.getenv("ENVIRONMENT", "development"),
        service_name=os.getenv("SERVICE_NAME", "finbrief-crawler"),
        enable_monitoring_api=os.getenv("ENABLE_MONITORING_API", "true").lower() == "true",
        monitoring_api_auth_required=os.getenv("MONITORING_API_AUTH", "true").lower() == "true"
    )
    
    # Logging configuration
    config.logging = LoggingConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_dir=os.getenv("LOG_DIR", "logs"),
        max_file_size_mb=int(os.getenv("LOG_MAX_FILE_SIZE_MB", "10")),
        backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        structured_format=os.getenv("LOG_STRUCTURED", "true").lower() == "true",
        console_output=os.getenv("LOG_CONSOLE", "true").lower() == "true"
    )
    
    # Metrics configuration
    config.metrics = MetricsConfig(
        retention_hours=int(os.getenv("METRICS_RETENTION_HOURS", "24")),
        flush_interval_seconds=int(os.getenv("METRICS_FLUSH_INTERVAL", "60")),
        max_memory_points=int(os.getenv("METRICS_MAX_MEMORY_POINTS", "10000")),
        collect_system_metrics=os.getenv("COLLECT_SYSTEM_METRICS", "true").lower() == "true",
        collect_api_metrics=os.getenv("COLLECT_API_METRICS", "true").lower() == "true",
        collect_database_metrics=os.getenv("COLLECT_DATABASE_METRICS", "true").lower() == "true",
        api_response_time_threshold_ms=float(os.getenv("API_RESPONSE_TIME_THRESHOLD_MS", "1000")),
        database_query_time_threshold_ms=float(os.getenv("DB_QUERY_TIME_THRESHOLD_MS", "500")),
        memory_usage_threshold_percent=float(os.getenv("MEMORY_USAGE_THRESHOLD", "85")),
        cpu_usage_threshold_percent=float(os.getenv("CPU_USAGE_THRESHOLD", "80")),
        disk_usage_threshold_percent=float(os.getenv("DISK_USAGE_THRESHOLD", "90"))
    )
    
    # Health check configuration
    config.health = HealthCheckConfig(
        check_interval_seconds=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
        timeout_seconds=float(os.getenv("HEALTH_CHECK_TIMEOUT", "5")),
        db_connection_timeout=float(os.getenv("DB_CONNECTION_TIMEOUT", "3")),
        db_query_timeout=float(os.getenv("DB_QUERY_TIMEOUT", "2")),
        api_timeout=float(os.getenv("API_TIMEOUT", "5"))
    )
    
    return config


def get_production_config() -> MonitoringConfig:
    """Get production-optimized monitoring configuration"""
    config = MonitoringConfig(
        environment="production",
        service_name="finbrief-crawler"
    )
    
    # Production logging settings
    config.logging = LoggingConfig(
        log_level="WARNING",  # Less verbose in production
        log_dir="/var/log/finbrief",
        max_file_size_mb=50,  # Larger files in production
        backup_count=10,
        structured_format=True,
        console_output=False,  # No console output in production
        component_levels={
            "crawler": "INFO",
            "api": "WARNING",
            "database": "ERROR",
            "nlp": "WARNING", 
            "strategy": "INFO",
            "monitoring": "ERROR"
        }
    )
    
    # Production metrics settings
    config.metrics = MetricsConfig(
        retention_hours=72,  # Keep metrics longer in production
        flush_interval_seconds=30,  # More frequent flushing
        max_memory_points=50000,  # More memory for metrics
        collect_system_metrics=True,
        collect_api_metrics=True,
        collect_database_metrics=True,
        api_response_time_threshold_ms=500.0,  # Stricter thresholds
        database_query_time_threshold_ms=200.0,
        memory_usage_threshold_percent=80.0,
        cpu_usage_threshold_percent=70.0,
        disk_usage_threshold_percent=85.0
    )
    
    # Production health check settings
    config.health = HealthCheckConfig(
        check_interval_seconds=15,  # More frequent health checks
        timeout_seconds=3.0,  # Shorter timeouts
        db_connection_timeout=2.0,
        db_query_timeout=1.0,
        api_timeout=3.0
    )
    
    return config


def get_development_config() -> MonitoringConfig:
    """Get development-optimized monitoring configuration"""
    config = MonitoringConfig(
        environment="development",
        service_name="finbrief-crawler-dev"
    )
    
    # Development logging settings (more verbose)
    config.logging = LoggingConfig(
        log_level="DEBUG",
        log_dir="logs",
        max_file_size_mb=5,
        backup_count=3,
        structured_format=True,
        console_output=True,
        component_levels={
            "crawler": "DEBUG",
            "api": "DEBUG",
            "database": "INFO",
            "nlp": "DEBUG",
            "strategy": "DEBUG",
            "monitoring": "DEBUG"
        }
    )
    
    # Development metrics settings
    config.metrics = MetricsConfig(
        retention_hours=6,  # Shorter retention in dev
        flush_interval_seconds=120,  # Less frequent flushing
        max_memory_points=5000,
        collect_system_metrics=True,
        collect_api_metrics=True,
        collect_database_metrics=True,
        api_response_time_threshold_ms=2000.0,  # More lenient thresholds
        database_query_time_threshold_ms=1000.0,
        memory_usage_threshold_percent=90.0,
        cpu_usage_threshold_percent=85.0,
        disk_usage_threshold_percent=95.0
    )
    
    # Development health check settings
    config.health = HealthCheckConfig(
        check_interval_seconds=60,  # Less frequent in dev
        timeout_seconds=10.0,  # Longer timeouts for debugging
        db_connection_timeout=5.0,
        db_query_timeout=3.0,
        api_timeout=10.0
    )
    
    return config


def get_test_config() -> MonitoringConfig:
    """Get test-optimized monitoring configuration"""
    config = MonitoringConfig(
        environment="test",
        service_name="finbrief-crawler-test"
    )
    
    # Test logging settings (minimal)
    config.logging = LoggingConfig(
        log_level="ERROR",  # Only errors in tests
        log_dir="test_logs",
        max_file_size_mb=1,
        backup_count=1,
        structured_format=False,
        console_output=False
    )
    
    # Test metrics settings (minimal)
    config.metrics = MetricsConfig(
        retention_hours=1,
        flush_interval_seconds=300,
        max_memory_points=100,
        collect_system_metrics=False,  # Don't collect system metrics in tests
        collect_api_metrics=False,
        collect_database_metrics=False
    )
    
    # Test health check settings (fast)
    config.health = HealthCheckConfig(
        check_interval_seconds=300,  # Infrequent in tests
        timeout_seconds=1.0,  # Very short timeouts
        db_connection_timeout=1.0,
        db_query_timeout=0.5,
        api_timeout=1.0,
        enabled_checks=["system_resources", "memory"]  # Only basic checks
    )
    
    return config


# Global configuration cache
_config_cache = None

def get_config() -> MonitoringConfig:
    """Get monitoring configuration based on environment"""
    global _config_cache
    
    if _config_cache is None:
        environment = os.getenv("ENVIRONMENT", "development").lower()
        
        if environment == "production":
            _config_cache = get_production_config()
        elif environment == "test":
            _config_cache = get_test_config()
        else:  # development or unknown
            _config_cache = get_development_config()
        
        # Override with environment variables if present
        env_config = load_config_from_env()
        if env_config.environment != "development":
            _config_cache = env_config
    
    return _config_cache


def reset_config():
    """Reset configuration cache (useful for testing)"""
    global _config_cache
    _config_cache = None


if __name__ == "__main__":
    # Test configuration loading
    import json
    
    print("Testing monitoring configuration...")
    
    # Test different environments
    environments = ["development", "production", "test"]
    
    for env in environments:
        os.environ["ENVIRONMENT"] = env
        reset_config()
        config = get_config()
        
        print(f"\n{env.title()} Configuration:")
        print(f"  Log Level: {config.logging.log_level}")
        print(f"  Log Directory: {config.logging.log_dir}")
        print(f"  Metrics Retention: {config.metrics.retention_hours} hours")
        print(f"  Health Check Interval: {config.health.check_interval_seconds} seconds")
        print(f"  Enabled Health Checks: {len(config.health.enabled_checks)}")
    
    # Test environment variable override
    print("\nTesting environment variable override...")
    os.environ.update({
        "LOG_LEVEL": "WARNING",
        "METRICS_RETENTION_HOURS": "48",
        "HEALTH_CHECK_TIMEOUT": "10"
    })
    
    reset_config()
    config = load_config_from_env()
    
    print(f"  Overridden Log Level: {config.logging.log_level}")
    print(f"  Overridden Metrics Retention: {config.metrics.retention_hours}")
    print(f"  Overridden Health Check Timeout: {config.health.timeout_seconds}")
    
    print("\nConfiguration testing completed.")