"""
Configuration settings for the FinBrief pipeline scheduler.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class SchedulerConfig:
    """Configuration for the pipeline scheduler"""
    
    # Database settings
    database_uri: str = os.getenv("DATABASE_URI", "sqlite:///./finbrief.db")
    
    # Scheduling settings
    run_interval_minutes: int = int(os.getenv("PIPELINE_INTERVAL_MINUTES", "30"))
    max_concurrent_sources: int = int(os.getenv("MAX_CONCURRENT_SOURCES", "3"))
    
    # Source settings - which sources to run and how often
    enabled_sources: List[str] = None
    source_intervals: Dict[str, int] = None  # Custom intervals per source (in minutes)
    
    # Error handling
    max_consecutive_failures: int = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "5"))
    failure_cooldown_minutes: int = int(os.getenv("FAILURE_COOLDOWN_MINUTES", "60"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/scheduler.log")
    max_log_size_mb: int = int(os.getenv("MAX_LOG_SIZE_MB", "100"))
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Strategy generation
    auto_generate_strategies: bool = os.getenv("AUTO_GENERATE_STRATEGIES", "true").lower() == "true"
    min_news_for_strategy: int = int(os.getenv("MIN_NEWS_FOR_STRATEGY", "5"))
    
    # Performance settings
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_memory_usage_mb: int = int(os.getenv("MAX_MEMORY_USAGE_MB", "1024"))
    
    # Notification settings
    enable_notifications: bool = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    notification_webhook: str = os.getenv("NOTIFICATION_WEBHOOK", "")
    notify_on_failure: bool = os.getenv("NOTIFY_ON_FAILURE", "true").lower() == "true"
    notify_on_success: bool = os.getenv("NOTIFY_ON_SUCCESS", "false").lower() == "true"
    
    def __post_init__(self):
        """Initialize default values"""
        if self.enabled_sources is None:
            self.enabled_sources = [
                'finnhub',
                'marketwatch', 
                'cafef',
                'gold_api'
                # 'reuters',  # Disabled by default due to access issues
                # 'vietstock', # Disabled by default due to parsing issues
                # 'vnexpress', # Disabled by default - requires feedparser
                # 'real_estate' # Disabled by default - might be too specialized
            ]
        
        if self.source_intervals is None:
            self.source_intervals = {
                'finnhub': 60,        # Every hour (API limits)
                'marketwatch': 30,    # Every 30 minutes
                'reuters': 45,        # Every 45 minutes
                'cafef': 30,         # Every 30 minutes
                'vnexpress': 30,     # Every 30 minutes
                'vietstock': 60,     # Every hour
                'gold_api': 120,     # Every 2 hours (slower moving)
                'real_estate': 240   # Every 4 hours (slower moving)
            }


@dataclass 
class ProductionConfig(SchedulerConfig):
    """Production-specific configuration"""
    
    run_interval_minutes: int = 30
    log_level: str = "INFO"
    max_concurrent_sources: int = 5
    enable_notifications: bool = True
    notify_on_failure: bool = True


@dataclass
class DevelopmentConfig(SchedulerConfig):
    """Development-specific configuration"""
    
    run_interval_minutes: int = 5  # More frequent for testing
    log_level: str = "DEBUG"
    max_concurrent_sources: int = 2
    enabled_sources: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Only enable reliable sources for development
        self.enabled_sources = ['gold_api', 'marketwatch']


@dataclass
class TestConfig(SchedulerConfig):
    """Test-specific configuration"""
    
    database_uri: str = "sqlite:///./test_scheduler.db"
    run_interval_minutes: int = 1  # Very frequent for testing
    log_level: str = "DEBUG"
    max_consecutive_failures: int = 2
    enabled_sources: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Only test sources for testing
        self.enabled_sources = ['gold_api']


def get_config() -> SchedulerConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()


# Configuration validation
def validate_config(config: SchedulerConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    if config.run_interval_minutes < 1:
        issues.append("run_interval_minutes must be at least 1")
    
    if config.max_concurrent_sources < 1:
        issues.append("max_concurrent_sources must be at least 1")
    
    if not config.database_uri:
        issues.append("database_uri is required")
    
    if config.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        issues.append("log_level must be a valid logging level")
    
    for source in config.enabled_sources:
        if source not in ['finnhub', 'marketwatch', 'reuters', 'cafef', 'vnexpress', 'vietstock', 'gold_api', 'real_estate']:
            issues.append(f"Unknown source: {source}")
    
    return issues