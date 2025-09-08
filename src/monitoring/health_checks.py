#!/usr/bin/env python3
"""
Health Check System

Comprehensive health monitoring for all application components
including database, external APIs, file systems, and services.
"""

import asyncio
import psutil
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    metadata: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    checks: List[HealthCheckResult]
    summary: Dict[str, Any]
    timestamp: datetime
    uptime: float


class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks: Dict[str, Callable] = {}
        self.check_intervals: Dict[str, int] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.start_time = datetime.now()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check("system_resources", self.check_system_resources, interval=30)
        self.register_check("disk_space", self.check_disk_space, interval=60)
        self.register_check("memory_usage", self.check_memory_usage, interval=30)
        self.register_check("process_health", self.check_process_health, interval=60)
        
        # Only register database check if SQLAlchemy is available
        if SQLALCHEMY_AVAILABLE:
            self.register_check("database_connectivity", self.check_database_connectivity, interval=30)
        
        self.register_check("log_directory", self.check_log_directory, interval=300)
        self.register_check("configuration", self.check_configuration, interval=300)
    
    def register_check(self, name: str, check_func: Callable, interval: int = 60):
        """Register a health check function"""
        self.checks[name] = check_func
        self.check_intervals[name] = interval
        self.logger.info(f"Registered health check: {name} (interval: {interval}s)")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown health check: {name}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error="Check not found"
            )
        
        start_time = time.time()
        
        try:
            check_func = self.checks[name]
            
            # Run check with timeout
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=30.0)
            else:
                result = check_func()
            
            response_time = time.time() - start_time
            
            if isinstance(result, HealthCheckResult):
                result.response_time = response_time
                result.timestamp = datetime.now()
                return result
            else:
                # Convert simple result to HealthCheckResult
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                return HealthCheckResult(
                    name=name,
                    status=status,
                    message=f"Check {name} {'passed' if result else 'failed'}",
                    response_time=response_time,
                    timestamp=datetime.now(),
                    metadata={}
                )
                
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check {name} timed out",
                response_time=response_time,
                timestamp=datetime.now(),
                metadata={},
                error="Timeout"
            )
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Health check {name} failed: {e}")
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check {name} failed: {str(e)}",
                response_time=response_time,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )
    
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks"""
        tasks = []
        
        for name in self.checks.keys():
            task = asyncio.create_task(self.run_check(name))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert them to error results
        health_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                name = list(self.checks.keys())[i]
                health_results.append(HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check {name} raised exception: {str(result)}",
                    response_time=0.0,
                    timestamp=datetime.now(),
                    metadata={},
                    error=str(result)
                ))
            else:
                health_results.append(result)
                # Cache result
                self.last_results[result.name] = result
        
        return health_results
    
    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status"""
        check_results = await self.run_all_checks()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        critical_count = 0
        warning_count = 0
        
        for result in check_results:
            if result.status == HealthStatus.CRITICAL:
                critical_count += 1
                overall_status = HealthStatus.CRITICAL
            elif result.status == HealthStatus.WARNING:
                warning_count += 1
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
        
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Generate summary
        summary = {
            "total_checks": len(check_results),
            "healthy_checks": len([r for r in check_results if r.status == HealthStatus.HEALTHY]),
            "warning_checks": warning_count,
            "critical_checks": critical_count,
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "average_response_time": sum(r.response_time for r in check_results) / len(check_results) if check_results else 0
        }
        
        return SystemHealth(
            status=overall_status,
            checks=check_results,
            summary=summary,
            timestamp=datetime.now(),
            uptime=uptime
        )
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    # Default health check implementations
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 75:
                status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > 80:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory.percent:.1f}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "System resources healthy" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )
    
    def check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability"""
        try:
            # Check multiple important directories
            paths_to_check = [
                ('/', 'root'),
                ('./logs', 'logs'),
                ('./data', 'data'),
                ('/tmp', 'temp')
            ]
            
            status = HealthStatus.HEALTHY
            messages = []
            metadata = {}
            
            for path_str, label in paths_to_check:
                path = Path(path_str)
                if path.exists():
                    try:
                        usage = psutil.disk_usage(str(path))
                        percent_used = (usage.used / usage.total) * 100
                        
                        metadata[f"{label}_usage_percent"] = percent_used
                        metadata[f"{label}_free_gb"] = usage.free / (1024**3)
                        
                        if percent_used > 95:
                            status = HealthStatus.CRITICAL
                            messages.append(f"{label} disk critically full: {percent_used:.1f}%")
                        elif percent_used > 85:
                            if status == HealthStatus.HEALTHY:
                                status = HealthStatus.WARNING
                            messages.append(f"{label} disk getting full: {percent_used:.1f}%")
                            
                    except Exception as e:
                        messages.append(f"Could not check {label} disk: {str(e)}")
            
            message = "Disk space healthy" if not messages else "; ".join(messages)
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Disk space check failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )
    
    def check_memory_usage(self) -> HealthCheckResult:
        """Check detailed memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check virtual memory
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Virtual memory critical: {memory.percent:.1f}%")
            elif memory.percent > 85:
                status = HealthStatus.WARNING
                issues.append(f"Virtual memory high: {memory.percent:.1f}%")
            
            # Check swap usage
            if swap.total > 0:  # Only check if swap exists
                swap_percent = (swap.used / swap.total) * 100
                if swap_percent > 80:
                    status = HealthStatus.WARNING
                    issues.append(f"Swap usage high: {swap_percent:.1f}%")
            
            message = "Memory usage healthy" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={
                    "virtual_memory_percent": memory.percent,
                    "virtual_memory_available_gb": memory.available / (1024**3),
                    "virtual_memory_used_gb": memory.used / (1024**3),
                    "virtual_memory_total_gb": memory.total / (1024**3),
                    "swap_percent": (swap.used / swap.total * 100) if swap.total > 0 else 0,
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_total_gb": swap.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Memory usage check failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )
    
    def check_process_health(self) -> HealthCheckResult:
        """Check process and thread health"""
        try:
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            
            # Get current process info
            cpu_percent = current_process.cpu_percent()
            memory_info = current_process.memory_info()
            thread_count = current_process.num_threads()
            
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check if we have too many threads
            if thread_count > 100:
                status = HealthStatus.WARNING
                issues.append(f"High thread count: {thread_count}")
            
            # Check process memory usage
            memory_mb = memory_info.rss / (1024 * 1024)
            if memory_mb > 1024:  # More than 1GB
                status = HealthStatus.WARNING
                issues.append(f"High process memory usage: {memory_mb:.1f}MB")
            
            message = "Process health good" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                name="process_health",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={
                    "system_process_count": process_count,
                    "current_process_cpu_percent": cpu_percent,
                    "current_process_memory_mb": memory_mb,
                    "current_process_thread_count": thread_count,
                    "current_process_pid": current_process.pid
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="process_health",
                status=HealthStatus.CRITICAL,
                message=f"Process health check failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )
    
    async def check_database_connectivity(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        if not SQLALCHEMY_AVAILABLE:
            return HealthCheckResult(
                name="database_connectivity",
                status=HealthStatus.UNKNOWN,
                message="SQLAlchemy not available",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={}
            )
        
        try:
            # Try to get database URL from environment or use default
            import os
            database_url = os.getenv('DATABASE_URI', 'postgresql://andy.huynh@localhost:5432/finbrief_prod')
            
            start_time = time.time()
            
            # Create engine and test connection
            engine = create_engine(database_url, pool_timeout=5, pool_recycle=300)
            
            with engine.connect() as conn:
                # Simple query to test connectivity
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get some database stats
                try:
                    stats_result = conn.execute(text("""
                        SELECT 
                            count(*) as connection_count,
                            current_database() as database_name
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                    """))
                    stats = stats_result.fetchone()
                    connection_count = stats[0] if stats else 0
                    db_name = stats[1] if stats else "unknown"
                except:
                    connection_count = 0
                    db_name = "unknown"
            
            response_time = time.time() - start_time
            
            # Determine status based on response time
            status = HealthStatus.HEALTHY
            if response_time > 5.0:
                status = HealthStatus.WARNING
            elif response_time > 10.0:
                status = HealthStatus.CRITICAL
            
            message = f"Database connection healthy ({response_time:.3f}s)"
            
            return HealthCheckResult(
                name="database_connectivity",
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                metadata={
                    "database_name": db_name,
                    "connection_count": connection_count,
                    "response_time_ms": response_time * 1000
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="database_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )
    
    def check_log_directory(self) -> HealthCheckResult:
        """Check log directory health"""
        try:
            log_dir = Path("logs")
            
            if not log_dir.exists():
                return HealthCheckResult(
                    name="log_directory",
                    status=HealthStatus.CRITICAL,
                    message="Log directory does not exist",
                    response_time=0.0,
                    timestamp=datetime.now(),
                    metadata={}
                )
            
            # Check if directory is writable
            test_file = log_dir / ".health_check_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                return HealthCheckResult(
                    name="log_directory",
                    status=HealthStatus.CRITICAL,
                    message=f"Log directory not writable: {str(e)}",
                    response_time=0.0,
                    timestamp=datetime.now(),
                    metadata={},
                    error=str(e)
                )
            
            # Get directory stats
            log_files = list(log_dir.glob("*.log"))
            total_size = sum(f.stat().st_size for f in log_files if f.exists())
            
            status = HealthStatus.HEALTHY
            message = "Log directory healthy"
            
            # Check if logs are too large (>100MB total)
            if total_size > 100 * 1024 * 1024:
                status = HealthStatus.WARNING
                message = f"Log directory size large: {total_size / (1024*1024):.1f}MB"
            
            return HealthCheckResult(
                name="log_directory",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={
                    "log_files_count": len(log_files),
                    "total_size_mb": total_size / (1024 * 1024),
                    "directory_path": str(log_dir.absolute())
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="log_directory",
                status=HealthStatus.CRITICAL,
                message=f"Log directory check failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )
    
    def check_configuration(self) -> HealthCheckResult:
        """Check configuration files and environment"""
        try:
            import os
            
            issues = []
            metadata = {}
            
            # Check important environment variables
            important_env_vars = [
                'DATABASE_URI',
                'SECRET_KEY',
                'LOG_LEVEL',
                'ENVIRONMENT'
            ]
            
            for var in important_env_vars:
                value = os.getenv(var)
                metadata[f"env_{var.lower()}"] = "set" if value else "not_set"
                if not value and var in ['DATABASE_URI']:
                    issues.append(f"Missing environment variable: {var}")
            
            # Check configuration files
            config_files = [
                'config/production.py',
                '.env.production',
                'requirements.txt'
            ]
            
            for config_file in config_files:
                path = Path(config_file)
                exists = path.exists()
                metadata[f"config_{path.stem}"] = "exists" if exists else "missing"
                
                if not exists and config_file == 'requirements.txt':
                    issues.append(f"Missing configuration file: {config_file}")
            
            status = HealthStatus.WARNING if issues else HealthStatus.HEALTHY
            message = "Configuration healthy" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                name="configuration",
                status=status,
                message=message,
                response_time=0.0,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                status=HealthStatus.CRITICAL,
                message=f"Configuration check failed: {str(e)}",
                response_time=0.0,
                timestamp=datetime.now(),
                metadata={},
                error=str(e)
            )


# Global health checker instance
_global_health_checker = None


def get_health_checker() -> HealthChecker:
    """Get or create global health checker"""
    global _global_health_checker
    
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    
    return _global_health_checker


# Convenience functions

async def check_system_health() -> SystemHealth:
    """Get comprehensive system health status"""
    checker = get_health_checker()
    return await checker.get_system_health()


async def quick_health_check() -> Dict[str, Any]:
    """Get a quick health summary"""
    health = await check_system_health()
    
    return {
        "status": health.status.value,
        "uptime": health.uptime,
        "checks_passed": health.summary["healthy_checks"],
        "checks_total": health.summary["total_checks"],
        "timestamp": health.timestamp.isoformat()
    }