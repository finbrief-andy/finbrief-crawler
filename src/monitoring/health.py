"""
Comprehensive health check system for FinBrief application.
Monitors system components and provides health status endpoints.
"""
import time
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import traceback

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any] = None


class HealthChecker:
    """Individual health check implementation"""
    
    def __init__(self, name: str, check_func: Callable, timeout: float = 5.0):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
    
    async def run_check(self) -> HealthCheckResult:
        """Execute the health check"""
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Run the check function with timeout
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(self.check_func(), timeout=self.timeout)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.check_func
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                message = result.get("message", "OK")
                details = result.get("details", {})
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                details = {}
            else:
                status = HealthStatus.HEALTHY
                message = str(result) if result else "OK"
                details = {}
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=timestamp,
                details=details
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=duration_ms,
                timestamp=timestamp,
                details={"error": "timeout"}
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=timestamp,
                details={
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )


class HealthMonitor:
    """
    Comprehensive health monitoring system for FinBrief application.
    """
    
    def __init__(self):
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.system_start_time = datetime.now()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks"""
        
        # System resources check
        self.register_check("system_resources", self._check_system_resources)
        
        # Database connectivity check
        self.register_check("database", self._check_database)
        
        # Disk space check
        self.register_check("disk_space", self._check_disk_space)
        
        # Memory usage check
        self.register_check("memory", self._check_memory_usage)
        
        # CPU usage check
        self.register_check("cpu", self._check_cpu_usage)
        
        # Vector store check (if available)
        self.register_check("vector_store", self._check_vector_store)
        
        # NLP models check
        self.register_check("nlp_models", self._check_nlp_models)
        
        # External APIs check
        self.register_check("external_apis", self._check_external_apis)
    
    def register_check(self, name: str, check_func: Callable, timeout: float = 5.0):
        """Register a new health check"""
        self.health_checkers[name] = HealthChecker(name, check_func, timeout)
    
    def unregister_check(self, name: str):
        """Remove a health check"""
        self.health_checkers.pop(name, None)
        self.last_results.pop(name, None)
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        # Run checks concurrently
        tasks = [
            checker.run_check() 
            for checker in self.health_checkers.values()
        ]
        
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in check_results:
            if isinstance(result, Exception):
                # Handle unexpected errors
                results["unknown"] = HealthCheckResult(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Unexpected error: {str(result)}",
                    duration_ms=0,
                    timestamp=datetime.now()
                )
            else:
                results[result.name] = result
                self.last_results[result.name] = result
        
        return results
    
    async def run_single_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a single health check by name"""
        checker = self.health_checkers.get(name)
        if not checker:
            return None
        
        result = await checker.run_check()
        self.last_results[name] = result
        return result
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        if not self.last_results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks have been performed yet",
                "uptime_seconds": (datetime.now() - self.system_start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        
        # Determine overall health status
        statuses = [result.status for result in self.last_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Count statuses
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(
                1 for s in statuses if s == status
            )
        
        return {
            "status": overall_status.value,
            "message": f"System is {overall_status.value}",
            "uptime_seconds": (datetime.now() - self.system_start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "total": len(self.last_results),
                **status_counts
            },
            "details": {
                name: asdict(result) for name, result in self.last_results.items()
            }
        }
    
    # Default health check implementations
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check overall system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically high"
            elif cpu_percent > 75 or memory.percent > 75 or disk.percent > 85:
                status = HealthStatus.DEGRADED  
                message = "System resources elevated"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Failed to check system resources: {str(e)}"
            }
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            import os
            database_uri = os.getenv("DATABASE_URI", "postgresql://andy.huynh@localhost:5432/finbrief_prod")
            
            engine = create_engine(database_uri, pool_pre_ping=True)
            
            start_time = time.time()
            with engine.connect() as conn:
                # Test basic connectivity
                conn.execute(text("SELECT 1"))
                
                # Test query performance
                result = conn.execute(text("SELECT COUNT(*) as count FROM news"))
                news_count = result.fetchone()[0]
                
            query_duration = (time.time() - start_time) * 1000
            
            if query_duration > 5000:  # 5 seconds
                status = HealthStatus.UNHEALTHY
                message = "Database queries extremely slow"
            elif query_duration > 1000:  # 1 second
                status = HealthStatus.DEGRADED
                message = "Database queries slow"
            else:
                status = HealthStatus.HEALTHY
                message = "Database connectivity OK"
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "query_duration_ms": query_duration,
                    "news_count": news_count,
                    "database_uri": database_uri.split('@')[1] if '@' in database_uri else "unknown"
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Database check failed: {str(e)}"
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            if disk.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Disk space critically low"
            elif disk.percent > 85:
                status = HealthStatus.DEGRADED
                message = "Disk space running low"
            else:
                status = HealthStatus.HEALTHY
                message = "Disk space sufficient"
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": free_gb,
                    "percent_used": disk.percent
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Disk space check failed: {str(e)}"
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            if memory.percent > 90 or swap.percent > 50:
                status = HealthStatus.UNHEALTHY
                message = "Memory usage critically high"
            elif memory.percent > 75 or swap.percent > 25:
                status = HealthStatus.DEGRADED
                message = "Memory usage elevated"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal"
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "swap_percent": swap.percent,
                    "swap_used_gb": swap.used / (1024**3)
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Memory check failed: {str(e)}"
            }
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "CPU usage critically high"
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                message = "CPU usage elevated"
            else:
                status = HealthStatus.HEALTHY
                message = "CPU usage normal"
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "per_cpu": psutil.cpu_percent(percpu=True)
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"CPU check failed: {str(e)}"
            }
    
    def _check_vector_store(self) -> Dict[str, Any]:
        """Check vector store availability"""
        try:
            from src.services.enhanced_vector_store import get_enhanced_vector_store
            
            store = get_enhanced_vector_store()
            if store.is_available():
                # Test basic functionality
                stats = store.get_collection_stats()
                
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": "Vector store available",
                    "details": {
                        "collections": stats.get("collections", {}),
                        "available": True
                    }
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED.value,
                    "message": "Vector store not available (dependencies missing)"
                }
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED.value,
                "message": f"Vector store check failed: {str(e)}"
            }
    
    def _check_nlp_models(self) -> Dict[str, Any]:
        """Check NLP model availability"""
        try:
            from src.crawlers.processors.enhanced_nlp_processor import EnhancedNLPProcessor
            
            processor = EnhancedNLPProcessor()
            if processor.is_available():
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": "NLP models available",
                    "details": {
                        "models_loaded": processor.get_loaded_models() if hasattr(processor, 'get_loaded_models') else [],
                        "available": True
                    }
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED.value,
                    "message": "NLP models not available (dependencies missing)"
                }
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED.value,
                "message": f"NLP models check failed: {str(e)}"
            }
    
    def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        try:
            import os
            
            # Check OpenAI API key availability
            openai_key = os.getenv("OPENAI_API_KEY")
            
            if openai_key:
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": "External APIs configured",
                    "details": {
                        "openai_configured": True,
                        "api_keys_present": ["OPENAI_API_KEY"]
                    }
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED.value,
                    "message": "Some external APIs not configured",
                    "details": {
                        "openai_configured": False,
                        "missing_keys": ["OPENAI_API_KEY"]
                    }
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"External API check failed: {str(e)}"
            }


# Global health monitor instance
_global_health_monitor = None

def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor instance"""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


# Convenience functions for common health checks
async def check_system_health() -> Dict[str, Any]:
    """Quick system health check"""
    monitor = get_health_monitor()
    return monitor.get_system_health_summary()

async def check_all_health() -> Dict[str, Any]:
    """Run all health checks and return results"""
    monitor = get_health_monitor()
    results = await monitor.run_all_checks()
    return {
        "summary": monitor.get_system_health_summary(),
        "individual_checks": {name: asdict(result) for name, result in results.items()}
    }


if __name__ == "__main__":
    # Test health monitoring system
    import asyncio
    
    async def test_health_checks():
        monitor = get_health_monitor()
        
        print("Running all health checks...")
        results = await monitor.run_all_checks()
        
        print("\nHealth Check Results:")
        for name, result in results.items():
            print(f"  {name}: {result.status.value} - {result.message} ({result.duration_ms:.1f}ms)")
        
        print("\nSystem Health Summary:")
        summary = monitor.get_system_health_summary()
        print(f"  Overall Status: {summary['status']}")
        print(f"  Uptime: {summary['uptime_seconds']:.1f} seconds")
        print(f"  Total Checks: {summary['checks']['total']}")
        
        return summary
    
    # Run the test
    asyncio.run(test_health_checks())
    print("\nHealth monitoring test completed.")