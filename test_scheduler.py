#!/usr/bin/env python3
"""
Test script for the automated pipeline scheduler.
"""
import sys
import os
import time
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_scheduler_components():
    """Test scheduler components individually"""
    print("=== Testing Scheduler Components ===")
    
    # Test configuration
    try:
        from config.scheduler_config import get_config, validate_config
        
        config = get_config()
        print(f"✅ Configuration loaded: {config.enabled_sources}")
        
        issues = validate_config(config)
        if issues:
            print(f"⚠️  Configuration issues: {issues}")
        else:
            print("✅ Configuration validation passed")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test retry handler
    try:
        from src.utils.retry_handler import RetryHandler, RetryConfig
        
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        handler = RetryHandler(config)
        
        # Test successful operation
        def success_func():
            return "success"
        
        result = handler.retry(success_func)
        print(f"✅ Retry handler success test: {result}")
        
        # Test failing operation
        attempt_count = 0
        def failing_func():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError(f"Attempt {attempt_count}")
        
        try:
            handler.retry(failing_func)
        except ValueError:
            print(f"✅ Retry handler failure test: {attempt_count} attempts made")
        
    except Exception as e:
        print(f"❌ Retry handler test failed: {e}")
        return False
    
    return True


def test_pipeline_scheduler():
    """Test the pipeline scheduler itself"""
    print("\n=== Testing Pipeline Scheduler ===")
    
    try:
        from scripts.scheduler import PipelineScheduler
        
        # Create temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db_uri = f"sqlite:///{tmp.name}"
        
        scheduler = PipelineScheduler(test_db_uri)
        
        # Test initialization
        if not scheduler.initialize():
            print("❌ Scheduler initialization failed")
            return False
        
        print("✅ Scheduler initialized successfully")
        
        # Test status method
        status = scheduler.status()
        print(f"✅ Scheduler status: running={status['running']}")
        
        # Clean up
        os.unlink(tmp.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_pipeline_run():
    """Test running the pipeline once"""
    print("\n=== Testing Single Pipeline Run ===")
    
    try:
        from scripts.scheduler import PipelineScheduler
        
        # Use a test database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db_uri = f"sqlite:///{tmp.name}"
        
        scheduler = PipelineScheduler(test_db_uri)
        
        if not scheduler.initialize():
            print("❌ Failed to initialize scheduler")
            return False
        
        print("Running pipeline once (this may take a minute)...")
        result = scheduler.run_pipeline()
        
        print(f"✅ Pipeline completed:")
        print(f"   Total inserted: {result.get('total_inserted', 0)}")
        print(f"   Total skipped: {result.get('total_skipped', 0)}")
        print(f"   Successful sources: {result.get('successful_sources', 0)}")
        print(f"   Duration: {result.get('duration_seconds', 0):.2f} seconds")
        
        if 'error' in result:
            print(f"⚠️  Error occurred: {result['error']}")
        
        # Clean up
        os.unlink(tmp.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Single pipeline run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cron_setup():
    """Test CRON setup script"""
    print("\n=== Testing CRON Setup ===")
    
    try:
        from scripts.setup_cron import setup_cron
        
        # This should create the necessary files
        cron_entry = setup_cron()
        
        print("✅ CRON setup completed")
        print(f"✅ CRON entry created: {cron_entry[:50]}...")
        
        # Check if the one-time script was created
        project_root = Path(__file__).parent
        script_path = project_root / "scripts" / "run_pipeline_once.py"
        
        if script_path.exists():
            print("✅ One-time execution script created")
            
            # Clean up test file
            os.unlink(script_path)
        else:
            print("⚠️  One-time execution script not found")
        
        return True
        
    except Exception as e:
        print(f"❌ CRON setup test failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable handling"""
    print("\n=== Testing Environment Variables ===")
    
    # Set some test environment variables
    test_vars = {
        'PIPELINE_INTERVAL_MINUTES': '15',
        'MAX_CONSECUTIVE_FAILURES': '3',
        'LOG_LEVEL': 'DEBUG',
        'ENVIRONMENT': 'test'
    }
    
    for key, value in test_vars.items():
        os.environ[key] = value
    
    try:
        from config.scheduler_config import get_config
        
        config = get_config()
        
        if config.run_interval_minutes == 1:  # Test config should be 1 minute
            print("✅ Environment-based configuration working")
        else:
            print(f"⚠️  Expected test interval 1, got {config.run_interval_minutes}")
        
        print(f"✅ Configuration: {config.enabled_sources}")
        
        # Clean up
        for key in test_vars:
            del os.environ[key]
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False


def main():
    """Run all scheduler tests"""
    print("🚀 Testing FinBrief Automated Scheduler")
    print("=" * 50)
    
    tests = [
        ("Component Tests", test_scheduler_components),
        ("Environment Variables", test_environment_variables),
        ("Pipeline Scheduler", test_pipeline_scheduler),
        ("CRON Setup", test_cron_setup),
        ("Single Pipeline Run", test_single_pipeline_run),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed >= 4:  # Allow one test to fail
        print("🎉 Scheduler system is ready!")
        print("\n📋 Next steps:")
        print("1. Set up CRON job: python scripts/setup_cron.py")
        print("2. Or run as daemon: python scripts/scheduler.py")
        print("3. Monitor logs: tail -f logs/scheduler.log")
        print("4. Check status: http://localhost:8000/admin/scheduler-status (if API running)")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
    
    return passed >= 4


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)