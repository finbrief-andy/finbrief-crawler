#!/usr/bin/env python3
"""
Quick test for scheduler initialization without running full pipeline.
"""
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_quick():
    """Quick test of scheduler components"""
    print("🚀 Quick Scheduler Test")
    print("=" * 30)
    
    # Test 1: Configuration
    try:
        from config.scheduler_config import get_config, validate_config
        config = get_config()
        issues = validate_config(config)
        assert not issues, f"Config issues: {issues}"
        print("✅ Configuration OK")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test 2: Scheduler initialization (no pipeline run)
    try:
        from scripts.scheduler import PipelineScheduler
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db_uri = f"sqlite:///{tmp.name}"
        
        scheduler = PipelineScheduler(test_db_uri)
        
        if scheduler.initialize():
            print("✅ Scheduler initialization OK")
            status = scheduler.status()
            print(f"✅ Status method OK: {status['run_interval_minutes']} min interval")
        else:
            print("❌ Scheduler initialization failed")
            return False
        
        os.unlink(tmp.name)
        
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        return False
    
    # Test 3: CRON setup
    try:
        from scripts.setup_cron import setup_cron
        cron_entry = setup_cron()
        print("✅ CRON setup OK")
        
        # Clean up test file
        from pathlib import Path
        script_path = Path(__file__).parent / "scripts" / "run_pipeline_once.py"
        if script_path.exists():
            os.unlink(script_path)
            
    except Exception as e:
        print(f"❌ CRON setup failed: {e}")
        return False
    
    print("\n🎉 All quick tests passed!")
    print("📋 Scheduler system ready for deployment")
    return True

if __name__ == "__main__":
    success = test_quick()
    if success:
        print("\n📋 To deploy:")
        print("1. Run: python scripts/setup_cron.py")
        print("2. Or: python scripts/scheduler.py")
        print("3. Monitor: tail -f logs/scheduler.log")
    sys.exit(0 if success else 1)