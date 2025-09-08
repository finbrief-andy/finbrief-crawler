#!/usr/bin/env python3
"""
Setup script for CRON job to run FinBrief pipeline automatically.
"""
import os
import subprocess
import sys
from pathlib import Path


def setup_cron():
    """Setup CRON job for pipeline execution"""
    
    # Get absolute paths
    project_root = Path(__file__).parent.parent.absolute()
    scheduler_script = project_root / "scripts" / "run_pipeline_once.py"
    log_file = project_root / "logs" / "cron.log"
    
    # Ensure logs directory exists
    (project_root / "logs").mkdir(exist_ok=True)
    
    # Create the one-time execution script
    with open(scheduler_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
One-time pipeline execution for CRON.
"""
import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append("{project_root}")

# Set up environment
os.environ.setdefault("DATABASE_URI", "{os.getenv('DATABASE_URI', 'sqlite:///./finbrief.db')}")
os.environ.setdefault("SECRET_KEY", "{os.getenv('SECRET_KEY', 'fallback-key')}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("{log_file}"),
        logging.StreamHandler()
    ]
)

def main():
    """Run pipeline once"""
    try:
        from scripts.scheduler import PipelineScheduler
        
        scheduler = PipelineScheduler()
        if not scheduler.initialize():
            logging.error("Failed to initialize scheduler")
            return 1
        
        logging.info("Starting scheduled pipeline run")
        result = scheduler.run_pipeline()
        
        if 'error' in result:
            logging.error(f"Pipeline failed: {{result['error']}}")
            return 1
        else:
            logging.info(f"Pipeline completed successfully: {{result.get('total_inserted', 0)}} items inserted")
            return 0
            
    except Exception as e:
        logging.error(f"CRON execution failed: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
''')
    
    # Make script executable
    os.chmod(scheduler_script, 0o755)
    
    # CRON job entry (runs every 30 minutes)
    python_path = sys.executable
    cron_entry = f"*/30 * * * * cd {project_root} && {python_path} {scheduler_script} >> {log_file} 2>&1"
    
    print("CRON setup completed!")
    print(f"Script created: {scheduler_script}")
    print(f"Log file: {log_file}")
    print(f"\\nTo install the CRON job, run:")
    print(f"echo '{cron_entry}' | crontab -")
    print("\\nTo view current CRON jobs:")
    print("crontab -l")
    print("\\nTo remove CRON jobs:")
    print("crontab -r")
    
    return cron_entry


def setup_systemd_service():
    """Create systemd service file for long-running scheduler"""
    
    project_root = Path(__file__).parent.parent.absolute()
    python_path = sys.executable
    user = os.getenv('USER', 'ubuntu')
    
    service_content = f"""[Unit]
Description=FinBrief Pipeline Scheduler
After=network.target postgresql.service
Requires=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={project_root}
Environment=DATABASE_URI={os.getenv('DATABASE_URI', 'sqlite:///./finbrief.db')}
Environment=SECRET_KEY={os.getenv('SECRET_KEY', 'fallback-key')}
Environment=PIPELINE_INTERVAL_MINUTES=30
ExecStart={python_path} {project_root}/scripts/scheduler.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path("finbrief-scheduler.service")
    with open(service_file, 'w') as f:
        f.write(service_content)
    
    print(f"\\nSystemd service file created: {service_file}")
    print("To install and start the service:")
    print(f"sudo cp {service_file} /etc/systemd/system/")
    print("sudo systemctl daemon-reload")
    print("sudo systemctl enable finbrief-scheduler")
    print("sudo systemctl start finbrief-scheduler")
    print("\\nTo check service status:")
    print("sudo systemctl status finbrief-scheduler")
    print("To view logs:")
    print("sudo journalctl -u finbrief-scheduler -f")


def main():
    """Main setup function"""
    print("FinBrief Pipeline Scheduler Setup")
    print("=" * 40)
    
    choice = input("Choose setup method:\\n1. CRON job (runs every 30 minutes)\\n2. Systemd service (long-running daemon)\\n3. Both\\nChoice (1/2/3): ")
    
    if choice in ['1', '3']:
        print("\\n--- CRON Job Setup ---")
        setup_cron()
    
    if choice in ['2', '3']:
        print("\\n--- Systemd Service Setup ---")
        setup_systemd_service()
    
    print("\\n--- Manual Testing ---")
    print("To test the pipeline manually:")
    print("python scripts/scheduler.py")


if __name__ == "__main__":
    main()