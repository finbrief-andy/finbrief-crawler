#!/usr/bin/env python3
"""
Scheduled pipeline execution system for FinBrief.
Implements automated news collection and strategy generation.
"""
import os
import sys
import logging
import time
import signal
from datetime import datetime, timedelta
from typing import Dict, Any
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crawlers.unified_pipeline import UnifiedNewsPipeline
from src.services.strategy_generator import StrategyGenerator
from src.database.models_migration import init_db_and_create, StrategyHorizonEnum, MarketEnum
from sqlalchemy.orm import sessionmaker


class PipelineScheduler:
    """Scheduler for automated pipeline execution"""
    
    def __init__(self, database_uri: str = None):
        self.database_uri = database_uri or os.getenv("DATABASE_URI", "sqlite:///./finbrief.db")
        self.running = False
        self.last_run = None
        self.run_interval = int(os.getenv("PIPELINE_INTERVAL_MINUTES", "30")) * 60  # Default 30 minutes
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler("logs/scheduler.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pipeline = None
        self.strategy_generator = None
        self.engine = None
        self.SessionLocal = None
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_success': None,
            'last_failure': None
        }
    
    def initialize(self):
        """Initialize database and components"""
        try:
            self.logger.info("Initializing scheduler components...")
            
            # Initialize database
            self.engine = init_db_and_create(self.database_uri)
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Initialize pipeline and strategy generator
            self.pipeline = UnifiedNewsPipeline(self.database_uri)
            self.strategy_generator = StrategyGenerator()
            
            self.logger.info("Scheduler initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scheduler: {e}")
            return False
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the news pipeline with error handling and retry logic"""
        run_start = datetime.utcnow()
        self.stats['total_runs'] += 1
        
        try:
            self.logger.info(f"Starting pipeline run #{self.stats['total_runs']}")
            
            # Run pipeline for all sources
            results = self.pipeline.run_pipeline()
            
            # Calculate totals
            total_inserted = sum(r.get('inserted', 0) for r in results.values())
            total_skipped = sum(r.get('skipped', 0) for r in results.values())
            successful_sources = sum(1 for r in results.values() if r.get('status') == 'success')
            failed_sources = sum(1 for r in results.values() if r.get('status') == 'error')
            
            self.logger.info(f"Pipeline completed: {total_inserted} inserted, {total_skipped} skipped")
            self.logger.info(f"Sources: {successful_sources} successful, {failed_sources} failed")
            
            # Generate strategies if we have new data
            if total_inserted > 0:
                self.logger.info("Generating strategies...")
                strategy_results = self.generate_strategies()
                results['strategy_generation'] = strategy_results
            
            # Update statistics
            if successful_sources > 0:
                self.stats['successful_runs'] += 1
                self.stats['last_success'] = run_start
            else:
                self.stats['failed_runs'] += 1
                self.stats['last_failure'] = run_start
            
            self.last_run = run_start
            
            # Log results
            run_summary = {
                'timestamp': run_start.isoformat(),
                'duration_seconds': (datetime.utcnow() - run_start).total_seconds(),
                'total_inserted': total_inserted,
                'total_skipped': total_skipped,
                'successful_sources': successful_sources,
                'failed_sources': failed_sources,
                'results': results
            }
            
            with open('logs/pipeline_runs.jsonl', 'a') as f:
                f.write(json.dumps(run_summary) + '\n')
            
            return run_summary
            
        except Exception as e:
            self.logger.error(f"Pipeline run failed: {e}")
            self.stats['failed_runs'] += 1
            self.stats['last_failure'] = run_start
            
            error_summary = {
                'timestamp': run_start.isoformat(),
                'error': str(e),
                'duration_seconds': (datetime.utcnow() - run_start).total_seconds()
            }
            
            with open('logs/pipeline_runs.jsonl', 'a') as f:
                f.write(json.dumps(error_summary) + '\n')
            
            return error_summary
    
    def generate_strategies(self) -> Dict[str, Any]:
        """Generate investment strategies"""
        try:
            session = self.SessionLocal()
            strategies_created = 0
            
            # Generate daily strategy for global market
            strategy = self.strategy_generator.create_strategy(
                session, StrategyHorizonEnum.daily, MarketEnum.global_market
            )
            if strategy:
                strategies_created += 1
                self.logger.info(f"Created daily global strategy: {strategy.title}")
            
            # Generate daily strategy for Vietnam market if we have VN news
            from src.database.models_migration import News
            vn_news_count = session.query(News).filter(News.market == MarketEnum.vn).count()
            if vn_news_count > 0:
                strategy = self.strategy_generator.create_strategy(
                    session, StrategyHorizonEnum.daily, MarketEnum.vn
                )
                if strategy:
                    strategies_created += 1
                    self.logger.info(f"Created daily VN strategy: {strategy.title}")
            
            session.close()
            
            return {
                'strategies_created': strategies_created,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Strategy generation failed: {e}")
            return {
                'strategies_created': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def should_run(self) -> bool:
        """Check if pipeline should run based on schedule"""
        if self.last_run is None:
            return True
        
        time_since_last = (datetime.utcnow() - self.last_run).total_seconds()
        return time_since_last >= self.run_interval
    
    def start(self):
        """Start the scheduler"""
        if not self.initialize():
            self.logger.error("Failed to initialize scheduler")
            return False
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        self.running = True
        self.logger.info(f"Scheduler started with {self.run_interval/60:.1f} minute intervals")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
        try:
            while self.running:
                if self.should_run():
                    self.run_pipeline()
                    
                    # Wait before checking again
                    time.sleep(60)  # Check every minute
                else:
                    # Sleep for a shorter interval to check for shutdown
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
        finally:
            self.stop()
    
    def stop(self, signum=None, frame=None):
        """Stop the scheduler gracefully"""
        self.logger.info("Stopping scheduler...")
        self.running = False
    
    def status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            'running': self.running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'run_interval_minutes': self.run_interval / 60,
            'stats': self.stats,
            'next_run_in_seconds': max(0, self.run_interval - (
                (datetime.utcnow() - self.last_run).total_seconds() 
                if self.last_run else 0
            ))
        }


def main():
    """Main entry point for scheduler"""
    scheduler = PipelineScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()