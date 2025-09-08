#!/bin/bash
set -e

# FinBrief Docker Entrypoint Script
# Handles initialization, migrations, and service startup

echo "🚀 Starting FinBrief Financial News Crawler"
echo "Environment: ${ENVIRONMENT:-production}"
echo "Timestamp: $(date)"

# Function to wait for database
wait_for_db() {
    echo "⏳ Waiting for database connection..."
    
    if [ -n "$DATABASE_URI" ]; then
        python3 -c "
import os
import time
import sys
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        engine = create_engine('$DATABASE_URI', pool_pre_ping=True)
        conn = engine.connect()
        conn.close()
        print('✅ Database connection successful')
        break
    except OperationalError as e:
        retry_count += 1
        print(f'⏳ Database not ready ({retry_count}/{max_retries}): {e}')
        if retry_count >= max_retries:
            print('❌ Database connection failed after maximum retries')
            sys.exit(1)
        time.sleep(2)
"
    else
        echo "⚠️  No DATABASE_URI provided, skipping database check"
    fi
}

# Function to run database migrations
run_migrations() {
    echo "🔄 Running database migrations..."
    if [ -f "scripts/init_production_db.py" ]; then
        python3 scripts/init_production_db.py || {
            echo "❌ Migration failed"
            exit 1
        }
        echo "✅ Migrations completed"
    else
        echo "⚠️  No migration script found, skipping"
    fi
}

# Function to initialize monitoring
init_monitoring() {
    echo "📊 Initializing monitoring system..."
    
    # Create log directories with proper permissions
    mkdir -p /app/logs/metrics
    
    # Test monitoring system
    python3 -c "
from src.monitoring.logger import get_logger
from src.monitoring.metrics import get_metrics
from src.monitoring.health import get_health_monitor
import asyncio

try:
    # Initialize components
    logger = get_logger('docker-init')
    metrics = get_metrics()
    health_monitor = get_health_monitor()
    
    logger.info('Docker container starting', environment='$ENVIRONMENT', container_id='$(hostname)')
    print('✅ Monitoring system initialized')
    
except Exception as e:
    print(f'⚠️  Monitoring initialization warning: {e}')
"
}

# Function to verify dependencies
verify_dependencies() {
    echo "🔍 Verifying dependencies..."
    
    python3 -c "
import sys
required_modules = [
    'fastapi',
    'sqlalchemy', 
    'psycopg2',
    'openai',
    'requests',
    'transformers'
]

optional_modules = [
    ('chromadb', 'Vector search functionality'),
    ('sentence_transformers', 'Enhanced NLP processing'),
    ('scikit_learn', 'Machine learning features'),
    ('psutil', 'System monitoring')
]

missing_required = []
missing_optional = []

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing_required.append(module)

for module, description in optional_modules:
    try:
        __import__(module)
    except ImportError:
        missing_optional.append((module, description))

if missing_required:
    print(f'❌ Missing required modules: {missing_required}')
    sys.exit(1)

if missing_optional:
    print('⚠️  Missing optional modules:')
    for module, desc in missing_optional:
        print(f'   - {module}: {desc}')

print('✅ Dependency verification completed')
"
}

# Function to start the server
start_server() {
    echo "🌐 Starting FinBrief API server..."
    
    # Set default values if not provided
    HOST=${HOST:-"0.0.0.0"}
    PORT=${PORT:-8000}
    WORKERS=${WORKERS:-1}
    
    if [ "$ENVIRONMENT" = "development" ]; then
        echo "🛠️  Starting in development mode"
        exec uvicorn scripts.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --reload \
            --log-level debug
    else
        echo "🏭 Starting in production mode with $WORKERS workers"
        exec gunicorn scripts.main:app \
            --bind "$HOST:$PORT" \
            --workers "$WORKERS" \
            --worker-class uvicorn.workers.UvicornWorker \
            --worker-connections 1000 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --timeout 30 \
            --keepalive 5 \
            --access-logfile - \
            --error-logfile - \
            --log-level info \
            --preload
    fi
}

# Function to start the crawler
start_crawler() {
    echo "📰 Starting FinBrief news crawler..."
    
    # Check if scheduler should run
    if [ "${ENABLE_SCHEDULER:-true}" = "true" ]; then
        echo "⏰ Starting with scheduler enabled"
        exec python3 scripts/scheduler.py
    else
        echo "🔄 Running one-time crawl"
        exec python3 -c "
from src.crawlers.unified_pipeline import UnifiedNewsPipeline
from scripts.main import get_db

# Run single crawl iteration
pipeline = UnifiedNewsPipeline()
db_gen = get_db()
session = next(db_gen)

try:
    results = pipeline.run_full_pipeline(session)
    print(f'✅ Crawl completed: {results}')
except Exception as e:
    print(f'❌ Crawl failed: {e}')
    exit(1)
finally:
    session.close()
"
    fi
}

# Function to run database initialization
init_db() {
    echo "🗄️  Initializing database..."
    wait_for_db
    run_migrations
    echo "✅ Database initialization completed"
}

# Function to run tests
run_tests() {
    echo "🧪 Running test suite..."
    
    # Set test environment
    export ENVIRONMENT=test
    export LOG_LEVEL=ERROR
    
    # Run tests
    if [ -d "tests" ]; then
        python3 -m pytest tests/ -v --tb=short || {
            echo "❌ Tests failed"
            exit 1
        }
    else
        echo "⚠️  No tests directory found, running basic import tests"
        python3 -c "
from src.crawlers.unified_pipeline import UnifiedNewsPipeline
from src.services.strategy_generator import StrategyGenerator
from src.monitoring.logger import get_logger
print('✅ Basic imports successful')
"
    fi
    
    echo "✅ Tests completed"
}

# Main execution logic
case "$1" in
    "server")
        verify_dependencies
        wait_for_db
        init_monitoring
        start_server
        ;;
    "crawler")
        verify_dependencies
        wait_for_db
        init_monitoring
        start_crawler
        ;;
    "init-db")
        init_db
        ;;
    "migrate")
        wait_for_db
        run_migrations
        ;;
    "test")
        run_tests
        ;;
    "shell")
        echo "🐚 Starting interactive shell..."
        exec python3 -c "
import sys
sys.path.append('/app')
from scripts.main import get_db
from src.crawlers.unified_pipeline import UnifiedNewsPipeline
from src.services.strategy_generator import StrategyGenerator
from src.monitoring.logger import get_logger

print('FinBrief Interactive Shell')
print('Available objects: get_db, UnifiedNewsPipeline, StrategyGenerator, get_logger')

import code
code.interact(local=locals())
"
        ;;
    "health-check")
        echo "🏥 Running health check..."
        python3 -c "
import asyncio
from src.monitoring.health import check_system_health

async def main():
    try:
        health = await check_system_health()
        print(f'Health Status: {health[\"status\"]}')
        print(f'Message: {health[\"message\"]}')
        if health['status'] != 'healthy':
            exit(1)
    except Exception as e:
        print(f'Health check failed: {e}')
        exit(1)

asyncio.run(main())
"
        ;;
    *)
        echo "Usage: $0 {server|crawler|init-db|migrate|test|shell|health-check}"
        echo ""
        echo "Commands:"
        echo "  server      - Start the FastAPI web server"
        echo "  crawler     - Start the news crawler"
        echo "  init-db     - Initialize database with tables and indexes"
        echo "  migrate     - Run database migrations only"
        echo "  test        - Run the test suite"
        echo "  shell       - Start interactive Python shell"
        echo "  health-check- Run system health check"
        echo ""
        echo "Environment Variables:"
        echo "  DATABASE_URI     - PostgreSQL connection string"
        echo "  OPENAI_API_KEY   - OpenAI API key for strategy generation"
        echo "  ENVIRONMENT      - Environment (development|production|test)"
        echo "  HOST             - Server host (default: 0.0.0.0)"
        echo "  PORT             - Server port (default: 8000)"
        echo "  WORKERS          - Number of worker processes (default: 1)"
        echo "  ENABLE_SCHEDULER - Enable automatic crawling (default: true)"
        echo "  LOG_LEVEL        - Logging level (DEBUG|INFO|WARNING|ERROR)"
        exit 1
        ;;
esac