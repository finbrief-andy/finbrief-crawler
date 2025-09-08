# Docker Deployment Guide

## Overview
FinBrief's Docker deployment configuration provides production-ready containerization with comprehensive orchestration, monitoring, and scaling capabilities. The system includes multiple deployment modes for development, staging, and production environments.

## Architecture

### Container Services

1. **FinBrief API Server** (`finbrief-api`)
   - FastAPI web server with Gunicorn/Uvicorn
   - Automatic health checks and monitoring
   - Horizontal scaling support
   - JWT authentication and API endpoints

2. **FinBrief News Crawler** (`finbrief-crawler`)
   - Automated news collection scheduler
   - Background processing with configurable intervals
   - Independent scaling and resource allocation
   - Comprehensive error handling and recovery

3. **PostgreSQL Database** (`postgres`)
   - Production-optimized configuration
   - Automatic backups and persistence
   - Connection pooling and performance tuning
   - Health monitoring and recovery

4. **Redis Cache** (`redis`) - Optional
   - Session storage and caching
   - Background task queue
   - Performance optimization
   - High availability configuration

5. **Nginx Reverse Proxy** (`nginx`) - Optional
   - Load balancing and SSL termination
   - Rate limiting and security headers
   - Static file serving
   - Production-grade performance

6. **Monitoring Stack** (`prometheus`, `grafana`) - Optional
   - Metrics collection and visualization
   - Alerting and notifications
   - Performance monitoring dashboards
   - Historical data analysis

## Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### 1. Clone and Configure
```bash
# Clone the repository
git clone <repository-url>
cd finbrief-crawler

# Copy environment configuration
cp .env.example .env

# Edit configuration
nano .env  # Set your database passwords, API keys, etc.
```

### 2. Production Deployment
```bash
# Make deployment script executable
chmod +x docker/deploy.sh

# Deploy with backup
./docker/deploy.sh deploy

# Or deploy without backup (first time)
./docker/deploy.sh deploy skip-backup
```

### 3. Verify Deployment
```bash
# Check service status
./docker/deploy.sh status

# Check health
./docker/deploy.sh health

# View logs
./docker/deploy.sh logs finbrief-api
```

## Configuration

### Environment Variables
```bash
# Core Settings
ENVIRONMENT=production
SECRET_KEY=your-32-character-secret-key-here
OPENAI_API_KEY=sk-your-openai-api-key

# Database
POSTGRES_DB=finbrief_prod
POSTGRES_USER=finbrief
POSTGRES_PASSWORD=secure_password_here
DATABASE_URI=postgresql://finbrief:secure_password_here@postgres:5432/finbrief_prod

# Server
API_PORT=8000
API_WORKERS=2
HTTP_PORT=80
HTTPS_PORT=443

# Monitoring
LOG_LEVEL=INFO
METRICS_RETENTION_HOURS=72
ENABLE_MONITORING_API=true

# Crawler
ENABLE_SCHEDULER=true
PIPELINE_INTERVAL_MINUTES=30
```

### Service Scaling
```yaml
# docker-compose.override.yml
services:
  finbrief-api:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## Deployment Modes

### Production Deployment
```bash
# Full production stack with monitoring
docker-compose up -d

# Production with Nginx proxy
docker-compose --profile nginx up -d

# Production with full monitoring stack
docker-compose --profile nginx --profile monitoring up -d
```

**Features:**
- Multi-worker API server with Gunicorn
- Automatic database migrations
- Health checks and automatic restarts
- Log aggregation and metrics collection
- SSL/TLS termination with Nginx
- Resource limits and monitoring

**Resource Requirements:**
- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ for full stack
- **Storage**: 20GB+ for logs and data
- **Network**: 1Gbps+ for high traffic

### Development Deployment
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# With crawler enabled
docker-compose -f docker-compose.dev.yml --profile crawler up -d

# With debugging tools
docker-compose -f docker-compose.dev.yml --profile tools up -d
```

**Features:**
- Hot code reloading with volume mounts
- Debug logging enabled
- Development database (isolated)
- Accessible container shells
- Testing and debugging tools

### Testing Deployment
```bash
# Run tests in containers
docker-compose -f docker-compose.dev.yml run --rm finbrief-tools test

# Interactive testing shell
docker-compose -f docker-compose.dev.yml run --rm finbrief-tools shell
```

## Service Management

### Deployment Script Commands
```bash
# Deployment
./docker/deploy.sh deploy              # Full deployment with backup
./docker/deploy.sh deploy skip-backup  # Deploy without backup
./docker/deploy.sh update              # Update deployment
./docker/deploy.sh update skip-backup  # Update without backup

# Service Control
./docker/deploy.sh start               # Start services
./docker/deploy.sh stop                # Stop services
./docker/deploy.sh restart             # Restart services

# Monitoring
./docker/deploy.sh health              # Check service health
./docker/deploy.sh status              # Show service status
./docker/deploy.sh logs [service]      # View service logs

# Maintenance
./docker/deploy.sh backup              # Create backup
./docker/deploy.sh cleanup             # Clean old resources
./docker/deploy.sh shell [service]     # Access service shell
```

### Manual Docker Compose Commands
```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d postgres finbrief-api

# Scale services
docker-compose up -d --scale finbrief-api=3

# View logs
docker-compose logs -f finbrief-api

# Execute commands
docker-compose exec finbrief-api /bin/bash
docker-compose exec postgres psql -U finbrief finbrief_prod

# Stop services
docker-compose down

# Stop and remove volumes (CAUTION: Data loss!)
docker-compose down -v
```

## Health Monitoring

### Built-in Health Checks
```bash
# API health check
curl http://localhost:8000/monitoring/ping

# Comprehensive health status
curl http://localhost:8000/monitoring/health

# Service-specific health
curl http://localhost:8000/monitoring/health/database
```

### Container Health Status
```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# View health check logs
docker inspect finbrief-api | grep -A 10 -B 5 Health
```

### Monitoring Endpoints
- **Health**: `http://localhost:8000/monitoring/ping`
- **Metrics**: `http://localhost:8000/monitoring/metrics/summary`
- **Status**: `http://localhost:8000/monitoring/health/detailed`
- **API Docs**: `http://localhost:8000/docs`

## Scaling and Performance

### Horizontal Scaling
```yaml
# Scale API servers
services:
  finbrief-api:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

```bash
# Using Docker Compose
docker-compose up -d --scale finbrief-api=3
```

### Resource Optimization
```yaml
# Resource limits
services:
  finbrief-api:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Database Tuning
```sql
-- PostgreSQL optimization (automatically applied)
shared_buffers = 256MB
effective_cache_size = 1GB
max_connections = 200
maintenance_work_mem = 64MB
```

### Performance Monitoring
```bash
# Container resource usage
docker stats

# Application metrics
curl http://localhost:8000/monitoring/metrics/performance

# Database performance
docker-compose exec postgres pg_stat_activity
```

## Backup and Recovery

### Automated Backups
```bash
# Manual backup
./docker/deploy.sh backup

# Automated backup (cron job)
0 2 * * * /path/to/finbrief/docker/deploy.sh backup
```

### Backup Contents
- **Database dump**: PostgreSQL full backup
- **Application logs**: Structured logs and metrics
- **Vector data**: Embeddings and search indexes
- **Configuration**: Environment and service configs

### Recovery Process
```bash
# Stop services
docker-compose down

# Restore database
docker-compose up -d postgres
docker-compose exec -T postgres psql -U finbrief finbrief_prod < backup.sql

# Restore volumes
docker run --rm -v finbrief_app_logs:/data -v /path/to/backup:/backup \
  alpine tar xzf /backup/volumes_backup.tar.gz -C /data

# Start services
docker-compose up -d
```

## Security

### Production Security Hardening
```yaml
# Security settings
services:
  finbrief-api:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
    user: "1000:1000"
```

### Network Security
```yaml
# Internal network isolation
networks:
  finbrief-network:
    internal: true
    driver: bridge
    
  finbrief-public:
    driver: bridge
```

### SSL/TLS Configuration
```bash
# Generate SSL certificates
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem
```

### Security Environment Variables
```bash
# Secrets management
SECRET_KEY=32-character-random-key
POSTGRES_PASSWORD=strong-database-password
JWT_SECRET_KEY=jwt-signing-key
```

## Troubleshooting

### Common Issues

1. **Database Connection Failures**
   ```bash
   # Check database status
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec finbrief-api python -c "
   from sqlalchemy import create_engine
   engine = create_engine('$DATABASE_URI')
   print('Connected successfully' if engine.connect() else 'Failed')
   "
   ```

2. **API Server Not Responding**
   ```bash
   # Check service logs
   docker-compose logs finbrief-api
   
   # Check port binding
   netstat -tlnp | grep 8000
   
   # Test health endpoint
   curl -v http://localhost:8000/monitoring/ping
   ```

3. **Memory Issues**
   ```bash
   # Monitor resource usage
   docker stats --no-stream
   
   # Check container limits
   docker inspect finbrief-api | grep -i memory
   
   # Increase memory limits
   # Edit docker-compose.yml memory settings
   ```

4. **Storage Issues**
   ```bash
   # Check disk space
   df -h
   docker system df
   
   # Clean up unused resources
   docker system prune -a
   
   # Check volume usage
   docker volume ls
   ```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run single container for debugging
docker-compose run --rm finbrief-api shell

# Interactive debugging session
docker-compose exec finbrief-api python -i
```

### Log Analysis
```bash
# Real-time log monitoring
docker-compose logs -f --tail=100

# Search logs
docker-compose logs finbrief-api 2>&1 | grep ERROR

# Export logs
docker-compose logs --no-color > finbrief-logs.txt
```

## Production Deployment Checklist

### Pre-Deployment
- [ ] Update `.env` with production values
- [ ] Set strong passwords and secrets
- [ ] Configure OpenAI API key
- [ ] Set up SSL certificates (if using HTTPS)
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerting

### Deployment
- [ ] Run `./docker/deploy.sh deploy`
- [ ] Verify all services are healthy
- [ ] Test API endpoints
- [ ] Check database connectivity
- [ ] Verify crawler is running
- [ ] Test monitoring endpoints

### Post-Deployment
- [ ] Set up automated backups
- [ ] Configure log rotation
- [ ] Set up monitoring dashboards
- [ ] Test disaster recovery procedures
- [ ] Document access credentials
- [ ] Set up alerting notifications

### Maintenance
- [ ] Regular security updates
- [ ] Database maintenance
- [ ] Log cleanup
- [ ] Performance monitoring
- [ ] Backup verification
- [ ] Capacity planning

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to server
      run: |
        ssh user@server "cd /path/to/finbrief && ./docker/deploy.sh update"
```

### GitLab CI Example
```yaml
deploy_production:
  stage: deploy
  script:
    - ssh user@server "cd /path/to/finbrief && ./docker/deploy.sh update"
  only:
    - main
```

---

*Last Updated: January 2025*
*Status: Production Ready - Comprehensive Docker Deployment System*