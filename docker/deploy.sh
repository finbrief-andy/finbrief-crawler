#!/bin/bash

# FinBrief Docker Deployment Script
# Handles production deployment, updates, and maintenance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
LOG_FILE="./deploy.log"

# Functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        warning "Environment file not found. Creating from example..."
        if [[ -f ".env.example" ]]; then
            cp .env.example "$ENV_FILE"
            warning "Please edit $ENV_FILE with your configuration"
            read -p "Press Enter to continue after editing $ENV_FILE..."
        else
            error "No .env.example file found"
        fi
    fi
    
    success "Prerequisites check completed"
}

# Validate environment configuration
validate_config() {
    info "Validating configuration..."
    
    # Check required environment variables
    source "$ENV_FILE"
    
    required_vars=(
        "DATABASE_URI"
        "SECRET_KEY"
        "POSTGRES_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Required environment variable $var is not set in $ENV_FILE"
        fi
    done
    
    # Check secret key length
    if [[ ${#SECRET_KEY} -lt 32 ]]; then
        error "SECRET_KEY must be at least 32 characters long"
    fi
    
    # Warn about default passwords
    if [[ "$POSTGRES_PASSWORD" == "finbrief_secure_password_change_me" ]]; then
        warning "You are using the default PostgreSQL password. Please change it in $ENV_FILE"
    fi
    
    if [[ -z "$OPENAI_API_KEY" || "$OPENAI_API_KEY" == "your-openai-api-key-here" ]]; then
        warning "OpenAI API key not set. Strategy generation will not work."
    fi
    
    success "Configuration validation completed"
}

# Build Docker images
build_images() {
    info "Building Docker images..."
    
    export BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    export VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    docker-compose -f "$COMPOSE_FILE" build --no-cache finbrief-api finbrief-crawler
    
    success "Docker images built successfully"
}

# Create backup
backup_data() {
    if [[ "$1" == "skip" ]]; then
        info "Skipping backup as requested"
        return
    fi
    
    info "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        BACKUP_FILE="$BACKUP_DIR/finbrief_db_$(date +%Y%m%d_%H%M%S).sql"
        
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump \
            -U "${POSTGRES_USER:-finbrief}" \
            -d "${POSTGRES_DB:-finbrief_prod}" \
            > "$BACKUP_FILE"
        
        if [[ -f "$BACKUP_FILE" && -s "$BACKUP_FILE" ]]; then
            success "Database backup created: $BACKUP_FILE"
        else
            warning "Database backup may have failed"
        fi
    else
        warning "PostgreSQL container not running, skipping database backup"
    fi
    
    # Backup volumes
    info "Backing up Docker volumes..."
    docker run --rm \
        -v finbrief-crawler_app_logs:/data/logs \
        -v finbrief-crawler_vector_data:/data/vectors \
        -v "$PWD/$BACKUP_DIR:/backup" \
        alpine tar czf "/backup/volumes_$(date +%Y%m%d_%H%M%S).tar.gz" -C /data .
    
    success "Backup completed"
}

# Deploy services
deploy() {
    info "Starting deployment..."
    
    # Pull latest images (if using pre-built images)
    # docker-compose -f "$COMPOSE_FILE" pull
    
    # Start core services first
    info "Starting database..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres
    
    # Wait for database to be ready
    info "Waiting for database to be ready..."
    timeout=60
    while ! docker-compose -f "$COMPOSE_FILE" exec postgres pg_isready -U "${POSTGRES_USER:-finbrief}" > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            error "Database failed to start within 60 seconds"
        fi
    done
    
    # Initialize database
    info "Initializing database..."
    docker-compose -f "$COMPOSE_FILE" run --rm finbrief-api init-db
    
    # Start all services
    info "Starting all services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    info "Waiting for services to be healthy..."
    sleep 10
    
    # Check service health
    check_health
    
    success "Deployment completed successfully"
}

# Check service health
check_health() {
    info "Checking service health..."
    
    services=("postgres" "finbrief-api" "finbrief-crawler")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            success "$service is running"
            
            # Check health status if available
            health=$(docker-compose -f "$COMPOSE_FILE" ps "$service" | grep "healthy" || true)
            if [[ -n "$health" ]]; then
                success "$service is healthy"
            fi
        else
            error "$service is not running"
        fi
    done
    
    # Test API endpoint
    info "Testing API endpoint..."
    sleep 5
    
    if curl -f http://localhost:${API_PORT:-8000}/monitoring/ping > /dev/null 2>&1; then
        success "API is responding"
    else
        warning "API is not responding on port ${API_PORT:-8000}"
    fi
}

# Update deployment
update() {
    info "Updating deployment..."
    
    # Create backup unless skipped
    backup_data "$1"
    
    # Pull latest code
    if [[ -d ".git" ]]; then
        info "Pulling latest code..."
        git pull origin main || warning "Failed to pull latest code"
    fi
    
    # Rebuild images
    build_images
    
    # Rolling update
    info "Performing rolling update..."
    docker-compose -f "$COMPOSE_FILE" up -d --force-recreate --no-deps finbrief-api finbrief-crawler
    
    # Check health after update
    sleep 10
    check_health
    
    success "Update completed"
}

# Stop services
stop() {
    info "Stopping services..."
    docker-compose -f "$COMPOSE_FILE" down
    success "Services stopped"
}

# Clean up old resources
cleanup() {
    info "Cleaning up old resources..."
    
    # Remove unused Docker resources
    docker system prune -f
    
    # Clean up old backups (keep last 7 days)
    find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete 2>/dev/null || true
    find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete 2>/dev/null || true
    
    success "Cleanup completed"
}

# Show logs
logs() {
    service="${1:-finbrief-api}"
    lines="${2:-100}"
    
    info "Showing logs for $service (last $lines lines)..."
    docker-compose -f "$COMPOSE_FILE" logs --tail="$lines" -f "$service"
}

# Show status
status() {
    info "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Main execution
case "$1" in
    "deploy")
        check_prerequisites
        validate_config
        build_images
        backup_data "$2"
        deploy
        ;;
    "update")
        check_prerequisites
        validate_config
        update "$2"
        ;;
    "stop")
        stop
        ;;
    "start")
        check_prerequisites
        validate_config
        docker-compose -f "$COMPOSE_FILE" up -d
        check_health
        ;;
    "restart")
        stop
        sleep 5
        check_prerequisites
        validate_config
        docker-compose -f "$COMPOSE_FILE" up -d
        check_health
        ;;
    "backup")
        backup_data
        ;;
    "health")
        check_health
        ;;
    "logs")
        logs "$2" "$3"
        ;;
    "status")
        status
        ;;
    "cleanup")
        cleanup
        ;;
    "shell")
        service="${2:-finbrief-api}"
        docker-compose -f "$COMPOSE_FILE" exec "$service" /bin/bash
        ;;
    *)
        echo "FinBrief Docker Deployment Script"
        echo ""
        echo "Usage: $0 {command} [options]"
        echo ""
        echo "Commands:"
        echo "  deploy [skip-backup]   - Full deployment (build, backup, deploy)"
        echo "  update [skip-backup]   - Update deployment (pull, build, restart)"
        echo "  start                  - Start services"
        echo "  stop                   - Stop services"
        echo "  restart                - Restart services"
        echo "  backup                 - Create backup"
        echo "  health                 - Check service health"
        echo "  logs [service] [lines] - Show service logs"
        echo "  status                 - Show service status and resource usage"
        echo "  cleanup                - Clean up old resources"
        echo "  shell [service]        - Access service shell"
        echo ""
        echo "Examples:"
        echo "  $0 deploy              # Full deployment with backup"
        echo "  $0 deploy skip-backup  # Deploy without backup"
        echo "  $0 update              # Update with backup"
        echo "  $0 logs finbrief-api   # Show API logs"
        echo "  $0 shell postgres      # Access PostgreSQL shell"
        exit 1
        ;;
esac