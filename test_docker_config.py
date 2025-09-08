#!/usr/bin/env python3
"""
Docker Configuration Test Suite
Tests Docker configuration files, environment setup, and deployment readiness.
"""
import os
import yaml
import json
import subprocess
from pathlib import Path
import sys


def test_dockerfile():
    """Test Dockerfile configuration"""
    print("🐳 Testing Dockerfile Configuration")
    print("=" * 50)
    
    # Test production Dockerfile
    dockerfile_path = Path("Dockerfile")
    if dockerfile_path.exists():
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check essential components
        checks = [
            ("FROM python:", "Python base image"),
            ("WORKDIR /app", "Working directory set"),
            ("COPY requirements", "Requirements copied"),
            ("RUN pip install", "Dependencies installed"),
            ("EXPOSE 8000", "Port exposed"),
            ("HEALTHCHECK", "Health check configured"),
            ("USER finbrief", "Non-root user"),
            ("ENTRYPOINT", "Entrypoint configured")
        ]
        
        passed = 0
        for check, description in checks:
            if check in content:
                print(f"✅ {description}")
                passed += 1
            else:
                print(f"❌ Missing: {description}")
        
        print(f"Dockerfile checks: {passed}/{len(checks)} passed")
        
        # Test development Dockerfile
        dev_dockerfile = Path("Dockerfile.dev")
        if dev_dockerfile.exists():
            print("✅ Development Dockerfile exists")
        else:
            print("⚠️  Development Dockerfile not found")
        
        return passed >= len(checks) * 0.8
    else:
        print("❌ Dockerfile not found")
        return False


def test_docker_compose():
    """Test Docker Compose configuration"""
    print("\n🐙 Testing Docker Compose Configuration")
    print("=" * 50)
    
    # Test production docker-compose
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    try:
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        print("✅ Docker Compose file is valid YAML")
        
        # Check services
        services = compose_config.get('services', {})
        expected_services = ['postgres', 'finbrief-api', 'finbrief-crawler']
        
        service_checks = 0
        for service in expected_services:
            if service in services:
                print(f"✅ Service '{service}' configured")
                service_checks += 1
                
                # Check service configuration
                service_config = services[service]
                if 'healthcheck' in service_config:
                    print(f"  ✅ Health check configured for {service}")
                if 'restart' in service_config:
                    print(f"  ✅ Restart policy set for {service}")
                if 'environment' in service_config:
                    print(f"  ✅ Environment variables configured for {service}")
            else:
                print(f"❌ Service '{service}' missing")
        
        # Check volumes
        volumes = compose_config.get('volumes', {})
        if volumes:
            print(f"✅ {len(volumes)} volumes configured")
        
        # Check networks
        networks = compose_config.get('networks', {})
        if networks:
            print(f"✅ {len(networks)} networks configured")
        
        # Test development compose
        dev_compose = Path("docker-compose.dev.yml")
        if dev_compose.exists():
            print("✅ Development Docker Compose exists")
            
            with open(dev_compose, 'r') as f:
                dev_config = yaml.safe_load(f)
            print("✅ Development Docker Compose is valid YAML")
        else:
            print("⚠️  Development Docker Compose not found")
        
        return service_checks >= len(expected_services)
        
    except yaml.YAMLError as e:
        print(f"❌ Docker Compose YAML error: {e}")
        return False


def test_environment_config():
    """Test environment configuration"""
    print("\n⚙️  Testing Environment Configuration")
    print("=" * 50)
    
    # Test .env.example
    env_example = Path(".env.example")
    if env_example.exists():
        with open(env_example, 'r') as f:
            env_content = f.read()
        
        print("✅ Environment example file exists")
        
        # Check required variables
        required_vars = [
            "DATABASE_URI",
            "SECRET_KEY", 
            "POSTGRES_PASSWORD",
            "ENVIRONMENT",
            "LOG_LEVEL",
            "API_PORT"
        ]
        
        var_checks = 0
        for var in required_vars:
            if var in env_content:
                print(f"✅ {var} configured")
                var_checks += 1
            else:
                print(f"❌ Missing: {var}")
        
        print(f"Environment variables: {var_checks}/{len(required_vars)} found")
        
        return var_checks >= len(required_vars) * 0.8
    else:
        print("❌ .env.example not found")
        return False


def test_deployment_scripts():
    """Test deployment scripts"""
    print("\n🚀 Testing Deployment Scripts")
    print("=" * 50)
    
    # Test entrypoint script
    entrypoint = Path("docker/entrypoint.sh")
    if entrypoint.exists():
        print("✅ Entrypoint script exists")
        
        if os.access(entrypoint, os.X_OK):
            print("✅ Entrypoint script is executable")
        else:
            print("⚠️  Entrypoint script not executable")
        
        with open(entrypoint, 'r') as f:
            script_content = f.read()
        
        # Check script features
        script_checks = [
            ("wait_for_db", "Database wait function"),
            ("run_migrations", "Migration function"),
            ("start_server", "Server start function"),
            ("start_crawler", "Crawler start function"),
            ("health-check", "Health check command")
        ]
        
        script_passed = 0
        for check, description in script_checks:
            if check in script_content:
                print(f"✅ {description}")
                script_passed += 1
            else:
                print(f"❌ Missing: {description}")
        
    else:
        print("❌ Entrypoint script not found")
        script_passed = 0
        script_checks = []
    
    # Test deployment script
    deploy_script = Path("docker/deploy.sh")
    if deploy_script.exists():
        print("✅ Deployment script exists")
        
        if os.access(deploy_script, os.X_OK):
            print("✅ Deployment script is executable")
        else:
            print("⚠️  Deployment script not executable")
    else:
        print("❌ Deployment script not found")
    
    return script_passed >= len(script_checks) * 0.7 if script_checks else False


def test_docker_ignore():
    """Test Docker ignore configuration"""
    print("\n🙈 Testing Docker Ignore Configuration")
    print("=" * 50)
    
    dockerignore = Path(".dockerignore")
    if dockerignore.exists():
        with open(dockerignore, 'r') as f:
            ignore_content = f.read()
        
        print("✅ .dockerignore file exists")
        
        # Check common ignore patterns
        ignore_checks = [
            (".git", "Git files ignored"),
            ("*.md", "Documentation ignored"),
            ("__pycache__", "Python cache ignored"),
            ("logs/", "Log files ignored"),
            (".env", "Environment files ignored"),
            ("tests/", "Test files ignored")
        ]
        
        ignore_passed = 0
        for pattern, description in ignore_checks:
            if pattern in ignore_content:
                print(f"✅ {description}")
                ignore_passed += 1
            else:
                print(f"⚠️  {description} not found")
        
        return ignore_passed >= len(ignore_checks) * 0.6
    else:
        print("❌ .dockerignore not found")
        return False


def test_requirements():
    """Test Python requirements"""
    print("\n📦 Testing Requirements Configuration")
    print("=" * 50)
    
    # Test main requirements
    main_req = Path("requirements.txt")
    if main_req.exists():
        print("✅ Main requirements.txt exists")
    else:
        print("❌ requirements.txt not found")
        return False
    
    # Test Docker requirements
    docker_req = Path("requirements-docker.txt")
    if docker_req.exists():
        with open(docker_req, 'r') as f:
            docker_content = f.read()
        
        print("✅ Docker requirements exist")
        
        # Check Docker-specific packages
        docker_packages = [
            "gunicorn",
            "uvicorn",
            "psutil"
        ]
        
        pkg_checks = 0
        for package in docker_packages:
            if package in docker_content:
                print(f"✅ {package} included")
                pkg_checks += 1
            else:
                print(f"⚠️  {package} missing")
        
        return pkg_checks >= len(docker_packages) * 0.7
    else:
        print("❌ requirements-docker.txt not found")
        return False


def test_docker_available():
    """Test if Docker is available"""
    print("\n🐋 Testing Docker Availability")
    print("=" * 50)
    
    try:
        # Test Docker command
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
            docker_available = True
        else:
            print("❌ Docker not available")
            docker_available = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker command not found")
        docker_available = False
    
    try:
        # Test Docker Compose
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker Compose available: {result.stdout.strip()}")
            compose_available = True
        else:
            # Try newer docker compose command
            result = subprocess.run(['docker', 'compose', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Docker Compose (plugin) available: {result.stdout.strip()}")
                compose_available = True
            else:
                print("❌ Docker Compose not available")
                compose_available = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker Compose command not found")
        compose_available = False
    
    return docker_available and compose_available


def test_nginx_config():
    """Test Nginx configuration"""
    print("\n🌐 Testing Nginx Configuration")
    print("=" * 50)
    
    nginx_conf = Path("docker/nginx/nginx.conf")
    if nginx_conf.exists():
        with open(nginx_conf, 'r') as f:
            nginx_content = f.read()
        
        print("✅ Nginx configuration exists")
        
        # Check Nginx features
        nginx_checks = [
            ("upstream finbrief_api", "Upstream configuration"),
            ("limit_req_zone", "Rate limiting"),
            ("ssl_certificate", "SSL configuration"),
            ("gzip on", "Compression enabled"),
            ("proxy_pass", "Proxy configuration"),
            ("add_header X-Frame-Options", "Security headers")
        ]
        
        nginx_passed = 0
        for check, description in nginx_checks:
            if check in nginx_content:
                print(f"✅ {description}")
                nginx_passed += 1
            else:
                print(f"⚠️  {description} not configured")
        
        return nginx_passed >= len(nginx_checks) * 0.6
    else:
        print("⚠️  Nginx configuration not found (optional)")
        return True  # Optional component


def main():
    """Run Docker configuration tests"""
    print("🐳 Docker Configuration Test Suite")
    print("=" * 70)
    print(f"Test started at: {os.popen('date').read().strip()}")
    print()
    
    tests = [
        ("Docker Availability", test_docker_available),
        ("Dockerfile Configuration", test_dockerfile),
        ("Docker Compose Configuration", test_docker_compose), 
        ("Environment Configuration", test_environment_config),
        ("Deployment Scripts", test_deployment_scripts),
        ("Docker Ignore Configuration", test_docker_ignore),
        ("Requirements Configuration", test_requirements),
        ("Nginx Configuration", test_nginx_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
        print()
    
    print("=" * 70)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("🎉 Docker configuration ready for deployment!")
        print()
        print("📋 Available Deployment Commands:")
        print("✅ Production deployment: ./docker/deploy.sh deploy")
        print("✅ Development environment: docker-compose -f docker-compose.dev.yml up -d")
        print("✅ Service management: ./docker/deploy.sh {start|stop|restart|status}")
        print("✅ Health monitoring: ./docker/deploy.sh health")
        print("✅ Log viewing: ./docker/deploy.sh logs [service]")
        
        print("\n🚀 Quick Start Guide:")
        print("1. Copy environment: cp .env.example .env")
        print("2. Edit configuration: nano .env")
        print("3. Deploy: ./docker/deploy.sh deploy")
        print("4. Check status: ./docker/deploy.sh status")
        print("5. View API docs: http://localhost:8000/docs")
        
        print("\n💡 Next Steps:")
        print("- Set up SSL certificates for production")
        print("- Configure monitoring dashboards")
        print("- Set up automated backups")
        print("- Configure CI/CD pipeline")
        
    else:
        print("⚠️  Docker configuration needs attention")
        print("💡 Fix failing tests before deployment")
        
        if not test_docker_available():
            print("\n🔧 Docker Installation:")
            print("- Install Docker: https://docs.docker.com/get-docker/")
            print("- Install Docker Compose: https://docs.docker.com/compose/install/")
    
    return passed >= total * 0.6


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)