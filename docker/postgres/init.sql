-- PostgreSQL initialization script for FinBrief
-- Creates necessary extensions and optimizations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create database if it doesn't exist (handled by POSTGRES_DB env var)
-- The database is already created by the postgres Docker image

-- Set recommended PostgreSQL settings for production
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Reload configuration
SELECT pg_reload_conf();

-- Create a read-only user for monitoring (optional)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'finbrief_readonly') THEN
        CREATE ROLE finbrief_readonly WITH LOGIN PASSWORD 'readonly_password';
        GRANT CONNECT ON DATABASE finbrief_prod TO finbrief_readonly;
        GRANT USAGE ON SCHEMA public TO finbrief_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO finbrief_readonly;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO finbrief_readonly;
    END IF;
END
$$;