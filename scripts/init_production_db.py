#!/usr/bin/env python3
"""
Initialize Production PostgreSQL Database
"""
import sys
import os
sys.path.append('.')

from config.production import DATABASE_URI
from src.database.models_migration import init_db_and_create
from sqlalchemy import create_engine, text

def init_production_database():
    """Initialize production PostgreSQL database with proper configuration"""
    print("🐘 Initializing Production PostgreSQL Database")
    print("=" * 50)
    
    print(f"Database URI: {DATABASE_URI}")
    
    try:
        # Test connection first
        engine = create_engine(DATABASE_URI)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connected to: {version}")
        
        # Initialize database with all tables and indexes
        print("\n📋 Creating tables and indexes...")
        engine = init_db_and_create(DATABASE_URI)
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
            
            print(f"\n📊 Created {len(tables)} tables:")
            for table in tables:
                print(f"  • {table}")
        
        # Verify indexes were created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT indexname, tablename
                FROM pg_indexes 
                WHERE schemaname = 'public' 
                AND indexname LIKE 'idx_%'
                ORDER BY tablename, indexname;
            """))
            indexes = result.fetchall()
            
            print(f"\n🔍 Created {len(indexes)} indexes:")
            for index_name, table_name in indexes:
                print(f"  • {index_name} on {table_name}")
        
        print(f"\n🎉 Production database initialized successfully!")
        print(f"🔗 Database URI: {DATABASE_URI}")
        
        return engine
        
    except Exception as e:
        print(f"❌ Failed to initialize production database: {e}")
        return None

if __name__ == "__main__":
    init_production_database()