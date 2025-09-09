#!/usr/bin/env python3
"""Backfill vector store with existing news and analysis data"""
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import sessionmaker
from src.database.models_migration import init_db_and_create
from src.services.vector_store import get_vector_store

def backfill_vector_store():
    """Backfill vector store with existing data"""
    print("Backfilling vector store with existing data...")
    
    # Initialize database
    engine = init_db_and_create()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    # Get vector store instance
    vs = get_vector_store()
    
    if not vs.is_available():
        print("❌ Vector store not available")
        return
    
    print(f"Initial state - News: {len(vs.news_collection.get()['ids'])}, Analysis: {len(vs.analysis_collection.get()['ids'])}")
    
    try:
        # Backfill embeddings
        vs.backfill_embeddings(session, limit=100)  # Limit to 100 items for testing
        
        # Check final state
        news_count = len(vs.news_collection.get()['ids'])
        analysis_count = len(vs.analysis_collection.get()['ids'])
        
        print(f"Final state - News: {news_count}, Analysis: {analysis_count}")
        
        if news_count > 0:
            print("✅ Successfully backfilled vector store with news data")
        else:
            print("⚠️  No news data found to backfill")
            
        if analysis_count > 0:
            print("✅ Successfully backfilled vector store with analysis data")
        else:
            print("⚠️  No analysis data found to backfill")
            
    except Exception as e:
        print(f"❌ Error during backfill: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    backfill_vector_store()