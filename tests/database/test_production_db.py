#!/usr/bin/env python3
"""
Test Production PostgreSQL Database with Real Data
"""
import sys
import os
sys.path.append('.')

from config.production import DATABASE_URI
from src.database.models_migration import init_db_and_create
from src.crawlers.unified_pipeline import UnifiedNewsPipeline
from sqlalchemy.orm import sessionmaker

def test_production_database():
    """Test production PostgreSQL database with real news data"""
    print("🧪 Testing Production PostgreSQL Database")
    print("=" * 50)
    
    print(f"Database URI: {DATABASE_URI}")
    
    # Initialize database connection
    engine = init_db_and_create(DATABASE_URI)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Test 1: Run unified pipeline
        print("\n📰 Test 1: Running Unified News Pipeline")
        print("-" * 30)
        
        pipeline = UnifiedNewsPipeline()
        results = pipeline.run_pipeline(session, max_articles_per_source=5)
        
        print(f"✅ Pipeline Results:")
        print(f"  • Total articles processed: {results.get('total_articles', 0)}")
        print(f"  • New articles: {results.get('new_articles', 0)}")
        print(f"  • Duplicates filtered: {results.get('duplicates_filtered', 0)}")
        print(f"  • Analysis completed: {results.get('analyses_created', 0)}")
        
        # Test 2: Query with PostgreSQL-specific features
        print("\n🔍 Test 2: Testing PostgreSQL JSONB Queries")
        print("-" * 30)
        
        from src.database.models_migration import News, Analysis
        from sqlalchemy import text
        
        # Test JSONB query
        jsonb_query = session.execute(text("""
            SELECT headline, tags->>'sentiment' as sentiment, source
            FROM news 
            WHERE tags ? 'sentiment' 
            ORDER BY published_at DESC 
            LIMIT 3;
        """))
        
        jsonb_results = jsonb_query.fetchall()
        print(f"✅ JSONB Query Results: {len(jsonb_results)} articles")
        for row in jsonb_results:
            print(f"  • {row[0][:50]}... (Sentiment: {row[1]}, Source: {row[2]})")
        
        # Test 3: Full-text search
        print("\n🔍 Test 3: Testing Full-Text Search")
        print("-" * 30)
        
        search_query = session.execute(text("""
            SELECT headline, source, published_at
            FROM news 
            WHERE to_tsvector('english', headline || ' ' || coalesce(content_summary, ''))
                  @@ to_tsquery('english', 'market | economy | stock')
            ORDER BY published_at DESC 
            LIMIT 3;
        """))
        
        search_results = search_query.fetchall()
        print(f"✅ Full-Text Search Results: {len(search_results)} articles")
        for row in search_results:
            print(f"  • {row[0][:60]}... ({row[1]}, {row[2].strftime('%Y-%m-%d %H:%M')})")
        
        # Test 4: Generate Strategy
        print("\n🤖 Test 4: Testing Strategy Generation")
        print("-" * 30)
        
        from src.services.strategy_generator import StrategyGenerator
        from src.database.models_migration import StrategyHorizonEnum, MarketEnum
        
        generator = StrategyGenerator()
        strategy = generator.create_strategy(
            session, 
            StrategyHorizonEnum.daily, 
            MarketEnum.global_market
        )
        
        if strategy:
            print(f"✅ Strategy Generated:")
            print(f"  • Title: {strategy.title}")
            print(f"  • Summary: {strategy.summary}")
            print(f"  • Confidence: {strategy.confidence_score}")
            print(f"  • Recommendations: {len(strategy.action_recommendations)} items")
        
        print(f"\n🎉 Production database testing completed successfully!")
        print(f"📊 Database is fully functional with PostgreSQL optimizations")
        
    except Exception as e:
        print(f"❌ Production database test failed: {e}")
        raise e
    finally:
        session.close()

if __name__ == "__main__":
    test_production_database()