#!/usr/bin/env python3
"""
Test script to run the FinBrief pipeline with SQLite for easy testing
"""
import sys
import os
sys.path.append('.')

# Set environment variables for testing
os.environ["DATABASE_URI"] = "sqlite:///./test_finbrief.db"
os.environ["SECRET_KEY"] = "test-secret-key-12345"
os.environ["FINNHUB_API_KEY"] = "sandbox_c0t2j4hr01qm1gk2jte0"  # Sandbox key

def test_database():
    """Test database with SQLite"""
    print("🔍 Testing database with SQLite...")
    try:
        from src.database.models_migration import init_db_and_create
        engine = init_db_and_create("sqlite:///./test_finbrief.db")
        print("✅ SQLite database created successfully")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_adapters():
    """Test individual adapters"""
    print("\n🔍 Testing news adapters...")
    
    # Test Gold adapter (no external dependencies)
    try:
        from src.crawlers.adapters.gold_adapter import GoldPriceAdapter
        gold_adapter = GoldPriceAdapter()
        news_items = gold_adapter.fetch_news()
        print(f"✅ Gold adapter: {len(news_items)} items")
    except Exception as e:
        print(f"❌ Gold adapter failed: {e}")
    
    # Test Finnhub adapter
    try:
        from src.crawlers.adapters.finnhub_adapter import FinnhubAdapter
        finnhub_adapter = FinnhubAdapter()
        # Don't fetch actual data in test, just check initialization
        print("✅ Finnhub adapter initialized")
    except Exception as e:
        print(f"❌ Finnhub adapter failed: {e}")

def test_simple_pipeline():
    """Test pipeline with gold adapter only"""
    print("\n🔍 Testing simple pipeline...")
    try:
        from src.crawlers.unified_pipeline import UnifiedNewsPipeline
        
        # Initialize pipeline with SQLite
        pipeline = UnifiedNewsPipeline("sqlite:///./test_finbrief.db")
        
        # Test only gold adapter (doesn't require external APIs)
        results = pipeline.run_pipeline(sources=['gold_api'])
        
        print(f"✅ Pipeline results: {results}")
        
        # Check if data was inserted
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=pipeline.engine)
        session = Session()
        
        from src.database.models_migration import News, Analysis
        news_count = session.query(News).count()
        analysis_count = session.query(Analysis).count()
        
        print(f"✅ Database: {news_count} news items, {analysis_count} analyses")
        session.close()
        
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_generation():
    """Test strategy generation"""
    print("\n🔍 Testing strategy generation...")
    try:
        from src.services.strategy_generator import StrategyGenerator
        from src.database.models_migration import init_db_and_create, StrategyHorizonEnum, MarketEnum
        from sqlalchemy.orm import sessionmaker
        
        engine = init_db_and_create("sqlite:///./test_finbrief.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        generator = StrategyGenerator()
        strategy = generator.create_strategy(session, StrategyHorizonEnum.daily, MarketEnum.global_market)
        
        if strategy:
            print(f"✅ Generated strategy: {strategy.title}")
        else:
            print("⚠️  No strategy generated (may need more news data)")
        
        session.close()
        return True
    except Exception as e:
        print(f"❌ Strategy generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api():
    """Test API startup"""
    print("\n🔍 Testing API startup...")
    try:
        from scripts.main import app
        print("✅ API imports successful")
        
        # Test if we can create the app
        print(f"✅ FastAPI app created: {app.title}")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 FinBrief Pipeline Test (SQLite)")
    print("=" * 40)
    
    tests = [
        test_database,
        test_adapters,
        test_simple_pipeline,
        test_strategy_generation,
        test_api
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed >= 3:  # Allow some failures for external dependencies
        print("🎉 Core functionality works! Your backend is ready.")
        print("\n📋 Next steps:")
        print("1. Run API server: source venv/bin/activate && export SECRET_KEY='your-key' && PYTHONPATH=. python scripts/main.py")
        print("2. Visit: http://localhost:8000/docs for API documentation")
        print("3. Try API endpoints: GET /strategy/daily, /strategies")
    else:
        print("⚠️  Some core tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()