#!/usr/bin/env python3
"""
Quick test script for FinBrief backend components
"""
import os
import sys
sys.path.append('.')

def test_database():
    """Test database initialization"""
    print("🔍 Testing database...")
    try:
        from src.database.models_migration import init_db_and_create
        engine = init_db_and_create()
        print("✅ Database schema created successfully")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_adapters():
    """Test news adapters"""
    print("\n🔍 Testing news adapters...")
    try:
        # Test Finnhub adapter
        from src.crawlers.adapters.finnhub_adapter import FinnhubAdapter
        finnhub = FinnhubAdapter()
        print("✅ Finnhub adapter initialized")
        
        # Test Gold adapter (doesn't require external dependencies)
        from src.crawlers.adapters.gold_adapter import GoldPriceAdapter
        gold = GoldPriceAdapter()
        print("✅ Gold adapter initialized")
        
        return True
    except Exception as e:
        print(f"❌ Adapters test failed: {e}")
        return False

def test_strategy_generator():
    """Test strategy generation"""
    print("\n🔍 Testing strategy generator...")
    try:
        from src.services.strategy_generator import StrategyGenerator
        from src.database.models_migration import StrategyHorizonEnum, MarketEnum
        
        generator = StrategyGenerator()
        print("✅ Strategy generator initialized")
        
        # Test enum access
        horizons = [h.value for h in StrategyHorizonEnum]
        markets = [m.value for m in MarketEnum]
        print(f"✅ Available horizons: {horizons}")
        print(f"✅ Available markets: {markets}")
        
        return True
    except Exception as e:
        print(f"❌ Strategy generator test failed: {e}")
        return False

def test_vector_store():
    """Test vector store (optional)"""
    print("\n🔍 Testing vector store...")
    try:
        from src.services.vector_store import VectorStore
        vs = VectorStore()
        
        if vs.is_available():
            print("✅ Vector store available")
        else:
            print("⚠️  Vector store not available (install: pip install chromadb sentence-transformers)")
        
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_api_schemas():
    """Test API response schemas"""
    print("\n🔍 Testing API schemas...")
    try:
        # Test if we can import and create basic API models
        from datetime import datetime
        from src.database.models_migration import (
            StrategyHorizonEnum, MarketEnum, AssetTypeEnum
        )
        
        # Test enum serialization
        test_data = {
            "horizon": StrategyHorizonEnum.daily.value,
            "market": MarketEnum.vn.value,
            "asset_type": AssetTypeEnum.stocks.value
        }
        
        print(f"✅ API schemas work: {test_data}")
        return True
    except Exception as e:
        print(f"❌ API schemas test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FinBrief Backend Test Suite")
    print("=" * 40)
    
    tests = [
        test_database,
        test_adapters, 
        test_strategy_generator,
        test_vector_store,
        test_api_schemas
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is ready.")
    else:
        print("⚠️  Some tests failed. Check dependencies and configuration.")
    
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set environment variables (DATABASE_URI, SECRET_KEY, FINNHUB_API_KEY)")
    print("3. Run pipeline: PYTHONPATH=. python3 src/crawlers/unified_pipeline.py")
    print("4. Start API: export SECRET_KEY='your-secret' && PYTHONPATH=. python3 scripts/main.py")

if __name__ == "__main__":
    main()