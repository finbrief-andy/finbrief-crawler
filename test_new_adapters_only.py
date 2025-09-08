#!/usr/bin/env python3
"""
Simple test script for the new news source adapters only.
Tests MarketWatch, Reuters, and VietStock adapters individually.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test MarketWatch
def test_marketwatch():
    """Test MarketWatch RSS integration"""
    print("=== Testing MarketWatch RSS ===")
    try:
        from src.crawlers.adapters.marketwatch_adapter import MarketWatchAdapter
        
        adapter = MarketWatchAdapter()
        print(f"Source: {adapter.source_name}")
        print(f"Market: {adapter.market}")
        
        # Test latest feed
        print(f"\nTesting latest feed:")
        raw_data = adapter.fetch_raw_data(feed='latest')
        print(f"Fetched {len(raw_data)} raw items")
        
        if raw_data:
            # Test normalization with first item
            try:
                news_item = adapter.normalize_item(raw_data[0])
                print(f"Sample headline: {news_item.headline[:100]}...")
                print(f"Published: {news_item.published_at}")
                print(f"URL: {news_item.url}")
                print(f"Asset type: {news_item.asset_type}")
                print(f"Source: {news_item.source}")
                print("✅ MarketWatch adapter working!")
            except Exception as e:
                print(f"❌ Error normalizing MarketWatch item: {e}")
        else:
            print("⚠️  No data fetched from MarketWatch")
            
    except Exception as e:
        print(f"❌ MarketWatch test failed: {e}")


def test_reuters():
    """Test Reuters RSS integration"""  
    print("\n\n=== Testing Reuters RSS ===")
    try:
        from src.crawlers.adapters.reuters_adapter import ReutersAdapter
        
        adapter = ReutersAdapter()
        print(f"Source: {adapter.source_name}")
        print(f"Market: {adapter.market}")
        
        # Test business feed
        print(f"\nTesting business feed:")
        raw_data = adapter.fetch_raw_data(feed='business')
        print(f"Fetched {len(raw_data)} raw items")
        
        if raw_data:
            # Test normalization with first item
            try:
                news_item = adapter.normalize_item(raw_data[0])
                print(f"Sample headline: {news_item.headline[:100]}...")
                print(f"Published: {news_item.published_at}")
                print(f"URL: {news_item.url}")
                print(f"Asset type: {news_item.asset_type}")
                print(f"Source: {news_item.source}")
                print("✅ Reuters adapter working!")
            except Exception as e:
                print(f"❌ Error normalizing Reuters item: {e}")
        else:
            print("⚠️  No data fetched from Reuters")
            
    except Exception as e:
        print(f"❌ Reuters test failed: {e}")


def test_vietstock():
    """Test VietStock RSS integration"""
    print("\n\n=== Testing VietStock RSS ===")
    try:
        from src.crawlers.adapters.vietstock_adapter import VietStockAdapter
        
        adapter = VietStockAdapter()
        print(f"Source: {adapter.source_name}")
        print(f"Market: {adapter.market}")
        
        # Test latest feed
        print(f"\nTesting latest feed:")
        raw_data = adapter.fetch_raw_data(feed='latest')
        print(f"Fetched {len(raw_data)} raw items")
        
        if raw_data:
            # Test normalization with first item
            try:
                news_item = adapter.normalize_item(raw_data[0])
                print(f"Sample headline: {news_item.headline[:100]}...")
                print(f"Published: {news_item.published_at}")
                print(f"URL: {news_item.url}")
                print(f"Asset type: {news_item.asset_type}")
                print(f"Source: {news_item.source}")
                print(f"Tickers: {news_item.tickers}")
                print("✅ VietStock adapter working!")
            except Exception as e:
                print(f"❌ Error normalizing VietStock item: {e}")
        else:
            print("⚠️  No data fetched from VietStock")
            
    except Exception as e:
        print(f"❌ VietStock test failed: {e}")


if __name__ == "__main__":
    print("Testing new news source adapters...")
    print("=" * 60)
    
    # Test individual adapters
    test_marketwatch()
    test_reuters() 
    test_vietstock()
    
    print("\n" + "=" * 60)
    print("Adapter testing completed!")