#!/usr/bin/env python3
"""
Test script for the new news sources integration.
Tests MarketWatch, Reuters, and VietStock adapters.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.crawlers.adapters.marketwatch_adapter import MarketWatchAdapter
from src.crawlers.adapters.reuters_adapter import ReutersAdapter
from src.crawlers.adapters.vietstock_adapter import VietStockAdapter
from src.crawlers.unified_pipeline import UnifiedNewsPipeline


def test_marketwatch():
    """Test MarketWatch RSS integration"""
    print("=== Testing MarketWatch RSS ===")
    try:
        adapter = MarketWatchAdapter()
        print(f"Source: {adapter.source_name}")
        print(f"Market: {adapter.market}")
        
        # Test different feeds
        for feed in ['latest', 'markets']:
            print(f"\nTesting {feed} feed:")
            news_items = adapter.fetch_news(feed=feed)
            print(f"Fetched {len(news_items)} items")
            
            if news_items:
                item = news_items[0]
                print(f"Sample headline: {item.headline[:100]}...")
                print(f"Published: {item.published_at}")
                print(f"URL: {item.url}")
                print(f"Asset type: {item.asset_type}")
                
    except Exception as e:
        print(f"MarketWatch test failed: {e}")


def test_reuters():
    """Test Reuters RSS integration"""
    print("\n\n=== Testing Reuters RSS ===")
    try:
        adapter = ReutersAdapter()
        print(f"Source: {adapter.source_name}")
        print(f"Market: {adapter.market}")
        
        # Test different feeds
        for feed in ['business', 'markets']:
            print(f"\nTesting {feed} feed:")
            news_items = adapter.fetch_news(feed=feed)
            print(f"Fetched {len(news_items)} items")
            
            if news_items:
                item = news_items[0]
                print(f"Sample headline: {item.headline[:100]}...")
                print(f"Published: {item.published_at}")
                print(f"URL: {item.url}")
                print(f"Asset type: {item.asset_type}")
                
    except Exception as e:
        print(f"Reuters test failed: {e}")


def test_vietstock():
    """Test VietStock RSS integration"""
    print("\n\n=== Testing VietStock RSS ===")
    try:
        adapter = VietStockAdapter()
        print(f"Source: {adapter.source_name}")
        print(f"Market: {adapter.market}")
        
        # Test different feeds
        for feed in ['latest', 'stocks']:
            print(f"\nTesting {feed} feed:")
            news_items = adapter.fetch_news(feed=feed)
            print(f"Fetched {len(news_items)} items")
            
            if news_items:
                item = news_items[0]
                print(f"Sample headline: {item.headline[:100]}...")
                print(f"Published: {item.published_at}")
                print(f"URL: {item.url}")
                print(f"Asset type: {item.asset_type}")
                print(f"Tickers: {item.tickers}")
                
    except Exception as e:
        print(f"VietStock test failed: {e}")


def test_unified_pipeline():
    """Test the new sources in the unified pipeline"""
    print("\n\n=== Testing Unified Pipeline with New Sources ===")
    try:
        pipeline = UnifiedNewsPipeline()
        
        # Test each new source individually (small batch)
        for source in ['marketwatch', 'reuters', 'vietstock']:
            print(f"\nTesting {source} via unified pipeline:")
            results = pipeline.run_pipeline(sources=[source])
            
            if source in results:
                result = results[source]
                print(f"Status: {result['status']}")
                print(f"Inserted: {result['inserted']}")
                print(f"Skipped: {result['skipped']}")
                if 'error' in result:
                    print(f"Error: {result['error']}")
            else:
                print(f"No results for {source}")
                
    except Exception as e:
        print(f"Unified pipeline test failed: {e}")


if __name__ == "__main__":
    print("Testing new news sources integration...")
    print("=" * 50)
    
    # Test individual adapters
    test_marketwatch()
    test_reuters() 
    test_vietstock()
    
    # Test unified pipeline integration
    test_unified_pipeline()
    
    print("\n" + "=" * 50)
    print("Testing completed!")