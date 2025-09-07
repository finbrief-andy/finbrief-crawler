"""
Gold Price API adapter for gold market data and news.
Fetches gold price data and related news from various sources.
"""
import requests
from datetime import datetime
from typing import List, Dict, Any
import json

from src.crawlers.base_adapter import BaseNewsAdapter, NewsItem
from src.database.models_migration import MarketEnum, AssetTypeEnum


class GoldPriceAdapter(BaseNewsAdapter):
    """Adapter for gold price data and news"""
    
    def __init__(self):
        super().__init__("gold_api", MarketEnum.global_market)
        
        # Multiple gold price APIs (free tiers)
        self.apis = {
            "metals_api": "https://api.metals.live/v1/spot/gold",
            "goldprice_api": "https://api.goldprice.org/",
            # Add more as needed
        }
    
    def fetch_gold_price_data(self) -> Dict[str, Any]:
        """Fetch current gold price data"""
        try:
            # Try metals.live API first (no key needed for basic data)
            response = requests.get(self.apis["metals_api"], timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "price_usd": data.get("price", 0),
                    "currency": "USD",
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "metals.live"
                }
        except Exception as e:
            print(f"Error fetching gold price from metals.live: {e}")
        
        return {}
    
    def generate_gold_news_items(self, price_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate synthetic news items from gold price data"""
        if not price_data or not price_data.get("price_usd"):
            return []
        
        price = price_data["price_usd"]
        timestamp = price_data.get("timestamp", datetime.utcnow().isoformat())
        
        # Generate different types of gold-related news items
        news_items = []
        
        # Current price update
        news_items.append({
            "title": f"Gold Price Update: ${price:.2f}/oz",
            "description": f"Current gold spot price: ${price:.2f} per ounce. Market data as of {timestamp}.",
            "url": f"https://goldprice.org/",
            "published": timestamp,
            "category": "price_update",
            "price_usd": price
        })
        
        # Price analysis (simplified)
        if price > 2000:
            news_items.append({
                "title": "Gold Maintains Above $2000 Resistance Level",
                "description": f"Gold continues trading above the key $2000 psychological level at ${price:.2f}/oz, indicating strong market sentiment.",
                "url": "https://goldprice.org/gold-price-analysis",
                "published": timestamp,
                "category": "analysis",
                "price_usd": price
            })
        elif price < 1800:
            news_items.append({
                "title": "Gold Below $1800 Support - Market Concerns",
                "description": f"Gold trading below $1800 support level at ${price:.2f}/oz may indicate shifting market dynamics.",
                "url": "https://goldprice.org/gold-price-analysis",
                "published": timestamp,
                "category": "analysis",
                "price_usd": price
            })
        
        return news_items
    
    def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw gold market data"""
        price_data = self.fetch_gold_price_data()
        return self.generate_gold_news_items(price_data)
    
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert gold data item to normalized format"""
        title = raw_item.get("title", "")
        description = raw_item.get("description", "")
        url = raw_item.get("url", "")
        published_str = raw_item.get("published", "")
        
        # Parse published date
        try:
            if isinstance(published_str, str):
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            else:
                published_at = datetime.utcnow()
        except:
            published_at = datetime.utcnow()
        
        # Gold-specific metadata
        metadata = {
            "asset_type": "gold",
            "category": raw_item.get("category", "general"),
            "price_usd": raw_item.get("price_usd"),
            "data_source": "synthetic_gold_news",
            "language": "en"
        }
        
        return NewsItem(
            headline=title,
            content_raw=description,
            url=url,
            published_at=published_at,
            source=self.source_name,
            market=self.market,
            asset_type=AssetTypeEnum.gold,
            tickers=["GOLD", "XAU"],  # Gold ticker symbols
            metadata=metadata
        )


class RealEstateNewsAdapter(BaseNewsAdapter):
    """Adapter for real estate market news (placeholder for RSS feeds)"""
    
    def __init__(self):
        super().__init__("real_estate_news", MarketEnum.global_market)
        
        # Real estate RSS feeds (examples)
        self.rss_feeds = {
            "global": "https://www.inman.com/feed/",
            "vietnam": "https://batdongsan.com.vn/rss"  # hypothetical
        }
    
    def fetch_raw_data(self, region: str = "global", **kwargs) -> List[Dict[str, Any]]:
        """Fetch real estate news from RSS feeds"""
        # For now, generate sample real estate news
        # In production, implement RSS parsing similar to CafeFAdapter
        
        sample_items = [
            {
                "title": "Housing Market Trends: Interest Rates Impact",
                "description": "Analysis of how changing interest rates affect the housing market dynamics and buyer behavior.",
                "url": "https://example.com/housing-trends",
                "published": datetime.utcnow().isoformat(),
                "category": "market_analysis"
            },
            {
                "title": "Commercial Real Estate Investment Outlook",
                "description": "Commercial property investment trends and forecasts for the upcoming quarter.",
                "url": "https://example.com/commercial-outlook",
                "published": datetime.utcnow().isoformat(),
                "category": "investment"
            }
        ]
        
        return sample_items
    
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert real estate news item to normalized format"""
        title = raw_item.get("title", "")
        description = raw_item.get("description", "")
        url = raw_item.get("url", "")
        
        # Parse published date
        try:
            published_str = raw_item.get("published", "")
            if isinstance(published_str, str):
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            else:
                published_at = datetime.utcnow()
        except:
            published_at = datetime.utcnow()
        
        # Real estate specific metadata
        metadata = {
            "asset_type": "real_estate",
            "category": raw_item.get("category", "general"),
            "region": "global",
            "language": "en"
        }
        
        return NewsItem(
            headline=title,
            content_raw=description,
            url=url,
            published_at=published_at,
            source=self.source_name,
            market=self.market,
            asset_type=AssetTypeEnum.real_estate,
            tickers=[],  # Real estate doesn't have traditional tickers
            metadata=metadata
        )


if __name__ == "__main__":
    # Test gold adapter
    print("Testing Gold Price Adapter...")
    gold_adapter = GoldPriceAdapter()
    gold_items = gold_adapter.fetch_news()
    print(f"Fetched {len(gold_items)} gold news items")
    
    # Test real estate adapter
    print("\nTesting Real Estate News Adapter...")
    re_adapter = RealEstateNewsAdapter()
    re_items = re_adapter.fetch_news()
    print(f"Fetched {len(re_items)} real estate news items")