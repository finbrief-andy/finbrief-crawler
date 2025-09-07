"""
CafeF RSS adapter for Vietnamese financial news.
Fetches news from CafeF RSS feeds covering stocks, gold, and real estate.
"""
import feedparser
import requests
from datetime import datetime
from typing import List, Dict, Any
import re

from src.crawlers.base_adapter import BaseNewsAdapter, NewsItem
from src.database.models_migration import MarketEnum, AssetTypeEnum


class CafeFAdapter(BaseNewsAdapter):
    """Adapter for CafeF Vietnamese financial news RSS"""
    
    def __init__(self):
        super().__init__("cafef", MarketEnum.vn)
        
        # CafeF RSS feed URLs for different asset types
        self.rss_feeds = {
            AssetTypeEnum.stocks: "https://cafef.vn/thi-truong-chung-khoan.rss",
            AssetTypeEnum.gold: "https://cafef.vn/vang-bac-da-quy.rss",
            AssetTypeEnum.real_estate: "https://cafef.vn/bat-dong-san.rss"
        }
        
        # Vietnamese stock tickers pattern
        self.vn_ticker_pattern = re.compile(r'\b([A-Z]{3})\b')
    
    def fetch_raw_data(self, asset_type: str = "stocks", **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw news data from CafeF RSS feeds"""
        try:
            asset_enum = AssetTypeEnum[asset_type.lower()]
        except KeyError:
            asset_enum = AssetTypeEnum.stocks
        
        rss_url = self.rss_feeds.get(asset_enum, self.rss_feeds[AssetTypeEnum.stocks])
        
        try:
            response = requests.get(rss_url, timeout=15)
            response.raise_for_status()
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            
            items = []
            for entry in feed.entries[:50]:  # Limit to recent 50 items
                items.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "description": entry.get("description", ""),
                    "published": entry.get("published", ""),
                    "published_parsed": entry.get("published_parsed"),
                    "asset_type": asset_enum.value,
                    "source_entry": entry
                })
            
            return items
            
        except Exception as e:
            print(f"Error fetching from CafeF RSS ({asset_type}): {e}")
            return []
    
    def extract_tickers_vn(self, text: str) -> List[str]:
        """Extract Vietnamese stock tickers from text"""
        if not text:
            return []
        
        # Common VN stock tickers
        matches = self.vn_ticker_pattern.findall(text.upper())
        
        # Filter to known VN tickers (simplified list)
        vn_tickers = ["VIC", "VCB", "VHM", "VNM", "BID", "CTG", "TCB", "MSN", "HPG", "SAB",
                     "GAS", "PLX", "VRE", "POW", "SSI", "ACB", "MBB", "STB", "HDB", "TPB"]
        
        return [ticker for ticker in matches if ticker in vn_tickers]
    
    def categorize_asset_type(self, title: str, description: str, declared_type: str) -> AssetTypeEnum:
        """Determine asset type from content"""
        text = f"{title} {description}".lower()
        
        # Gold keywords
        gold_keywords = ["vàng", "gold", "bạc", "silver", "kim loại quý", "precious metal"]
        if any(keyword in text for keyword in gold_keywords):
            return AssetTypeEnum.gold
        
        # Real estate keywords
        real_estate_keywords = ["bất động sản", "real estate", "nhà đất", "căn hộ", "chung cư", 
                               "property", "housing", "building", "construction"]
        if any(keyword in text for keyword in real_estate_keywords):
            return AssetTypeEnum.real_estate
        
        # Stock keywords or tickers present
        stock_keywords = ["chứng khoán", "stock", "cổ phiếu", "shares", "trading", "hose", "hnx"]
        if any(keyword in text for keyword in stock_keywords) or self.extract_tickers_vn(text):
            return AssetTypeEnum.stocks
        
        # Default to declared type or stocks
        try:
            return AssetTypeEnum[declared_type.lower()]
        except (KeyError, AttributeError):
            return AssetTypeEnum.stocks
    
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert CafeF RSS item to normalized format"""
        title = raw_item.get("title", "").strip()
        description = raw_item.get("description", "").strip()
        url = raw_item.get("link", "")
        
        # Parse published date
        published_parsed = raw_item.get("published_parsed")
        if published_parsed:
            published_at = datetime(*published_parsed[:6])
        else:
            published_at = datetime.utcnow()
        
        # Determine asset type
        declared_type = raw_item.get("asset_type", "stocks")
        asset_type = self.categorize_asset_type(title, description, declared_type)
        
        # Extract tickers if it's a stock-related article
        tickers = []
        if asset_type == AssetTypeEnum.stocks:
            tickers = self.extract_tickers_vn(f"{title} {description}")
        
        # Create metadata
        metadata = {
            "original_asset_type": declared_type,
            "detected_asset_type": asset_type.value,
            "tickers_extracted": tickers,
            "language": "vi",
            "source_feed": "cafef_rss"
        }
        
        return NewsItem(
            headline=title,
            content_raw=description,
            url=url,
            published_at=published_at,
            source=self.source_name,
            market=self.market,
            asset_type=asset_type,
            tickers=tickers,
            metadata=metadata
        )
    
    def fetch_news(self, **kwargs) -> List[NewsItem]:
        """Fetch and normalize news from all CafeF RSS feeds"""
        all_items = []
        
        # Fetch from all asset type feeds
        for asset_type in ["stocks", "gold", "real_estate"]:
            try:
                raw_items = self.fetch_raw_data(asset_type=asset_type)
                for raw_item in raw_items:
                    item = self.normalize_item(raw_item)
                    all_items.append(item)
            except Exception as e:
                print(f"Error processing {asset_type} feed from CafeF: {e}")
        
        return all_items


if __name__ == "__main__":
    # Test the adapter
    adapter = CafeFAdapter()
    news_items = adapter.fetch_news()
    print(f"Fetched {len(news_items)} news items from CafeF")
    
    # Show sample items by asset type
    for asset_type in AssetTypeEnum:
        items = [item for item in news_items if getattr(item, 'asset_type', None) == asset_type]
        print(f"{asset_type.value}: {len(items)} items")
        if items:
            print(f"  Sample: {items[0].headline[:100]}")