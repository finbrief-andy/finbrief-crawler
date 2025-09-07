"""
VnExpress RSS adapter for Vietnamese financial news.
Fetches Vietnamese business news from VnExpress RSS feed.
"""
import feedparser
from datetime import datetime
from typing import List, Dict, Any

from src.crawlers.base_adapter import BaseNewsAdapter, NewsItem
from src.database.models_migration import MarketEnum


class VnExpressAdapter(BaseNewsAdapter):
    """Adapter for VnExpress Vietnamese business news RSS feed"""
    
    def __init__(self):
        super().__init__("vnexpress", MarketEnum.vn)
        self.rss_url = "https://vnexpress.net/rss/kinh-doanh.rss"
    
    def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw RSS data from VnExpress"""
        try:
            feed = feedparser.parse(self.rss_url)
            if hasattr(feed, 'status') and feed.status != 200:
                print(f"RSS feed returned status {feed.status}")
                return []
            
            # Convert RSS entries to dict format
            items = []
            for entry in feed.entries:
                items.append({
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.get('published', ''),
                    'published_parsed': entry.get('published_parsed'),
                    'summary': entry.get('summary', ''),
                    'description': entry.get('description', ''),
                    'guid': entry.get('guid', ''),
                    'source': 'vnexpress'
                })
            
            return items
        except Exception as e:
            print(f"Error fetching VnExpress RSS: {e}")
            return []
    
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert VnExpress RSS item to normalized format"""
        headline = raw_item.get("title", "")
        content_raw = raw_item.get("summary") or raw_item.get("description", "")
        url = raw_item.get("link")
        
        # Parse published date
        published_parsed = raw_item.get("published_parsed")
        if published_parsed:
            published_at = datetime(*published_parsed[:6])
        else:
            published_at = datetime.utcnow()
        
        return NewsItem(
            headline=headline,
            content_raw=content_raw,
            url=url,
            published_at=published_at,
            source=self.source_name,
            market=self.market,
            metadata=raw_item  # Keep original RSS data as metadata
        )