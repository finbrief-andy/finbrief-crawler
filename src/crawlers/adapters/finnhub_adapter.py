"""
Finnhub API adapter for financial news.
Fetches global financial news from Finnhub API.
"""
import os
import requests
from datetime import datetime
from typing import List, Dict, Any

from src.crawlers.base_adapter import BaseNewsAdapter, NewsItem
from src.database.models_migration import MarketEnum


class FinnhubAdapter(BaseNewsAdapter):
    """Adapter for Finnhub financial news API"""
    
    def __init__(self):
        super().__init__("finnhub", MarketEnum.global_market)
        self.api_key = os.getenv("FINNHUB_API_KEY", "d2u2v59r01qo4hodrjagd2u2v59r01qo4hodrjb0")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY environment variable is required")
    
    def fetch_raw_data(self, category: str = "general", **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw news data from Finnhub API"""
        url = f"https://finnhub.io/api/v1/news?category={category}&token={self.api_key}"
        
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching from Finnhub: {e}")
            return []
    
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert Finnhub news item to normalized format"""
        headline = raw_item.get("headline") or raw_item.get("title") or ""
        summary_text = raw_item.get("summary") or ""
        url = raw_item.get("url") or raw_item.get("news_url")
        published_unix = raw_item.get("datetime")
        
        # Convert Unix timestamp to datetime
        if published_unix:
            published_at = datetime.utcfromtimestamp(published_unix)
        else:
            published_at = datetime.utcnow()
        
        # Use summary as content, fallback to headline
        content_raw = summary_text or headline
        source = raw_item.get("source") or raw_item.get("category") or self.source_name
        
        return NewsItem(
            headline=headline,
            content_raw=content_raw,
            url=url,
            published_at=published_at,
            source=source,
            market=self.market,
            metadata=raw_item  # Keep original data as metadata
        )