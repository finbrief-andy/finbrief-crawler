"""
Base adapter interface for news sources.
Defines the common structure for normalizing different news sources.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from src.database.models_migration import MarketEnum


@dataclass
class NewsItem:
    """Normalized news item structure from any source"""
    headline: str
    content_raw: str
    url: Optional[str]
    published_at: datetime
    source: str
    market: MarketEnum
    metadata: Dict[str, Any]  # Source-specific metadata


class BaseNewsAdapter(ABC):
    """Abstract base class for news source adapters"""
    
    def __init__(self, source_name: str, market: MarketEnum):
        self.source_name = source_name
        self.market = market
    
    @abstractmethod
    def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw data from the source"""
        pass
    
    @abstractmethod
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert raw item to normalized NewsItem"""
        pass
    
    def fetch_news(self, **kwargs) -> List[NewsItem]:
        """Fetch and normalize news items"""
        raw_data = self.fetch_raw_data(**kwargs)
        news_items = []
        
        for raw_item in raw_data:
            try:
                news_item = self.normalize_item(raw_item)
                news_items.append(news_item)
            except Exception as e:
                print(f"Error normalizing item from {self.source_name}: {e}")
                continue
        
        return news_items