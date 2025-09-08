"""
Reuters RSS adapter for business and financial news.
Fetches business news from Reuters RSS feeds.
"""
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any
from dateutil.parser import parse as parse_date

from src.crawlers.base_adapter import BaseNewsAdapter, NewsItem
from src.database.models_migration import MarketEnum, AssetTypeEnum


class ReutersAdapter(BaseNewsAdapter):
    """Adapter for Reuters RSS feeds"""
    
    def __init__(self):
        super().__init__("reuters", MarketEnum.global_market)
        self.rss_feeds = {
            'business': 'https://www.reuters.com/markets/rss',
            'markets': 'https://www.reuters.com/business/rss',
            'world': 'https://www.reuters.com/world/rss',
            'technology': 'https://www.reuters.com/technology/rss'
        }
    
    def fetch_raw_data(self, feed: str = 'business', **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw RSS data from Reuters"""
        if feed not in self.rss_feeds:
            print(f"Unknown Reuters feed: {feed}. Available: {list(self.rss_feeds.keys())}")
            feed = 'business'
        
        url = self.rss_feeds[feed]
        
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; FinBrief/1.0; +https://finbrief.com/bot)'
            })
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            items = []
            
            # Find all item elements in the RSS feed
            for item in root.findall('.//item'):
                item_data = {}
                
                # Extract standard RSS fields
                for child in item:
                    if child.tag in ['title', 'description', 'link', 'pubDate', 'guid']:
                        item_data[child.tag] = child.text
                    elif child.tag == 'category':
                        if 'categories' not in item_data:
                            item_data['categories'] = []
                        item_data['categories'].append(child.text)
                
                # Add feed type as metadata
                item_data['feed_type'] = feed
                
                if item_data.get('title'):  # Only include items with titles
                    items.append(item_data)
            
            print(f"Fetched {len(items)} items from Reuters {feed} feed")
            return items
            
        except requests.RequestException as e:
            print(f"Error fetching Reuters RSS ({feed}): {e}")
            return []
        except ET.ParseError as e:
            print(f"Error parsing Reuters RSS XML ({feed}): {e}")
            return []
    
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert Reuters RSS item to normalized format"""
        headline = raw_item.get('title', '').strip()
        content_raw = raw_item.get('description', '').strip()
        url = raw_item.get('link', '').strip()
        pub_date_str = raw_item.get('pubDate', '')
        
        # Parse publication date
        if pub_date_str:
            try:
                published_at = parse_date(pub_date_str)
                # Convert to UTC if timezone aware
                if published_at.tzinfo is not None:
                    published_at = published_at.utctimetuple()
                    published_at = datetime(*published_at[:6])
            except Exception as e:
                print(f"Error parsing date '{pub_date_str}': {e}")
                published_at = datetime.utcnow()
        else:
            published_at = datetime.utcnow()
        
        # Clean HTML tags from content if present
        import re
        if content_raw:
            content_raw = re.sub(r'<[^>]+>', '', content_raw)
            content_raw = content_raw.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        # Determine asset type based on feed type
        asset_type = AssetTypeEnum.stocks  # Default
        feed_type = raw_item.get('feed_type', '')
        
        if feed_type in ['economy', 'world']:
            asset_type = AssetTypeEnum.stocks  # Use stocks for economy/world news
        elif feed_type == 'technology':
            asset_type = AssetTypeEnum.stocks  # Tech stocks
        elif feed_type in ['markets', 'business']:
            asset_type = AssetTypeEnum.stocks
        
        return NewsItem(
            headline=headline,
            content_raw=content_raw or headline,
            url=url,
            published_at=published_at,
            source=f"{self.source_name}_{feed_type}" if feed_type else self.source_name,
            market=self.market,
            asset_type=asset_type,
            tickers=[],  # Could extract tickers from content if needed
            metadata=raw_item
        )