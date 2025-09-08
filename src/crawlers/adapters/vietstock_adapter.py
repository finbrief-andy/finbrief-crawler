"""
VietStock RSS adapter for Vietnamese financial news.
Fetches Vietnamese market news from VietStock RSS feeds.
"""
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any
from dateutil.parser import parse as parse_date

from src.crawlers.base_adapter import BaseNewsAdapter, NewsItem
from src.database.models_migration import MarketEnum, AssetTypeEnum


class VietStockAdapter(BaseNewsAdapter):
    """Adapter for VietStock RSS feeds"""
    
    def __init__(self):
        super().__init__("vietstock", MarketEnum.vn)
        self.rss_feeds = {
            'latest': 'https://vietstock.vn/rss/tintuc.rss',
            'analysis': 'https://vietstock.vn/rss/phan-tich-bao-cao.rss',
            'market': 'https://vietstock.vn/rss/thi-truong.rss',
            'macro': 'https://vietstock.vn/rss/vi-mo.rss',
            'stocks': 'https://vietstock.vn/rss/co-phieu.rss'
        }
    
    def fetch_raw_data(self, feed: str = 'latest', **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw RSS data from VietStock"""
        if feed not in self.rss_feeds:
            print(f"Unknown VietStock feed: {feed}. Available: {list(self.rss_feeds.keys())}")
            feed = 'latest'
        
        url = self.rss_feeds[feed]
        
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; FinBrief/1.0; +https://finbrief.com/bot)'
            })
            response.raise_for_status()
            
            # Parse XML with better error handling
            try:
                # Clean the response content to handle potential encoding issues
                content = response.content.decode('utf-8', errors='ignore')
                # Remove any problematic characters
                import re
                content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', content)
                root = ET.fromstring(content.encode('utf-8'))
            except UnicodeDecodeError:
                # Fallback to original content
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
            
            print(f"Fetched {len(items)} items from VietStock {feed} feed")
            return items
            
        except requests.RequestException as e:
            print(f"Error fetching VietStock RSS ({feed}): {e}")
            return []
        except ET.ParseError as e:
            print(f"Error parsing VietStock RSS XML ({feed}): {e}")
            return []
    
    def normalize_item(self, raw_item: Dict[str, Any]) -> NewsItem:
        """Convert VietStock RSS item to normalized format"""
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
        
        if feed_type in ['macro']:
            asset_type = AssetTypeEnum.stocks  # Use stocks for macro news
        elif feed_type in ['stocks', 'analysis', 'market']:
            asset_type = AssetTypeEnum.stocks
        
        # Extract potential ticker symbols from Vietnamese content
        tickers = []
        if headline or content_raw:
            text = f"{headline} {content_raw}".upper()
            # Look for Vietnamese stock patterns like VIC, VHM, etc.
            import re
            ticker_pattern = r'\b[A-Z]{3}\b'  # 3-letter stock codes are common in Vietnam
            potential_tickers = re.findall(ticker_pattern, text)
            # Filter out common non-ticker words
            exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
            tickers = [t for t in potential_tickers if t not in exclude_words][:5]  # Limit to 5 tickers
        
        return NewsItem(
            headline=headline,
            content_raw=content_raw or headline,
            url=url,
            published_at=published_at,
            source=f"{self.source_name}_{feed_type}" if feed_type else self.source_name,
            market=self.market,
            asset_type=asset_type,
            tickers=tickers,
            metadata=raw_item
        )