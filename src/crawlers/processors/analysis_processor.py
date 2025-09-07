"""
Analysis processing components.
Handles action mapping and analysis storage.
"""
import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy.orm import Session

from src.database.models_migration import (
    News, Analysis, MarketEnum, SentimentEnum, 
    ActionEnum, AnalysisTypeEnum
)
from src.crawlers.base_adapter import NewsItem


class AnalysisProcessor:
    """Handles news deduplication, storage, and analysis creation"""
    
    @staticmethod
    def sha256_hash(text: str) -> str:
        """Generate SHA256 hash for content deduplication"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def check_duplicate(self, session: Session, news_item: NewsItem) -> Optional[News]:
        """Check if news already exists based on URL or content hash"""
        content_hash = self.sha256_hash(news_item.headline + " " + news_item.content_raw)
        
        return session.query(News).filter(
            (News.url == news_item.url) | (News.content_hash == content_hash)
        ).first()
    
    def store_news(self, session: Session, news_item: NewsItem, content_summary: str) -> News:
        """Store news item in database"""
        content_hash = self.sha256_hash(news_item.headline + " " + news_item.content_raw)
        
        news = News(
            source=news_item.source,
            url=news_item.url,
            published_at=news_item.published_at,
            headline=news_item.headline,
            content_raw=news_item.content_raw,
            content_summary=content_summary,
            content_hash=content_hash,
            market=news_item.market  # SQLAlchemy will handle enum conversion
        )
        
        session.add(news)
        session.commit()
        return news
    
    def map_sentiment_to_action(self, sentiment_label: str) -> Dict[str, ActionEnum]:
        """Map sentiment to trading actions"""
        if sentiment_label == "positive":
            return {
                "short": ActionEnum.BUY,
                "mid": ActionEnum.HOLD,
                "long": ActionEnum.BUY
            }
        elif sentiment_label == "negative":
            return {
                "short": ActionEnum.SELL,
                "mid": ActionEnum.AVOID,
                "long": ActionEnum.HOLD
            }
        else:  # neutral
            return {
                "short": ActionEnum.HOLD,
                "mid": ActionEnum.HOLD,
                "long": ActionEnum.HOLD
            }
    
    def store_analysis(self, session: Session, news: News, sentiment_result: Dict, 
                      content_summary: str, raw_metadata: Dict) -> Analysis:
        """Store analysis results in database"""
        actions = self.map_sentiment_to_action(sentiment_result["sentiment"])
        
        raw_output = {
            "source_metadata": raw_metadata,
            "summarizer": {"summary": content_summary},
            "sentiment_model": sentiment_result
        }
        
        analysis = Analysis(
            news_id=news.id,
            analysis_type=AnalysisTypeEnum.model,
            model_name="finbert+bart",
            model_version="v0.1",
            created_by=None,
            sentiment=SentimentEnum(sentiment_result["sentiment"]),
            sentiment_score=sentiment_result["confidence"],
            impact_score=None,
            action_short=actions["short"],
            action_mid=actions["mid"],
            action_long=actions["long"],
            action_confidence=float(sentiment_result["confidence"]),
            rationale=f"Auto rule-based mapping from sentiment ({sentiment_result['sentiment']})",
            raw_output=raw_output,
            is_latest=True
        )
        
        session.add(analysis)
        session.commit()
        return analysis