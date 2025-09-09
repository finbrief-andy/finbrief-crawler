"""
Mobile-Optimized API endpoints.
Provides lightweight, mobile-friendly API responses and features.
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Header
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from src.api.auth import get_current_user
from src.database.models_migration import UserBase, News, Analysis, Strategy, MarketEnum, StrategyHorizonEnum
from src.services.vector_store import get_vector_store
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import desc, and_
from src.database.models_migration import init_db_and_create

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mobile", tags=["mobile"])

# Initialize database
engine = init_db_and_create()
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===============================
# Mobile-Optimized Response Models
# ===============================

class MobileNewsItem(BaseModel):
    """Lightweight news item for mobile"""
    id: int
    headline: str = Field(..., max_length=120)  # Truncated for mobile
    summary: str = Field(..., max_length=200)  # Short summary
    source: str
    published_at: datetime
    sentiment: Optional[str] = None
    action: Optional[str] = None
    tickers: List[str] = Field(default=[], max_items=3)  # Limit tickers
    url: Optional[str] = None
    image_url: Optional[str] = None  # For future use


class MobileStrategy(BaseModel):
    """Lightweight strategy for mobile"""
    id: int
    horizon: str
    market: str
    title: str = Field(..., max_length=100)
    summary: str = Field(..., max_length=300)  # Condensed summary
    key_points: List[str] = Field(default=[], max_items=3)  # Top 3 points
    confidence: Optional[float] = None
    updated_at: datetime


class MobilePortfolio(BaseModel):
    """Mobile portfolio summary"""
    total_value: float
    daily_change: float
    daily_change_percent: float
    top_performers: List[Dict[str, Any]] = Field(default=[], max_items=3)
    alerts_count: int
    last_updated: datetime


class MobileAlert(BaseModel):
    """Mobile alert notification"""
    id: int
    title: str = Field(..., max_length=80)
    message: str = Field(..., max_length=150)
    priority: str  # high, medium, low
    type: str  # price, news, strategy
    created_at: datetime
    read: bool = False


class SyncResponse(BaseModel):
    """Data synchronization response"""
    news: List[MobileNewsItem]
    strategies: List[MobileStrategy]
    alerts: List[MobileAlert]
    portfolio: Optional[MobilePortfolio] = None
    last_sync: datetime
    next_sync: datetime


# ===============================
# Mobile API Endpoints
# ===============================

@router.get("/news", response_model=List[MobileNewsItem])
async def get_mobile_news(
    limit: int = Query(20, ge=1, le=50, description="Number of items"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    market: Optional[str] = Query(None, description="Market filter"),
    db: Session = Depends(get_db),
    current_user: UserBase = Depends(get_current_user)
):
    """Get mobile-optimized news feed"""
    try:
        query = db.query(News).join(Analysis, News.id == Analysis.news_id, isouter=True)
        
        if market:
            try:
                market_enum = MarketEnum[market.lower()]
                query = query.filter(News.market == market_enum)
            except KeyError:
                pass  # Ignore invalid market
        
        news_items = query.order_by(desc(News.published_at)).offset(offset).limit(limit).all()
        
        mobile_news = []
        for news in news_items:
            # Get latest analysis for this news
            analysis = db.query(Analysis).filter(Analysis.news_id == news.id).order_by(desc(Analysis.created_at)).first()
            
            # Truncate headline for mobile
            headline = news.headline[:120] if len(news.headline) > 120 else news.headline
            
            # Create short summary
            summary = news.content_summary[:200] if news.content_summary and len(news.content_summary) > 200 else (news.content_summary or headline)
            
            mobile_news.append(MobileNewsItem(
                id=news.id,
                headline=headline,
                summary=summary,
                source=news.source,
                published_at=news.published_at,
                sentiment=analysis.sentiment.value if analysis and analysis.sentiment else None,
                action=analysis.action_short.value if analysis and analysis.action_short else None,
                tickers=news.tickers[:3] if news.tickers else [],
                url=news.url
            ))
        
        logger.info(f"Mobile user {current_user.username} fetched {len(mobile_news)} news items")
        return mobile_news
        
    except Exception as e:
        logger.error(f"Error fetching mobile news: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch news")


@router.get("/strategies", response_model=List[MobileStrategy])
async def get_mobile_strategies(
    market: str = Query("global", description="Market"),
    limit: int = Query(10, ge=1, le=20, description="Number of strategies"),
    db: Session = Depends(get_db),
    current_user: UserBase = Depends(get_current_user)
):
    """Get mobile-optimized strategies"""
    try:
        market_enum = MarketEnum[market.lower()]
        
        strategies = db.query(Strategy).filter(
            Strategy.market == market_enum
        ).order_by(desc(Strategy.strategy_date)).limit(limit).all()
        
        mobile_strategies = []
        for strategy in strategies:
            # Extract key points from key_drivers (limit to 3)
            key_points = strategy.key_drivers[:3] if strategy.key_drivers else []
            
            # Truncate title and summary for mobile
            title = strategy.title[:100] if len(strategy.title) > 100 else strategy.title
            summary = strategy.summary[:300] if len(strategy.summary) > 300 else strategy.summary
            
            mobile_strategies.append(MobileStrategy(
                id=strategy.id,
                horizon=strategy.horizon.value,
                market=strategy.market.value,
                title=title,
                summary=summary,
                key_points=key_points,
                confidence=strategy.confidence_score,
                updated_at=strategy.created_at
            ))
        
        logger.info(f"Mobile user {current_user.username} fetched {len(mobile_strategies)} strategies")
        return mobile_strategies
        
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid market")
    except Exception as e:
        logger.error(f"Error fetching mobile strategies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch strategies")


@router.get("/dashboard", response_model=SyncResponse)
async def get_mobile_dashboard(
    sync_since: Optional[datetime] = Query(None, description="Last sync timestamp"),
    db: Session = Depends(get_db),
    current_user: UserBase = Depends(get_current_user)
):
    """Get complete mobile dashboard data"""
    try:
        # Set sync window - if not provided, get last 24 hours
        if not sync_since:
            sync_since = datetime.utcnow() - timedelta(hours=24)
        
        # Get recent news (mobile optimized)
        news_query = db.query(News).join(Analysis, News.id == Analysis.news_id, isouter=True)
        news_query = news_query.filter(News.created_at >= sync_since)
        recent_news = news_query.order_by(desc(News.published_at)).limit(10).all()
        
        mobile_news = []
        for news in recent_news:
            analysis = db.query(Analysis).filter(Analysis.news_id == news.id).order_by(desc(Analysis.created_at)).first()
            
            mobile_news.append(MobileNewsItem(
                id=news.id,
                headline=news.headline[:120],
                summary=(news.content_summary or news.headline)[:200],
                source=news.source,
                published_at=news.published_at,
                sentiment=analysis.sentiment.value if analysis and analysis.sentiment else None,
                action=analysis.action_short.value if analysis and analysis.action_short else None,
                tickers=news.tickers[:3] if news.tickers else [],
                url=news.url
            ))
        
        # Get recent strategies
        strategies = db.query(Strategy).filter(
            Strategy.created_at >= sync_since
        ).order_by(desc(Strategy.strategy_date)).limit(5).all()
        
        mobile_strategies = []
        for strategy in strategies:
            mobile_strategies.append(MobileStrategy(
                id=strategy.id,
                horizon=strategy.horizon.value,
                market=strategy.market.value,
                title=strategy.title[:100],
                summary=strategy.summary[:300],
                key_points=strategy.key_drivers[:3] if strategy.key_drivers else [],
                confidence=strategy.confidence_score,
                updated_at=strategy.created_at
            ))
        
        # Mock alerts for now (could be implemented with a real alerts system)
        mock_alerts = [
            MobileAlert(
                id=1,
                title="Market Update",
                message="New strategy generated for global market",
                priority="medium",
                type="strategy",
                created_at=datetime.utcnow(),
                read=False
            )
        ] if mobile_strategies else []
        
        # Mock portfolio summary
        mock_portfolio = MobilePortfolio(
            total_value=100000.0,
            daily_change=2500.0,
            daily_change_percent=2.5,
            top_performers=[
                {"symbol": "AAPL", "change": 3.2},
                {"symbol": "NVDA", "change": 5.1},
                {"symbol": "MSFT", "change": 1.8}
            ],
            alerts_count=len(mock_alerts),
            last_updated=datetime.utcnow()
        )
        
        now = datetime.utcnow()
        next_sync = now + timedelta(minutes=15)  # Suggest next sync in 15 minutes
        
        return SyncResponse(
            news=mobile_news,
            strategies=mobile_strategies,
            alerts=mock_alerts,
            portfolio=mock_portfolio,
            last_sync=now,
            next_sync=next_sync
        )
        
    except Exception as e:
        logger.error(f"Error fetching mobile dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard data")


@router.get("/search", response_model=List[MobileNewsItem])
async def mobile_search(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=20, description="Number of results"),
    current_user: UserBase = Depends(get_current_user)
):
    """Mobile-optimized semantic search"""
    try:
        vs = get_vector_store()
        
        if not vs.is_available():
            # Fallback to simple text search if vector search unavailable
            raise HTTPException(status_code=503, detail="Search temporarily unavailable")
        
        # Perform semantic search
        results = vs.search_similar_news(q, limit=limit)
        
        mobile_results = []
        for result in results:
            metadata = result['metadata']
            
            mobile_results.append(MobileNewsItem(
                id=int(result['id'].replace('news_', '')),
                headline=metadata.get('headline', '')[:120],
                summary=result['text'][:200],
                source=metadata.get('source', ''),
                published_at=datetime.fromisoformat(metadata.get('published_at', datetime.utcnow().isoformat())),
                tickers=[],  # Could extract from metadata if available
                url=metadata.get('url', '')
            ))
        
        logger.info(f"Mobile user {current_user.username} searched: '{q}' -> {len(mobile_results)} results")
        return mobile_results
        
    except Exception as e:
        logger.error(f"Error in mobile search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("/device/register")
async def register_mobile_device(
    device_token: str = Query(..., description="Push notification token"),
    platform: str = Query(..., description="ios or android"),
    app_version: str = Query("1.0.0", description="App version"),
    current_user: UserBase = Depends(get_current_user)
):
    """Register mobile device for push notifications"""
    try:
        # In a real implementation, you would save this to a DeviceTokens table
        # For now, we'll just log it
        logger.info(f"Registered device for user {current_user.username}: {platform} - {device_token[:20]}...")
        
        return {
            "status": "registered",
            "device_id": f"{platform}_{device_token[:8]}",
            "push_enabled": True,
            "message": "Device registered for push notifications"
        }
        
    except Exception as e:
        logger.error(f"Error registering device: {e}")
        raise HTTPException(status_code=500, detail="Failed to register device")


@router.get("/offline/sync")
async def offline_sync_data(
    last_sync: Optional[datetime] = Query(None, description="Last successful sync"),
    current_user: UserBase = Depends(get_current_user)
):
    """Get data for offline synchronization"""
    try:
        # Determine sync window
        if not last_sync:
            last_sync = datetime.utcnow() - timedelta(days=7)  # Default: last 7 days
        
        # This endpoint would return data optimized for offline storage
        # Including compressed data, essential content only, etc.
        
        return {
            "sync_id": f"sync_{int(datetime.utcnow().timestamp())}",
            "data_size_kb": 150,  # Estimated size
            "items_available": 25,
            "sync_recommended": True,
            "next_sync_in_minutes": 30,
            "message": "Offline sync data available"
        }
        
    except Exception as e:
        logger.error(f"Error preparing offline sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to prepare sync data")


@router.get("/status")
async def mobile_api_status():
    """Mobile API health and status"""
    return {
        "status": "available",
        "version": "1.0.0",
        "features": {
            "push_notifications": True,
            "offline_sync": True,
            "semantic_search": True,
            "real_time_data": True
        },
        "endpoints": [
            "/mobile/news",
            "/mobile/strategies", 
            "/mobile/dashboard",
            "/mobile/search",
            "/mobile/device/register",
            "/mobile/offline/sync"
        ]
    }