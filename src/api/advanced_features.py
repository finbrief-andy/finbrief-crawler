#!/usr/bin/env python3
"""
Advanced Features API Endpoints

Provides REST API endpoints for personalization, backtesting, 
real-time alerts, and portfolio tracking features.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging

from ..services.personalization_engine import PersonalizationEngine, UserPreferences
from ..services.backtesting_engine import BacktestingEngine, BacktestRequest, StrategyParams
from ..services.realtime_alerts import RealTimeAlerts, AlertSubscription, AlertType, AlertPriority
from ..services.portfolio_tracker import PortfolioTracker, Portfolio, Transaction, TransactionType
from ..auth.jwt_handler import get_current_user
from ..database.models import User

router = APIRouter(prefix="/api/v1/advanced", tags=["Advanced Features"])
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Initialize services
personalization_engine = PersonalizationEngine()
backtesting_engine = BacktestingEngine()
realtime_alerts = RealTimeAlerts()
portfolio_tracker = PortfolioTracker()

# Pydantic models for request/response validation

class UserPreferencesRequest(BaseModel):
    risk_tolerance: str
    investment_horizon: str
    preferred_markets: List[str]
    preferred_sectors: List[str]
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    preferred_countries: Optional[List[str]] = None
    esg_preference: Optional[str] = None

class BacktestRequestModel(BaseModel):
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    strategy_params: Dict[str, Any]
    benchmark: Optional[str] = "SPY"

class AlertSubscriptionRequest(BaseModel):
    alert_types: List[str]
    symbols: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    delivery_channels: List[str]
    min_priority: str = "medium"

class PortfolioCreateRequest(BaseModel):
    name: str
    description: str = ""
    initial_cash: float = 0.0

class TransactionRequest(BaseModel):
    symbol: str
    transaction_type: str
    quantity: float
    price: float
    commission: float = 0.0
    notes: str = ""

# Personalization Endpoints

@router.post("/personalization/preferences")
async def set_user_preferences(
    preferences: UserPreferencesRequest,
    current_user: User = Depends(get_current_user)
):
    """Set user preferences for personalized content"""
    try:
        user_prefs = UserPreferences(
            user_id=current_user.id,
            risk_tolerance=preferences.risk_tolerance,
            investment_horizon=preferences.investment_horizon,
            preferred_markets=preferences.preferred_markets,
            preferred_sectors=preferences.preferred_sectors,
            min_market_cap=preferences.min_market_cap,
            max_market_cap=preferences.max_market_cap,
            preferred_countries=preferences.preferred_countries or [],
            esg_preference=preferences.esg_preference
        )
        
        success = await personalization_engine.update_user_preferences(user_prefs)
        
        if success:
            return {"message": "User preferences updated successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
            
    except Exception as e:
        logger.error(f"Failed to set user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/personalization/preferences")
async def get_user_preferences(current_user: User = Depends(get_current_user)):
    """Get current user preferences"""
    try:
        preferences = await personalization_engine.get_user_preferences(current_user.id)
        
        if preferences:
            return {
                "preferences": preferences.__dict__,
                "status": "success"
            }
        else:
            return {
                "message": "No preferences found",
                "preferences": None,
                "status": "not_found"
            }
            
    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/personalization/feed")
async def get_personalized_feed(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """Get personalized news feed"""
    try:
        articles = await personalization_engine.get_personalized_articles(current_user.id, limit)
        
        return {
            "articles": articles,
            "count": len(articles),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get personalized feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/personalization/strategies")
async def get_personalized_strategies(
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_user)
):
    """Get personalized investment strategies"""
    try:
        strategies = await personalization_engine.get_personalized_strategies(current_user.id, limit)
        
        return {
            "strategies": strategies,
            "count": len(strategies),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get personalized strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting Endpoints

@router.post("/backtesting/run")
async def run_backtest(
    request: BacktestRequestModel,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Run a strategy backtest"""
    try:
        backtest_request = BacktestRequest(
            user_id=current_user.id,
            strategy_name=request.strategy_name,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            strategy_params=StrategyParams(**request.strategy_params),
            benchmark=request.benchmark
        )
        
        # Run backtest in background
        backtest_id = await backtesting_engine.submit_backtest(backtest_request)
        background_tasks.add_task(backtesting_engine.run_backtest, backtest_id)
        
        return {
            "backtest_id": backtest_id,
            "message": "Backtest submitted successfully",
            "status": "submitted"
        }
        
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtesting/results/{backtest_id}")
async def get_backtest_results(
    backtest_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get backtest results by ID"""
    try:
        results = await backtesting_engine.get_backtest_results(backtest_id)
        
        if results:
            # Verify user owns this backtest
            if results.user_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return {
                "results": results.__dict__,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail="Backtest not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtesting/history")
async def get_backtest_history(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """Get user's backtest history"""
    try:
        history = await backtesting_engine.get_user_backtest_history(current_user.id, limit)
        
        return {
            "backtests": history,
            "count": len(history),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get backtest history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/backtesting/results/{backtest_id}")
async def delete_backtest(
    backtest_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a backtest result"""
    try:
        # Verify ownership
        results = await backtesting_engine.get_backtest_results(backtest_id)
        if not results or results.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await backtesting_engine.delete_backtest(backtest_id)
        
        if success:
            return {"message": "Backtest deleted successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete backtest")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time Alerts Endpoints

@router.post("/alerts/subscribe")
async def subscribe_to_alerts(
    subscription: AlertSubscriptionRequest,
    current_user: User = Depends(get_current_user)
):
    """Subscribe to real-time alerts"""
    try:
        alert_subscription = AlertSubscription(
            user_id=current_user.id,
            alert_types=[AlertType(t) for t in subscription.alert_types],
            symbols=subscription.symbols or [],
            keywords=subscription.keywords or [],
            delivery_channels=subscription.delivery_channels,
            min_priority=AlertPriority(subscription.min_priority)
        )
        
        success = await realtime_alerts.create_subscription(alert_subscription)
        
        if success:
            return {"message": "Alert subscription created successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create subscription")
            
    except Exception as e:
        logger.error(f"Failed to subscribe to alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/subscriptions")
async def get_alert_subscriptions(current_user: User = Depends(get_current_user)):
    """Get user's alert subscriptions"""
    try:
        subscriptions = await realtime_alerts.get_user_subscriptions(current_user.id)
        
        return {
            "subscriptions": [sub.__dict__ for sub in subscriptions],
            "count": len(subscriptions),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get alert subscriptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/recent")
async def get_recent_alerts(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """Get recent alerts for user"""
    try:
        alerts = await realtime_alerts.get_user_alerts(current_user.id, limit)
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/alerts/subscriptions/{subscription_id}")
async def unsubscribe_from_alerts(
    subscription_id: str,
    current_user: User = Depends(get_current_user)
):
    """Unsubscribe from alerts"""
    try:
        success = await realtime_alerts.delete_subscription(subscription_id, current_user.id)
        
        if success:
            return {"message": "Unsubscribed successfully", "status": "success"}
        else:
            raise HTTPException(status_code=404, detail="Subscription not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unsubscribe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio Tracking Endpoints

@router.post("/portfolios")
async def create_portfolio(
    portfolio_request: PortfolioCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new portfolio"""
    try:
        portfolio = portfolio_tracker.create_portfolio(
            user_id=current_user.id,
            name=portfolio_request.name,
            description=portfolio_request.description,
            initial_cash=portfolio_request.initial_cash
        )
        
        return {
            "portfolio": {
                "portfolio_id": portfolio.portfolio_id,
                "name": portfolio.name,
                "description": portfolio.description,
                "total_value": float(portfolio.total_value),
                "created_at": portfolio.created_at.isoformat()
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to create portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios")
async def get_user_portfolios(current_user: User = Depends(get_current_user)):
    """Get all user portfolios"""
    try:
        portfolios = portfolio_tracker.get_user_portfolios(current_user.id)
        
        portfolio_summaries = []
        for portfolio in portfolios:
            summary = portfolio_tracker.get_portfolio_summary(portfolio.portfolio_id)
            if summary:
                portfolio_summaries.append(summary)
        
        return {
            "portfolios": portfolio_summaries,
            "count": len(portfolio_summaries),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get user portfolios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios/{portfolio_id}")
async def get_portfolio_details(
    portfolio_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get detailed portfolio information"""
    try:
        portfolio = portfolio_tracker.get_portfolio(portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Verify ownership
        if portfolio.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        summary = portfolio_tracker.get_portfolio_summary(portfolio_id)
        return {
            "portfolio": summary,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolios/{portfolio_id}/transactions")
async def add_transaction(
    portfolio_id: int,
    transaction_request: TransactionRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a transaction to portfolio"""
    try:
        # Verify portfolio ownership
        portfolio = portfolio_tracker.get_portfolio(portfolio_id)
        if not portfolio or portfolio.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        transaction = Transaction(
            transaction_id=f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            portfolio_id=portfolio_id,
            symbol=transaction_request.symbol,
            transaction_type=TransactionType(transaction_request.transaction_type),
            quantity=transaction_request.quantity,
            price=transaction_request.price,
            commission=transaction_request.commission,
            timestamp=datetime.now(),
            notes=transaction_request.notes
        )
        
        success = portfolio_tracker.add_transaction(portfolio_id, transaction)
        
        if success:
            return {"message": "Transaction added successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add transaction")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: int,
    period: str = Query(default="1Y", regex="^(1D|1W|1M|3M|6M|1Y|YTD|ALL)$"),
    current_user: User = Depends(get_current_user)
):
    """Get portfolio performance metrics"""
    try:
        # Verify ownership
        portfolio = portfolio_tracker.get_portfolio(portfolio_id)
        if not portfolio or portfolio.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        performance = portfolio_tracker.calculate_performance_metrics(portfolio_id, period)
        
        if performance:
            return {
                "performance": performance.__dict__,
                "status": "success"
            }
        else:
            return {
                "message": "Insufficient data for performance calculation",
                "performance": None,
                "status": "no_data"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios/{portfolio_id}/risk")
async def get_portfolio_risk(
    portfolio_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get portfolio risk metrics"""
    try:
        # Verify ownership
        portfolio = portfolio_tracker.get_portfolio(portfolio_id)
        if not portfolio or portfolio.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        risk_metrics = portfolio_tracker.calculate_risk_metrics(portfolio_id)
        
        if risk_metrics:
            return {
                "risk_metrics": risk_metrics.__dict__,
                "status": "success"
            }
        else:
            return {
                "message": "Insufficient data for risk calculation",
                "risk_metrics": None,
                "status": "no_data"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios/{portfolio_id}/alerts")
async def get_portfolio_alerts(
    portfolio_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get active alerts for portfolio"""
    try:
        # Verify ownership
        portfolio = portfolio_tracker.get_portfolio(portfolio_id)
        if not portfolio or portfolio.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        alerts = await portfolio_tracker.get_portfolio_alerts(portfolio_id)
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def advanced_features_health():
    """Health check for advanced features services"""
    try:
        health_status = {
            "personalization": "healthy",
            "backtesting": "healthy", 
            "alerts": "healthy",
            "portfolio": "healthy",
            "timestamp": datetime.now().isoformat(),
            "status": "healthy"
        }
        
        # Could add actual health checks for each service
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")