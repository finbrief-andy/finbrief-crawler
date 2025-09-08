"""
Real-time Alert System for FinBrief
Provides instant notifications for breaking news, market events, and portfolio alerts.
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import re

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from src.database.models_migration import User, News, Analysis, Strategy
from src.monitoring.logger import get_logger


class AlertType(Enum):
    """Types of alerts"""
    BREAKING_NEWS = "breaking_news"
    PRICE_ALERT = "price_alert"
    STRATEGY_UPDATE = "strategy_update"
    PORTFOLIO_ALERT = "portfolio_alert"
    MARKET_SUMMARY = "market_summary"
    EARNINGS_ALERT = "earnings_alert"
    CUSTOM_ALERT = "custom_alert"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    PUSH_NOTIFICATION = "push_notification"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    user_id: int
    alert_type: AlertType
    priority: AlertPriority
    channels: List[AlertChannel]
    conditions: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class Alert:
    """Individual alert instance"""
    id: str
    rule_id: str
    user_id: int
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    data: Dict[str, Any]
    channels: List[AlertChannel]
    created_at: datetime
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    is_delivered: bool = False
    is_acknowledged: bool = False


class RealTimeAlertsEngine:
    """
    Real-time alert system for financial news and market events.
    """
    
    def __init__(self):
        self.logger = get_logger("realtime_alerts")
        
        # Active alert rules by user
        self.alert_rules: Dict[int, List[AlertRule]] = defaultdict(list)
        
        # Alert delivery callbacks
        self.delivery_callbacks: Dict[AlertChannel, Callable] = {}
        
        # Alert history (in-memory for demo, would use database in production)
        self.alert_history: List[Alert] = []
        
        # Market data cache for price alerts
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Alert processing queue
        self.alert_queue = asyncio.Queue()
        
        # Background task handle
        self.background_task = None
        
        # Initialize default alert types
        self._init_default_alert_types()
    
    def _init_default_alert_types(self):
        """Initialize default alert configurations"""
        self.default_conditions = {
            AlertType.BREAKING_NEWS: {
                "urgency_threshold": 8,
                "keywords": ["breaking", "urgent", "alert", "developing"],
                "sources": ["reuters", "bloomberg", "cnbc"]
            },
            AlertType.PRICE_ALERT: {
                "price_change_threshold": 0.05,  # 5%
                "volume_threshold": 1.5,  # 1.5x average volume
                "timeframe": "1h"
            },
            AlertType.STRATEGY_UPDATE: {
                "confidence_threshold": 0.8,
                "action_changes": True,
                "new_strategies": True
            },
            AlertType.PORTFOLIO_ALERT: {
                "portfolio_change_threshold": 0.02,  # 2%
                "position_change_threshold": 0.1,   # 10%
                "risk_threshold": 0.15  # 15%
            },
            AlertType.MARKET_SUMMARY: {
                "frequency": "daily",
                "time": "17:00",
                "include_top_movers": True,
                "include_news_summary": True
            },
            AlertType.EARNINGS_ALERT: {
                "days_before": 3,
                "include_estimates": True,
                "include_guidance": True
            }
        }
    
    async def start_alert_processing(self):
        """Start the background alert processing"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._process_alerts())
            self.logger.info("Real-time alerts processing started")
    
    async def stop_alert_processing(self):
        """Stop the background alert processing"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            self.logger.info("Real-time alerts processing stopped")
    
    async def _process_alerts(self):
        """Background task to process alerts"""
        while True:
            try:
                # Process alerts from queue
                while not self.alert_queue.empty():
                    alert = await self.alert_queue.get()
                    await self._deliver_alert(alert)
                
                # Check for new alerts
                await self._check_news_alerts()
                await self._check_price_alerts()
                await self._check_portfolio_alerts()
                
                # Sleep between checks
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error("Alert processing error", exception=e)
                await asyncio.sleep(60)  # Wait longer on error
    
    def create_alert_rule(
        self, 
        user_id: int, 
        alert_type: AlertType, 
        conditions: Dict[str, Any],
        priority: AlertPriority = AlertPriority.MEDIUM,
        channels: List[AlertChannel] = None
    ) -> AlertRule:
        """Create a new alert rule for user"""
        try:
            if channels is None:
                channels = [AlertChannel.IN_APP, AlertChannel.PUSH_NOTIFICATION]
            
            rule = AlertRule(
                id=f"rule_{user_id}_{alert_type.value}_{datetime.now().timestamp()}",
                user_id=user_id,
                alert_type=alert_type,
                priority=priority,
                channels=channels,
                conditions={**self.default_conditions.get(alert_type, {}), **conditions},
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.alert_rules[user_id].append(rule)
            
            self.logger.info(
                "Alert rule created",
                user_id=user_id,
                alert_type=alert_type.value,
                rule_id=rule.id
            )
            
            return rule
            
        except Exception as e:
            self.logger.error("Failed to create alert rule", exception=e, user_id=user_id)
            raise
    
    def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule"""
        try:
            for user_rules in self.alert_rules.values():
                for rule in user_rules:
                    if rule.id == rule_id:
                        for key, value in updates.items():
                            if hasattr(rule, key):
                                setattr(rule, key, value)
                        rule.updated_at = datetime.now()
                        
                        self.logger.info("Alert rule updated", rule_id=rule_id, updates=updates)
                        return True
            
            self.logger.warning("Alert rule not found", rule_id=rule_id)
            return False
            
        except Exception as e:
            self.logger.error("Failed to update alert rule", exception=e, rule_id=rule_id)
            return False
    
    def delete_alert_rule(self, user_id: int, rule_id: str) -> bool:
        """Delete an alert rule"""
        try:
            user_rules = self.alert_rules.get(user_id, [])
            for i, rule in enumerate(user_rules):
                if rule.id == rule_id:
                    del user_rules[i]
                    self.logger.info("Alert rule deleted", user_id=user_id, rule_id=rule_id)
                    return True
            
            self.logger.warning("Alert rule not found for deletion", user_id=user_id, rule_id=rule_id)
            return False
            
        except Exception as e:
            self.logger.error("Failed to delete alert rule", exception=e, rule_id=rule_id)
            return False
    
    def get_user_alert_rules(self, user_id: int) -> List[AlertRule]:
        """Get all alert rules for a user"""
        return self.alert_rules.get(user_id, [])
    
    async def create_alert(
        self,
        rule_id: str,
        user_id: int,
        alert_type: AlertType,
        title: str,
        message: str,
        data: Dict[str, Any] = None,
        priority: AlertPriority = AlertPriority.MEDIUM,
        channels: List[AlertChannel] = None
    ):
        """Create and queue an alert for delivery"""
        try:
            alert = Alert(
                id=f"alert_{user_id}_{datetime.now().timestamp()}",
                rule_id=rule_id,
                user_id=user_id,
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                data=data or {},
                channels=channels or [AlertChannel.IN_APP],
                created_at=datetime.now()
            )
            
            await self.alert_queue.put(alert)
            
            self.logger.info(
                "Alert created",
                user_id=user_id,
                alert_type=alert_type.value,
                priority=priority.value,
                title=title
            )
            
        except Exception as e:
            self.logger.error("Failed to create alert", exception=e, user_id=user_id)
    
    async def _deliver_alert(self, alert: Alert):
        """Deliver alert through specified channels"""
        try:
            delivery_results = {}
            
            for channel in alert.channels:
                callback = self.delivery_callbacks.get(channel)
                if callback:
                    try:
                        result = await callback(alert)
                        delivery_results[channel.value] = result
                    except Exception as e:
                        delivery_results[channel.value] = {"success": False, "error": str(e)}
                        self.logger.error(
                            "Alert delivery failed",
                            exception=e,
                            channel=channel.value,
                            alert_id=alert.id
                        )
                else:
                    # Default delivery (log for demo)
                    delivery_results[channel.value] = await self._default_delivery(alert, channel)
            
            alert.is_delivered = any(result.get("success", False) for result in delivery_results.values())
            alert.delivered_at = datetime.now() if alert.is_delivered else None
            
            # Store in history
            self.alert_history.append(alert)
            
            self.logger.info(
                "Alert delivered",
                alert_id=alert.id,
                user_id=alert.user_id,
                channels=list(delivery_results.keys()),
                success=alert.is_delivered
            )
            
        except Exception as e:
            self.logger.error("Alert delivery failed", exception=e, alert_id=alert.id)
    
    async def _default_delivery(self, alert: Alert, channel: AlertChannel) -> Dict[str, Any]:
        """Default alert delivery (logging for demo purposes)"""
        self.logger.info(
            f"📱 {channel.value.upper()} Alert",
            user_id=alert.user_id,
            priority=alert.priority.value,
            title=alert.title,
            message=alert.message,
            alert_type=alert.alert_type.value
        )
        return {"success": True, "method": "logging", "timestamp": datetime.now().isoformat()}
    
    def register_delivery_callback(self, channel: AlertChannel, callback: Callable):
        """Register a delivery callback for a specific channel"""
        self.delivery_callbacks[channel] = callback
        self.logger.info("Delivery callback registered", channel=channel.value)
    
    async def _check_news_alerts(self):
        """Check for breaking news alerts"""
        try:
            # Get recent news (last 30 minutes)
            cutoff_time = datetime.now() - timedelta(minutes=30)
            
            # This would normally query the database
            # For demo, we'll simulate checking conditions
            
            for user_id, rules in self.alert_rules.items():
                for rule in rules:
                    if (rule.alert_type == AlertType.BREAKING_NEWS and 
                        rule.is_active):
                        
                        # Check if conditions are met
                        if await self._evaluate_breaking_news_conditions(rule):
                            await self.create_alert(
                                rule_id=rule.id,
                                user_id=user_id,
                                alert_type=AlertType.BREAKING_NEWS,
                                title="🚨 Breaking News Alert",
                                message="Important market development detected",
                                data={
                                    "source": "market_monitor",
                                    "timestamp": datetime.now().isoformat(),
                                    "urgency": "high"
                                },
                                priority=AlertPriority.HIGH,
                                channels=rule.channels
                            )
                            
        except Exception as e:
            self.logger.error("News alert check failed", exception=e)
    
    async def _check_price_alerts(self):
        """Check for price movement alerts"""
        try:
            for user_id, rules in self.alert_rules.items():
                for rule in rules:
                    if (rule.alert_type == AlertType.PRICE_ALERT and 
                        rule.is_active):
                        
                        # Evaluate price conditions
                        if await self._evaluate_price_conditions(rule):
                            await self.create_alert(
                                rule_id=rule.id,
                                user_id=user_id,
                                alert_type=AlertType.PRICE_ALERT,
                                title="📈 Price Alert Triggered",
                                message="Significant price movement detected",
                                data={
                                    "ticker": rule.conditions.get("ticker", "UNKNOWN"),
                                    "price_change": rule.conditions.get("price_change_threshold", 0),
                                    "timestamp": datetime.now().isoformat()
                                },
                                priority=AlertPriority.MEDIUM,
                                channels=rule.channels
                            )
                            
        except Exception as e:
            self.logger.error("Price alert check failed", exception=e)
    
    async def _check_portfolio_alerts(self):
        """Check for portfolio-related alerts"""
        try:
            for user_id, rules in self.alert_rules.items():
                for rule in rules:
                    if (rule.alert_type == AlertType.PORTFOLIO_ALERT and 
                        rule.is_active):
                        
                        # Evaluate portfolio conditions
                        if await self._evaluate_portfolio_conditions(rule):
                            await self.create_alert(
                                rule_id=rule.id,
                                user_id=user_id,
                                alert_type=AlertType.PORTFOLIO_ALERT,
                                title="💼 Portfolio Alert",
                                message="Portfolio threshold exceeded",
                                data={
                                    "portfolio_change": rule.conditions.get("portfolio_change_threshold", 0),
                                    "timestamp": datetime.now().isoformat()
                                },
                                priority=AlertPriority.MEDIUM,
                                channels=rule.channels
                            )
                            
        except Exception as e:
            self.logger.error("Portfolio alert check failed", exception=e)
    
    async def _evaluate_breaking_news_conditions(self, rule: AlertRule) -> bool:
        """Evaluate breaking news alert conditions"""
        # Simulate condition evaluation
        # In production, this would check against actual news data
        import random
        return random.random() < 0.01  # 1% chance for demo
    
    async def _evaluate_price_conditions(self, rule: AlertRule) -> bool:
        """Evaluate price alert conditions"""
        # Simulate price condition evaluation
        import random
        return random.random() < 0.02  # 2% chance for demo
    
    async def _evaluate_portfolio_conditions(self, rule: AlertRule) -> bool:
        """Evaluate portfolio alert conditions"""
        # Simulate portfolio condition evaluation
        import random
        return random.random() < 0.005  # 0.5% chance for demo
    
    def acknowledge_alert(self, alert_id: str, user_id: int) -> bool:
        """Mark an alert as acknowledged"""
        try:
            for alert in self.alert_history:
                if alert.id == alert_id and alert.user_id == user_id:
                    alert.is_acknowledged = True
                    alert.acknowledged_at = datetime.now()
                    
                    self.logger.info("Alert acknowledged", alert_id=alert_id, user_id=user_id)
                    return True
            
            self.logger.warning("Alert not found for acknowledgment", alert_id=alert_id, user_id=user_id)
            return False
            
        except Exception as e:
            self.logger.error("Failed to acknowledge alert", exception=e, alert_id=alert_id)
            return False
    
    def get_user_alerts(
        self, 
        user_id: int, 
        limit: int = 50, 
        alert_type: Optional[AlertType] = None,
        acknowledged: Optional[bool] = None
    ) -> List[Alert]:
        """Get alerts for a user with optional filtering"""
        try:
            user_alerts = [alert for alert in self.alert_history if alert.user_id == user_id]
            
            # Apply filters
            if alert_type:
                user_alerts = [alert for alert in user_alerts if alert.alert_type == alert_type]
            
            if acknowledged is not None:
                user_alerts = [alert for alert in user_alerts if alert.is_acknowledged == acknowledged]
            
            # Sort by creation time (newest first)
            user_alerts.sort(key=lambda x: x.created_at, reverse=True)
            
            return user_alerts[:limit]
            
        except Exception as e:
            self.logger.error("Failed to get user alerts", exception=e, user_id=user_id)
            return []
    
    def get_alert_statistics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get alert statistics for a user"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            user_alerts = [
                alert for alert in self.alert_history 
                if alert.user_id == user_id and alert.created_at >= cutoff_date
            ]
            
            if not user_alerts:
                return {"total_alerts": 0, "period_days": days}
            
            # Calculate statistics
            stats = {
                "total_alerts": len(user_alerts),
                "period_days": days,
                "alerts_by_type": defaultdict(int),
                "alerts_by_priority": defaultdict(int),
                "delivery_rate": 0,
                "acknowledgment_rate": 0,
                "average_response_time_minutes": 0,
                "busiest_hour": 0,
                "most_common_type": "",
                "trend": "stable"
            }
            
            delivered_count = 0
            acknowledged_count = 0
            response_times = []
            hourly_distribution = defaultdict(int)
            
            for alert in user_alerts:
                # Type and priority distribution
                stats["alerts_by_type"][alert.alert_type.value] += 1
                stats["alerts_by_priority"][alert.priority.value] += 1
                
                # Delivery rate
                if alert.is_delivered:
                    delivered_count += 1
                
                # Acknowledgment rate and response time
                if alert.is_acknowledged and alert.acknowledged_at and alert.delivered_at:
                    acknowledged_count += 1
                    response_time = (alert.acknowledged_at - alert.delivered_at).total_seconds() / 60
                    response_times.append(response_time)
                
                # Hourly distribution
                hourly_distribution[alert.created_at.hour] += 1
            
            # Calculate rates and averages
            stats["delivery_rate"] = delivered_count / len(user_alerts)
            stats["acknowledgment_rate"] = acknowledged_count / delivered_count if delivered_count > 0 else 0
            stats["average_response_time_minutes"] = np.mean(response_times) if response_times else 0
            stats["busiest_hour"] = max(hourly_distribution, key=hourly_distribution.get) if hourly_distribution else 0
            stats["most_common_type"] = max(stats["alerts_by_type"], key=stats["alerts_by_type"].get) if stats["alerts_by_type"] else ""
            
            # Convert defaultdicts to regular dicts
            stats["alerts_by_type"] = dict(stats["alerts_by_type"])
            stats["alerts_by_priority"] = dict(stats["alerts_by_priority"])
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get alert statistics", exception=e, user_id=user_id)
            return {"total_alerts": 0, "period_days": days, "error": str(e)}
    
    async def send_test_alert(self, user_id: int, alert_type: AlertType = AlertType.CUSTOM_ALERT):
        """Send a test alert to verify delivery"""
        await self.create_alert(
            rule_id="test_rule",
            user_id=user_id,
            alert_type=alert_type,
            title="🧪 Test Alert",
            message="This is a test alert to verify your notification settings are working correctly.",
            data={
                "test": True,
                "timestamp": datetime.now().isoformat()
            },
            priority=AlertPriority.LOW,
            channels=[AlertChannel.IN_APP, AlertChannel.PUSH_NOTIFICATION]
        )
        
        self.logger.info("Test alert sent", user_id=user_id)


# Global real-time alerts engine instance
_alerts_engine = None

async def get_alerts_engine() -> RealTimeAlertsEngine:
    """Get or create global real-time alerts engine instance"""
    global _alerts_engine
    if _alerts_engine is None:
        _alerts_engine = RealTimeAlertsEngine()
        await _alerts_engine.start_alert_processing()
    return _alerts_engine


# Convenience functions for common alert types
async def create_breaking_news_alert(user_id: int, urgency_threshold: int = 8):
    """Create a breaking news alert rule"""
    engine = await get_alerts_engine()
    return engine.create_alert_rule(
        user_id=user_id,
        alert_type=AlertType.BREAKING_NEWS,
        conditions={"urgency_threshold": urgency_threshold},
        priority=AlertPriority.HIGH,
        channels=[AlertChannel.PUSH_NOTIFICATION, AlertChannel.IN_APP]
    )

async def create_price_alert(user_id: int, ticker: str, threshold: float = 0.05):
    """Create a price movement alert rule"""
    engine = await get_alerts_engine()
    return engine.create_alert_rule(
        user_id=user_id,
        alert_type=AlertType.PRICE_ALERT,
        conditions={
            "ticker": ticker,
            "price_change_threshold": threshold
        },
        priority=AlertPriority.MEDIUM,
        channels=[AlertChannel.PUSH_NOTIFICATION, AlertChannel.IN_APP]
    )

async def create_portfolio_alert(user_id: int, threshold: float = 0.02):
    """Create a portfolio change alert rule"""
    engine = await get_alerts_engine()
    return engine.create_alert_rule(
        user_id=user_id,
        alert_type=AlertType.PORTFOLIO_ALERT,
        conditions={"portfolio_change_threshold": threshold},
        priority=AlertPriority.MEDIUM,
        channels=[AlertChannel.PUSH_NOTIFICATION, AlertChannel.IN_APP]
    )


if __name__ == "__main__":
    # Test real-time alerts engine
    import asyncio
    
    async def test_alerts():
        engine = RealTimeAlertsEngine()
        await engine.start_alert_processing()
        
        # Create test alert rule
        rule = engine.create_alert_rule(
            user_id=1,
            alert_type=AlertType.BREAKING_NEWS,
            conditions={"urgency_threshold": 7},
            priority=AlertPriority.HIGH
        )
        
        print(f"Created alert rule: {rule.id}")
        
        # Send test alert
        await engine.send_test_alert(1)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get user alerts
        alerts = engine.get_user_alerts(1)
        print(f"User has {len(alerts)} alerts")
        
        # Get statistics
        stats = engine.get_alert_statistics(1)
        print(f"Alert statistics: {stats}")
        
        await engine.stop_alert_processing()
    
    # Run test
    asyncio.run(test_alerts())
    print("Real-time alerts test completed.")