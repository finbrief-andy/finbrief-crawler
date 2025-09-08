"""
Advanced Personalization Engine for FinBrief
Provides user preference-based content curation and strategy personalization.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from src.database.models_migration import User, News, Analysis, Strategy
from src.monitoring.logger import get_logger


@dataclass
class UserPreferences:
    """User preference profile"""
    user_id: int
    risk_tolerance: str  # conservative, moderate, aggressive
    investment_horizon: str  # short, medium, long
    preferred_markets: List[str]
    preferred_sectors: List[str]
    preferred_asset_types: List[str]
    news_frequency: str  # real_time, daily, weekly
    strategy_complexity: str  # simple, moderate, advanced
    notification_preferences: Dict[str, bool]
    updated_at: datetime


@dataclass
class PersonalizationScore:
    """Personalization scoring for content"""
    content_id: str
    content_type: str  # news, analysis, strategy
    relevance_score: float
    preference_match: float
    freshness_score: float
    engagement_score: float
    final_score: float
    explanation: Dict[str, Any]


class PersonalizationEngine:
    """
    Advanced personalization engine for content curation and strategy personalization.
    """
    
    def __init__(self):
        self.logger = get_logger("personalization")
        
        # Scoring weights
        self.weights = {
            "relevance": 0.35,
            "preference": 0.25,
            "freshness": 0.20,
            "engagement": 0.20
        }
        
        # Default preferences
        self.default_preferences = {
            "risk_tolerance": "moderate",
            "investment_horizon": "medium",
            "preferred_markets": ["global"],
            "preferred_sectors": ["technology", "finance", "healthcare"],
            "preferred_asset_types": ["stocks"],
            "news_frequency": "daily",
            "strategy_complexity": "moderate",
            "notification_preferences": {
                "breaking_news": True,
                "strategy_updates": True,
                "portfolio_alerts": True,
                "market_summaries": False
            }
        }
    
    def get_user_preferences(self, session: Session, user_id: int) -> UserPreferences:
        """Get user preferences with fallback to defaults"""
        try:
            user = session.query(User).filter(User.id == user_id).first()
            
            if user and hasattr(user, 'preferences') and user.preferences:
                prefs = user.preferences if isinstance(user.preferences, dict) else json.loads(user.preferences)
            else:
                prefs = self.default_preferences.copy()
            
            # Ensure all required fields are present
            for key, default_value in self.default_preferences.items():
                if key not in prefs:
                    prefs[key] = default_value
            
            return UserPreferences(
                user_id=user_id,
                updated_at=datetime.now(),
                **prefs
            )
            
        except Exception as e:
            self.logger.error("Failed to get user preferences", exception=e, user_id=user_id)
            return UserPreferences(
                user_id=user_id,
                updated_at=datetime.now(),
                **self.default_preferences
            )
    
    def update_user_preferences(self, session: Session, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                self.logger.warning("User not found for preference update", user_id=user_id)
                return False
            
            # Validate preferences
            validated_prefs = self._validate_preferences(preferences)
            
            # Update user preferences
            if hasattr(user, 'preferences'):
                user.preferences = json.dumps(validated_prefs)
            else:
                # Add preferences field if it doesn't exist
                setattr(user, 'preferences', json.dumps(validated_prefs))
            
            session.commit()
            
            self.logger.info("User preferences updated", user_id=user_id, preferences=validated_prefs)
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error("Failed to update user preferences", exception=e, user_id=user_id)
            return False
    
    def _validate_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize user preferences"""
        validated = {}
        
        # Risk tolerance validation
        risk_options = ["conservative", "moderate", "aggressive"]
        validated["risk_tolerance"] = preferences.get("risk_tolerance", "moderate")
        if validated["risk_tolerance"] not in risk_options:
            validated["risk_tolerance"] = "moderate"
        
        # Investment horizon validation
        horizon_options = ["short", "medium", "long"]
        validated["investment_horizon"] = preferences.get("investment_horizon", "medium")
        if validated["investment_horizon"] not in horizon_options:
            validated["investment_horizon"] = "medium"
        
        # Markets validation
        valid_markets = ["global", "vn", "us", "asia", "europe"]
        preferred_markets = preferences.get("preferred_markets", ["global"])
        validated["preferred_markets"] = [m for m in preferred_markets if m in valid_markets]
        if not validated["preferred_markets"]:
            validated["preferred_markets"] = ["global"]
        
        # Sectors validation
        valid_sectors = [
            "technology", "finance", "healthcare", "energy", "consumer_goods",
            "industrial", "materials", "utilities", "real_estate", "telecommunications"
        ]
        preferred_sectors = preferences.get("preferred_sectors", ["technology", "finance"])
        validated["preferred_sectors"] = [s for s in preferred_sectors if s in valid_sectors]
        if not validated["preferred_sectors"]:
            validated["preferred_sectors"] = ["technology", "finance"]
        
        # Asset types validation
        valid_assets = ["stocks", "bonds", "commodities", "crypto", "forex"]
        preferred_assets = preferences.get("preferred_asset_types", ["stocks"])
        validated["preferred_asset_types"] = [a for a in preferred_assets if a in valid_assets]
        if not validated["preferred_asset_types"]:
            validated["preferred_asset_types"] = ["stocks"]
        
        # Frequency validation
        freq_options = ["real_time", "daily", "weekly"]
        validated["news_frequency"] = preferences.get("news_frequency", "daily")
        if validated["news_frequency"] not in freq_options:
            validated["news_frequency"] = "daily"
        
        # Complexity validation
        complexity_options = ["simple", "moderate", "advanced"]
        validated["strategy_complexity"] = preferences.get("strategy_complexity", "moderate")
        if validated["strategy_complexity"] not in complexity_options:
            validated["strategy_complexity"] = "moderate"
        
        # Notification preferences
        default_notifications = self.default_preferences["notification_preferences"]
        validated["notification_preferences"] = preferences.get("notification_preferences", default_notifications)
        
        return validated
    
    def personalize_news_feed(self, session: Session, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Generate personalized news feed for user"""
        try:
            user_prefs = self.get_user_preferences(session, user_id)
            
            # Get recent news
            cutoff_time = datetime.now() - timedelta(days=7)
            
            query = session.query(News).filter(
                News.created_at >= cutoff_time
            )
            
            # Apply market filters
            if "global" not in user_prefs.preferred_markets:
                query = query.filter(News.market.in_(user_prefs.preferred_markets))
            
            # Apply asset type filters
            query = query.filter(News.asset_type.in_(user_prefs.preferred_asset_types))
            
            # Order by recency
            news_items = query.order_by(desc(News.created_at)).limit(limit * 3).all()
            
            # Score and rank news items
            scored_news = []
            for news in news_items:
                score = self._calculate_news_score(news, user_prefs)
                if score.final_score > 0.3:  # Minimum relevance threshold
                    scored_news.append({
                        "news": news,
                        "score": score,
                        "personalization_reason": score.explanation
                    })
            
            # Sort by final score
            scored_news.sort(key=lambda x: x["score"].final_score, reverse=True)
            
            # Return top items
            personalized_feed = []
            for item in scored_news[:limit]:
                news_data = {
                    "id": item["news"].id,
                    "headline": item["news"].headline,
                    "content_summary": item["news"].content_summary,
                    "source": item["news"].source,
                    "published_at": item["news"].published_at.isoformat(),
                    "tickers": item["news"].tickers,
                    "market": item["news"].market.value,
                    "asset_type": item["news"].asset_type.value,
                    "personalization_score": item["score"].final_score,
                    "relevance_explanation": item["personalization_reason"]
                }
                personalized_feed.append(news_data)
            
            self.logger.info("Generated personalized news feed", 
                           user_id=user_id, 
                           total_items=len(personalized_feed),
                           avg_score=np.mean([item["personalization_score"] for item in personalized_feed]))
            
            return personalized_feed
            
        except Exception as e:
            self.logger.error("Failed to generate personalized news feed", exception=e, user_id=user_id)
            return []
    
    def personalize_strategies(self, session: Session, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Generate personalized investment strategies"""
        try:
            user_prefs = self.get_user_preferences(session, user_id)
            
            # Get recent strategies
            cutoff_time = datetime.now() - timedelta(days=30)
            
            strategies = session.query(Strategy).filter(
                Strategy.created_at >= cutoff_time
            ).order_by(desc(Strategy.created_at)).limit(limit * 2).all()
            
            # Score and rank strategies
            scored_strategies = []
            for strategy in strategies:
                score = self._calculate_strategy_score(strategy, user_prefs)
                if score.final_score > 0.4:  # Higher threshold for strategies
                    scored_strategies.append({
                        "strategy": strategy,
                        "score": score,
                        "personalization_reason": score.explanation
                    })
            
            # Sort by final score
            scored_strategies.sort(key=lambda x: x["score"].final_score, reverse=True)
            
            # Return top strategies
            personalized_strategies = []
            for item in scored_strategies[:limit]:
                strategy_data = {
                    "id": item["strategy"].id,
                    "title": item["strategy"].title,
                    "rationale": item["strategy"].rationale,
                    "action_short": item["strategy"].action_short.value,
                    "action_mid": item["strategy"].action_mid.value,
                    "action_long": item["strategy"].action_long.value,
                    "confidence_score": item["strategy"].confidence_score,
                    "created_at": item["strategy"].created_at.isoformat(),
                    "personalization_score": item["score"].final_score,
                    "relevance_explanation": item["personalization_reason"]
                }
                personalized_strategies.append(strategy_data)
            
            self.logger.info("Generated personalized strategies",
                           user_id=user_id,
                           total_strategies=len(personalized_strategies),
                           avg_score=np.mean([s["personalization_score"] for s in personalized_strategies]))
            
            return personalized_strategies
            
        except Exception as e:
            self.logger.error("Failed to generate personalized strategies", exception=e, user_id=user_id)
            return []
    
    def _calculate_news_score(self, news: News, user_prefs: UserPreferences) -> PersonalizationScore:
        """Calculate personalization score for news item"""
        scores = {}
        explanation = {}
        
        # Relevance score (based on content analysis)
        relevance = 0.7  # Base relevance
        if news.tickers:
            relevance += 0.2  # Has specific tickers
        if news.content_summary and len(news.content_summary) > 100:
            relevance += 0.1  # Has substantial content
        scores["relevance"] = min(relevance, 1.0)
        explanation["relevance"] = f"Content relevance: {scores['relevance']:.2f}"
        
        # Preference match score
        preference = 0.0
        
        # Market preference
        if news.market.value in user_prefs.preferred_markets:
            preference += 0.4
            explanation["market_match"] = f"Matches preferred market: {news.market.value}"
        
        # Asset type preference
        if news.asset_type.value in user_prefs.preferred_asset_types:
            preference += 0.3
            explanation["asset_match"] = f"Matches preferred asset: {news.asset_type.value}"
        
        # Sector matching (basic heuristic based on content)
        sector_matches = self._detect_sectors_in_content(news.headline + " " + (news.content_summary or ""))
        matching_sectors = set(sector_matches) & set(user_prefs.preferred_sectors)
        if matching_sectors:
            preference += 0.3
            explanation["sector_match"] = f"Matches sectors: {list(matching_sectors)}"
        
        scores["preference"] = min(preference, 1.0)
        
        # Freshness score
        hours_old = (datetime.now() - news.created_at).total_seconds() / 3600
        if hours_old < 1:
            freshness = 1.0
        elif hours_old < 6:
            freshness = 0.9
        elif hours_old < 24:
            freshness = 0.7
        elif hours_old < 72:
            freshness = 0.5
        else:
            freshness = 0.3
        
        scores["freshness"] = freshness
        explanation["freshness"] = f"Content age: {hours_old:.1f} hours"
        
        # Engagement score (placeholder - could use actual engagement metrics)
        engagement = 0.6  # Base engagement
        if news.urgency_score and news.urgency_score > 7:
            engagement += 0.3
        scores["engagement"] = min(engagement, 1.0)
        explanation["engagement"] = f"Estimated engagement: {scores['engagement']:.2f}"
        
        # Calculate final score
        final_score = sum(scores[key] * self.weights[key] for key in scores)
        
        return PersonalizationScore(
            content_id=str(news.id),
            content_type="news",
            relevance_score=scores["relevance"],
            preference_match=scores["preference"],
            freshness_score=scores["freshness"],
            engagement_score=scores["engagement"],
            final_score=final_score,
            explanation=explanation
        )
    
    def _calculate_strategy_score(self, strategy: Strategy, user_prefs: UserPreferences) -> PersonalizationScore:
        """Calculate personalization score for strategy"""
        scores = {}
        explanation = {}
        
        # Relevance score
        relevance = 0.8  # Base relevance for strategies
        if strategy.confidence_score and strategy.confidence_score > 0.7:
            relevance += 0.2
        scores["relevance"] = min(relevance, 1.0)
        explanation["relevance"] = f"Strategy confidence: {strategy.confidence_score or 0:.2f}"
        
        # Preference match score
        preference = 0.0
        
        # Risk tolerance matching
        strategy_risk = self._assess_strategy_risk(strategy)
        if strategy_risk == user_prefs.risk_tolerance:
            preference += 0.4
            explanation["risk_match"] = f"Matches risk tolerance: {user_prefs.risk_tolerance}"
        elif self._risk_compatibility(strategy_risk, user_prefs.risk_tolerance):
            preference += 0.2
            explanation["risk_match"] = f"Compatible risk level: {strategy_risk}"
        
        # Investment horizon matching
        strategy_horizon = self._assess_strategy_horizon(strategy)
        if strategy_horizon == user_prefs.investment_horizon:
            preference += 0.3
            explanation["horizon_match"] = f"Matches investment horizon: {user_prefs.investment_horizon}"
        elif self._horizon_compatibility(strategy_horizon, user_prefs.investment_horizon):
            preference += 0.15
        
        # Complexity matching
        strategy_complexity = self._assess_strategy_complexity(strategy)
        if strategy_complexity == user_prefs.strategy_complexity:
            preference += 0.3
            explanation["complexity_match"] = f"Matches complexity preference: {user_prefs.strategy_complexity}"
        
        scores["preference"] = min(preference, 1.0)
        
        # Freshness score
        days_old = (datetime.now() - strategy.created_at).days
        if days_old < 1:
            freshness = 1.0
        elif days_old < 3:
            freshness = 0.9
        elif days_old < 7:
            freshness = 0.7
        elif days_old < 30:
            freshness = 0.5
        else:
            freshness = 0.3
        
        scores["freshness"] = freshness
        explanation["freshness"] = f"Strategy age: {days_old} days"
        
        # Engagement score
        engagement = 0.7  # Base engagement for strategies
        scores["engagement"] = engagement
        
        # Calculate final score
        final_score = sum(scores[key] * self.weights[key] for key in scores)
        
        return PersonalizationScore(
            content_id=str(strategy.id),
            content_type="strategy",
            relevance_score=scores["relevance"],
            preference_match=scores["preference"],
            freshness_score=scores["freshness"],
            engagement_score=scores["engagement"],
            final_score=final_score,
            explanation=explanation
        )
    
    def _detect_sectors_in_content(self, content: str) -> List[str]:
        """Detect sectors mentioned in content (basic keyword matching)"""
        content_lower = content.lower()
        sectors = []
        
        sector_keywords = {
            "technology": ["tech", "software", "ai", "artificial intelligence", "cloud", "digital", "cyber"],
            "finance": ["bank", "financial", "fintech", "insurance", "lending", "payment"],
            "healthcare": ["health", "medical", "pharmaceutical", "biotech", "drug", "hospital"],
            "energy": ["oil", "gas", "energy", "renewable", "solar", "wind", "coal"],
            "consumer_goods": ["retail", "consumer", "brand", "products", "food", "beverage"],
            "industrial": ["manufacturing", "industrial", "machinery", "construction", "aerospace"],
            "materials": ["materials", "metals", "mining", "chemicals", "steel"],
            "utilities": ["utilities", "electric", "water", "power", "grid"],
            "real_estate": ["real estate", "property", "housing", "construction", "reit"],
            "telecommunications": ["telecom", "wireless", "network", "broadband", "mobile"]
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                sectors.append(sector)
        
        return sectors
    
    def _assess_strategy_risk(self, strategy: Strategy) -> str:
        """Assess risk level of strategy based on actions"""
        if not strategy.action_short or not strategy.action_mid or not strategy.action_long:
            return "moderate"
        
        actions = [strategy.action_short.value, strategy.action_mid.value, strategy.action_long.value]
        
        # Count aggressive actions
        aggressive_actions = ["BUY", "STRONG_BUY", "SELL", "SHORT"]
        aggressive_count = sum(1 for action in actions if action in aggressive_actions)
        
        if aggressive_count >= 2:
            return "aggressive"
        elif aggressive_count == 0:
            return "conservative"
        else:
            return "moderate"
    
    def _assess_strategy_horizon(self, strategy: Strategy) -> str:
        """Assess investment horizon of strategy"""
        if not strategy.action_long:
            return "short"
        
        # Simple heuristic based on long-term action
        if strategy.action_long.value in ["BUY", "STRONG_BUY", "HOLD"]:
            return "long"
        elif strategy.action_mid and strategy.action_mid.value in ["BUY", "HOLD"]:
            return "medium"
        else:
            return "short"
    
    def _assess_strategy_complexity(self, strategy: Strategy) -> str:
        """Assess complexity of strategy based on content"""
        if not strategy.rationale:
            return "simple"
        
        rationale_length = len(strategy.rationale)
        
        if rationale_length > 500:
            return "advanced"
        elif rationale_length > 200:
            return "moderate"
        else:
            return "simple"
    
    def _risk_compatibility(self, strategy_risk: str, user_risk: str) -> bool:
        """Check if strategy risk is compatible with user risk tolerance"""
        compatibility_matrix = {
            "conservative": ["conservative"],
            "moderate": ["conservative", "moderate"],
            "aggressive": ["moderate", "aggressive"]
        }
        
        return strategy_risk in compatibility_matrix.get(user_risk, [])
    
    def _horizon_compatibility(self, strategy_horizon: str, user_horizon: str) -> bool:
        """Check if strategy horizon is compatible with user horizon"""
        compatibility_matrix = {
            "short": ["short", "medium"],
            "medium": ["short", "medium", "long"],
            "long": ["medium", "long"]
        }
        
        return strategy_horizon in compatibility_matrix.get(user_horizon, [])
    
    def get_user_insights(self, session: Session, user_id: int) -> Dict[str, Any]:
        """Generate user behavior insights for personalization improvement"""
        try:
            user_prefs = self.get_user_preferences(session, user_id)
            
            insights = {
                "profile_summary": {
                    "risk_tolerance": user_prefs.risk_tolerance,
                    "investment_horizon": user_prefs.investment_horizon,
                    "preferred_markets": user_prefs.preferred_markets,
                    "preferred_sectors": user_prefs.preferred_sectors
                },
                "activity_patterns": self._analyze_user_activity(session, user_id),
                "content_preferences": self._analyze_content_preferences(session, user_id),
                "recommendations": self._generate_profile_recommendations(user_prefs)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error("Failed to generate user insights", exception=e, user_id=user_id)
            return {}
    
    def _analyze_user_activity(self, session: Session, user_id: int) -> Dict[str, Any]:
        """Analyze user activity patterns (placeholder for future implementation)"""
        return {
            "login_frequency": "daily",
            "peak_activity_hours": [9, 10, 17, 18],
            "content_engagement": "moderate",
            "session_duration": "15-30 minutes"
        }
    
    def _analyze_content_preferences(self, session: Session, user_id: int) -> Dict[str, Any]:
        """Analyze user content preferences (placeholder for future implementation)"""
        return {
            "preferred_content_length": "medium",
            "preferred_news_sources": ["bloomberg", "reuters"],
            "topics_of_interest": ["earnings", "market_trends", "economic_indicators"],
            "strategy_preference": "data_driven"
        }
    
    def _generate_profile_recommendations(self, user_prefs: UserPreferences) -> List[str]:
        """Generate recommendations for improving user profile"""
        recommendations = []
        
        if len(user_prefs.preferred_sectors) < 3:
            recommendations.append("Consider adding more sectors to diversify your interests")
        
        if len(user_prefs.preferred_markets) == 1:
            recommendations.append("Explore additional markets for broader opportunities")
        
        if user_prefs.strategy_complexity == "simple":
            recommendations.append("Try moderate complexity strategies as you gain experience")
        
        return recommendations


# Global personalization engine instance
_personalization_engine = None

def get_personalization_engine() -> PersonalizationEngine:
    """Get or create global personalization engine instance"""
    global _personalization_engine
    if _personalization_engine is None:
        _personalization_engine = PersonalizationEngine()
    return _personalization_engine


if __name__ == "__main__":
    # Test personalization engine
    engine = PersonalizationEngine()
    
    # Test preference validation
    test_prefs = {
        "risk_tolerance": "aggressive",
        "investment_horizon": "long",
        "preferred_markets": ["global", "us"],
        "preferred_sectors": ["technology", "healthcare"],
        "preferred_asset_types": ["stocks", "crypto"],
        "news_frequency": "daily",
        "strategy_complexity": "advanced"
    }
    
    validated = engine._validate_preferences(test_prefs)
    print("Validated preferences:", validated)
    
    # Test sector detection
    content = "Apple Inc. announced new AI technology for healthcare applications"
    sectors = engine._detect_sectors_in_content(content)
    print("Detected sectors:", sectors)
    
    print("Personalization engine test completed.")