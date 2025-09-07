"""
Strategy Generator Service
Converts news analysis into actionable investment strategies across different time horizons.
Enhanced with OpenAI GPT-4 for intelligent strategy generation.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from src.database.models_migration import (
    Analysis, News, Strategy, StrategyHorizonEnum, 
    MarketEnum, AssetTypeEnum, SentimentEnum
)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")


class StrategyGenerator:
    """Generates investment strategies from news analysis"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                self.logger.info("OpenAI client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        elif not self.openai_api_key:
            self.logger.warning("No OpenAI API key provided. Using rule-based strategy generation.")
        
    def get_recent_analyses(self, session: Session, horizon: StrategyHorizonEnum, 
                          market: MarketEnum = None, limit: int = 20) -> List[Analysis]:
        """Retrieve recent analyses based on strategy horizon"""
        
        # Define time windows for different horizons
        time_windows = {
            StrategyHorizonEnum.daily: timedelta(days=1),
            StrategyHorizonEnum.weekly: timedelta(days=7),
            StrategyHorizonEnum.monthly: timedelta(days=30),
            StrategyHorizonEnum.yearly: timedelta(days=365)
        }
        
        cutoff_date = datetime.utcnow() - time_windows[horizon]
        
        query = session.query(Analysis).join(News).filter(
            and_(
                News.published_at >= cutoff_date,
                Analysis.is_latest == True
            )
        )
        
        if market:
            query = query.filter(News.market == market)
            
        return query.order_by(desc(News.published_at)).limit(limit).all()
    
    def analyze_sentiment_distribution(self, analyses: List[Analysis]) -> Dict[str, float]:
        """Calculate sentiment distribution from analyses"""
        if not analyses:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for analysis in analyses:
            if analysis.sentiment:
                sentiment_counts[analysis.sentiment.value] += 1
        
        total = len(analyses)
        return {k: v/total for k, v in sentiment_counts.items()}
    
    def extract_key_drivers(self, analyses: List[Analysis]) -> List[str]:
        """Extract key market drivers from recent analyses"""
        drivers = []
        
        for analysis in analyses:
            if analysis.rationale:
                # Simple extraction - in production, use NLP to extract key themes
                drivers.append(analysis.rationale[:200])
        
        # Return top drivers (simplified - should use clustering/ranking)
        return drivers[:5]
    
    def generate_action_recommendations(self, sentiment_dist: Dict[str, float], 
                                      horizon: StrategyHorizonEnum) -> List[Dict[str, Any]]:
        """Generate action recommendations based on sentiment and horizon"""
        recommendations = []
        
        # Simple rule-based recommendations (replace with LLM in production)
        positive_ratio = sentiment_dist.get("positive", 0)
        negative_ratio = sentiment_dist.get("negative", 0)
        
        if positive_ratio > 0.6:
            recommendations.append({
                "action": "BUY",
                "confidence": min(positive_ratio, 0.9),
                "rationale": f"Strong positive sentiment ({positive_ratio:.1%}) suggests favorable market conditions",
                "asset_focus": "stocks"
            })
        elif negative_ratio > 0.6:
            recommendations.append({
                "action": "HOLD",
                "confidence": min(negative_ratio, 0.8),
                "rationale": f"High negative sentiment ({negative_ratio:.1%}) suggests caution",
                "asset_focus": "defensive"
            })
        else:
            recommendations.append({
                "action": "DIVERSIFY",
                "confidence": 0.7,
                "rationale": "Mixed sentiment suggests balanced approach",
                "asset_focus": "balanced"
            })
        
        # Adjust for time horizon
        if horizon == StrategyHorizonEnum.daily:
            recommendations[0]["time_sensitivity"] = "high"
        elif horizon == StrategyHorizonEnum.yearly:
            recommendations[0]["time_sensitivity"] = "low"
        
        return recommendations
    
    def generate_strategy_with_openai(self, analyses: List[Analysis], 
                                    horizon: StrategyHorizonEnum, market: MarketEnum) -> Dict[str, Any]:
        """Generate strategy using OpenAI GPT-4"""
        if not self.openai_client:
            return self.generate_strategy_with_rules(analyses, horizon)
        
        # Prepare news context for OpenAI
        news_context = []
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for analysis in analyses[:15]:  # Use top 15 most recent
            if analysis.news and analysis.news.headline:
                sentiment_val = analysis.sentiment.value if analysis.sentiment else "neutral"
                sentiment_counts[sentiment_val] += 1
                
                news_context.append({
                    "headline": analysis.news.headline,
                    "summary": analysis.news.content_summary or "No summary available",
                    "sentiment": sentiment_val,
                    "asset_type": analysis.news.asset_type.value if analysis.news.asset_type else "stocks",
                    "source": analysis.news.source
                })
        
        # Create context string for prompt
        news_text = "\n".join([
            f"• {item['headline']} (Sentiment: {item['sentiment']}, Asset: {item['asset_type']}, Source: {item['source']})"
            for item in news_context[:10]
        ])
        
        # Determine time horizon context
        horizon_context = {
            StrategyHorizonEnum.daily: "short-term trading decisions for the next 24 hours",
            StrategyHorizonEnum.weekly: "portfolio positioning for the next week", 
            StrategyHorizonEnum.monthly: "sector allocation and medium-term strategy for the next month",
            StrategyHorizonEnum.yearly: "long-term investment outlook and risk management for the next year"
        }
        
        market_context = "Vietnamese" if market == MarketEnum.vn else "global"
        
        # Create the prompt
        system_prompt = f"""You are a professional financial analyst specializing in {market_context} markets. 
        Generate actionable investment strategies based on recent financial news analysis."""
        
        user_prompt = f"""Based on the following recent financial news analysis, generate a {horizon.value} investment strategy for {horizon_context[horizon]}:

RECENT NEWS ANALYSIS:
{news_text}

MARKET SENTIMENT DISTRIBUTION:
- Positive: {sentiment_counts['positive']} articles
- Neutral: {sentiment_counts['neutral']} articles  
- Negative: {sentiment_counts['negative']} articles

Please provide a strategy in the following JSON format:
{{
    "title": "Brief descriptive title",
    "summary": "2-3 sentence executive summary",
    "key_drivers": ["Driver 1", "Driver 2", "Driver 3"],
    "action_recommendations": [
        {{
            "action": "BUY/HOLD/SELL",
            "asset_focus": "stocks/bonds/gold/real_estate/mixed",
            "rationale": "Explanation for this recommendation",
            "confidence": 0.8,
            "time_sensitivity": "high/medium/low"
        }}
    ],
    "confidence_score": 0.85
}}

Focus on actionable insights rather than generic advice. Consider the market sentiment and specific news themes."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use more cost-effective model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            # Parse the JSON response
            strategy_text = response.choices[0].message.content
            
            # Extract JSON from response (handle potential markdown formatting)
            if "```json" in strategy_text:
                strategy_text = strategy_text.split("```json")[1].split("```")[0].strip()
            elif "```" in strategy_text:
                strategy_text = strategy_text.split("```")[1].strip()
            
            strategy_data = json.loads(strategy_text)
            
            self.logger.info(f"Generated OpenAI strategy for {horizon.value} horizon")
            return strategy_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse OpenAI JSON response: {e}")
            return self.generate_strategy_with_rules(analyses, horizon)
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self.generate_strategy_with_rules(analyses, horizon)
    
    def generate_strategy_with_rules(self, analyses: List[Analysis], 
                                   horizon: StrategyHorizonEnum) -> Dict[str, Any]:
        """Fallback rule-based strategy generation"""
        sentiment_dist = self.analyze_sentiment_distribution(analyses)
        recommendations = self.generate_action_recommendations(sentiment_dist, horizon)
        
        return {
            "title": f"{horizon.value.title()} Market Strategy",
            "summary": f"Rule-based analysis of {len(analyses)} recent news items",
            "key_drivers": [
                f"Market sentiment: {max(sentiment_dist, key=sentiment_dist.get)}",
                f"News volume: {len(analyses)} articles analyzed",
                "Automated sentiment-based analysis"
            ],
            "action_recommendations": recommendations,
            "confidence_score": recommendations[0]["confidence"] if recommendations else 0.5
        }
    
    def generate_strategy_with_llm(self, analyses: List[Analysis], 
                                 horizon: StrategyHorizonEnum) -> Dict[str, Any]:
        """Generate strategy using LLM (delegates to OpenAI or rule-based)"""
        return self.generate_strategy_with_openai(analyses, horizon, MarketEnum.global_market)
    
    def create_strategy(self, session: Session, horizon: StrategyHorizonEnum,
                       market: MarketEnum = MarketEnum.global_market,
                       strategy_date: datetime = None) -> Strategy:
        """Create a new strategy for the given horizon and market"""
        
        if strategy_date is None:
            strategy_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get recent analyses
        analyses = self.get_recent_analyses(session, horizon, market)
        
        if not analyses:
            self.logger.warning(f"No analyses found for {horizon.value} strategy generation")
            return None
        
        # Generate strategy content using OpenAI or fallback to rules
        strategy_content = self.generate_strategy_with_openai(analyses, horizon, market)
        
        # Store analysis IDs for traceability (as JSON string for SQLite compatibility)
        import json
        source_analysis_ids = json.dumps([a.id for a in analyses])
        
        # Create strategy record
        strategy = Strategy(
            horizon=horizon,
            market=market,
            strategy_date=strategy_date,
            title=strategy_content["title"],
            summary=strategy_content["summary"],
            key_drivers=strategy_content["key_drivers"],
            action_recommendations=strategy_content["action_recommendations"],
            confidence_score=strategy_content["confidence_score"],
            source_analysis_ids=source_analysis_ids,
            generated_by="strategy_generator_v1.0"
        )
        
        session.add(strategy)
        session.commit()
        session.refresh(strategy)
        
        self.logger.info(f"Generated {horizon.value} strategy (id={strategy.id}) from {len(analyses)} analyses")
        return strategy
    
    def get_latest_strategy(self, session: Session, horizon: StrategyHorizonEnum,
                          market: MarketEnum = MarketEnum.global_market) -> Optional[Strategy]:
        """Get the most recent strategy for given horizon and market"""
        return session.query(Strategy).filter(
            and_(
                Strategy.horizon == horizon,
                Strategy.market == market
            )
        ).order_by(desc(Strategy.strategy_date)).first()
    
    def generate_all_strategies(self, session: Session, 
                              market: MarketEnum = MarketEnum.global_market) -> Dict[str, Strategy]:
        """Generate strategies for all time horizons"""
        results = {}
        
        for horizon in StrategyHorizonEnum:
            try:
                strategy = self.create_strategy(session, horizon, market)
                if strategy:
                    results[horizon.value] = strategy
            except Exception as e:
                self.logger.error(f"Failed to generate {horizon.value} strategy: {e}")
        
        return results


# Convenience function for scheduled strategy generation
def generate_daily_strategies():
    """Generate strategies for all horizons - can be called by CRON"""
    from src.database.models_migration import init_db_and_create
    from sqlalchemy.orm import sessionmaker
    
    engine = init_db_and_create()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        generator = StrategyGenerator()
        
        # Generate for both markets
        for market in [MarketEnum.global_market, MarketEnum.vn]:
            strategies = generator.generate_all_strategies(session, market)
            print(f"Generated {len(strategies)} strategies for {market.value} market")
            
    finally:
        session.close()


if __name__ == "__main__":
    generate_daily_strategies()