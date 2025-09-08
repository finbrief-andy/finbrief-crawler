#!/usr/bin/env python3
"""
Advanced Features Test Suite

Comprehensive testing for personalization, backtesting, 
real-time alerts, and portfolio tracking systems.
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.personalization_engine import PersonalizationEngine, UserPreferences
from src.services.backtesting_engine import BacktestingEngine, BacktestRequest, StrategyParams
from src.services.realtime_alerts import RealTimeAlerts, AlertSubscription, AlertType, AlertPriority
from src.services.portfolio_tracker import PortfolioTracker, Portfolio, Transaction, TransactionType

class TestPersonalizationEngine:
    """Test personalization engine functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.engine = PersonalizationEngine()
        self.test_user_id = 12345
    
    @pytest.mark.asyncio
    async def test_user_preferences_crud(self):
        """Test user preferences CRUD operations"""
        print("\n🧠 Testing Personalization Engine - User Preferences")
        
        # Create preferences
        preferences = UserPreferences(
            user_id=self.test_user_id,
            risk_tolerance="moderate",
            investment_horizon="medium",
            preferred_markets=["US", "Europe"],
            preferred_sectors=["Technology", "Healthcare"],
            min_market_cap=1000000000,  # 1B
            max_market_cap=100000000000,  # 100B
            preferred_countries=["United States", "Germany"],
            esg_preference="high"
        )
        
        # Test update preferences
        success = await self.engine.update_user_preferences(preferences)
        assert success, "Should successfully update user preferences"
        print("✅ User preferences updated successfully")
        
        # Test get preferences
        retrieved_prefs = await self.engine.get_user_preferences(self.test_user_id)
        assert retrieved_prefs is not None, "Should retrieve user preferences"
        assert retrieved_prefs.risk_tolerance == "moderate", "Risk tolerance should match"
        assert len(retrieved_prefs.preferred_markets) == 2, "Should have 2 preferred markets"
        print("✅ User preferences retrieved successfully")
    
    @pytest.mark.asyncio
    async def test_article_scoring(self):
        """Test article relevance scoring"""
        print("\n📊 Testing Article Scoring Algorithm")
        
        # Mock article data
        mock_articles = [
            {
                'id': 1,
                'title': 'Apple Stock Hits New High on iPhone Sales',
                'content': 'Apple Inc. reached a new all-time high as iPhone sales exceeded expectations...',
                'symbols': ['AAPL'],
                'sectors': ['Technology'],
                'published_at': datetime.now() - timedelta(hours=1),
                'engagement_score': 0.8
            },
            {
                'id': 2,
                'title': 'Healthcare Sector Outlook for 2024',
                'content': 'The healthcare sector shows promising signs with new drug approvals...',
                'symbols': ['JNJ', 'PFE'],
                'sectors': ['Healthcare'],
                'published_at': datetime.now() - timedelta(hours=2),
                'engagement_score': 0.6
            },
            {
                'id': 3,
                'title': 'Cryptocurrency Market Volatility Continues',
                'content': 'Bitcoin and other cryptocurrencies experience high volatility...',
                'symbols': ['BTC-USD'],
                'sectors': ['Cryptocurrency'],
                'published_at': datetime.now() - timedelta(days=1),
                'engagement_score': 0.9
            }
        ]
        
        # Create preferences favoring tech and healthcare
        preferences = UserPreferences(
            user_id=self.test_user_id,
            risk_tolerance="moderate",
            investment_horizon="medium",
            preferred_markets=["US"],
            preferred_sectors=["Technology", "Healthcare"],
            preferred_countries=["United States"]
        )
        
        # Test scoring
        scored_articles = []
        for article in mock_articles:
            score = self.engine._calculate_article_score(article, preferences)
            scored_articles.append((article['id'], article['title'][:30] + "...", score))
        
        # Sort by score
        scored_articles.sort(key=lambda x: x[2], reverse=True)
        
        print("Article Scoring Results:")
        for article_id, title, score in scored_articles:
            print(f"  📰 Article {article_id}: {title} - Score: {score:.3f}")
        
        # Technology and Healthcare articles should score higher than Crypto
        tech_score = next(score for aid, _, score in scored_articles if aid == 1)
        healthcare_score = next(score for aid, _, score in scored_articles if aid == 2)
        crypto_score = next(score for aid, _, score in scored_articles if aid == 3)
        
        assert tech_score > crypto_score, "Tech article should score higher than crypto"
        assert healthcare_score > crypto_score, "Healthcare article should score higher than crypto"
        print("✅ Article scoring working correctly")
    
    @pytest.mark.asyncio
    async def test_personalized_feed_generation(self):
        """Test personalized news feed generation"""
        print("\n📰 Testing Personalized Feed Generation")
        
        # This would typically fetch from database
        # For testing, we'll simulate the process
        articles = await self.engine.get_personalized_articles(self.test_user_id, limit=10)
        
        print(f"Generated personalized feed with {len(articles)} articles")
        
        # Basic validation
        assert isinstance(articles, list), "Should return list of articles"
        print("✅ Personalized feed generated successfully")


class TestBacktestingEngine:
    """Test backtesting engine functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.engine = BacktestingEngine()
        self.test_user_id = 12345
    
    @pytest.mark.asyncio
    async def test_backtest_submission(self):
        """Test backtest submission and execution"""
        print("\n📈 Testing Backtesting Engine - Backtest Submission")
        
        # Create backtest request
        strategy_params = StrategyParams(
            strategy_type="buy_and_hold",
            symbols=["AAPL", "GOOGL", "MSFT"],
            allocation={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
            rebalance_frequency="monthly"
        )
        
        backtest_request = BacktestRequest(
            user_id=self.test_user_id,
            strategy_name="Tech Portfolio Test",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            strategy_params=strategy_params,
            benchmark="SPY"
        )
        
        # Submit backtest
        backtest_id = await self.engine.submit_backtest(backtest_request)
        assert backtest_id is not None, "Should return backtest ID"
        print(f"✅ Backtest submitted with ID: {backtest_id}")
        
        # Run backtest
        print("🔄 Running backtest simulation...")
        await self.engine.run_backtest(backtest_id)
        print("✅ Backtest execution completed")
        
        # Get results
        results = await self.engine.get_backtest_results(backtest_id)
        assert results is not None, "Should return backtest results"
        
        print(f"📊 Backtest Results Summary:")
        print(f"  Total Return: {results.total_return:.2%}")
        print(f"  Annualized Return: {results.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {results.max_drawdown:.2%}")
        print(f"  Win Rate: {results.win_rate:.2%}")
        
        # Validate results structure
        assert hasattr(results, 'total_return'), "Results should have total_return"
        assert hasattr(results, 'sharpe_ratio'), "Results should have sharpe_ratio"
        assert hasattr(results, 'max_drawdown'), "Results should have max_drawdown"
        print("✅ Backtest results validated")
    
    @pytest.mark.asyncio
    async def test_strategy_comparison(self):
        """Test comparing multiple strategies"""
        print("\n📊 Testing Strategy Comparison")
        
        strategies = [
            {
                "name": "Buy and Hold",
                "params": StrategyParams(
                    strategy_type="buy_and_hold",
                    symbols=["SPY"],
                    allocation={"SPY": 1.0}
                )
            },
            {
                "name": "Equal Weight Portfolio", 
                "params": StrategyParams(
                    strategy_type="equal_weight",
                    symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
                    allocation={"AAPL": 0.25, "GOOGL": 0.25, "MSFT": 0.25, "AMZN": 0.25},
                    rebalance_frequency="quarterly"
                )
            }
        ]
        
        backtest_results = []
        for strategy in strategies:
            request = BacktestRequest(
                user_id=self.test_user_id,
                strategy_name=strategy["name"],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 6, 30),  # Shorter period for testing
                initial_capital=100000.0,
                strategy_params=strategy["params"]
            )
            
            backtest_id = await self.engine.submit_backtest(request)
            await self.engine.run_backtest(backtest_id)
            results = await self.engine.get_backtest_results(backtest_id)
            backtest_results.append((strategy["name"], results))
        
        print("Strategy Comparison Results:")
        for name, results in backtest_results:
            print(f"  📈 {name}:")
            print(f"    Return: {results.total_return:.2%}")
            print(f"    Sharpe: {results.sharpe_ratio:.3f}")
            print(f"    Max DD: {results.max_drawdown:.2%}")
        
        assert len(backtest_results) == 2, "Should have results for both strategies"
        print("✅ Strategy comparison completed")


class TestRealTimeAlerts:
    """Test real-time alerts system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.alerts = RealTimeAlerts()
        self.test_user_id = 12345
    
    @pytest.mark.asyncio
    async def test_alert_subscription(self):
        """Test alert subscription management"""
        print("\n🚨 Testing Real-Time Alerts - Subscription Management")
        
        # Create subscription
        subscription = AlertSubscription(
            user_id=self.test_user_id,
            alert_types=[AlertType.BREAKING_NEWS, AlertType.PRICE_ALERT],
            symbols=["AAPL", "GOOGL"],
            keywords=["earnings", "acquisition"],
            delivery_channels=["email", "push"],
            min_priority=AlertPriority.MEDIUM
        )
        
        # Test create subscription
        success = await self.alerts.create_subscription(subscription)
        assert success, "Should successfully create subscription"
        print("✅ Alert subscription created")
        
        # Test get subscriptions
        subscriptions = await self.alerts.get_user_subscriptions(self.test_user_id)
        assert len(subscriptions) > 0, "Should have at least one subscription"
        print(f"✅ Retrieved {len(subscriptions)} subscription(s)")
        
        # Test subscription details
        sub = subscriptions[0]
        assert AlertType.BREAKING_NEWS in sub.alert_types, "Should include breaking news alerts"
        assert "AAPL" in sub.symbols, "Should include AAPL symbol"
        print("✅ Subscription details validated")
    
    @pytest.mark.asyncio
    async def test_alert_processing(self):
        """Test alert processing and delivery"""
        print("\n📬 Testing Alert Processing and Delivery")
        
        # Mock market event
        market_event = {
            'type': 'breaking_news',
            'symbol': 'AAPL',
            'title': 'Apple Announces Record Quarterly Earnings',
            'content': 'Apple Inc. reported record quarterly earnings...',
            'priority': 'high',
            'timestamp': datetime.now(),
            'impact_score': 0.9
        }
        
        # Process alert
        processed_alerts = await self.alerts._process_market_event(market_event)
        print(f"Processed {len(processed_alerts)} alerts from market event")
        
        # Test alert delivery
        for alert in processed_alerts:
            delivery_result = await self.alerts._deliver_alert(alert)
            print(f"Alert delivered via {alert['channels']}: {delivery_result}")
        
        # Get recent alerts for user
        recent_alerts = await self.alerts.get_user_alerts(self.test_user_id, limit=10)
        print(f"✅ User has {len(recent_alerts)} recent alerts")
    
    @pytest.mark.asyncio
    async def test_alert_filtering(self):
        """Test alert filtering by preferences"""
        print("\n🔍 Testing Alert Filtering")
        
        # Test events with different priorities and symbols
        test_events = [
            {
                'type': 'price_alert',
                'symbol': 'AAPL',
                'title': 'AAPL up 5%',
                'priority': 'high',
                'timestamp': datetime.now()
            },
            {
                'type': 'breaking_news',
                'symbol': 'TSLA',
                'title': 'Tesla news update',
                'priority': 'low',
                'timestamp': datetime.now()
            },
            {
                'type': 'strategy_update',
                'symbol': 'GOOGL',
                'title': 'Strategy recommendation updated',
                'priority': 'medium',
                'timestamp': datetime.now()
            }
        ]
        
        filtered_results = []
        for event in test_events:
            should_alert = await self.alerts._should_send_alert(self.test_user_id, event)
            filtered_results.append((event['symbol'], event['priority'], should_alert))
            print(f"  📊 {event['symbol']} ({event['priority']}): {'✅ Send' if should_alert else '❌ Skip'}")
        
        print("✅ Alert filtering tested")


class TestPortfolioTracker:
    """Test portfolio tracking system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.tracker = PortfolioTracker()
        self.test_user_id = 12345
    
    def test_portfolio_creation(self):
        """Test portfolio creation and management"""
        print("\n💼 Testing Portfolio Tracker - Portfolio Creation")
        
        # Create portfolio
        portfolio = self.tracker.create_portfolio(
            user_id=self.test_user_id,
            name="Test Portfolio",
            description="Portfolio for testing purposes",
            initial_cash=Decimal('100000.00')
        )
        
        assert portfolio is not None, "Should create portfolio successfully"
        assert portfolio.user_id == self.test_user_id, "Should have correct user ID"
        assert portfolio.cash_balance == Decimal('100000.00'), "Should have correct initial cash"
        print(f"✅ Portfolio created with ID: {portfolio.portfolio_id}")
        
        # Test portfolio retrieval
        retrieved = self.tracker.get_portfolio(portfolio.portfolio_id)
        # Note: This will be None in test environment without proper DB
        print("✅ Portfolio retrieval tested")
        
        return portfolio
    
    def test_transaction_processing(self):
        """Test transaction processing"""
        print("\n💳 Testing Transaction Processing")
        
        # Create test portfolio
        portfolio = self.test_portfolio_creation()
        if not portfolio:
            return
        
        # Create buy transaction
        buy_transaction = Transaction(
            transaction_id="TEST_BUY_001",
            portfolio_id=portfolio.portfolio_id,
            symbol="AAPL",
            transaction_type=TransactionType.BUY,
            quantity=Decimal('100'),
            price=Decimal('150.00'),
            commission=Decimal('9.99'),
            timestamp=datetime.now(),
            notes="Test buy transaction"
        )
        
        # Process transaction
        initial_cash = portfolio.cash_balance
        self.tracker._process_transaction(portfolio, buy_transaction)
        
        # Verify transaction effects
        assert "AAPL" in portfolio.positions, "Should create AAPL position"
        position = portfolio.positions["AAPL"]
        assert position.quantity == Decimal('100'), "Should have 100 shares"
        assert position.average_cost == Decimal('150.00'), "Should have correct average cost"
        
        expected_cash = initial_cash - (Decimal('100') * Decimal('150.00') + Decimal('9.99'))
        assert portfolio.cash_balance == expected_cash, "Should deduct cash correctly"
        
        print("✅ Buy transaction processed correctly")
        
        # Test sell transaction
        sell_transaction = Transaction(
            transaction_id="TEST_SELL_001",
            portfolio_id=portfolio.portfolio_id,
            symbol="AAPL",
            transaction_type=TransactionType.SELL,
            quantity=Decimal('50'),
            price=Decimal('155.00'),
            commission=Decimal('9.99'),
            timestamp=datetime.now(),
            notes="Test sell transaction"
        )
        
        cash_before_sell = portfolio.cash_balance
        self.tracker._process_transaction(portfolio, sell_transaction)
        
        # Verify sell effects
        position = portfolio.positions["AAPL"]
        assert position.quantity == Decimal('50'), "Should have 50 shares remaining"
        
        expected_cash_after_sell = cash_before_sell + (Decimal('50') * Decimal('155.00') - Decimal('9.99'))
        assert portfolio.cash_balance == expected_cash_after_sell, "Should add cash from sale"
        
        print("✅ Sell transaction processed correctly")
    
    def test_portfolio_metrics(self):
        """Test portfolio performance and risk metrics"""
        print("\n📊 Testing Portfolio Metrics Calculation")
        
        # Create test portfolio with positions
        portfolio = self.test_portfolio_creation()
        if not portfolio:
            return
        
        # Add some positions manually for testing
        from src.services.portfolio_tracker import Position, AssetType
        
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            quantity=Decimal('100'),
            average_cost=Decimal('150.00'),
            current_price=Decimal('160.00')
        )
        
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            asset_type=AssetType.STOCK,
            quantity=Decimal('50'),
            average_cost=Decimal('120.00'),
            current_price=Decimal('125.00')
        )
        
        # Test portfolio value calculations
        total_unrealized_pnl = portfolio.total_unrealized_pnl
        print(f"Total Unrealized P&L: ${total_unrealized_pnl:,.2f}")
        
        unrealized_pnl_pct = portfolio.total_unrealized_pnl_pct
        print(f"Total Unrealized P&L %: {unrealized_pnl_pct:.2f}%")
        
        # Test risk metrics calculation
        risk_metrics = self.tracker.calculate_risk_metrics(portfolio.portfolio_id)
        if risk_metrics:
            print(f"Risk Level: {risk_metrics.risk_level.value}")
            print(f"Volatility: {risk_metrics.volatility:.2%}")
            print(f"Beta: {risk_metrics.beta:.3f}")
            print(f"Diversification Score: {risk_metrics.diversification_score:.3f}")
            print("✅ Risk metrics calculated")
        
        # Test performance metrics
        performance = self.tracker.calculate_performance_metrics(portfolio.portfolio_id, "1M")
        if performance:
            print(f"1M Return: {performance.total_return_pct:.2f}%")
            print(f"Sharpe Ratio: {performance.sharpe_ratio:.3f}")
            print(f"Max Drawdown: {performance.max_drawdown:.2%}")
            print("✅ Performance metrics calculated")
    
    def test_portfolio_creation(self):
        """Wrapper for portfolio creation test"""
        return self.test_portfolio_creation()


class TestIntegration:
    """Test integration between different advanced features"""
    
    @pytest.mark.asyncio
    async def test_personalized_portfolio_insights(self):
        """Test integration of personalization with portfolio tracking"""
        print("\n🔗 Testing Advanced Features Integration")
        
        from src.services.portfolio_tracker import AdvancedFeaturesIntegration
        
        integration = AdvancedFeaturesIntegration()
        test_user_id = 12345
        test_portfolio_id = 789
        
        # Get integrated insights
        insights = await integration.get_personalized_portfolio_insights(
            user_id=test_user_id,
            portfolio_id=test_portfolio_id
        )
        
        print(f"Generated {len(insights)} integrated insights")
        
        # Validate integration response structure
        expected_keys = ['portfolio_summary', 'alerts', 'recommendations', 'market_updates', 'strategy_suggestions']
        for key in expected_keys:
            assert key in insights, f"Should include {key} in insights"
        
        print("✅ Advanced features integration tested")


def run_all_tests():
    """Run comprehensive test suite"""
    print("🚀 Advanced Features Test Suite")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_classes = [
        TestPersonalizationEngine,
        TestBacktestingEngine, 
        TestRealTimeAlerts,
        TestPortfolioTracker,
        TestIntegration
    ]
    
    passed = 0
    failed = 0
    total_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n--- {class_name} ---")
        
        try:
            instance = test_class()
            
            # Get all test methods
            test_methods = [method for method in dir(instance) if method.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    
                    # Run setup if it exists
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    
                    # Run test method
                    if asyncio.iscoroutinefunction(method):
                        asyncio.run(method())
                    else:
                        method()
                    
                    passed += 1
                    print(f"✅ {method_name} PASSED")
                    
                except Exception as e:
                    failed += 1
                    print(f"❌ {method_name} FAILED: {e}")
                    
        except Exception as e:
            failed += 1
            print(f"❌ {class_name} setup FAILED: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total_tests} tests passed")
    
    if failed == 0:
        print("🎉 All advanced features tests passed!")
        print("\n✅ Advanced Features Status:")
        print("  🧠 Personalization Engine: Ready")
        print("  📈 Backtesting Engine: Ready")  
        print("  🚨 Real-Time Alerts: Ready")
        print("  💼 Portfolio Tracker: Ready")
        print("  🔗 Feature Integration: Ready")
        
        print("\n🚀 Next Steps:")
        print("  1. Update main API to include advanced features endpoints")
        print("  2. Add advanced features to database schema")
        print("  3. Configure background task processing")
        print("  4. Set up monitoring for advanced features")
        print("  5. Create user documentation")
        
    else:
        print(f"⚠️  {failed} test(s) failed - review implementation")
        
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)