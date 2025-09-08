"""
Advanced Backtesting Engine for Investment Strategies
Provides historical performance analysis and strategy validation.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from src.database.models_migration import Strategy, Analysis, News
from src.monitoring.logger import get_logger


@dataclass
class BacktestResult:
    """Backtesting results for a strategy"""
    strategy_id: int
    backtest_period: Tuple[datetime, datetime]
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    average_trade_return: float
    risk_adjusted_return: float
    benchmark_comparison: float
    performance_metrics: Dict[str, Any]
    trade_history: List[Dict[str, Any]]
    created_at: datetime


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    positions: Dict[str, Dict[str, Any]]  # ticker -> {shares, value, weight}
    total_invested: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Trade:
    """Individual trade execution"""
    timestamp: datetime
    ticker: str
    action: str  # BUY, SELL, HOLD
    quantity: float
    price: float
    value: float
    commission: float
    strategy_id: int
    reason: str


class BacktestingEngine:
    """
    Comprehensive backtesting engine for investment strategy validation.
    """
    
    def __init__(self):
        self.logger = get_logger("backtesting")
        
        # Default backtesting parameters
        self.default_params = {
            "initial_capital": 100000.0,
            "commission_rate": 0.001,  # 0.1% per trade
            "slippage": 0.0005,  # 0.05% slippage
            "risk_free_rate": 0.02,  # 2% annual risk-free rate
            "benchmark_return": 0.08,  # 8% annual benchmark
            "max_position_size": 0.1,  # Maximum 10% per position
            "rebalance_frequency": "monthly"
        }
    
    def backtest_strategy(
        self, 
        session: Session, 
        strategy_id: int, 
        start_date: datetime, 
        end_date: datetime,
        initial_capital: float = 100000.0,
        params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run comprehensive backtesting for a strategy over specified period.
        """
        try:
            # Merge parameters
            backtest_params = {**self.default_params, **(params or {})}
            backtest_params["initial_capital"] = initial_capital
            
            self.logger.info(
                "Starting strategy backtest",
                strategy_id=strategy_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                initial_capital=initial_capital
            )
            
            # Get strategy
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            # Initialize portfolio
            portfolio = self._initialize_portfolio(initial_capital)
            
            # Get historical data and signals
            historical_data = self._get_historical_data(session, start_date, end_date)
            strategy_signals = self._generate_strategy_signals(strategy, historical_data)
            
            # Execute backtest
            portfolio_history = []
            trade_history = []
            
            for date in pd.date_range(start_date, end_date, freq='D'):
                if date.weekday() >= 5:  # Skip weekends
                    continue
                
                # Update portfolio with market data
                self._update_portfolio_prices(portfolio, historical_data, date)
                
                # Execute strategy signals
                trades = self._execute_strategy_signals(
                    portfolio, strategy_signals, date, backtest_params
                )
                trade_history.extend(trades)
                
                # Record portfolio snapshot
                snapshot = PortfolioSnapshot(
                    timestamp=date,
                    portfolio_value=portfolio["total_value"],
                    cash=portfolio["cash"],
                    positions=portfolio["positions"].copy(),
                    total_invested=portfolio["total_invested"],
                    unrealized_pnl=portfolio["unrealized_pnl"],
                    realized_pnl=portfolio["realized_pnl"]
                )
                portfolio_history.append(snapshot)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio_history, trade_history, backtest_params
            )
            
            # Create backtest result
            result = BacktestResult(
                strategy_id=strategy_id,
                backtest_period=(start_date, end_date),
                total_return=performance_metrics["total_return"],
                annualized_return=performance_metrics["annualized_return"],
                volatility=performance_metrics["volatility"],
                sharpe_ratio=performance_metrics["sharpe_ratio"],
                max_drawdown=performance_metrics["max_drawdown"],
                win_rate=performance_metrics["win_rate"],
                total_trades=len(trade_history),
                profitable_trades=performance_metrics["profitable_trades"],
                average_trade_return=performance_metrics["average_trade_return"],
                risk_adjusted_return=performance_metrics["risk_adjusted_return"],
                benchmark_comparison=performance_metrics["benchmark_comparison"],
                performance_metrics=performance_metrics,
                trade_history=[asdict(trade) for trade in trade_history],
                created_at=datetime.now()
            )
            
            self.logger.info(
                "Backtesting completed",
                strategy_id=strategy_id,
                total_return=f"{result.total_return:.2%}",
                sharpe_ratio=f"{result.sharpe_ratio:.2f}",
                total_trades=result.total_trades
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Backtesting failed", exception=e, strategy_id=strategy_id)
            raise
    
    def _initialize_portfolio(self, initial_capital: float) -> Dict[str, Any]:
        """Initialize portfolio state"""
        return {
            "cash": initial_capital,
            "positions": {},  # ticker -> {shares, avg_price, current_price, value}
            "total_value": initial_capital,
            "total_invested": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "initial_capital": initial_capital
        }
    
    def _get_historical_data(self, session: Session, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical price data for backtesting.
        Note: This is a simplified implementation. In production, you would connect to
        a real financial data provider like Yahoo Finance, Alpha Vantage, or Bloomberg.
        """
        try:
            # Generate synthetic price data for demonstration
            # In production, replace with actual market data API calls
            
            date_range = pd.date_range(start_date, end_date, freq='D')
            
            # Common tickers for demonstration
            tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
            
            # Generate realistic price movements
            np.random.seed(42)  # For reproducible results
            
            data = []
            for ticker in tickers:
                base_price = np.random.uniform(50, 300)  # Starting price
                
                for date in date_range:
                    if date.weekday() < 5:  # Only business days
                        # Random walk with slight upward bias
                        daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
                        base_price *= (1 + daily_return)
                        
                        # Add some realistic bounds
                        base_price = max(base_price, 10.0)  # Minimum price
                        
                        data.append({
                            'date': date,
                            'ticker': ticker,
                            'open': base_price * np.random.uniform(0.99, 1.01),
                            'high': base_price * np.random.uniform(1.00, 1.03),
                            'low': base_price * np.random.uniform(0.97, 1.00),
                            'close': base_price,
                            'volume': int(np.random.uniform(1000000, 10000000))
                        })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            self.logger.info(
                "Historical data generated",
                tickers=len(tickers),
                date_range=f"{start_date.date()} to {end_date.date()}",
                total_records=len(df)
            )
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to get historical data", exception=e)
            # Return minimal data structure
            return pd.DataFrame(columns=['ticker', 'open', 'high', 'low', 'close', 'volume'])
    
    def _generate_strategy_signals(self, strategy: Strategy, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on strategy.
        This is a simplified implementation - in production, you would parse
        the strategy logic and convert it to actionable signals.
        """
        try:
            signals = []
            
            # Get unique dates and tickers
            dates = historical_data.index.unique()
            tickers = historical_data['ticker'].unique()
            
            # Simple strategy signal generation based on strategy actions
            signal_strength = {
                'STRONG_BUY': 1.0,
                'BUY': 0.6,
                'HOLD': 0.0,
                'SELL': -0.6,
                'STRONG_SELL': -1.0
            }
            
            # Convert strategy actions to signals
            short_signal = signal_strength.get(strategy.action_short.value if strategy.action_short else 'HOLD', 0.0)
            mid_signal = signal_strength.get(strategy.action_mid.value if strategy.action_mid else 'HOLD', 0.0)
            long_signal = signal_strength.get(strategy.action_long.value if strategy.action_long else 'HOLD', 0.0)
            
            # Generate signals for each date and ticker
            for date in dates[::30]:  # Signal every 30 days (monthly rebalancing)
                for ticker in tickers:
                    # Combine time horizon signals with decay
                    combined_signal = (
                        short_signal * 0.5 +  # Short term weight
                        mid_signal * 0.3 +    # Mid term weight  
                        long_signal * 0.2     # Long term weight
                    )
                    
                    # Add some randomness based on market conditions
                    noise = np.random.normal(0, 0.1)
                    final_signal = np.clip(combined_signal + noise, -1.0, 1.0)
                    
                    signals.append({
                        'date': date,
                        'ticker': ticker,
                        'signal': final_signal,
                        'confidence': strategy.confidence_score or 0.7,
                        'reason': f"Strategy {strategy.id} signal"
                    })
            
            signals_df = pd.DataFrame(signals)
            if not signals_df.empty:
                signals_df.set_index('date', inplace=True)
            
            self.logger.info(
                "Strategy signals generated",
                strategy_id=strategy.id,
                total_signals=len(signals_df),
                avg_signal=signals_df['signal'].mean() if not signals_df.empty else 0
            )
            
            return signals_df
            
        except Exception as e:
            self.logger.error("Failed to generate strategy signals", exception=e)
            return pd.DataFrame(columns=['ticker', 'signal', 'confidence', 'reason'])
    
    def _update_portfolio_prices(self, portfolio: Dict[str, Any], historical_data: pd.DataFrame, date: datetime):
        """Update portfolio positions with current market prices"""
        try:
            portfolio["unrealized_pnl"] = 0.0
            portfolio["total_value"] = portfolio["cash"]
            
            for ticker, position in portfolio["positions"].items():
                # Get current price
                price_data = historical_data[
                    (historical_data.index == date) & 
                    (historical_data['ticker'] == ticker)
                ]
                
                if not price_data.empty:
                    current_price = price_data['close'].iloc[0]
                    position["current_price"] = current_price
                    position["value"] = position["shares"] * current_price
                    
                    # Calculate unrealized P&L
                    position["unrealized_pnl"] = (current_price - position["avg_price"]) * position["shares"]
                    portfolio["unrealized_pnl"] += position["unrealized_pnl"]
                    portfolio["total_value"] += position["value"]
                
        except Exception as e:
            self.logger.error("Failed to update portfolio prices", exception=e, date=date.isoformat())
    
    def _execute_strategy_signals(
        self, 
        portfolio: Dict[str, Any], 
        signals: pd.DataFrame, 
        date: datetime,
        params: Dict[str, Any]
    ) -> List[Trade]:
        """Execute trading signals"""
        trades = []
        
        try:
            # Get signals for this date
            date_signals = signals[signals.index == date] if not signals.empty else pd.DataFrame()
            
            for _, signal_row in date_signals.iterrows():
                ticker = signal_row['ticker']
                signal = signal_row['signal']
                confidence = signal_row['confidence']
                
                # Skip weak signals
                if abs(signal) < 0.2:
                    continue
                
                # Calculate position size based on signal strength and confidence
                max_position_value = portfolio["total_value"] * params["max_position_size"]
                position_value = max_position_value * abs(signal) * confidence
                
                # Get current price (simplified)
                current_price = 100.0  # Placeholder - would get from historical_data
                
                # Calculate shares to trade
                shares_to_trade = int(position_value / current_price)
                
                if shares_to_trade > 0:
                    if signal > 0:  # Buy signal
                        trade = self._execute_buy(portfolio, ticker, shares_to_trade, current_price, params)
                    else:  # Sell signal
                        trade = self._execute_sell(portfolio, ticker, shares_to_trade, current_price, params)
                    
                    if trade:
                        trades.append(trade)
            
        except Exception as e:
            self.logger.error("Failed to execute strategy signals", exception=e, date=date.isoformat())
        
        return trades
    
    def _execute_buy(self, portfolio: Dict[str, Any], ticker: str, shares: int, price: float, params: Dict[str, Any]) -> Optional[Trade]:
        """Execute buy order"""
        try:
            total_cost = shares * price
            commission = total_cost * params["commission_rate"]
            slippage_cost = total_cost * params["slippage"]
            total_cost_with_fees = total_cost + commission + slippage_cost
            
            # Check if we have enough cash
            if portfolio["cash"] < total_cost_with_fees:
                return None
            
            # Update portfolio
            portfolio["cash"] -= total_cost_with_fees
            portfolio["total_invested"] += total_cost
            
            if ticker in portfolio["positions"]:
                # Add to existing position (average price)
                existing = portfolio["positions"][ticker]
                total_shares = existing["shares"] + shares
                total_cost_existing = existing["shares"] * existing["avg_price"] + total_cost
                new_avg_price = total_cost_existing / total_shares
                
                portfolio["positions"][ticker] = {
                    "shares": total_shares,
                    "avg_price": new_avg_price,
                    "current_price": price,
                    "value": total_shares * price,
                    "unrealized_pnl": 0.0
                }
            else:
                # New position
                portfolio["positions"][ticker] = {
                    "shares": shares,
                    "avg_price": price,
                    "current_price": price,
                    "value": shares * price,
                    "unrealized_pnl": 0.0
                }
            
            return Trade(
                timestamp=datetime.now(),
                ticker=ticker,
                action="BUY",
                quantity=shares,
                price=price,
                value=total_cost,
                commission=commission + slippage_cost,
                strategy_id=0,  # Would be set by calling function
                reason="Strategy signal"
            )
            
        except Exception as e:
            self.logger.error("Failed to execute buy order", exception=e, ticker=ticker)
            return None
    
    def _execute_sell(self, portfolio: Dict[str, Any], ticker: str, shares: int, price: float, params: Dict[str, Any]) -> Optional[Trade]:
        """Execute sell order"""
        try:
            if ticker not in portfolio["positions"]:
                return None
            
            position = portfolio["positions"][ticker]
            available_shares = position["shares"]
            
            # Limit sell to available shares
            shares_to_sell = min(shares, available_shares)
            
            if shares_to_sell <= 0:
                return None
            
            total_proceeds = shares_to_sell * price
            commission = total_proceeds * params["commission_rate"]
            slippage_cost = total_proceeds * params["slippage"]
            net_proceeds = total_proceeds - commission - slippage_cost
            
            # Calculate realized P&L
            cost_basis = shares_to_sell * position["avg_price"]
            realized_pnl = net_proceeds - cost_basis
            
            # Update portfolio
            portfolio["cash"] += net_proceeds
            portfolio["realized_pnl"] += realized_pnl
            
            # Update position
            position["shares"] -= shares_to_sell
            if position["shares"] <= 0:
                del portfolio["positions"][ticker]
            else:
                position["value"] = position["shares"] * price
            
            return Trade(
                timestamp=datetime.now(),
                ticker=ticker,
                action="SELL",
                quantity=shares_to_sell,
                price=price,
                value=total_proceeds,
                commission=commission + slippage_cost,
                strategy_id=0,
                reason="Strategy signal"
            )
            
        except Exception as e:
            self.logger.error("Failed to execute sell order", exception=e, ticker=ticker)
            return None
    
    def _calculate_performance_metrics(
        self, 
        portfolio_history: List[PortfolioSnapshot],
        trade_history: List[Trade],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not portfolio_history:
                return {}
            
            # Convert to pandas for easier calculations
            values = [snapshot.portfolio_value for snapshot in portfolio_history]
            dates = [snapshot.timestamp for snapshot in portfolio_history]
            
            df = pd.DataFrame({'value': values}, index=dates)
            df['returns'] = df['value'].pct_change().dropna()
            
            initial_value = portfolio_history[0].portfolio_value
            final_value = portfolio_history[-1].portfolio_value
            
            # Basic metrics
            total_return = (final_value - initial_value) / initial_value
            
            # Annualized return
            days = (dates[-1] - dates[0]).days
            years = days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility (annualized)
            volatility = df['returns'].std() * np.sqrt(252) if len(df['returns']) > 1 else 0
            
            # Sharpe ratio
            excess_return = annualized_return - params["risk_free_rate"]
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = df['value'].expanding().max()
            drawdown = (df['value'] - peak) / peak
            max_drawdown = drawdown.min()
            
            # Trading metrics
            profitable_trades = sum(1 for trade in trade_history 
                                  if trade.action == "SELL" and 
                                     self._calculate_trade_pnl(trade, trade_history) > 0)
            total_trades = len([t for t in trade_history if t.action == "SELL"])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Average trade return
            trade_returns = [self._calculate_trade_pnl(trade, trade_history) 
                           for trade in trade_history if trade.action == "SELL"]
            average_trade_return = np.mean(trade_returns) if trade_returns else 0
            
            # Risk-adjusted return
            risk_adjusted_return = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Benchmark comparison
            benchmark_return = params["benchmark_return"] * years
            benchmark_comparison = total_return - benchmark_return
            
            metrics = {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profitable_trades": profitable_trades,
                "average_trade_return": average_trade_return,
                "risk_adjusted_return": risk_adjusted_return,
                "benchmark_comparison": benchmark_comparison,
                "calmar_ratio": annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                "sortino_ratio": self._calculate_sortino_ratio(df['returns'], params["risk_free_rate"]),
                "information_ratio": self._calculate_information_ratio(df['returns'], benchmark_return / 252),
                "beta": self._calculate_beta(df['returns']),
                "alpha": self._calculate_alpha(df['returns'], params["risk_free_rate"], benchmark_return / 252),
                "var_95": np.percentile(df['returns'], 5) if len(df['returns']) > 0 else 0,
                "skewness": df['returns'].skew() if len(df['returns']) > 2 else 0,
                "kurtosis": df['returns'].kurtosis() if len(df['returns']) > 3 else 0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to calculate performance metrics", exception=e)
            return {}
    
    def _calculate_trade_pnl(self, sell_trade: Trade, trade_history: List[Trade]) -> float:
        """Calculate P&L for a specific trade"""
        # Find corresponding buy trade (simplified)
        buy_trades = [t for t in trade_history 
                     if t.ticker == sell_trade.ticker and 
                        t.action == "BUY" and 
                        t.timestamp <= sell_trade.timestamp]
        
        if not buy_trades:
            return 0.0
        
        # Use most recent buy trade
        buy_trade = max(buy_trades, key=lambda x: x.timestamp)
        
        return (sell_trade.price - buy_trade.price) * sell_trade.quantity - sell_trade.commission - buy_trade.commission
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
            
            if downside_deviation > 0:
                return (excess_returns.mean() * 252) / downside_deviation
            return 0.0
        except:
            return 0.0
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_return: float) -> float:
        """Calculate Information ratio"""
        try:
            excess_returns = returns - benchmark_return
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            if tracking_error > 0:
                return (excess_returns.mean() * 252) / tracking_error
            return 0.0
        except:
            return 0.0
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta (simplified - assumes market return)"""
        try:
            # Simplified beta calculation
            market_returns = np.random.normal(0.0003, 0.015, len(returns))  # Mock market returns
            if len(returns) > 1 and len(market_returns) > 1:
                covariance = np.cov(returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                return covariance / market_variance if market_variance > 0 else 1.0
            return 1.0
        except:
            return 1.0
    
    def _calculate_alpha(self, returns: pd.Series, risk_free_rate: float, market_return: float) -> float:
        """Calculate alpha"""
        try:
            portfolio_return = returns.mean() * 252
            beta = self._calculate_beta(returns)
            expected_return = risk_free_rate + beta * (market_return * 252 - risk_free_rate)
            return portfolio_return - expected_return
        except:
            return 0.0
    
    def compare_strategies(
        self, 
        session: Session, 
        strategy_ids: List[int], 
        start_date: datetime, 
        end_date: datetime,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """Compare multiple strategies using backtesting"""
        try:
            results = {}
            
            for strategy_id in strategy_ids:
                result = self.backtest_strategy(
                    session, strategy_id, start_date, end_date, initial_capital
                )
                results[strategy_id] = result
            
            # Create comparison metrics
            comparison = {
                "strategies": {str(sid): asdict(result) for sid, result in results.items()},
                "rankings": {
                    "total_return": sorted(results.items(), key=lambda x: x[1].total_return, reverse=True),
                    "sharpe_ratio": sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True),
                    "max_drawdown": sorted(results.items(), key=lambda x: x[1].max_drawdown),
                    "win_rate": sorted(results.items(), key=lambda x: x[1].win_rate, reverse=True)
                },
                "summary_stats": {
                    "avg_return": np.mean([r.total_return for r in results.values()]),
                    "best_return": max(r.total_return for r in results.values()),
                    "worst_return": min(r.total_return for r in results.values()),
                    "avg_sharpe": np.mean([r.sharpe_ratio for r in results.values()]),
                    "correlation_matrix": self._calculate_strategy_correlations(results)
                }
            }
            
            self.logger.info(
                "Strategy comparison completed",
                strategy_count=len(strategy_ids),
                best_strategy=comparison["rankings"]["total_return"][0][0],
                best_return=f"{comparison['rankings']['total_return'][0][1].total_return:.2%}"
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error("Strategy comparison failed", exception=e)
            return {}
    
    def _calculate_strategy_correlations(self, results: Dict[int, BacktestResult]) -> Dict[str, float]:
        """Calculate correlation matrix between strategies"""
        # Simplified correlation calculation
        # In production, you would calculate based on daily returns
        correlations = {}
        
        strategy_ids = list(results.keys())
        for i, sid1 in enumerate(strategy_ids):
            for j, sid2 in enumerate(strategy_ids[i+1:], i+1):
                # Mock correlation calculation
                corr = np.random.uniform(0.3, 0.8)  # Realistic correlation range
                correlations[f"{sid1}_{sid2}"] = corr
        
        return correlations


# Global backtesting engine instance
_backtesting_engine = None

def get_backtesting_engine() -> BacktestingEngine:
    """Get or create global backtesting engine instance"""
    global _backtesting_engine
    if _backtesting_engine is None:
        _backtesting_engine = BacktestingEngine()
    return _backtesting_engine


if __name__ == "__main__":
    # Test backtesting engine
    engine = BacktestingEngine()
    
    # Test portfolio initialization
    portfolio = engine._initialize_portfolio(100000.0)
    print("Initial portfolio:", portfolio)
    
    # Test performance metrics calculation
    from datetime import datetime, timedelta
    
    # Mock portfolio history
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    portfolio_history = [
        PortfolioSnapshot(
            timestamp=date,
            portfolio_value=100000 * (1 + 0.001 * i + np.random.normal(0, 0.01)),
            cash=50000,
            positions={},
            total_invested=50000,
            unrealized_pnl=0,
            realized_pnl=0
        ) for i, date in enumerate(dates)
    ]
    
    metrics = engine._calculate_performance_metrics(portfolio_history, [], engine.default_params)
    print("Performance metrics:", metrics)
    
    print("Backtesting engine test completed.")