#!/usr/bin/env python
"""
Integration tests for custom indicators within the backtesting framework.

This module tests how custom indicators integrate with the full backtesting workflow,
including strategy development, signal generation, and performance evaluation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components for testing
from platon_light.indicators.base import BaseIndicator
from platon_light.indicators.basic import SMA, EMA, RSI, BollingerBands, MACD
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.strategy import BaseStrategy


class CustomMomentumIndicator(BaseIndicator):
    """
    Custom momentum indicator for testing.
    
    This indicator calculates momentum as the percentage change over a specified period.
    """
    
    def __init__(self, period=14, input_column='close', output_column=None):
        """
        Initialize the indicator.
        
        Args:
            period: Number of periods for momentum calculation
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
    
    @property
    def name(self):
        """Get the indicator name with period."""
        return f"Momentum_{self.period}"
    
    def calculate(self, data):
        """
        Calculate the momentum indicator.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with momentum values
        """
        # Calculate momentum as percentage change
        momentum = data[self.input_column].pct_change(self.period) * 100
        return momentum


class CustomVolatilityIndicator(BaseIndicator):
    """
    Custom volatility indicator for testing.
    
    This indicator calculates volatility as the standard deviation of returns
    over a specified period, annualized.
    """
    
    def __init__(self, period=20, annualization_factor=252, input_column='close', output_column=None):
        """
        Initialize the indicator.
        
        Args:
            period: Number of periods for volatility calculation
            annualization_factor: Factor to annualize volatility (252 for daily, 52 for weekly, etc.)
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
        self.annualization_factor = annualization_factor
    
    @property
    def name(self):
        """Get the indicator name with period."""
        return f"Volatility_{self.period}"
    
    def calculate(self, data):
        """
        Calculate the volatility indicator.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with volatility values
        """
        # Calculate returns
        returns = data[self.input_column].pct_change()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=self.period).std()
        
        # Annualize volatility
        volatility = volatility * np.sqrt(self.annualization_factor)
        
        return volatility


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy for testing custom indicators.
    
    This strategy uses the custom momentum indicator to generate buy/sell signals.
    """
    
    def __init__(self, momentum_period=14, momentum_threshold=2.0):
        """
        Initialize the strategy.
        
        Args:
            momentum_period: Period for momentum calculation
            momentum_threshold: Threshold for buy/sell signals
        """
        super().__init__()
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.momentum = CustomMomentumIndicator(period=momentum_period)
    
    def prepare_data(self, data):
        """
        Prepare data for the strategy.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with added indicators
        """
        # Add momentum indicator
        data = self.momentum(data)
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals.
        
        Args:
            data: DataFrame containing price data and indicators
            
        Returns:
            DataFrame with added signal column
        """
        # Initialize signal column
        data['signal'] = 0
        
        # Generate buy signals when momentum is above threshold
        data.loc[data[self.momentum.output_column] > self.momentum_threshold, 'signal'] = 1
        
        # Generate sell signals when momentum is below negative threshold
        data.loc[data[self.momentum.output_column] < -self.momentum_threshold, 'signal'] = -1
        
        return data


class VolatilityAdjustedStrategy(BaseStrategy):
    """
    Volatility-adjusted trading strategy for testing custom indicators.
    
    This strategy uses both momentum and volatility indicators to generate
    buy/sell signals with position sizing based on volatility.
    """
    
    def __init__(self, momentum_period=14, volatility_period=20, momentum_threshold=2.0):
        """
        Initialize the strategy.
        
        Args:
            momentum_period: Period for momentum calculation
            volatility_period: Period for volatility calculation
            momentum_threshold: Threshold for buy/sell signals
        """
        super().__init__()
        self.momentum_period = momentum_period
        self.volatility_period = volatility_period
        self.momentum_threshold = momentum_threshold
        self.momentum = CustomMomentumIndicator(period=momentum_period)
        self.volatility = CustomVolatilityIndicator(period=volatility_period)
    
    def prepare_data(self, data):
        """
        Prepare data for the strategy.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with added indicators
        """
        # Add momentum indicator
        data = self.momentum(data)
        
        # Add volatility indicator
        data = self.volatility(data)
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals with position sizing.
        
        Args:
            data: DataFrame containing price data and indicators
            
        Returns:
            DataFrame with added signal and position_size columns
        """
        # Initialize signal column
        data['signal'] = 0
        
        # Generate buy signals when momentum is above threshold
        data.loc[data[self.momentum.output_column] > self.momentum_threshold, 'signal'] = 1
        
        # Generate sell signals when momentum is below negative threshold
        data.loc[data[self.momentum.output_column] < -self.momentum_threshold, 'signal'] = -1
        
        # Add position sizing based on inverse volatility
        # Higher volatility = smaller position size
        data['position_size'] = 0.0
        
        # Avoid division by zero
        min_volatility = data[self.volatility.output_column].min()
        if min_volatility == 0:
            min_volatility = 0.001
        
        # Calculate position size as inverse of normalized volatility
        # Scale to range [0.1, 1.0]
        volatility_normalized = data[self.volatility.output_column] / data[self.volatility.output_column].max()
        data['position_size'] = 1.0 - (0.9 * volatility_normalized)
        
        # Apply position size only to non-zero signals
        data.loc[data['signal'] == 0, 'position_size'] = 0.0
        
        return data


class MultiIndicatorStrategy(BaseStrategy):
    """
    Multi-indicator trading strategy for testing.
    
    This strategy combines multiple indicators (SMA, RSI, custom momentum)
    to generate trading signals.
    """
    
    def __init__(self, sma_short_period=20, sma_long_period=50, 
                 rsi_period=14, momentum_period=10):
        """
        Initialize the strategy.
        
        Args:
            sma_short_period: Period for short SMA
            sma_long_period: Period for long SMA
            rsi_period: Period for RSI
            momentum_period: Period for momentum
        """
        super().__init__()
        self.sma_short = SMA(period=sma_short_period)
        self.sma_long = SMA(period=sma_long_period)
        self.rsi = RSI(period=rsi_period)
        self.momentum = CustomMomentumIndicator(period=momentum_period)
    
    def prepare_data(self, data):
        """
        Prepare data for the strategy.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with added indicators
        """
        # Add indicators
        data = self.sma_short(data)
        data = self.sma_long(data)
        data = self.rsi(data)
        data = self.momentum(data)
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on multiple indicators.
        
        Args:
            data: DataFrame containing price data and indicators
            
        Returns:
            DataFrame with added signal column
        """
        # Initialize signal column
        data['signal'] = 0
        
        # Generate buy signals when:
        # 1. Short SMA crosses above long SMA
        # 2. RSI is below 70 (not overbought)
        # 3. Momentum is positive
        sma_cross_up = (
            (data[self.sma_short.output_column].shift(1) <= data[self.sma_long.output_column].shift(1)) &
            (data[self.sma_short.output_column] > data[self.sma_long.output_column])
        )
        
        rsi_not_overbought = data[self.rsi.output_column] < 70
        momentum_positive = data[self.momentum.output_column] > 0
        
        buy_condition = sma_cross_up & rsi_not_overbought & momentum_positive
        data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals when:
        # 1. Short SMA crosses below long SMA
        # 2. RSI is above 30 (not oversold)
        # 3. Momentum is negative
        sma_cross_down = (
            (data[self.sma_short.output_column].shift(1) >= data[self.sma_long.output_column].shift(1)) &
            (data[self.sma_short.output_column] < data[self.sma_long.output_column])
        )
        
        rsi_not_oversold = data[self.rsi.output_column] > 30
        momentum_negative = data[self.momentum.output_column] < 0
        
        sell_condition = sma_cross_down & rsi_not_oversold & momentum_negative
        data.loc[sell_condition, 'signal'] = -1
        
        return data


class TestIndicatorIntegration(unittest.TestCase):
    """Test cases for indicator integration with the backtesting framework."""
    
    def setUp(self):
        """Set up test data for integration testing."""
        # Create sample price data
        self.start_date = '2020-01-01'
        self.end_date = '2021-01-01'
        
        # Create a DataLoader instance
        self.data_loader = DataLoader()
        
        # Try to load real data if available, otherwise use synthetic data
        try:
            self.data = self.data_loader.load_data(
                symbol='BTC/USDT',
                timeframe='1d',
                start_date=self.start_date,
                end_date=self.end_date
            )
        except Exception as e:
            print(f"Could not load real data: {e}")
            print("Using synthetic data instead")
            
            # Create synthetic data
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            np.random.seed(42)  # For reproducibility
            
            # Create price series with trend, cycles, and noise
            t = np.linspace(0, 4*np.pi, len(dates))
            trend = np.linspace(5000, 15000, len(dates))
            cycles = 2000 * np.sin(t) + 1000 * np.sin(2*t)
            noise = np.random.normal(0, 500, len(dates))
            
            close_prices = trend + cycles + noise
            
            # Create OHLC data
            self.data = pd.DataFrame({
                'open': close_prices - np.random.normal(0, 100, len(dates)),
                'high': close_prices + np.random.normal(200, 50, len(dates)),
                'low': close_prices - np.random.normal(200, 50, len(dates)),
                'close': close_prices,
                'volume': np.random.normal(1000000, 200000, len(dates))
            }, index=dates)
        
        # Create a BacktestEngine instance
        self.backtest_engine = BacktestEngine()
        
        # Create a PerformanceAnalyzer instance
        self.performance_analyzer = PerformanceAnalyzer()
    
    def test_custom_momentum_indicator(self):
        """Test custom momentum indicator in isolation."""
        # Create the indicator
        momentum = CustomMomentumIndicator(period=14)
        
        # Apply to data
        result = momentum(self.data)
        
        # Check that output column exists
        self.assertIn('Momentum_14', result.columns)
        
        # Check that values are calculated correctly
        expected_momentum = self.data['close'].pct_change(14) * 100
        pd.testing.assert_series_equal(result['Momentum_14'], expected_momentum)
    
    def test_custom_volatility_indicator(self):
        """Test custom volatility indicator in isolation."""
        # Create the indicator
        volatility = CustomVolatilityIndicator(period=20)
        
        # Apply to data
        result = volatility(self.data)
        
        # Check that output column exists
        self.assertIn('Volatility_20', result.columns)
        
        # Check that values are calculated correctly
        returns = self.data['close'].pct_change()
        expected_volatility = returns.rolling(window=20).std() * np.sqrt(252)
        pd.testing.assert_series_equal(result['Volatility_20'], expected_volatility)
    
    def test_momentum_strategy_backtest(self):
        """Test momentum strategy with custom indicator in backtest."""
        # Create the strategy
        strategy = MomentumStrategy(momentum_period=14, momentum_threshold=2.0)
        
        # Run backtest
        results = self.backtest_engine.run(
            data=self.data,
            strategy=strategy,
            initial_capital=10000,
            position_size=0.1
        )
        
        # Check that backtest completes without error
        self.assertIsNotNone(results)
        self.assertIn('equity', results.columns)
        self.assertIn('position', results.columns)
        self.assertIn('signal', results.columns)
        
        # Check that signals are generated
        self.assertTrue((results['signal'] != 0).any())
        
        # Check that trades are executed
        self.assertTrue((results['position'] != 0).any())
        
        # Analyze performance
        performance = self.performance_analyzer.analyze(results)
        
        # Check that performance metrics are calculated
        self.assertIn('total_return', performance)
        self.assertIn('sharpe_ratio', performance)
        self.assertIn('max_drawdown', performance)
        
        # Print performance summary
        print("\nMomentum Strategy Performance")
        print("=============================")
        for metric, value in performance.items():
            print(f"{metric}: {value}")
    
    def test_volatility_adjusted_strategy_backtest(self):
        """Test volatility-adjusted strategy with custom indicators in backtest."""
        # Create the strategy
        strategy = VolatilityAdjustedStrategy(
            momentum_period=14,
            volatility_period=20,
            momentum_threshold=2.0
        )
        
        # Run backtest
        results = self.backtest_engine.run(
            data=self.data,
            strategy=strategy,
            initial_capital=10000,
            position_size=1.0  # Full position size, will be adjusted by strategy
        )
        
        # Check that backtest completes without error
        self.assertIsNotNone(results)
        self.assertIn('equity', results.columns)
        self.assertIn('position', results.columns)
        self.assertIn('signal', results.columns)
        self.assertIn('position_size', results.columns)
        
        # Check that signals are generated
        self.assertTrue((results['signal'] != 0).any())
        
        # Check that trades are executed with varying position sizes
        self.assertTrue((results['position'] != 0).any())
        self.assertTrue((results.loc[results['position'] != 0, 'position_size'] > 0).all())
        self.assertTrue((results.loc[results['position'] != 0, 'position_size'] <= 1).all())
        
        # Analyze performance
        performance = self.performance_analyzer.analyze(results)
        
        # Print performance summary
        print("\nVolatility-Adjusted Strategy Performance")
        print("=======================================")
        for metric, value in performance.items():
            print(f"{metric}: {value}")
    
    def test_multi_indicator_strategy_backtest(self):
        """Test multi-indicator strategy in backtest."""
        # Create the strategy
        strategy = MultiIndicatorStrategy(
            sma_short_period=20,
            sma_long_period=50,
            rsi_period=14,
            momentum_period=10
        )
        
        # Run backtest
        results = self.backtest_engine.run(
            data=self.data,
            strategy=strategy,
            initial_capital=10000,
            position_size=0.1
        )
        
        # Check that backtest completes without error
        self.assertIsNotNone(results)
        self.assertIn('equity', results.columns)
        self.assertIn('position', results.columns)
        self.assertIn('signal', results.columns)
        
        # Check that signals are generated
        self.assertTrue((results['signal'] != 0).any())
        
        # Check that trades are executed
        self.assertTrue((results['position'] != 0).any())
        
        # Analyze performance
        performance = self.performance_analyzer.analyze(results)
        
        # Print performance summary
        print("\nMulti-Indicator Strategy Performance")
        print("===================================")
        for metric, value in performance.items():
            print(f"{metric}: {value}")
    
    def test_compare_strategies(self):
        """Compare performance of different strategies using custom indicators."""
        # Create strategies
        momentum_strategy = MomentumStrategy(momentum_period=14, momentum_threshold=2.0)
        volatility_strategy = VolatilityAdjustedStrategy(
            momentum_period=14,
            volatility_period=20,
            momentum_threshold=2.0
        )
        multi_indicator_strategy = MultiIndicatorStrategy(
            sma_short_period=20,
            sma_long_period=50,
            rsi_period=14,
            momentum_period=10
        )
        
        # Run backtests
        momentum_results = self.backtest_engine.run(
            data=self.data,
            strategy=momentum_strategy,
            initial_capital=10000,
            position_size=0.1
        )
        
        volatility_results = self.backtest_engine.run(
            data=self.data,
            strategy=volatility_strategy,
            initial_capital=10000,
            position_size=1.0
        )
        
        multi_indicator_results = self.backtest_engine.run(
            data=self.data,
            strategy=multi_indicator_strategy,
            initial_capital=10000,
            position_size=0.1
        )
        
        # Analyze performance
        momentum_performance = self.performance_analyzer.analyze(momentum_results)
        volatility_performance = self.performance_analyzer.analyze(volatility_results)
        multi_indicator_performance = self.performance_analyzer.analyze(multi_indicator_results)
        
        # Compare performance
        print("\nStrategy Performance Comparison")
        print("==============================")
        
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        strategies = {
            'Momentum': momentum_performance,
            'Volatility-Adjusted': volatility_performance,
            'Multi-Indicator': multi_indicator_performance
        }
        
        for metric in metrics:
            print(f"\n{metric}:")
            for strategy_name, performance in strategies.items():
                if metric in performance:
                    print(f"{strategy_name}: {performance[metric]}")
        
        # Plot equity curves
        plt.figure(figsize=(12, 6))
        plt.plot(momentum_results.index, momentum_results['equity'], label='Momentum')
        plt.plot(volatility_results.index, volatility_results['equity'], label='Volatility-Adjusted')
        plt.plot(multi_indicator_results.index, multi_indicator_results['equity'], label='Multi-Indicator')
        plt.title('Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'))
        print(f"\nEquity curves comparison saved to {os.path.join(output_dir, 'strategy_comparison.png')}")


if __name__ == '__main__':
    unittest.main()
