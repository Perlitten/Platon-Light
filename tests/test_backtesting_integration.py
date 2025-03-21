#!/usr/bin/env python
"""
Backtesting Integration Tests

This script provides comprehensive tests for validating the integration
of all components in the Platon Light backtesting module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import tempfile
import json

# Add parent directory to path to import Platon Light modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.backtesting.optimization import ParameterOptimizer
from platon_light.core.strategy_factory import StrategyFactory
from platon_light.core.base_strategy import BaseStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_backtesting_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MovingAverageCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover strategy for testing.
    """
    
    def __init__(self, config):
        super().__init__(config)
        strategy_config = config.get('strategy', {})
        self.fast_period = strategy_config.get('fast_period', 20)
        self.slow_period = strategy_config.get('slow_period', 50)
        
    def generate_signals(self, data):
        """Generate trading signals based on moving average crossovers."""
        # Calculate moving averages
        data['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
        data['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
        
        # Initialize signal column
        data['signal'] = 0
        
        # Generate signals
        for i in range(1, len(data)):
            # Buy signal: fast MA crosses above slow MA
            if (data['fast_ma'].iloc[i-1] <= data['slow_ma'].iloc[i-1] and 
                data['fast_ma'].iloc[i] > data['slow_ma'].iloc[i]):
                data.loc[data.index[i], 'signal'] = 1
            
            # Sell signal: fast MA crosses below slow MA
            elif (data['fast_ma'].iloc[i-1] >= data['slow_ma'].iloc[i-1] and 
                  data['fast_ma'].iloc[i] < data['slow_ma'].iloc[i]):
                data.loc[data.index[i], 'signal'] = -1
        
        return data


class RSIStrategy(BaseStrategy):
    """
    RSI Strategy for testing.
    """
    
    def __init__(self, config):
        super().__init__(config)
        strategy_config = config.get('strategy', {})
        self.rsi_period = strategy_config.get('rsi_period', 14)
        self.oversold = strategy_config.get('oversold', 30)
        self.overbought = strategy_config.get('overbought', 70)
        
    def generate_signals(self, data):
        """Generate trading signals based on RSI indicator."""
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize signal column
        data['signal'] = 0
        
        # Generate signals
        for i in range(1, len(data)):
            # Buy signal: RSI crosses below oversold and then back above
            if (data['rsi'].iloc[i-1] <= self.oversold and 
                data['rsi'].iloc[i] > self.oversold):
                data.loc[data.index[i], 'signal'] = 1
            
            # Sell signal: RSI crosses above overbought and then back below
            elif (data['rsi'].iloc[i-1] >= self.overbought and 
                  data['rsi'].iloc[i] < self.overbought):
                data.loc[data.index[i], 'signal'] = -1
        
        return data


# Register strategies
StrategyFactory.register_strategy("moving_average_crossover", MovingAverageCrossover)
StrategyFactory.register_strategy("rsi_strategy", RSIStrategy)


class TestBacktestingIntegration(unittest.TestCase):
    """Test suite for validating the backtesting integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_dir = cls.temp_dir.name
        
        # Create sample data files
        cls._create_sample_data()
        
        # Create basic configuration
        cls.config = {
            'backtesting': {
                'initial_capital': 10000,
                'commission': 0.001,
                'slippage': 0.0005,
                'position_size': 1.0,
                'allow_short': True
            },
            'strategy': {
                'name': 'moving_average_crossover',
                'fast_period': 20,
                'slow_period': 50
            },
            'data': {
                'source': 'csv',
                'directory': cls.data_dir
            }
        }
        
        # Create components
        cls.data_loader = DataLoader(cls.config)
        cls.backtest_engine = BacktestEngine(cls.config)
        cls.performance_analyzer = PerformanceAnalyzer(cls.config)
        cls.visualizer = BacktestVisualizer(cls.config)
        cls.optimizer = ParameterOptimizer(cls.config)
        
        logger.info("Test environment set up successfully")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are done"""
        cls.temp_dir.cleanup()
        logger.info("Test environment cleaned up")
    
    @classmethod
    def _create_sample_data(cls):
        """Create sample data files for testing"""
        # Create directory structure
        os.makedirs(os.path.join(cls.data_dir, 'binance'), exist_ok=True)
        
        # Generate sample OHLCV data
        symbols = ['BTCUSDT', 'ETHUSDT']
        timeframes = ['1h', '4h', '1d']
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 3, 31)
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Determine time delta based on timeframe
                if timeframe == '1h':
                    delta = timedelta(hours=1)
                elif timeframe == '4h':
                    delta = timedelta(hours=4)
                else:  # '1d'
                    delta = timedelta(days=1)
                
                # Generate timestamps
                current_date = start_date
                timestamps = []
                while current_date <= end_date:
                    timestamps.append(current_date)
                    current_date += delta
                
                # Generate OHLCV data
                n = len(timestamps)
                
                # Start with a base price and add random walk
                base_price = 100 if symbol == 'ETHUSDT' else 40000
                price_volatility = base_price * 0.01  # 1% volatility
                
                # Generate price series with random walk
                np.random.seed(42)  # For reproducibility
                random_walk = np.random.normal(0, price_volatility, n).cumsum()
                prices = base_price + random_walk
                
                # Ensure prices are positive
                prices = np.maximum(prices, base_price * 0.1)
                
                # Generate OHLCV data
                opens = prices.copy()
                highs = prices * (1 + np.random.uniform(0, 0.02, n))
                lows = prices * (1 - np.random.uniform(0, 0.02, n))
                closes = prices * (1 + np.random.normal(0, 0.01, n))
                volumes = np.random.uniform(base_price * 10, base_price * 100, n)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'timestamp': [int(t.timestamp() * 1000) for t in timestamps],
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                })
                
                # Save to CSV
                filename = f"{symbol}_{timeframe}.csv"
                filepath = os.path.join(cls.data_dir, 'binance', filename)
                df.to_csv(filepath, index=False)
                
                logger.info(f"Created sample data file: {filepath}")
    
    def test_end_to_end_workflow(self):
        """Test the entire backtesting workflow end-to-end"""
        symbol = 'BTCUSDT'
        timeframe = '1d'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 3, 31)
        
        # 1. Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        self.assertFalse(data.empty)
        logger.info(f"Loaded {len(data)} data points for {symbol} {timeframe}")
        
        # 2. Run backtest
        strategy = StrategyFactory.create_strategy(self.config['strategy']['name'], self.config)
        results = self.backtest_engine.run(strategy, data)
        
        self.assertIsNotNone(results)
        self.assertIn('trades', results)
        self.assertIn('equity_curve', results)
        logger.info(f"Backtest completed with {len(results['trades'])} trades")
        
        # 3. Analyze performance
        metrics = self.performance_analyzer.analyze(results)
        
        self.assertIsNotNone(metrics)
        self.assertIn('return_percent', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown_percent', metrics)
        logger.info(f"Performance analysis completed: Return={metrics['return_percent']:.2f}%, Sharpe={metrics['sharpe_ratio']:.2f}")
        
        # 4. Visualize results
        fig = self.visualizer.plot_equity_curve(results)
        self.assertIsNotNone(fig)
        
        fig = self.visualizer.plot_drawdown_curve(results)
        self.assertIsNotNone(fig)
        
        fig = self.visualizer.plot_monthly_returns(results)
        self.assertIsNotNone(fig)
        
        logger.info("Visualization completed")
        
        # 5. Optimize parameters
        param_grid = {
            'fast_period': [10, 20, 30],
            'slow_period': [40, 50, 60]
        }
        
        optimization_results = self.optimizer.grid_search(
            data=data,
            param_grid=param_grid,
            metric='sharpe_ratio',
            maximize=True
        )
        
        self.assertIsNotNone(optimization_results)
        self.assertIn('best_params', optimization_results)
        self.assertIn('best_metrics', optimization_results)
        
        logger.info(f"Optimization completed: Best params={optimization_results['best_params']}")
        
        # 6. Run optimized strategy
        optimized_config = self.config.copy()
        optimized_config['strategy'].update(optimization_results['best_params'])
        
        optimized_strategy = StrategyFactory.create_strategy(self.config['strategy']['name'], optimized_config)
        optimized_results = self.backtest_engine.run(optimized_strategy, data)
        
        optimized_metrics = self.performance_analyzer.analyze(optimized_results)
        
        logger.info(f"Optimized strategy: Return={optimized_metrics['return_percent']:.2f}%, Sharpe={optimized_metrics['sharpe_ratio']:.2f}")
        
        # 7. Compare original vs optimized
        self.assertGreaterEqual(
            optimized_metrics['sharpe_ratio'],
            metrics['sharpe_ratio'],
            "Optimized strategy should have better or equal Sharpe ratio"
        )
        
        logger.info("End-to-end workflow test passed")
    
    def test_multi_strategy_comparison(self):
        """Test comparing multiple strategies"""
        symbol = 'BTCUSDT'
        timeframe = '1d'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 3, 31)
        
        # Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create strategies
        ma_config = self.config.copy()
        ma_config['strategy'] = {
            'name': 'moving_average_crossover',
            'fast_period': 20,
            'slow_period': 50
        }
        ma_strategy = StrategyFactory.create_strategy(ma_config['strategy']['name'], ma_config)
        
        rsi_config = self.config.copy()
        rsi_config['strategy'] = {
            'name': 'rsi_strategy',
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70
        }
        rsi_strategy = StrategyFactory.create_strategy(rsi_config['strategy']['name'], rsi_config)
        
        # Run backtests
        ma_results = self.backtest_engine.run(ma_strategy, data)
        rsi_results = self.backtest_engine.run(rsi_strategy, data)
        
        # Analyze performance
        ma_metrics = self.performance_analyzer.analyze(ma_results)
        rsi_metrics = self.performance_analyzer.analyze(rsi_results)
        
        logger.info(f"MA Strategy: Return={ma_metrics['return_percent']:.2f}%, Sharpe={ma_metrics['sharpe_ratio']:.2f}")
        logger.info(f"RSI Strategy: Return={rsi_metrics['return_percent']:.2f}%, Sharpe={rsi_metrics['sharpe_ratio']:.2f}")
        
        # Compare strategies
        comparison = self.performance_analyzer.compare_strategies({
            'Moving Average': ma_results,
            'RSI': rsi_results
        })
        
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison), 2)
        
        # Visualize comparison
        fig = self.visualizer.plot_strategy_comparison({
            'Moving Average': ma_results,
            'RSI': rsi_results
        })
        
        self.assertIsNotNone(fig)
        
        logger.info("Multi-strategy comparison test passed")
    
    def test_walk_forward_optimization(self):
        """Test walk-forward optimization"""
        symbol = 'BTCUSDT'
        timeframe = '1d'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 3, 31)
        
        # Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Define parameter grid
        param_grid = {
            'fast_period': [10, 20, 30],
            'slow_period': [40, 50, 60]
        }
        
        # Run walk-forward optimization
        wfo_results = self.optimizer.walk_forward_test(
            data=data,
            param_grid=param_grid,
            train_size=0.7,
            test_size=0.3,
            n_segments=3,
            metric='sharpe_ratio',
            maximize=True
        )
        
        self.assertIsNotNone(wfo_results)
        self.assertIn('segment_params', wfo_results)
        self.assertIn('segment_train_metrics', wfo_results)
        self.assertIn('segment_test_metrics', wfo_results)
        self.assertIn('combined_test_metrics', wfo_results)
        
        logger.info(f"Walk-forward optimization completed")
        logger.info(f"Combined test metrics: Return={wfo_results['combined_test_metrics']['return_percent']:.2f}%, Sharpe={wfo_results['combined_test_metrics']['sharpe_ratio']:.2f}")
        
        # Visualize WFO results
        fig = self.visualizer.plot_wfo_equity_curve(wfo_results)
        self.assertIsNotNone(fig)
        
        fig = self.visualizer.plot_wfo_parameter_stability(wfo_results)
        self.assertIsNotNone(fig)
        
        logger.info("Walk-forward optimization test passed")
    
    def test_multi_timeframe_backtesting(self):
        """Test backtesting with multiple timeframes"""
        symbol = 'BTCUSDT'
        timeframes = ['1h', '4h', '1d']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 3, 31)
        
        # Load data for all timeframes
        data = {}
        for timeframe in timeframes:
            data[timeframe] = self.data_loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        
        # Create multi-timeframe strategy configuration
        mt_config = self.config.copy()
        mt_config['strategy'] = {
            'name': 'moving_average_crossover',
            'fast_period': 20,
            'slow_period': 50,
            'signal_timeframe': '4h',
            'execution_timeframe': '1h'
        }
        
        # Run multi-timeframe backtest
        mt_results = self.backtest_engine.run_multi_timeframe(
            strategy_name=mt_config['strategy']['name'],
            config=mt_config,
            data=data,
            signal_timeframe=mt_config['strategy']['signal_timeframe'],
            execution_timeframe=mt_config['strategy']['execution_timeframe']
        )
        
        self.assertIsNotNone(mt_results)
        self.assertIn('trades', mt_results)
        self.assertIn('equity_curve', mt_results)
        
        # Analyze performance
        mt_metrics = self.performance_analyzer.analyze(mt_results)
        
        logger.info(f"Multi-timeframe backtest completed: Return={mt_metrics['return_percent']:.2f}%, Sharpe={mt_metrics['sharpe_ratio']:.2f}")
        
        # Compare with single timeframe
        st_strategy = StrategyFactory.create_strategy(self.config['strategy']['name'], self.config)
        st_results = self.backtest_engine.run(st_strategy, data['1d'])
        st_metrics = self.performance_analyzer.analyze(st_results)
        
        logger.info(f"Single timeframe backtest: Return={st_metrics['return_percent']:.2f}%, Sharpe={st_metrics['sharpe_ratio']:.2f}")
        
        # Visualize comparison
        fig = self.visualizer.plot_strategy_comparison({
            'Multi-Timeframe': mt_results,
            'Single-Timeframe': st_results
        })
        
        self.assertIsNotNone(fig)
        
        logger.info("Multi-timeframe backtesting test passed")
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        symbol = 'BTCUSDT'
        timeframe = '1d'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 3, 31)
        
        # Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Run backtest
        strategy = StrategyFactory.create_strategy(self.config['strategy']['name'], self.config)
        results = self.backtest_engine.run(strategy, data)
        
        # Run Monte Carlo simulation
        mc_results = self.performance_analyzer.monte_carlo_simulation(
            results=results,
            num_simulations=100,
            method='trade_resample'
        )
        
        self.assertIsNotNone(mc_results)
        self.assertIn('simulated_equity_curves', mc_results)
        self.assertIn('var_95', mc_results)
        self.assertIn('var_99', mc_results)
        self.assertIn('expected_return', mc_results)
        
        logger.info(f"Monte Carlo simulation completed")
        logger.info(f"Expected Return: {mc_results['expected_return']:.2f}%")
        logger.info(f"VaR 95%: {mc_results['var_95']:.2f}%")
        
        # Visualize Monte Carlo results
        fig = self.visualizer.plot_monte_carlo_simulations(mc_results)
        self.assertIsNotNone(fig)
        
        logger.info("Monte Carlo simulation test passed")
    
    def test_portfolio_backtesting(self):
        """Test backtesting with a portfolio of assets"""
        symbols = ['BTCUSDT', 'ETHUSDT']
        timeframe = '1d'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 3, 31)
        
        # Load data for all symbols
        data = {}
        for symbol in symbols:
            data[symbol] = self.data_loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        
        # Create portfolio configuration
        portfolio_config = self.config.copy()
        portfolio_config['backtesting']['portfolio'] = {
            'allocation': {
                'BTCUSDT': 0.6,
                'ETHUSDT': 0.4
            },
            'rebalance_frequency': 'monthly'
        }
        
        # Run portfolio backtest
        portfolio_results = self.backtest_engine.run_portfolio(
            strategy_name=portfolio_config['strategy']['name'],
            config=portfolio_config,
            data=data
        )
        
        self.assertIsNotNone(portfolio_results)
        self.assertIn('trades', portfolio_results)
        self.assertIn('equity_curve', portfolio_results)
        self.assertIn('asset_allocation', portfolio_results)
        
        # Analyze performance
        portfolio_metrics = self.performance_analyzer.analyze(portfolio_results)
        
        logger.info(f"Portfolio backtest completed: Return={portfolio_metrics['return_percent']:.2f}%, Sharpe={portfolio_metrics['sharpe_ratio']:.2f}")
        
        # Compare with single asset
        single_strategy = StrategyFactory.create_strategy(self.config['strategy']['name'], self.config)
        btc_results = self.backtest_engine.run(single_strategy, data['BTCUSDT'])
        btc_metrics = self.performance_analyzer.analyze(btc_results)
        
        logger.info(f"BTC only backtest: Return={btc_metrics['return_percent']:.2f}%, Sharpe={btc_metrics['sharpe_ratio']:.2f}")
        
        # Visualize comparison
        fig = self.visualizer.plot_strategy_comparison({
            'Portfolio': portfolio_results,
            'BTC Only': btc_results
        })
        
        self.assertIsNotNone(fig)
        
        # Visualize asset allocation
        fig = self.visualizer.plot_portfolio_allocation(portfolio_results)
        self.assertIsNotNone(fig)
        
        logger.info("Portfolio backtesting test passed")


if __name__ == '__main__':
    unittest.main()
