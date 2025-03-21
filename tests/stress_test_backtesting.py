#!/usr/bin/env python
"""
Stress Test Suite for Platon Light Backtesting Module

This script performs stress testing on the backtesting module to ensure it can handle:
1. Large datasets
2. Edge cases
3. Extreme market conditions
4. Resource constraints
5. Error conditions
"""

import unittest
import sys
import os
import time
import gc
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import psutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path to import Platon Light modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.backtesting.optimization import StrategyOptimizer
from platon_light.core.strategy_factory import StrategyFactory
from platon_light.core.base_strategy import BaseStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stress_test_backtesting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Define test strategies
class SimpleStrategy(BaseStrategy):
    """Simple strategy for testing"""
    def __init__(self, config):
        super().__init__(config)
        self.sma_period = config.get('strategy', {}).get('sma_period', 20)
    
    def generate_signals(self, data):
        """Generate simple SMA crossover signals"""
        data['sma'] = data['close'].rolling(window=self.sma_period).mean()
        data['signal'] = 0
        data.loc[data['close'] > data['sma'], 'signal'] = 1
        data.loc[data['close'] < data['sma'], 'signal'] = -1
        return data


class ErrorStrategy(BaseStrategy):
    """Strategy that deliberately raises errors for testing"""
    def __init__(self, config):
        super().__init__(config)
        self.error_type = config.get('strategy', {}).get('error_type', 'random')
        self.error_frequency = config.get('strategy', {}).get('error_frequency', 0.1)
    
    def generate_signals(self, data):
        """Generate signals with deliberate errors"""
        # Random errors
        if self.error_type == 'random' and np.random.random() < self.error_frequency:
            raise RuntimeError("Deliberate random error in strategy")
        
        # Division by zero error
        if self.error_type == 'division_by_zero':
            # Create a column with some zeros
            data['divisor'] = data['close'] - data['close'].shift(1)
            data['bad_indicator'] = data['close'] / data['divisor']
        
        # Memory error simulation
        if self.error_type == 'memory':
            # Create a large array to consume memory
            large_array = np.ones((10000, 10000)) * data['close'].mean()
            data['memory_hog'] = large_array.mean()
        
        # Normal operation
        data['signal'] = 0
        data.loc[data.index % 20 == 0, 'signal'] = 1
        data.loc[data.index % 20 == 10, 'signal'] = -1
        
        return data


class HighFrequencyStrategy(BaseStrategy):
    """Strategy that generates many trades for stress testing"""
    def generate_signals(self, data):
        """Generate high-frequency signals"""
        data['signal'] = 0
        # Alternate buy/sell signals frequently
        data.loc[data.index % 2 == 0, 'signal'] = 1
        data.loc[data.index % 2 == 1, 'signal'] = -1
        return data


class ResourceIntensiveStrategy(BaseStrategy):
    """Strategy that consumes significant computational resources"""
    def __init__(self, config):
        super().__init__(config)
        self.complexity = config.get('strategy', {}).get('complexity', 5)
    
    def generate_signals(self, data):
        """Generate signals with resource-intensive calculations"""
        # Create multiple rolling windows and calculations
        for i in range(1, self.complexity + 1):
            period = i * 10
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'std_{period}'] = data['close'].rolling(window=period).std()
            data[f'upper_{period}'] = data[f'sma_{period}'] + data[f'std_{period}'] * 2
            data[f'lower_{period}'] = data[f'sma_{period}'] - data[f'std_{period}'] * 2
        
        # Perform computationally expensive operations
        for i in range(1, self.complexity + 1):
            for j in range(1, self.complexity + 1):
                if i != j:
                    data[f'cross_{i}_{j}'] = data[f'sma_{i*10}'] - data[f'sma_{j*10}']
        
        # Generate signals based on multiple conditions
        data['signal'] = 0
        
        buy_condition = True
        sell_condition = True
        
        for i in range(1, self.complexity + 1):
            buy_condition = buy_condition & (data[f'sma_{i*10}'] > data[f'sma_{i*10}'].shift(1))
            sell_condition = sell_condition & (data[f'sma_{i*10}'] < data[f'sma_{i*10}'].shift(1))
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data


# Register test strategies
StrategyFactory.register_strategy("simple", SimpleStrategy)
StrategyFactory.register_strategy("error", ErrorStrategy)
StrategyFactory.register_strategy("high_frequency", HighFrequencyStrategy)
StrategyFactory.register_strategy("resource_intensive", ResourceIntensiveStrategy)


class StressTestBacktesting(unittest.TestCase):
    """Stress test suite for backtesting module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logger.info("Setting up stress test environment")
        
        # Base configuration
        cls.base_config = {
            'backtesting': {
                'initial_capital': 10000,
                'commission': 0.001,
                'slippage': 0.0005,
                'position_size': 1.0
            },
            'logging': {
                'level': 'INFO',
                'console_output': True,
                'file_output': True
            }
        }
        
        # Create test data directory if it doesn't exist
        os.makedirs('test_data', exist_ok=True)
        
        # Generate synthetic data for testing
        cls.generate_test_data()
    
    @classmethod
    def generate_test_data(cls):
        """Generate synthetic data for testing"""
        logger.info("Generating synthetic test data")
        
        # Generate normal market data
        cls.generate_market_data('normal', 1000)
        
        # Generate large dataset
        cls.generate_market_data('large', 100000)
        
        # Generate extreme market data (high volatility)
        cls.generate_market_data('volatile', 1000, volatility=0.05)
        
        # Generate extreme market data (trending)
        cls.generate_market_data('trending', 1000, trend=0.001)
        
        # Generate extreme market data (gaps)
        cls.generate_market_data('gaps', 1000, gap_probability=0.05, gap_size=0.05)
        
        # Generate extreme market data (flat)
        cls.generate_market_data('flat', 1000, volatility=0.0001)
    
    @classmethod
    def generate_market_data(cls, name, size, volatility=0.01, trend=0.0, gap_probability=0.0, gap_size=0.0):
        """Generate synthetic market data with specified characteristics"""
        logger.info(f"Generating {name} market data with {size} data points")
        
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=size)
        timestamps = pd.date_range(start=start_date, end=end_date, periods=size)
        
        # Generate price data
        price = 100.0
        prices = []
        
        for i in range(size):
            # Apply trend
            price *= (1 + trend)
            
            # Apply random price movement
            price *= (1 + np.random.normal(0, volatility))
            
            # Apply gaps
            if np.random.random() < gap_probability:
                gap_direction = 1 if np.random.random() > 0.5 else -1
                price *= (1 + gap_direction * gap_size)
            
            prices.append(price)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': np.random.randint(100, 10000, size=size)
        })
        
        # Adjust high and low prices
        for i in range(size):
            data.loc[i, 'high'] = data.loc[i, 'open'] * (1 + np.random.random() * volatility)
            data.loc[i, 'low'] = data.loc[i, 'open'] * (1 - np.random.random() * volatility)
        
        # Save to CSV
        data.to_csv(f'test_data/{name}_data.csv', index=False)
        
        return data
    
    def setUp(self):
        """Set up before each test"""
        # Clear memory
        gc.collect()
    
    def test_large_dataset(self):
        """Test backtesting with a large dataset"""
        logger.info("Running large dataset test")
        
        # Configure test
        config = self.base_config.copy()
        config['strategy'] = {'name': 'simple', 'sma_period': 20}
        
        # Load large dataset
        data = pd.read_csv('test_data/large_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Measure memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)
        
        # Create backtest engine
        backtest_engine = BacktestEngine(config)
        
        # Time the backtest
        start_time = time.time()
        
        # Run backtest with large dataset
        results = backtest_engine.run_with_data(data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Measure memory usage after
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_increase = memory_after - memory_before
        
        logger.info(f"Large dataset test completed in {execution_time:.2f} seconds")
        logger.info(f"Memory usage increased by {memory_increase:.2f} MB")
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)
        
        # Performance assertions
        self.assertLess(execution_time, 60, "Backtest took too long to execute")
        self.assertLess(memory_increase, 1000, "Memory usage increased too much")
    
    def test_extreme_market_conditions(self):
        """Test backtesting under extreme market conditions"""
        logger.info("Running extreme market conditions test")
        
        # Market conditions to test
        conditions = ['volatile', 'trending', 'gaps', 'flat']
        
        for condition in conditions:
            logger.info(f"Testing {condition} market condition")
            
            # Configure test
            config = self.base_config.copy()
            config['strategy'] = {'name': 'simple', 'sma_period': 20}
            
            # Load dataset
            data = pd.read_csv(f'test_data/{condition}_data.csv')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Create backtest engine
            backtest_engine = BacktestEngine(config)
            
            # Run backtest
            results = backtest_engine.run_with_data(data)
            
            # Verify results
            self.assertIsNotNone(results)
            self.assertIn('equity_curve', results)
            self.assertIn('trades', results)
            self.assertIn('metrics', results)
            
            # Analyze performance metrics
            metrics = results['metrics']
            logger.info(f"{condition} market results: Return: {metrics['return_percent']:.2f}%, "
                       f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                       f"Max DD: {metrics['max_drawdown_percent']:.2f}%")
    
    def test_high_frequency_trading(self):
        """Test backtesting with high-frequency trading strategy"""
        logger.info("Running high-frequency trading test")
        
        # Configure test
        config = self.base_config.copy()
        config['strategy'] = {'name': 'high_frequency'}
        
        # Load dataset
        data = pd.read_csv('test_data/normal_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Create backtest engine
        backtest_engine = BacktestEngine(config)
        
        # Time the backtest
        start_time = time.time()
        
        # Run backtest
        results = backtest_engine.run_with_data(data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f"High-frequency test completed in {execution_time:.2f} seconds")
        
        # Verify results
        self.assertIsNotNone(results)
        
        # Check number of trades
        self.assertGreater(len(results['trades']), 400, "Not enough trades generated")
        
        # Performance assertions
        self.assertLess(execution_time, 10, "High-frequency backtest took too long to execute")
    
    def test_resource_intensive_strategy(self):
        """Test backtesting with resource-intensive strategy"""
        logger.info("Running resource-intensive strategy test")
        
        # Configure test with increasing complexity
        for complexity in [1, 3, 5]:
            logger.info(f"Testing complexity level {complexity}")
            
            config = self.base_config.copy()
            config['strategy'] = {'name': 'resource_intensive', 'complexity': complexity}
            
            # Load dataset
            data = pd.read_csv('test_data/normal_data.csv')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Measure resource usage
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)
            cpu_percent_before = process.cpu_percent()
            
            # Create backtest engine
            backtest_engine = BacktestEngine(config)
            
            # Time the backtest
            start_time = time.time()
            
            # Run backtest
            results = backtest_engine.run_with_data(data)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Measure resource usage after
            memory_after = process.memory_info().rss / (1024 * 1024)
            cpu_percent_after = process.cpu_percent()
            
            memory_increase = memory_after - memory_before
            cpu_increase = cpu_percent_after - cpu_percent_before
            
            logger.info(f"Complexity {complexity} test completed in {execution_time:.2f} seconds")
            logger.info(f"Memory usage increased by {memory_increase:.2f} MB")
            logger.info(f"CPU usage increased by {cpu_increase:.2f}%")
            
            # Verify results
            self.assertIsNotNone(results)
            
            # Log performance metrics
            metrics = results['metrics']
            logger.info(f"Complexity {complexity} results: Return: {metrics['return_percent']:.2f}%, "
                       f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    def test_error_handling(self):
        """Test error handling in backtesting module"""
        logger.info("Running error handling test")
        
        error_types = ['random', 'division_by_zero', 'memory']
        
        for error_type in error_types:
            logger.info(f"Testing error type: {error_type}")
            
            # Configure test
            config = self.base_config.copy()
            config['strategy'] = {'name': 'error', 'error_type': error_type, 'error_frequency': 0.2}
            
            # Load dataset
            data = pd.read_csv('test_data/normal_data.csv')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Create backtest engine
            backtest_engine = BacktestEngine(config)
            
            # Run backtest and check if it handles errors
            try:
                results = backtest_engine.run_with_data(data)
                
                # If we got here, the backtest engine handled the error
                logger.info(f"Error type {error_type} was handled successfully")
                
                # Verify results still exist
                self.assertIsNotNone(results)
                self.assertIn('equity_curve', results)
                
            except Exception as e:
                logger.error(f"Error type {error_type} was not handled: {str(e)}")
                
                # Fail the test if error_type is 'random' (should be handled)
                # For other error types, it's acceptable if they're not handled
                if error_type == 'random':
                    self.fail(f"Random error should have been handled but raised: {str(e)}")
    
    def test_optimization_stress(self):
        """Test strategy optimization under stress"""
        logger.info("Running optimization stress test")
        
        # Configure test
        config = self.base_config.copy()
        config['strategy'] = {'name': 'simple'}
        
        # Load dataset
        data = pd.read_csv('test_data/normal_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Define parameter grid
        param_grid = {
            'sma_period': [5, 10, 20, 50, 100]
        }
        
        # Create optimizer
        optimizer = StrategyOptimizer(config)
        
        # Time the optimization
        start_time = time.time()
        
        # Run optimization
        optimization_results = optimizer.grid_search(
            data=data,
            param_grid=param_grid,
            metric='sharpe_ratio'
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {execution_time:.2f} seconds")
        
        # Verify results
        self.assertIsNotNone(optimization_results)
        self.assertGreaterEqual(len(optimization_results), len(param_grid['sma_period']))
        
        # Check best parameters
        best_params = optimization_results[0]['parameters']
        best_metric = optimization_results[0]['metrics']['sharpe_ratio']
        
        logger.info(f"Best parameters: {best_params}, Sharpe ratio: {best_metric:.2f}")
        
        # Performance assertions
        self.assertLess(execution_time, 30, "Optimization took too long to execute")
    
    def test_visualization_stress(self):
        """Test visualization module under stress"""
        logger.info("Running visualization stress test")
        
        # Configure test
        config = self.base_config.copy()
        config['strategy'] = {'name': 'simple', 'sma_period': 20}
        
        # Load dataset
        data = pd.read_csv('test_data/normal_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Create backtest engine and run backtest
        backtest_engine = BacktestEngine(config)
        results = backtest_engine.run_with_data(data)
        
        # Create visualizer
        visualizer = BacktestVisualizer(config)
        
        # Time the visualization
        start_time = time.time()
        
        # Create visualizations
        equity_curve_fig = visualizer.plot_equity_curve(results)
        drawdown_fig = visualizer.plot_drawdown(results)
        returns_fig = visualizer.plot_returns_distribution(results)
        
        # Save figures to test output
        equity_curve_fig.savefig('test_data/equity_curve.png')
        drawdown_fig.savefig('test_data/drawdown.png')
        returns_fig.savefig('test_data/returns.png')
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f"Visualization completed in {execution_time:.2f} seconds")
        
        # Verify results
        self.assertTrue(os.path.exists('test_data/equity_curve.png'))
        self.assertTrue(os.path.exists('test_data/drawdown.png'))
        self.assertTrue(os.path.exists('test_data/returns.png'))
        
        # Performance assertions
        self.assertLess(execution_time, 10, "Visualization took too long to execute")
    
    def test_multiple_symbols(self):
        """Test backtesting with multiple symbols simultaneously"""
        logger.info("Running multiple symbols test")
        
        # Generate data for multiple symbols
        symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'ADA']
        symbol_data = {}
        
        for symbol in symbols:
            # Generate slightly different data for each symbol
            data = pd.read_csv('test_data/normal_data.csv')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Apply random factor to prices
            factor = 0.8 + np.random.random() * 0.4  # 0.8 to 1.2
            for col in ['open', 'high', 'low', 'close']:
                data[col] = data[col] * factor
            
            symbol_data[symbol] = data
        
        # Configure test
        config = self.base_config.copy()
        config['strategy'] = {'name': 'simple', 'sma_period': 20}
        
        # Create backtest engine
        backtest_engine = BacktestEngine(config)
        
        # Time the backtest
        start_time = time.time()
        
        # Run backtest for each symbol
        all_results = {}
        for symbol, data in symbol_data.items():
            results = backtest_engine.run_with_data(data, symbol=symbol)
            all_results[symbol] = results
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f"Multiple symbols test completed in {execution_time:.2f} seconds")
        
        # Verify results
        for symbol, results in all_results.items():
            self.assertIsNotNone(results)
            self.assertIn('equity_curve', results)
            self.assertIn('trades', results)
            self.assertIn('metrics', results)
            
            logger.info(f"{symbol} results: Return: {results['metrics']['return_percent']:.2f}%, "
                       f"Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
        
        # Performance assertions
        self.assertLess(execution_time, 60, "Multiple symbols backtest took too long to execute")
    
    def test_parallel_backtesting(self):
        """Test parallel backtesting performance"""
        logger.info("Running parallel backtesting test")
        
        try:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            
            # Configure test
            config = self.base_config.copy()
            config['strategy'] = {'name': 'simple', 'sma_period': 20}
            
            # Load dataset
            data = pd.read_csv('test_data/normal_data.csv')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Define parameter variations
            variations = [
                {'sma_period': 10},
                {'sma_period': 20},
                {'sma_period': 50},
                {'sma_period': 100},
                {'sma_period': 200}
            ]
            
            # Function to run a single backtest
            def run_single_backtest(params):
                test_config = config.copy()
                test_config['strategy'].update(params)
                
                backtest_engine = BacktestEngine(test_config)
                results = backtest_engine.run_with_data(data)
                
                return {
                    'parameters': params,
                    'metrics': results['metrics']
                }
            
            # Time sequential execution
            start_time = time.time()
            
            sequential_results = []
            for params in variations:
                result = run_single_backtest(params)
                sequential_results.append(result)
            
            sequential_time = time.time() - start_time
            
            # Time parallel execution
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                parallel_results = list(executor.map(run_single_backtest, variations))
            
            parallel_time = time.time() - start_time
            
            logger.info(f"Sequential execution time: {sequential_time:.2f} seconds")
            logger.info(f"Parallel execution time: {parallel_time:.2f} seconds")
            logger.info(f"Speedup factor: {sequential_time / parallel_time:.2f}x")
            
            # Verify results
            self.assertEqual(len(sequential_results), len(parallel_results))
            
            # Performance assertions
            if multiprocessing.cpu_count() > 1:
                self.assertLess(parallel_time, sequential_time, 
                               "Parallel execution should be faster than sequential")
        
        except ImportError:
            logger.warning("multiprocessing or concurrent.futures not available, skipping parallel test")
            self.skipTest("multiprocessing or concurrent.futures not available")
    
    def tearDown(self):
        """Clean up after each test"""
        # Force garbage collection
        gc.collect()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        logger.info("Stress tests completed")


if __name__ == '__main__':
    unittest.main()
