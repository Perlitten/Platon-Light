import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.core.base_strategy import BaseStrategy
from platon_light.core.strategy_factory import StrategyFactory


class TestCustomStrategy(BaseStrategy):
    """Test strategy implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        # Initialize strategy parameters
        self.rsi_period = config.get('strategy', {}).get('rsi_period', 14)
        self.rsi_overbought = config.get('strategy', {}).get('rsi_overbought', 70)
        self.rsi_oversold = config.get('strategy', {}).get('rsi_oversold', 30)
    
    def generate_signals(self, data):
        """Generate trading signals based on RSI"""
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['rsi'] < self.rsi_oversold, 'signal'] = 1  # Buy signal
        data.loc[data['rsi'] > self.rsi_overbought, 'signal'] = -1  # Sell signal
        
        return data


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy for testing"""
    
    def __init__(self, config):
        super().__init__(config)
        # Initialize strategy parameters
        self.fast_period = config.get('strategy', {}).get('fast_period', 10)
        self.slow_period = config.get('strategy', {}).get('slow_period', 30)
    
    def generate_signals(self, data):
        """Generate trading signals based on MA crossover"""
        # Calculate moving averages
        data['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
        data['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        # Buy signal when fast MA crosses above slow MA
        data.loc[(data['fast_ma'] > data['slow_ma']) & 
                (data['fast_ma'].shift(1) <= data['slow_ma'].shift(1)), 'signal'] = 1
        # Sell signal when fast MA crosses below slow MA
        data.loc[(data['fast_ma'] < data['slow_ma']) & 
                (data['fast_ma'].shift(1) >= data['slow_ma'].shift(1)), 'signal'] = -1
        
        return data


class TestStrategyImplementation(unittest.TestCase):
    """Test cases for custom strategy implementation"""

    def setUp(self):
        """Set up test fixtures"""
        # Register test strategies
        StrategyFactory.register_strategy("test_rsi", TestCustomStrategy)
        StrategyFactory.register_strategy("ma_crossover", MACrossoverStrategy)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H')
        self.sample_data = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'datetime': dates,
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(100, 5, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        })
        
        # Add some trend to the close prices for better testing
        trend = np.linspace(0, 20, len(dates))
        self.sample_data['close'] = self.sample_data['close'] + trend
        
        # Create a mock data loader that returns our sample data
        self.mock_data_loader = type('MockDataLoader', (), {
            'load_data': lambda self, symbol, timeframe, start_date, end_date: self.sample_data
        })()
        self.mock_data_loader.sample_data = self.sample_data

    def test_strategy_registration(self):
        """Test that strategies can be registered and retrieved"""
        # Check that our test strategies are registered
        available_strategies = StrategyFactory.get_available_strategies()
        self.assertIn("test_rsi", available_strategies)
        self.assertIn("ma_crossover", available_strategies)
        
        # Create strategy instances
        config = {'strategy': {'rsi_period': 14, 'fast_period': 10, 'slow_period': 30}}
        rsi_strategy = StrategyFactory.create_strategy("test_rsi", config)
        ma_strategy = StrategyFactory.create_strategy("ma_crossover", config)
        
        # Check that instances are of the correct type
        self.assertIsInstance(rsi_strategy, TestCustomStrategy)
        self.assertIsInstance(ma_strategy, MACrossoverStrategy)

    def test_rsi_strategy_signals(self):
        """Test that RSI strategy generates correct signals"""
        # Configure RSI strategy
        config = {
            'strategy': {
                'name': 'test_rsi',
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            }
        }
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(config)
        backtest_engine.data_loader = self.mock_data_loader
        
        # Run backtest
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        results = backtest_engine.run(symbol, timeframe, start_date, end_date)
        
        # Check that signals were generated
        self.assertIn('data', results)
        data = results['data']
        self.assertIn('signal', data.columns)
        self.assertIn('rsi', data.columns)
        
        # Verify signal logic
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        
        # All buy signals should have RSI < oversold threshold
        if not buy_signals.empty:
            self.assertTrue(all(buy_signals['rsi'] < config['strategy']['rsi_oversold']))
        
        # All sell signals should have RSI > overbought threshold
        if not sell_signals.empty:
            self.assertTrue(all(sell_signals['rsi'] > config['strategy']['rsi_overbought']))

    def test_ma_crossover_strategy_signals(self):
        """Test that MA Crossover strategy generates correct signals"""
        # Configure MA Crossover strategy
        config = {
            'strategy': {
                'name': 'ma_crossover',
                'fast_period': 10,
                'slow_period': 30
            }
        }
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(config)
        backtest_engine.data_loader = self.mock_data_loader
        
        # Run backtest
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        results = backtest_engine.run(symbol, timeframe, start_date, end_date)
        
        # Check that signals were generated
        self.assertIn('data', results)
        data = results['data']
        self.assertIn('signal', data.columns)
        self.assertIn('fast_ma', data.columns)
        self.assertIn('slow_ma', data.columns)
        
        # Verify signal logic
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        
        # Check a few buy signals to verify the crossover logic
        for idx in buy_signals.index:
            if idx > 0:  # Skip first row
                # Current: fast_ma > slow_ma
                self.assertGreater(data.loc[idx, 'fast_ma'], data.loc[idx, 'slow_ma'])
                # Previous: fast_ma <= slow_ma
                self.assertLessEqual(data.loc[idx-1, 'fast_ma'], data.loc[idx-1, 'slow_ma'])
        
        # Check a few sell signals to verify the crossover logic
        for idx in sell_signals.index:
            if idx > 0:  # Skip first row
                # Current: fast_ma < slow_ma
                self.assertLess(data.loc[idx, 'fast_ma'], data.loc[idx, 'slow_ma'])
                # Previous: fast_ma >= slow_ma
                self.assertGreaterEqual(data.loc[idx-1, 'fast_ma'], data.loc[idx-1, 'slow_ma'])

    def test_strategy_parameter_setting(self):
        """Test that strategy parameters can be set and updated"""
        # Initial configuration
        config = {
            'strategy': {
                'name': 'test_rsi',
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            }
        }
        
        # Create strategy
        strategy = StrategyFactory.create_strategy("test_rsi", config)
        
        # Check initial parameters
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.rsi_overbought, 70)
        self.assertEqual(strategy.rsi_oversold, 30)
        
        # Update parameters
        new_params = {
            'rsi_period': 21,
            'rsi_overbought': 80,
            'rsi_oversold': 20
        }
        strategy.set_parameters(new_params)
        
        # Check updated parameters
        self.assertEqual(strategy.rsi_period, 21)
        self.assertEqual(strategy.rsi_overbought, 80)
        self.assertEqual(strategy.rsi_oversold, 20)

    def test_multiple_strategies_comparison(self):
        """Test comparing multiple strategies"""
        # Define configurations for different strategies
        configs = [
            {
                'strategy': {
                    'name': 'test_rsi',
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                }
            },
            {
                'strategy': {
                    'name': 'ma_crossover',
                    'fast_period': 10,
                    'slow_period': 30
                }
            }
        ]
        
        results = []
        
        # Run backtests for each strategy
        for config in configs:
            backtest_engine = BacktestEngine(config)
            backtest_engine.data_loader = self.mock_data_loader
            
            result = backtest_engine.run(
                symbol='BTCUSDT',
                timeframe='1h',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31)
            )
            
            results.append({
                'strategy_name': config['strategy']['name'],
                'metrics': result['metrics']
            })
        
        # Check that we have results for both strategies
        self.assertEqual(len(results), 2)
        
        # Verify that the strategy names are correct
        strategy_names = [r['strategy_name'] for r in results]
        self.assertIn('test_rsi', strategy_names)
        self.assertIn('ma_crossover', strategy_names)
        
        # Check that metrics were calculated for each strategy
        for result in results:
            self.assertIn('return_percent', result['metrics'])
            self.assertIn('sharpe_ratio', result['metrics'])
            self.assertIn('max_drawdown_percent', result['metrics'])


class TestStrategyExtensibility(unittest.TestCase):
    """Test cases for strategy extensibility features"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H')
        self.sample_data = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'datetime': dates,
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(100, 5, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        })

    def test_strategy_inheritance(self):
        """Test that strategies can inherit from each other"""
        # Define a base strategy
        class BaseTestStrategy(BaseStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.base_param = config.get('strategy', {}).get('base_param', 10)
            
            def calculate_base_indicator(self, data):
                data['base_indicator'] = data['close'].rolling(window=self.base_param).mean()
                return data
            
            def generate_signals(self, data):
                data = self.calculate_base_indicator(data)
                data['signal'] = 0
                return data
        
        # Define a derived strategy
        class DerivedStrategy(BaseTestStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.derived_param = config.get('strategy', {}).get('derived_param', 20)
            
            def calculate_derived_indicator(self, data):
                data['derived_indicator'] = data['close'].rolling(window=self.derived_param).std()
                return data
            
            def generate_signals(self, data):
                # Call base class method to calculate base indicator
                data = super().calculate_base_indicator(data)
                # Calculate derived indicator
                data = self.calculate_derived_indicator(data)
                
                # Generate signals based on both indicators
                data['signal'] = 0
                data.loc[data['base_indicator'] > data['derived_indicator'], 'signal'] = 1
                data.loc[data['base_indicator'] < data['derived_indicator'], 'signal'] = -1
                
                return data
        
        # Register strategies
        StrategyFactory.register_strategy("base_test", BaseTestStrategy)
        StrategyFactory.register_strategy("derived_test", DerivedStrategy)
        
        # Create instances
        config = {
            'strategy': {
                'base_param': 15,
                'derived_param': 25
            }
        }
        
        base_strategy = StrategyFactory.create_strategy("base_test", config)
        derived_strategy = StrategyFactory.create_strategy("derived_test", config)
        
        # Check parameter inheritance
        self.assertEqual(base_strategy.base_param, 15)
        self.assertEqual(derived_strategy.base_param, 15)
        self.assertEqual(derived_strategy.derived_param, 25)
        
        # Test signal generation
        base_data = base_strategy.generate_signals(self.sample_data.copy())
        derived_data = derived_strategy.generate_signals(self.sample_data.copy())
        
        # Check that base indicator is calculated in both strategies
        self.assertIn('base_indicator', base_data.columns)
        self.assertIn('base_indicator', derived_data.columns)
        
        # Check that derived indicator is only in derived strategy
        self.assertNotIn('derived_indicator', base_data.columns)
        self.assertIn('derived_indicator', derived_data.columns)
        
        # Check signal generation logic
        self.assertTrue(all(base_data['signal'] == 0))  # Base strategy doesn't generate real signals
        self.assertTrue(any(derived_data['signal'] != 0))  # Derived strategy should generate some signals

    def test_strategy_composition(self):
        """Test that strategies can be composed together"""
        # Define component strategies
        class RSIComponent(BaseStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.rsi_period = config.get('strategy', {}).get('rsi_period', 14)
            
            def calculate_rsi(self, data):
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=self.rsi_period).mean()
                avg_loss = loss.rolling(window=self.rsi_period).mean()
                
                rs = avg_gain / avg_loss
                data['rsi'] = 100 - (100 / (1 + rs))
                return data
        
        class MAComponent(BaseStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.fast_period = config.get('strategy', {}).get('fast_period', 10)
                self.slow_period = config.get('strategy', {}).get('slow_period', 30)
            
            def calculate_ma(self, data):
                data['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
                data['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
                return data
        
        # Define composite strategy
        class CompositeStrategy(BaseStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.rsi_component = RSIComponent(config)
                self.ma_component = MAComponent(config)
                
                self.rsi_threshold = config.get('strategy', {}).get('rsi_threshold', 50)
            
            def generate_signals(self, data):
                # Calculate indicators using components
                data = self.rsi_component.calculate_rsi(data)
                data = self.ma_component.calculate_ma(data)
                
                # Generate signals based on both components
                data['signal'] = 0
                
                # Buy when fast MA > slow MA AND RSI < threshold
                buy_condition = (data['fast_ma'] > data['slow_ma']) & (data['rsi'] < self.rsi_threshold)
                data.loc[buy_condition, 'signal'] = 1
                
                # Sell when fast MA < slow MA AND RSI > threshold
                sell_condition = (data['fast_ma'] < data['slow_ma']) & (data['rsi'] > self.rsi_threshold)
                data.loc[sell_condition, 'signal'] = -1
                
                return data
        
        # Register strategy
        StrategyFactory.register_strategy("composite_test", CompositeStrategy)
        
        # Create instance
        config = {
            'strategy': {
                'rsi_period': 14,
                'fast_period': 10,
                'slow_period': 30,
                'rsi_threshold': 50
            }
        }
        
        composite_strategy = StrategyFactory.create_strategy("composite_test", config)
        
        # Test signal generation
        data = composite_strategy.generate_signals(self.sample_data.copy())
        
        # Check that all indicators are calculated
        self.assertIn('rsi', data.columns)
        self.assertIn('fast_ma', data.columns)
        self.assertIn('slow_ma', data.columns)
        
        # Check signal generation logic
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        
        # Check a few buy signals
        for _, row in buy_signals.iterrows():
            self.assertGreater(row['fast_ma'], row['slow_ma'])
            self.assertLess(row['rsi'], config['strategy']['rsi_threshold'])
        
        # Check a few sell signals
        for _, row in sell_signals.iterrows():
            self.assertLess(row['fast_ma'], row['slow_ma'])
            self.assertGreater(row['rsi'], config['strategy']['rsi_threshold'])


if __name__ == '__main__':
    unittest.main()
