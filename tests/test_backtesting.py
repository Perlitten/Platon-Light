import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.backtesting.optimization import StrategyOptimizer


class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'backtesting': {
                'data_dir': tempfile.mkdtemp(),
                'use_cache': False,
                'default_timeframe': '1h'
            }
        }
        self.data_loader = DataLoader(self.config)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')
        self.sample_data = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(100, 5, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        })
        
        # Save sample data to CSV for testing
        self.sample_data_path = os.path.join(self.config['backtesting']['data_dir'], 'BTCUSDT_1h.csv')
        self.sample_data.to_csv(self.sample_data_path, index=False)

    def test_load_data_from_csv(self):
        """Test loading data from CSV file"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        data = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertTrue('timestamp' in data.columns)
        self.assertTrue('open' in data.columns)
        self.assertTrue('high' in data.columns)
        self.assertTrue('low' in data.columns)
        self.assertTrue('close' in data.columns)
        self.assertTrue('volume' in data.columns)

    def test_date_filtering(self):
        """Test that data is correctly filtered by date range"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 3)
        end_date = datetime(2023, 1, 5)
        
        data = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
        
        # Convert timestamps to datetime for easier comparison
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Check that all data points are within the specified range
        self.assertTrue(all(data['datetime'] >= start_date))
        self.assertTrue(all(data['datetime'] <= end_date + timedelta(days=1)))  # Add 1 day to include end_date

    def test_resample_data(self):
        """Test resampling data to a different timeframe"""
        # Load 1h data
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        hourly_data = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
        
        # Resample to 4h
        resampled_data = self.data_loader.resample_data(hourly_data, '4h')
        
        self.assertIsNotNone(resampled_data)
        self.assertIsInstance(resampled_data, pd.DataFrame)
        self.assertLess(len(resampled_data), len(hourly_data))  # 4h data should have fewer rows than 1h data

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory and files
        if os.path.exists(self.sample_data_path):
            os.remove(self.sample_data_path)
        if os.path.exists(self.config['backtesting']['data_dir']):
            os.rmdir(self.config['backtesting']['data_dir'])


class TestBacktestEngine(unittest.TestCase):
    """Test cases for the BacktestEngine class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'backtesting': {
                'initial_capital': 10000,
                'commission': 0.001,  # 0.1%
                'slippage': 0.0005,   # 0.05%
                'position_size': 0.1  # 10% of capital per trade
            },
            'strategy': {
                'name': 'moving_average',
                'fast_period': 10,
                'slow_period': 30
            }
        }
        
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
        
        # Add moving averages for testing
        self.sample_data['fast_ma'] = self.sample_data['close'].rolling(window=10).mean()
        self.sample_data['slow_ma'] = self.sample_data['close'].rolling(window=30).mean()
        
        # Add signals
        self.sample_data['signal'] = 0
        # Buy signal when fast_ma crosses above slow_ma
        self.sample_data.loc[(self.sample_data['fast_ma'] > self.sample_data['slow_ma']) & 
                            (self.sample_data['fast_ma'].shift(1) <= self.sample_data['slow_ma'].shift(1)), 'signal'] = 1
        # Sell signal when fast_ma crosses below slow_ma
        self.sample_data.loc[(self.sample_data['fast_ma'] < self.sample_data['slow_ma']) & 
                            (self.sample_data['fast_ma'].shift(1) >= self.sample_data['slow_ma'].shift(1)), 'signal'] = -1
        
        # Create a mock data loader that returns our sample data
        self.mock_data_loader = type('MockDataLoader', (), {
            'load_data': lambda self, symbol, timeframe, start_date, end_date: self.sample_data
        })()
        self.mock_data_loader.sample_data = self.sample_data
        
        # Initialize backtest engine with mock data loader
        self.backtest_engine = BacktestEngine(self.config)
        self.backtest_engine.data_loader = self.mock_data_loader

    def test_run_backtest(self):
        """Test running a backtest"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        results = self.backtest_engine.run(symbol, timeframe, start_date, end_date)
        
        self.assertIsNotNone(results)
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)
        
        # Check that equity curve has expected fields
        equity_curve = results['equity_curve']
        self.assertGreater(len(equity_curve), 0)
        self.assertTrue(all(key in equity_curve[0] for key in ['timestamp', 'equity', 'cash', 'position_value']))
        
        # Check that trades list has expected fields
        trades = results['trades']
        if trades:  # If any trades were executed
            self.assertTrue(all(key in trades[0] for key in ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit_loss']))
        
        # Check that metrics include key performance indicators
        metrics = results['metrics']
        self.assertIn('return_percent', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown_percent', metrics)

    def test_position_sizing(self):
        """Test that position sizing works correctly"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        # Set position size to 20%
        self.config['backtesting']['position_size'] = 0.2
        self.backtest_engine = BacktestEngine(self.config)
        self.backtest_engine.data_loader = self.mock_data_loader
        
        results_20_pct = self.backtest_engine.run(symbol, timeframe, start_date, end_date)
        
        # Set position size to 10%
        self.config['backtesting']['position_size'] = 0.1
        self.backtest_engine = BacktestEngine(self.config)
        self.backtest_engine.data_loader = self.mock_data_loader
        
        results_10_pct = self.backtest_engine.run(symbol, timeframe, start_date, end_date)
        
        # Trades with 20% position size should have approximately twice the profit/loss
        # of trades with 10% position size (not exactly due to compounding)
        if results_20_pct['trades'] and results_10_pct['trades']:
            avg_pl_20_pct = sum(t['profit_loss'] for t in results_20_pct['trades']) / len(results_20_pct['trades'])
            avg_pl_10_pct = sum(t['profit_loss'] for t in results_10_pct['trades']) / len(results_10_pct['trades'])
            
            # The ratio should be approximately 2, but allow for some deviation due to compounding
            ratio = abs(avg_pl_20_pct / avg_pl_10_pct) if avg_pl_10_pct != 0 else 0
            self.assertGreater(ratio, 1.5)  # Should be close to 2, but allow some margin

    def test_commission_impact(self):
        """Test that commission affects returns correctly"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        # Run with 0.1% commission
        self.config['backtesting']['commission'] = 0.001
        self.backtest_engine = BacktestEngine(self.config)
        self.backtest_engine.data_loader = self.mock_data_loader
        
        results_with_commission = self.backtest_engine.run(symbol, timeframe, start_date, end_date)
        
        # Run with no commission
        self.config['backtesting']['commission'] = 0.0
        self.backtest_engine = BacktestEngine(self.config)
        self.backtest_engine.data_loader = self.mock_data_loader
        
        results_no_commission = self.backtest_engine.run(symbol, timeframe, start_date, end_date)
        
        # Returns with commission should be lower than returns without commission
        self.assertLessEqual(
            results_with_commission['metrics']['return_percent'],
            results_no_commission['metrics']['return_percent']
        )


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for the PerformanceAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {}
        self.analyzer = PerformanceAnalyzer(self.config)
        
        # Create sample backtest results
        self.sample_results = {
            'equity_curve': [
                {'timestamp': 1672531200000, 'equity': 10000, 'cash': 10000, 'position_value': 0},
                {'timestamp': 1672534800000, 'equity': 10050, 'cash': 5000, 'position_value': 5050},
                {'timestamp': 1672538400000, 'equity': 10100, 'cash': 5000, 'position_value': 5100},
                {'timestamp': 1672542000000, 'equity': 10150, 'cash': 5000, 'position_value': 5150},
                {'timestamp': 1672545600000, 'equity': 10200, 'cash': 5000, 'position_value': 5200},
                {'timestamp': 1672549200000, 'equity': 10150, 'cash': 5000, 'position_value': 5150},
                {'timestamp': 1672552800000, 'equity': 10100, 'cash': 5000, 'position_value': 5100},
                {'timestamp': 1672556400000, 'equity': 10050, 'cash': 5000, 'position_value': 5050},
                {'timestamp': 1672560000000, 'equity': 10000, 'cash': 10000, 'position_value': 0},
                {'timestamp': 1672563600000, 'equity': 9950, 'cash': 4950, 'position_value': 5000},
                {'timestamp': 1672567200000, 'equity': 9900, 'cash': 4950, 'position_value': 4950},
                {'timestamp': 1672570800000, 'equity': 9950, 'cash': 4950, 'position_value': 5000},
                {'timestamp': 1672574400000, 'equity': 10000, 'cash': 4950, 'position_value': 5050},
                {'timestamp': 1672578000000, 'equity': 10050, 'cash': 4950, 'position_value': 5100},
                {'timestamp': 1672581600000, 'equity': 10100, 'cash': 10100, 'position_value': 0}
            ],
            'trades': [
                {
                    'entry_time': 1672534800000,
                    'exit_time': 1672560000000,
                    'entry_price': 100,
                    'exit_price': 100,
                    'size': 50,
                    'side': 'long',
                    'profit_loss': 0,
                    'profit_loss_pct': 0
                },
                {
                    'entry_time': 1672563600000,
                    'exit_time': 1672581600000,
                    'entry_price': 100,
                    'exit_price': 102,
                    'size': 50,
                    'side': 'long',
                    'profit_loss': 100,
                    'profit_loss_pct': 2
                }
            ],
            'metrics': {
                'return_percent': 1.0,
                'annualized_return': 365.0,  # Simplified for testing
                'sharpe_ratio': 1.5,
                'max_drawdown_percent': -2.0,
                'win_rate': 50.0
            }
        }

    def test_calculate_metrics(self):
        """Test calculating performance metrics"""
        metrics = self.analyzer.calculate_metrics(self.sample_results)
        
        self.assertIsNotNone(metrics)
        self.assertIn('return_percent', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown_percent', metrics)
        self.assertIn('win_rate', metrics)
        
        # Check specific metrics
        self.assertAlmostEqual(metrics['return_percent'], 1.0, places=1)
        self.assertLessEqual(metrics['max_drawdown_percent'], 0)  # Drawdown should be negative or zero

    def test_calculate_drawdowns(self):
        """Test calculating drawdowns"""
        drawdowns = self.analyzer.calculate_drawdowns(self.sample_results)
        
        self.assertIsNotNone(drawdowns)
        self.assertGreater(len(drawdowns), 0)
        
        # Check that maximum drawdown matches the metrics
        max_dd = min(drawdowns, key=lambda x: x['drawdown_percent'])
        self.assertAlmostEqual(max_dd['drawdown_percent'], self.sample_results['metrics']['max_drawdown_percent'], places=1)

    def test_calculate_trade_statistics(self):
        """Test calculating trade statistics"""
        trade_stats = self.analyzer.calculate_trade_statistics(self.sample_results)
        
        self.assertIsNotNone(trade_stats)
        self.assertIn('total_trades', trade_stats)
        self.assertIn('winning_trades', trade_stats)
        self.assertIn('losing_trades', trade_stats)
        self.assertIn('win_rate', trade_stats)
        self.assertIn('avg_profit_per_trade', trade_stats)
        self.assertIn('avg_profit_per_winning_trade', trade_stats)
        self.assertIn('avg_loss_per_losing_trade', trade_stats)
        
        # Check specific statistics
        self.assertEqual(trade_stats['total_trades'], 2)
        self.assertEqual(trade_stats['winning_trades'], 1)
        self.assertEqual(trade_stats['losing_trades'], 0)  # One trade is break-even
        self.assertAlmostEqual(trade_stats['win_rate'], 50.0, places=1)


class TestVisualization(unittest.TestCase):
    """Test cases for the BacktestVisualizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'visualization': {
                'output_dir': tempfile.mkdtemp()
            }
        }
        self.visualizer = BacktestVisualizer(self.config)
        
        # Create sample backtest results (same as in TestPerformanceAnalyzer)
        self.sample_results = {
            'equity_curve': [
                {'timestamp': 1672531200000, 'equity': 10000, 'cash': 10000, 'position_value': 0},
                {'timestamp': 1672534800000, 'equity': 10050, 'cash': 5000, 'position_value': 5050},
                {'timestamp': 1672538400000, 'equity': 10100, 'cash': 5000, 'position_value': 5100},
                {'timestamp': 1672542000000, 'equity': 10150, 'cash': 5000, 'position_value': 5150},
                {'timestamp': 1672545600000, 'equity': 10200, 'cash': 5000, 'position_value': 5200},
                {'timestamp': 1672549200000, 'equity': 10150, 'cash': 5000, 'position_value': 5150},
                {'timestamp': 1672552800000, 'equity': 10100, 'cash': 5000, 'position_value': 5100},
                {'timestamp': 1672556400000, 'equity': 10050, 'cash': 5000, 'position_value': 5050},
                {'timestamp': 1672560000000, 'equity': 10000, 'cash': 10000, 'position_value': 0},
                {'timestamp': 1672563600000, 'equity': 9950, 'cash': 4950, 'position_value': 5000},
                {'timestamp': 1672567200000, 'equity': 9900, 'cash': 4950, 'position_value': 4950},
                {'timestamp': 1672570800000, 'equity': 9950, 'cash': 4950, 'position_value': 5000},
                {'timestamp': 1672574400000, 'equity': 10000, 'cash': 4950, 'position_value': 5050},
                {'timestamp': 1672578000000, 'equity': 10050, 'cash': 4950, 'position_value': 5100},
                {'timestamp': 1672581600000, 'equity': 10100, 'cash': 10100, 'position_value': 0}
            ],
            'trades': [
                {
                    'entry_time': 1672534800000,
                    'exit_time': 1672560000000,
                    'entry_price': 100,
                    'exit_price': 100,
                    'size': 50,
                    'side': 'long',
                    'profit_loss': 0,
                    'profit_loss_pct': 0
                },
                {
                    'entry_time': 1672563600000,
                    'exit_time': 1672581600000,
                    'entry_price': 100,
                    'exit_price': 102,
                    'size': 50,
                    'side': 'long',
                    'profit_loss': 100,
                    'profit_loss_pct': 2
                }
            ],
            'metrics': {
                'return_percent': 1.0,
                'annualized_return': 365.0,
                'sharpe_ratio': 1.5,
                'max_drawdown_percent': -2.0,
                'win_rate': 50.0
            }
        }

    def test_plot_equity_curve(self):
        """Test plotting equity curve"""
        # Test saving to file
        output_file = os.path.join(self.config['visualization']['output_dir'], 'equity_curve.png')
        self.visualizer.plot_equity_curve(self.sample_results, save=True, show=False, filename=output_file)
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_file))

    def test_plot_drawdown_chart(self):
        """Test plotting drawdown chart"""
        # Test saving to file
        output_file = os.path.join(self.config['visualization']['output_dir'], 'drawdown.png')
        self.visualizer.plot_drawdown_chart(self.sample_results, save=True, show=False, filename=output_file)
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_file))

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory and files
        import shutil
        if os.path.exists(self.config['visualization']['output_dir']):
            shutil.rmtree(self.config['visualization']['output_dir'])


class TestOptimization(unittest.TestCase):
    """Test cases for the StrategyOptimizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'backtesting': {
                'initial_capital': 10000,
                'commission': 0.001,
                'slippage': 0.0005
            },
            'strategy': {
                'name': 'moving_average'
            },
            'optimization': {
                'parallel': False
            }
        }
        
        # Create a mock backtest engine that returns predetermined results
        def mock_run(symbol, timeframe, start_date, end_date, **kwargs):
            # Return different results based on parameters
            fast_period = self.config['strategy'].get('fast_period', 10)
            slow_period = self.config['strategy'].get('slow_period', 30)
            
            # Simple model: better results when fast_period is around 15 and slow_period is around 50
            sharpe = 1.0 - abs(fast_period - 15) * 0.05 - abs(slow_period - 50) * 0.01
            return_pct = 10.0 - abs(fast_period - 15) * 0.5 - abs(slow_period - 50) * 0.1
            
            return {
                'metrics': {
                    'sharpe_ratio': sharpe,
                    'return_percent': return_pct,
                    'max_drawdown_percent': -5.0
                }
            }
        
        self.mock_backtest_engine = type('MockBacktestEngine', (), {
            'run': mock_run,
            'strategy': type('MockStrategy', (), {
                'set_parameters': lambda self, params: None
            })()
        })()
        
        # Initialize optimizer with mock backtest engine
        self.optimizer = StrategyOptimizer(self.config)
        self.optimizer.backtest_engine = self.mock_backtest_engine

    def test_grid_search(self):
        """Test grid search optimization"""
        param_grid = {
            'fast_period': [5, 10, 15, 20, 25],
            'slow_period': [30, 40, 50, 60, 70]
        }
        
        results = self.optimizer.grid_search(
            param_grid=param_grid,
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            metric='sharpe_ratio'
        )
        
        self.assertIsNotNone(results)
        self.assertIn('best_params', results)
        self.assertIn('best_metrics', results)
        self.assertIn('all_results', results)
        
        # Check that best parameters match our mock model
        best_params = results['best_params']
        self.assertEqual(best_params['fast_period'], 15)
        self.assertEqual(best_params['slow_period'], 50)
        
        # Check that all parameter combinations were tested
        all_results = results['all_results']
        self.assertEqual(len(all_results), len(param_grid['fast_period']) * len(param_grid['slow_period']))

    def test_optimize_multiple_metrics(self):
        """Test optimization with multiple metrics"""
        param_grid = {
            'fast_period': [10, 15, 20],
            'slow_period': [40, 50, 60]
        }
        
        # Optimize for Sharpe ratio
        sharpe_results = self.optimizer.grid_search(
            param_grid=param_grid,
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            metric='sharpe_ratio'
        )
        
        # Optimize for return percentage
        return_results = self.optimizer.grid_search(
            param_grid=param_grid,
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            metric='return_percent'
        )
        
        # Both metrics should lead to the same optimal parameters in our mock model
        self.assertEqual(sharpe_results['best_params'], return_results['best_params'])


if __name__ == '__main__':
    unittest.main()
