import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.backtesting.optimization import StrategyOptimizer
from platon_light.core.strategy_factory import StrategyFactory


# Create test configuration
test_config = {
    'backtesting': {
        'initial_capital': 10000,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0005,   # 0.05%
        'position_size': 0.1, # 10% of capital per trade
        'data_dir': './test_data'
    },
    'strategy': {
        'name': 'moving_average',
        'fast_period': 10,
        'slow_period': 30
    },
    'visualization': {
        'output_dir': './test_output'
    }
}

# Ensure test directories exist
os.makedirs(test_config['backtesting']['data_dir'], exist_ok=True)
os.makedirs(test_config['visualization']['output_dir'], exist_ok=True)


def generate_sample_data(symbol, timeframe, start_date, end_date):
    """
    Generate sample market data for testing.
    
    Args:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe for the data
        start_date (datetime): Start date for the data
        end_date (datetime): End date for the data
        
    Returns:
        pd.DataFrame: Sample market data
    """
    # Determine frequency based on timeframe
    if timeframe == '1m':
        freq = '1min'
    elif timeframe == '5m':
        freq = '5min'
    elif timeframe == '15m':
        freq = '15min'
    elif timeframe == '1h':
        freq = '1H'
    elif timeframe == '4h':
        freq = '4H'
    elif timeframe == '1d':
        freq = '1D'
    else:
        freq = '1H'  # Default to 1 hour
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate price data with trend and some randomness
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.normal(0, 1, len(dates))
    
    # Create price series with trend, noise, and some cyclical patterns
    close_prices = base_price + trend + noise + 5 * np.sin(np.linspace(0, 10 * np.pi, len(dates)))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': [int(d.timestamp() * 1000) for d in dates],
        'datetime': dates,
        'open': close_prices - np.random.normal(0, 0.5, len(dates)),
        'high': close_prices + np.random.normal(1, 0.5, len(dates)),
        'low': close_prices - np.random.normal(1, 0.5, len(dates)),
        'close': close_prices,
        'volume': np.random.normal(1000, 200, len(dates))
    })
    
    # Ensure high is always the highest and low is always the lowest
    for i in range(len(data)):
        data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'high'])
        data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'low'])
    
    # Ensure volume is positive
    data['volume'] = data['volume'].abs()
    
    # Save data to CSV for testing
    os.makedirs(os.path.join(test_config['backtesting']['data_dir']), exist_ok=True)
    data_path = os.path.join(test_config['backtesting']['data_dir'], f'{symbol}_{timeframe}.csv')
    data.to_csv(data_path, index=False)
    
    print(f"Generated sample data for {symbol} {timeframe} with {len(data)} data points")
    return data


def run_integration_test(symbol, timeframe, start_date, end_date, strategy_name, strategy_params):
    """
    Run an integration test with the specified parameters.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        timeframe (str): Timeframe for the backtest (e.g., '1h', '4h', '1d')
        start_date (datetime): Start date for the backtest
        end_date (datetime): End date for the backtest
        strategy_name (str): Name of the strategy to test
        strategy_params (dict): Strategy parameters
        
    Returns:
        dict: Backtest results
    """
    # Update configuration
    test_config['strategy'] = {'name': strategy_name, **strategy_params}
    
    # Generate sample data if needed
    generate_sample_data(symbol, timeframe, start_date, end_date)
    
    # Initialize components
    data_loader = DataLoader(test_config)
    backtest_engine = BacktestEngine(test_config)
    performance_analyzer = PerformanceAnalyzer(test_config)
    visualizer = BacktestVisualizer(test_config)
    
    # Run backtest
    print(f"Running backtest for {strategy_name} on {symbol} {timeframe}")
    results = backtest_engine.run(symbol, timeframe, start_date, end_date)
    
    # Analyze results
    print("Analyzing backtest results")
    metrics = performance_analyzer.analyze(results)
    
    # Generate visualizations
    print("Generating visualizations")
    equity_curve_path = os.path.join(test_config['visualization']['output_dir'], f'{symbol}_{timeframe}_{strategy_name}_equity.png')
    drawdown_path = os.path.join(test_config['visualization']['output_dir'], f'{symbol}_{timeframe}_{strategy_name}_drawdown.png')
    
    visualizer.plot_equity_curve(results, save=True, show=False, filename=equity_curve_path)
    visualizer.plot_drawdown_chart(results, save=True, show=False, filename=drawdown_path)
    
    print(f"Integration test completed for {strategy_name} on {symbol} {timeframe}")
    print(f"Return: {metrics['return_percent']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}, Max DD: {metrics['max_drawdown_percent']:.2f}%")
    
    return results


def test_end_to_end_workflow():
    """Test the complete backtesting workflow"""
    print("\n=== Testing End-to-End Workflow ===")
    
    symbol = 'BTCUSDT'
    timeframe = '1h'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    # Test with Moving Average Crossover strategy
    print("\nTesting Moving Average Crossover strategy")
    ma_params = {
        'fast_period': 10,
        'slow_period': 30
    }
    
    ma_results = run_integration_test(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        strategy_name='moving_average',
        strategy_params=ma_params
    )
    
    # Test with RSI strategy
    print("\nTesting RSI strategy")
    rsi_params = {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30
    }
    
    rsi_results = run_integration_test(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        strategy_name='rsi',
        strategy_params=rsi_params
    )
    
    # Compare strategy performance
    print("\nStrategy Comparison:")
    print(f"MA Return: {ma_results['metrics']['return_percent']:.2f}%, RSI Return: {rsi_results['metrics']['return_percent']:.2f}%")
    print(f"MA Sharpe: {ma_results['metrics']['sharpe_ratio']:.2f}, RSI Sharpe: {rsi_results['metrics']['sharpe_ratio']:.2f}")
    
    # Verify results contain expected data
    assert 'equity_curve' in ma_results, "Missing equity curve in results"
    assert 'trades' in ma_results, "Missing trades in results"
    assert 'metrics' in ma_results, "Missing metrics in results"
    
    # Verify trades have expected fields
    if ma_results['trades']:
        trade = ma_results['trades'][0]
        assert 'entry_time' in trade, "Missing entry_time in trade"
        assert 'exit_time' in trade, "Missing exit_time in trade"
        assert 'profit_loss' in trade, "Missing profit_loss in trade"
    
    print("End-to-end workflow test completed successfully")


def test_with_sample_data():
    """Test the backtesting system with sample market data"""
    print("\n=== Testing with Sample Market Data ===")
    
    # Define test parameters
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['1h', '4h', '1d']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\nTesting with sample market data: {symbol} {timeframe}")
            
            # Generate and load data
            data = generate_sample_data(symbol, timeframe, start_date, end_date)
            
            # Verify data quality
            assert not data.empty, f"Failed to generate data for {symbol} {timeframe}"
            assert 'open' in data.columns, "Missing open price column"
            assert 'high' in data.columns, "Missing high price column"
            assert 'low' in data.columns, "Missing low price column"
            assert 'close' in data.columns, "Missing close price column"
            assert 'volume' in data.columns, "Missing volume column"
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            print(f"Missing values: {missing_values}")
            assert missing_values == 0, "Data contains missing values"
            
            # Check for duplicate timestamps
            duplicate_timestamps = data.duplicated(subset=['timestamp']).sum()
            print(f"Duplicate timestamps: {duplicate_timestamps}")
            assert duplicate_timestamps == 0, "Data contains duplicate timestamps"
            
            # Run a simple backtest to verify the data works with the engine
            backtest_engine = BacktestEngine(test_config)
            results = backtest_engine.run(symbol, timeframe, start_date, end_date)
            
            print(f"Backtest completed with {len(results['equity_curve'])} data points")
            print(f"Number of trades: {len(results['trades'])}")
    
    print("Sample data test completed successfully")


def test_multiple_timeframes():
    """Test the backtesting system with multiple timeframes"""
    print("\n=== Testing Multiple Timeframes ===")
    
    symbol = 'BTCUSDT'
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)  # Use a shorter period for higher timeframes
    
    strategy_params = {
        'fast_period': 10,
        'slow_period': 30
    }
    
    results = {}
    
    for timeframe in timeframes:
        print(f"\nTesting timeframe: {timeframe}")
        
        # Adjust periods based on timeframe
        if timeframe in ['5m', '15m']:
            # For shorter timeframes, use shorter periods
            strategy_params['fast_period'] = 20
            strategy_params['slow_period'] = 60
        else:
            strategy_params['fast_period'] = 10
            strategy_params['slow_period'] = 30
        
        # Run backtest
        result = run_integration_test(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_name='moving_average',
            strategy_params=strategy_params
        )
        
        results[timeframe] = result
    
    # Compare results across timeframes
    print("\nTimeframe Comparison:")
    for timeframe, result in results.items():
        print(f"{timeframe}: Return: {result['metrics']['return_percent']:.2f}%, "
              f"Sharpe: {result['metrics']['sharpe_ratio']:.2f}, "
              f"Trades: {len(result['trades'])}")
    
    print("Multiple timeframes test completed successfully")


def test_multiple_symbols():
    """Test the backtesting system with multiple symbols"""
    print("\n=== Testing Multiple Symbols ===")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    timeframe = '1h'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 15)  # Use a shorter period for multiple symbols
    
    strategy_params = {
        'fast_period': 10,
        'slow_period': 30
    }
    
    results = {}
    
    for symbol in symbols:
        print(f"\nTesting symbol: {symbol}")
        
        # Run backtest
        result = run_integration_test(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_name='moving_average',
            strategy_params=strategy_params
        )
        
        results[symbol] = result
    
    # Compare results across symbols
    print("\nSymbol Comparison:")
    for symbol, result in results.items():
        print(f"{symbol}: Return: {result['metrics']['return_percent']:.2f}%, "
              f"Sharpe: {result['metrics']['sharpe_ratio']:.2f}, "
              f"Trades: {len(result['trades'])}")
    
    print("Multiple symbols test completed successfully")


def test_performance():
    """Test the performance of the backtesting system"""
    print("\n=== Performance Testing ===")
    
    symbol = 'BTCUSDT'
    timeframes = ['1h', '4h', '1d']
    
    # Test with different time periods
    test_periods = [
        (datetime(2023, 1, 1), datetime(2023, 1, 7)),   # 1 week
        (datetime(2023, 1, 1), datetime(2023, 1, 15)),  # 2 weeks
        (datetime(2023, 1, 1), datetime(2023, 1, 31))   # 1 month
    ]
    
    strategy_params = {
        'fast_period': 10,
        'slow_period': 30
    }
    
    print("\nPerformance Testing:")
    
    for timeframe in timeframes:
        print(f"\nTimeframe: {timeframe}")
        
        for start_date, end_date in test_periods:
            period_name = f"{(end_date - start_date).days} days"
            
            # Generate sample data
            data = generate_sample_data(symbol, timeframe, start_date, end_date)
            
            # Measure data loading time
            start_time = time.time()
            data_loader = DataLoader(test_config)
            data = data_loader.load_data(symbol, timeframe, start_date, end_date)
            data_load_time = time.time() - start_time
            
            # Measure backtest execution time
            start_time = time.time()
            backtest_engine = BacktestEngine(test_config)
            results = backtest_engine.run(symbol, timeframe, start_date, end_date)
            backtest_time = time.time() - start_time
            
            # Measure analysis time
            start_time = time.time()
            performance_analyzer = PerformanceAnalyzer(test_config)
            metrics = performance_analyzer.analyze(results)
            analysis_time = time.time() - start_time
            
            # Print performance metrics
            print(f"{period_name}: "
                  f"Data points: {len(data)}, "
                  f"Load time: {data_load_time:.2f}s, "
                  f"Backtest time: {backtest_time:.2f}s, "
                  f"Analysis time: {analysis_time:.2f}s, "
                  f"Total time: {data_load_time + backtest_time + analysis_time:.2f}s")
    
    print("Performance test completed successfully")


def test_optimization():
    """Test the optimization functionality"""
    print("\n=== Testing Optimization ===")
    
    symbol = 'BTCUSDT'
    timeframe = '1h'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 15)
    
    # Define parameter grid for optimization
    param_grid = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50]
    }
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(test_config)
    
    # Generate sample data
    generate_sample_data(symbol, timeframe, start_date, end_date)
    
    # Run grid search optimization
    print("\nRunning grid search optimization")
    start_time = time.time()
    grid_results = optimizer.grid_search(
        param_grid=param_grid,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        metric='sharpe_ratio'
    )
    grid_time = time.time() - start_time
    
    # Print optimization results
    print(f"Grid search completed in {grid_time:.2f}s")
    print(f"Best parameters: {grid_results['best_params']}")
    print(f"Best Sharpe ratio: {grid_results['best_metrics']['sharpe_ratio']:.2f}")
    print(f"Best return: {grid_results['best_metrics']['return_percent']:.2f}%")
    
    # Verify optimization results
    assert 'best_params' in grid_results, "Missing best parameters in optimization results"
    assert 'best_metrics' in grid_results, "Missing best metrics in optimization results"
    assert 'all_results' in grid_results, "Missing all results in optimization results"
    
    # Check that all parameter combinations were tested
    expected_combinations = len(param_grid['fast_period']) * len(param_grid['slow_period'])
    actual_combinations = len(grid_results['all_results'])
    assert actual_combinations == expected_combinations, f"Expected {expected_combinations} parameter combinations, got {actual_combinations}"
    
    print("Optimization test completed successfully")


def run_all_tests():
    """Run all integration tests"""
    print("Running Platon Light Backtesting Integration Tests")
    print("=" * 50)
    
    # Run end-to-end workflow test
    test_end_to_end_workflow()
    
    # Run tests with sample market data
    test_with_sample_data()
    
    # Run tests with multiple timeframes
    test_multiple_timeframes()
    
    # Run tests with multiple symbols
    test_multiple_symbols()
    
    # Run performance tests
    test_performance()
    
    # Run optimization tests
    test_optimization()
    
    print("\nAll integration tests completed successfully!")


if __name__ == "__main__":
    # Run all integration tests
    run_all_tests()
