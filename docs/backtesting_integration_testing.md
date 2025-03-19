# Backtesting Integration Testing Guide

This guide provides instructions on how to perform integration testing for the Platon Light backtesting module. Integration testing ensures that all components of the backtesting system work together correctly in real-world scenarios.

## Table of Contents

1. [Introduction](#introduction)
2. [Setting Up the Test Environment](#setting-up-the-test-environment)
3. [Creating Integration Test Cases](#creating-integration-test-cases)
4. [End-to-End Testing Workflow](#end-to-end-testing-workflow)
5. [Testing with Real Market Data](#testing-with-real-market-data)
6. [Testing Multiple Timeframes](#testing-multiple-timeframes)
7. [Testing Multiple Symbols](#testing-multiple-symbols)
8. [Performance Testing](#performance-testing)
9. [Continuous Integration](#continuous-integration)
10. [Best Practices](#best-practices)

## Introduction

Integration testing for the backtesting module verifies that all components (data loading, strategy execution, performance analysis, etc.) work together correctly. Unlike unit tests that focus on individual components, integration tests ensure the entire backtesting pipeline functions as expected.

## Setting Up the Test Environment

Before running integration tests, set up a proper test environment:

```python
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
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
```

## Creating Integration Test Cases

Create comprehensive test cases that cover different aspects of the backtesting system:

```python
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
    
    # Initialize components
    data_loader = DataLoader(test_config)
    backtest_engine = BacktestEngine(test_config)
    performance_analyzer = PerformanceAnalyzer(test_config)
    visualizer = BacktestVisualizer(test_config)
    
    # Run backtest
    results = backtest_engine.run(symbol, timeframe, start_date, end_date)
    
    # Analyze results
    metrics = performance_analyzer.analyze(results)
    
    # Generate visualizations
    visualizer.plot_equity_curve(results, save=True, show=False)
    visualizer.plot_drawdown_chart(results, save=True, show=False)
    
    print(f"Integration test completed for {strategy_name} on {symbol} {timeframe}")
    print(f"Return: {metrics['return_percent']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}, Max DD: {metrics['max_drawdown_percent']:.2f}%")
    
    return results
```

## End-to-End Testing Workflow

Test the entire backtesting workflow from data loading to results visualization:

```python
# Example of an end-to-end test
def test_end_to_end_workflow():
    """Test the complete backtesting workflow"""
    symbol = 'BTCUSDT'
    timeframe = '1h'
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Test with Moving Average Crossover strategy
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
```

## Testing with Real Market Data

Test the backtesting system with real market data to ensure it handles real-world scenarios:

```python
def test_with_real_market_data():
    """Test the backtesting system with real market data"""
    # Define test parameters
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['1h', '4h', '1d']
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\nTesting with real market data: {symbol} {timeframe}")
            
            # Load data
            data_loader = DataLoader(test_config)
            data = data_loader.load_data(symbol, timeframe, start_date, end_date)
            
            # Verify data quality
            assert not data.empty, f"Failed to load data for {symbol} {timeframe}"
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
```

## Testing Multiple Timeframes

Test the backtesting system with different timeframes to ensure it handles timeframe resampling correctly:

```python
def test_multiple_timeframes():
    """Test the backtesting system with multiple timeframes"""
    symbol = 'BTCUSDT'
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 31)  # Use a shorter period for higher timeframes
    
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
```

## Testing Multiple Symbols

Test the backtesting system with different trading pairs to ensure it works across various markets:

```python
def test_multiple_symbols():
    """Test the backtesting system with multiple symbols"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    timeframe = '1h'
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 3, 31)
    
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
```

## Performance Testing

Test the performance of the backtesting system to ensure it can handle large datasets efficiently:

```python
import time

def test_performance():
    """Test the performance of the backtesting system"""
    symbol = 'BTCUSDT'
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    # Test with different time periods
    test_periods = [
        (datetime(2022, 1, 1), datetime(2022, 1, 31)),  # 1 month
        (datetime(2022, 1, 1), datetime(2022, 3, 31)),  # 3 months
        (datetime(2022, 1, 1), datetime(2022, 6, 30)),  # 6 months
        (datetime(2022, 1, 1), datetime(2022, 12, 31))  # 1 year
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
```

## Continuous Integration

Set up continuous integration for your backtesting tests to ensure they run automatically when code changes:

```python
def run_ci_tests():
    """Run all integration tests for CI pipeline"""
    print("Running integration tests for CI...")
    
    # Run end-to-end workflow test
    test_end_to_end_workflow()
    
    # Run tests with real market data
    test_with_real_market_data()
    
    # Run tests with multiple timeframes
    test_multiple_timeframes()
    
    # Run tests with multiple symbols
    test_multiple_symbols()
    
    # Run performance tests
    test_performance()
    
    print("\nAll integration tests completed successfully!")
```

## Best Practices

Follow these best practices for effective integration testing:

1. **Use realistic data**: Always test with real market data that includes various market conditions.

2. **Test edge cases**: Include tests for market gaps, high volatility periods, and low liquidity scenarios.

3. **Validate results**: Don't just run tests, validate that the results make sense and match expectations.

4. **Compare against benchmarks**: Compare your strategy performance against market benchmarks or simple strategies.

5. **Test across multiple timeframes and symbols**: Ensure your system works consistently across different markets and timeframes.

6. **Monitor performance**: Keep track of execution times to identify performance bottlenecks.

7. **Automate testing**: Set up continuous integration to run tests automatically when code changes.

8. **Document test results**: Keep a record of test results to track improvements over time.

9. **Test with different configurations**: Verify that the system works with various parameter combinations.

10. **Separate test data**: Maintain separate datasets for testing to avoid contaminating production data.

Example of a comprehensive integration test script:

```python
if __name__ == "__main__":
    # Run all integration tests
    print("Running Platon Light Backtesting Integration Tests")
    print("=" * 50)
    
    # Run end-to-end workflow test
    print("\n1. End-to-End Workflow Test")
    test_end_to_end_workflow()
    
    # Run tests with real market data
    print("\n2. Real Market Data Test")
    test_with_real_market_data()
    
    # Run tests with multiple timeframes
    print("\n3. Multiple Timeframes Test")
    test_multiple_timeframes()
    
    # Run tests with multiple symbols
    print("\n4. Multiple Symbols Test")
    test_multiple_symbols()
    
    # Run performance tests
    print("\n5. Performance Test")
    test_performance()
    
    print("\nAll integration tests completed successfully!")
```

By following this guide, you can ensure that your backtesting system is thoroughly tested and works reliably in real-world scenarios.
