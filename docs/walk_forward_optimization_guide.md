# Walk-Forward Optimization Guide

This guide explains how to implement and use walk-forward optimization (WFO) to develop robust trading strategies using the Platon Light backtesting module.

## Table of Contents

1. [Introduction](#introduction)
2. [The Problem of Overfitting](#the-problem-of-overfitting)
3. [Walk-Forward Optimization Explained](#walk-forward-optimization-explained)
4. [Implementation in Platon Light](#implementation-in-platon-light)
5. [Analyzing WFO Results](#analyzing-wfo-results)
6. [Best Practices](#best-practices)
7. [Example Implementation](#example-implementation)

## Introduction

Walk-forward optimization (WFO) is an advanced backtesting technique that helps develop more robust trading strategies by reducing the risk of overfitting. It combines parameter optimization with out-of-sample testing in a systematic way, providing a more realistic assessment of strategy performance.

## The Problem of Overfitting

Overfitting occurs when a trading strategy is too closely tailored to historical data, capturing noise rather than genuine market patterns. An overfitted strategy typically performs well in backtests but fails in live trading. Signs of overfitting include:

1. Exceptional backtest performance that seems too good to be true
2. Complex strategies with many parameters
3. High sensitivity to small parameter changes
4. Poor performance on out-of-sample data
5. Strategies that only work in specific market conditions

## Walk-Forward Optimization Explained

Walk-forward optimization addresses overfitting by dividing historical data into multiple segments and performing the following steps:

1. **Divide Data**: Split historical data into multiple segments
2. **Optimize In-Sample**: For each segment, optimize strategy parameters using in-sample data
3. **Test Out-of-Sample**: Apply the optimized parameters to the subsequent out-of-sample data
4. **Combine Results**: Combine all out-of-sample results to evaluate overall strategy performance

This approach simulates how a strategy would perform if you had been developing and trading it over time, periodically re-optimizing parameters based on recent data.

### Types of Walk-Forward Optimization

1. **Standard Walk-Forward**: Data is divided into consecutive segments with no overlap
2. **Anchored Walk-Forward**: The in-sample period starts at the same point but grows longer with each iteration
3. **Rolling Walk-Forward**: Both in-sample and out-of-sample periods move forward with each iteration, maintaining the same length

## Implementation in Platon Light

The Platon Light backtesting module provides built-in support for walk-forward optimization through the `ParameterOptimizer` class.

### Basic Usage

```python
from platon_light.backtesting.optimization import ParameterOptimizer

# Create optimizer
optimizer = ParameterOptimizer(config)

# Define parameter grid
param_grid = {
    'fast_period': range(5, 30, 5),
    'slow_period': range(20, 100, 10)
}

# Run walk-forward optimization
wfo_results = optimizer.walk_forward_test(
    data=data,
    param_grid=param_grid,
    train_size=0.7,      # 70% of each segment for training
    test_size=0.3,       # 30% of each segment for testing
    n_segments=5,        # Number of segments to divide data into
    metric='sharpe_ratio',
    maximize=True
)
```

### Configuration Options

- **train_size**: Proportion of each segment to use for in-sample optimization
- **test_size**: Proportion of each segment to use for out-of-sample testing
- **n_segments**: Number of segments to divide the data into
- **overlap**: Overlap between consecutive segments (for rolling walk-forward)
- **anchor_start**: Whether to use anchored walk-forward optimization
- **metric**: Performance metric to optimize (e.g., 'sharpe_ratio', 'return_percent')
- **maximize**: Whether to maximize or minimize the metric

## Analyzing WFO Results

The walk-forward optimization results contain valuable information about strategy robustness:

### Key Metrics to Analyze

1. **Out-of-Sample Performance**: The combined performance across all out-of-sample periods
2. **Parameter Stability**: How much parameters vary across different segments
3. **Performance Consistency**: Consistency of performance across different segments
4. **Robustness Ratio**: Ratio of out-of-sample performance to in-sample performance

### Visualization

The Platon Light backtesting module provides several visualization tools for analyzing WFO results:

```python
# Plot out-of-sample equity curve
visualizer.plot_wfo_equity_curve(wfo_results)

# Plot parameter stability
visualizer.plot_wfo_parameter_stability(wfo_results)

# Plot performance consistency
visualizer.plot_wfo_performance_consistency(wfo_results)

# Plot robustness ratio
visualizer.plot_wfo_robustness_ratio(wfo_results)
```

## Best Practices

1. **Use Sufficient Data**: Each segment should have enough data for meaningful optimization
2. **Avoid Too Many Parameters**: Limit the number of parameters to reduce the risk of overfitting
3. **Choose Appropriate Segment Size**: Balance between having enough data for optimization and enough segments for robust testing
4. **Consider Market Regimes**: Ensure segments cover different market conditions
5. **Focus on Parameter Stability**: Strategies with stable parameters across segments are more robust
6. **Evaluate Performance Consistency**: Look for consistent performance across all out-of-sample periods
7. **Compare with Simple Benchmark**: Compare WFO results with a simple buy-and-hold or other benchmark strategy
8. **Use Multiple Metrics**: Don't rely on a single performance metric
9. **Forward Test**: Even after WFO, forward testing with paper trading is recommended
10. **Regularly Re-Optimize**: In live trading, periodically re-optimize parameters as new data becomes available

## Example Implementation

Here's a complete example of implementing walk-forward optimization for a moving average crossover strategy:

```python
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.backtesting.optimization import ParameterOptimizer
from platon_light.core.strategy_factory import StrategyFactory
from platon_light.core.base_strategy import BaseStrategy


class MovingAverageCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover strategy.
    Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Get strategy parameters from config
        strategy_config = config.get('strategy', {})
        self.fast_period = strategy_config.get('fast_period', 20)
        self.slow_period = strategy_config.get('slow_period', 50)
        
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (DataFrame): Market data with OHLCV columns
            
        Returns:
            DataFrame: Data with added signal column
        """
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


# Register strategy
StrategyFactory.register_strategy("moving_average_crossover", MovingAverageCrossover)


def run_walk_forward_optimization():
    """Run walk-forward optimization for moving average crossover strategy"""
    # Define backtest parameters
    symbol = 'BTCUSDT'
    timeframe = '1d'
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Create configuration
    config = {
        'backtesting': {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005
        },
        'strategy': {
            'name': 'moving_average_crossover',
            'fast_period': 20,
            'slow_period': 50
        },
        'data': {
            'source': 'csv',
            'directory': 'data/historical'
        }
    }
    
    # Create data loader
    data_loader = DataLoader(config)
    
    # Load historical data
    data = data_loader.load_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Loaded {len(data)} data points for {symbol} {timeframe}")
    
    # Define parameter grid
    param_grid = {
        'fast_period': range(5, 30, 5),
        'slow_period': range(20, 100, 10)
    }
    
    # Create optimizer
    optimizer = ParameterOptimizer(config)
    
    # Run standard walk-forward optimization
    print("\nRunning standard walk-forward optimization...")
    standard_wfo_results = optimizer.walk_forward_test(
        data=data,
        param_grid=param_grid,
        train_size=0.7,
        test_size=0.3,
        n_segments=5,
        metric='sharpe_ratio',
        maximize=True
    )
    
    # Run anchored walk-forward optimization
    print("\nRunning anchored walk-forward optimization...")
    anchored_wfo_results = optimizer.walk_forward_test(
        data=data,
        param_grid=param_grid,
        train_size=0.7,
        test_size=0.3,
        n_segments=5,
        metric='sharpe_ratio',
        maximize=True,
        anchor_start=True
    )
    
    # Run rolling walk-forward optimization
    print("\nRunning rolling walk-forward optimization...")
    rolling_wfo_results = optimizer.walk_forward_test(
        data=data,
        param_grid=param_grid,
        train_size=0.7,
        test_size=0.3,
        n_segments=5,
        metric='sharpe_ratio',
        maximize=True,
        overlap=0.5
    )
    
    # Create visualizer
    visualizer = BacktestVisualizer(config)
    
    # Analyze standard WFO results
    print("\nStandard Walk-Forward Optimization Results:")
    analyze_wfo_results(standard_wfo_results, visualizer, "standard")
    
    # Analyze anchored WFO results
    print("\nAnchored Walk-Forward Optimization Results:")
    analyze_wfo_results(anchored_wfo_results, visualizer, "anchored")
    
    # Analyze rolling WFO results
    print("\nRolling Walk-Forward Optimization Results:")
    analyze_wfo_results(rolling_wfo_results, visualizer, "rolling")
    
    # Compare WFO methods
    compare_wfo_methods(
        standard_wfo_results,
        anchored_wfo_results,
        rolling_wfo_results,
        visualizer
    )
    
    return {
        'standard': standard_wfo_results,
        'anchored': anchored_wfo_results,
        'rolling': rolling_wfo_results
    }


def analyze_wfo_results(wfo_results, visualizer, wfo_type):
    """Analyze walk-forward optimization results"""
    # Extract key metrics
    combined_metrics = wfo_results['combined_test_metrics']
    
    print(f"Combined Out-of-Sample Performance:")
    print(f"Total Return: {combined_metrics['return_percent']:.2f}%")
    print(f"Sharpe Ratio: {combined_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {combined_metrics['max_drawdown_percent']:.2f}%")
    print(f"Win Rate: {combined_metrics['win_rate']:.2f}%")
    
    # Calculate parameter stability
    param_stability = {}
    for param in wfo_results['segment_params'][0].keys():
        values = [segment_params[param] for segment_params in wfo_results['segment_params']]
        param_stability[param] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    print("\nParameter Stability:")
    for param, stats in param_stability.items():
        print(f"{param}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']}, {stats['max']}]")
    
    # Calculate performance consistency
    segment_returns = [metrics['return_percent'] for metrics in wfo_results['segment_test_metrics']]
    segment_sharpes = [metrics['sharpe_ratio'] for metrics in wfo_results['segment_test_metrics']]
    
    print("\nPerformance Consistency:")
    print(f"Returns: mean={np.mean(segment_returns):.2f}%, std={np.std(segment_returns):.2f}%")
    print(f"Sharpe Ratios: mean={np.mean(segment_sharpes):.2f}, std={np.std(segment_sharpes):.2f}")
    
    # Calculate robustness ratio
    train_sharpes = [metrics['sharpe_ratio'] for metrics in wfo_results['segment_train_metrics']]
    test_sharpes = segment_sharpes
    
    robustness_ratios = [test / train if train > 0 else 0 for test, train in zip(test_sharpes, train_sharpes)]
    avg_robustness_ratio = np.mean(robustness_ratios)
    
    print(f"\nRobustness Ratio (Out-of-Sample / In-Sample): {avg_robustness_ratio:.2f}")
    
    # Plot results
    visualizer.plot_wfo_equity_curve(wfo_results, title=f"{wfo_type.capitalize()} WFO Equity Curve")
    visualizer.plot_wfo_parameter_stability(wfo_results, title=f"{wfo_type.capitalize()} WFO Parameter Stability")
    visualizer.plot_wfo_performance_consistency(wfo_results, title=f"{wfo_type.capitalize()} WFO Performance Consistency")
    visualizer.plot_wfo_robustness_ratio(wfo_results, title=f"{wfo_type.capitalize()} WFO Robustness Ratio")


def compare_wfo_methods(standard_results, anchored_results, rolling_results, visualizer):
    """Compare different walk-forward optimization methods"""
    # Extract key metrics
    methods = ['Standard', 'Anchored', 'Rolling']
    results = [standard_results, anchored_results, rolling_results]
    
    # Compare returns
    returns = [result['combined_test_metrics']['return_percent'] for result in results]
    sharpes = [result['combined_test_metrics']['sharpe_ratio'] for result in results]
    drawdowns = [result['combined_test_metrics']['max_drawdown_percent'] for result in results]
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Method': methods,
        'Return (%)': returns,
        'Sharpe Ratio': sharpes,
        'Max Drawdown (%)': drawdowns
    })
    
    print("\nWFO Method Comparison:")
    print(comparison)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot returns
    plt.subplot(2, 2, 1)
    plt.bar(methods, returns)
    plt.title('Total Return (%)')
    plt.grid(axis='y')
    
    # Plot Sharpe ratios
    plt.subplot(2, 2, 2)
    plt.bar(methods, sharpes)
    plt.title('Sharpe Ratio')
    plt.grid(axis='y')
    
    # Plot drawdowns
    plt.subplot(2, 2, 3)
    plt.bar(methods, [abs(d) for d in drawdowns])
    plt.title('Max Drawdown (%)')
    plt.grid(axis='y')
    
    # Plot combined equity curves
    plt.subplot(2, 2, 4)
    for i, result in enumerate(results):
        equity_curve = result['combined_test_equity_curve']
        plt.plot(range(len(equity_curve)), equity_curve, label=methods[i])
    
    plt.title('Combined Equity Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('wfo_method_comparison.png')
    plt.close()
    
    print("\nWFO method comparison chart saved as 'wfo_method_comparison.png'")


if __name__ == "__main__":
    results = run_walk_forward_optimization()
```

## Conclusion

Walk-forward optimization is a powerful technique for developing robust trading strategies while minimizing the risk of overfitting. By systematically combining parameter optimization with out-of-sample testing, WFO provides a more realistic assessment of strategy performance.

The Platon Light backtesting module offers comprehensive support for different types of walk-forward optimization, along with tools for analyzing and visualizing the results. By following the best practices outlined in this guide, you can develop trading strategies that are more likely to perform well in live trading.

Remember that no backtesting method, including WFO, can guarantee future performance. Always start with small position sizes when transitioning a strategy from backtesting to live trading, and continuously monitor performance to ensure it aligns with expectations.
