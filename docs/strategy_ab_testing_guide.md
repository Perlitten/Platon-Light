# Strategy A/B Testing Guide

This guide explains how to conduct A/B testing of trading strategies using the Platon Light backtesting module.

## Table of Contents

1. [Introduction](#introduction)
2. [A/B Testing Principles](#ab-testing-principles)
3. [Setting Up A/B Tests](#setting-up-ab-tests)
4. [Statistical Analysis](#statistical-analysis)
5. [Avoiding Biases](#avoiding-biases)
6. [Implementation Examples](#implementation-examples)
7. [Best Practices](#best-practices)

## Introduction

A/B testing (also known as split testing) is a method of comparing two versions of a trading strategy to determine which performs better. By systematically testing strategy variants, you can make data-driven decisions about which features or parameters to implement in your final strategy.

## A/B Testing Principles

Effective A/B testing of trading strategies follows these principles:

1. **Single Variable Testing**: Change only one variable at a time to isolate its impact
2. **Controlled Environment**: Use identical market data and testing conditions
3. **Statistical Significance**: Ensure results are statistically significant, not due to chance
4. **Adequate Sample Size**: Use enough data to draw meaningful conclusions
5. **Unbiased Evaluation**: Use objective metrics to evaluate performance

## Setting Up A/B Tests

### Basic A/B Test Setup

```python
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

# Define test parameters
symbol = 'BTCUSDT'
timeframe = '1d'
start_date = datetime(2020, 1, 1)
end_date = datetime(2022, 12, 31)

# Base configuration
base_config = {
    'backtesting': {
        'initial_capital': 10000,
        'commission': 0.001,
        'slippage': 0.0005
    }
}

# Strategy A configuration (original)
config_a = base_config.copy()
config_a['strategy'] = {
    'name': 'moving_average',
    'fast_period': 20,
    'slow_period': 50
}

# Strategy B configuration (variant)
config_b = base_config.copy()
config_b['strategy'] = {
    'name': 'moving_average',
    'fast_period': 10,
    'slow_period': 30
}

# Run backtests
backtest_engine_a = BacktestEngine(config_a)
results_a = backtest_engine_a.run(symbol, timeframe, start_date, end_date)

backtest_engine_b = BacktestEngine(config_b)
results_b = backtest_engine_b.run(symbol, timeframe, start_date, end_date)

# Compare results
print("Strategy A Results:")
print(f"Total Return: {results_a['metrics']['return_percent']:.2f}%")
print(f"Sharpe Ratio: {results_a['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results_a['metrics']['max_drawdown_percent']:.2f}%")

print("\nStrategy B Results:")
print(f"Total Return: {results_b['metrics']['return_percent']:.2f}%")
print(f"Sharpe Ratio: {results_b['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results_b['metrics']['max_drawdown_percent']:.2f}%")
```

### Visualizing A/B Test Results

```python
def compare_equity_curves(results_a, results_b, title="Strategy A/B Comparison"):
    """Compare equity curves of two strategies"""
    plt.figure(figsize=(12, 6))
    
    # Extract equity curve data
    equity_a = pd.DataFrame(results_a['equity_curve'])
    equity_a['timestamp'] = pd.to_datetime(equity_a['timestamp'], unit='ms')
    
    equity_b = pd.DataFrame(results_b['equity_curve'])
    equity_b['timestamp'] = pd.to_datetime(equity_b['timestamp'], unit='ms')
    
    # Normalize to percentage returns
    initial_equity_a = equity_a['equity'].iloc[0]
    equity_a['equity_pct'] = (equity_a['equity'] / initial_equity_a - 1) * 100
    
    initial_equity_b = equity_b['equity'].iloc[0]
    equity_b['equity_pct'] = (equity_b['equity'] / initial_equity_b - 1) * 100
    
    # Plot equity curves
    plt.plot(equity_a['timestamp'], equity_a['equity_pct'], label='Strategy A')
    plt.plot(equity_b['timestamp'], equity_b['equity_pct'], label='Strategy B')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_ab_comparison.png')
    plt.show()

# Compare equity curves
compare_equity_curves(results_a, results_b)
```

## Statistical Analysis

### Calculating Statistical Significance

```python
def test_statistical_significance(results_a, results_b, alpha=0.05):
    """
    Test if the performance difference between two strategies is statistically significant.
    
    Args:
        results_a (dict): Results from strategy A
        results_b (dict): Results from strategy B
        alpha (float): Significance level
        
    Returns:
        dict: Dictionary containing test results
    """
    # Calculate daily returns
    equity_a = pd.DataFrame(results_a['equity_curve'])
    equity_a['timestamp'] = pd.to_datetime(equity_a['timestamp'], unit='ms')
    equity_a.set_index('timestamp', inplace=True)
    equity_a['daily_return'] = equity_a['equity'].pct_change()
    
    equity_b = pd.DataFrame(results_b['equity_curve'])
    equity_b['timestamp'] = pd.to_datetime(equity_b['timestamp'], unit='ms')
    equity_b.set_index('timestamp', inplace=True)
    equity_b['daily_return'] = equity_b['equity'].pct_change()
    
    # Align the time series
    common_index = equity_a.index.intersection(equity_b.index)
    returns_a = equity_a.loc[common_index, 'daily_return'].dropna()
    returns_b = equity_b.loc[common_index, 'daily_return'].dropna()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(returns_a, returns_b)
    
    # Check if the difference is significant
    is_significant = p_value < alpha
    
    # Calculate effect size (Cohen's d)
    mean_a = returns_a.mean()
    mean_b = returns_b.mean()
    std_pooled = ((returns_a.std() ** 2 + returns_b.std() ** 2) / 2) ** 0.5
    effect_size = abs(mean_a - mean_b) / std_pooled
    
    # Interpret effect size
    if effect_size < 0.2:
        effect_interpretation = "negligible"
    elif effect_size < 0.5:
        effect_interpretation = "small"
    elif effect_size < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    # Print results
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Statistically significant: {is_significant}")
    print(f"Effect size (Cohen's d): {effect_size:.4f} ({effect_interpretation})")
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'effect_size': effect_size,
        'effect_interpretation': effect_interpretation
    }

# Test statistical significance
significance_results = test_statistical_significance(results_a, results_b)
```

### Comparing Multiple Metrics

```python
def compare_multiple_metrics(results_a, results_b):
    """Compare multiple performance metrics between two strategies"""
    metrics = ['return_percent', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown_percent', 
               'win_rate', 'profit_factor', 'avg_trade_percent']
    
    comparison = pd.DataFrame({
        'Strategy A': [results_a['metrics'].get(m, None) for m in metrics],
        'Strategy B': [results_b['metrics'].get(m, None) for m in metrics],
    }, index=metrics)
    
    # Calculate difference and percent improvement
    comparison['Difference'] = comparison['Strategy B'] - comparison['Strategy A']
    comparison['% Improvement'] = (comparison['Difference'] / comparison['Strategy A'].abs()) * 100
    
    # Print comparison table
    print("Performance Metrics Comparison:")
    print(comparison)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    comparison[['Strategy A', 'Strategy B']].plot(kind='bar')
    plt.title('Strategy A/B Metrics Comparison')
    plt.ylabel('Value')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.show()
    
    return comparison

# Compare multiple metrics
metrics_comparison = compare_multiple_metrics(results_a, results_b)
```

## Avoiding Biases

When conducting A/B tests, be aware of these potential biases:

1. **Selection Bias**: Avoid cherry-picking time periods that favor one strategy
2. **Look-Ahead Bias**: Ensure strategy B doesn't use future information
3. **Survivorship Bias**: Test on a complete dataset including delisted assets
4. **Overfitting**: Don't optimize strategy B based on the test period
5. **Multiple Testing Bias**: Adjust significance levels when testing many variants

### Techniques to Reduce Bias

1. **Time Period Splitting**:
   ```python
   # Split test period into multiple sub-periods
   periods = [
       (datetime(2020, 1, 1), datetime(2020, 6, 30)),
       (datetime(2020, 7, 1), datetime(2020, 12, 31)),
       (datetime(2021, 1, 1), datetime(2021, 6, 30)),
       (datetime(2021, 7, 1), datetime(2021, 12, 31))
   ]
   
   # Test strategies across all periods
   for i, (start, end) in enumerate(periods):
       print(f"\nPeriod {i+1}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
       
       results_a = backtest_engine_a.run(symbol, timeframe, start, end)
       results_b = backtest_engine_b.run(symbol, timeframe, start, end)
       
       print(f"Strategy A Return: {results_a['metrics']['return_percent']:.2f}%")
       print(f"Strategy B Return: {results_b['metrics']['return_percent']:.2f}%")
   ```

2. **Multiple Asset Testing**:
   ```python
   # Test strategies across multiple assets
   symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'XRPUSDT', 'BNBUSDT']
   
   for symbol in symbols:
       print(f"\nAsset: {symbol}")
       
       results_a = backtest_engine_a.run(symbol, timeframe, start_date, end_date)
       results_b = backtest_engine_b.run(symbol, timeframe, start_date, end_date)
       
       print(f"Strategy A Return: {results_a['metrics']['return_percent']:.2f}%")
       print(f"Strategy B Return: {results_b['metrics']['return_percent']:.2f}%")
   ```

3. **Walk-Forward Testing**:
   ```python
   # Perform walk-forward testing
   window_size = timedelta(days=180)
   step_size = timedelta(days=30)
   
   current_start = start_date
   results_a_wf = []
   results_b_wf = []
   
   while current_start + window_size <= end_date:
       current_end = current_start + window_size
       
       # Run backtests for this window
       window_results_a = backtest_engine_a.run(symbol, timeframe, current_start, current_end)
       window_results_b = backtest_engine_b.run(symbol, timeframe, current_start, current_end)
       
       results_a_wf.append({
           'start': current_start,
           'end': current_end,
           'return': window_results_a['metrics']['return_percent']
       })
       
       results_b_wf.append({
           'start': current_start,
           'end': current_end,
           'return': window_results_b['metrics']['return_percent']
       })
       
       # Move to next window
       current_start += step_size
   
   # Convert to DataFrame for analysis
   wf_results = pd.DataFrame({
       'start': [r['start'] for r in results_a_wf],
       'end': [r['end'] for r in results_a_wf],
       'strategy_a_return': [r['return'] for r in results_a_wf],
       'strategy_b_return': [r['return'] for r in results_b_wf]
   })
   
   # Calculate win rate for strategy B
   wf_results['b_wins'] = wf_results['strategy_b_return'] > wf_results['strategy_a_return']
   win_rate = wf_results['b_wins'].mean() * 100
   
   print(f"Strategy B win rate across {len(wf_results)} windows: {win_rate:.2f}%")
   ```

## Implementation Examples

### Complete A/B Test Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer

def run_ab_test(symbol, timeframe, start_date, end_date, config_a, config_b, name_a="Strategy A", name_b="Strategy B"):
    """
    Run a complete A/B test between two strategy configurations.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe for data
        start_date (datetime): Start date for backtest
        end_date (datetime): End date for backtest
        config_a (dict): Configuration for strategy A
        config_b (dict): Configuration for strategy B
        name_a (str): Name for strategy A
        name_b (str): Name for strategy B
        
    Returns:
        dict: Dictionary containing test results
    """
    print(f"Running A/B test for {symbol} from {start_date} to {end_date}")
    print(f"Testing {name_a} vs {name_b}")
    
    # Run backtests
    backtest_engine_a = BacktestEngine(config_a)
    results_a = backtest_engine_a.run(symbol, timeframe, start_date, end_date)
    
    backtest_engine_b = BacktestEngine(config_b)
    results_b = backtest_engine_b.run(symbol, timeframe, start_date, end_date)
    
    # Compare key metrics
    metrics = ['return_percent', 'sharpe_ratio', 'max_drawdown_percent', 'win_rate']
    
    print("\nKey Metrics Comparison:")
    for metric in metrics:
        value_a = results_a['metrics'].get(metric, 0)
        value_b = results_b['metrics'].get(metric, 0)
        difference = value_b - value_a
        
        if metric in ['max_drawdown_percent']:  # Lower is better
            is_better = difference < 0
            difference_str = f"{difference:.2f}"
        else:  # Higher is better
            is_better = difference > 0
            difference_str = f"+{difference:.2f}" if difference > 0 else f"{difference:.2f}"
        
        result_symbol = "✓" if is_better else "✗"
        
        print(f"{metric}: {value_a:.2f} vs {value_b:.2f} ({difference_str}) {result_symbol}")
    
    # Calculate statistical significance
    significance_results = test_statistical_significance(results_a, results_b)
    
    # Compare equity curves
    compare_equity_curves(results_a, results_b, f"{name_a} vs {name_b}")
    
    # Compare drawdowns
    compare_drawdowns(results_a, results_b, name_a, name_b)
    
    # Compare trade distributions
    compare_trade_distributions(results_a, results_b, name_a, name_b)
    
    # Determine winner
    if significance_results['is_significant']:
        # If statistically significant, choose based on Sharpe ratio
        if results_b['metrics']['sharpe_ratio'] > results_a['metrics']['sharpe_ratio']:
            winner = name_b
        else:
            winner = name_a
        
        print(f"\nWinner: {winner} (statistically significant difference)")
    else:
        # If not statistically significant, consider them equivalent
        print("\nNo statistically significant difference between strategies")
        
        # Still provide a recommendation based on Sharpe ratio
        if results_b['metrics']['sharpe_ratio'] > results_a['metrics']['sharpe_ratio']:
            print(f"Slight preference for {name_b} based on Sharpe ratio")
        else:
            print(f"Slight preference for {name_a} based on Sharpe ratio")
    
    return {
        'results_a': results_a,
        'results_b': results_b,
        'significance': significance_results,
        'metrics_comparison': {metric: (results_a['metrics'].get(metric, 0), 
                                       results_b['metrics'].get(metric, 0)) 
                              for metric in metrics}
    }

def compare_drawdowns(results_a, results_b, name_a="Strategy A", name_b="Strategy B"):
    """Compare drawdown profiles of two strategies"""
    # Extract equity data
    equity_a = pd.DataFrame(results_a['equity_curve'])
    equity_a['timestamp'] = pd.to_datetime(equity_a['timestamp'], unit='ms')
    equity_a.set_index('timestamp', inplace=True)
    
    equity_b = pd.DataFrame(results_b['equity_curve'])
    equity_b['timestamp'] = pd.to_datetime(equity_b['timestamp'], unit='ms')
    equity_b.set_index('timestamp', inplace=True)
    
    # Calculate drawdowns
    equity_a['peak'] = equity_a['equity'].cummax()
    equity_a['drawdown'] = (equity_a['equity'] - equity_a['peak']) / equity_a['peak'] * 100
    
    equity_b['peak'] = equity_b['equity'].cummax()
    equity_b['drawdown'] = (equity_b['equity'] - equity_b['peak']) / equity_b['peak'] * 100
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    plt.plot(equity_a.index, equity_a['drawdown'], label=name_a)
    plt.plot(equity_b.index, equity_b['drawdown'], label=name_b)
    plt.title('Drawdown Comparison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('drawdown_comparison.png')
    plt.show()

def compare_trade_distributions(results_a, results_b, name_a="Strategy A", name_b="Strategy B"):
    """Compare trade profit/loss distributions"""
    # Extract trade data
    trades_a = pd.DataFrame(results_a['trades'])
    trades_b = pd.DataFrame(results_b['trades'])
    
    # Calculate profit/loss percentages
    if not trades_a.empty and 'profit_loss' in trades_a.columns:
        trades_a['profit_loss_pct'] = trades_a['profit_loss_percent']
    
    if not trades_b.empty and 'profit_loss' in trades_b.columns:
        trades_b['profit_loss_pct'] = trades_b['profit_loss_percent']
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    
    if not trades_a.empty and 'profit_loss_pct' in trades_a.columns:
        plt.hist(trades_a['profit_loss_pct'], alpha=0.5, bins=20, label=name_a)
    
    if not trades_b.empty and 'profit_loss_pct' in trades_b.columns:
        plt.hist(trades_b['profit_loss_pct'], alpha=0.5, bins=20, label=name_b)
    
    plt.title('Trade Profit/Loss Distribution')
    plt.xlabel('Profit/Loss (%)')
    plt.ylabel('Number of Trades')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trade_distribution_comparison.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define test parameters
    symbol = 'BTCUSDT'
    timeframe = '1d'
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Base configuration
    base_config = {
        'backtesting': {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005
        }
    }
    
    # Strategy A: Original SMA Crossover
    config_a = base_config.copy()
    config_a['strategy'] = {
        'name': 'moving_average',
        'fast_period': 20,
        'slow_period': 50
    }
    
    # Strategy B: Modified SMA Crossover
    config_b = base_config.copy()
    config_b['strategy'] = {
        'name': 'moving_average',
        'fast_period': 10,
        'slow_period': 30
    }
    
    # Run A/B test
    test_results = run_ab_test(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config_a=config_a,
        config_b=config_b,
        name_a="SMA(20,50)",
        name_b="SMA(10,30)"
    )
```

## Best Practices

1. **Define Clear Hypotheses**: Clearly state what you're testing and why
2. **Use Adequate Sample Sizes**: Test over long enough periods to capture different market conditions
3. **Control for External Factors**: Ensure both strategies are tested under identical conditions
4. **Focus on Risk-Adjusted Returns**: Compare Sharpe ratios, not just total returns
5. **Test Multiple Metrics**: Evaluate performance across various dimensions (returns, drawdowns, win rate, etc.)
6. **Use Statistical Tests**: Confirm results are statistically significant
7. **Test Across Assets**: Verify performance across multiple trading instruments
8. **Document Everything**: Keep detailed records of all tests and results
9. **Avoid Overfitting**: Don't repeatedly modify and test on the same dataset
10. **Implement Gradually**: Roll out winning strategies gradually, monitoring performance

By following these A/B testing principles, you can systematically improve your trading strategies and make data-driven decisions about which features to implement.
