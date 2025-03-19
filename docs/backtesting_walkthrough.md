# Backtesting Walkthrough Guide

This step-by-step walkthrough guide demonstrates how to use the Platon Light backtesting module to develop, test, and optimize a trading strategy from start to finish.

## Table of Contents

1. [Introduction](#introduction)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Defining Your Strategy](#defining-your-strategy)
4. [Loading Historical Data](#loading-historical-data)
5. [Running Your First Backtest](#running-your-first-backtest)
6. [Analyzing Backtest Results](#analyzing-backtest-results)
7. [Visualizing Performance](#visualizing-performance)
8. [Optimizing Strategy Parameters](#optimizing-strategy-parameters)
9. [Validating Strategy Robustness](#validating-strategy-robustness)
10. [Implementing in Live Trading](#implementing-in-live-trading)
11. [Complete Example](#complete-example)

## Introduction

Backtesting is the process of testing a trading strategy on historical data to evaluate its performance before risking real capital. This walkthrough will guide you through the entire backtesting process using the Platon Light backtesting module.

## Setting Up Your Environment

Before you begin backtesting, make sure you have installed all the required dependencies:

```bash
pip install -r requirements.txt
```

Next, import the necessary modules:

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
```

## Defining Your Strategy

The first step in backtesting is to define your trading strategy. In Platon Light, strategies are created by subclassing the `BaseStrategy` class and implementing the required methods:

```python
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

# Register the strategy with the factory
StrategyFactory.register_strategy("moving_average_crossover", MovingAverageCrossover)
```

## Loading Historical Data

Next, you need to load historical market data for backtesting:

```python
# Define backtest parameters
symbol = 'BTCUSDT'
timeframe = '1h'
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 12, 31)

# Create configuration
config = {
    'backtesting': {
        'initial_capital': 10000,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0005    # 0.05%
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
print(data.head())
```

## Running Your First Backtest

With your strategy defined and historical data loaded, you can now run your first backtest:

```python
# Create backtest engine
backtest_engine = BacktestEngine(config)

# Run backtest
results = backtest_engine.run_with_data(data)

print("Backtest completed successfully!")
```

## Analyzing Backtest Results

After running the backtest, you need to analyze the results to evaluate your strategy's performance:

```python
# Extract key metrics
metrics = results['metrics']

print("\nPerformance Metrics:")
print(f"Total Return: {metrics['return_percent']:.2f}%")
print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Total Trades: {metrics['total_trades']}")
print(f"Average Trade: {metrics['avg_trade_percent']:.2f}%")
print(f"Average Win: {metrics['avg_win_percent']:.2f}%")
print(f"Average Loss: {metrics['avg_loss_percent']:.2f}%")
print(f"Average Hold Time: {metrics['avg_hold_time']} hours")

# Create performance analyzer for detailed analysis
performance_analyzer = PerformanceAnalyzer(config)
detailed_analysis = performance_analyzer.analyze(results)

# Print monthly returns
print("\nMonthly Returns:")
print(detailed_analysis['monthly_returns'])

# Print drawdown periods
print("\nDrawdown Periods:")
print(detailed_analysis['drawdown_periods'])
```

## Visualizing Performance

Visualization is crucial for understanding your strategy's performance:

```python
# Create visualizer
visualizer = BacktestVisualizer(config)

# Plot equity curve
visualizer.plot_equity_curve(results)

# Plot drawdown
visualizer.plot_drawdown(results)

# Plot trade outcomes
visualizer.plot_trade_outcomes(results)

# Plot monthly returns
visualizer.plot_monthly_returns(detailed_analysis['monthly_returns'])

# Plot moving averages and trades on price chart
visualizer.plot_strategy(data, results['trades'])

# Save all plots to a report
visualizer.save_report(results, 'backtest_report.html')
```

## Optimizing Strategy Parameters

Once you have a working strategy, you can optimize its parameters:

```python
# Define parameter ranges to test
param_grid = {
    'fast_period': range(5, 30, 5),
    'slow_period': range(20, 100, 10)
}

# Create optimizer
optimizer = ParameterOptimizer(config)

# Run optimization
optimization_results = optimizer.optimize(
    data=data,
    param_grid=param_grid,
    metric='sharpe_ratio',  # Optimize for Sharpe ratio
    maximize=True           # We want to maximize this metric
)

# Print best parameters
best_params = optimization_results['best_params']
best_metrics = optimization_results['best_metrics']

print("\nOptimization Results:")
print(f"Best Parameters: {best_params}")
print(f"Best Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
print(f"Best Return: {best_metrics['return_percent']:.2f}%")

# Visualize optimization results
visualizer.plot_optimization_heatmap(
    optimization_results['all_results'],
    param_x='fast_period',
    param_y='slow_period',
    metric='sharpe_ratio'
)
```

## Validating Strategy Robustness

After optimizing your strategy, it's important to validate its robustness:

```python
# 1. Walk-forward testing
walk_forward_results = optimizer.walk_forward_test(
    data=data,
    param_grid=param_grid,
    train_size=0.7,
    test_size=0.3,
    metric='sharpe_ratio',
    maximize=True
)

print("\nWalk-Forward Test Results:")
print(f"Train Performance: {walk_forward_results['train_metrics']['return_percent']:.2f}%")
print(f"Test Performance: {walk_forward_results['test_metrics']['return_percent']:.2f}%")

# 2. Monte Carlo simulation
from platon_light.tests.monte_carlo_simulation import MonteCarloSimulator

simulator = MonteCarloSimulator()
simulation_results = simulator.run_simulation(
    backtest_results=results,
    num_simulations=1000,
    confidence_level=0.95
)

print("\nMonte Carlo Simulation Results:")
print(f"95% Confidence Interval for Final Equity: "
      f"[{simulation_results['trade_based_simulation']['confidence_interval']['lower_bound']:.2f}%, "
      f"{simulation_results['trade_based_simulation']['confidence_interval']['upper_bound']:.2f}%]")

# 3. Multi-timeframe testing
timeframes = ['15m', '1h', '4h', '1d']
multi_tf_results = {}

for tf in timeframes:
    print(f"\nTesting on {tf} timeframe...")
    tf_data = data_loader.load_data(
        symbol=symbol,
        timeframe=tf,
        start_date=start_date,
        end_date=end_date
    )
    
    multi_tf_results[tf] = backtest_engine.run_with_data(tf_data)
    print(f"Return: {multi_tf_results[tf]['metrics']['return_percent']:.2f}%")
    print(f"Sharpe Ratio: {multi_tf_results[tf]['metrics']['sharpe_ratio']:.2f}")
```

## Implementing in Live Trading

Once you're satisfied with your strategy's backtest performance, you can implement it in live trading:

```python
# Example code to transition from backtest to live trading
from platon_light.live.strategy_runner import StrategyRunner

# Use the optimized parameters
live_config = config.copy()
live_config['strategy'].update(best_params)

# Create strategy runner
strategy_runner = StrategyRunner(live_config)

# Connect to exchange
strategy_runner.connect()

# Start trading
strategy_runner.start()
```

## Complete Example

Here's a complete example that ties everything together:

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


# Define strategy
class MovingAverageCrossover(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        strategy_config = config.get('strategy', {})
        self.fast_period = strategy_config.get('fast_period', 20)
        self.slow_period = strategy_config.get('slow_period', 50)
        
    def generate_signals(self, data):
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


def main():
    # Define backtest parameters
    symbol = 'BTCUSDT'
    timeframe = '1h'
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    
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
    
    # Create backtest engine
    backtest_engine = BacktestEngine(config)
    
    # Run backtest
    results = backtest_engine.run_with_data(data)
    
    print("Backtest completed successfully!")
    
    # Analyze results
    metrics = results['metrics']
    
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['return_percent']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    
    # Create visualizer
    visualizer = BacktestVisualizer(config)
    
    # Plot equity curve
    visualizer.plot_equity_curve(results)
    
    # Plot drawdown
    visualizer.plot_drawdown(results)
    
    # Plot strategy
    visualizer.plot_strategy(data, results['trades'])
    
    # Optimize strategy parameters
    param_grid = {
        'fast_period': range(5, 30, 5),
        'slow_period': range(20, 100, 10)
    }
    
    optimizer = ParameterOptimizer(config)
    
    optimization_results = optimizer.optimize(
        data=data,
        param_grid=param_grid,
        metric='sharpe_ratio',
        maximize=True
    )
    
    best_params = optimization_results['best_params']
    best_metrics = optimization_results['best_metrics']
    
    print("\nOptimization Results:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"Best Return: {best_metrics['return_percent']:.2f}%")
    
    # Visualize optimization results
    visualizer.plot_optimization_heatmap(
        optimization_results['all_results'],
        param_x='fast_period',
        param_y='slow_period',
        metric='sharpe_ratio'
    )
    
    # Save report
    visualizer.save_report(results, 'backtest_report.html')
    
    print("\nBacktesting walkthrough completed!")


if __name__ == "__main__":
    main()
```

This walkthrough guide provides a comprehensive overview of the backtesting process using the Platon Light backtesting module. By following these steps, you can develop, test, optimize, and validate your trading strategies before deploying them in live trading.

Remember that past performance does not guarantee future results. Always validate your strategies thoroughly and start with small position sizes when transitioning to live trading.

For more detailed information on specific aspects of backtesting, refer to the other guides in the documentation:

- [Backtesting API Reference](backtesting_api_reference.md)
- [Custom Strategies Guide](custom_strategies_guide.md)
- [Strategy Optimization Guide](strategy_optimization_guide.md)
- [Performance Metrics Reference](backtesting_performance_metrics.md)
- [Multi-Timeframe Backtesting Guide](multi_timeframe_backtesting.md)
- [Monte Carlo Simulation Guide](monte_carlo_simulation_guide.md)
- [Strategy A/B Testing Guide](strategy_ab_testing_guide.md)
