# Monte Carlo Simulation Guide for Backtesting

This guide explains how to use Monte Carlo simulations to evaluate the robustness of trading strategies and estimate the range of possible outcomes using the Platon Light backtesting module.

## Table of Contents

1. [Introduction](#introduction)
2. [Why Use Monte Carlo Simulations](#why-use-monte-carlo-simulations)
3. [Types of Monte Carlo Simulations](#types-of-monte-carlo-simulations)
4. [Implementation in Platon Light](#implementation-in-platon-light)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)
7. [Example Usage](#example-usage)

## Introduction

Monte Carlo simulation is a computational technique that uses repeated random sampling to obtain a range of possible outcomes for a process that has inherent uncertainty. In trading strategy backtesting, Monte Carlo simulations help assess the robustness of strategies by simulating thousands of possible performance scenarios based on historical data.

## Why Use Monte Carlo Simulations

Traditional backtesting provides a single equity curve based on historical data, which may not represent the full range of possible outcomes. Monte Carlo simulations offer several advantages:

1. **Estimate Performance Range**: Determine the range of possible returns and drawdowns
2. **Assess Robustness**: Evaluate how sensitive a strategy is to the specific sequence of trades
3. **Calculate Probabilities**: Estimate the probability of achieving specific performance targets
4. **Stress Testing**: Understand how a strategy might perform in extreme scenarios
5. **Risk Management**: Make more informed position sizing decisions based on worst-case scenarios
6. **Avoid Overfitting**: Strategies that perform well across many simulations are less likely to be overfitted

## Types of Monte Carlo Simulations

The Platon Light backtesting module supports two main types of Monte Carlo simulations:

### 1. Trade-Based Simulation

This approach resamples historical trades with replacement to create new equity curves:

- **Methodology**: Randomly selects trades from the historical record (with replacement) to create new trade sequences
- **Advantages**: Preserves the distribution of trade outcomes while varying their sequence
- **Use Case**: Best for evaluating the impact of trade sequence on overall performance

### 2. Returns-Based Simulation

This approach resamples daily (or other period) returns to create new equity curves:

- **Methodology**: Randomly selects daily returns from the historical record to create new return sequences
- **Advantages**: Captures the distribution of market returns and their impact on strategy performance
- **Use Case**: Best for estimating the range of possible future performance

## Implementation in Platon Light

The Platon Light backtesting module provides a `MonteCarloSimulator` class for performing Monte Carlo simulations on backtest results.

### Basic Usage

```python
from platon_light.tests.monte_carlo_simulation import MonteCarloSimulator

# Run a backtest
backtest_engine = BacktestEngine(config)
backtest_results = backtest_engine.run(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31)
)

# Create Monte Carlo simulator
simulator = MonteCarloSimulator(output_dir='monte_carlo_results')

# Run simulation
simulation_results = simulator.run_simulation(
    backtest_results=backtest_results,
    num_simulations=1000,
    confidence_level=0.95
)
```

### Configuration Parameters

- **num_simulations**: Number of simulations to run (default: 1000)
- **confidence_level**: Confidence level for intervals (default: 0.95)
- **output_dir**: Directory to save simulation results and visualizations

## Interpreting Results

Monte Carlo simulations generate a wealth of information to help evaluate strategy robustness:

### Key Metrics

1. **Final Equity Distribution**:
   - **Mean/Median Final Equity**: The average/median final equity across all simulations
   - **Confidence Interval**: Range of final equity values within the specified confidence level
   - **Probability of Profit**: Percentage of simulations that end with a profit

2. **Drawdown Distribution**:
   - **Mean/Median Maximum Drawdown**: The average/median maximum drawdown across all simulations
   - **Worst Drawdown**: The worst drawdown observed across all simulations
   - **Drawdown Confidence Interval**: Range of maximum drawdowns within the specified confidence level

3. **Worst-Case Scenarios**:
   - **5th Percentile**: Final equity value that 5% of simulations fall below
   - **1st Percentile**: Final equity value that 1% of simulations fall below

### Visualizations

The Monte Carlo simulator generates several visualizations to help interpret the results:

1. **Equity Curves**: Shows a subset of simulated equity curves along with median and confidence intervals
2. **Final Equity Distribution**: Histogram of final equity values across all simulations
3. **Drawdown Distribution**: Histogram of maximum drawdowns across all simulations

## Best Practices

1. **Run Sufficient Simulations**: Use at least 1,000 simulations for stable results
2. **Use Multiple Confidence Levels**: Analyze results at different confidence levels (e.g., 90%, 95%, 99%)
3. **Compare Both Simulation Types**: Trade-based and returns-based simulations provide different insights
4. **Focus on Worst-Case Scenarios**: Pay special attention to the lower tail of the distribution
5. **Consider Out-of-Sample Testing**: Validate Monte Carlo results with out-of-sample testing
6. **Adjust Position Sizing**: Use Monte Carlo results to inform position sizing decisions
7. **Evaluate Multiple Market Regimes**: Run simulations on different market periods
8. **Look Beyond Averages**: Examine the full distribution, not just mean/median values
9. **Update Regularly**: Re-run simulations as new trade data becomes available
10. **Combine with Other Techniques**: Use Monte Carlo alongside other validation methods

## Example Usage

Here's a complete example of how to use Monte Carlo simulations to evaluate a trading strategy:

```python
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add parent directory to path to import Platon Light modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.tests.monte_carlo_simulation import MonteCarloSimulator

# Define configuration
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
    }
}

# Run backtest
data_loader = DataLoader(config)
backtest_engine = BacktestEngine(config)

backtest_results = backtest_engine.run(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31)
)

# Create Monte Carlo simulator
simulator = MonteCarloSimulator(output_dir='monte_carlo_results')

# Run simulation
simulation_results = simulator.run_simulation(
    backtest_results=backtest_results,
    num_simulations=1000,
    confidence_level=0.95
)

# Analyze trade-based simulation results
trade_sim = simulation_results['trade_based_simulation']
print(f"Trade-Based Simulation Results:")
print(f"Mean Final Equity: {trade_sim['mean_final_equity']:.2f}%")
print(f"95% Confidence Interval: [{trade_sim['confidence_interval']['lower_bound']:.2f}%, "
      f"{trade_sim['confidence_interval']['upper_bound']:.2f}%]")
print(f"Probability of Profit: {trade_sim['probability_of_profit']*100:.2f}%")
print(f"Mean Maximum Drawdown: {trade_sim['drawdown_statistics']['mean_max_drawdown']:.2f}%")
print(f"Worst Drawdown: {trade_sim['drawdown_statistics']['worst_drawdown']:.2f}%")

# Analyze returns-based simulation results
returns_sim = simulation_results['returns_based_simulation']
print(f"\nReturns-Based Simulation Results:")
print(f"Mean Final Equity: {returns_sim['mean_final_equity']:.2f}%")
print(f"95% Confidence Interval: [{returns_sim['confidence_interval']['lower_bound']:.2f}%, "
      f"{returns_sim['confidence_interval']['upper_bound']:.2f}%]")
print(f"Probability of Profit: {returns_sim['probability_of_profit']*100:.2f}%")
print(f"Mean Maximum Drawdown: {returns_sim['drawdown_statistics']['mean_max_drawdown']:.2f}%")
print(f"Worst Drawdown: {returns_sim['drawdown_statistics']['worst_drawdown']:.2f}%")

# Make position sizing decisions based on simulation results
worst_case_drawdown = returns_sim['drawdown_statistics']['worst_drawdown']
max_acceptable_drawdown = -20  # e.g., -20%

if worst_case_drawdown < max_acceptable_drawdown:
    position_size_factor = max_acceptable_drawdown / worst_case_drawdown
    print(f"\nBased on worst-case drawdown analysis, consider reducing position size to "
          f"{position_size_factor:.2f}x current size")
else:
    print("\nCurrent position sizing appears acceptable based on drawdown analysis")

# Save simulation results
with open('monte_carlo_summary.json', 'w') as f:
    json.dump(simulation_results, f, indent=2, default=str)
```

### Command Line Usage

You can also run Monte Carlo simulations from the command line:

```bash
python monte_carlo_simulation.py --backtest-results path/to/backtest_results.json --num-simulations 1000 --confidence-level 0.95
```

## Conclusion

Monte Carlo simulations are a powerful tool for evaluating trading strategy robustness and estimating the range of possible outcomes. By incorporating Monte Carlo analysis into your backtesting workflow, you can make more informed decisions about strategy selection, position sizing, and risk management.

Remember that Monte Carlo simulations are based on historical data and assumptions about the distribution of returns. While they provide valuable insights, they should be used as one of several tools in your strategy evaluation toolkit, alongside out-of-sample testing, forward testing, and fundamental analysis.
