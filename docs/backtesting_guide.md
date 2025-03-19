# Platon Light Backtesting Guide

## Introduction

The Platon Light backtesting module provides a comprehensive framework for testing and optimizing trading strategies using historical market data. This guide explains how to use the backtesting module effectively to evaluate strategy performance before deploying in a live trading environment.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Running a Backtest](#running-a-backtest)
4. [Analyzing Results](#analyzing-results)
5. [Visualizing Results](#visualizing-results)
6. [Optimizing Strategies](#optimizing-strategies)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using the backtesting module, ensure you have:

- Python 3.8 or higher
- Required dependencies installed (see `requirements.txt`)
- Historical market data or access to an exchange API

### Installation

The backtesting module is included in the Platon Light package. No additional installation is required.

### Directory Structure

```
platon_light/
├── backtesting/
│   ├── __init__.py
│   ├── data_loader.py         # Historical data loading
│   ├── backtest_engine.py     # Backtesting simulation engine
│   ├── performance_analyzer.py # Performance analysis
│   ├── visualization.py       # Results visualization
│   ├── optimization.py        # Parameter optimization
│   └── cli.py                 # Command-line interface
```

## Configuration

### Configuration File

Create a backtesting configuration file based on the `backtest_config.example.yaml` template. This file contains settings for:

- Data sources
- Simulation parameters
- Strategy parameters
- Visualization options
- Optimization settings

Example configuration:

```yaml
# General settings
general:
  mode: "backtest"
  exchange: "binance"

# Trading parameters
trading:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframes: ["1m", "5m", "15m"]
  
# Strategy configuration
strategy:
  name: "scalping_rsi"
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70

# Backtesting specific configuration
backtesting:
  data_source: "binance"
  initial_capital: 10000.0
  commission: 0.04
  slippage: 0.01
```

### Environment Variables

You can override configuration settings using environment variables:

```
PLATON_INITIAL_CAPITAL=20000
PLATON_COMMISSION=0.05
```

## Running a Backtest

### Using the Command-Line Interface

The simplest way to run a backtest is using the command-line interface:

```bash
python -m platon_light.backtesting.cli --config backtest_config.yaml --symbol BTCUSDT --timeframe 1m --start-date 2023-01-01 --end-date 2023-01-31
```

#### Available CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Path to configuration file | `--config backtest_config.yaml` |
| `--symbol` | Trading pair symbol | `--symbol BTCUSDT` |
| `--timeframe` | Timeframe | `--timeframe 1m` |
| `--start-date` | Start date (YYYY-MM-DD) | `--start-date 2023-01-01` |
| `--end-date` | End date (YYYY-MM-DD) | `--end-date 2023-01-31` |
| `--strategy` | Strategy name | `--strategy scalping_rsi` |
| `--strategy-params` | Strategy parameters (JSON) | `--strategy-params '{"rsi_period": 14}'` |
| `--commission` | Commission rate | `--commission 0.04` |
| `--slippage` | Slippage rate | `--slippage 0.01` |
| `--initial-capital` | Initial capital | `--initial-capital 10000` |
| `--output-dir` | Output directory | `--output-dir results/` |
| `--report` | Generate HTML report | `--report` |
| `--plot` | Generate plots | `--plot` |
| `--download-data` | Download historical data | `--download-data` |
| `--use-cache` | Use cached data | `--use-cache` |
| `--compare` | Compare multiple strategies | `--compare` |
| `--strategies` | Strategies to compare | `--strategies "strategy1,strategy2"` |
| `--log-level` | Logging level | `--log-level DEBUG` |

### Using the Python API

You can also run backtests programmatically:

```python
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.utils.config_manager import ConfigManager
from datetime import datetime

# Load configuration
config_manager = ConfigManager("backtest_config.yaml")
config = config_manager.get_config()

# Initialize components
data_loader = DataLoader(config)
backtest_engine = BacktestEngine(config)
performance_analyzer = PerformanceAnalyzer(config)

# Define backtest parameters
symbol = "BTCUSDT"
timeframe = "1m"
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 31)

# Run backtest
results = backtest_engine.run(symbol, timeframe, start_date, end_date)

# Analyze results
analysis = performance_analyzer.analyze(results)

# Print metrics
print(f"Total Return: {analysis['metrics']['return_percent']:.2f}%")
print(f"Sharpe Ratio: {analysis['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {analysis['metrics']['max_drawdown_percent']:.2f}%")
print(f"Win Rate: {analysis['metrics']['win_rate']:.2f}%")
```

## Analyzing Results

### Performance Metrics

The backtesting module calculates the following performance metrics:

| Metric | Description |
|--------|-------------|
| Return Percentage | Total return as a percentage of initial capital |
| Sharpe Ratio | Risk-adjusted return measure |
| Sortino Ratio | Downside risk-adjusted return measure |
| Max Drawdown | Maximum peak-to-trough decline |
| Win Rate | Percentage of winning trades |
| Profit Factor | Ratio of gross profit to gross loss |
| Calmar Ratio | Return divided by maximum drawdown |
| Average Trade | Average profit/loss per trade |
| Average Win/Loss Ratio | Ratio of average winning trade to average losing trade |
| Expectancy | Expected return per dollar risked |

### Trade Analysis

The `performance_analyzer.py` module provides detailed trade analysis:

```python
# Get trade statistics
trade_stats = performance_analyzer.get_trade_statistics(results)

# Print trade statistics
print(f"Total Trades: {trade_stats['total_trades']}")
print(f"Winning Trades: {trade_stats['winning_trades']}")
print(f"Losing Trades: {trade_stats['losing_trades']}")
print(f"Average Win: {trade_stats['average_win']:.2f}%")
print(f"Average Loss: {trade_stats['average_loss']:.2f}%")
print(f"Largest Win: {trade_stats['largest_win']:.2f}%")
print(f"Largest Loss: {trade_stats['largest_loss']:.2f}%")
print(f"Average Trade Duration: {trade_stats['average_duration']} minutes")
```

## Visualizing Results

### Available Visualizations

The `visualization.py` module provides the following visualizations:

1. **Equity Curve**: Shows the growth of capital over time
2. **Drawdown Chart**: Shows the drawdowns over time
3. **Trade Distribution**: Shows the distribution of trade profits
4. **Monthly Returns**: Shows the monthly returns as a heatmap
5. **Performance Metrics**: Shows key performance metrics
6. **Trade Analysis**: Shows trade-specific analysis charts

### Generating Visualizations

```python
from platon_light.backtesting.visualization import BacktestVisualizer

# Initialize visualizer
visualizer = BacktestVisualizer(config)

# Generate visualizations
visualizer.plot_equity_curve(results)
visualizer.plot_drawdown(results)
visualizer.plot_trade_distribution(results)
visualizer.plot_monthly_returns(results)
visualizer.plot_performance_metrics(results)
visualizer.plot_trade_analysis(results)

# Generate HTML report with all visualizations
report_path = visualizer.create_report(results)
print(f"Report generated at: {report_path}")
```

### HTML Reports

The backtesting module can generate comprehensive HTML reports with all visualizations and metrics:

```bash
python -m platon_light.backtesting.cli --config backtest_config.yaml --symbol BTCUSDT --timeframe 1m --start-date 2023-01-01 --end-date 2023-01-31 --report
```

## Optimizing Strategies

### Optimization Methods

The backtesting module supports the following optimization methods:

1. **Grid Search**: Exhaustive search over specified parameter values
2. **Genetic Algorithm**: Evolutionary optimization technique
3. **Walk-Forward Optimization**: Time-series optimization with out-of-sample validation

### Grid Search Optimization

```python
from platon_light.backtesting.optimization import StrategyOptimizer
from datetime import datetime

# Initialize optimizer
optimizer = StrategyOptimizer(config)

# Define parameter grid
param_grid = {
    "rsi_period": [7, 14, 21],
    "rsi_oversold": [20, 25, 30],
    "rsi_overbought": [70, 75, 80]
}

# Define backtest parameters
symbol = "BTCUSDT"
timeframe = "1m"
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 31)

# Run grid search
results = optimizer.grid_search(param_grid, symbol, timeframe, start_date, end_date)

# Print best parameters
print(f"Best Parameters: {results['best_params']}")
print(f"Best Metrics: {results['best_metrics']}")
```

### Genetic Algorithm Optimization

```python
# Define parameter ranges
param_ranges = {
    "rsi_period": [5, 30, 1],  # [min, max, step]
    "rsi_oversold": [10, 40, 5],
    "rsi_overbought": [60, 90, 5]
}

# Run genetic algorithm
results = optimizer.genetic_algorithm(
    param_ranges, 
    symbol, 
    timeframe, 
    start_date, 
    end_date,
    population_size=50,
    generations=10,
    mutation_rate=0.1,
    crossover_rate=0.7
)

# Print best parameters
print(f"Best Parameters: {results['best_params']}")
print(f"Best Metrics: {results['best_metrics']}")
```

### Walk-Forward Optimization

```python
# Run walk-forward optimization
results = optimizer.walk_forward_optimization(
    param_grid,
    symbol,
    timeframe,
    start_date,
    end_date,
    window_size=30,  # Size of in-sample window in days
    step_size=7      # Size of out-of-sample window in days
)

# Print results
for window in results['windows']:
    print(f"Window {window['window']}")
    print(f"  In-Sample: {window['in_sample']['start']} to {window['in_sample']['end']}")
    print(f"  Out-Sample: {window['out_sample']['start']} to {window['out_sample']['end']}")
    print(f"  Best Parameters: {window['in_sample']['best_params']}")
    print(f"  In-Sample Return: {window['in_sample']['metrics']['return_percent']:.2f}%")
    print(f"  Out-Sample Return: {window['out_sample']['metrics']['return_percent']:.2f}%")
```

## Advanced Features

### Multi-Timeframe Analysis

The backtesting module supports multi-timeframe analysis:

```python
# Load data for multiple timeframes
data_1m = data_loader.load_data(symbol, "1m", start_date, end_date)
data_5m = data_loader.load_data(symbol, "5m", start_date, end_date)
data_15m = data_loader.load_data(symbol, "15m", start_date, end_date)

# Combine data
combined_data = data_loader.combine_timeframes(data_1m, data_5m, data_15m)

# Run backtest with combined data
results = backtest_engine.run_multi_timeframe(symbol, combined_data, start_date, end_date)
```

### Multi-Asset Backtesting

The backtesting module supports multi-asset backtesting:

```python
# Define symbols
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Run backtest for each symbol
all_results = {}
for symbol in symbols:
    results = backtest_engine.run(symbol, timeframe, start_date, end_date)
    all_results[symbol] = results

# Analyze combined results
combined_analysis = performance_analyzer.analyze_multi_asset(all_results)
```

### Custom Strategies

You can implement custom strategies by extending the base `Strategy` class:

```python
from platon_light.core.strategy import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        # Initialize strategy parameters
        
    def generate_signals(self, data):
        # Generate buy/sell signals
        signals = []
        
        # Your strategy logic here
        
        return signals
```

## Best Practices

### Data Quality

- **Sufficient History**: Ensure you have sufficient historical data for your strategy's lookback period.
- **Data Integrity**: Check for gaps, outliers, and other data quality issues.
- **Realistic Prices**: Use realistic prices that account for liquidity and market impact.

### Avoiding Overfitting

- **Out-of-Sample Testing**: Always validate your strategy on out-of-sample data.
- **Walk-Forward Analysis**: Use walk-forward analysis to simulate real-world performance.
- **Parameter Sensitivity**: Test your strategy with slightly different parameters to ensure robustness.
- **Multiple Assets**: Test your strategy on multiple assets to ensure generalizability.
- **Multiple Timeframes**: Test your strategy on multiple timeframes to ensure consistency.

### Realistic Simulation

- **Commission and Slippage**: Include realistic commission and slippage rates.
- **Execution Latency**: Simulate realistic execution latency.
- **Market Impact**: Consider market impact for larger positions.
- **Liquidity Constraints**: Consider liquidity constraints for less liquid markets.

### Risk Management

- **Position Sizing**: Use proper position sizing based on risk tolerance.
- **Stop Losses**: Implement and test stop loss strategies.
- **Correlation**: Consider correlation between assets in multi-asset strategies.
- **Drawdown Limits**: Implement drawdown limits to control risk.

## Troubleshooting

### Common Issues

#### No Data Available

If you encounter a "No data available" error:

1. Check that the symbol and timeframe are valid
2. Verify that data is available for the specified date range
3. Check your data source configuration
4. Try downloading data explicitly with the `--download-data` option

#### Strategy Not Found

If you encounter a "Strategy not found" error:

1. Check that the strategy name is correct in your configuration
2. Verify that the strategy class is properly implemented and registered

#### Performance Issues

If backtesting is slow:

1. Use cached data with the `--use-cache` option
2. Reduce the date range or number of symbols
3. Simplify your strategy logic
4. Use a more powerful machine

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs for detailed error messages
2. Review the source code documentation
3. Open an issue on the project repository

## Conclusion

The Platon Light backtesting module provides a comprehensive framework for testing and optimizing trading strategies. By following this guide, you can effectively evaluate your strategies before deploying them in a live trading environment.

Remember that past performance is not indicative of future results. Always validate your strategies thoroughly and implement proper risk management.
