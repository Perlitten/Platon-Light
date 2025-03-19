# Backtesting Module

## Overview

The backtesting module provides a comprehensive framework for testing trading strategies using historical data. It allows users to evaluate strategy performance, optimize parameters, and visualize results before deploying strategies in a live trading environment.

## Components

The backtesting module consists of the following components:

1. **Data Loader (`data_loader.py`)**: Loads historical market data from various sources.
2. **Backtest Engine (`backtest_engine.py`)**: Simulates trading strategies on historical data.
3. **Performance Analyzer (`performance_analyzer.py`)**: Analyzes backtest results and calculates performance metrics.
4. **Visualization (`visualization.py`)**: Creates visualizations of backtest results.
5. **Optimization (`optimization.py`)**: Optimizes strategy parameters using various techniques.
6. **Command-Line Interface (`cli.py`)**: Provides a command-line interface for running backtests.

## Usage

### Basic Usage

To run a backtest using the command-line interface:

```bash
python -m platon_light.backtesting.cli --config backtest_config.yaml --symbol BTCUSDT --timeframe 1m --start-date 2023-01-01 --end-date 2023-01-31
```

### Configuration

Create a configuration file based on the `backtest_config.example.yaml` template. This file contains settings for data sources, simulation parameters, strategy parameters, and visualization options.

### Optimization

To optimize strategy parameters:

```bash
python -m platon_light.backtesting.cli --config backtest_config.yaml --symbol BTCUSDT --timeframe 1m --start-date 2023-01-01 --end-date 2023-01-31 --optimize
```

### Comparing Strategies

To compare multiple strategies:

```bash
python -m platon_light.backtesting.cli --config backtest_config.yaml --symbol BTCUSDT --timeframe 1m --start-date 2023-01-01 --end-date 2023-01-31 --compare --strategies "strategy1,strategy2,strategy3"
```

## Directory Structure

```
backtesting/
├── __init__.py
├── data_loader.py         # Historical data loading
├── backtest_engine.py     # Backtesting simulation engine
├── performance_analyzer.py # Performance analysis
├── visualization.py       # Results visualization
├── optimization.py        # Parameter optimization
├── cli.py                 # Command-line interface
└── README.md              # This documentation
```

## Performance Metrics

The backtesting module calculates the following performance metrics:

- **Return Percentage**: Total return as a percentage of initial capital
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return measure
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Calmar Ratio**: Return divided by maximum drawdown
- **Average Trade**: Average profit/loss per trade
- **Average Win/Loss Ratio**: Ratio of average winning trade to average losing trade
- **Expectancy**: Expected return per dollar risked

## Optimization Methods

The backtesting module supports the following optimization methods:

1. **Grid Search**: Exhaustive search over specified parameter values
2. **Genetic Algorithm**: Evolutionary optimization technique
3. **Walk-Forward Optimization**: Time-series optimization with out-of-sample validation

## Visualization Options

The visualization module provides the following charts:

- Equity curve
- Drawdown chart
- Trade distribution
- Monthly returns heatmap
- Performance metrics comparison
- Trade analysis charts
- Strategy comparison charts

## Example

```python
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.utils.config_manager import ConfigManager
from datetime import datetime

# Load configuration
config_manager = ConfigManager("backtest_config.yaml")
config = config_manager.get_config()

# Initialize components
data_loader = DataLoader(config)
backtest_engine = BacktestEngine(config)
performance_analyzer = PerformanceAnalyzer(config)
visualizer = BacktestVisualizer(config)

# Define backtest parameters
symbol = "BTCUSDT"
timeframe = "1m"
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 31)

# Run backtest
results = backtest_engine.run(symbol, timeframe, start_date, end_date)

# Analyze results
analysis = performance_analyzer.analyze(results)

# Visualize results
visualizer.plot_equity_curve(results)
visualizer.plot_drawdown(results)
visualizer.plot_trade_distribution(results)

# Generate report
report_path = visualizer.create_report(results)
print(f"Report generated at: {report_path}")
```

## Best Practices

1. **Data Quality**: Ensure historical data is accurate and includes sufficient history for your strategy.
2. **Avoid Overfitting**: Be cautious of overfitting when optimizing parameters. Use walk-forward testing and out-of-sample validation.
3. **Realistic Simulation**: Configure realistic commission rates, slippage, and execution latency.
4. **Multiple Timeframes**: Test strategies across multiple timeframes to ensure robustness.
5. **Multiple Symbols**: Test strategies across multiple symbols to ensure generalizability.
6. **Stress Testing**: Test strategies under various market conditions, including high volatility periods.
