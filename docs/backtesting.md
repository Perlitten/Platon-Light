# Platon Light Backtesting Module Documentation

## Overview

The Platon Light backtesting module provides a comprehensive framework for testing and optimizing trading strategies using historical market data. This module allows traders to validate their strategies before deploying them in live trading environments.

## Architecture

The backtesting module consists of the following components:

### Core Components

1. **BaseStrategy** (`platon_light/backtesting/strategy.py`): Abstract base class that all trading strategies must inherit from. Defines the interface for strategy implementation.

2. **BacktestEngine** (`platon_light/backtesting/backtest_engine.py`): Core engine that handles data loading, strategy execution, trade simulation, and performance calculation.

3. **DataLoader** (`platon_light/backtesting/data_loader.py`): Responsible for loading and preparing historical market data for backtesting.

### Strategies

1. **MovingAverageCrossover** (`platon_light/backtesting/strategies/moving_average_crossover.py`): Implementation of a moving average crossover strategy with optional RSI and Bollinger Bands filters.

### Indicators

The backtesting module leverages the following indicators:

1. **SMA** (`platon_light/indicators/basic/sma.py`): Simple Moving Average calculation.
2. **EMA** (`platon_light/indicators/basic/ema.py`): Exponential Moving Average calculation.
3. **RSI** (`platon_light/indicators/basic/rsi.py`): Relative Strength Index calculation.
4. **Bollinger Bands** (`platon_light/indicators/basic/bollinger_bands.py`): Bollinger Bands calculation.
5. **MACD** (`platon_light/indicators/basic/macd.py`): Moving Average Convergence Divergence calculation.

## Workflow

### 1. Data Preparation

Before running backtests, you need historical market data. You can either:

- Generate synthetic data using `generate_sample_data.py`
- Use real market data by fetching it from exchanges using `launch_live_backtest.py`

Data should be in OHLCV (Open, High, Low, Close, Volume) format with a timestamp index.

### 2. Strategy Implementation

To implement a new strategy:

1. Create a new Python file in the `platon_light/backtesting/strategies/` directory
2. Define a class that inherits from `BaseStrategy`
3. Implement the required methods:
   - `prepare_data(self, data)`: Add indicators and prepare data for signal generation
   - `generate_signals(self, data)`: Generate buy/sell signals based on the prepared data

Example:

```python
from platon_light.backtesting.strategy import BaseStrategy
from platon_light.indicators.basic.sma import SMA

class MySMAStrategy(BaseStrategy):
    def __init__(self, period=20):
        self.period = period
        
    def prepare_data(self, data):
        # Add SMA indicator
        sma = SMA(self.period)
        data['sma'] = sma.calculate(data['close'])
        return data
        
    def generate_signals(self, data):
        # Generate signals
        data['signal'] = 0
        data.loc[data['close'] > data['sma'], 'signal'] = 1  # Buy signal
        data.loc[data['close'] < data['sma'], 'signal'] = -1  # Sell signal
        return data
```

### 3. Running Backtests

#### Basic Backtest

Use `run_backtest.py` to run a basic backtest with default parameters:

```bash
python run_backtest.py
```

#### Parameter Optimization

Use `optimize_strategy_parameters.py` to find optimal parameters for your strategy:

```bash
python optimize_strategy_parameters.py
```

This script performs a grid search over specified parameter ranges and saves the best parameter combinations.

#### Comprehensive Backtest

Use `run_comprehensive_backtest.py` to run backtests with optimized parameters across multiple symbols and timeframes:

```bash
python run_comprehensive_backtest.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h 1d
```

### 4. Performance Analysis

Use `analyze_backtest_performance.py` to analyze the results of your backtests:

```bash
python analyze_backtest_performance.py
```

This script calculates various performance metrics, including:

- Total return
- Annualized return
- Maximum drawdown
- Sharpe ratio
- Win rate
- Profit factor

It also generates visualizations of equity curves, drawdowns, and trade distributions.

## Security

The backtesting module includes secure credential handling for API keys when fetching real market data. Credentials are encrypted using Fernet symmetric encryption and stored in a secure location.

To set up secure credentials:

```bash
python launch_live_backtest.py --setup-credentials
```

## Best Practices

1. **Start with synthetic data**: Use synthetic data for initial testing before moving to real market data.
2. **Validate indicators**: Ensure that your indicators are correctly implemented and validated against known reference implementations.
3. **Avoid overfitting**: Be cautious of overfitting your strategy to historical data. Use out-of-sample testing to validate performance.
4. **Consider transaction costs**: Include realistic transaction costs in your backtests to get more accurate performance estimates.
5. **Use multiple timeframes**: Test your strategy across multiple timeframes to ensure robustness.
6. **Test with different market conditions**: Validate your strategy in different market conditions (trending, ranging, volatile).

## Extending the Framework

### Adding New Indicators

To add a new indicator:

1. Create a new Python file in the `platon_light/indicators/` directory
2. Implement the indicator calculation logic
3. Add tests to validate the indicator implementation

### Adding New Strategies

To add a new strategy:

1. Create a new Python file in the `platon_light/backtesting/strategies/` directory
2. Implement the strategy logic as described in the "Strategy Implementation" section
3. Update the `platon_light/backtesting/strategies/__init__.py` file to include your new strategy

### Customizing Performance Metrics

To add custom performance metrics:

1. Modify the `calculate_performance_metrics` function in `analyze_backtest_performance.py`
2. Add your custom metric calculation
3. Update the visualization and reporting functions to include your new metric

## Troubleshooting

### Common Issues

1. **NaN values in indicators**: Ensure that your data has sufficient history for indicator calculation. Some indicators require a minimum amount of data points.
2. **No signals generated**: Check your signal generation logic and ensure that the conditions for generating signals are being met.
3. **Poor performance**: Review your strategy logic and consider parameter optimization.

### Debugging Tips

1. Use logging to track the execution of your strategy
2. Visualize intermediate results to understand how your strategy is behaving
3. Test with a small subset of data first before running full backtests

## Future Enhancements

Planned enhancements for the backtesting module include:

1. **Monte Carlo simulation**: Implement Monte Carlo simulation to estimate the distribution of possible outcomes.
2. **Walk-forward optimization**: Implement walk-forward optimization to reduce overfitting.
3. **Machine learning integration**: Add support for machine learning models in strategy development.
4. **Portfolio backtesting**: Extend the framework to support multi-asset portfolio backtesting.
5. **Event-driven backtesting**: Implement event-driven backtesting for more realistic simulation of market conditions.

## References

- [Investopedia - Backtesting](https://www.investopedia.com/terms/b/backtesting.asp)
- [Quantitative Trading: How to Build Your Own Algorithmic Trading Business](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/1119800064)
- [Python for Finance: Mastering Data-Driven Finance](https://www.amazon.com/Python-Finance-Mastering-Data-Driven/dp/1492024333)
