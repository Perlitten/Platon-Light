# Trading Strategies

This document provides an overview of the trading strategies available in Platon Light and explains how to configure and use them.

## Available Strategies

### Moving Average Crossover

The Moving Average Crossover strategy is a trend-following strategy that generates buy and sell signals based on the crossover of fast and slow moving averages.

#### Parameters

- `fast_ma_type`: Type of fast moving average ('SMA' or 'EMA')
- `slow_ma_type`: Type of slow moving average ('SMA' or 'EMA')
- `fast_period`: Period for fast moving average
- `slow_period`: Period for slow moving average
- `rsi_period`: Period for RSI calculation
- `rsi_overbought`: RSI overbought threshold
- `rsi_oversold`: RSI oversold threshold
- `use_filters`: Whether to use additional filters (RSI, Bollinger Bands)

#### Signal Logic

- **Buy Signal**: Generated when the fast moving average crosses above the slow moving average
- **Sell Signal**: Generated when the fast moving average crosses below the slow moving average

#### Filters

When `use_filters` is enabled, the strategy applies additional filters to reduce false signals:

- **RSI Filter**: Buy signals are only considered when RSI is below the oversold threshold, and sell signals are only considered when RSI is above the overbought threshold
- **Bollinger Bands Filter**: Buy signals are strengthened when price is near the lower Bollinger Band, and sell signals are strengthened when price is near the upper Bollinger Band

#### Configuration Example

```yaml
strategy:
  name: MovingAverageCrossover
  params:
    fast_ma_type: EMA
    slow_ma_type: SMA
    fast_period: 9
    slow_period: 21
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    use_filters: true
```

### Custom Strategies

You can create custom strategies by inheriting from the `BaseStrategy` class and implementing the required methods.

## Creating a Custom Strategy

To create a custom strategy, follow these steps:

1. Create a new Python file in the `platon_light/backtesting/strategies` directory
2. Import the `BaseStrategy` class
3. Create a new class that inherits from `BaseStrategy`
4. Implement the required methods: `prepare_data` and `generate_signals`

### Example Custom Strategy

```python
from platon_light.backtesting.strategy import BaseStrategy
import pandas as pd
import numpy as np

class MyCustomStrategy(BaseStrategy):
    """
    A custom trading strategy example.
    """
    
    def __init__(self, param1=10, param2=20):
        """
        Initialize the strategy with custom parameters.
        
        Args:
            param1 (int): First parameter
            param2 (int): Second parameter
        """
        self.param1 = param1
        self.param2 = param2
        
    def prepare_data(self, data):
        """
        Prepare data by adding indicators.
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Data with added indicators
        """
        # Add your indicators here
        data['indicator1'] = self._calculate_indicator1(data, self.param1)
        data['indicator2'] = self._calculate_indicator2(data, self.param2)
        
        return data
        
    def generate_signals(self, data):
        """
        Generate buy and sell signals.
        
        Args:
            data (pd.DataFrame): Prepared data with indicators
            
        Returns:
            pd.DataFrame: Data with added 'signal' column (1 for buy, -1 for sell, 0 for hold)
        """
        # Initialize signals
        data['signal'] = 0
        
        # Generate buy signals
        buy_condition = (data['indicator1'] > data['indicator2'])
        data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals
        sell_condition = (data['indicator1'] < data['indicator2'])
        data.loc[sell_condition, 'signal'] = -1
        
        return data
        
    def _calculate_indicator1(self, data, period):
        """
        Calculate the first indicator.
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Indicator period
            
        Returns:
            pd.Series: Calculated indicator values
        """
        # Example: Simple Moving Average
        return data['close'].rolling(window=period).mean()
        
    def _calculate_indicator2(self, data, period):
        """
        Calculate the second indicator.
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Indicator period
            
        Returns:
            pd.Series: Calculated indicator values
        """
        # Example: Exponential Moving Average
        return data['close'].ewm(span=period, adjust=False).mean()
```

## Strategy Optimization

Platon Light provides tools for optimizing strategy parameters through the backtesting module. See the [Strategy Optimization Guide](strategy_optimization_guide.md) for more details.

## Strategy Performance Metrics

When evaluating a strategy, Platon Light calculates the following performance metrics:

- **Total Return**: Overall percentage return
- **Annualized Return**: Return normalized to a yearly basis
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Maximum observed loss from a peak to a trough
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Average Win**: Average profit of winning trades
- **Average Loss**: Average loss of losing trades
- **Risk-Reward Ratio**: Ratio of average win to average loss

## Best Practices

1. **Test thoroughly**: Always backtest your strategy on different market conditions and timeframes
2. **Start simple**: Begin with simple strategies and gradually add complexity
3. **Avoid overfitting**: Be cautious of strategies that perform exceptionally well on historical data but may fail in live trading
4. **Consider transaction costs**: Include realistic transaction costs in your backtests
5. **Use proper risk management**: Even the best strategy needs proper risk management
6. **Monitor and adapt**: Regularly review your strategy's performance and be prepared to adapt to changing market conditions
