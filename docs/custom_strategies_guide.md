# Creating Custom Strategies for Platon Light Backtesting

This guide explains how to create and implement custom trading strategies for the Platon Light backtesting module.

## Table of Contents

1. [Strategy Architecture](#strategy-architecture)
2. [Creating a Basic Strategy](#creating-a-basic-strategy)
3. [Advanced Strategy Development](#advanced-strategy-development)
4. [Testing Your Strategy](#testing-your-strategy)
5. [Strategy Registration](#strategy-registration)
6. [Best Practices](#best-practices)

## Strategy Architecture

In Platon Light, all trading strategies inherit from the base `Strategy` class. This architecture provides a consistent interface for the backtesting engine to interact with different strategies.

### Strategy Class Hierarchy

```
Strategy (Base Class)
├── ScalpingStrategy
│   ├── RSIScalpingStrategy
│   ├── MAScalpingStrategy
│   └── BBScalpingStrategy
├── TrendFollowingStrategy
│   ├── MACrossStrategy
│   └── ADXTrendStrategy
└── MeanReversionStrategy
    ├── RSIMeanReversionStrategy
    └── BollingerBandReversionStrategy
```

### Core Strategy Components

Every strategy consists of these key components:

1. **Signal Generation**: Logic to identify entry and exit points
2. **Risk Management**: Position sizing and risk control
3. **Parameter Management**: Handling strategy-specific parameters
4. **Execution Logic**: Rules for executing trades

## Creating a Basic Strategy

### Step 1: Create a New Strategy Class

Create a new Python file in the `platon_light/strategies` directory:

```python
# platon_light/strategies/my_custom_strategy.py

from platon_light.core.strategy import Strategy
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class MyCustomStrategy(Strategy):
    """
    My custom trading strategy
    
    This strategy implements [describe your strategy logic here]
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the strategy
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get strategy-specific parameters from config
        strategy_config = config.get("strategy", {})
        self.param1 = strategy_config.get("param1", 14)  # Default value: 14
        self.param2 = strategy_config.get("param2", 0.5)  # Default value: 0.5
        
        # Initialize any other required variables
        self.logger.info("MyCustomStrategy initialized")
```

### Step 2: Implement Signal Generation

Add the `generate_signals` method to your strategy class:

```python
def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals
    
    Args:
        data: DataFrame with OHLCV data and indicators
        
    Returns:
        DataFrame with added signal columns
    """
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Initialize signal columns
    df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    
    # Implement your signal generation logic here
    # Example: Simple moving average crossover
    df['short_ma'] = df['close'].rolling(window=10).mean()
    df['long_ma'] = df['close'].rolling(window=30).mean()
    
    # Generate buy signals
    buy_condition = (df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1))
    df.loc[buy_condition, 'signal'] = 1
    df.loc[buy_condition, 'entry_price'] = df['close']
    df.loc[buy_condition, 'stop_loss'] = df['close'] * (1 - self.param2)
    df.loc[buy_condition, 'take_profit'] = df['close'] * (1 + self.param2 * 2)
    
    # Generate sell signals
    sell_condition = (df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1))
    df.loc[sell_condition, 'signal'] = -1
    df.loc[sell_condition, 'exit_price'] = df['close']
    
    return df
```

### Step 3: Implement Position Sizing

Add a method to determine position size:

```python
def calculate_position_size(self, capital: float, price: float, risk_per_trade: float) -> float:
    """
    Calculate position size based on risk parameters
    
    Args:
        capital: Available capital
        price: Current price
        risk_per_trade: Risk per trade as a percentage of capital
        
    Returns:
        Position size
    """
    # Get risk parameters
    risk_params = self.config.get("risk_management", {})
    max_position_size = risk_params.get("max_position_size", 0.1)
    
    # Calculate position size based on risk
    risk_amount = capital * risk_per_trade
    position_value = capital * max_position_size
    
    # Return the smaller of the two position sizes
    return min(risk_amount / (price * self.param2), position_value / price)
```

## Advanced Strategy Development

### Adding Technical Indicators

You can add custom technical indicators to your strategy:

```python
def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for the strategy
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    df = data.copy()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=self.param1).mean()
    avg_loss = loss.rolling(window=self.param1).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['sma'] = df['close'].rolling(window=20).mean()
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['sma'] + 2 * df['std']
    df['lower_band'] = df['sma'] - 2 * df['std']
    
    return df
```

### Implementing Advanced Exit Rules

Add sophisticated exit rules to your strategy:

```python
def generate_exit_signals(self, data: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Generate exit signals for an open position
    
    Args:
        data: DataFrame with OHLCV data and indicators
        position: Current position ('long' or 'short')
        
    Returns:
        DataFrame with added exit signal columns
    """
    df = data.copy()
    
    # Initialize exit signal column
    df['exit_signal'] = 0  # 0: no exit, 1: exit
    
    if position == 'long':
        # Exit long position when RSI is overbought
        exit_condition = df['rsi'] > 70
        df.loc[exit_condition, 'exit_signal'] = 1
        df.loc[exit_condition, 'exit_price'] = df['close']
        df.loc[exit_condition, 'exit_reason'] = 'rsi_overbought'
        
    elif position == 'short':
        # Exit short position when RSI is oversold
        exit_condition = df['rsi'] < 30
        df.loc[exit_condition, 'exit_signal'] = 1
        df.loc[exit_condition, 'exit_price'] = df['close']
        df.loc[exit_condition, 'exit_reason'] = 'rsi_oversold'
    
    return df
```

## Testing Your Strategy

### Unit Testing

Create unit tests for your strategy in the `tests/strategies` directory:

```python
# tests/strategies/test_my_custom_strategy.py

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from platon_light.strategies.my_custom_strategy import MyCustomStrategy


class TestMyCustomStrategy(unittest.TestCase):
    
    def setUp(self):
        # Create a simple configuration
        self.config = {
            "strategy": {
                "name": "my_custom_strategy",
                "param1": 14,
                "param2": 0.5
            },
            "risk_management": {
                "max_position_size": 0.1
            }
        }
        
        # Create a sample DataFrame
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
        self.data = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        
        # Initialize the strategy
        self.strategy = MyCustomStrategy(self.config)
    
    def test_generate_signals(self):
        # Test signal generation
        signals = self.strategy.generate_signals(self.data)
        
        # Check if signal columns exist
        self.assertIn('signal', signals.columns)
        self.assertIn('entry_price', signals.columns)
        self.assertIn('exit_price', signals.columns)
        
        # Check if signals are generated
        self.assertTrue((signals['signal'] != 0).any())
    
    def test_calculate_position_size(self):
        # Test position sizing
        position_size = self.strategy.calculate_position_size(10000, 100, 0.01)
        
        # Check if position size is reasonable
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 10)


if __name__ == '__main__':
    unittest.main()
```

### Backtesting

Test your strategy using the backtesting module:

```python
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.strategies.my_custom_strategy import MyCustomStrategy
from datetime import datetime

# Create configuration
config = {
    "strategy": {
        "name": "my_custom_strategy",
        "param1": 14,
        "param2": 0.5
    },
    "backtesting": {
        "initial_capital": 10000,
        "commission": 0.001,
        "slippage": 0.0005
    }
}

# Initialize components
data_loader = DataLoader(config)
backtest_engine = BacktestEngine(config)
performance_analyzer = PerformanceAnalyzer(config)

# Define backtest parameters
symbol = "BTCUSDT"
timeframe = "1h"
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 3, 31)

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

## Strategy Registration

To make your strategy available in the backtesting module, you need to register it in the strategy factory:

### Step 1: Update Strategy Factory

Edit the `platon_light/core/strategy_factory.py` file:

```python
# platon_light/core/strategy_factory.py

from typing import Dict
from platon_light.core.strategy import Strategy
from platon_light.strategies.scalping_strategy import ScalpingStrategy
from platon_light.strategies.my_custom_strategy import MyCustomStrategy


class StrategyFactory:
    """
    Factory class for creating strategy instances
    """
    
    @staticmethod
    def create_strategy(config: Dict) -> Strategy:
        """
        Create a strategy instance based on configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Strategy instance
        """
        strategy_name = config.get("strategy", {}).get("name", "scalping")
        
        if strategy_name == "scalping":
            return ScalpingStrategy(config)
        elif strategy_name == "my_custom_strategy":
            return MyCustomStrategy(config)
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")
```

### Step 2: Update Configuration Template

Add your strategy parameters to the configuration template:

```yaml
# backtest_config.example.yaml

# Strategy configuration
strategy:
  name: "my_custom_strategy"  # Strategy name
  param1: 14                  # Custom parameter 1
  param2: 0.5                 # Custom parameter 2
```

## Best Practices

### 1. Keep Strategies Simple and Focused

Each strategy should have a clear, focused trading logic. Avoid creating overly complex strategies that try to do too much.

### 2. Use Proper Risk Management

Always implement proper risk management in your strategies. This includes:
- Position sizing based on risk
- Stop-loss and take-profit levels
- Maximum drawdown limits

### 3. Document Your Strategy

Include detailed documentation for your strategy:
- Strategy description and logic
- Required parameters and their meanings
- Expected behavior in different market conditions
- Known limitations

### 4. Test Thoroughly

Test your strategy under various market conditions:
- Bull markets
- Bear markets
- Sideways markets
- High volatility periods
- Low volatility periods

### 5. Avoid Overfitting

Be cautious of overfitting your strategy to historical data:
- Use out-of-sample testing
- Implement walk-forward analysis
- Keep the number of parameters reasonable
- Test on multiple symbols and timeframes

### 6. Implement Logging

Add proper logging to your strategy for debugging and analysis:

```python
def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    # ... strategy logic ...
    
    # Log signal generation
    self.logger.info(f"Generated {buy_signals.sum()} buy signals and {sell_signals.sum()} sell signals")
    self.logger.debug(f"Signal generation completed in {time.time() - start_time:.2f} seconds")
    
    return df
```

### 7. Optimize Performance

Optimize your strategy for performance, especially for high-frequency strategies:
- Use vectorized operations instead of loops
- Cache calculations where possible
- Use efficient data structures

## Conclusion

By following this guide, you can create custom trading strategies for the Platon Light backtesting module. Remember to thoroughly test your strategies before deploying them in a live trading environment.
