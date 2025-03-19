# Position Sizing and Risk Management Guide

This guide explains effective position sizing and risk management techniques for trading strategies using the Platon Light backtesting module.

## Table of Contents

1. [Introduction](#introduction)
2. [Position Sizing Techniques](#position-sizing-techniques)
3. [Risk Management Strategies](#risk-management-strategies)
4. [Implementation in Platon Light](#implementation-in-platon-light)
5. [Backtesting Risk Management](#backtesting-risk-management)
6. [Best Practices](#best-practices)
7. [Example Implementation](#example-implementation)

## Introduction

Position sizing and risk management are critical components of successful trading. They determine:

- How much capital to allocate to each trade
- How to manage risk across your portfolio
- When to exit trades to protect capital
- How to adjust position sizes based on market conditions

Proper risk management often makes the difference between a profitable strategy and one that fails, regardless of the quality of entry and exit signals.

## Position Sizing Techniques

### 1. Fixed Position Size

The simplest approach is to use a fixed position size for all trades.

```python
position_size = 100  # Fixed number of units
```

**Pros**: Simple to implement and understand.
**Cons**: Doesn't account for varying risk levels between trades.

### 2. Percentage of Capital

Allocate a fixed percentage of your current capital to each trade.

```python
percentage = 0.02  # 2% of capital
position_size = account_balance * percentage / entry_price
```

**Pros**: Adjusts position size as account grows or shrinks.
**Cons**: Still doesn't account for varying risk levels between trades.

### 3. Volatility-Based Position Sizing

Adjust position size based on market volatility, typically using Average True Range (ATR).

```python
risk_per_trade = account_balance * 0.01  # 1% risk per trade
stop_loss_distance = 2 * atr_value  # 2 ATR stop loss
position_size = risk_per_trade / stop_loss_distance
```

**Pros**: Adapts to market conditions, smaller positions in volatile markets.
**Cons**: Requires accurate volatility measurement.

### 4. Kelly Criterion

Optimize position size based on the probability of winning and the win/loss ratio.

```python
win_probability = 0.55  # 55% win rate
win_loss_ratio = 1.5    # Average win is 1.5x average loss
kelly_percentage = win_probability - ((1 - win_probability) / win_loss_ratio)
position_size = account_balance * kelly_percentage / entry_price
```

**Pros**: Mathematically optimal for long-term capital growth.
**Cons**: Can suggest aggressive position sizes; often used with a fractional multiplier (e.g., half-Kelly).

### 5. Fixed Risk Position Sizing

Risk a fixed percentage of capital on each trade based on stop loss level.

```python
risk_percentage = 0.01  # 1% risk per trade
stop_loss_price = entry_price * 0.95  # 5% below entry price
risk_amount = account_balance * risk_percentage
position_size = risk_amount / (entry_price - stop_loss_price)
```

**Pros**: Consistent risk exposure per trade.
**Cons**: Requires predefined stop loss levels.

## Risk Management Strategies

### 1. Stop Loss Orders

Set predetermined exit points to limit losses on individual trades.

- **Fixed Percentage Stop**: Exit when price moves against you by a fixed percentage
- **ATR-Based Stop**: Exit when price moves against you by a multiple of ATR
- **Support/Resistance Stop**: Place stops beyond key support/resistance levels
- **Time-Based Stop**: Exit if the trade hasn't reached profit target within a specific time

### 2. Take Profit Orders

Set predetermined profit targets to secure gains.

- **Fixed Percentage Target**: Exit when price moves in your favor by a fixed percentage
- **Risk/Reward Ratio**: Set profit target at a multiple of your risk (e.g., 2:1, 3:1)
- **Technical Level Target**: Place targets at key resistance/support levels
- **Trailing Stop**: Lock in profits by moving stop loss as price moves in your favor

### 3. Position Scaling

Manage risk by scaling in or out of positions.

- **Scaling In**: Add to positions as the trade moves in your favor
- **Scaling Out**: Take partial profits at different price levels
- **Pyramiding**: Add to winning positions while maintaining constant risk

### 4. Portfolio-Level Risk Management

Manage risk across your entire portfolio.

- **Correlation Analysis**: Avoid highly correlated positions
- **Sector Exposure Limits**: Limit exposure to any single market sector
- **Maximum Open Positions**: Limit the number of concurrent open positions
- **Maximum Portfolio Heat**: Limit total portfolio risk exposure

### 5. Drawdown Management

Implement rules to protect capital during losing streaks.

- **Reduce Position Size**: Decrease position size after consecutive losses
- **Trading Pause**: Temporarily stop trading after reaching drawdown threshold
- **Strategy Rotation**: Switch to alternative strategies during unfavorable market conditions

## Implementation in Platon Light

The Platon Light backtesting module provides built-in support for position sizing and risk management through the `BacktestEngine` class.

### Configuration

```python
config = {
    'backtesting': {
        'initial_capital': 10000,
        'position_sizing': {
            'method': 'risk_percentage',  # Options: fixed, percentage, volatility, kelly, risk_percentage
            'params': {
                'risk_percentage': 0.01,  # 1% risk per trade
                'stop_loss_type': 'atr',  # Options: fixed, percentage, atr, support_resistance
                'stop_loss_params': {
                    'atr_multiple': 2.0   # 2 ATR stop loss
                }
            }
        },
        'risk_management': {
            'max_open_positions': 5,
            'max_correlation': 0.7,
            'max_sector_exposure': 0.3,
            'max_drawdown': 0.2,          # 20% maximum drawdown
            'trailing_stop': {
                'enabled': True,
                'activation_percentage': 0.02,  # Activate after 2% profit
                'trail_percentage': 0.01        # 1% trailing stop
            }
        }
    }
}
```

### Example Usage

```python
# Create backtest engine with risk management
backtest_engine = BacktestEngine(config)

# Run backtest with risk management
results = backtest_engine.run(strategy, data)
```

## Backtesting Risk Management

When backtesting with risk management, consider the following:

### 1. Realistic Simulation

Ensure your backtesting engine accurately simulates:

- **Slippage**: Price movement between signal generation and execution
- **Liquidity**: Availability of shares/contracts at desired price
- **Commission**: Trading costs that impact overall profitability
- **Spread**: Difference between bid and ask prices

### 2. Risk Metrics to Monitor

Track these key risk metrics during backtesting:

- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at a given confidence level
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return relative to maximum drawdown
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit divided by gross loss
- **Recovery Factor**: Net profit divided by maximum drawdown

### 3. Stress Testing

Test your strategy under extreme conditions:

- **Historical Crashes**: Test performance during known market crashes
- **Volatility Shocks**: Simulate sudden increases in volatility
- **Liquidity Crises**: Simulate periods of reduced liquidity
- **Correlation Breakdowns**: Test when normal correlations fail
- **Black Swan Events**: Simulate extreme, unexpected market moves

## Best Practices

1. **Start Conservative**: Begin with smaller position sizes than mathematically optimal
2. **Consistency is Key**: Apply risk management rules consistently
3. **Diversify Risk Management**: Use multiple risk management techniques
4. **Adapt to Market Conditions**: Adjust position sizes based on volatility and market regime
5. **Regular Review**: Periodically review and adjust risk parameters
6. **Psychological Factors**: Choose risk levels you can stick with emotionally
7. **Account for Correlation**: Reduce position sizes for correlated assets
8. **Plan for Worst Case**: Design for survival in extreme market conditions
9. **Document Rules**: Create a written risk management plan
10. **Automate When Possible**: Remove emotion from risk management decisions

## Example Implementation

Here's a comprehensive example implementing position sizing and risk management in Platon Light:

```python
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.core.strategy_factory import StrategyFactory
from platon_light.core.base_strategy import BaseStrategy


class RiskManagedStrategy(BaseStrategy):
    """
    Example strategy with integrated risk management.
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Strategy parameters
        strategy_config = config.get('strategy', {})
        self.rsi_period = strategy_config.get('rsi_period', 14)
        self.rsi_overbought = strategy_config.get('rsi_overbought', 70)
        self.rsi_oversold = strategy_config.get('rsi_oversold', 30)
        
        # Risk management parameters
        risk_config = config.get('backtesting', {}).get('risk_management', {})
        self.use_trailing_stop = risk_config.get('trailing_stop', {}).get('enabled', False)
        self.trail_activation = risk_config.get('trailing_stop', {}).get('activation_percentage', 0.02)
        self.trail_percentage = risk_config.get('trailing_stop', {}).get('trail_percentage', 0.01)
        
        # Position sizing parameters
        sizing_config = config.get('backtesting', {}).get('position_sizing', {})
        self.position_sizing_method = sizing_config.get('method', 'fixed')
        self.position_sizing_params = sizing_config.get('params', {})
        
    def calculate_position_size(self, data, current_index, account_balance, entry_price):
        """
        Calculate position size based on configured method.
        """
        if self.position_sizing_method == 'fixed':
            return self.position_sizing_params.get('size', 1.0)
            
        elif self.position_sizing_method == 'percentage':
            percentage = self.position_sizing_params.get('percentage', 0.02)
            return account_balance * percentage / entry_price
            
        elif self.position_sizing_method == 'volatility':
            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[current_index]
            
            risk_percentage = self.position_sizing_params.get('risk_percentage', 0.01)
            atr_multiple = self.position_sizing_params.get('atr_multiple', 2.0)
            
            risk_amount = account_balance * risk_percentage
            stop_distance = atr * atr_multiple
            
            return risk_amount / stop_distance if stop_distance > 0 else 0
            
        elif self.position_sizing_method == 'risk_percentage':
            risk_percentage = self.position_sizing_params.get('risk_percentage', 0.01)
            stop_type = self.position_sizing_params.get('stop_loss_type', 'percentage')
            
            if stop_type == 'percentage':
                stop_percentage = self.position_sizing_params.get('stop_loss_params', {}).get('percentage', 0.05)
                stop_distance = entry_price * stop_percentage
            elif stop_type == 'atr':
                # Calculate ATR as above
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift(1))
                low_close = np.abs(data['low'] - data['close'].shift(1))
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                atr = true_range.rolling(14).mean().iloc[current_index]
                
                atr_multiple = self.position_sizing_params.get('stop_loss_params', {}).get('atr_multiple', 2.0)
                stop_distance = atr * atr_multiple
            else:
                # Default to 5% stop
                stop_distance = entry_price * 0.05
            
            risk_amount = account_balance * risk_percentage
            return risk_amount / stop_distance if stop_distance > 0 else 0
            
        else:
            # Default to fixed size
            return 1.0
    
    def calculate_stop_loss(self, data, current_index, entry_price, position_type):
        """
        Calculate stop loss price based on configured method.
        """
        stop_type = self.position_sizing_params.get('stop_loss_type', 'percentage')
        
        if stop_type == 'percentage':
            stop_percentage = self.position_sizing_params.get('stop_loss_params', {}).get('percentage', 0.05)
            if position_type == 'long':
                return entry_price * (1 - stop_percentage)
            else:  # short
                return entry_price * (1 + stop_percentage)
                
        elif stop_type == 'atr':
            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[current_index]
            
            atr_multiple = self.position_sizing_params.get('stop_loss_params', {}).get('atr_multiple', 2.0)
            if position_type == 'long':
                return entry_price - (atr * atr_multiple)
            else:  # short
                return entry_price + (atr * atr_multiple)
                
        elif stop_type == 'support_resistance':
            # This would require additional logic to identify support/resistance levels
            # For simplicity, we'll use a default percentage stop
            if position_type == 'long':
                return entry_price * 0.95  # 5% below entry
            else:  # short
                return entry_price * 1.05  # 5% above entry
                
        else:
            # Default to 5% stop
            if position_type == 'long':
                return entry_price * 0.95
            else:  # short
                return entry_price * 1.05
    
    def calculate_take_profit(self, entry_price, stop_loss, position_type):
        """
        Calculate take profit price based on risk-reward ratio.
        """
        risk = abs(entry_price - stop_loss)
        risk_reward_ratio = self.position_sizing_params.get('risk_reward_ratio', 2.0)
        
        if position_type == 'long':
            return entry_price + (risk * risk_reward_ratio)
        else:  # short
            return entry_price - (risk * risk_reward_ratio)
    
    def update_trailing_stop(self, current_price, highest_price, lowest_price, 
                             current_stop, position_type):
        """
        Update trailing stop if needed.
        """
        if not self.use_trailing_stop:
            return current_stop
            
        if position_type == 'long':
            # Check if price has moved enough to activate trailing stop
            if (highest_price / current_stop - 1) >= self.trail_activation:
                new_stop = current_price * (1 - self.trail_percentage)
                return max(new_stop, current_stop)
            return current_stop
            
        else:  # short
            # Check if price has moved enough to activate trailing stop
            if (current_stop / lowest_price - 1) >= self.trail_activation:
                new_stop = current_price * (1 + self.trail_percentage)
                return min(new_stop, current_stop)
            return current_stop
    
    def generate_signals(self, data):
        """
        Generate trading signals with integrated risk management.
        """
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize columns
        data['signal'] = 0
        data['position_size'] = 0
        data['stop_loss'] = 0
        data['take_profit'] = 0
        
        # Track account balance for position sizing
        account_balance = self.config.get('backtesting', {}).get('initial_capital', 10000)
        
        # Track highest/lowest prices for trailing stops
        highest_price = data['close'][0]
        lowest_price = data['close'][0]
        
        # Current position tracking
        in_position = False
        position_type = None
        entry_price = 0
        current_stop = 0
        
        for i in range(1, len(data)):
            # Update highest/lowest prices
            highest_price = max(highest_price, data['close'][i])
            lowest_price = min(lowest_price, data['close'][i])
            
            # Check for exit signals if in position
            if in_position:
                # Update trailing stop if enabled
                current_stop = self.update_trailing_stop(
                    data['close'][i], highest_price, lowest_price, 
                    current_stop, position_type
                )
                
                # Check if stop loss or take profit hit
                if position_type == 'long':
                    if data['low'][i] <= current_stop:
                        # Stop loss hit
                        data.loc[data.index[i], 'signal'] = -1
                        in_position = False
                        
                        # Update account balance (simplified)
                        price_change = current_stop / entry_price - 1
                        account_balance *= (1 + price_change)
                        
                    elif data['high'][i] >= data.loc[data.index[i-1], 'take_profit']:
                        # Take profit hit
                        data.loc[data.index[i], 'signal'] = -1
                        in_position = False
                        
                        # Update account balance (simplified)
                        price_change = data.loc[data.index[i-1], 'take_profit'] / entry_price - 1
                        account_balance *= (1 + price_change)
                        
                else:  # short
                    if data['high'][i] >= current_stop:
                        # Stop loss hit
                        data.loc[data.index[i], 'signal'] = 1
                        in_position = False
                        
                        # Update account balance (simplified)
                        price_change = entry_price / current_stop - 1
                        account_balance *= (1 + price_change)
                        
                    elif data['low'][i] <= data.loc[data.index[i-1], 'take_profit']:
                        # Take profit hit
                        data.loc[data.index[i], 'signal'] = 1
                        in_position = False
                        
                        # Update account balance (simplified)
                        price_change = entry_price / data.loc[data.index[i-1], 'take_profit'] - 1
                        account_balance *= (1 + price_change)
            
            # Check for entry signals if not in position
            if not in_position:
                # Buy signal: RSI crosses below oversold and then back above
                if (data['rsi'].iloc[i-1] <= self.rsi_oversold and 
                    data['rsi'].iloc[i] > self.rsi_oversold):
                    
                    data.loc[data.index[i], 'signal'] = 1
                    in_position = True
                    position_type = 'long'
                    entry_price = data['close'][i]
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        data, i, account_balance, entry_price
                    )
                    data.loc[data.index[i], 'position_size'] = position_size
                    
                    # Calculate stop loss and take profit
                    stop_loss = self.calculate_stop_loss(data, i, entry_price, position_type)
                    take_profit = self.calculate_take_profit(entry_price, stop_loss, position_type)
                    
                    data.loc[data.index[i], 'stop_loss'] = stop_loss
                    data.loc[data.index[i], 'take_profit'] = take_profit
                    
                    # Initialize trailing stop
                    current_stop = stop_loss
                    highest_price = entry_price
                    
                # Sell signal: RSI crosses above overbought and then back below
                elif (data['rsi'].iloc[i-1] >= self.rsi_overbought and 
                      data['rsi'].iloc[i] < self.rsi_overbought):
                    
                    data.loc[data.index[i], 'signal'] = -1
                    in_position = True
                    position_type = 'short'
                    entry_price = data['close'][i]
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        data, i, account_balance, entry_price
                    )
                    data.loc[data.index[i], 'position_size'] = position_size
                    
                    # Calculate stop loss and take profit
                    stop_loss = self.calculate_stop_loss(data, i, entry_price, position_type)
                    take_profit = self.calculate_take_profit(entry_price, stop_loss, position_type)
                    
                    data.loc[data.index[i], 'stop_loss'] = stop_loss
                    data.loc[data.index[i], 'take_profit'] = take_profit
                    
                    # Initialize trailing stop
                    current_stop = stop_loss
                    lowest_price = entry_price
        
        return data


# Register strategy
StrategyFactory.register_strategy("risk_managed_strategy", RiskManagedStrategy)


def run_risk_managed_backtest():
    """Run backtest with risk management"""
    # Define backtest parameters
    symbol = 'BTCUSDT'
    timeframe = '1d'
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Create configuration
    config = {
        'backtesting': {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'position_sizing': {
                'method': 'risk_percentage',
                'params': {
                    'risk_percentage': 0.01,  # 1% risk per trade
                    'stop_loss_type': 'atr',
                    'stop_loss_params': {
                        'atr_multiple': 2.0
                    },
                    'risk_reward_ratio': 2.0  # 2:1 reward-to-risk ratio
                }
            },
            'risk_management': {
                'max_open_positions': 3,
                'max_drawdown': 0.2,  # 20% maximum drawdown
                'trailing_stop': {
                    'enabled': True,
                    'activation_percentage': 0.02,  # Activate after 2% profit
                    'trail_percentage': 0.01  # 1% trailing stop
                }
            }
        },
        'strategy': {
            'name': 'risk_managed_strategy',
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
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
    
    # Create strategy
    strategy = StrategyFactory.create_strategy(config['strategy']['name'], config)
    
    # Run backtest
    results = backtest_engine.run(strategy, data)
    
    print(f"Backtest completed with {len(results['trades'])} trades")
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer(config)
    
    # Analyze results
    metrics = analyzer.analyze(results)
    
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['return_percent']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
    print(f"Risk-Adjusted Return: {metrics['risk_adjusted_return']:.2f}%")
    
    # Create visualizer
    visualizer = BacktestVisualizer(config)
    
    # Plot equity curve
    visualizer.plot_equity_curve(results)
    
    # Plot drawdown
    visualizer.plot_drawdown_curve(results)
    
    # Plot trade distribution
    visualizer.plot_trade_distribution(results)
    
    # Plot position sizing
    visualizer.plot_position_sizes(results)
    
    # Plot risk management
    visualizer.plot_risk_metrics(results)
    
    return results, metrics


if __name__ == "__main__":
    results, metrics = run_risk_managed_backtest()
```

## Conclusion

Effective position sizing and risk management are essential components of successful trading strategies. By implementing these techniques in your Platon Light backtesting workflow, you can:

1. Protect your capital during adverse market conditions
2. Maximize returns during favorable conditions
3. Maintain consistent risk exposure
4. Avoid catastrophic losses
5. Improve the psychological aspects of trading

Remember that no strategy is perfect, and even the best risk management cannot eliminate all risk. However, proper position sizing and risk management can significantly improve your strategy's robustness and long-term performance.

Always test your risk management rules thoroughly in backtesting before applying them to live trading, and be prepared to adapt your approach as market conditions change.
