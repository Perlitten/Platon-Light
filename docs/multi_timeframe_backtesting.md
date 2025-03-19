# Multi-Timeframe Backtesting Guide

This guide explains how to implement and analyze backtests across multiple timeframes using the Platon Light backtesting module.

## Table of Contents

1. [Introduction](#introduction)
2. [Why Use Multiple Timeframes](#why-use-multiple-timeframes)
3. [Implementation Approaches](#implementation-approaches)
4. [Data Preparation](#data-preparation)
5. [Strategy Implementation](#strategy-implementation)
6. [Performance Analysis](#performance-analysis)
7. [Best Practices](#best-practices)
8. [Example Implementation](#example-implementation)

## Introduction

Multi-timeframe backtesting involves testing a trading strategy across different timeframes (e.g., 1-minute, 15-minute, 1-hour, 4-hour, daily) to evaluate its consistency and robustness. This approach provides deeper insights into strategy performance and helps identify optimal trading frequencies.

## Why Use Multiple Timeframes

Testing across multiple timeframes offers several benefits:

1. **Robustness Verification**: A truly robust strategy should perform well across different timeframes
2. **Timeframe Optimization**: Identify which timeframe works best for your strategy
3. **Market Dynamics Understanding**: Reveal how market behavior changes across timeframes
4. **Noise Reduction**: Higher timeframes filter out market noise
5. **Trade Frequency Optimization**: Balance between trade frequency and transaction costs
6. **Improved Signal Quality**: Combine signals from multiple timeframes for better entries/exits

## Implementation Approaches

There are two main approaches to multi-timeframe backtesting:

### 1. Sequential Testing

Run separate backtests for each timeframe and compare results:

```python
timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
results = {}

for tf in timeframes:
    backtest_engine = BacktestEngine(config)
    results[tf] = backtest_engine.run(
        symbol='BTCUSDT',
        timeframe=tf,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 12, 31)
    )
```

### 2. Integrated Multi-Timeframe Strategy

Create a strategy that processes data from multiple timeframes simultaneously:

```python
class MultiTimeframeStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.primary_tf = config.get('strategy', {}).get('primary_timeframe', '1h')
        self.secondary_tf = config.get('strategy', {}).get('secondary_timeframe', '4h')
        
    def generate_signals(self, data, secondary_data=None):
        # Process data from both timeframes
        # Generate signals based on combined analysis
        return data
```

## Data Preparation

Proper data preparation is crucial for multi-timeframe backtesting:

### Loading Data for Multiple Timeframes

```python
def load_multi_timeframe_data(symbol, timeframes, start_date, end_date):
    """Load data for multiple timeframes"""
    data_loader = DataLoader(config)
    multi_tf_data = {}
    
    for tf in timeframes:
        multi_tf_data[tf] = data_loader.load_data(
            symbol=symbol,
            timeframe=tf,
            start_date=start_date,
            end_date=end_date
        )
    
    return multi_tf_data
```

### Aligning Timestamps

When working with multiple timeframes, ensure proper timestamp alignment:

```python
def align_timeframes(primary_data, secondary_data):
    """Align data from different timeframes by timestamp"""
    # Convert timestamps to datetime if they're not already
    primary_data['datetime'] = pd.to_datetime(primary_data['timestamp'], unit='ms')
    secondary_data['datetime'] = pd.to_datetime(secondary_data['timestamp'], unit='ms')
    
    # Create a copy of secondary data for merging
    secondary_data_copy = secondary_data.copy()
    
    # For each row in primary data, find the most recent row in secondary data
    aligned_secondary = []
    
    for idx, row in primary_data.iterrows():
        current_time = row['datetime']
        # Find the most recent data point in secondary data
        valid_rows = secondary_data_copy[secondary_data_copy['datetime'] <= current_time]
        
        if not valid_rows.empty:
            most_recent = valid_rows.iloc[-1].to_dict()
            aligned_secondary.append(most_recent)
        else:
            # If no valid secondary data found, use NaN values
            aligned_secondary.append({col: np.nan for col in secondary_data_copy.columns})
    
    # Convert to DataFrame and add suffix to column names
    aligned_df = pd.DataFrame(aligned_secondary)
    aligned_df = aligned_df.add_suffix('_secondary')
    
    # Combine with primary data
    combined_data = pd.concat([primary_data.reset_index(drop=True), 
                              aligned_df.reset_index(drop=True)], axis=1)
    
    return combined_data
```

## Strategy Implementation

### Multi-Timeframe Strategy Base Class

```python
class MultiTimeframeStrategy(BaseStrategy):
    """Base class for strategies that use multiple timeframes"""
    
    def __init__(self, config):
        super().__init__(config)
        self.timeframes = config.get('strategy', {}).get('timeframes', ['1h', '4h'])
        self.primary_tf = self.timeframes[0]
        
    def preprocess_data(self, multi_tf_data):
        """Preprocess data from multiple timeframes"""
        # Implement preprocessing logic
        return multi_tf_data
    
    def generate_signals(self, data, secondary_data=None):
        """Generate trading signals using data from multiple timeframes"""
        # Implement signal generation logic
        return data
    
    def run(self, multi_tf_data):
        """Run strategy on multiple timeframe data"""
        # Preprocess data
        processed_data = self.preprocess_data(multi_tf_data)
        
        # Generate signals
        primary_data = processed_data[self.primary_tf]
        secondary_data = {tf: processed_data[tf] for tf in self.timeframes if tf != self.primary_tf}
        
        signals = self.generate_signals(primary_data, secondary_data)
        
        return signals
```

### Example: Trend Following with Multiple Timeframes

```python
class MultiTimeframeTrendStrategy(MultiTimeframeStrategy):
    """Trend following strategy using multiple timeframes"""
    
    def __init__(self, config):
        super().__init__(config)
        self.fast_period = config.get('strategy', {}).get('fast_period', 20)
        self.slow_period = config.get('strategy', {}).get('slow_period', 50)
        
    def generate_signals(self, data, secondary_data=None):
        """Generate signals based on trend alignment across timeframes"""
        # Calculate EMAs on primary timeframe
        data['ema_fast'] = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate trend on secondary timeframe
        secondary_tf = list(secondary_data.keys())[0]  # Get first secondary timeframe
        secondary = secondary_data[secondary_tf]
        secondary['ema_fast'] = secondary['close'].ewm(span=self.fast_period, adjust=False).mean()
        secondary['ema_slow'] = secondary['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Align secondary timeframe data with primary
        combined = align_timeframes(data, secondary)
        
        # Generate signals based on trend alignment
        data['primary_trend'] = np.where(data['ema_fast'] > data['ema_slow'], 1, -1)
        data['secondary_trend'] = np.where(
            combined['ema_fast_secondary'] > combined['ema_slow_secondary'], 1, -1)
        
        # Signal when trends align
        data['signal'] = 0
        data.loc[(data['primary_trend'] == 1) & (data['secondary_trend'] == 1) & 
                 (data['primary_trend'].shift(1) != 1), 'signal'] = 1  # Buy when both trends are up
        data.loc[(data['primary_trend'] == -1) & (data['secondary_trend'] == -1) & 
                 (data['primary_trend'].shift(1) != -1), 'signal'] = -1  # Sell when both trends are down
        
        return data
```

## Performance Analysis

### Comparing Performance Across Timeframes

```python
def compare_timeframe_performance(results):
    """Compare performance metrics across timeframes"""
    metrics = ['return_percent', 'sharpe_ratio', 'max_drawdown_percent', 'win_rate']
    comparison = {}
    
    for tf, result in results.items():
        comparison[tf] = {metric: result['metrics'][metric] for metric in metrics}
    
    # Convert to DataFrame for easy comparison
    comparison_df = pd.DataFrame(comparison).T
    
    # Print comparison table
    print("Performance Across Timeframes:")
    print(comparison_df)
    
    # Create bar charts for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        comparison_df[metric].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('timeframe_comparison.png')
    plt.show()
    
    return comparison_df
```

### Visualizing Equity Curves Across Timeframes

```python
def compare_equity_curves(results):
    """Compare equity curves across timeframes"""
    plt.figure(figsize=(12, 6))
    
    for tf, result in results.items():
        # Extract equity curve data
        equity_data = pd.DataFrame(result['equity_curve'])
        equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'], unit='ms')
        
        # Normalize to percentage returns
        initial_equity = equity_data['equity'].iloc[0]
        equity_data['equity_pct'] = (equity_data['equity'] / initial_equity - 1) * 100
        
        # Plot equity curve
        plt.plot(equity_data['timestamp'], equity_data['equity_pct'], label=f'{tf}')
    
    plt.title("Equity Curve Comparison Across Timeframes")
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_curves_comparison.png')
    plt.show()
```

## Best Practices

1. **Start with Higher Timeframes**: Begin testing with higher timeframes (daily, 4-hour) before moving to lower ones
2. **Ensure Sufficient Data**: Lower timeframes require more data points for statistical significance
3. **Consider Computational Resources**: Testing on very low timeframes can be computationally intensive
4. **Align Entry/Exit Logic**: Ensure your strategy's logic is consistent across timeframes
5. **Account for Transaction Costs**: Lower timeframes typically involve more trades and higher costs
6. **Test Market Regimes**: Evaluate performance across different market conditions for each timeframe
7. **Avoid Curve-Fitting**: A strategy that works across multiple timeframes is less likely to be curve-fitted
8. **Consider Trading Hours**: Some timeframes may be more active during specific market hours
9. **Validate with Out-of-Sample Data**: Test the best-performing timeframe on unseen data
10. **Combine Timeframes Strategically**: Use higher timeframes for trend direction and lower timeframes for entry/exit timing

## Example Implementation

Here's a complete example of multi-timeframe backtesting:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.core.strategy_factory import StrategyFactory
from platon_light.core.base_strategy import BaseStrategy

# Define multi-timeframe strategy
class MultiTimeframeMACD(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.fast_period = config.get('strategy', {}).get('fast_period', 12)
        self.slow_period = config.get('strategy', {}).get('slow_period', 26)
        self.signal_period = config.get('strategy', {}).get('signal_period', 9)
        self.higher_tf = config.get('strategy', {}).get('higher_timeframe', '4h')
        self.higher_tf_data = None
    
    def set_higher_timeframe_data(self, data):
        """Set higher timeframe data for the strategy"""
        self.higher_tf_data = data
    
    def calculate_macd(self, data):
        """Calculate MACD indicator"""
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def generate_signals(self, data):
        """Generate trading signals using MACD from multiple timeframes"""
        # Calculate MACD for primary timeframe
        primary_macd, primary_signal, primary_hist = self.calculate_macd(data)
        data['macd'] = primary_macd
        data['signal_line'] = primary_signal
        data['histogram'] = primary_hist
        
        # Initialize signal column
        data['signal'] = 0
        
        # If higher timeframe data is available
        if self.higher_tf_data is not None:
            # Calculate MACD for higher timeframe
            higher_macd, higher_signal, higher_hist = self.calculate_macd(self.higher_tf_data)
            self.higher_tf_data['macd'] = higher_macd
            self.higher_tf_data['signal_line'] = higher_signal
            self.higher_tf_data['histogram'] = higher_hist
            
            # Align higher timeframe data with primary timeframe
            aligned_data = align_timeframes(data, self.higher_tf_data)
            
            # Generate signals based on both timeframes
            for i in range(1, len(data)):
                # Higher timeframe trend (positive histogram = bullish, negative = bearish)
                higher_tf_bullish = aligned_data.iloc[i].get('histogram_secondary', 0) > 0
                
                # Primary timeframe signal conditions
                macd_crossover = (primary_hist[i] > 0 and primary_hist[i-1] <= 0)
                macd_crossunder = (primary_hist[i] < 0 and primary_hist[i-1] >= 0)
                
                # Generate signals when primary timeframe signal aligns with higher timeframe trend
                if macd_crossover and higher_tf_bullish:
                    data.loc[data.index[i], 'signal'] = 1  # Buy signal
                elif macd_crossunder and not higher_tf_bullish:
                    data.loc[data.index[i], 'signal'] = -1  # Sell signal
        else:
            # If no higher timeframe data, use only primary timeframe
            for i in range(1, len(data)):
                if primary_hist[i] > 0 and primary_hist[i-1] <= 0:
                    data.loc[data.index[i], 'signal'] = 1  # Buy signal
                elif primary_hist[i] < 0 and primary_hist[i-1] >= 0:
                    data.loc[data.index[i], 'signal'] = -1  # Sell signal
        
        return data

# Register strategy
StrategyFactory.register_strategy("multi_tf_macd", MultiTimeframeMACD)

# Main function to run multi-timeframe backtest
def run_multi_timeframe_backtest():
    # Define test parameters
    symbol = 'BTCUSDT'
    primary_timeframes = ['15m', '1h', '4h', '1d']
    higher_timeframe = '4h'
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    
    # Base configuration
    base_config = {
        'backtesting': {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005
        },
        'strategy': {
            'name': 'multi_tf_macd',
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'higher_timeframe': higher_timeframe
        }
    }
    
    # Create data loader
    data_loader = DataLoader(base_config)
    
    # Load higher timeframe data once
    higher_tf_data = data_loader.load_data(
        symbol=symbol,
        timeframe=higher_timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    # Run backtests for each primary timeframe
    results = {}
    
    for tf in primary_timeframes:
        print(f"Running backtest for {tf} timeframe...")
        
        # Skip if primary timeframe is the same as higher timeframe
        if tf == higher_timeframe:
            config = base_config.copy()
            config['strategy']['higher_timeframe'] = None
        else:
            config = base_config.copy()
        
        # Load primary timeframe data
        primary_data = data_loader.load_data(
            symbol=symbol,
            timeframe=tf,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create backtest engine
        backtest_engine = BacktestEngine(config)
        
        # Set higher timeframe data for the strategy
        strategy = backtest_engine.get_strategy()
        if tf != higher_timeframe:
            strategy.set_higher_timeframe_data(higher_tf_data)
        
        # Run backtest
        results[tf] = backtest_engine.run_with_data(primary_data)
        
        print(f"Completed backtest for {tf} timeframe")
    
    # Compare results across timeframes
    comparison = compare_timeframe_performance(results)
    
    # Compare equity curves
    compare_equity_curves(results)
    
    # Determine best timeframe based on Sharpe ratio
    best_tf = comparison['sharpe_ratio'].idxmax()
    print(f"\nBest performing timeframe: {best_tf}")
    print(f"Sharpe ratio: {comparison.loc[best_tf, 'sharpe_ratio']:.2f}")
    print(f"Return: {comparison.loc[best_tf, 'return_percent']:.2f}%")
    
    return results, comparison

# Run the multi-timeframe backtest
if __name__ == "__main__":
    results, comparison = run_multi_timeframe_backtest()
```

This example demonstrates how to:
1. Implement a strategy that uses data from multiple timeframes
2. Run backtests across different primary timeframes
3. Compare performance metrics to identify the optimal timeframe
4. Visualize results to gain insights into timeframe-specific behavior

By testing your strategies across multiple timeframes, you can develop more robust trading systems that adapt to different market conditions and trading frequencies.
