# Custom Indicators: Basic Implementation

This guide explains how to create and implement custom technical indicators in the Platon Light backtesting framework.

## Table of Contents

1. [Introduction](#introduction)
2. [Indicator Framework Overview](#indicator-framework-overview)
3. [Creating Basic Indicators](#creating-basic-indicators)
4. [Testing Indicators](#testing-indicators)
5. [Integration with Strategies](#integration-with-strategies)
6. [Best Practices](#best-practices)

## Introduction

Technical indicators are mathematical calculations based on price, volume, or open interest data that aim to forecast financial market direction. Custom indicators allow you to implement proprietary analysis methods that may provide an edge in your trading strategies.

The Platon Light backtesting framework provides a flexible system for creating, testing, and using custom indicators in your trading strategies.

## Indicator Framework Overview

### Core Components

The indicator framework in Platon Light consists of:

1. **BaseIndicator**: Abstract base class that all indicators inherit from
2. **IndicatorRegistry**: Central registry for all available indicators
3. **IndicatorFactory**: Factory class for creating indicator instances
4. **IndicatorMixin**: Mixin class for adding indicator functionality to strategies

### BaseIndicator Class

The `BaseIndicator` class provides the foundation for all indicators:

```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    
    All custom indicators should inherit from this class and implement
    the calculate method.
    """
    
    def __init__(self, input_column='close', output_column=None):
        """
        Initialize the indicator.
        
        Args:
            input_column: Column name to use as input (default: 'close')
            output_column: Column name for the output (default: None, will use indicator name)
        """
        self.input_column = input_column
        self._output_column = output_column
    
    @property
    def name(self):
        """Get the indicator name (class name by default)."""
        return self.__class__.__name__
    
    @property
    def output_column(self):
        """Get the output column name."""
        return self._output_column or self.name
    
    @abstractmethod
    def calculate(self, data):
        """
        Calculate the indicator values.
        
        Args:
            data: DataFrame containing price/volume data
            
        Returns:
            Series or DataFrame with indicator values
        """
        pass
    
    def __call__(self, data):
        """
        Calculate indicator and return the updated dataframe.
        
        Args:
            data: DataFrame containing price/volume data
            
        Returns:
            DataFrame with indicator values added
        """
        result = data.copy()
        indicator_values = self.calculate(data)
        
        if isinstance(indicator_values, pd.Series):
            result[self.output_column] = indicator_values
        elif isinstance(indicator_values, pd.DataFrame):
            for column in indicator_values.columns:
                result[f"{self.output_column}_{column}"] = indicator_values[column]
        
        return result
```

## Creating Basic Indicators

### Simple Moving Average (SMA)

Let's implement a Simple Moving Average indicator:

```python
class SMA(BaseIndicator):
    """Simple Moving Average indicator."""
    
    def __init__(self, period=20, input_column='close', output_column=None):
        """
        Initialize the SMA indicator.
        
        Args:
            period: Number of periods for the moving average
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
    
    @property
    def name(self):
        """Get the indicator name with period."""
        return f"SMA_{self.period}"
    
    def calculate(self, data):
        """
        Calculate the Simple Moving Average.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with SMA values
        """
        return data[self.input_column].rolling(window=self.period).mean()
```

### Relative Strength Index (RSI)

Now let's implement a more complex indicator, the RSI:

```python
class RSI(BaseIndicator):
    """Relative Strength Index indicator."""
    
    def __init__(self, period=14, input_column='close', output_column=None):
        """
        Initialize the RSI indicator.
        
        Args:
            period: Number of periods for RSI calculation
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
    
    @property
    def name(self):
        """Get the indicator name with period."""
        return f"RSI_{self.period}"
    
    def calculate(self, data):
        """
        Calculate the Relative Strength Index.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with RSI values
        """
        # Get price series
        price = data[self.input_column]
        
        # Calculate price changes
        delta = price.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
```

### Bollinger Bands

Let's implement Bollinger Bands, which return multiple values:

```python
class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator."""
    
    def __init__(self, period=20, std_dev=2, input_column='close', output_column=None):
        """
        Initialize the Bollinger Bands indicator.
        
        Args:
            period: Number of periods for the moving average
            std_dev: Number of standard deviations for the bands
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
        self.std_dev = std_dev
    
    @property
    def name(self):
        """Get the indicator name with parameters."""
        return f"BB_{self.period}_{self.std_dev}"
    
    def calculate(self, data):
        """
        Calculate the Bollinger Bands.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with middle, upper, and lower band values
        """
        # Calculate middle band (SMA)
        middle_band = data[self.input_column].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std = data[self.input_column].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        # Return all bands as a DataFrame
        return pd.DataFrame({
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        })
```

### Custom Momentum Indicator

Now let's create a custom momentum indicator that's not commonly available:

```python
class MomentumOscillator(BaseIndicator):
    """
    Custom Momentum Oscillator.
    
    This indicator measures price momentum by comparing the current price
    to a price n periods ago, then normalizes the result between 0 and 100.
    """
    
    def __init__(self, period=14, smooth_period=3, input_column='close', output_column=None):
        """
        Initialize the Momentum Oscillator.
        
        Args:
            period: Number of periods for momentum calculation
            smooth_period: Number of periods for smoothing
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
        self.smooth_period = smooth_period
    
    @property
    def name(self):
        """Get the indicator name with parameters."""
        return f"MomentumOsc_{self.period}_{self.smooth_period}"
    
    def calculate(self, data):
        """
        Calculate the Momentum Oscillator.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with oscillator values
        """
        # Get price series
        price = data[self.input_column]
        
        # Calculate raw momentum (current price / price n periods ago)
        momentum = price / price.shift(self.period)
        
        # Convert to percentage change
        momentum_pct = (momentum - 1) * 100
        
        # Apply smoothing
        smoothed = momentum_pct.rolling(window=self.smooth_period).mean()
        
        # Normalize between 0 and 100 using recent range
        # (this is what makes this indicator unique)
        high = smoothed.rolling(window=self.period*2).max()
        low = smoothed.rolling(window=self.period*2).min()
        normalized = 100 * (smoothed - low) / (high - low)
        
        return normalized
```

## Testing Indicators

### Unit Testing

Create unit tests for your indicators to ensure they calculate correctly:

```python
import unittest
import pandas as pd
import numpy as np
from platon_light.indicators.base import BaseIndicator
from platon_light.indicators.basic import SMA, RSI, BollingerBands

class TestIndicators(unittest.TestCase):
    """Test cases for technical indicators."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)
    
    def test_sma(self):
        """Test Simple Moving Average calculation."""
        # Create SMA indicator
        sma = SMA(period=20)
        
        # Calculate SMA
        result = sma(self.data)
        
        # Check that output column exists
        self.assertIn('SMA_20', result.columns)
        
        # Check that first 19 values are NaN (not enough data)
        self.assertTrue(result['SMA_20'].iloc[:19].isna().all())
        
        # Check that remaining values are not NaN
        self.assertTrue(result['SMA_20'].iloc[19:].notna().all())
        
        # Manually calculate SMA for verification
        expected_sma = self.data['close'].rolling(window=20).mean()
        
        # Check that values match expected
        pd.testing.assert_series_equal(result['SMA_20'], expected_sma)
    
    def test_rsi(self):
        """Test Relative Strength Index calculation."""
        # Create RSI indicator
        rsi = RSI(period=14)
        
        # Calculate RSI
        result = rsi(self.data)
        
        # Check that output column exists
        self.assertIn('RSI_14', result.columns)
        
        # Check that first 14 values are NaN (not enough data)
        self.assertTrue(result['RSI_14'].iloc[:14].isna().all())
        
        # Check that remaining values are not NaN
        self.assertTrue(result['RSI_14'].iloc[14:].notna().all())
        
        # Check that values are between 0 and 100
        self.assertTrue((result['RSI_14'].dropna() >= 0).all())
        self.assertTrue((result['RSI_14'].dropna() <= 100).all())
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        # Create Bollinger Bands indicator
        bb = BollingerBands(period=20, std_dev=2)
        
        # Calculate Bollinger Bands
        result = bb(self.data)
        
        # Check that output columns exist
        self.assertIn('BB_20_2_middle', result.columns)
        self.assertIn('BB_20_2_upper', result.columns)
        self.assertIn('BB_20_2_lower', result.columns)
        
        # Check that first 19 values are NaN (not enough data)
        self.assertTrue(result['BB_20_2_middle'].iloc[:19].isna().all())
        
        # Check that remaining values are not NaN
        self.assertTrue(result['BB_20_2_middle'].iloc[19:].notna().all())
        
        # Check that upper band is always greater than middle band
        self.assertTrue((result['BB_20_2_upper'] >= result['BB_20_2_middle']).all())
        
        # Check that lower band is always less than middle band
        self.assertTrue((result['BB_20_2_lower'] <= result['BB_20_2_middle']).all())

if __name__ == '__main__':
    unittest.main()
```

### Visual Testing

Create a visualization script to visually inspect your indicators:

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from platon_light.indicators.basic import SMA, RSI, BollingerBands, MomentumOscillator

def visualize_indicators(data, save_path=None):
    """
    Visualize indicators on price data.
    
    Args:
        data: DataFrame containing price data
        save_path: Optional path to save the visualization
    """
    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Plot price data
    axs[0].plot(data.index, data['close'], label='Close Price')
    axs[0].set_title('Price')
    axs[0].legend()
    
    # Plot SMA
    sma_short = SMA(period=20)(data)
    sma_long = SMA(period=50)(data)
    axs[1].plot(data.index, data['close'], label='Close Price', alpha=0.5)
    axs[1].plot(data.index, sma_short['SMA_20'], label='SMA(20)')
    axs[1].plot(data.index, sma_long['SMA_50'], label='SMA(50)')
    axs[1].set_title('Simple Moving Averages')
    axs[1].legend()
    
    # Plot Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2)(data)
    axs[2].plot(data.index, data['close'], label='Close Price')
    axs[2].plot(data.index, bb['BB_20_2_middle'], label='Middle Band')
    axs[2].plot(data.index, bb['BB_20_2_upper'], label='Upper Band')
    axs[2].plot(data.index, bb['BB_20_2_lower'], label='Lower Band')
    axs[2].fill_between(data.index, bb['BB_20_2_upper'], bb['BB_20_2_lower'], alpha=0.2)
    axs[2].set_title('Bollinger Bands')
    axs[2].legend()
    
    # Plot oscillators
    rsi = RSI(period=14)(data)
    momentum = MomentumOscillator(period=14, smooth_period=3)(data)
    axs[3].plot(data.index, rsi['RSI_14'], label='RSI(14)')
    axs[3].plot(data.index, momentum['MomentumOsc_14_3'], label='Momentum Oscillator')
    axs[3].axhline(y=70, color='r', linestyle='--', alpha=0.3)
    axs[3].axhline(y=30, color='g', linestyle='--', alpha=0.3)
    axs[3].set_title('Oscillators')
    axs[3].set_ylim(0, 100)
    axs[3].legend()
    
    # Format x-axis
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Example usage
if __name__ == '__main__':
    # Load sample data
    data = pd.read_csv('sample_data.csv', index_col='date', parse_dates=True)
    
    # Visualize indicators
    visualize_indicators(data, save_path='indicator_visualization.png')
```

## Integration with Strategies

### Using Indicators in Strategies

Here's how to use custom indicators in your trading strategies:

```python
from platon_light.core.strategy import BaseStrategy
from platon_light.indicators.basic import SMA, RSI, BollingerBands

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates buy signals when a short-term moving average
    crosses above a long-term moving average, and sell signals when it
    crosses below.
    """
    
    def __init__(self, short_period=20, long_period=50):
        """
        Initialize the strategy.
        
        Args:
            short_period: Period for short-term moving average
            long_period: Period for long-term moving average
        """
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
        
        # Initialize indicators
        self.short_ma = SMA(period=short_period)
        self.long_ma = SMA(period=long_period)
    
    def prepare_data(self, data):
        """
        Prepare data by calculating indicators.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added indicator columns
        """
        # Apply indicators
        data = self.short_ma(data)
        data = self.long_ma(data)
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            DataFrame with added signal column
        """
        # Get indicator column names
        short_col = self.short_ma.output_column
        long_col = self.long_ma.output_column
        
        # Initialize signals
        data['signal'] = 0
        
        # Generate signals based on moving average crossover
        data.loc[data[short_col] > data[long_col], 'signal'] = 1  # Buy signal
        data.loc[data[short_col] < data[long_col], 'signal'] = -1  # Sell signal
        
        return data
```

### Combining Multiple Indicators

Here's a more complex strategy that combines multiple indicators:

```python
class MultiIndicatorStrategy(BaseStrategy):
    """
    Multi-Indicator Strategy.
    
    This strategy combines RSI, Bollinger Bands, and moving averages
    to generate trading signals.
    """
    
    def __init__(self, ma_short=20, ma_long=50, rsi_period=14, bb_period=20, bb_std=2):
        """
        Initialize the strategy.
        
        Args:
            ma_short: Period for short-term moving average
            ma_long: Period for long-term moving average
            rsi_period: Period for RSI
            bb_period: Period for Bollinger Bands
            bb_std: Standard deviation for Bollinger Bands
        """
        super().__init__()
        
        # Initialize indicators
        self.short_ma = SMA(period=ma_short)
        self.long_ma = SMA(period=ma_long)
        self.rsi = RSI(period=rsi_period)
        self.bb = BollingerBands(period=bb_period, std_dev=bb_std)
    
    def prepare_data(self, data):
        """
        Prepare data by calculating indicators.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added indicator columns
        """
        # Apply indicators
        data = self.short_ma(data)
        data = self.long_ma(data)
        data = self.rsi(data)
        data = self.bb(data)
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            DataFrame with added signal column
        """
        # Get indicator column names
        short_col = self.short_ma.output_column
        long_col = self.long_ma.output_column
        rsi_col = self.rsi.output_column
        bb_middle = f"{self.bb.output_column}_middle"
        bb_upper = f"{self.bb.output_column}_upper"
        bb_lower = f"{self.bb.output_column}_lower"
        
        # Initialize signals
        data['signal'] = 0
        
        # Generate buy signals
        buy_condition = (
            (data[short_col] > data[long_col]) &  # MA crossover
            (data[rsi_col] < 30) &  # RSI oversold
            (data['close'] < data[bb_lower])  # Price below lower BB
        )
        data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals
        sell_condition = (
            (data[short_col] < data[long_col]) &  # MA crossover
            (data[rsi_col] > 70) &  # RSI overbought
            (data['close'] > data[bb_upper])  # Price above upper BB
        )
        data.loc[sell_condition, 'signal'] = -1
        
        return data
```

## Best Practices

### Efficiency

1. **Vectorize calculations**: Use NumPy and Pandas vectorized operations instead of loops
2. **Cache results**: Avoid recalculating indicators when possible
3. **Use rolling windows efficiently**: Be aware of the computational cost of rolling windows

### Robustness

1. **Handle edge cases**: Ensure your indicator handles NaN values, zero values, and other edge cases
2. **Validate inputs**: Check that input data contains required columns
3. **Document limitations**: Clearly document any limitations or assumptions

### Maintainability

1. **Follow naming conventions**: Use consistent naming for indicator classes and output columns
2. **Document parameters**: Clearly document what each parameter does
3. **Write unit tests**: Test your indicators with different parameters and edge cases

### Example of a Well-Documented Indicator

```python
class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price (VWAP) indicator.
    
    VWAP is calculated by adding up the dollars traded for every transaction
    (price multiplied by the number of shares traded) and then dividing by the
    total shares traded.
    
    This indicator requires 'high', 'low', 'close', and 'volume' columns in the data.
    
    Parameters:
        period: Number of periods for calculation (default: None for daily VWAP)
        session_start: Time to reset VWAP calculation (default: None)
    
    Example:
        # Calculate daily VWAP
        vwap = VWAP()
        data = vwap(data)
        
        # Calculate 20-period VWAP
        vwap20 = VWAP(period=20)
        data = vwap20(data)
    """
    
    def __init__(self, period=None, session_start=None, output_column=None):
        """
        Initialize the VWAP indicator.
        
        Args:
            period: Number of periods for calculation (default: None for daily VWAP)
            session_start: Time to reset VWAP calculation (default: None)
            output_column: Column name for the output
        """
        super().__init__(input_column=None, output_column=output_column)
        self.period = period
        self.session_start = session_start
    
    @property
    def name(self):
        """Get the indicator name with period if specified."""
        if self.period:
            return f"VWAP_{self.period}"
        return "VWAP"
    
    def calculate(self, data):
        """
        Calculate the Volume Weighted Average Price.
        
        Args:
            data: DataFrame containing price and volume data
            
        Returns:
            Series with VWAP values
        """
        # Validate required columns
        required_columns = ['high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate price * volume
        price_volume = typical_price * data['volume']
        
        if self.period:
            # Calculate rolling VWAP
            cumulative_price_volume = price_volume.rolling(window=self.period).sum()
            cumulative_volume = data['volume'].rolling(window=self.period).sum()
        else:
            # Calculate daily VWAP
            if self.session_start and isinstance(data.index, pd.DatetimeIndex):
                # Reset calculation at session start
                session_mask = data.index.time >= pd.Timestamp(self.session_start).time()
                session_groups = session_mask.cumsum()
                
                cumulative_price_volume = price_volume.groupby(session_groups).cumsum()
                cumulative_volume = data['volume'].groupby(session_groups).cumsum()
            else:
                # Simple cumulative calculation
                cumulative_price_volume = price_volume.cumsum()
                cumulative_volume = data['volume'].cumsum()
        
        # Calculate VWAP
        vwap = cumulative_price_volume / cumulative_volume
        
        return vwap
```

This guide covers the basics of implementing custom indicators in the Platon Light backtesting framework. For more advanced techniques, see the "Custom Indicators: Advanced Techniques" guide.
