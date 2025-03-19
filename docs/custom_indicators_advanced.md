# Custom Indicators: Advanced Techniques

This guide covers advanced techniques for creating and optimizing custom indicators in the Platon Light backtesting framework.

## Table of Contents

1. [Introduction](#introduction)
2. [Multi-Timeframe Indicators](#multi-timeframe-indicators)
3. [Adaptive Indicators](#adaptive-indicators)
4. [Machine Learning Enhanced Indicators](#machine-learning-enhanced-indicators)
5. [Composite Indicators](#composite-indicators)
6. [Performance Optimization](#performance-optimization)
7. [Advanced Implementation Examples](#advanced-implementation-examples)

## Introduction

Advanced indicators can provide deeper insights into market behavior and potentially improve trading strategy performance. This guide builds on the basics covered in "Custom Indicators: Basic Implementation" and explores more sophisticated techniques.

## Multi-Timeframe Indicators

Multi-timeframe analysis allows you to incorporate data from different timeframes into your trading decisions, providing a more comprehensive view of market conditions.

### Implementation Framework

```python
import pandas as pd
import numpy as np
from platon_light.indicators.base import BaseIndicator
from platon_light.data.resampler import DataResampler

class MultiTimeframeIndicator(BaseIndicator):
    """
    Base class for multi-timeframe indicators.
    
    This class provides functionality to calculate an indicator across
    multiple timeframes and combine the results.
    """
    
    def __init__(self, base_indicator, timeframes, input_column='close', output_column=None):
        """
        Initialize the multi-timeframe indicator.
        
        Args:
            base_indicator: The indicator to apply across timeframes
            timeframes: List of timeframes to use (e.g., ['1h', '4h', '1d'])
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.base_indicator = base_indicator
        self.timeframes = timeframes
        self.resampler = DataResampler()
    
    @property
    def name(self):
        """Get the indicator name."""
        base_name = self.base_indicator.name
        timeframe_str = '_'.join(self.timeframes)
        return f"MTF_{base_name}_{timeframe_str}"
    
    def calculate(self, data):
        """
        Calculate the indicator across multiple timeframes.
        
        Args:
            data: DataFrame containing price/volume data
            
        Returns:
            DataFrame with indicator values for each timeframe
        """
        # Initialize results DataFrame
        results = pd.DataFrame(index=data.index)
        
        # Calculate indicator for each timeframe
        for tf in self.timeframes:
            # Resample data to the current timeframe
            resampled_data = self.resampler.resample(data, tf)
            
            # Calculate indicator on resampled data
            indicator_values = self.base_indicator.calculate(resampled_data)
            
            # Map values back to original timeframe
            mapped_values = self._map_to_original_timeframe(indicator_values, data.index)
            
            # Add to results
            results[f"{tf}"] = mapped_values
        
        return results
    
    def _map_to_original_timeframe(self, resampled_values, original_index):
        """
        Map resampled indicator values back to the original timeframe.
        
        Args:
            resampled_values: Series with indicator values at resampled timeframe
            original_index: Index of the original timeframe
            
        Returns:
            Series with indicator values mapped to original timeframe
        """
        # Forward fill values to ensure all original timestamps have a value
        if isinstance(resampled_values.index, pd.DatetimeIndex):
            full_idx = pd.date_range(
                start=min(original_index.min(), resampled_values.index.min()),
                end=max(original_index.max(), resampled_values.index.max()),
                freq=pd.infer_freq(original_index)
            )
            filled_values = resampled_values.reindex(full_idx, method='ffill')
            return filled_values.reindex(original_index)
        else:
            # For non-datetime indices, use simple reindexing with forward fill
            return resampled_values.reindex(original_index, method='ffill')
```

### Example: Multi-Timeframe RSI

```python
from platon_light.indicators.basic import RSI

# Create a multi-timeframe RSI indicator
mtf_rsi = MultiTimeframeIndicator(
    base_indicator=RSI(period=14),
    timeframes=['1h', '4h', '1d']
)

# Apply to data
result = mtf_rsi(data)

# Access individual timeframe values
hourly_rsi = result['1h']
daily_rsi = result['1d']

# Create a strategy using multi-timeframe RSI
class MultiTimeframeRSIStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.mtf_rsi = MultiTimeframeIndicator(
            base_indicator=RSI(period=14),
            timeframes=['1h', '4h', '1d']
        )
    
    def prepare_data(self, data):
        return self.mtf_rsi(data)
    
    def generate_signals(self, data):
        # Buy when hourly RSI is oversold but daily RSI is bullish
        buy_condition = (
            (data['1h'] < 30) &  # Hourly RSI oversold
            (data['1d'] > 50)    # Daily RSI bullish
        )
        
        # Sell when hourly RSI is overbought but daily RSI is bearish
        sell_condition = (
            (data['1h'] > 70) &  # Hourly RSI overbought
            (data['1d'] < 50)    # Daily RSI bearish
        )
        
        data['signal'] = 0
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
```

## Adaptive Indicators

Adaptive indicators automatically adjust their parameters based on market conditions, potentially improving performance across different market regimes.

### Volatility-Adjusted Indicators

```python
class VolatilityAdjustedSMA(BaseIndicator):
    """
    Volatility-Adjusted Simple Moving Average.
    
    This indicator adjusts the SMA period based on market volatility.
    Higher volatility leads to shorter periods, while lower volatility
    leads to longer periods.
    """
    
    def __init__(self, base_period=20, volatility_window=100, 
                 min_period=5, max_period=50, input_column='close', 
                 output_column=None):
        """
        Initialize the indicator.
        
        Args:
            base_period: Base period for the SMA
            volatility_window: Window for volatility calculation
            min_period: Minimum allowed period
            max_period: Maximum allowed period
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.base_period = base_period
        self.volatility_window = volatility_window
        self.min_period = min_period
        self.max_period = max_period
    
    @property
    def name(self):
        """Get the indicator name."""
        return f"VolAdjSMA_{self.base_period}"
    
    def calculate(self, data):
        """
        Calculate the Volatility-Adjusted SMA.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with indicator values
        """
        # Calculate historical volatility
        returns = data[self.input_column].pct_change()
        historical_vol = returns.rolling(window=self.volatility_window).std()
        
        # Normalize volatility between 0 and 1
        min_vol = historical_vol.rolling(window=self.volatility_window).min()
        max_vol = historical_vol.rolling(window=self.volatility_window).max()
        norm_vol = (historical_vol - min_vol) / (max_vol - min_vol)
        
        # Calculate adaptive period
        # Higher volatility -> shorter period, Lower volatility -> longer period
        period_range = self.max_period - self.min_period
        adaptive_period = self.max_period - (norm_vol * period_range)
        adaptive_period = adaptive_period.fillna(self.base_period)
        
        # Calculate SMA with variable period
        result = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            if i < self.volatility_window:
                # Use base period for initial data points
                period = self.base_period
            else:
                # Use adaptive period
                period = max(self.min_period, min(self.max_period, int(adaptive_period.iloc[i])))
            
            if i < period:
                result.iloc[i] = np.nan
            else:
                result.iloc[i] = data[self.input_column].iloc[i-period:i].mean()
        
        return result
```

## Machine Learning Enhanced Indicators

Machine learning can enhance traditional indicators by identifying patterns and relationships that may not be apparent with standard technical analysis.

### Regression-Based Indicator

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class MLTrendStrength(BaseIndicator):
    """
    Machine Learning Trend Strength Indicator.
    
    This indicator uses linear regression to measure trend strength
    and predict short-term price movement.
    """
    
    def __init__(self, lookback_period=50, prediction_period=10, 
                 feature_window=5, input_column='close', output_column=None):
        """
        Initialize the indicator.
        
        Args:
            lookback_period: Period for training the model
            prediction_period: Period to predict ahead
            feature_window: Window for feature generation
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.lookback_period = lookback_period
        self.prediction_period = prediction_period
        self.feature_window = feature_window
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    @property
    def name(self):
        """Get the indicator name."""
        return f"MLTrendStrength_{self.lookback_period}_{self.prediction_period}"
    
    def _create_features(self, price_series):
        """Create features for the model."""
        features = pd.DataFrame(index=price_series.index)
        
        # Price change features
        for i in range(1, self.feature_window + 1):
            features[f'return_{i}'] = price_series.pct_change(i)
        
        # Moving average features
        for window in [5, 10, 20]:
            features[f'ma_{window}'] = price_series.rolling(window=window).mean() / price_series - 1
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'vol_{window}'] = price_series.pct_change().rolling(window=window).std()
        
        return features
    
    def calculate(self, data):
        """
        Calculate the ML Trend Strength indicator.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with indicator values
        """
        price = data[self.input_column]
        features = self._create_features(price)
        
        # Initialize result series
        result = pd.Series(index=data.index, dtype=float)
        
        # Calculate indicator for each point after the initial lookback period
        for i in range(self.lookback_period, len(data)):
            # Get training data
            train_features = features.iloc[i-self.lookback_period:i].dropna()
            
            if len(train_features) < self.lookback_period / 2:
                result.iloc[i] = np.nan
                continue
            
            # Create target: future return
            future_return = price.iloc[i:i+self.prediction_period].pct_change(self.prediction_period-1).iloc[-1]
            if pd.isna(future_return):
                # If we don't have enough future data, use what we have
                future_return = price.iloc[i:].pct_change(len(price.iloc[i:])-1).iloc[-1]
            
            # Prepare training data
            X = train_features.values
            y = price.pct_change(self.prediction_period).iloc[i-self.lookback_period:i].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Predict trend strength
            current_features = features.iloc[i:i+1].values
            current_features_scaled = self.scaler.transform(current_features)
            prediction = self.model.predict(current_features_scaled)[0]
            
            # Store prediction as trend strength
            result.iloc[i] = prediction
        
        # Normalize to range [-1, 1]
        result = result.clip(-1, 1)
        
        return result
```

## Composite Indicators

Composite indicators combine multiple indicators to create a more comprehensive view of market conditions.

### Market Regime Indicator

```python
class MarketRegimeIndicator(BaseIndicator):
    """
    Market Regime Indicator.
    
    This composite indicator identifies the current market regime
    (trending, ranging, volatile) by combining multiple indicators.
    """
    
    def __init__(self, lookback_period=100, input_column='close', output_column=None):
        """
        Initialize the indicator.
        
        Args:
            lookback_period: Period for regime analysis
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.lookback_period = lookback_period
        
        # Component indicators
        self.adx = ADX(period=14)  # Trend strength
        self.bb = BollingerBands(period=20, std_dev=2)  # Volatility
        self.rsi = RSI(period=14)  # Momentum
    
    @property
    def name(self):
        """Get the indicator name."""
        return f"MarketRegime_{self.lookback_period}"
    
    def calculate(self, data):
        """
        Calculate the Market Regime Indicator.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with regime probabilities
        """
        # Calculate component indicators
        data_with_indicators = data.copy()
        data_with_indicators = self.adx(data_with_indicators)
        data_with_indicators = self.bb(data_with_indicators)
        data_with_indicators = self.rsi(data_with_indicators)
        
        # Extract indicator values
        adx_values = data_with_indicators['ADX_14']
        bb_width = (data_with_indicators['BB_20_2_upper'] - 
                    data_with_indicators['BB_20_2_lower']) / data_with_indicators['BB_20_2_middle']
        rsi_values = data_with_indicators['RSI_14']
        
        # Initialize regime probabilities
        regimes = pd.DataFrame(index=data.index, columns=['trending', 'ranging', 'volatile'])
        
        # Calculate regime probabilities
        
        # Trending regime: high ADX, moderate BB width, strong RSI direction
        trending_score = (
            (adx_values / 100) *  # Normalized ADX (0-1)
            (1 - abs(rsi_values - 50) / 50)  # RSI direction strength
        )
        
        # Ranging regime: low ADX, narrow BB width, RSI in middle range
        ranging_score = (
            (1 - adx_values / 100) *  # Inverse of normalized ADX
            (1 - bb_width / bb_width.rolling(window=self.lookback_period).max()) *  # Narrow BB width
            (1 - abs(rsi_values - 50) / 50)  # RSI in middle range
        )
        
        # Volatile regime: increasing BB width, RSI at extremes
        volatile_score = (
            (bb_width / bb_width.rolling(window=self.lookback_period).max()) *  # Wide BB width
            (abs(rsi_values - 50) / 50)  # RSI at extremes
        )
        
        # Normalize scores to probabilities
        total_score = trending_score + ranging_score + volatile_score
        regimes['trending'] = trending_score / total_score
        regimes['ranging'] = ranging_score / total_score
        regimes['volatile'] = volatile_score / total_score
        
        # Fill NaN values
        regimes = regimes.fillna(1/3)  # Equal probability if not enough data
        
        return regimes
```

## Performance Optimization

Optimizing indicator performance is crucial for efficient backtesting, especially with large datasets.

### Caching Results

```python
from functools import lru_cache

class CachedIndicator(BaseIndicator):
    """
    Base class for cached indicators.
    
    This class adds result caching to indicators to improve performance
    when the same indicator is calculated multiple times.
    """
    
    def __init__(self, base_indicator):
        """
        Initialize the cached indicator.
        
        Args:
            base_indicator: The indicator to cache
        """
        super().__init__(
            input_column=base_indicator.input_column,
            output_column=base_indicator.output_column
        )
        self.base_indicator = base_indicator
    
    @property
    def name(self):
        """Get the indicator name."""
        return f"Cached_{self.base_indicator.name}"
    
    @lru_cache(maxsize=32)
    def _cached_calculate(self, data_key):
        """Cached calculation method."""
        # Convert the key back to a DataFrame
        import pickle
        data = pickle.loads(data_key)
        return self.base_indicator.calculate(data)
    
    def calculate(self, data):
        """
        Calculate the indicator with caching.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series or DataFrame with indicator values
        """
        # Convert DataFrame to a hashable key
        import pickle
        data_key = pickle.dumps(data)
        
        # Use cached calculation
        return self._cached_calculate(data_key)
```

### Numba Acceleration

```python
import numba

class NumbaAcceleratedRSI(BaseIndicator):
    """
    Numba-accelerated RSI implementation.
    
    This indicator uses Numba to compile the calculation to machine code,
    significantly improving performance.
    """
    
    def __init__(self, period=14, input_column='close', output_column=None):
        """
        Initialize the indicator.
        
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
    
    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_rsi(prices, period):
        """Numba-accelerated RSI calculation."""
        # Pre-allocate output array
        rsi = np.empty_like(prices)
        rsi[:] = np.nan
        
        # Calculate price changes
        deltas = np.zeros_like(prices)
        deltas[1:] = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gains = np.zeros_like(deltas)
        losses = np.zeros_like(deltas)
        
        for i in range(1, len(prices)):
            if deltas[i] > 0:
                gains[i] = deltas[i]
            elif deltas[i] < 0:
                losses[i] = -deltas[i]
        
        # Calculate average gains and losses
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average is simple average
        if period <= len(prices):
            avg_gain[period] = np.sum(gains[1:period+1]) / period
            avg_loss[period] = np.sum(losses[1:period+1]) / period
        
        # Subsequent averages are smoothed
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
        
        # Calculate RS and RSI
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def calculate(self, data):
        """
        Calculate the RSI.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with RSI values
        """
        prices = data[self.input_column].values
        rsi_values = self._calculate_rsi(prices, self.period)
        return pd.Series(rsi_values, index=data.index)
```

## Advanced Implementation Examples

### Ichimoku Cloud Indicator

```python
class IchimokuCloud(BaseIndicator):
    """
    Ichimoku Cloud indicator.
    
    The Ichimoku Cloud is a comprehensive indicator that provides information
    about support/resistance, trend direction, and momentum.
    """
    
    def __init__(self, tenkan_period=9, kijun_period=26, senkou_b_period=52,
                 displacement=26, output_column=None):
        """
        Initialize the Ichimoku Cloud indicator.
        
        Args:
            tenkan_period: Period for Tenkan-sen (Conversion Line)
            kijun_period: Period for Kijun-sen (Base Line)
            senkou_b_period: Period for Senkou Span B (Leading Span B)
            displacement: Displacement period for Senkou Span
            output_column: Column name for the output
        """
        super().__init__(input_column=None, output_column=output_column)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
    
    @property
    def name(self):
        """Get the indicator name."""
        return "Ichimoku"
    
    def _donchian_channel_middle(self, high, low, period):
        """Calculate the middle of the Donchian Channel."""
        high_val = high.rolling(window=period).max()
        low_val = low.rolling(window=period).min()
        return (high_val + low_val) / 2
    
    def calculate(self, data):
        """
        Calculate the Ichimoku Cloud components.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with Ichimoku Cloud components
        """
        # Validate required columns
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_sen = self._donchian_channel_middle(
            data['high'], data['low'], self.tenkan_period
        )
        
        # Calculate Kijun-sen (Base Line)
        kijun_sen = self._donchian_channel_middle(
            data['high'], data['low'], self.kijun_period
        )
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_span_b = self._donchian_channel_middle(
            data['high'], data['low'], self.senkou_b_period
        ).shift(self.displacement)
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = data['close'].shift(-self.displacement)
        
        # Return all components
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })
```

### Heikin-Ashi Candles

```python
class HeikinAshi(BaseIndicator):
    """
    Heikin-Ashi candles indicator.
    
    Heikin-Ashi candles are a modified version of candlestick charts
    that filter out market noise to better identify trends.
    """
    
    def __init__(self, output_column=None):
        """
        Initialize the Heikin-Ashi indicator.
        
        Args:
            output_column: Column name for the output
        """
        super().__init__(input_column=None, output_column=output_column)
    
    @property
    def name(self):
        """Get the indicator name."""
        return "HeikinAshi"
    
    def calculate(self, data):
        """
        Calculate Heikin-Ashi candles.
        
        Args:
            data: DataFrame containing OHLC data
            
        Returns:
            DataFrame with Heikin-Ashi OHLC values
        """
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Initialize result DataFrame
        ha = pd.DataFrame(index=data.index)
        
        # Calculate Heikin-Ashi values
        ha['open'] = np.nan
        ha['high'] = np.nan
        ha['low'] = np.nan
        ha['close'] = np.nan
        
        # First Heikin-Ashi candle
        ha.iloc[0, ha.columns.get_indexer(['open'])[0]] = data['open'].iloc[0]
        ha.iloc[0, ha.columns.get_indexer(['high'])[0]] = data['high'].iloc[0]
        ha.iloc[0, ha.columns.get_indexer(['low'])[0]] = data['low'].iloc[0]
        ha.iloc[0, ha.columns.get_indexer(['close'])[0]] = data['close'].iloc[0]
        
        # Calculate remaining Heikin-Ashi candles
        for i in range(1, len(data)):
            # HA Close = (Open + High + Low + Close) / 4
            ha.iloc[i, ha.columns.get_indexer(['close'])[0]] = (
                data['open'].iloc[i] + data['high'].iloc[i] + 
                data['low'].iloc[i] + data['close'].iloc[i]
            ) / 4
            
            # HA Open = (Previous HA Open + Previous HA Close) / 2
            ha.iloc[i, ha.columns.get_indexer(['open'])[0]] = (
                ha['open'].iloc[i-1] + ha['close'].iloc[i-1]
            ) / 2
            
            # HA High = max(High, HA Open, HA Close)
            ha.iloc[i, ha.columns.get_indexer(['high'])[0]] = max(
                data['high'].iloc[i],
                ha['open'].iloc[i],
                ha['close'].iloc[i]
            )
            
            # HA Low = min(Low, HA Open, HA Close)
            ha.iloc[i, ha.columns.get_indexer(['low'])[0]] = min(
                data['low'].iloc[i],
                ha['open'].iloc[i],
                ha['close'].iloc[i]
            )
        
        return ha
```

This guide covers advanced techniques for creating and optimizing custom indicators in the Platon Light backtesting framework. For testing and validating your custom indicators, see the "Custom Indicators: Testing Framework" guide.
