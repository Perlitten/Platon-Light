# Custom Indicators Testing Framework

This guide provides a comprehensive approach to testing custom indicators in the Platon Light backtesting framework. Proper testing ensures that your indicators are reliable, accurate, and perform well in various market conditions.

## Table of Contents

1. [Introduction](#introduction)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [Performance Testing](#performance-testing)
5. [Visualization Testing](#visualization-testing)
6. [Edge Case Testing](#edge-case-testing)
7. [Test Automation](#test-automation)
8. [Continuous Integration](#continuous-integration)
9. [Best Practices](#best-practices)

## Introduction

Testing custom indicators is crucial for ensuring their reliability and accuracy in trading strategies. The Platon Light framework provides a comprehensive testing infrastructure that allows you to:

- Verify indicator calculations against known values
- Test integration with the backtesting engine
- Benchmark performance with large datasets
- Validate behavior in edge cases
- Visualize indicator outputs for manual inspection

## Unit Testing

Unit tests verify that individual indicators calculate correct values under various conditions.

### Basic Unit Testing

```python
import unittest
import pandas as pd
import numpy as np
from platon_light.indicators.basic import SMA

class TestSMA(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.data = pd.DataFrame({
            'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })
        
    def test_sma_calculation(self):
        # Create SMA indicator with period 3
        sma = SMA(period=3)
        
        # Apply to data
        result = sma(self.data)
        
        # Expected values (manually calculated)
        expected = pd.Series([None, None, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], 
                             name='SMA_3')
        
        # Assert that calculated values match expected values
        pd.testing.assert_series_equal(result['SMA_3'].reset_index(drop=True), 
                                      expected.reset_index(drop=True), 
                                      check_names=False)
```

### Testing Against Reference Implementations

For complex indicators, it's useful to test against established implementations:

```python
def test_rsi_against_reference(self):
    # Calculate RSI using our implementation
    rsi = RSI(period=14)
    result = rsi(self.data)
    
    # Calculate RSI using a reference library (e.g., pandas_ta, ta-lib)
    import pandas_ta as ta
    reference_rsi = ta.rsi(self.data['close'], length=14)
    
    # Compare results (allowing for small numerical differences)
    np.testing.assert_allclose(
        result['RSI_14'].dropna().values,
        reference_rsi.dropna().values,
        rtol=1e-10
    )
```

## Integration Testing

Integration tests verify that indicators work correctly within the full backtesting workflow.

### Testing with Strategies

```python
def test_indicator_in_strategy(self):
    # Create a strategy that uses the indicator
    class SMAStrategy(BaseStrategy):
        def __init__(self):
            self.sma_short = SMA(period=10)
            self.sma_long = SMA(period=30)
            
        def prepare_data(self, data):
            data = self.sma_short(data)
            data = self.sma_long(data)
            return data
            
        def generate_signals(self, data):
            data['signal'] = 0
            # Buy when short SMA crosses above long SMA
            data.loc[data['SMA_10'] > data['SMA_30'], 'signal'] = 1
            # Sell when short SMA crosses below long SMA
            data.loc[data['SMA_10'] < data['SMA_30'], 'signal'] = -1
            return data
    
    # Run backtest with the strategy
    backtest_engine = BacktestEngine()
    results = backtest_engine.run(
        data=self.data,
        strategy=SMAStrategy(),
        initial_capital=10000
    )
    
    # Verify that signals were generated
    self.assertTrue((results['signal'] != 0).any())
    
    # Verify that trades were executed
    self.assertTrue((results['position'] != 0).any())
```

### Testing Multiple Indicators Together

```python
def test_multiple_indicators(self):
    # Create a strategy that uses multiple indicators
    class MultiIndicatorStrategy(BaseStrategy):
        def __init__(self):
            self.sma = SMA(period=20)
            self.rsi = RSI(period=14)
            self.bb = BollingerBands(period=20, std_dev=2)
            
        def prepare_data(self, data):
            data = self.sma(data)
            data = self.rsi(data)
            data = self.bb(data)
            return data
            
        def generate_signals(self, data):
            # Strategy logic using multiple indicators
            # ...
            return data
    
    # Run backtest and verify results
    # ...
```

## Performance Testing

Performance tests ensure that indicators calculate efficiently and scale well with large datasets.

### Speed Testing

```python
def test_indicator_speed(self):
    # Create datasets of different sizes
    sizes = [100, 1000, 10000, 100000]
    datasets = {}
    
    for size in sizes:
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=size, freq='5min')
        close_prices = 100 + np.cumsum(np.random.normal(0, 0.1, size))
        datasets[size] = pd.DataFrame({'close': close_prices}, index=dates)
    
    # Measure execution time for each dataset
    for size, data in datasets.items():
        sma = SMA(period=20)
        
        start_time = time.time()
        result = sma(data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"SMA with {size} data points: {execution_time:.6f} seconds")
```

### Memory Usage Testing

```python
from memory_profiler import profile

@profile
def test_indicator_memory(self):
    # Create a large dataset
    size = 100000
    dates = pd.date_range(start='2020-01-01', periods=size, freq='5min')
    close_prices = 100 + np.cumsum(np.random.normal(0, 0.1, size))
    data = pd.DataFrame({'close': close_prices}, index=dates)
    
    # Calculate indicator
    sma = SMA(period=20)
    result = sma(data)
```

## Visualization Testing

Visualization tests help identify issues that may not be apparent from numerical tests alone.

### Plotting Indicator Values

```python
def test_indicator_visualization(self):
    # Calculate indicator
    sma = SMA(period=20)
    result = sma(self.data)
    
    # Plot price and indicator
    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result['close'], label='Price')
    plt.plot(result.index, result['SMA_20'], label='SMA(20)')
    plt.title('Price vs SMA(20)')
    plt.legend()
    
    # Save the plot for manual inspection
    plt.savefig('sma_visualization.png')
```

### Comparing Multiple Indicators

```python
def test_compare_indicators(self):
    # Calculate multiple indicators
    sma20 = SMA(period=20)(self.data)
    sma50 = SMA(period=50)(self.data)
    ema20 = EMA(period=20)(self.data)
    
    # Plot for comparison
    plt.figure(figsize=(12, 6))
    plt.plot(sma20.index, sma20['close'], label='Price')
    plt.plot(sma20.index, sma20['SMA_20'], label='SMA(20)')
    plt.plot(sma50.index, sma50['SMA_50'], label='SMA(50)')
    plt.plot(ema20.index, ema20['EMA_20'], label='EMA(20)')
    plt.title('Comparison of Moving Averages')
    plt.legend()
    
    # Save the plot
    plt.savefig('moving_averages_comparison.png')
```

## Edge Case Testing

Edge case tests verify that indicators handle unusual or extreme data correctly.

### Empty Dataset

```python
def test_empty_dataset(self):
    # Create an empty dataset
    empty_data = pd.DataFrame({'close': []})
    
    # Apply indicator
    sma = SMA(period=20)
    result = sma(empty_data)
    
    # Verify that result is also empty
    self.assertEqual(len(result), 0)
```

### Missing Values

```python
def test_missing_values(self):
    # Create data with missing values
    data_with_na = self.data.copy()
    data_with_na.loc[5:7, 'close'] = np.nan
    
    # Apply indicator
    sma = SMA(period=3)
    result = sma(data_with_na)
    
    # Verify that indicator handles NaN values correctly
    self.assertTrue(np.isnan(result.loc[5:9, 'SMA_3']).all())
```

### Extreme Values

```python
def test_extreme_values(self):
    # Create data with extreme values
    extreme_data = self.data.copy()
    extreme_data.loc[5, 'close'] = 1000000  # Extremely high value
    extreme_data.loc[6, 'close'] = 0.00001  # Extremely low value
    
    # Apply indicator
    rsi = RSI(period=14)
    result = rsi(extreme_data)
    
    # Verify that indicator doesn't produce NaN or infinity
    self.assertFalse(np.isinf(result['RSI_14']).any())
    
    # Verify that RSI stays within bounds [0, 100]
    self.assertTrue((result['RSI_14'].dropna() >= 0).all())
    self.assertTrue((result['RSI_14'].dropna() <= 100).all())
```

## Test Automation

Automating tests ensures that indicators remain reliable as the codebase evolves.

### Test Runner

```python
def run_all_tests():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestSMA))
    suite.addTest(unittest.makeSuite(TestEMA))
    suite.addTest(unittest.makeSuite(TestRSI))
    # ...
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    with open('test_report.txt', 'w') as f:
        f.write(f"Tests run: {result.testsRun}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Failures: {len(result.failures)}\n")

if __name__ == '__main__':
    run_all_tests()
```

### Parameterized Tests

```python
import unittest
from parameterized import parameterized

class TestIndicatorsParameterized(unittest.TestCase):
    @parameterized.expand([
        ("SMA", SMA, {'period': 20}),
        ("EMA", EMA, {'period': 20}),
        ("RSI", RSI, {'period': 14}),
        ("Bollinger Bands", BollingerBands, {'period': 20, 'std_dev': 2})
    ])
    def test_indicator_calculation(self, name, indicator_class, params):
        # Create indicator
        indicator = indicator_class(**params)
        
        # Apply to data
        result = indicator(self.data)
        
        # Verify that output column exists
        self.assertIn(indicator.output_column, result.columns)
        
        # Verify that indicator produces valid values
        self.assertFalse(result[indicator.output_column].isnull().all())
```

## Continuous Integration

Integrating tests into a CI/CD pipeline ensures that indicators are tested automatically with each code change.

### GitHub Actions Example

```yaml
name: Run Indicator Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/test_indicators/ --cov=platon_light/indicators
    
    - name: Upload coverage report
      uses: codecov/codecov-action@v1
```

## Best Practices

### 1. Test Indicator Properties

Verify that indicators maintain their mathematical properties:

```python
def test_rsi_bounds(self):
    # RSI should always be between 0 and 100
    rsi = RSI(period=14)
    result = rsi(self.data)
    
    self.assertTrue((result['RSI_14'].dropna() >= 0).all())
    self.assertTrue((result['RSI_14'].dropna() <= 100).all())
```

### 2. Test Parameter Variations

Test indicators with different parameter values:

```python
def test_sma_periods(self):
    periods = [5, 10, 20, 50, 200]
    
    for period in periods:
        sma = SMA(period=period)
        result = sma(self.data)
        
        # Verify that output column exists
        self.assertIn(f'SMA_{period}', result.columns)
        
        # Verify that first (period-1) values are NaN
        self.assertTrue(result[f'SMA_{period}'].iloc[:period-1].isnull().all())
        
        # Verify that remaining values are not NaN
        self.assertFalse(result[f'SMA_{period}'].iloc[period:].isnull().any())
```

### 3. Test Indicator Combinations

Test how indicators work together:

```python
def test_indicator_combination(self):
    # Apply multiple indicators
    data = self.data.copy()
    data = SMA(period=20)(data)
    data = RSI(period=14)(data)
    data = BollingerBands(period=20, std_dev=2)(data)
    
    # Verify that all output columns exist
    self.assertIn('SMA_20', data.columns)
    self.assertIn('RSI_14', data.columns)
    self.assertIn('BB_Upper_20_2', data.columns)
    self.assertIn('BB_Middle_20_2', data.columns)
    self.assertIn('BB_Lower_20_2', data.columns)
```

### 4. Test Indicator Chaining

Test indicators that depend on other indicators:

```python
def test_indicator_chaining(self):
    # Create a composite indicator that uses SMA
    class SMASlopeIndicator(BaseIndicator):
        def __init__(self, period=20):
            super().__init__(input_column='close')
            self.period = period
            self.sma = SMA(period=period)
        
        @property
        def name(self):
            return f"SMA_Slope_{self.period}"
        
        def calculate(self, data):
            # First calculate SMA
            data = self.sma(data)
            
            # Then calculate slope of SMA
            sma_column = self.sma.output_column
            slope = data[sma_column].diff() / data.index.to_series().diff().dt.total_seconds() * 86400
            return slope
    
    # Test the chained indicator
    slope_indicator = SMASlopeIndicator(period=20)
    result = slope_indicator(self.data)
    
    # Verify that both indicators' outputs exist
    self.assertIn('SMA_20', result.columns)
    self.assertIn('SMA_Slope_20', result.columns)
```

### 5. Document Test Cases

Document each test case with clear descriptions:

```python
def test_bollinger_bands_width(self):
    """
    Test that Bollinger Bands width increases during volatile periods
    and decreases during stable periods.
    """
    # Create data with varying volatility
    data = create_data_with_volatility_regimes()
    
    # Calculate Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2)
    result = bb(data)
    
    # Calculate BB width
    result['BB_Width'] = (result['BB_Upper_20_2'] - result['BB_Lower_20_2']) / result['BB_Middle_20_2']
    
    # Verify that BB width is higher during volatile periods
    high_volatility_width = result.loc[result['volatility_regime'] == 'high', 'BB_Width'].mean()
    low_volatility_width = result.loc[result['volatility_regime'] == 'low', 'BB_Width'].mean()
    
    self.assertGreater(high_volatility_width, low_volatility_width)
```

By following this comprehensive testing framework, you can ensure that your custom indicators are reliable, accurate, and perform well in various market conditions, ultimately leading to more robust trading strategies in the Platon Light backtesting framework.
