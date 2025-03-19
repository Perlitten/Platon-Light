# Platon Light Backtesting FAQ

This document addresses frequently asked questions about the Platon Light backtesting module.

## Table of Contents

1. [General Questions](#general-questions)
2. [Data Questions](#data-questions)
3. [Strategy Questions](#strategy-questions)
4. [Performance Questions](#performance-questions)
5. [Optimization Questions](#optimization-questions)
6. [Technical Questions](#technical-questions)

## General Questions

### What is backtesting and why is it important?

Backtesting is the process of testing a trading strategy on historical data to evaluate its performance before risking real capital. It's important because it allows you to:

- Validate trading ideas before implementing them in live markets
- Understand strategy behavior under different market conditions
- Identify and fix potential issues in your strategy
- Optimize parameters for better performance
- Build confidence in your trading approach

### How accurate is backtesting in predicting future performance?

Backtesting provides an indication of how a strategy might perform, but it's not a guarantee of future results. The accuracy depends on:

- **Data quality**: High-quality, clean historical data improves accuracy
- **Realistic assumptions**: Accounting for slippage, commission, and liquidity
- **Avoiding overfitting**: Ensuring the strategy isn't just fitted to past data
- **Market stability**: Whether future market conditions resemble the past

A properly conducted backtest with out-of-sample validation can provide a reasonable estimate of future performance, but always remember that past performance doesn't guarantee future results.

### What's the difference between backtesting and forward testing?

| Backtesting | Forward Testing (Paper Trading) |
|-------------|--------------------------------|
| Uses historical data | Uses real-time data |
| Results available immediately | Takes time to generate results |
| May suffer from look-ahead bias | No look-ahead bias |
| Can test years of data quickly | Limited to real-time progression |
| May not account for all real-world factors | More closely simulates real trading |

Ideally, use both: backtest to develop and refine strategies, then forward test to validate them before live trading.

### How much historical data should I use for backtesting?

The ideal amount of historical data depends on your strategy:

- **Minimum recommendation**: At least 200-300 trades or 1-2 market cycles
- **Long-term strategies**: 5-10+ years of data covering multiple market cycles
- **Short-term strategies**: 1-3 years of high-quality data may be sufficient
- **Intraday strategies**: Several months of tick or minute-level data

More data generally provides more robust results, but ensure the data is relevant to current market conditions. Very old data may represent market dynamics that no longer exist.

## Data Questions

### What data sources does Platon Light support?

Platon Light supports multiple data sources:

- **Exchange APIs**: Direct connection to Binance and other supported exchanges
- **CSV files**: Local CSV files in the specified format
- **Database**: SQL databases with proper schema
- **Custom data sources**: Custom data providers via the data loader interface

To configure data sources, modify the `data_sources` section in your configuration file.

### How do I handle missing or bad data?

Platon Light provides several methods to handle data issues:

```python
# Fill gaps in data
data = data_loader.fill_gaps(data, method='linear')

# Remove outliers
data = data_loader.remove_outliers(data, std_threshold=3)

# Validate data quality
issues = data_loader.validate_data(data)
if issues:
    print(f"Data quality issues found: {issues}")
```

For more advanced data cleaning, you can implement custom preprocessing functions:

```python
def my_preprocessing(data):
    # Custom data cleaning logic
    return cleaned_data

# Apply custom preprocessing
data_loader.set_preprocessing_function(my_preprocessing)
```

### Can I use multiple timeframes in my backtest?

Yes, Platon Light supports multi-timeframe backtesting:

```python
# Configure multiple timeframes
config['backtesting']['timeframes'] = ['1m', '5m', '1h', '1d']

# In your strategy
def generate_signals(self, data, timeframes):
    # Access different timeframe data
    m1_data = timeframes['1m']
    h1_data = timeframes['1h']
    
    # Generate signals using multiple timeframes
    # ...
```

Multi-timeframe backtesting is more computationally intensive but provides a more realistic simulation for strategies that use multiple timeframes.

### How does the data caching system work?

Platon Light implements a data caching system to improve performance:

- **Cache location**: Data is cached in the directory specified in `config['backtesting']['cache_dir']`
- **Cache format**: HDF5 or pickle format for efficient storage and retrieval
- **Cache invalidation**: Cache is automatically invalidated when requesting new date ranges
- **Manual control**: Enable/disable caching via `config['backtesting']['use_cache']`

To manually clear the cache:

```python
data_loader.clear_cache()  # Clear all cached data
data_loader.clear_cache(symbol="BTCUSDT")  # Clear specific symbol
```

## Strategy Questions

### How do I create a custom strategy?

To create a custom strategy:

1. Create a new Python file in the `strategies` directory
2. Inherit from the `BaseStrategy` class
3. Implement required methods
4. Register your strategy with the `StrategyFactory`

Example:

```python
# my_strategy.py
from platon_light.core.base_strategy import BaseStrategy
from platon_light.core.strategy_factory import StrategyFactory

class MyCustomStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        # Initialize strategy parameters
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 30)
    
    def generate_signals(self, data):
        # Calculate indicators
        data['fast_ma'] = data['close'].rolling(self.fast_period).mean()
        data['slow_ma'] = data['close'].rolling(self.slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
        data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1
        
        return data

# Register strategy
StrategyFactory.register_strategy("my_strategy", MyCustomStrategy)
```

Then use your strategy in backtesting:

```python
config['strategy']['name'] = 'my_strategy'
config['strategy']['fast_period'] = 12
config['strategy']['slow_period'] = 26

backtest_engine = BacktestEngine(config)
results = backtest_engine.run(...)
```

### How do I implement position sizing and risk management?

Position sizing and risk management are crucial for realistic backtesting:

```python
class MyStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)  # 1% risk per trade
        self.max_positions = config.get('max_positions', 5)
        
    def calculate_position_size(self, data, index, capital):
        # Get current price
        current_price = data.loc[index, 'close']
        
        # Calculate stop loss level (example: 2% below entry)
        stop_loss_pct = 0.02
        stop_loss = current_price * (1 - stop_loss_pct)
        
        # Calculate risk amount
        risk_amount = capital * self.risk_per_trade
        
        # Calculate position size based on risk
        risk_per_unit = current_price - stop_loss
        position_size = risk_amount / risk_per_unit
        
        # Convert to units (e.g., BTC)
        units = position_size / current_price
        
        return units, stop_loss
```

You can also implement portfolio-level risk management:

```python
def manage_portfolio_risk(self, open_positions, new_signal, capital):
    # Check number of open positions
    if len(open_positions) >= self.max_positions and new_signal > 0:
        return 0  # Don't open new position
    
    # Check correlation with existing positions
    # ...
    
    # Check sector exposure
    # ...
    
    return new_signal  # Return original signal if all checks pass
```

### Can I backtest multiple strategies simultaneously?

Yes, you can backtest multiple strategies for comparison:

```python
# Define strategies
strategies = [
    {"name": "moving_average", "fast_period": 10, "slow_period": 30},
    {"name": "rsi", "rsi_period": 14, "oversold": 30, "overbought": 70},
    {"name": "bollinger", "period": 20, "std_dev": 2}
]

# Run backtests
results = {}
for strategy_config in strategies:
    # Update config with strategy settings
    config['strategy'] = strategy_config
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config)
    
    # Run backtest
    strategy_results = backtest_engine.run(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 12, 31)
    )
    
    # Store results
    results[strategy_config['name']] = strategy_results

# Compare strategies
for name, result in results.items():
    print(f"Strategy: {name}")
    print(f"  Return: {result['metrics']['return_percent']:.2f}%")
    print(f"  Sharpe: {result['metrics']['sharpe_ratio']:.2f}")
    print(f"  Max DD: {result['metrics']['max_drawdown_percent']:.2f}%")
```

You can also use the `StrategyEnsemble` class to combine multiple strategies:

```python
from platon_light.backtesting.strategy_ensemble import StrategyEnsemble

# Create ensemble
ensemble = StrategyEnsemble(config)

# Add strategies
ensemble.add_strategy("moving_average", weight=0.4)
ensemble.add_strategy("rsi", weight=0.3)
ensemble.add_strategy("bollinger", weight=0.3)

# Run ensemble backtest
results = ensemble.run(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31)
)
```

## Performance Questions

### How do I interpret backtest results?

Platon Light provides comprehensive performance metrics:

```python
# Run backtest
results = backtest_engine.run(...)

# Access key metrics
return_pct = results['metrics']['return_percent']
sharpe = results['metrics']['sharpe_ratio']
max_dd = results['metrics']['max_drawdown_percent']
win_rate = results['metrics']['win_rate']

# Print summary
print(f"Total Return: {return_pct:.2f}%")
print(f"Annualized Return: {results['metrics']['annualized_return']:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
```

For a detailed analysis, use the `PerformanceAnalyzer`:

```python
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(config)
detailed_analysis = analyzer.analyze(results)

# Access detailed metrics
monthly_returns = detailed_analysis['monthly_returns']
drawdowns = detailed_analysis['drawdowns']
trade_statistics = detailed_analysis['trade_statistics']
```

Refer to the [Performance Metrics Reference](backtesting_performance_metrics.md) for a complete guide to interpreting metrics.

### How can I visualize backtest results?

Platon Light provides built-in visualization tools:

```python
from platon_light.backtesting.visualization import BacktestVisualizer

# Create visualizer
visualizer = BacktestVisualizer(config)

# Plot equity curve
visualizer.plot_equity_curve(results)

# Plot drawdowns
visualizer.plot_drawdown_chart(results)

# Plot monthly returns heatmap
visualizer.plot_monthly_returns(results)

# Plot trade distribution
visualizer.plot_trade_distribution(results)

# Plot complete performance dashboard
visualizer.plot_performance_dashboard(results)
```

For custom visualizations:

```python
import matplotlib.pyplot as plt
import pandas as pd

# Create equity curve dataframe
equity_df = pd.DataFrame(results['equity_curve'])
equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='ms')
equity_df.set_index('datetime', inplace=True)

# Custom plot
plt.figure(figsize=(12, 6))
plt.plot(equity_df.index, equity_df['equity'], label='Strategy')
plt.plot(equity_df.index, equity_df['benchmark_equity'], label='Benchmark')
plt.title('Strategy vs Benchmark Performance')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### How do I compare my strategy to a benchmark?

Platon Light supports benchmark comparison:

```python
# Configure benchmark
config['backtesting']['benchmark'] = {
    'symbol': 'BTCUSDT',  # Same as strategy or different
    'type': 'buy_hold'    # buy_hold, index, or custom
}

# Run backtest with benchmark
results = backtest_engine.run(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    include_benchmark=True
)

# Access benchmark metrics
benchmark_return = results['benchmark_metrics']['return_percent']
strategy_return = results['metrics']['return_percent']
alpha = results['metrics']['alpha']
beta = results['metrics']['beta']

print(f"Strategy Return: {strategy_return:.2f}%")
print(f"Benchmark Return: {benchmark_return:.2f}%")
print(f"Alpha: {alpha:.4f}")
print(f"Beta: {beta:.4f}")

# Plot comparison
visualizer.plot_benchmark_comparison(results)
```

For custom benchmarks:

```python
# Load custom benchmark data
benchmark_data = data_loader.load_benchmark_data(
    symbol="SPY",  # Example: S&P 500 ETF
    timeframe="1d",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31)
)

# Run backtest with custom benchmark
results = backtest_engine.run(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    benchmark_data=benchmark_data
)
```

## Optimization Questions

### How do I optimize strategy parameters?

Platon Light provides several optimization methods:

```python
from platon_light.backtesting.optimization import StrategyOptimizer

# Initialize optimizer
optimizer = StrategyOptimizer(config)

# Define parameter grid
param_grid = {
    "fast_period": [5, 10, 15, 20],
    "slow_period": [20, 30, 40, 50],
    "signal_threshold": [0.0, 0.001, 0.002]
}

# Run grid search
results = optimizer.grid_search(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 6, 30),
    metric="sharpe_ratio"
)

# Get best parameters
best_params = results["best_params"]
print(f"Best parameters: {best_params}")
print(f"Best Sharpe ratio: {results['best_metrics']['sharpe_ratio']}")

# Validate on out-of-sample data
backtest_engine = BacktestEngine(config)
backtest_engine.strategy.set_parameters(best_params)
validation_results = backtest_engine.run(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 7, 1),
    end_date=datetime(2022, 12, 31)
)
print(f"Out-of-sample Sharpe: {validation_results['metrics']['sharpe_ratio']}")
```

For more efficient optimization of large parameter spaces:

```python
# Genetic algorithm optimization
param_ranges = {
    "fast_period": [5, 30, 1],  # [min, max, step]
    "slow_period": [20, 100, 5],
    "signal_threshold": [0.0, 0.01, 0.001]
}

ga_results = optimizer.genetic_algorithm(
    param_ranges=param_ranges,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 6, 30),
    population_size=50,
    generations=20,
    metric="sharpe_ratio"
)
```

See the [Strategy Optimization Guide](strategy_optimization_guide.md) for detailed instructions.

### How do I avoid overfitting during optimization?

To avoid overfitting:

1. **Use proper data splitting**:
   ```python
   # Split data into train, validation, and test sets
   train_period = (datetime(2020, 1, 1), datetime(2021, 6, 30))
   validation_period = (datetime(2021, 7, 1), datetime(2022, 6, 30))
   test_period = (datetime(2022, 7, 1), datetime(2023, 6, 30))
   ```

2. **Limit parameter combinations**:
   ```python
   # Use fewer parameters with reasonable ranges
   param_grid = {
       "fast_period": [10, 20, 30],  # 3 values instead of 20+
       "slow_period": [40, 60, 80]   # 3 values instead of 20+
   }
   ```

3. **Use walk-forward optimization**:
   ```python
   # Define time windows
   windows = [
       (datetime(2021, 1, 1), datetime(2021, 6, 30), datetime(2021, 7, 1), datetime(2021, 9, 30)),
       (datetime(2021, 4, 1), datetime(2021, 9, 30), datetime(2021, 10, 1), datetime(2021, 12, 31)),
       (datetime(2021, 7, 1), datetime(2021, 12, 31), datetime(2022, 1, 1), datetime(2022, 3, 31))
   ]
   
   # Run walk-forward optimization
   wfo_results = optimizer.walk_forward_optimization(
       param_grid=param_grid,
       symbol="BTCUSDT",
       timeframe="1h",
       windows=windows,
       metric="sharpe_ratio"
   )
   ```

4. **Test across multiple markets**:
   ```python
   # Optimize across multiple symbols
   symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
   multi_results = optimizer.multi_market_optimization(
       param_grid=param_grid,
       symbols=symbols,
       timeframe="1h",
       start_date=datetime(2022, 1, 1),
       end_date=datetime(2022, 6, 30),
       metric="sharpe_ratio"
   )
   ```

5. **Use robust metrics**:
   ```python
   # Define a custom metric that penalizes inconsistency
   def robust_metric(results):
       sharpe = results['metrics']['sharpe_ratio']
       consistency = results['metrics']['monthly_win_rate']
       return sharpe * (consistency / 100)  # Weight by consistency
   
   # Use robust metric for optimization
   results = optimizer.grid_search(
       param_grid=param_grid,
       symbol="BTCUSDT",
       timeframe="1h",
       start_date=datetime(2022, 1, 1),
       end_date=datetime(2022, 6, 30),
       metric=robust_metric
   )
   ```

### Can I optimize multiple objectives simultaneously?

Yes, Platon Light supports multi-objective optimization:

```python
# Define multiple objectives
objectives = [
    {"name": "sharpe_ratio", "weight": 0.6},
    {"name": "max_drawdown_percent", "weight": -0.4}  # Negative weight for metrics to minimize
]

# Run multi-objective optimization
results = optimizer.multi_objective_optimization(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 6, 30),
    objectives=objectives
)

# Get Pareto-optimal solutions
pareto_solutions = results["pareto_solutions"]
for solution in pareto_solutions[:5]:  # Top 5 solutions
    print(f"Parameters: {solution['params']}")
    print(f"Sharpe: {solution['metrics']['sharpe_ratio']:.2f}")
    print(f"Max DD: {solution['metrics']['max_drawdown_percent']:.2f}%")
    print("---")
```

You can also create a custom composite metric:

```python
def custom_objective(results):
    sharpe = results['metrics']['sharpe_ratio']
    drawdown = abs(results['metrics']['max_drawdown_percent'])
    win_rate = results['metrics']['win_rate']
    
    # Custom formula balancing multiple objectives
    score = (sharpe * 0.5) + (win_rate / 100 * 0.3) - (drawdown / 100 * 0.2)
    return score

# Use custom objective
results = optimizer.grid_search(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 6, 30),
    metric=custom_objective
)
```

## Technical Questions

### How do I handle different order types in backtesting?

Platon Light supports various order types:

```python
class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Generate basic signals
        data['signal'] = 0
        # Your signal logic here...
        
        # Add order type information
        data['order_type'] = 'market'  # Default order type
        
        # Set limit orders for some signals
        limit_condition = (data['signal'] == 1) & (some_condition)
        data.loc[limit_condition, 'order_type'] = 'limit'
        data.loc[limit_condition, 'limit_price'] = data['close'] * 0.99  # 1% below close
        
        # Set stop orders for other signals
        stop_condition = (data['signal'] == 1) & (other_condition)
        data.loc[stop_condition, 'order_type'] = 'stop'
        data.loc[stop_condition, 'stop_price'] = data['close'] * 1.02  # 2% above close
        
        return data
```

The backtest engine will simulate these order types accordingly:

- **Market orders**: Executed at the next available price (open, high, low, close depending on configuration)
- **Limit orders**: Executed only if the price reaches the limit price
- **Stop orders**: Executed only if the price reaches the stop price
- **Stop-limit orders**: Combines features of stop and limit orders

Configure order execution in the config:

```yaml
backtesting:
  order_execution:
    market_execution_price: "open"  # open, high, low, close, or vwap
    limit_execution: true
    partial_fills: true
    slippage_model: "fixed"  # fixed, percentage, or custom
```

### How does Platon Light handle slippage and commission?

Slippage and commission are crucial for realistic backtesting:

```yaml
backtesting:
  commission:
    type: "percentage"  # percentage or fixed
    value: 0.001  # 0.1% commission
  slippage:
    type: "percentage"  # percentage, fixed, or custom
    value: 0.0005  # 0.05% slippage
```

For more advanced models:

```python
# Custom slippage model
def custom_slippage(price, volume, symbol, side):
    """Custom slippage model based on volume"""
    base_slippage = 0.0001  # 0.01% base slippage
    volume_factor = min(1.0, volume / 100000)  # Normalize volume
    return price * (base_slippage + (volume_factor * 0.001))

# Set custom slippage model
backtest_engine.set_slippage_model(custom_slippage)

# Custom commission model
def custom_commission(price, volume, symbol, side):
    """Tiered commission structure"""
    if volume * price < 10000:
        return volume * price * 0.002  # 0.2% for small trades
    else:
        return volume * price * 0.001  # 0.1% for large trades

# Set custom commission model
backtest_engine.set_commission_model(custom_commission)
```

### Can I run backtests in parallel?

Yes, Platon Light supports parallel backtesting for improved performance:

```python
from platon_light.backtesting.parallel import ParallelBacktester

# Initialize parallel backtester
parallel_tester = ParallelBacktester(config, n_jobs=-1)  # Use all available cores

# Define backtest tasks
tasks = [
    {"symbol": "BTCUSDT", "timeframe": "1h", "start_date": datetime(2022, 1, 1), "end_date": datetime(2022, 12, 31)},
    {"symbol": "ETHUSDT", "timeframe": "1h", "start_date": datetime(2022, 1, 1), "end_date": datetime(2022, 12, 31)},
    {"symbol": "BNBUSDT", "timeframe": "1h", "start_date": datetime(2022, 1, 1), "end_date": datetime(2022, 12, 31)}
]

# Run parallel backtests
results = parallel_tester.run_parallel(tasks)

# Access results
for symbol, result in results.items():
    print(f"Symbol: {symbol}")
    print(f"Return: {result['metrics']['return_percent']:.2f}%")
```

For parallel optimization:

```python
# Enable parallel processing for optimization
config['optimization']['parallel'] = True
config['optimization']['n_jobs'] = -1  # Use all available cores

# Initialize optimizer with parallel processing
optimizer = StrategyOptimizer(config)

# Run optimization in parallel
results = optimizer.grid_search(param_grid, ...)
```

### How do I save and load backtest results?

Platon Light provides functionality to save and load backtest results:

```python
# Save backtest results
from platon_light.backtesting.utils import save_results

# Run backtest
results = backtest_engine.run(...)

# Save results
save_results(results, "my_backtest_results.pkl")

# Load results
from platon_light.backtesting.utils import load_results

loaded_results = load_results("my_backtest_results.pkl")

# Generate report from saved results
from platon_light.backtesting.reporting import generate_report

report = generate_report(loaded_results, "My Backtest Report")
report.save("backtest_report.html")
```

For automated backtest versioning:

```python
from platon_light.backtesting.versioning import BacktestVersionManager

# Initialize version manager
version_manager = BacktestVersionManager(config)

# Run backtest with versioning
results = backtest_engine.run(...)
version_id = version_manager.save_version(results, "Initial RSI strategy test")

# List available versions
versions = version_manager.list_versions()
for v in versions:
    print(f"ID: {v['id']}, Date: {v['date']}, Description: {v['description']}")

# Load specific version
loaded_results = version_manager.load_version(version_id)

# Compare versions
comparison = version_manager.compare_versions(version_id1, version_id2)
print(f"Return difference: {comparison['return_diff']:.2f}%")
```

---

If you have additional questions not covered in this FAQ, please refer to the detailed documentation or contact support.
