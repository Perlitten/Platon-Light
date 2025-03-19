# Strategy Optimization Guide

This guide walks you through the process of optimizing trading strategies using the Platon Light backtesting module.

## Table of Contents

1. [Introduction to Strategy Optimization](#introduction-to-strategy-optimization)
2. [Optimization Methods](#optimization-methods)
3. [Parameter Optimization Workflow](#parameter-optimization-workflow)
4. [Walk-Forward Optimization](#walk-forward-optimization)
5. [Multi-Market Optimization](#multi-market-optimization)
6. [Avoiding Overfitting](#avoiding-overfitting)
7. [Optimization Case Studies](#optimization-case-studies)

## Introduction to Strategy Optimization

Strategy optimization is the process of finding the best parameters for a trading strategy to maximize performance metrics like returns, Sharpe ratio, or other custom objectives.

### Why Optimize?

- **Improve Performance**: Find parameter combinations that yield better results
- **Understand Sensitivity**: Learn how strategy performance varies with different parameters
- **Validate Robustness**: Ensure strategy works across different market conditions
- **Adapt to Markets**: Update parameters as market conditions change

### Optimization vs. Curve Fitting

- **Optimization**: Finding genuinely better parameters that work across various market conditions
- **Curve Fitting**: Overfitting parameters to historical data that won't work in the future

The goal is to optimize without curve fitting, which we'll address throughout this guide.

## Optimization Methods

The Platon Light backtesting module offers several optimization methods:

### Grid Search

Exhaustively tests all combinations of parameters within specified ranges.

```python
from platon_light.backtesting.optimization import StrategyOptimizer

# Initialize optimizer
optimizer = StrategyOptimizer(config)

# Define parameter grid
param_grid = {
    "rsi_period": [7, 14, 21],
    "rsi_overbought": [70, 75, 80],
    "rsi_oversold": [20, 25, 30],
    "stop_loss_pct": [1.0, 2.0, 3.0]
}

# Run grid search
results = optimizer.grid_search(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    metric="sharpe_ratio"  # Optimization objective
)

# Get best parameters
best_params = results["best_params"]
best_metrics = results["best_metrics"]
print(f"Best parameters: {best_params}")
print(f"Best Sharpe ratio: {best_metrics['sharpe_ratio']}")
```

**Pros**:
- Simple to understand and implement
- Guaranteed to find global optimum within the grid
- Easy to visualize results

**Cons**:
- Computationally expensive for many parameters
- "Curse of dimensionality" - exponential growth in combinations

### Genetic Algorithm

Uses evolutionary principles to efficiently search the parameter space.

```python
# Define parameter ranges
param_ranges = {
    "rsi_period": [5, 30, 1],  # [min, max, step]
    "rsi_overbought": [60, 90, 5],
    "rsi_oversold": [10, 40, 5],
    "stop_loss_pct": [0.5, 5.0, 0.5]
}

# Run genetic algorithm
results = optimizer.genetic_algorithm(
    param_ranges=param_ranges,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    population_size=50,
    generations=20,
    mutation_rate=0.1,
    crossover_rate=0.8,
    metric="sharpe_ratio"
)
```

**Pros**:
- More efficient for large parameter spaces
- Can find good solutions without testing all combinations
- Works well with non-linear relationships

**Cons**:
- May not find the absolute global optimum
- Results can vary between runs
- More complex to configure

### Bayesian Optimization

Uses probabilistic model to efficiently search parameter space by learning from previous evaluations.

```python
# Define parameter ranges
param_bounds = {
    "rsi_period": (5, 30),  # (min, max)
    "rsi_overbought": (60, 90),
    "rsi_oversold": (10, 40),
    "stop_loss_pct": (0.5, 5.0)
}

# Run Bayesian optimization
results = optimizer.bayesian_optimization(
    param_bounds=param_bounds,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    n_iterations=50,
    initial_points=10,
    metric="sharpe_ratio"
)
```

**Pros**:
- Very efficient for expensive-to-evaluate functions
- Balances exploration and exploitation
- Works well with noisy objective functions

**Cons**:
- More complex mathematically
- May struggle with discrete parameters
- Requires careful tuning of the acquisition function

## Parameter Optimization Workflow

Follow this step-by-step workflow for effective strategy optimization:

### 1. Define Optimization Objectives

Choose the metrics to optimize for:

```python
# Common optimization metrics
optimization_metrics = [
    "sharpe_ratio",  # Risk-adjusted returns
    "sortino_ratio",  # Downside risk-adjusted returns
    "calmar_ratio",  # Drawdown-adjusted returns
    "return_percent",  # Total return percentage
    "profit_factor",  # Gross profit / gross loss
    "expectancy"  # Expected return per trade
]

# Custom metric function
def custom_metric(results):
    """Custom metric balancing returns and drawdown"""
    return (results["metrics"]["return_percent"] * 0.7) - \
           (abs(results["metrics"]["max_drawdown_percent"]) * 0.3)

# Use custom metric
results = optimizer.grid_search(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    metric=custom_metric  # Custom optimization objective
)
```

### 2. Select Parameters to Optimize

Choose which parameters to optimize and their ranges:

```python
# For a moving average crossover strategy
param_grid = {
    # Essential parameters - wide ranges
    "fast_ma_period": [5, 10, 15, 20, 25, 30],
    "slow_ma_period": [20, 30, 40, 50, 60],
    
    # Secondary parameters - narrower ranges
    "stop_loss_pct": [1.0, 2.0, 3.0],
    "take_profit_pct": [2.0, 3.0, 4.0],
    
    # Fixed parameters (not included in grid)
    # "position_size_pct": 0.1  # Set in config
}
```

### 3. Split Data Properly

Split data into in-sample (optimization) and out-of-sample (validation) periods:

```python
# Load full dataset
data = data_loader.load_data(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2023, 1, 1)
)

# Split into in-sample and out-of-sample
split_date = datetime(2022, 6, 1)  # 18 months in-sample, 6 months out-of-sample

# Optimize on in-sample data
results = optimizer.grid_search(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2021, 1, 1),
    end_date=split_date,
    metric="sharpe_ratio"
)

# Test best parameters on out-of-sample data
best_params = results["best_params"]
backtest_engine = BacktestEngine(config)
backtest_engine.strategy.set_parameters(best_params)
out_of_sample_results = backtest_engine.run(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=split_date,
    end_date=datetime(2023, 1, 1)
)

# Compare performance
print(f"In-sample Sharpe: {results['best_metrics']['sharpe_ratio']}")
print(f"Out-of-sample Sharpe: {out_of_sample_results['metrics']['sharpe_ratio']}")
```

### 4. Analyze Optimization Results

Visualize and analyze optimization results:

```python
# Get all results from grid search
all_results = results["all_results"]

# Create parameter heatmap
optimizer.plot_parameter_heatmap(
    all_results,
    x_param="fast_ma_period",
    y_param="slow_ma_period",
    metric="sharpe_ratio"
)

# Plot parameter sensitivity
optimizer.plot_parameter_sensitivity(
    all_results,
    param="stop_loss_pct",
    metric="sharpe_ratio"
)

# 3D surface plot
optimizer.plot_3d_surface(
    all_results,
    x_param="fast_ma_period",
    y_param="slow_ma_period",
    z_metric="sharpe_ratio"
)
```

### 5. Validate Robustness

Test optimized strategy across different conditions:

```python
# Test across different market conditions
market_conditions = [
    ("Bull Market", datetime(2020, 4, 1), datetime(2021, 4, 1)),
    ("Sideways Market", datetime(2019, 1, 1), datetime(2019, 12, 31)),
    ("Bear Market", datetime(2022, 1, 1), datetime(2022, 12, 31))
]

for condition, start, end in market_conditions:
    condition_results = backtest_engine.run(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date=start,
        end_date=end
    )
    print(f"{condition} - Sharpe: {condition_results['metrics']['sharpe_ratio']}")
    print(f"{condition} - Return: {condition_results['metrics']['return_percent']}%")
    print(f"{condition} - Max DD: {condition_results['metrics']['max_drawdown_percent']}%")
    print("---")

# Test across different symbols
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
for symbol in symbols:
    symbol_results = backtest_engine.run(
        symbol=symbol,
        timeframe="1h",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 12, 31)
    )
    print(f"{symbol} - Sharpe: {symbol_results['metrics']['sharpe_ratio']}")
    print(f"{symbol} - Return: {symbol_results['metrics']['return_percent']}%")
```

## Walk-Forward Optimization

Walk-forward optimization (WFO) is a technique to reduce overfitting by continuously re-optimizing parameters as new data becomes available.

### Basic Walk-Forward Optimization

```python
# Define time windows
windows = [
    (datetime(2021, 1, 1), datetime(2021, 6, 30), datetime(2021, 7, 1), datetime(2021, 9, 30)),
    (datetime(2021, 4, 1), datetime(2021, 9, 30), datetime(2021, 10, 1), datetime(2021, 12, 31)),
    (datetime(2021, 7, 1), datetime(2021, 12, 31), datetime(2022, 1, 1), datetime(2022, 3, 31)),
    (datetime(2021, 10, 1), datetime(2022, 3, 31), datetime(2022, 4, 1), datetime(2022, 6, 30))
]

# Run walk-forward optimization
wfo_results = optimizer.walk_forward_optimization(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    windows=windows,
    metric="sharpe_ratio"
)

# Analyze WFO results
for i, window_result in enumerate(wfo_results["window_results"]):
    print(f"Window {i+1}:")
    print(f"  Best parameters: {window_result['best_params']}")
    print(f"  In-sample Sharpe: {window_result['in_sample_metrics']['sharpe_ratio']}")
    print(f"  Out-of-sample Sharpe: {window_result['out_of_sample_metrics']['sharpe_ratio']}")
    print("---")

# Get combined out-of-sample performance
combined_performance = wfo_results["combined_performance"]
print(f"Combined out-of-sample Sharpe: {combined_performance['sharpe_ratio']}")
print(f"Combined out-of-sample Return: {combined_performance['return_percent']}%")
```

### Anchored Walk-Forward Optimization

Anchored WFO keeps the start date fixed and extends the optimization window:

```python
# Define anchored windows
anchored_windows = [
    (datetime(2021, 1, 1), datetime(2021, 6, 30), datetime(2021, 7, 1), datetime(2021, 9, 30)),
    (datetime(2021, 1, 1), datetime(2021, 9, 30), datetime(2021, 10, 1), datetime(2021, 12, 31)),
    (datetime(2021, 1, 1), datetime(2021, 12, 31), datetime(2022, 1, 1), datetime(2022, 3, 31)),
    (datetime(2021, 1, 1), datetime(2022, 3, 31), datetime(2022, 4, 1), datetime(2022, 6, 30))
]

# Run anchored walk-forward optimization
anchored_wfo_results = optimizer.walk_forward_optimization(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    windows=anchored_windows,
    metric="sharpe_ratio"
)
```

## Multi-Market Optimization

Optimize across multiple markets simultaneously to find robust parameters:

```python
# Define symbols to optimize across
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Run multi-market optimization
multi_market_results = optimizer.multi_market_optimization(
    param_grid=param_grid,
    symbols=symbols,
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    metric="sharpe_ratio",
    aggregation_method="mean"  # Options: "mean", "min", "median"
)

# Get best parameters across all markets
best_params = multi_market_results["best_params"]
print(f"Best parameters across all markets: {best_params}")

# Test best parameters on each market
for symbol in symbols:
    backtest_engine.strategy.set_parameters(best_params)
    symbol_results = backtest_engine.run(
        symbol=symbol,
        timeframe="1h",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 12, 31)
    )
    print(f"{symbol} - Sharpe: {symbol_results['metrics']['sharpe_ratio']}")
```

## Avoiding Overfitting

Strategies to avoid overfitting during optimization:

### 1. Use Proper Validation

Always validate on out-of-sample data:

```python
# Proper validation workflow
# 1. Split data into training, validation, and test sets
train_period = (datetime(2020, 1, 1), datetime(2021, 6, 30))
validation_period = (datetime(2021, 7, 1), datetime(2022, 6, 30))
test_period = (datetime(2022, 7, 1), datetime(2023, 6, 30))

# 2. Optimize on training data
train_results = optimizer.grid_search(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=train_period[0],
    end_date=train_period[1],
    metric="sharpe_ratio"
)

# 3. Select top N parameter sets
top_params = optimizer.get_top_parameters(
    all_results=train_results["all_results"],
    metric="sharpe_ratio",
    n=5
)

# 4. Evaluate top parameters on validation data
validation_performances = []
for params in top_params:
    backtest_engine.strategy.set_parameters(params)
    validation_results = backtest_engine.run(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date=validation_period[0],
        end_date=validation_period[1]
    )
    validation_performances.append({
        "params": params,
        "sharpe": validation_results["metrics"]["sharpe_ratio"]
    })

# 5. Select best performing on validation
best_validation_params = max(validation_performances, key=lambda x: x["sharpe"])["params"]

# 6. Final evaluation on test data
backtest_engine.strategy.set_parameters(best_validation_params)
test_results = backtest_engine.run(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=test_period[0],
    end_date=test_period[1]
)
print(f"Final test Sharpe: {test_results['metrics']['sharpe_ratio']}")
```

### 2. Limit Parameter Combinations

Reduce the parameter space to avoid overfitting:

```python
# Instead of this (1,000,000 combinations)
excessive_param_grid = {
    "ma_type": ["sma", "ema", "wma", "dema", "tema"],
    "fast_period": list(range(5, 55, 1)),  # 50 values
    "slow_period": list(range(10, 210, 2)),  # 100 values
    "signal_period": list(range(3, 23, 1)),  # 20 values
    "stop_loss_pct": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # 10 values
}

# Use this (720 combinations)
reasonable_param_grid = {
    "ma_type": ["sma", "ema"],  # 2 values
    "fast_period": [5, 10, 15, 20, 25, 30],  # 6 values
    "slow_period": [30, 50, 70, 100, 150, 200],  # 6 values
    "signal_period": [5, 9, 14],  # 3 values
    "stop_loss_pct": [1.0, 2.0, 3.0, 4.0]  # 4 values
}
```

### 3. Use Robust Optimization Objectives

Choose metrics that penalize overfitting:

```python
# Custom robustness-focused metric
def robustness_metric(results):
    """Metric that rewards consistency across different periods"""
    # Split results into sub-periods
    periods = split_results_into_periods(results)
    
    # Calculate Sharpe ratio for each period
    period_sharpes = [calculate_sharpe(period) for period in periods]
    
    # Calculate mean and standard deviation of Sharpe ratios
    mean_sharpe = sum(period_sharpes) / len(period_sharpes)
    std_sharpe = (sum((s - mean_sharpe)**2 for s in period_sharpes) / len(period_sharpes))**0.5
    
    # Reward high mean and penalize high standard deviation
    return mean_sharpe - std_sharpe

# Use robustness metric
results = optimizer.grid_search(
    param_grid=param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 12, 31),
    metric=robustness_metric
)
```

### 4. Analyze Parameter Sensitivity

Examine how sensitive performance is to parameter changes:

```python
# Get all results from optimization
all_results = results["all_results"]

# Analyze parameter sensitivity
sensitivity = optimizer.analyze_parameter_sensitivity(
    all_results=all_results,
    metric="sharpe_ratio"
)

# Print sensitivity for each parameter
for param, sensitivity_score in sensitivity.items():
    print(f"Parameter: {param}, Sensitivity: {sensitivity_score}")

# Choose parameters with lower sensitivity
robust_params = {
    param: results["best_params"][param] 
    for param in results["best_params"] 
    if sensitivity[param] < 0.5  # Low sensitivity threshold
}
```

## Optimization Case Studies

### Case Study 1: RSI Strategy Optimization

```python
# Define parameter grid for RSI strategy
rsi_param_grid = {
    "rsi_period": [7, 14, 21],
    "rsi_overbought": [70, 75, 80],
    "rsi_oversold": [20, 25, 30],
    "exit_rsi": [40, 50, 60]
}

# Run optimization
results = optimizer.grid_search(
    param_grid=rsi_param_grid,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    metric="sharpe_ratio"
)

# Best parameters
best_params = results["best_params"]
print(f"Best RSI parameters: {best_params}")

# Visualize parameter relationships
optimizer.plot_parameter_heatmap(
    all_results=results["all_results"],
    x_param="rsi_period",
    y_param="rsi_oversold",
    metric="sharpe_ratio"
)
```

### Case Study 2: Multi-Timeframe Strategy

```python
# Define parameter grid for multi-timeframe strategy
mtf_param_grid = {
    "fast_tf": ["5m", "15m", "30m"],
    "slow_tf": ["1h", "4h", "1d"],
    "fast_ma_period": [10, 20, 30],
    "slow_ma_period": [50, 100, 200]
}

# Custom validation function to ensure fast_tf < slow_tf
def validate_params(params):
    tf_order = {"5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1d": 6}
    return tf_order[params["fast_tf"]] < tf_order[params["slow_tf"]]

# Run optimization with validation
results = optimizer.grid_search(
    param_grid=mtf_param_grid,
    symbol="BTCUSDT",
    timeframe="1h",  # Base timeframe
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    metric="sharpe_ratio",
    validate_params=validate_params
)
```

By following this guide, you'll be able to effectively optimize your trading strategies while avoiding common pitfalls like overfitting. Remember that the goal is to find robust parameters that work well across different market conditions, not just parameters that fit historical data perfectly.
