# Platon Light Backtesting Troubleshooting Guide

This guide provides solutions for common issues encountered when using the Platon Light backtesting module.

## Table of Contents

1. [Data Loading Issues](#data-loading-issues)
2. [Backtesting Engine Issues](#backtesting-engine-issues)
3. [Strategy Issues](#strategy-issues)
4. [Performance Issues](#performance-issues)
5. [Visualization Issues](#visualization-issues)
6. [Optimization Issues](#optimization-issues)
7. [CLI Issues](#cli-issues)
8. [Configuration Issues](#configuration-issues)
9. [Common Error Messages](#common-error-messages)
10. [Debugging Tips](#debugging-tips)

## Data Loading Issues

### No Data Available

**Symptoms:**
- "No data available for backtest" error message
- Empty DataFrame returned by `data_loader.load_data()`

**Possible Causes:**
1. Invalid symbol or timeframe
2. Date range with no available data
3. API connection issues
4. Missing local data files

**Solutions:**
1. **Verify Symbol and Timeframe**
   ```python
   # Check available symbols
   available_symbols = data_loader.get_available_symbols()
   print(available_symbols)
   
   # Check available timeframes
   available_timeframes = data_loader.get_available_timeframes()
   print(available_timeframes)
   ```

2. **Check Date Range**
   ```python
   # Try loading data for a known good date range
   data = data_loader.load_data(symbol, timeframe, 
                               datetime(2023, 1, 1), 
                               datetime(2023, 1, 31))
   ```

3. **Force Data Download**
   ```python
   # Force download from API
   data = data_loader.download_data(symbol, timeframe, 
                                   start_date, end_date)
   ```

4. **Check Data Directory**
   ```python
   # Check if data directory exists and has files
   data_dir = Path(config["backtesting"]["data_dir"])
   print(f"Data directory exists: {data_dir.exists()}")
   print(f"Files in directory: {list(data_dir.glob('*.csv'))}")
   ```

### Data Quality Issues

**Symptoms:**
- Gaps in data
- Abnormal price values
- Missing indicators

**Solutions:**
1. **Check for Gaps**
   ```python
   # Check for gaps in timestamp
   data['timestamp_diff'] = data['timestamp'].diff()
   gaps = data[data['timestamp_diff'] > expected_interval]
   print(f"Found {len(gaps)} gaps in data")
   ```

2. **Check for Outliers**
   ```python
   # Check for price outliers
   mean = data['close'].mean()
   std = data['close'].std()
   outliers = data[(data['close'] > mean + 3*std) | (data['close'] < mean - 3*std)]
   print(f"Found {len(outliers)} outliers in price data")
   ```

3. **Fill Gaps**
   ```python
   # Fill gaps in data
   data = data_loader.fill_gaps(data, method='linear')
   ```

## Backtesting Engine Issues

### No Trades Generated

**Symptoms:**
- Backtest runs without errors but no trades are generated
- `results['trades']` is empty

**Possible Causes:**
1. Strategy not generating signals
2. Signal thresholds too strict
3. Risk management preventing trade execution

**Solutions:**
1. **Check Signal Generation**
   ```python
   # Run strategy signal generation separately
   data = data_loader.load_data(symbol, timeframe, start_date, end_date)
   data = backtest_engine.strategy.generate_signals(data)
   
   # Check if signals are generated
   signal_count = (data['signal'] != 0).sum()
   print(f"Generated {signal_count} signals")
   ```

2. **Adjust Strategy Parameters**
   ```python
   # Adjust strategy parameters to be less strict
   config['strategy']['rsi_oversold'] = 35  # Less strict oversold threshold
   config['strategy']['rsi_overbought'] = 65  # Less strict overbought threshold
   
   # Reinitialize with new config
   backtest_engine = BacktestEngine(config)
   ```

3. **Check Risk Management Settings**
   ```python
   # Adjust risk management settings
   config['risk_management']['max_position_size'] = 0.2  # Increase max position size
   config['risk_management']['max_open_positions'] = 5  # Increase max open positions
   
   # Reinitialize with new config
   backtest_engine = BacktestEngine(config)
   ```

### Unrealistic Results

**Symptoms:**
- Extremely high returns
- Perfect win rate
- No drawdowns

**Possible Causes:**
1. Look-ahead bias in strategy
2. Unrealistic commission/slippage settings
3. Overfitting to historical data

**Solutions:**
1. **Check for Look-ahead Bias**
   ```python
   # Ensure strategy only uses past data
   def generate_signals(self, data):
       for i in range(len(data)):
           # Only use data up to current index
           current_data = data.iloc[:i+1]
           # Generate signal based only on current_data
   ```

2. **Use Realistic Commission and Slippage**
   ```python
   # Set realistic commission and slippage
   config['backtesting']['commission'] = 0.001  # 0.1%
   config['backtesting']['slippage'] = 0.0005  # 0.05%
   
   # Reinitialize with new config
   backtest_engine = BacktestEngine(config)
   ```

3. **Implement Out-of-Sample Testing**
   ```python
   # Split data into in-sample and out-of-sample
   split_date = start_date + (end_date - start_date) * 0.7
   
   # Train on in-sample
   in_sample_results = backtest_engine.run(symbol, timeframe, start_date, split_date)
   
   # Test on out-of-sample
   out_sample_results = backtest_engine.run(symbol, timeframe, split_date, end_date)
   
   # Compare results
   print(f"In-sample return: {in_sample_results['metrics']['return_percent']:.2f}%")
   print(f"Out-sample return: {out_sample_results['metrics']['return_percent']:.2f}%")
   ```

## Strategy Issues

### Strategy Not Found

**Symptoms:**
- "Unsupported strategy: [strategy_name]" error message

**Solutions:**
1. **Check Strategy Name**
   ```python
   # Print available strategies
   from platon_light.core.strategy_factory import StrategyFactory
   print(f"Available strategies: {StrategyFactory.get_available_strategies()}")
   ```

2. **Register Custom Strategy**
   ```python
   # Add your strategy to the factory
   from platon_light.core.strategy_factory import StrategyFactory
   from my_custom_strategy import MyCustomStrategy
   
   # Register strategy
   StrategyFactory.register_strategy("my_strategy", MyCustomStrategy)
   ```

### Strategy Parameter Issues

**Symptoms:**
- Strategy behaves unexpectedly
- Parameters not taking effect

**Solutions:**
1. **Check Parameter Loading**
   ```python
   # Print loaded parameters
   print("Strategy parameters:")
   for param, value in backtest_engine.strategy.__dict__.items():
       if not param.startswith('_'):
           print(f"  {param}: {value}")
   ```

2. **Override Parameters Directly**
   ```python
   # Override parameters directly
   backtest_engine.strategy.rsi_period = 14
   backtest_engine.strategy.rsi_oversold = 30
   backtest_engine.strategy.rsi_overbought = 70
   ```

## Performance Issues

### Slow Backtesting

**Symptoms:**
- Backtests take a long time to complete
- High memory usage

**Solutions:**
1. **Use Cached Data**
   ```python
   # Enable data caching
   config['backtesting']['use_cache'] = True
   data_loader = DataLoader(config)
   ```

2. **Reduce Date Range**
   ```python
   # Use a smaller date range
   start_date = datetime(2023, 1, 1)
   end_date = datetime(2023, 1, 31)  # One month instead of longer period
   ```

3. **Optimize Strategy Calculations**
   ```python
   # Use vectorized operations instead of loops
   # Instead of:
   for i in range(len(data)):
       data.loc[i, 'signal'] = calculate_signal(data.iloc[i])
   
   # Use:
   data['signal'] = calculate_signal_vectorized(data)
   ```

4. **Use Parallel Processing for Optimization**
   ```python
   # Enable parallel processing
   config['optimization']['parallel'] = True
   config['optimization']['n_jobs'] = -1  # Use all available cores
   optimizer = StrategyOptimizer(config)
   ```

## Visualization Issues

### Plots Not Showing

**Symptoms:**
- No plots displayed
- Empty figures

**Solutions:**
1. **Check Matplotlib Backend**
   ```python
   import matplotlib.pyplot as plt
   print(f"Current backend: {plt.get_backend()}")
   
   # Set a different backend if needed
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc.
   ```

2. **Save Plots to File**
   ```python
   # Save plots to file instead of displaying
   visualizer.plot_equity_curve(results, save=True, show=False)
   print(f"Plot saved to: {visualizer.output_dir}")
   ```

3. **Check Data Validity**
   ```python
   # Check if equity curve data exists
   if not results['equity_curve']:
       print("No equity curve data available")
   else:
       print(f"Equity curve has {len(results['equity_curve'])} data points")
   ```

### Incorrect Visualizations

**Symptoms:**
- Plots show incorrect data
- Axis labels or scales are wrong

**Solutions:**
1. **Check Data Format**
   ```python
   # Check data format
   print("Equity curve data format:")
   print(results['equity_curve'][0])
   
   # Ensure timestamps are converted to datetime
   import pandas as pd
   equity_df = pd.DataFrame(results['equity_curve'])
   equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='ms')
   ```

2. **Customize Plot Settings**
   ```python
   # Customize plot settings
   visualizer.plot_equity_curve(
       results,
       title="Custom Equity Curve",
       xlabel="Date",
       ylabel="Equity ($)",
       figsize=(12, 6),
       grid=True
   )
   ```

## Optimization Issues

### Optimization Not Improving Results

**Symptoms:**
- Optimization completes but doesn't improve strategy performance
- All parameter combinations give similar results

**Possible Causes:**
1. Parameter ranges too narrow
2. Strategy not sensitive to parameters
3. Overfitting to noise

**Solutions:**
1. **Widen Parameter Ranges**
   ```python
   # Use wider parameter ranges
   param_grid = {
       "rsi_period": [7, 14, 21, 28],  # More values
       "rsi_oversold": [20, 25, 30, 35, 40],  # More values
       "rsi_overbought": [60, 65, 70, 75, 80]  # More values
   }
   ```

2. **Use Different Optimization Metric**
   ```python
   # Optimize for a different metric
   optimizer.grid_search(param_grid, symbol, timeframe, start_date, end_date, 
                        metric="sharpe_ratio")  # Instead of return_percent
   ```

3. **Implement Cross-Validation**
   ```python
   # Use time-series cross-validation
   from sklearn.model_selection import TimeSeriesSplit
   
   # Split data into multiple train/test sets
   tscv = TimeSeriesSplit(n_splits=5)
   
   # For each split
   for train_idx, test_idx in tscv.split(data):
       train_data = data.iloc[train_idx]
       test_data = data.iloc[test_idx]
       
       # Run optimization on train data
       train_results = optimizer.grid_search(param_grid, train_data)
       
       # Test best parameters on test data
       test_results = backtest_engine.run_on_data(test_data, train_results['best_params'])
       
       print(f"Train return: {train_results['best_metrics']['return_percent']:.2f}%")
       print(f"Test return: {test_results['metrics']['return_percent']:.2f}%")
   ```

### Optimization Taking Too Long

**Symptoms:**
- Optimization process is very slow
- High CPU/memory usage

**Solutions:**
1. **Reduce Parameter Combinations**
   ```python
   # Reduce number of parameter combinations
   param_grid = {
       "rsi_period": [7, 14, 21],  # Fewer values
       "rsi_oversold": [30],  # Fixed value
       "rsi_overbought": [70]  # Fixed value
   }
   ```

2. **Use Genetic Algorithm Instead of Grid Search**
   ```python
   # Use genetic algorithm for faster optimization
   param_ranges = {
       "rsi_period": [5, 30, 1],  # [min, max, step]
       "rsi_oversold": [20, 40, 5],
       "rsi_overbought": [60, 80, 5]
   }
   
   results = optimizer.genetic_algorithm(
       param_ranges, 
       symbol, 
       timeframe, 
       start_date, 
       end_date,
       population_size=20,  # Smaller population
       generations=5  # Fewer generations
   )
   ```

3. **Use Smaller Dataset for Initial Optimization**
   ```python
   # Use a smaller dataset for initial optimization
   small_start_date = datetime(2023, 1, 1)
   small_end_date = datetime(2023, 1, 31)  # One month
   
   # Run initial optimization on small dataset
   initial_results = optimizer.grid_search(param_grid, symbol, timeframe, 
                                         small_start_date, small_end_date)
   
   # Fine-tune on full dataset with reduced parameter grid
   reduced_param_grid = {
       param: [initial_results['best_params'][param]] for param in param_grid
   }
   
   # Add small variations around best values
   for param, value in initial_results['best_params'].items():
       if isinstance(value, int):
           reduced_param_grid[param] = [value-1, value, value+1]
       elif isinstance(value, float):
           reduced_param_grid[param] = [value*0.9, value, value*1.1]
   
   # Run fine-tuning on full dataset
   final_results = optimizer.grid_search(reduced_param_grid, symbol, timeframe, 
                                       start_date, end_date)
   ```

## CLI Issues

### Command-Line Arguments Not Working

**Symptoms:**
- CLI arguments not taking effect
- Default values used instead of provided arguments

**Solutions:**
1. **Check Argument Parsing**
   ```bash
   # Run with verbose output
   python -m platon_light.backtesting.cli --config backtest_config.yaml --verbose
   ```

2. **Use Environment Variables Instead**
   ```bash
   # Set environment variables
   export PLATON_INITIAL_CAPITAL=20000
   export PLATON_COMMISSION=0.001
   
   # Run CLI
   python -m platon_light.backtesting.cli --config backtest_config.yaml
   ```

3. **Check Configuration Override Logic**
   ```python
   # Print final configuration
   python -m platon_light.backtesting.cli --config backtest_config.yaml --print-config
   ```

### CLI Command Fails

**Symptoms:**
- CLI command exits with error
- No output generated

**Solutions:**
1. **Run with Debug Logging**
   ```bash
   # Enable debug logging
   python -m platon_light.backtesting.cli --config backtest_config.yaml --log-level DEBUG
   ```

2. **Check Required Arguments**
   ```bash
   # Show help to see required arguments
   python -m platon_light.backtesting.cli --help
   ```

3. **Run Step by Step**
   ```python
   # Run each step manually
   from platon_light.backtesting.cli import parse_args, load_config, run_backtest
   
   # Parse arguments
   args = parse_args(['--config', 'backtest_config.yaml'])
   
   # Load configuration
   config = load_config(args.config)
   
   # Run backtest
   results = run_backtest(config, args)
   ```

## Configuration Issues

### Configuration Not Loading

**Symptoms:**
- "Configuration file not found" error
- Default configuration used instead of custom configuration

**Solutions:**
1. **Check File Path**
   ```python
   import os
   config_path = "backtest_config.yaml"
   print(f"Config file exists: {os.path.exists(config_path)}")
   print(f"Current working directory: {os.getcwd()}")
   ```

2. **Use Absolute Path**
   ```python
   from pathlib import Path
   config_path = Path.cwd() / "backtest_config.yaml"
   print(f"Absolute path: {config_path}")
   ```

3. **Create Default Configuration**
   ```python
   import yaml
   
   # Create default configuration
   default_config = {
       "backtesting": {
           "initial_capital": 10000,
           "commission": 0.001,
           "slippage": 0.0005
       },
       "strategy": {
           "name": "scalping",
           "rsi_period": 14
       }
   }
   
   # Save default configuration
   with open("backtest_config.yaml", "w") as f:
       yaml.dump(default_config, f)
   ```

### Invalid Configuration Values

**Symptoms:**
- "Invalid configuration value" error
- Unexpected behavior due to incorrect configuration

**Solutions:**
1. **Validate Configuration**
   ```python
   def validate_config(config):
       """Validate configuration values"""
       errors = []
       
       # Check required sections
       required_sections = ["backtesting", "strategy"]
       for section in required_sections:
           if section not in config:
               errors.append(f"Missing required section: {section}")
       
       # Check backtesting parameters
       if "backtesting" in config:
           backtesting = config["backtesting"]
           if "initial_capital" in backtesting and backtesting["initial_capital"] <= 0:
               errors.append("initial_capital must be positive")
           if "commission" in backtesting and (backtesting["commission"] < 0 or backtesting["commission"] > 1):
               errors.append("commission must be between 0 and 1")
       
       # Check strategy parameters
       if "strategy" in config:
           strategy = config["strategy"]
           if "name" not in strategy:
               errors.append("strategy name is required")
       
       return errors
   
   # Validate configuration
   errors = validate_config(config)
   if errors:
       print("Configuration errors:")
       for error in errors:
           print(f"  - {error}")
   ```

2. **Set Default Values**
   ```python
   def set_default_values(config):
       """Set default values for missing configuration options"""
       # Ensure backtesting section exists
       if "backtesting" not in config:
           config["backtesting"] = {}
       
       # Set default backtesting parameters
       backtesting = config["backtesting"]
       backtesting.setdefault("initial_capital", 10000)
       backtesting.setdefault("commission", 0.001)
       backtesting.setdefault("slippage", 0.0005)
       
       # Ensure strategy section exists
       if "strategy" not in config:
           config["strategy"] = {}
       
       # Set default strategy parameters
       strategy = config["strategy"]
       strategy.setdefault("name", "scalping")
       
       return config
   
   # Set default values
   config = set_default_values(config)
   ```

## Common Error Messages

### "No module named 'platon_light'"

**Cause:** Python cannot find the Platon Light package.

**Solutions:**
1. **Install the Package**
   ```bash
   # Install package in development mode
   pip install -e .
   ```

2. **Add to Python Path**
   ```python
   import sys
   from pathlib import Path
   
   # Add project root to path
   project_root = Path(__file__).parent.parent
   sys.path.append(str(project_root))
   ```

### "AttributeError: 'NoneType' object has no attribute 'X'"

**Cause:** Trying to access an attribute of a None object, often due to a function returning None.

**Solutions:**
1. **Check Return Values**
   ```python
   # Check if function returns None
   result = some_function()
   if result is None:
       print("Function returned None")
       # Handle the None case
   else:
       # Proceed normally
       result.some_attribute
   ```

2. **Add Null Checks**
   ```python
   # Add null checks before accessing attributes
   if hasattr(obj, 'attribute') and obj.attribute is not None:
       # Use obj.attribute
   ```

### "KeyError: 'X'"

**Cause:** Trying to access a non-existent key in a dictionary.

**Solutions:**
1. **Use get() Method**
   ```python
   # Use get() with default value
   value = config.get("section", {}).get("key", default_value)
   ```

2. **Check Key Existence**
   ```python
   # Check if key exists before accessing
   if "section" in config and "key" in config["section"]:
       value = config["section"]["key"]
   else:
       value = default_value
   ```

## Debugging Tips

### Enable Debug Logging

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get logger
logger = logging.getLogger("platon_light.backtesting")
```

### Use Breakpoints

```python
# Add breakpoints in your code
import pdb

def problematic_function():
    # Some code
    pdb.set_trace()  # Debugger will stop here
    # More code
```

### Print Intermediate Values

```python
# Add print statements for debugging
def calculate_signal(data):
    rsi = calculate_rsi(data)
    print(f"RSI: {rsi}")
    
    if rsi < 30:
        signal = 1
    elif rsi > 70:
        signal = -1
    else:
        signal = 0
    
    print(f"Signal: {signal}")
    return signal
```

### Check Data Types

```python
# Check data types
def debug_data_types(obj):
    if isinstance(obj, dict):
        print("Dictionary:")
        for key, value in obj.items():
            print(f"  {key}: {type(value)}")
    elif isinstance(obj, list):
        print("List:")
        if obj:
            print(f"  First item type: {type(obj[0])}")
        print(f"  Length: {len(obj)}")
    else:
        print(f"Type: {type(obj)}")
        print(f"Value: {obj}")
```

### Isolate Components

When debugging complex issues, isolate components to identify the source of the problem:

```python
# Test data loading separately
data = data_loader.load_data(symbol, timeframe, start_date, end_date)
print(f"Data loaded: {len(data)} rows")

# Test signal generation separately
signals = backtest_engine.strategy.generate_signals(data)
print(f"Signals generated: {(signals['signal'] != 0).sum()} signals")

# Test trade simulation separately
trades = backtest_engine._simulate_trading(signals, symbol, timeframe)
print(f"Trades simulated: {len(trades['trades'])} trades")
```

## Getting Help

If you're still experiencing issues after trying the solutions in this guide, you can:

1. **Check the Documentation**
   - Read the API reference
   - Review the backtesting guide

2. **Search for Similar Issues**
   - Check if others have encountered the same problem
   - Look for solutions in online forums or communities

3. **Create a Minimal Reproducible Example**
   - Create a simple script that reproduces the issue
   - Share the script and error message when asking for help

4. **Contact Support**
   - Provide detailed information about the issue
   - Include error messages, configuration, and steps to reproduce
