# Platon Light Backtesting API Reference

This document provides a comprehensive reference for the Platon Light backtesting module API. It includes detailed information about each class, method, and function.

## Table of Contents

1. [DataLoader](#dataloader)
2. [BacktestEngine](#backtestengine)
3. [PerformanceAnalyzer](#performanceanalyzer)
4. [BacktestVisualizer](#backtestvisualizer)
5. [StrategyOptimizer](#strategyoptimizer)
6. [CLI Interface](#cli-interface)

---

## DataLoader

The `DataLoader` class is responsible for loading and preparing historical market data for backtesting.

### Class: `DataLoader`

```python
from platon_light.backtesting.data_loader import DataLoader
```

#### Constructor

```python
DataLoader(config: Dict)
```

- **Parameters:**
  - `config`: Configuration dictionary containing data loading settings

#### Methods

##### `load_data`

```python
load_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime, use_cache: bool = True) -> pd.DataFrame
```

- **Description:** Loads historical market data for the specified symbol, timeframe, and date range
- **Parameters:**
  - `symbol`: Trading pair symbol (e.g., 'BTCUSDT')
  - `timeframe`: Timeframe (e.g., '1m', '5m', '1h')
  - `start_date`: Start date for data loading
  - `end_date`: End date for data loading
  - `use_cache`: Whether to use cached data (default: True)
- **Returns:** DataFrame with OHLCV data

##### `prepare_data`

```python
prepare_data(data: pd.DataFrame) -> pd.DataFrame
```

- **Description:** Prepares data for backtesting by calculating indicators and other features
- **Parameters:**
  - `data`: DataFrame with OHLCV data
- **Returns:** DataFrame with OHLCV data and calculated indicators

##### `download_data`

```python
download_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame
```

- **Description:** Downloads historical market data from the exchange API
- **Parameters:**
  - `symbol`: Trading pair symbol (e.g., 'BTCUSDT')
  - `timeframe`: Timeframe (e.g., '1m', '5m', '1h')
  - `start_date`: Start date for data downloading
  - `end_date`: End date for data downloading
- **Returns:** DataFrame with downloaded OHLCV data

##### `resample_data`

```python
resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame
```

- **Description:** Resamples data to a different timeframe
- **Parameters:**
  - `data`: DataFrame with OHLCV data
  - `timeframe`: Target timeframe (e.g., '5m', '1h')
- **Returns:** Resampled DataFrame

##### `combine_timeframes`

```python
combine_timeframes(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame
```

- **Description:** Combines data from multiple timeframes into a single DataFrame
- **Parameters:**
  - `data_dict`: Dictionary mapping timeframes to DataFrames
- **Returns:** Combined DataFrame

##### `save_data`

```python
save_data(data: pd.DataFrame, symbol: str, timeframe: str, format: str = 'csv') -> str
```

- **Description:** Saves data to a file
- **Parameters:**
  - `data`: DataFrame with OHLCV data
  - `symbol`: Trading pair symbol
  - `timeframe`: Timeframe
  - `format`: File format ('csv' or 'parquet')
- **Returns:** Path to saved file

---

## BacktestEngine

The `BacktestEngine` class is responsible for simulating trading strategies on historical data.

### Class: `BacktestEngine`

```python
from platon_light.backtesting.backtest_engine import BacktestEngine
```

#### Constructor

```python
BacktestEngine(config: Dict)
```

- **Parameters:**
  - `config`: Configuration dictionary containing backtesting settings

#### Methods

##### `run`

```python
run(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Dict
```

- **Description:** Runs a backtest for the specified symbol, timeframe, and date range
- **Parameters:**
  - `symbol`: Trading pair symbol (e.g., 'BTCUSDT')
  - `timeframe`: Timeframe (e.g., '1m', '5m', '1h')
  - `start_date`: Start date for backtesting
  - `end_date`: End date for backtesting
- **Returns:** Dictionary with backtest results

##### `run_multi_timeframe`

```python
run_multi_timeframe(symbol: str, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> Dict
```

- **Description:** Runs a backtest using multiple timeframes
- **Parameters:**
  - `symbol`: Trading pair symbol
  - `data`: Dictionary mapping timeframes to DataFrames
  - `start_date`: Start date for backtesting
  - `end_date`: End date for backtesting
- **Returns:** Dictionary with backtest results

##### `generate_report`

```python
generate_report(results: Dict, output_dir: Optional[str] = None) -> str
```

- **Description:** Generates a backtest report
- **Parameters:**
  - `results`: Backtest results dictionary
  - `output_dir`: Output directory for the report
- **Returns:** Path to the generated report

##### `get_strategy_info`

```python
get_strategy_info() -> Dict
```

- **Description:** Returns information about the current strategy
- **Returns:** Dictionary with strategy information

---

## PerformanceAnalyzer

The `PerformanceAnalyzer` class is responsible for analyzing backtest results and calculating performance metrics.

### Class: `PerformanceAnalyzer`

```python
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
```

#### Constructor

```python
PerformanceAnalyzer(config: Dict)
```

- **Parameters:**
  - `config`: Configuration dictionary containing analysis settings

#### Methods

##### `analyze`

```python
analyze(results: Dict, save_plots: bool = False) -> Dict
```

- **Description:** Analyzes backtest results and calculates performance metrics
- **Parameters:**
  - `results`: Backtest results dictionary
  - `save_plots`: Whether to save plots (default: False)
- **Returns:** Dictionary with analysis results

##### `get_trade_statistics`

```python
get_trade_statistics(results: Dict) -> Dict
```

- **Description:** Calculates trade statistics from backtest results
- **Parameters:**
  - `results`: Backtest results dictionary
- **Returns:** Dictionary with trade statistics

##### `calculate_metrics`

```python
calculate_metrics(equity_curve: List[Dict], trades: List[Dict]) -> Dict
```

- **Description:** Calculates performance metrics from equity curve and trades
- **Parameters:**
  - `equity_curve`: List of equity curve points
  - `trades`: List of trades
- **Returns:** Dictionary with performance metrics

##### `calculate_drawdowns`

```python
calculate_drawdowns(equity_curve: List[Dict]) -> Tuple[float, float, List[Dict]]
```

- **Description:** Calculates drawdowns from equity curve
- **Parameters:**
  - `equity_curve`: List of equity curve points
- **Returns:** Tuple with max drawdown, max drawdown percentage, and list of drawdowns

##### `compare_strategies`

```python
compare_strategies(results_list: List[Dict], strategy_names: List[str]) -> Dict
```

- **Description:** Compares multiple strategies
- **Parameters:**
  - `results_list`: List of backtest results dictionaries
  - `strategy_names`: List of strategy names
- **Returns:** Dictionary with comparison results

##### `generate_performance_summary`

```python
generate_performance_summary(results: Dict) -> str
```

- **Description:** Generates a performance summary
- **Parameters:**
  - `results`: Backtest results dictionary
- **Returns:** Performance summary string

---

## BacktestVisualizer

The `BacktestVisualizer` class is responsible for creating visualizations of backtest results.

### Class: `BacktestVisualizer`

```python
from platon_light.backtesting.visualization import BacktestVisualizer
```

#### Constructor

```python
BacktestVisualizer(config: Dict)
```

- **Parameters:**
  - `config`: Configuration dictionary containing visualization settings

#### Methods

##### `plot_equity_curve`

```python
plot_equity_curve(results: Dict, save: bool = True) -> plt.Figure
```

- **Description:** Plots the equity curve
- **Parameters:**
  - `results`: Backtest results dictionary
  - `save`: Whether to save the plot (default: True)
- **Returns:** Matplotlib figure

##### `plot_drawdown`

```python
plot_drawdown(results: Dict, save: bool = True) -> plt.Figure
```

- **Description:** Plots the drawdown chart
- **Parameters:**
  - `results`: Backtest results dictionary
  - `save`: Whether to save the plot (default: True)
- **Returns:** Matplotlib figure

##### `plot_trade_distribution`

```python
plot_trade_distribution(results: Dict, save: bool = True) -> plt.Figure
```

- **Description:** Plots the trade profit distribution
- **Parameters:**
  - `results`: Backtest results dictionary
  - `save`: Whether to save the plot (default: True)
- **Returns:** Matplotlib figure

##### `plot_monthly_returns`

```python
plot_monthly_returns(results: Dict, save: bool = True) -> plt.Figure
```

- **Description:** Plots the monthly returns heatmap
- **Parameters:**
  - `results`: Backtest results dictionary
  - `save`: Whether to save the plot (default: True)
- **Returns:** Matplotlib figure

##### `plot_performance_metrics`

```python
plot_performance_metrics(results: Dict, save: bool = True) -> plt.Figure
```

- **Description:** Plots the performance metrics
- **Parameters:**
  - `results`: Backtest results dictionary
  - `save`: Whether to save the plot (default: True)
- **Returns:** Matplotlib figure

##### `plot_trade_analysis`

```python
plot_trade_analysis(results: Dict, save: bool = True) -> plt.Figure
```

- **Description:** Plots trade analysis charts
- **Parameters:**
  - `results`: Backtest results dictionary
  - `save`: Whether to save the plot (default: True)
- **Returns:** Matplotlib figure

##### `plot_strategy_comparison`

```python
plot_strategy_comparison(results_list: List[Dict], strategy_names: List[str], save: bool = True) -> plt.Figure
```

- **Description:** Plots strategy comparison charts
- **Parameters:**
  - `results_list`: List of backtest results dictionaries
  - `strategy_names`: List of strategy names
  - `save`: Whether to save the plot (default: True)
- **Returns:** Matplotlib figure

##### `create_report`

```python
create_report(results: Dict, output_path: Optional[str] = None) -> str
```

- **Description:** Creates an HTML report with all visualizations
- **Parameters:**
  - `results`: Backtest results dictionary
  - `output_path`: Output path for the report
- **Returns:** Path to the generated report

---

## StrategyOptimizer

The `StrategyOptimizer` class is responsible for optimizing strategy parameters.

### Class: `StrategyOptimizer`

```python
from platon_light.backtesting.optimization import StrategyOptimizer
```

#### Constructor

```python
StrategyOptimizer(config: Dict)
```

- **Parameters:**
  - `config`: Configuration dictionary containing optimization settings

#### Methods

##### `grid_search`

```python
grid_search(param_grid: Dict, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Dict
```

- **Description:** Performs grid search optimization
- **Parameters:**
  - `param_grid`: Dictionary with parameter names and possible values
  - `symbol`: Trading pair symbol
  - `timeframe`: Timeframe
  - `start_date`: Start date
  - `end_date`: End date
- **Returns:** Dictionary with optimization results

##### `genetic_algorithm`

```python
genetic_algorithm(param_ranges: Dict, symbol: str, timeframe: str, start_date: datetime, end_date: datetime, population_size: int = 50, generations: int = 10, mutation_rate: float = 0.1, crossover_rate: float = 0.7) -> Dict
```

- **Description:** Performs genetic algorithm optimization
- **Parameters:**
  - `param_ranges`: Dictionary with parameter names and (min, max, step) tuples
  - `symbol`: Trading pair symbol
  - `timeframe`: Timeframe
  - `start_date`: Start date
  - `end_date`: End date
  - `population_size`: Size of the population (default: 50)
  - `generations`: Number of generations (default: 10)
  - `mutation_rate`: Mutation rate (default: 0.1)
  - `crossover_rate`: Crossover rate (default: 0.7)
- **Returns:** Dictionary with optimization results

##### `walk_forward_optimization`

```python
walk_forward_optimization(param_grid: Dict, symbol: str, timeframe: str, start_date: datetime, end_date: datetime, window_size: int = 30, step_size: int = 7) -> Dict
```

- **Description:** Performs walk-forward optimization
- **Parameters:**
  - `param_grid`: Dictionary with parameter names and possible values
  - `symbol`: Trading pair symbol
  - `timeframe`: Timeframe
  - `start_date`: Start date
  - `end_date`: End date
  - `window_size`: Size of the in-sample window in days (default: 30)
  - `step_size`: Size of the out-of-sample window in days (default: 7)
- **Returns:** Dictionary with optimization results

##### `optimize_multi_objective`

```python
optimize_multi_objective(param_grid: Dict, symbol: str, timeframe: str, start_date: datetime, end_date: datetime, objectives: List[str]) -> Dict
```

- **Description:** Performs multi-objective optimization
- **Parameters:**
  - `param_grid`: Dictionary with parameter names and possible values
  - `symbol`: Trading pair symbol
  - `timeframe`: Timeframe
  - `start_date`: Start date
  - `end_date`: End date
  - `objectives`: List of objective metrics
- **Returns:** Dictionary with optimization results

---

## CLI Interface

The command-line interface provides a convenient way to run backtests, analyze results, and optimize strategies.

### Module: `cli.py`

```python
from platon_light.backtesting.cli import main
```

#### Functions

##### `parse_args`

```python
parse_args() -> argparse.Namespace
```

- **Description:** Parses command-line arguments
- **Returns:** Namespace with parsed arguments

##### `setup_logging`

```python
setup_logging(log_level: str, output_dir: Optional[str] = None)
```

- **Description:** Sets up logging configuration
- **Parameters:**
  - `log_level`: Logging level
  - `output_dir`: Output directory for log file

##### `load_config`

```python
load_config(config_path: str) -> Dict
```

- **Description:** Loads configuration from file
- **Parameters:**
  - `config_path`: Path to configuration file
- **Returns:** Configuration dictionary

##### `update_config_from_args`

```python
update_config_from_args(config: Dict, args: argparse.Namespace) -> Dict
```

- **Description:** Updates configuration with command-line arguments
- **Parameters:**
  - `config`: Configuration dictionary
  - `args`: Command-line arguments
- **Returns:** Updated configuration dictionary

##### `run_backtest`

```python
run_backtest(config: Dict, args: argparse.Namespace) -> Dict
```

- **Description:** Runs a backtest with the given configuration
- **Parameters:**
  - `config`: Configuration dictionary
  - `args`: Command-line arguments
- **Returns:** Backtest results

##### `compare_strategies`

```python
compare_strategies(config: Dict, args: argparse.Namespace) -> Dict
```

- **Description:** Compares multiple strategies
- **Parameters:**
  - `config`: Configuration dictionary
  - `args`: Command-line arguments
- **Returns:** Comparison results

##### `print_summary`

```python
print_summary(results: Dict)
```

- **Description:** Prints backtest summary
- **Parameters:**
  - `results`: Backtest results dictionary

##### `print_comparison`

```python
print_comparison(comparison: Dict, strategy_names: List[str])
```

- **Description:** Prints strategy comparison
- **Parameters:**
  - `comparison`: Comparison results dictionary
  - `strategy_names`: List of strategy names

##### `main`

```python
main()
```

- **Description:** Main entry point for the CLI
- **Usage:**
  ```bash
  python -m platon_light.backtesting.cli --config backtest_config.yaml
  ```

---

## Example Usage

### Basic Backtesting

```python
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.utils.config_manager import ConfigManager
from datetime import datetime

# Load configuration
config_manager = ConfigManager("backtest_config.yaml")
config = config_manager.get_config()

# Initialize components
data_loader = DataLoader(config)
backtest_engine = BacktestEngine(config)
performance_analyzer = PerformanceAnalyzer(config)
visualizer = BacktestVisualizer(config)

# Define backtest parameters
symbol = "BTCUSDT"
timeframe = "1m"
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 31)

# Run backtest
results = backtest_engine.run(symbol, timeframe, start_date, end_date)

# Analyze results
analysis = performance_analyzer.analyze(results)

# Visualize results
visualizer.plot_equity_curve(results)
visualizer.plot_drawdown(results)
visualizer.create_report(results)
```

### Parameter Optimization

```python
from platon_light.backtesting.optimization import StrategyOptimizer
from datetime import datetime

# Initialize optimizer
optimizer = StrategyOptimizer(config)

# Define parameter grid
param_grid = {
    "rsi_period": [7, 14, 21],
    "rsi_oversold": [20, 25, 30],
    "rsi_overbought": [70, 75, 80]
}

# Define backtest parameters
symbol = "BTCUSDT"
timeframe = "1m"
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 31)

# Run grid search
results = optimizer.grid_search(param_grid, symbol, timeframe, start_date, end_date)

# Print best parameters
print(f"Best Parameters: {results['best_params']}")
print(f"Best Metrics: {results['best_metrics']}")
```

### Strategy Comparison

```python
# Define strategies to compare
strategies = ["scalping_rsi", "scalping_macd", "scalping_bollinger"]

# Run backtest for each strategy
results_list = []
for strategy in strategies:
    # Update config with strategy
    config["strategy"]["name"] = strategy
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config)
    
    # Run backtest
    results = backtest_engine.run(symbol, timeframe, start_date, end_date)
    results_list.append(results)

# Compare strategies
comparison = performance_analyzer.compare_strategies(results_list, strategies)

# Visualize comparison
visualizer.plot_strategy_comparison(results_list, strategies)
```
