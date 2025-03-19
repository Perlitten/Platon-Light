# Backtesting Monitoring and Logging Guide

This guide explains how to implement effective monitoring and logging for your backtesting processes in Platon Light.

## Table of Contents

1. [Introduction](#introduction)
2. [Logging Configuration](#logging-configuration)
3. [Monitoring Backtest Progress](#monitoring-backtest-progress)
4. [Logging Trade Information](#logging-trade-information)
5. [Performance Metrics Logging](#performance-metrics-logging)
6. [Error Handling and Debugging](#error-handling-and-debugging)
7. [Log Analysis](#log-analysis)
8. [Best Practices](#best-practices)

## Introduction

Proper monitoring and logging are essential for backtesting as they help you:

- Track the progress of long-running backtests
- Debug issues in your strategy implementation
- Analyze trade decisions and performance
- Create audit trails for strategy development
- Compare results across multiple backtest runs

## Logging Configuration

Platon Light uses Python's built-in logging module. Here's how to configure it:

```python
import logging
import os
from datetime import datetime

def setup_logging(config):
    """
    Set up logging for backtesting.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    logs_dir = config.get('logging', {}).get('logs_dir', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    strategy_name = config.get('strategy', {}).get('name', 'unknown')
    log_filename = f"{logs_dir}/{strategy_name}_{timestamp}.log"
    
    # Configure logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Set up logger
    logger = logging.getLogger('platon_light.backtesting')
    logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for strategy: {strategy_name}")
    logger.info(f"Log file: {log_filename}")
    
    return logger
```

Add this to your configuration file:

```yaml
logging:
  logs_dir: logs
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  console_output: true
  file_output: true
  trade_details: true
  performance_metrics: true
```

## Monitoring Backtest Progress

For long-running backtests, implement progress monitoring:

```python
class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(config)
        # Other initialization code...
    
    def run(self, symbol, timeframe, start_date, end_date):
        self.logger.info(f"Starting backtest for {symbol} {timeframe} from {start_date} to {end_date}")
        
        # Load data
        self.logger.info("Loading market data...")
        data = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
        self.logger.info(f"Loaded {len(data)} data points")
        
        # Initialize variables for tracking progress
        total_bars = len(data)
        progress_interval = max(1, total_bars // 20)  # Report progress ~20 times
        start_time = datetime.now()
        
        # Run backtest
        self.logger.info("Running backtest simulation...")
        
        for i, (idx, bar) in enumerate(data.iterrows()):
            # Process bar...
            
            # Log progress
            if i % progress_interval == 0 or i == total_bars - 1:
                progress_pct = (i + 1) / total_bars * 100
                elapsed = datetime.now() - start_time
                bars_per_second = (i + 1) / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                
                estimated_remaining = None
                if bars_per_second > 0:
                    remaining_bars = total_bars - (i + 1)
                    estimated_remaining = remaining_bars / bars_per_second
                    estimated_remaining = str(timedelta(seconds=int(estimated_remaining)))
                
                self.logger.info(f"Progress: {progress_pct:.1f}% ({i+1}/{total_bars}) - "
                                f"Speed: {bars_per_second:.1f} bars/sec - "
                                f"Est. remaining: {estimated_remaining}")
        
        elapsed_time = datetime.now() - start_time
        self.logger.info(f"Backtest completed in {elapsed_time}")
        
        # Return results...
```

## Logging Trade Information

Log detailed information about each trade:

```python
def execute_trade(self, bar, signal):
    """Execute a trade based on the signal"""
    # Trade execution logic...
    
    if trade_executed:
        self.logger.info(
            f"TRADE: {trade_type} {symbol} at {price:.2f} - "
            f"Size: {size:.4f} - Position: {position_size:.4f} - "
            f"Equity: {equity:.2f}"
        )
        
        # Log detailed trade information at DEBUG level
        self.logger.debug(
            f"TRADE DETAILS: {trade_type} {symbol} - "
            f"Price: {price:.2f} - Size: {size:.4f} - "
            f"Commission: {commission:.2f} - Slippage: {slippage:.2f} - "
            f"Cash: {cash:.2f} - Position Value: {position_value:.2f} - "
            f"Equity: {equity:.2f}"
        )

def close_trade(self, bar, reason):
    """Close an existing trade"""
    # Trade closing logic...
    
    if trade_closed:
        self.logger.info(
            f"CLOSE: {symbol} at {price:.2f} - "
            f"P/L: {profit_loss:.2f} ({profit_loss_pct:.2f}%) - "
            f"Equity: {equity:.2f} - Reason: {reason}"
        )
        
        # Log trade summary
        self.logger.info(
            f"TRADE SUMMARY: {symbol} - "
            f"Entry: {entry_price:.2f} on {entry_time} - "
            f"Exit: {exit_price:.2f} on {exit_time} - "
            f"P/L: {profit_loss:.2f} ({profit_loss_pct:.2f}%) - "
            f"Duration: {duration}"
        )
```

## Performance Metrics Logging

Log performance metrics during and after the backtest:

```python
class PerformanceAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('platon_light.backtesting')
    
    def analyze(self, results):
        """Analyze backtest results and log performance metrics"""
        self.logger.info("Analyzing backtest results...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        # Log overall performance
        self.logger.info("===== PERFORMANCE SUMMARY =====")
        self.logger.info(f"Total Return: {metrics['return_percent']:.2f}%")
        self.logger.info(f"Annualized Return: {metrics['annualized_return']:.2f}%")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
        
        # Log trade statistics
        trade_stats = self.calculate_trade_statistics(results)
        self.logger.info("===== TRADE STATISTICS =====")
        self.logger.info(f"Total Trades: {trade_stats['total_trades']}")
        self.logger.info(f"Win Rate: {trade_stats['win_rate']:.2f}%")
        self.logger.info(f"Profit Factor: {trade_stats['profit_factor']:.2f}")
        self.logger.info(f"Average Profit per Trade: {trade_stats['avg_profit_per_trade']:.2f}")
        self.logger.info(f"Average Profit per Winning Trade: {trade_stats['avg_profit_per_winning_trade']:.2f}")
        self.logger.info(f"Average Loss per Losing Trade: {trade_stats['avg_loss_per_losing_trade']:.2f}")
        
        # Log monthly returns if available
        if 'monthly_returns' in metrics:
            self.logger.info("===== MONTHLY RETURNS =====")
            for month, return_pct in metrics['monthly_returns'].items():
                self.logger.info(f"{month}: {return_pct:.2f}%")
        
        return metrics
```

## Error Handling and Debugging

Implement robust error handling and debugging:

```python
def generate_signals(self, data):
    """Generate trading signals based on strategy logic"""
    try:
        # Strategy-specific calculations
        self.logger.debug("Calculating strategy indicators...")
        
        # Generate signals
        self.logger.debug("Generating trading signals...")
        
        return data
    except Exception as e:
        self.logger.error(f"Error generating signals: {str(e)}")
        self.logger.exception("Exception details:")
        
        # Return data without signals to avoid crashing the backtest
        data['signal'] = 0
        return data
```

## Log Analysis

Create utilities to analyze log files:

```python
def analyze_log_file(log_file_path):
    """
    Analyze a backtest log file to extract key information.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary containing extracted information
    """
    import re
    
    # Initialize results dictionary
    results = {
        'trades': [],
        'performance': {},
        'errors': [],
        'warnings': []
    }
    
    # Regular expressions for parsing
    trade_pattern = re.compile(r'TRADE: (BUY|SELL) (\w+) at (\d+\.\d+)')
    close_pattern = re.compile(r'CLOSE: (\w+) at (\d+\.\d+) - P/L: ([-\d\.]+) \(([-\d\.]+)%\)')
    performance_pattern = re.compile(r'Total Return: ([-\d\.]+)%')
    error_pattern = re.compile(r'ERROR - (.+)')
    warning_pattern = re.compile(r'WARNING - (.+)')
    
    # Parse log file
    with open(log_file_path, 'r') as f:
        for line in f:
            # Extract trades
            trade_match = trade_pattern.search(line)
            if trade_match:
                trade_type, symbol, price = trade_match.groups()
                results['trades'].append({
                    'type': trade_type,
                    'symbol': symbol,
                    'price': float(price)
                })
            
            # Extract trade closures
            close_match = close_pattern.search(line)
            if close_match:
                symbol, price, profit_loss, profit_loss_pct = close_match.groups()
                results['trades'].append({
                    'type': 'CLOSE',
                    'symbol': symbol,
                    'price': float(price),
                    'profit_loss': float(profit_loss),
                    'profit_loss_pct': float(profit_loss_pct)
                })
            
            # Extract performance metrics
            performance_match = performance_pattern.search(line)
            if performance_match:
                results['performance']['total_return'] = float(performance_match.group(1))
            
            # Extract errors
            error_match = error_pattern.search(line)
            if error_match:
                results['errors'].append(error_match.group(1))
            
            # Extract warnings
            warning_match = warning_pattern.search(line)
            if warning_match:
                results['warnings'].append(warning_match.group(1))
    
    return results
```

## Best Practices

1. **Use Appropriate Log Levels**:
   - DEBUG: Detailed information for debugging
   - INFO: General information about backtest progress
   - WARNING: Potential issues that don't stop the backtest
   - ERROR: Serious problems that may affect results
   - CRITICAL: Fatal errors that stop the backtest

2. **Log Key Decision Points**:
   - Signal generation logic
   - Trade entry and exit decisions
   - Position sizing calculations
   - Risk management actions

3. **Include Context in Log Messages**:
   - Timestamp and bar information
   - Current market conditions
   - Strategy state and parameters
   - Account state (equity, positions)

4. **Implement Rotating Logs**:
   For long-running or repeated backtests:

   ```python
   from logging.handlers import RotatingFileHandler
   
   # Use rotating file handler
   file_handler = RotatingFileHandler(
       log_filename,
       maxBytes=10*1024*1024,  # 10 MB
       backupCount=5
   )
   ```

5. **Create Structured Logs**:
   For easier parsing and analysis:

   ```python
   import json
   
   def log_trade(self, trade_data):
       """Log trade information in structured format"""
       self.logger.info(f"TRADE: {json.dumps(trade_data)}")
   ```

6. **Log Configuration and Environment**:
   At the start of each backtest:

   ```python
   def log_backtest_config(self):
       """Log backtest configuration"""
       self.logger.info("===== BACKTEST CONFIGURATION =====")
       self.logger.info(f"Strategy: {self.config['strategy']['name']}")
       self.logger.info(f"Symbol: {self.symbol}")
       self.logger.info(f"Timeframe: {self.timeframe}")
       self.logger.info(f"Period: {self.start_date} to {self.end_date}")
       self.logger.info(f"Initial Capital: {self.config['backtesting']['initial_capital']}")
       self.logger.info(f"Commission: {self.config['backtesting']['commission']}")
       self.logger.info(f"Slippage: {self.config['backtesting']['slippage']}")
       
       # Log strategy-specific parameters
       self.logger.info("Strategy Parameters:")
       for key, value in self.config['strategy'].items():
           if key != 'name':
               self.logger.info(f"  {key}: {value}")
   ```

7. **Monitor Resource Usage**:
   For large backtests:

   ```python
   import psutil
   import os
   
   def log_resource_usage(self):
       """Log current resource usage"""
       process = psutil.Process(os.getpid())
       memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
       cpu_percent = process.cpu_percent()
       
       self.logger.info(f"Resource usage - Memory: {memory_usage:.2f} MB, CPU: {cpu_percent:.1f}%")
   ```

8. **Create Summary Reports**:
   Generate summary reports at the end of backtests:

   ```python
   def generate_summary_report(self, results, output_file=None):
       """Generate a summary report of backtest results"""
       # Create report content
       report = []
       report.append("===== BACKTEST SUMMARY REPORT =====")
       report.append(f"Strategy: {self.config['strategy']['name']}")
       report.append(f"Symbol: {self.symbol}")
       report.append(f"Timeframe: {self.timeframe}")
       report.append(f"Period: {self.start_date} to {self.end_date}")
       report.append("")
       
       report.append("Performance Metrics:")
       for key, value in results['metrics'].items():
           if isinstance(value, float):
               report.append(f"  {key}: {value:.2f}")
           else:
               report.append(f"  {key}: {value}")
       
       report.append("")
       report.append(f"Total Trades: {len(results['trades'])}")
       
       # Write to file if specified
       if output_file:
           with open(output_file, 'w') as f:
               f.write('\n'.join(report))
       
       # Log the report
       for line in report:
           self.logger.info(line)
       
       return '\n'.join(report)
   ```

By implementing these monitoring and logging practices, you'll have better visibility into your backtesting process, making it easier to debug issues, analyze results, and improve your trading strategies.
