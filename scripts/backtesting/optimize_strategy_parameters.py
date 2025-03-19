#!/usr/bin/env python
"""
Strategy parameter optimization script for Platon Light.

This script performs a grid search to find the optimal parameters
for the Moving Average Crossover strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import logging
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from platon_light.backtesting.strategies.moving_average_crossover import MovingAverageCrossover


def load_sample_data(file_path=None):
    """
    Load sample data for backtesting.
    
    Args:
        file_path: Path to the CSV file with historical data
        
    Returns:
        DataFrame with OHLCV data
    """
    if file_path is None:
        # Use a default sample data file
        file_path = Path(__file__).parent / 'data' / 'sample' / 'BTCUSDT_1h.csv'
    
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Load data from CSV
        data = pd.read_csv(file_path)
        
        # Ensure we have the required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        # Convert timestamp to datetime
        if 'timestamp' in data.columns:
            if data['timestamp'].dtype == np.int64 or data['timestamp'].dtype == np.float64:
                # Convert milliseconds to seconds if needed
                if data['timestamp'].iloc[0] > 1e10:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                else:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            else:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Set timestamp as index
        data.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(data)} data points")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def run_backtest_with_params(data, params):
    """
    Run a backtest with specific parameters.
    
    Args:
        data: DataFrame with OHLCV data
        params: Dictionary with strategy parameters
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Create strategy instance
        strategy = MovingAverageCrossover(**params)
        
        # Run strategy
        result = strategy.run(data)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(result)
        
        # Add parameters to metrics
        metrics['params'] = params
        
        return metrics
    except Exception as e:
        logger.error(f"Error running backtest with params {params}: {e}")
        return {'error': str(e), 'params': params}


def calculate_performance_metrics(data):
    """
    Calculate performance metrics for the backtest.
    
    Args:
        data: DataFrame with backtest results
        
    Returns:
        Dictionary with performance metrics
    """
    # Extract signals
    signals = data[data['signal'] != 0].copy()
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    # Basic signal metrics
    metrics = {
        'total_signals': len(signals),
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
    }
    
    # Calculate returns (simple simulation)
    # Assuming we buy when signal=1 and sell when signal=-1
    
    # Initialize position and portfolio value columns
    data['position'] = 0
    data['portfolio_value'] = 100  # Start with $100
    
    # Simulate trading
    position = 0
    entry_price = 0
    
    for i in range(1, len(data)):
        # Update position based on signals
        if data['signal'].iloc[i] == 1 and position == 0:
            # Buy
            position = 1
            entry_price = data['close'].iloc[i]
        elif data['signal'].iloc[i] == -1 and position == 1:
            # Sell
            position = 0
        
        # Record position
        data.loc[data.index[i], 'position'] = position
        
        # Calculate portfolio value
        if position == 0:
            # Not in market, value stays the same
            data.loc[data.index[i], 'portfolio_value'] = data['portfolio_value'].iloc[i-1]
        else:
            # In market, value changes with price
            pct_change = data['close'].iloc[i] / data['close'].iloc[i-1] - 1
            data.loc[data.index[i], 'portfolio_value'] = data['portfolio_value'].iloc[i-1] * (1 + pct_change)
    
    # Calculate returns
    data['returns'] = data['portfolio_value'].pct_change()
    
    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
    
    # Calculate strategy performance metrics
    if len(data) > 1:
        # Total return
        total_return = data['portfolio_value'].iloc[-1] / data['portfolio_value'].iloc[0] - 1
        metrics['total_return'] = total_return
        
        # Annualized return (assuming 365 days per year)
        days = (data.index[-1] - data.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
            metrics['annualized_return'] = annualized_return
        
        # Maximum drawdown
        cumulative_max = data['portfolio_value'].cummax()
        drawdown = (data['portfolio_value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = max_drawdown
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if data['returns'].std() > 0:
            sharpe_ratio = data['returns'].mean() / data['returns'].std() * np.sqrt(365)
            metrics['sharpe_ratio'] = sharpe_ratio
        
        # Win rate (percentage of profitable trades)
        if 'position' in data.columns:
            # Find position changes
            position_changes = data['position'].diff()
            
            # Entry points (position changes from 0 to 1)
            entries = data[position_changes == 1].index
            
            # Exit points (position changes from 1 to 0)
            exits = data[position_changes == -1].index
            
            # Calculate trade returns
            trade_returns = []
            
            for i in range(min(len(entries), len(exits))):
                if entries[i] < exits[i]:
                    entry_price = data.loc[entries[i], 'close']
                    exit_price = data.loc[exits[i], 'close']
                    trade_return = exit_price / entry_price - 1
                    trade_returns.append(trade_return)
            
            if trade_returns:
                metrics['win_rate'] = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
                metrics['avg_win'] = sum(r for r in trade_returns if r > 0) / sum(1 for r in trade_returns if r > 0) if sum(1 for r in trade_returns if r > 0) > 0 else 0
                metrics['avg_loss'] = sum(r for r in trade_returns if r <= 0) / sum(1 for r in trade_returns if r <= 0) if sum(1 for r in trade_returns if r <= 0) > 0 else 0
                metrics['profit_factor'] = abs(sum(r for r in trade_returns if r > 0) / sum(r for r in trade_returns if r <= 0)) if sum(r for r in trade_returns if r <= 0) < 0 else float('inf')
    
    return metrics


def grid_search(data, param_grid, n_jobs=-1):
    """
    Perform grid search to find optimal parameters.
    
    Args:
        data: DataFrame with OHLCV data
        param_grid: Dictionary with parameter values to search
        n_jobs: Number of parallel jobs (-1 for all available cores)
        
    Returns:
        List of dictionaries with performance metrics for each parameter combination
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Running grid search with {len(param_combinations)} parameter combinations")
    
    # Set number of workers
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Run backtests in parallel
    results = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit tasks
        futures = []
        for params_tuple in param_combinations:
            params = dict(zip(param_names, params_tuple))
            future = executor.submit(run_backtest_with_params, data, params)
            futures.append(future)
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(param_combinations):
                logger.info(f"Completed {i + 1}/{len(param_combinations)} parameter combinations")
    
    # Sort results by performance metric (e.g., Sharpe ratio)
    results.sort(key=lambda x: x.get('sharpe_ratio', float('-inf')), reverse=True)
    
    return results


def visualize_optimization_results(results, output_dir=None):
    """
    Visualize optimization results.
    
    Args:
        results: List of dictionaries with performance metrics
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract parameters and metrics
    param_names = list(results[0]['params'].keys())
    metric_names = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    # Create dataframe with results
    results_df = pd.DataFrame([
        {**r['params'], **{k: r.get(k, None) for k in metric_names}}
        for r in results if 'error' not in r
    ])
    
    # 1. Parameter distribution of top performers
    plt.figure(figsize=(12, 8))
    
    # Select top performers (top 10%)
    top_n = max(1, int(len(results_df) * 0.1))
    top_performers = results_df.nlargest(top_n, 'sharpe_ratio')
    
    # Plot parameter distributions for top performers
    for i, param in enumerate(param_names):
        if isinstance(results_df[param].iloc[0], (int, float)):
            plt.subplot(len(param_names), 1, i + 1)
            plt.hist(top_performers[param], bins=10, alpha=0.7)
            plt.title(f'Distribution of {param} in Top Performers')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"param_distribution_{timestamp}.png")
    plt.close()
    
    # 2. Parameter importance (correlation with performance)
    plt.figure(figsize=(12, 8))
    
    # Calculate correlation between parameters and metrics
    corr_data = {}
    for param in param_names:
        if isinstance(results_df[param].iloc[0], (int, float)):
            for metric in metric_names:
                if metric in results_df.columns:
                    corr = results_df[param].corr(results_df[metric])
                    corr_data[(param, metric)] = corr
    
    # Convert to dataframe for plotting
    corr_df = pd.DataFrame([
        {'Parameter': param, 'Metric': metric, 'Correlation': corr}
        for (param, metric), corr in corr_data.items()
    ])
    
    # Plot heatmap
    if not corr_df.empty:
        pivot_df = corr_df.pivot(index='Parameter', columns='Metric', values='Correlation')
        plt.imshow(pivot_df.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        plt.title('Parameter Importance (Correlation with Performance)')
        
        # Add correlation values
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                plt.text(j, i, f"{pivot_df.iloc[i, j]:.2f}", 
                        ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"param_importance_{timestamp}.png")
        plt.close()
    
    # 3. Performance distribution
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metric_names):
        if metric in results_df.columns:
            plt.subplot(len(metric_names), 1, i + 1)
            plt.hist(results_df[metric], bins=20, alpha=0.7)
            plt.title(f'Distribution of {metric}')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"performance_distribution_{timestamp}.png")
    plt.close()
    
    # 4. Scatter plots for key parameters vs performance
    plt.figure(figsize=(15, 10))
    
    # Select top 2 parameters by correlation with Sharpe ratio
    if 'sharpe_ratio' in results_df.columns:
        param_corrs = []
        for param in param_names:
            if isinstance(results_df[param].iloc[0], (int, float)):
                corr = abs(results_df[param].corr(results_df['sharpe_ratio']))
                param_corrs.append((param, corr))
        
        param_corrs.sort(key=lambda x: x[1], reverse=True)
        top_params = [p for p, _ in param_corrs[:min(2, len(param_corrs))]]
        
        if len(top_params) >= 2:
            plt.scatter(results_df[top_params[0]], results_df[top_params[1]], 
                        c=results_df['sharpe_ratio'], cmap='viridis', alpha=0.7)
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel(top_params[0])
            plt.ylabel(top_params[1])
            plt.title(f'Parameter Space: {top_params[0]} vs {top_params[1]}')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"param_space_{timestamp}.png")
            plt.close()
    
    logger.info(f"Optimization visualizations saved to {output_dir}")


def save_optimization_results(results, output_dir=None):
    """
    Save optimization results to CSV and JSON files.
    
    Args:
        results: List of dictionaries with performance metrics
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {**r['params'], **{k: v for k, v in r.items() if k != 'params'}}
        for r in results if 'error' not in r
    ])
    
    # Save to CSV
    csv_file = output_dir / f"optimization_results_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    
    # Save top 10 results to JSON
    top_results = results[:10]
    json_file = output_dir / f"top_parameters_{timestamp}.json"
    
    with open(json_file, 'w') as f:
        json.dump(top_results, f, indent=4, default=str)
    
    logger.info(f"Optimization results saved to {csv_file}")
    logger.info(f"Top parameters saved to {json_file}")


def main():
    """Main function to optimize strategy parameters."""
    logger.info("Starting strategy parameter optimization")
    
    # Load sample data
    data_dir = Path(__file__).parent / 'data' / 'sample'
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if sample data exists
    sample_file = data_dir / 'BTCUSDT_1h.csv'
    if not sample_file.exists():
        logger.warning(f"Sample data file not found at {sample_file}")
        logger.info("Please provide a CSV file with OHLCV data")
        return
    
    # Load data
    data = load_sample_data(sample_file)
    if data is None:
        logger.error("Failed to load data")
        return
    
    # Define parameter grid
    param_grid = {
        'fast_ma_type': ['EMA', 'SMA'],
        'slow_ma_type': ['EMA', 'SMA'],
        'fast_period': [10, 20],
        'slow_period': [40, 50],
        'rsi_period': [14],
        'rsi_overbought': [70],
        'rsi_oversold': [30],
        'use_filters': [True, False]
    }
    
    # Run grid search
    results = grid_search(data, param_grid)
    
    # Visualize results
    visualize_optimization_results(results)
    
    # Save results
    save_optimization_results(results)
    
    # Print best parameters
    if results:
        best_params = results[0]['params']
        best_metrics = {k: v for k, v in results[0].items() if k != 'params'}
        
        logger.info("Best parameters found:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        logger.info("Performance with best parameters:")
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    
    logger.info("Strategy parameter optimization completed")


if __name__ == "__main__":
    main()
