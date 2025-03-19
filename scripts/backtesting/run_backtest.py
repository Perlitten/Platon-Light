#!/usr/bin/env python
"""
Backtest runner script for Platon Light.

This script runs a backtest using the Moving Average Crossover strategy
and displays the results.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from platon_light.backtesting.strategies.moving_average_crossover import MovingAverageCrossover
from platon_light.backtesting.data_loader import DataLoader


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


def run_backtest(data, strategy_params=None):
    """
    Run a backtest using the Moving Average Crossover strategy.
    
    Args:
        data: DataFrame with OHLCV data
        strategy_params: Dictionary with strategy parameters
        
    Returns:
        DataFrame with backtest results
    """
    if strategy_params is None:
        strategy_params = {
            'fast_ma_type': 'EMA',
            'slow_ma_type': 'SMA',
            'fast_period': 20,
            'slow_period': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'use_filters': True
        }
    
    logger.info(f"Running backtest with strategy parameters: {strategy_params}")
    
    # Create strategy instance
    strategy = MovingAverageCrossover(**strategy_params)
    
    # Run strategy
    result = strategy.run(data)
    
    logger.info(f"Backtest completed with {len(result[result['signal'] != 0])} signals")
    
    return result


def analyze_results(result):
    """
    Analyze backtest results.
    
    Args:
        result: DataFrame with backtest results
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate basic metrics
    signals = result[result['signal'] != 0]
    buy_signals = result[result['signal'] == 1]
    sell_signals = result[result['signal'] == -1]
    
    metrics = {
        'total_signals': len(signals),
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'signal_ratio': len(buy_signals) / len(sell_signals) if len(sell_signals) > 0 else float('inf')
    }
    
    logger.info(f"Analysis results: {metrics}")
    
    return metrics


def visualize_results(result, title="Moving Average Crossover Backtest"):
    """
    Visualize backtest results.
    
    Args:
        result: DataFrame with backtest results
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(result.index, result['close'], label='Close Price', alpha=0.5)
    
    # Plot moving averages
    plt.plot(result.index, result['fast_ma'], label='Fast MA', alpha=0.8)
    plt.plot(result.index, result['slow_ma'], label='Slow MA', alpha=0.8)
    
    # Plot buy and sell signals
    buy_signals = result[result['signal'] == 1]
    sell_signals = result[result['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], 
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['close'], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f"{title} - Price and Signals")
    plt.legend()
    plt.grid(True)
    
    # Plot RSI if available
    if 'RSI_14' in result.columns:
        plt.subplot(2, 1, 2)
        plt.plot(result.index, result['RSI_14'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        plt.title("RSI Indicator")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backtest_result_{timestamp}.png"
    
    plt.savefig(output_file)
    logger.info(f"Visualization saved to {output_file}")
    
    # Don't show the plot interactively
    plt.close()


def main():
    """Main function to run the backtest."""
    logger.info("Starting backtest")
    
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
    
    # Run backtest
    result = run_backtest(data)
    
    # Analyze results
    metrics = analyze_results(result)
    
    # Visualize results
    visualize_results(result)
    
    # Save results to CSV for further analysis
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backtest_data_{timestamp}.csv"
    
    result.to_csv(output_file)
    logger.info(f"Backtest data saved to {output_file}")
    
    logger.info("Backtest completed successfully")


if __name__ == "__main__":
    main()
