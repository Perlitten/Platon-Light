#!/usr/bin/env python
"""
Backtest performance analysis script for Platon Light.

This script analyzes the results of a backtest and generates performance metrics
and visualizations to evaluate the strategy's effectiveness.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_backtest_results(file_path=None):
    """
    Load backtest results from a CSV file.
    
    Args:
        file_path: Path to the CSV file with backtest results
        
    Returns:
        DataFrame with backtest results
    """
    if file_path is None:
        # Find the most recent backtest results file
        results_dir = Path(__file__).parent / 'results'
        csv_files = list(results_dir.glob('backtest_data_*.csv'))
        
        if not csv_files:
            logger.error("No backtest results found")
            return None
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        file_path = csv_files[0]
    
    logger.info(f"Loading backtest results from {file_path}")
    
    try:
        # Load data from CSV
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        logger.info(f"Loaded {len(data)} data points")
        return data
        
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}")
        return None


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
        'signal_ratio': len(buy_signals) / len(sell_signals) if len(sell_signals) > 0 else float('inf')
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
    
    return metrics, data


def visualize_performance(data, metrics, output_dir=None):
    """
    Visualize backtest performance.
    
    Args:
        data: DataFrame with backtest results
        metrics: Dictionary with performance metrics
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Equity curve with buy/sell signals
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['portfolio_value'], label='Portfolio Value', color='blue')
    
    # Plot buy and sell signals
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['portfolio_value'], 
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['portfolio_value'], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title('Equity Curve with Signals')
    plt.legend()
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    cumulative_max = data['portfolio_value'].cummax()
    drawdown = (data['portfolio_value'] - cumulative_max) / cumulative_max
    plt.fill_between(data.index, drawdown, 0, color='red', alpha=0.3)
    plt.title('Drawdown')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"equity_curve_{timestamp}.png")
    plt.close()
    
    # 2. Performance metrics table
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    
    # Create table data
    table_data = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate', 'avg_win', 'avg_loss']:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        # Format key for display
        display_key = key.replace('_', ' ').title()
        
        table_data.append([display_key, formatted_value])
    
    # Create table
    table = plt.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left', colWidths=[0.6, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Performance Metrics', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"performance_metrics_{timestamp}.png")
    plt.close()
    
    # 3. Indicator analysis
    plt.figure(figsize=(12, 10))
    
    # Plot price with moving averages
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price', alpha=0.5)
    plt.plot(data.index, data['fast_ma'], label='Fast MA', alpha=0.8)
    plt.plot(data.index, data['slow_ma'], label='Slow MA', alpha=0.8)
    
    # Plot buy and sell signals
    plt.scatter(buy_signals.index, buy_signals['close'], 
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['close'], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title('Price with Moving Averages')
    plt.legend()
    plt.grid(True)
    
    # Plot RSI if available
    if 'RSI_14' in data.columns:
        plt.subplot(3, 1, 2)
        plt.plot(data.index, data['RSI_14'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        plt.title('RSI Indicator')
        plt.legend()
        plt.grid(True)
    
    # Plot MACD if available
    if 'MACD_12_26_9' in data.columns:
        plt.subplot(3, 1, 3)
        plt.plot(data.index, data['MACD_12_26_9'], label='MACD', color='blue')
        plt.plot(data.index, data['MACD_Signal_12_26_9'], label='Signal', color='red')
        plt.bar(data.index, data['MACD_Hist_12_26_9'], label='Histogram', color='green', alpha=0.3)
        plt.title('MACD Indicator')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"indicator_analysis_{timestamp}.png")
    plt.close()
    
    logger.info(f"Performance visualizations saved to {output_dir}")


def generate_performance_report(data, metrics, output_dir=None):
    """
    Generate a performance report in HTML format.
    
    Args:
        data: DataFrame with backtest results
        metrics: Dictionary with performance metrics
        output_dir: Directory to save the report
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Performance Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .metrics-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .metric-box {{
                background-color: #f2f2f2;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                width: 30%;
                box-sizing: border-box;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .positive {{
                color: green;
            }}
            .negative {{
                color: red;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Backtest Performance Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Performance Summary</h2>
        <div class="metrics-container">
    """
    
    # Add key metrics
    key_metrics = [
        ('Total Return', metrics.get('total_return', 0), True),
        ('Annualized Return', metrics.get('annualized_return', 0), True),
        ('Max Drawdown', metrics.get('max_drawdown', 0), False),
        ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), True),
        ('Win Rate', metrics.get('win_rate', 0), True),
        ('Profit Factor', metrics.get('profit_factor', 0), True)
    ]
    
    for name, value, is_positive in key_metrics:
        if isinstance(value, float):
            if name in ['Total Return', 'Annualized Return', 'Max Drawdown', 'Win Rate']:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        color_class = ''
        if is_positive:
            if value > 0:
                color_class = 'positive'
            elif value < 0:
                color_class = 'negative'
        
        html_content += f"""
            <div class="metric-box">
                <h3>{name}</h3>
                <div class="metric-value {color_class}">{formatted_value}</div>
            </div>
        """
    
    html_content += """
        </div>
        
        <h2>Detailed Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
    """
    
    # Add all metrics to table
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate', 'avg_win', 'avg_loss']:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        # Format key for display
        display_key = key.replace('_', ' ').title()
        
        html_content += f"""
            <tr>
                <td>{display_key}</td>
                <td>{formatted_value}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
    """
    
    # Add images
    image_files = list(output_dir.glob(f"*_{timestamp}.png"))
    for img_file in image_files:
        html_content += f"""
        <div>
            <h3>{img_file.stem.replace('_' + timestamp, '').replace('_', ' ').title()}</h3>
            <img src="{img_file.name}" alt="{img_file.stem}">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML report
    report_file = output_dir / f"performance_report_{timestamp}.html"
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Performance report saved to {report_file}")
    
    return report_file


def main():
    """Main function to analyze backtest performance."""
    logger.info("Starting backtest performance analysis")
    
    # Load backtest results
    data = load_backtest_results()
    if data is None:
        logger.error("Failed to load backtest results")
        return
    
    # Calculate performance metrics
    metrics, data_with_metrics = calculate_performance_metrics(data)
    
    # Log metrics
    logger.info("Performance metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Visualize performance
    visualize_performance(data_with_metrics, metrics, output_dir)
    
    # Generate performance report
    report_file = generate_performance_report(data_with_metrics, metrics, output_dir)
    
    logger.info("Backtest performance analysis completed")
    logger.info(f"Performance report saved to {report_file}")


if __name__ == "__main__":
    main()
