#!/usr/bin/env python
"""
Comprehensive backtest runner script for Platon Light.

This script runs backtests using optimized parameters for different
strategies, timeframes, and symbols, and generates a comprehensive report.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import json
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from platon_light.backtesting.strategies.moving_average_crossover import MovingAverageCrossover


def load_sample_data(symbol='BTCUSDT', timeframe='1h'):
    """
    Load sample data for backtesting.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe (e.g., '1h', '4h', '1d')
        
    Returns:
        DataFrame with OHLCV data
    """
    # Construct file path
    file_path = Path(__file__).parent / 'data' / 'sample' / f"{symbol}_{timeframe}.csv"
    
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


def load_optimized_parameters(strategy_name='moving_average_crossover'):
    """
    Load optimized parameters for a strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary with optimized parameters
    """
    # Find the most recent optimization results
    results_dir = Path(__file__).parent / 'results'
    json_files = list(results_dir.glob('top_parameters_*.json'))
    
    if not json_files:
        logger.warning("No optimization results found, using default parameters")
        return {
            'fast_ma_type': 'EMA',
            'slow_ma_type': 'SMA',
            'fast_period': 20,
            'slow_period': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'use_filters': True
        }
    
    # Sort by modification time (most recent first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    json_file = json_files[0]
    
    logger.info(f"Loading optimized parameters from {json_file}")
    
    try:
        # Load parameters from JSON
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        # Get parameters from the best result
        if results and 'params' in results[0]:
            params = results[0]['params']
            logger.info(f"Loaded optimized parameters: {params}")
            return params
        else:
            logger.warning("Invalid optimization results format, using default parameters")
            return {
                'fast_ma_type': 'EMA',
                'slow_ma_type': 'SMA',
                'fast_period': 20,
                'slow_period': 50,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'use_filters': True
            }
        
    except Exception as e:
        logger.error(f"Error loading optimized parameters: {e}")
        logger.warning("Using default parameters")
        return {
            'fast_ma_type': 'EMA',
            'slow_ma_type': 'SMA',
            'fast_period': 20,
            'slow_period': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'use_filters': True
        }


def run_backtest(data, strategy_params):
    """
    Run a backtest using the specified strategy parameters.
    
    Args:
        data: DataFrame with OHLCV data
        strategy_params: Dictionary with strategy parameters
        
    Returns:
        DataFrame with backtest results
    """
    logger.info(f"Running backtest with strategy parameters: {strategy_params}")
    
    # Create strategy instance
    strategy = MovingAverageCrossover(**strategy_params)
    
    # Run strategy
    result = strategy.run(data)
    
    logger.info(f"Backtest completed with {len(result[result['signal'] != 0])} signals")
    
    return result


def calculate_performance_metrics(data):
    """
    Calculate performance metrics for the backtest.
    
    Args:
        data: DataFrame with backtest results
        
    Returns:
        Dictionary with performance metrics and DataFrame with trade data
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
    entry_time = None
    
    trades = []
    
    for i in range(1, len(data)):
        # Update position based on signals
        if data['signal'].iloc[i] == 1 and position == 0:
            # Buy
            position = 1
            entry_price = data['close'].iloc[i]
            entry_time = data.index[i]
        elif data['signal'].iloc[i] == -1 and position == 1:
            # Sell
            position = 0
            exit_price = data['close'].iloc[i]
            exit_time = data.index[i]
            
            # Record trade
            trade_return = exit_price / entry_price - 1
            trade_duration = (exit_time - entry_time).total_seconds() / 3600  # hours
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': trade_return,
                'duration': trade_duration
            })
        
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
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
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
        
        # Win rate and other trade metrics
        if not trades_df.empty:
            metrics['win_rate'] = (trades_df['return'] > 0).mean()
            metrics['avg_win'] = trades_df.loc[trades_df['return'] > 0, 'return'].mean() if (trades_df['return'] > 0).any() else 0
            metrics['avg_loss'] = trades_df.loc[trades_df['return'] <= 0, 'return'].mean() if (trades_df['return'] <= 0).any() else 0
            
            # Profit factor
            gross_profit = trades_df.loc[trades_df['return'] > 0, 'return'].sum()
            gross_loss = abs(trades_df.loc[trades_df['return'] <= 0, 'return'].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average trade duration
            metrics['avg_trade_duration'] = trades_df['duration'].mean()
    
    return metrics, data, trades_df


def visualize_backtest_results(data, trades_df, metrics, symbol, timeframe, output_dir=None):
    """
    Visualize backtest results.
    
    Args:
        data: DataFrame with backtest results
        trades_df: DataFrame with trade data
        metrics: Dictionary with performance metrics
        symbol: Trading pair symbol
        timeframe: Timeframe
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Price chart with signals and moving averages
    plt.figure(figsize=(12, 8))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price', alpha=0.5)
    
    # Plot moving averages
    plt.plot(data.index, data['fast_ma'], label='Fast MA', alpha=0.8)
    plt.plot(data.index, data['slow_ma'], label='Slow MA', alpha=0.8)
    
    # Plot buy and sell signals
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], 
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['close'], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f"{symbol} {timeframe} - Price and Signals")
    plt.legend()
    plt.grid(True)
    
    # Plot portfolio value
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['portfolio_value'], label='Portfolio Value', color='blue')
    
    # Plot buy and sell signals on portfolio value
    plt.scatter(buy_signals.index, buy_signals['portfolio_value'], 
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['portfolio_value'], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f"{symbol} {timeframe} - Portfolio Value")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{symbol}_{timeframe}_backtest_{timestamp}.png")
    plt.close()
    
    # 2. Trade analysis
    if not trades_df.empty:
        plt.figure(figsize=(12, 10))
        
        # Plot trade returns
        plt.subplot(2, 2, 1)
        plt.bar(range(len(trades_df)), trades_df['return'], color=['green' if r > 0 else 'red' for r in trades_df['return']])
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Trade Returns')
        plt.xlabel('Trade Number')
        plt.ylabel('Return')
        plt.grid(True)
        
        # Plot trade durations
        plt.subplot(2, 2, 2)
        plt.hist(trades_df['duration'], bins=20, alpha=0.7)
        plt.title('Trade Duration Distribution')
        plt.xlabel('Duration (hours)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot cumulative returns
        plt.subplot(2, 2, 3)
        cumulative_returns = (1 + trades_df['return']).cumprod() - 1
        plt.plot(range(len(cumulative_returns)), cumulative_returns)
        plt.title('Cumulative Trade Returns')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 2, 4)
        cumulative_max = data['portfolio_value'].cummax()
        drawdown = (data['portfolio_value'] - cumulative_max) / cumulative_max
        plt.fill_between(data.index, drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{symbol}_{timeframe}_trade_analysis_{timestamp}.png")
        plt.close()
    
    # 3. Performance metrics table
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
    
    plt.title(f"{symbol} {timeframe} - Performance Metrics", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"{symbol}_{timeframe}_metrics_{timestamp}.png")
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def generate_html_report(results, output_dir=None):
    """
    Generate an HTML report with backtest results.
    
    Args:
        results: List of dictionaries with backtest results
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
        <title>Comprehensive Backtest Report</title>
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
                margin-bottom: 20px;
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
            .backtest-container {{
                margin-bottom: 40px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Comprehensive Backtest Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Timeframe</th>
                <th>Total Return</th>
                <th>Annualized Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Profit Factor</th>
            </tr>
    """
    
    # Add summary rows
    for result in results:
        symbol = result['symbol']
        timeframe = result['timeframe']
        metrics = result['metrics']
        
        html_content += f"""
            <tr>
                <td>{symbol}</td>
                <td>{timeframe}</td>
                <td class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">{metrics.get('total_return', 0):.2%}</td>
                <td class="{'positive' if metrics.get('annualized_return', 0) > 0 else 'negative'}">{metrics.get('annualized_return', 0):.2%}</td>
                <td>{metrics.get('sharpe_ratio', 0):.4f}</td>
                <td>{metrics.get('max_drawdown', 0):.2%}</td>
                <td>{metrics.get('win_rate', 0):.2%}</td>
                <td>{metrics.get('profit_factor', 0):.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Detailed Results</h2>
    """
    
    # Add detailed sections for each backtest
    for result in results:
        symbol = result['symbol']
        timeframe = result['timeframe']
        metrics = result['metrics']
        image_files = result['image_files']
        
        html_content += f"""
        <div class="backtest-container">
            <h3>{symbol} {timeframe}</h3>
            
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
            color_class = ''
            if is_positive:
                if value > 0:
                    color_class = 'positive'
                elif value < 0:
                    color_class = 'negative'
            
            html_content += f"""
                <div class="metric-box">
                    <h4>{name}</h4>
                    <div class="metric-value {color_class}">{value:.2%}</div>
                </div>
            """
        
        html_content += """
            </div>
            
            <h4>Performance Metrics</h4>
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
            
            <h4>Visualizations</h4>
        """
        
        # Add images
        for img_file in image_files:
            img_name = Path(img_file).name
            img_title = img_name.split('_')[2].replace('.png', '').title()
            
            html_content += f"""
            <div>
                <h5>{img_title}</h5>
                <img src="{img_name}" alt="{img_title}">
            </div>
            """
        
        html_content += """
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML report
    report_file = output_dir / f"comprehensive_backtest_report_{timestamp}.html"
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Comprehensive report saved to {report_file}")
    
    return report_file


def main():
    """Main function to run comprehensive backtests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comprehensive backtests')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                        help='Symbols to backtest')
    parser.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'],
                        help='Timeframes to backtest')
    args = parser.parse_args()
    
    logger.info("Starting comprehensive backtests")
    
    # Load optimized parameters
    params = load_optimized_parameters()
    
    # Create results directory
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run backtests for each symbol and timeframe
    results = []
    
    for symbol in args.symbols:
        for timeframe in args.timeframes:
            logger.info(f"Running backtest for {symbol} {timeframe}")
            
            # Load data
            data = load_sample_data(symbol, timeframe)
            if data is None:
                logger.error(f"Failed to load data for {symbol} {timeframe}")
                continue
            
            # Run backtest
            backtest_result = run_backtest(data, params)
            
            # Calculate performance metrics
            metrics, data_with_metrics, trades_df = calculate_performance_metrics(backtest_result)
            
            # Visualize results
            visualize_backtest_results(data_with_metrics, trades_df, metrics, symbol, timeframe, results_dir)
            
            # Find image files
            image_files = list(results_dir.glob(f"{symbol}_{timeframe}_*_{timestamp}.png"))
            
            # Add to results
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'metrics': metrics,
                'image_files': [str(f) for f in image_files]
            })
            
            # Save backtest data
            backtest_data_file = results_dir / f"{symbol}_{timeframe}_backtest_data_{timestamp}.csv"
            data_with_metrics.to_csv(backtest_data_file)
            
            # Save trades data
            if not trades_df.empty:
                trades_data_file = results_dir / f"{symbol}_{timeframe}_trades_{timestamp}.csv"
                trades_df.to_csv(trades_data_file)
            
            logger.info(f"Backtest for {symbol} {timeframe} completed")
    
    # Generate comprehensive report
    if results:
        report_file = generate_html_report(results, results_dir)
        logger.info(f"Comprehensive report generated: {report_file}")
    
    logger.info("Comprehensive backtests completed")


if __name__ == "__main__":
    main()
