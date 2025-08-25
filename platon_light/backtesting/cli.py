"""
Command-line interface for backtesting module

This module provides a command-line interface for running backtests
and analyzing results from the command line.
"""

import os
import sys
import logging
import argparse
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.utils.config_manager import ConfigManager


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Platon Light Backtesting CLI")

    # Required arguments
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )

    # Backtest parameters
    parser.add_argument(
        "--symbol", type=str, help="Trading pair symbol (e.g., BTCUSDT)"
    )
    parser.add_argument("--timeframe", type=str, help="Timeframe (e.g., 1m, 5m, 1h)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")

    # Strategy parameters
    parser.add_argument("--strategy", type=str, help="Strategy name")
    parser.add_argument(
        "--strategy-params", type=str, help="Strategy parameters (JSON string)"
    )

    # Risk parameters
    parser.add_argument("--commission", type=float, help="Commission rate")
    parser.add_argument("--slippage", type=float, help="Slippage rate")
    parser.add_argument("--initial-capital", type=float, help="Initial capital")

    # Output parameters
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    # Data parameters
    parser.add_argument(
        "--download-data", action="store_true", help="Download historical data"
    )
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")

    # Compare multiple strategies
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple strategies"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of strategy names to compare",
    )

    # Logging parameters
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def setup_logging(log_level: str, output_dir: Optional[str] = None):
    """
    Set up logging configuration

    Args:
        log_level: Logging level
        output_dir: Output directory for log file
    """
    # Create log directory if specified
    if output_dir:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path: str) -> Dict:
    """
    Load configuration from file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # Check if file exists
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Load configuration
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def update_config_from_args(config: Dict, args) -> Dict:
    """
    Update configuration with command line arguments

    Args:
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        Updated configuration dictionary
    """
    # Create a copy of the configuration
    updated_config = config.copy()

    # Update symbol and timeframe
    if args.symbol:
        updated_config.setdefault("trading", {})["symbols"] = [args.symbol]

    if args.timeframe:
        updated_config.setdefault("trading", {})["timeframes"] = [args.timeframe]

    # Update strategy
    if args.strategy:
        updated_config.setdefault("strategy", {})["name"] = args.strategy

    # Update strategy parameters
    if args.strategy_params:
        try:
            strategy_params = json.loads(args.strategy_params)
            updated_config.setdefault("strategy", {}).update(strategy_params)
        except json.JSONDecodeError:
            print(f"Error: Invalid strategy parameters JSON: {args.strategy_params}")
            sys.exit(1)

    # Update risk parameters
    if args.commission:
        updated_config.setdefault("backtesting", {})["commission"] = args.commission

    if args.slippage:
        updated_config.setdefault("backtesting", {})["slippage"] = args.slippage

    if args.initial_capital:
        updated_config.setdefault("backtesting", {})[
            "initial_capital"
        ] = args.initial_capital

    # Update output parameters
    if args.output_dir:
        updated_config.setdefault("backtesting", {})["output_dir"] = args.output_dir

    # Update data parameters
    if args.use_cache:
        updated_config.setdefault("backtesting", {})["use_cache"] = args.use_cache

    return updated_config


def run_backtest(config: Dict, args) -> Dict:
    """
    Run backtest with the given configuration

    Args:
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        Backtest results
    """
    # Get symbol and timeframe
    symbols = config.get("trading", {}).get("symbols", ["BTCUSDT"])
    timeframes = config.get("trading", {}).get("timeframes", ["1m"])

    symbol = args.symbol or symbols[0]
    timeframe = args.timeframe or timeframes[0]

    # Get date range
    end_date = datetime.now()
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print(
                f"Error: Invalid end date format: {args.end_date}. Expected YYYY-MM-DD"
            )
            sys.exit(1)

    start_date = end_date - timedelta(days=30)  # Default to 30 days
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(
                f"Error: Invalid start date format: {args.start_date}. Expected YYYY-MM-DD"
            )
            sys.exit(1)

    # Initialize components
    data_loader = DataLoader(config)
    backtest_engine = BacktestEngine(config)
    performance_analyzer = PerformanceAnalyzer(config)

    # Download data if requested
    if args.download_data:
        print(
            f"Downloading historical data for {symbol} ({timeframe}) from {start_date} to {end_date}"
        )
        data_loader.download_data(symbol, timeframe, start_date, end_date)

    # Run backtest
    print(
        f"Running backtest for {symbol} ({timeframe}) from {start_date} to {end_date}"
    )
    results = backtest_engine.run(symbol, timeframe, start_date, end_date)

    # Check for errors
    if "error" in results:
        print(f"Error running backtest: {results['error']}")
        sys.exit(1)

    # Analyze results
    print("Analyzing backtest results")
    analysis = performance_analyzer.analyze(results, save_plots=args.plot)

    # Generate report if requested
    if args.report:
        print("Generating backtest report")
        report_summary = backtest_engine.generate_report(
            analysis, output_dir=args.output_dir
        )
        print(report_summary)

    # Print summary
    print_summary(analysis)

    return analysis


def compare_strategies(config: Dict, args) -> Dict:
    """
    Compare multiple strategies

    Args:
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        Comparison results
    """
    # Get strategies to compare
    if not args.strategies:
        print("Error: No strategies specified for comparison")
        sys.exit(1)

    strategy_names = args.strategies.split(",")

    # Get symbol and timeframe
    symbols = config.get("trading", {}).get("symbols", ["BTCUSDT"])
    timeframes = config.get("trading", {}).get("timeframes", ["1m"])

    symbol = args.symbol or symbols[0]
    timeframe = args.timeframe or timeframes[0]

    # Get date range
    end_date = datetime.now()
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print(
                f"Error: Invalid end date format: {args.end_date}. Expected YYYY-MM-DD"
            )
            sys.exit(1)

    start_date = end_date - timedelta(days=30)  # Default to 30 days
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(
                f"Error: Invalid start date format: {args.start_date}. Expected YYYY-MM-DD"
            )
            sys.exit(1)

    # Initialize performance analyzer
    performance_analyzer = PerformanceAnalyzer(config)

    # Run backtest for each strategy
    results_list = []

    for strategy_name in strategy_names:
        print(f"Running backtest for strategy: {strategy_name}")

        # Update config with strategy
        strategy_config = config.copy()
        strategy_config.setdefault("strategy", {})["name"] = strategy_name

        # Initialize components
        backtest_engine = BacktestEngine(strategy_config)

        # Run backtest
        results = backtest_engine.run(symbol, timeframe, start_date, end_date)

        # Check for errors
        if "error" in results:
            print(f"Error running backtest for {strategy_name}: {results['error']}")
            continue

        # Analyze results
        analysis = performance_analyzer.analyze(results, save_plots=False)

        results_list.append(analysis)

    # Compare strategies
    if len(results_list) < 2:
        print("Error: Need at least two valid strategy results to compare")
        sys.exit(1)

    print("Comparing strategies")
    comparison = performance_analyzer.compare_strategies(results_list, strategy_names)

    # Print comparison
    print_comparison(comparison, strategy_names)

    return comparison


def print_summary(results: Dict):
    """
    Print backtest summary

    Args:
        results: Backtest results
    """
    metrics = results.get("metrics", {})

    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)

    print(f"Initial Capital: {metrics.get('initial_capital', 0):.2f}")
    print(f"Final Capital: {metrics.get('final_capital', 0):.2f}")
    print(
        f"Net Profit: {metrics.get('net_profit', 0):.2f} ({metrics.get('return_percent', 0):.2f}%)"
    )
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(
        f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f} ({metrics.get('max_drawdown_percent', 0):.2f}%)"
    )
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    print("=" * 50)


def print_comparison(comparison: Dict, strategy_names: List[str]):
    """
    Print strategy comparison

    Args:
        comparison: Comparison results
        strategy_names: List of strategy names
    """
    if "error" in comparison:
        print(f"Error in comparison: {comparison['error']}")
        return

    metrics_comparison = comparison.get("metrics_comparison", {})

    print("\n" + "=" * 50)
    print("STRATEGY COMPARISON")
    print("=" * 50)

    # Print key metrics for each strategy
    metrics = [
        "return_percent",
        "win_rate",
        "profit_factor",
        "max_drawdown_percent",
        "sharpe_ratio",
    ]

    # Print header
    print(f"{'Metric':<20}", end="")
    for strategy in strategy_names:
        print(f"{strategy:<15}", end="")
    print()

    # Print metrics
    for metric in metrics:
        print(f"{metric:<20}", end="")
        for strategy in strategy_names:
            value = metrics_comparison.get(strategy, {}).get(metric, 0)
            print(f"{value:<15.2f}", end="")
        print()

    print("=" * 50)


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Update configuration with command line arguments
    config = update_config_from_args(config, args)

    # Set up logging
    setup_logging(args.log_level, args.output_dir)

    # Run backtest or compare strategies
    if args.compare:
        compare_strategies(config, args)
    else:
        run_backtest(config, args)


if __name__ == "__main__":
    main()
