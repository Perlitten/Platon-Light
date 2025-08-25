#!/usr/bin/env python
"""
Automated Backtesting Workflow for Platon Light

This script automates the entire backtesting workflow:
1. Generate sample data (if needed)
2. Run basic backtest
3. Optimize strategy parameters
4. Run comprehensive backtest with optimized parameters
5. Generate performance reports

Usage:
    python run_backtest_workflow.py [options]

Options:
    --use-real-data: Use real market data instead of sample data
    --symbols: Comma-separated list of symbols to test (default: BTCUSDT,ETHUSDT)
    --timeframes: Comma-separated list of timeframes to test (default: 1h,4h,1d)
    --start-date: Start date for real data (default: 90 days ago)
    --end-date: End date for real data (default: today)
    --skip-steps: Comma-separated list of steps to skip (e.g., "generate,optimize")
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run automated backtesting workflow")
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real market data instead of sample data",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated list of symbols to test",
    )
    parser.add_argument(
        "--timeframes",
        default="1h,4h,1d",
        help="Comma-separated list of timeframes to test",
    )
    parser.add_argument(
        "--start-date",
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date for real data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for real data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-steps",
        default="",
        help='Comma-separated list of steps to skip (e.g., "generate,optimize")',
    )

    return parser.parse_args()


def run_command(command, cwd=None):
    """Run a command and log its output."""
    logger.info(f"Running command: {command}")

    try:
        result = subprocess.run(
            command, cwd=cwd, shell=True, check=True, text=True, capture_output=True
        )

        logger.info(f"Command completed successfully")
        logger.debug(result.stdout)

        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stderr)

        return False, e.stderr


def generate_sample_data(args):
    """Generate sample data for backtesting."""
    logger.info("Step 1: Generating sample data")

    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")

    success, output = run_command(
        f"python generate_sample_data.py --symbols {' '.join(symbols)} --timeframes {' '.join(timeframes)}",
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    return success


def run_basic_backtest(args):
    """Run a basic backtest to validate the strategy."""
    logger.info("Step 2: Running basic backtest")

    if args.use_real_data:
        # Check if credentials are set up
        success, _ = run_command(
            "python launch_live_backtest.py --symbol BTCUSDT --timeframe 1h "
            f"--start-date {args.start_date} --end-date {args.end_date}",
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    else:
        success, _ = run_command(
            "python run_backtest.py", cwd=os.path.dirname(os.path.abspath(__file__))
        )

    return success


def optimize_strategy_parameters(args):
    """Optimize strategy parameters using grid search."""
    logger.info("Step 3: Optimizing strategy parameters")

    if args.use_real_data:
        # For real data, we need to specify the symbol and timeframe
        success, _ = run_command(
            "python optimize_strategy_parameters.py --use-real-data "
            f"--symbol BTCUSDT --timeframe 1h --start-date {args.start_date} --end-date {args.end_date}",
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    else:
        success, _ = run_command(
            "python optimize_strategy_parameters.py",
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

    return success


def run_comprehensive_backtest(args):
    """Run comprehensive backtest with optimized parameters."""
    logger.info("Step 4: Running comprehensive backtest")

    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")

    if args.use_real_data:
        # For real data, we need to run each symbol/timeframe combination separately
        all_success = True

        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"Running comprehensive backtest for {symbol} {timeframe}")

                success, _ = run_command(
                    f"python launch_live_backtest.py --symbol {symbol} --timeframe {timeframe} "
                    f"--start-date {args.start_date} --end-date {args.end_date}",
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

                if not success:
                    all_success = False

        return all_success
    else:
        # For sample data, we can run all combinations at once
        success, _ = run_command(
            f"python run_comprehensive_backtest.py --symbols {' '.join(symbols)} --timeframes {' '.join(timeframes)}",
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        return success


def analyze_performance(args):
    """Analyze backtest performance and generate reports."""
    logger.info("Step 5: Analyzing performance")

    # Find the most recent backtest results
    results_dir = Path(__file__).parent / "results"
    csv_files = list(results_dir.glob("*_backtest_*.csv"))

    if not csv_files:
        logger.warning("No backtest results found to analyze")
        return False

    # Sort by modification time (most recent first)
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    csv_file = csv_files[0]

    logger.info(f"Analyzing performance for {csv_file}")

    success, _ = run_command(
        f"python analyze_backtest_performance.py --input-file {csv_file}",
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    return success


def serve_reports():
    """Serve the generated reports."""
    logger.info("Step 6: Serving reports")

    # Start the report server in the background
    subprocess.Popen(
        "python serve_report.py",
        cwd=os.path.dirname(os.path.abspath(__file__)),
        shell=True,
    )

    logger.info("Report server started at http://localhost:8080")
    logger.info("Press Ctrl+C to stop the server when done")

    return True


def main():
    """Main function to run the automated backtesting workflow."""
    args = parse_arguments()

    logger.info("Starting automated backtesting workflow")

    # Determine which steps to skip
    skip_steps = args.skip_steps.split(",") if args.skip_steps else []

    # Step 1: Generate sample data (if not using real data)
    if not args.use_real_data and "generate" not in skip_steps:
        if not generate_sample_data(args):
            logger.error("Failed to generate sample data")
            return

    # Step 2: Run basic backtest
    if "basic" not in skip_steps:
        if not run_basic_backtest(args):
            logger.error("Failed to run basic backtest")
            return

    # Step 3: Optimize strategy parameters
    if "optimize" not in skip_steps:
        if not optimize_strategy_parameters(args):
            logger.error("Failed to optimize strategy parameters")
            return

    # Step 4: Run comprehensive backtest
    if "comprehensive" not in skip_steps:
        if not run_comprehensive_backtest(args):
            logger.error("Failed to run comprehensive backtest")
            return

    # Step 5: Analyze performance
    if "analyze" not in skip_steps:
        if not analyze_performance(args):
            logger.error("Failed to analyze performance")
            return

    # Step 6: Serve reports
    if "serve" not in skip_steps:
        if not serve_reports():
            logger.error("Failed to serve reports")
            return

    logger.info("Automated backtesting workflow completed successfully")
    logger.info("View the reports at http://localhost:8080")


if __name__ == "__main__":
    main()
