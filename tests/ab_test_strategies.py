#!/usr/bin/env python
"""
A/B Testing Script for Trading Strategies

This script provides a framework for conducting A/B tests between different trading strategies
or strategy variants using the Platon Light backtesting module.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import json
import argparse
import logging

# Add parent directory to path to import Platon Light modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.core.strategy_factory import StrategyFactory
from platon_light.core.base_strategy import BaseStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ab_test_strategies.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ABTester:
    """Class for conducting A/B tests between trading strategies"""

    def __init__(self, output_dir="ab_test_results"):
        """
        Initialize the A/B tester.

        Args:
            output_dir (str): Directory to save test results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logger

    def run_test(
        self,
        symbol,
        timeframe,
        start_date,
        end_date,
        config_a,
        config_b,
        name_a="Strategy A",
        name_b="Strategy B",
        save_results=True,
    ):
        """
        Run an A/B test between two strategies.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            config_a (dict): Configuration for strategy A
            config_b (dict): Configuration for strategy B
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B
            save_results (bool): Whether to save results to disk

        Returns:
            dict: Dictionary containing test results
        """
        self.logger.info(
            f"Running A/B test for {symbol} from {start_date} to {end_date}"
        )
        self.logger.info(f"Testing {name_a} vs {name_b}")

        # Run backtests
        backtest_engine_a = BacktestEngine(config_a)
        results_a = backtest_engine_a.run(symbol, timeframe, start_date, end_date)

        backtest_engine_b = BacktestEngine(config_b)
        results_b = backtest_engine_b.run(symbol, timeframe, start_date, end_date)

        # Compare key metrics
        metrics = [
            "return_percent",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_percent",
            "win_rate",
            "profit_factor",
            "avg_trade_percent",
        ]

        metrics_comparison = {}

        self.logger.info("\nKey Metrics Comparison:")
        for metric in metrics:
            value_a = results_a["metrics"].get(metric, 0)
            value_b = results_b["metrics"].get(metric, 0)
            difference = value_b - value_a

            if metric in ["max_drawdown_percent"]:  # Lower is better
                is_better = difference < 0
                difference_str = f"{difference:.2f}"
            else:  # Higher is better
                is_better = difference > 0
                difference_str = (
                    f"+{difference:.2f}" if difference > 0 else f"{difference:.2f}"
                )

            result_symbol = "✓" if is_better else "✗"

            self.logger.info(
                f"{metric}: {value_a:.2f} vs {value_b:.2f} ({difference_str}) {result_symbol}"
            )

            metrics_comparison[metric] = {
                "value_a": value_a,
                "value_b": value_b,
                "difference": difference,
                "is_better": is_better,
            }

        # Calculate statistical significance
        significance_results = self.test_statistical_significance(results_a, results_b)

        # Compare equity curves
        equity_curve_path = self.compare_equity_curves(
            results_a, results_b, name_a, name_b
        )

        # Compare drawdowns
        drawdown_path = self.compare_drawdowns(results_a, results_b, name_a, name_b)

        # Compare trade distributions
        trade_dist_path = self.compare_trade_distributions(
            results_a, results_b, name_a, name_b
        )

        # Compare monthly returns
        monthly_returns_path = self.compare_monthly_returns(
            results_a, results_b, name_a, name_b
        )

        # Determine winner
        if significance_results["is_significant"]:
            # If statistically significant, choose based on Sharpe ratio
            if (
                results_b["metrics"]["sharpe_ratio"]
                > results_a["metrics"]["sharpe_ratio"]
            ):
                winner = name_b
            else:
                winner = name_a

            self.logger.info(
                f"\nWinner: {winner} (statistically significant difference)"
            )
        else:
            # If not statistically significant, consider them equivalent
            self.logger.info(
                "\nNo statistically significant difference between strategies"
            )

            # Still provide a recommendation based on Sharpe ratio
            if (
                results_b["metrics"]["sharpe_ratio"]
                > results_a["metrics"]["sharpe_ratio"]
            ):
                self.logger.info(
                    f"Slight preference for {name_b} based on Sharpe ratio"
                )
                winner = f"{name_b} (slight preference)"
            else:
                self.logger.info(
                    f"Slight preference for {name_a} based on Sharpe ratio"
                )
                winner = f"{name_a} (slight preference)"

        # Compile results
        test_results = {
            "test_parameters": {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "strategy_a_name": name_a,
                "strategy_b_name": name_b,
            },
            "metrics_comparison": metrics_comparison,
            "significance_results": significance_results,
            "winner": winner,
            "visualizations": {
                "equity_curve": equity_curve_path,
                "drawdown": drawdown_path,
                "trade_distribution": trade_dist_path,
                "monthly_returns": monthly_returns_path,
            },
        }

        # Save results if requested
        if save_results:
            self.save_test_results(test_results, name_a, name_b)

        return test_results

    def test_statistical_significance(self, results_a, results_b, alpha=0.05):
        """
        Test if the performance difference between two strategies is statistically significant.

        Args:
            results_a (dict): Results from strategy A
            results_b (dict): Results from strategy B
            alpha (float): Significance level

        Returns:
            dict: Dictionary containing test results
        """
        # Calculate daily returns
        equity_a = pd.DataFrame(results_a["equity_curve"])
        equity_a["timestamp"] = pd.to_datetime(equity_a["timestamp"], unit="ms")
        equity_a.set_index("timestamp", inplace=True)
        equity_a["daily_return"] = equity_a["equity"].pct_change()

        equity_b = pd.DataFrame(results_b["equity_curve"])
        equity_b["timestamp"] = pd.to_datetime(equity_b["timestamp"], unit="ms")
        equity_b.set_index("timestamp", inplace=True)
        equity_b["daily_return"] = equity_b["equity"].pct_change()

        # Align the time series
        common_index = equity_a.index.intersection(equity_b.index)
        returns_a = equity_a.loc[common_index, "daily_return"].dropna()
        returns_b = equity_b.loc[common_index, "daily_return"].dropna()

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(returns_a, returns_b)

        # Check if the difference is significant
        is_significant = p_value < alpha

        # Calculate effect size (Cohen's d)
        mean_a = returns_a.mean()
        mean_b = returns_b.mean()
        std_pooled = ((returns_a.std() ** 2 + returns_b.std() ** 2) / 2) ** 0.5
        effect_size = abs(mean_a - mean_b) / std_pooled if std_pooled > 0 else 0

        # Interpret effect size
        if effect_size < 0.2:
            effect_interpretation = "negligible"
        elif effect_size < 0.5:
            effect_interpretation = "small"
        elif effect_size < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        # Print results
        self.logger.info(f"T-statistic: {t_stat:.4f}")
        self.logger.info(f"P-value: {p_value:.4f}")
        self.logger.info(f"Statistically significant: {is_significant}")
        self.logger.info(
            f"Effect size (Cohen's d): {effect_size:.4f} ({effect_interpretation})"
        )

        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "effect_size": effect_size,
            "effect_interpretation": effect_interpretation,
        }

    def compare_equity_curves(
        self, results_a, results_b, name_a="Strategy A", name_b="Strategy B"
    ):
        """
        Compare equity curves of two strategies.

        Args:
            results_a (dict): Results from strategy A
            results_b (dict): Results from strategy B
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B

        Returns:
            str: Path to saved figure
        """
        plt.figure(figsize=(12, 6))

        # Extract equity curve data
        equity_a = pd.DataFrame(results_a["equity_curve"])
        equity_a["timestamp"] = pd.to_datetime(equity_a["timestamp"], unit="ms")

        equity_b = pd.DataFrame(results_b["equity_curve"])
        equity_b["timestamp"] = pd.to_datetime(equity_b["timestamp"], unit="ms")

        # Normalize to percentage returns
        initial_equity_a = equity_a["equity"].iloc[0]
        equity_a["equity_pct"] = (equity_a["equity"] / initial_equity_a - 1) * 100

        initial_equity_b = equity_b["equity"].iloc[0]
        equity_b["equity_pct"] = (equity_b["equity"] / initial_equity_b - 1) * 100

        # Plot equity curves
        plt.plot(equity_a["timestamp"], equity_a["equity_pct"], label=name_a)
        plt.plot(equity_b["timestamp"], equity_b["equity_pct"], label=name_b)

        plt.title(f"Equity Curve: {name_a} vs {name_b}")
        plt.xlabel("Date")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        filename = f"{self.output_dir}/equity_curve_comparison.png"
        plt.savefig(filename)
        plt.close()

        return filename

    def compare_drawdowns(
        self, results_a, results_b, name_a="Strategy A", name_b="Strategy B"
    ):
        """
        Compare drawdown profiles of two strategies.

        Args:
            results_a (dict): Results from strategy A
            results_b (dict): Results from strategy B
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B

        Returns:
            str: Path to saved figure
        """
        # Extract equity data
        equity_a = pd.DataFrame(results_a["equity_curve"])
        equity_a["timestamp"] = pd.to_datetime(equity_a["timestamp"], unit="ms")
        equity_a.set_index("timestamp", inplace=True)

        equity_b = pd.DataFrame(results_b["equity_curve"])
        equity_b["timestamp"] = pd.to_datetime(equity_b["timestamp"], unit="ms")
        equity_b.set_index("timestamp", inplace=True)

        # Calculate drawdowns
        equity_a["peak"] = equity_a["equity"].cummax()
        equity_a["drawdown"] = (
            (equity_a["equity"] - equity_a["peak"]) / equity_a["peak"] * 100
        )

        equity_b["peak"] = equity_b["equity"].cummax()
        equity_b["drawdown"] = (
            (equity_b["equity"] - equity_b["peak"]) / equity_b["peak"] * 100
        )

        # Plot drawdowns
        plt.figure(figsize=(12, 6))
        plt.plot(equity_a.index, equity_a["drawdown"], label=name_a)
        plt.plot(equity_b.index, equity_b["drawdown"], label=name_b)
        plt.title("Drawdown Comparison")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        filename = f"{self.output_dir}/drawdown_comparison.png"
        plt.savefig(filename)
        plt.close()

        return filename

    def compare_trade_distributions(
        self, results_a, results_b, name_a="Strategy A", name_b="Strategy B"
    ):
        """
        Compare trade profit/loss distributions.

        Args:
            results_a (dict): Results from strategy A
            results_b (dict): Results from strategy B
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B

        Returns:
            str: Path to saved figure
        """
        # Extract trade data
        trades_a = (
            pd.DataFrame(results_a["trades"])
            if "trades" in results_a
            else pd.DataFrame()
        )
        trades_b = (
            pd.DataFrame(results_b["trades"])
            if "trades" in results_b
            else pd.DataFrame()
        )

        # Plot distributions
        plt.figure(figsize=(12, 6))

        if not trades_a.empty and "profit_loss_percent" in trades_a.columns:
            plt.hist(trades_a["profit_loss_percent"], alpha=0.5, bins=20, label=name_a)

        if not trades_b.empty and "profit_loss_percent" in trades_b.columns:
            plt.hist(trades_b["profit_loss_percent"], alpha=0.5, bins=20, label=name_b)

        plt.title("Trade Profit/Loss Distribution")
        plt.xlabel("Profit/Loss (%)")
        plt.ylabel("Number of Trades")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        filename = f"{self.output_dir}/trade_distribution_comparison.png"
        plt.savefig(filename)
        plt.close()

        return filename

    def compare_monthly_returns(
        self, results_a, results_b, name_a="Strategy A", name_b="Strategy B"
    ):
        """
        Compare monthly returns of two strategies.

        Args:
            results_a (dict): Results from strategy A
            results_b (dict): Results from strategy B
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B

        Returns:
            str: Path to saved figure
        """
        # Extract equity data
        equity_a = pd.DataFrame(results_a["equity_curve"])
        equity_a["timestamp"] = pd.to_datetime(equity_a["timestamp"], unit="ms")
        equity_a.set_index("timestamp", inplace=True)

        equity_b = pd.DataFrame(results_b["equity_curve"])
        equity_b["timestamp"] = pd.to_datetime(equity_b["timestamp"], unit="ms")
        equity_b.set_index("timestamp", inplace=True)

        # Calculate daily returns
        equity_a["daily_return"] = equity_a["equity"].pct_change()
        equity_b["daily_return"] = equity_b["equity"].pct_change()

        # Resample to monthly returns
        monthly_returns_a = (equity_a["daily_return"] + 1).resample("M").prod() - 1
        monthly_returns_b = (equity_b["daily_return"] + 1).resample("M").prod() - 1

        # Create DataFrame for plotting
        monthly_df = pd.DataFrame(
            {name_a: monthly_returns_a, name_b: monthly_returns_b}
        )

        # Plot monthly returns
        plt.figure(figsize=(14, 7))
        monthly_df.plot(kind="bar", figsize=(14, 7))
        plt.title("Monthly Returns Comparison")
        plt.xlabel("Month")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()

        # Save figure
        filename = f"{self.output_dir}/monthly_returns_comparison.png"
        plt.savefig(filename)
        plt.close()

        return filename

    def save_test_results(self, test_results, name_a, name_b):
        """
        Save test results to disk.

        Args:
            test_results (dict): Test results
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B

        Returns:
            str: Path to saved results
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/ab_test_{name_a}_vs_{name_b}_{timestamp}.json"

        # Convert datetime objects to strings
        results_json = json.dumps(test_results, default=str, indent=2)

        # Save to file
        with open(filename, "w") as f:
            f.write(results_json)

        self.logger.info(f"Test results saved to {filename}")

        return filename

    def run_multi_period_test(
        self,
        symbol,
        timeframe,
        periods,
        config_a,
        config_b,
        name_a="Strategy A",
        name_b="Strategy B",
    ):
        """
        Run A/B tests across multiple time periods.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            periods (list): List of (start_date, end_date) tuples
            config_a (dict): Configuration for strategy A
            config_b (dict): Configuration for strategy B
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B

        Returns:
            dict: Dictionary containing test results for each period
        """
        self.logger.info(f"Running multi-period A/B test for {symbol}")
        self.logger.info(f"Testing {name_a} vs {name_b} across {len(periods)} periods")

        period_results = {}

        for i, (start_date, end_date) in enumerate(periods):
            period_name = f"Period {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            self.logger.info(f"\nTesting {period_name}")

            # Run test for this period
            results = self.run_test(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                config_a=config_a,
                config_b=config_b,
                name_a=name_a,
                name_b=name_b,
                save_results=False,  # Don't save individual period results
            )

            period_results[period_name] = results

        # Calculate overall win rate for strategy B
        wins = sum(
            1 for result in period_results.values() if name_b in result["winner"]
        )
        win_rate = (wins / len(periods)) * 100

        self.logger.info(f"\nOverall Results:")
        self.logger.info(
            f"{name_b} win rate across {len(periods)} periods: {win_rate:.2f}%"
        )

        if win_rate > 50:
            self.logger.info(f"Overall winner: {name_b}")
            overall_winner = name_b
        else:
            self.logger.info(f"Overall winner: {name_a}")
            overall_winner = name_a

        # Compile overall results
        overall_results = {
            "test_parameters": {
                "symbol": symbol,
                "timeframe": timeframe,
                "periods": [
                    (start.isoformat(), end.isoformat()) for start, end in periods
                ],
                "strategy_a_name": name_a,
                "strategy_b_name": name_b,
            },
            "period_results": period_results,
            "overall_stats": {
                "total_periods": len(periods),
                "strategy_b_wins": wins,
                "strategy_b_win_rate": win_rate,
                "overall_winner": overall_winner,
            },
        }

        # Save overall results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{self.output_dir}/multi_period_test_{name_a}_vs_{name_b}_{timestamp}.json"
        )

        with open(filename, "w") as f:
            f.write(json.dumps(overall_results, default=str, indent=2))

        self.logger.info(f"Multi-period test results saved to {filename}")

        return overall_results

    def run_multi_asset_test(
        self,
        symbols,
        timeframe,
        start_date,
        end_date,
        config_a,
        config_b,
        name_a="Strategy A",
        name_b="Strategy B",
    ):
        """
        Run A/B tests across multiple assets.

        Args:
            symbols (list): List of trading symbols
            timeframe (str): Timeframe for data
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            config_a (dict): Configuration for strategy A
            config_b (dict): Configuration for strategy B
            name_a (str): Name for strategy A
            name_b (str): Name for strategy B

        Returns:
            dict: Dictionary containing test results for each asset
        """
        self.logger.info(f"Running multi-asset A/B test across {len(symbols)} symbols")
        self.logger.info(
            f"Testing {name_a} vs {name_b} from {start_date} to {end_date}"
        )

        asset_results = {}

        for symbol in symbols:
            self.logger.info(f"\nTesting {symbol}")

            # Run test for this asset
            results = self.run_test(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                config_a=config_a,
                config_b=config_b,
                name_a=name_a,
                name_b=name_b,
                save_results=False,  # Don't save individual asset results
            )

            asset_results[symbol] = results

        # Calculate overall win rate for strategy B
        wins = sum(1 for result in asset_results.values() if name_b in result["winner"])
        win_rate = (wins / len(symbols)) * 100

        self.logger.info(f"\nOverall Results:")
        self.logger.info(
            f"{name_b} win rate across {len(symbols)} assets: {win_rate:.2f}%"
        )

        if win_rate > 50:
            self.logger.info(f"Overall winner: {name_b}")
            overall_winner = name_b
        else:
            self.logger.info(f"Overall winner: {name_a}")
            overall_winner = name_a

        # Compile overall results
        overall_results = {
            "test_parameters": {
                "symbols": symbols,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "strategy_a_name": name_a,
                "strategy_b_name": name_b,
            },
            "asset_results": asset_results,
            "overall_stats": {
                "total_assets": len(symbols),
                "strategy_b_wins": wins,
                "strategy_b_win_rate": win_rate,
                "overall_winner": overall_winner,
            },
        }

        # Save overall results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{self.output_dir}/multi_asset_test_{name_a}_vs_{name_b}_{timestamp}.json"
        )

        with open(filename, "w") as f:
            f.write(json.dumps(overall_results, default=str, indent=2))

        self.logger.info(f"Multi-asset test results saved to {filename}")

        return overall_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="A/B Test Trading Strategies")

    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading symbol to test"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d", help="Timeframe for data"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2022-12-31",
        help="End date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--strategy-a", type=str, required=True, help="Name of strategy A"
    )
    parser.add_argument(
        "--strategy-b", type=str, required=True, help="Name of strategy B"
    )
    parser.add_argument(
        "--config-a", type=str, required=True, help="Path to config file for strategy A"
    )
    parser.add_argument(
        "--config-b", type=str, required=True, help="Path to config file for strategy B"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ab_test_results",
        help="Directory to save test results",
    )
    parser.add_argument(
        "--multi-period",
        action="store_true",
        help="Run test across multiple time periods",
    )
    parser.add_argument(
        "--multi-asset", action="store_true", help="Run test across multiple assets"
    )
    parser.add_argument(
        "--symbols", type=str, nargs="+", help="List of symbols for multi-asset test"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Load configurations
    with open(args.config_a, "r") as f:
        config_a = json.load(f)

    with open(args.config_b, "r") as f:
        config_b = json.load(f)

    # Create A/B tester
    ab_tester = ABTester(output_dir=args.output_dir)

    if args.multi_period:
        # Define periods (1-year periods with 3-month overlap)
        period_length = timedelta(days=365)
        overlap = timedelta(days=90)

        periods = []
        period_start = start_date

        while period_start + period_length <= end_date:
            period_end = period_start + period_length
            periods.append((period_start, period_end))
            period_start = period_end - overlap

        # Run multi-period test
        ab_tester.run_multi_period_test(
            symbol=args.symbol,
            timeframe=args.timeframe,
            periods=periods,
            config_a=config_a,
            config_b=config_b,
            name_a=args.strategy_a,
            name_b=args.strategy_b,
        )

    elif args.multi_asset:
        # Run multi-asset test
        symbols = (
            args.symbols
            if args.symbols
            else ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
        )

        ab_tester.run_multi_asset_test(
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            config_a=config_a,
            config_b=config_b,
            name_a=args.strategy_a,
            name_b=args.strategy_b,
        )

    else:
        # Run single test
        ab_tester.run_test(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            config_a=config_a,
            config_b=config_b,
            name_a=args.strategy_a,
            name_b=args.strategy_b,
        )


if __name__ == "__main__":
    main()
