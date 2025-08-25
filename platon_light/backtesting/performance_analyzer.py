"""
Performance analyzer for backtesting results

This module provides functionality to analyze and visualize backtest results.
It calculates performance metrics, generates visualizations, and creates reports
to help evaluate trading strategy performance.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path


class PerformanceAnalyzer:
    """
    Performance analyzer for backtesting results

    Features:
    - Calculate performance metrics
    - Generate visualizations
    - Create performance reports
    - Compare multiple strategies
    """

    def __init__(self, config: Dict):
        """
        Initialize the performance analyzer

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Set output directory
        output_dir = config.get("backtesting", {}).get("output_dir", "backtest_results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Performance analyzer initialized")

    def analyze(self, results: Dict, save_plots: bool = True) -> Dict:
        """
        Analyze backtest results

        Args:
            results: Backtest results dictionary
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Analyzing backtest results")

        # Extract data
        trades = results.get("trades", [])
        equity_curve = results.get("equity_curve", [])
        metrics = results.get("metrics", {})

        if not trades or not equity_curve:
            self.logger.warning("No trades or equity data to analyze")
            return {"error": "No data to analyze"}

        # Convert to DataFrames
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)

        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(trades_df, equity_df)
        metrics.update(additional_metrics)

        # Generate plots if requested
        if save_plots:
            self._generate_plots(trades_df, equity_df, metrics)

        # Return updated results
        results["metrics"] = metrics

        return results

    def _calculate_additional_metrics(
        self, trades_df: pd.DataFrame, equity_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate additional performance metrics

        Args:
            trades_df: DataFrame with trade data
            equity_df: DataFrame with equity curve data

        Returns:
            Dictionary with additional metrics
        """
        metrics = {}

        # Skip if no trades
        if trades_df.empty or equity_df.empty:
            return metrics

        try:
            # Calculate trade metrics
            if not trades_df.empty:
                # Convert datetime columns if they're strings
                for col in ["entry_time", "exit_time"]:
                    if col in trades_df.columns and trades_df[col].dtype == "object":
                        trades_df[col] = pd.to_datetime(trades_df[col])

                # Calculate trade durations
                if (
                    "entry_time" in trades_df.columns
                    and "exit_time" in trades_df.columns
                ):
                    trades_df["duration"] = (
                        trades_df["exit_time"] - trades_df["entry_time"]
                    ).dt.total_seconds() / 60  # minutes

                    metrics["avg_trade_duration"] = trades_df["duration"].mean()
                    metrics["max_trade_duration"] = trades_df["duration"].max()
                    metrics["min_trade_duration"] = trades_df["duration"].min()

                # Calculate consecutive wins/losses
                if "pnl" in trades_df.columns:
                    trades_df["is_win"] = trades_df["pnl"] > 0

                    # Calculate streaks
                    trades_df["streak_group"] = (
                        trades_df["is_win"] != trades_df["is_win"].shift()
                    ).cumsum()
                    streak_counts = trades_df.groupby(["streak_group", "is_win"]).size()

                    # Get max consecutive wins/losses
                    max_wins = (
                        streak_counts[
                            streak_counts.index.get_level_values("is_win")
                        ].max()
                        if True in streak_counts.index.get_level_values("is_win")
                        else 0
                    )
                    max_losses = (
                        streak_counts[
                            ~streak_counts.index.get_level_values("is_win")
                        ].max()
                        if False in streak_counts.index.get_level_values("is_win")
                        else 0
                    )

                    metrics["max_consecutive_wins"] = max_wins
                    metrics["max_consecutive_losses"] = max_losses

                # Calculate monthly returns
                if "exit_time" in trades_df.columns and "pnl" in trades_df.columns:
                    trades_df["month"] = trades_df["exit_time"].dt.to_period("M")
                    monthly_returns = trades_df.groupby("month")["pnl"].sum()

                    metrics["monthly_returns"] = monthly_returns.to_dict()
                    metrics["best_month"] = monthly_returns.max()
                    metrics["worst_month"] = monthly_returns.min()
                    metrics["profitable_months"] = (monthly_returns > 0).sum()
                    metrics["losing_months"] = (monthly_returns <= 0).sum()

                # Calculate per-symbol performance
                if "symbol" in trades_df.columns and "pnl" in trades_df.columns:
                    symbol_performance = trades_df.groupby("symbol").agg(
                        {"pnl": ["sum", "mean", "count"], "is_win": "mean"}  # Win rate
                    )

                    # Flatten multi-index columns
                    symbol_performance.columns = [
                        "_".join(col).strip()
                        for col in symbol_performance.columns.values
                    ]

                    # Rename columns
                    symbol_performance = symbol_performance.rename(
                        columns={
                            "pnl_sum": "total_pnl",
                            "pnl_mean": "avg_pnl",
                            "pnl_count": "trade_count",
                            "is_win_mean": "win_rate",
                        }
                    )

                    # Convert win rate to percentage
                    symbol_performance["win_rate"] = (
                        symbol_performance["win_rate"] * 100
                    )

                    metrics["symbol_performance"] = symbol_performance.to_dict(
                        orient="index"
                    )

            # Calculate equity curve metrics
            if not equity_df.empty:
                # Convert datetime if it's a string
                if (
                    "datetime" in equity_df.columns
                    and equity_df["datetime"].dtype == "object"
                ):
                    equity_df["datetime"] = pd.to_datetime(equity_df["datetime"])

                # Calculate daily returns
                if "datetime" in equity_df.columns and "equity" in equity_df.columns:
                    equity_df["date"] = equity_df["datetime"].dt.date
                    daily_equity = (
                        equity_df.groupby("date")["equity"].last().reset_index()
                    )
                    daily_equity["return"] = daily_equity["equity"].pct_change()

                    # Calculate volatility (annualized)
                    metrics["daily_volatility"] = daily_equity["return"].std()
                    metrics["annualized_volatility"] = metrics[
                        "daily_volatility"
                    ] * np.sqrt(252)

                    # Calculate Sharpe and Sortino ratios
                    risk_free_rate = self.config.get("backtesting", {}).get(
                        "risk_free_rate", 0.0
                    )

                    avg_daily_return = daily_equity["return"].mean()
                    excess_return = avg_daily_return - (risk_free_rate / 252)

                    if metrics["daily_volatility"] > 0:
                        metrics["sharpe_ratio"] = (
                            excess_return / metrics["daily_volatility"]
                        ) * np.sqrt(252)
                    else:
                        metrics["sharpe_ratio"] = 0

                    # Sortino ratio (downside deviation)
                    negative_returns = daily_equity["return"][
                        daily_equity["return"] < 0
                    ]
                    downside_deviation = negative_returns.std()

                    if downside_deviation > 0:
                        metrics["sortino_ratio"] = (
                            excess_return / downside_deviation
                        ) * np.sqrt(252)
                    else:
                        metrics["sortino_ratio"] = 0

                    # Calculate Calmar ratio
                    if (
                        "max_drawdown_percent" in metrics
                        and metrics["max_drawdown_percent"] > 0
                    ):
                        annualized_return = avg_daily_return * 252
                        metrics["calmar_ratio"] = annualized_return / (
                            metrics["max_drawdown_percent"] / 100
                        )
                    else:
                        metrics["calmar_ratio"] = 0

                    # Calculate underwater periods
                    equity_df["peak"] = equity_df["equity"].cummax()
                    equity_df["drawdown"] = (
                        (equity_df["equity"] - equity_df["peak"])
                        / equity_df["peak"]
                        * 100
                    )
                    equity_df["is_underwater"] = equity_df["drawdown"] < 0

                    # Calculate underwater periods
                    equity_df["underwater_group"] = (
                        equity_df["is_underwater"] != equity_df["is_underwater"].shift()
                    ).cumsum()
                    underwater_periods = equity_df[equity_df["is_underwater"]].groupby(
                        "underwater_group"
                    )

                    if not underwater_periods.empty:
                        underwater_durations = underwater_periods.apply(
                            lambda x: (
                                x["datetime"].max() - x["datetime"].min()
                            ).total_seconds()
                            / (60 * 60 * 24)
                        )  # days

                        metrics["avg_underwater_duration"] = underwater_durations.mean()
                        metrics["max_underwater_duration"] = underwater_durations.max()
                        metrics["underwater_periods"] = len(underwater_durations)

        except Exception as e:
            self.logger.error(f"Error calculating additional metrics: {e}")

        return metrics

    def _generate_plots(
        self, trades_df: pd.DataFrame, equity_df: pd.DataFrame, metrics: Dict
    ):
        """
        Generate performance plots

        Args:
            trades_df: DataFrame with trade data
            equity_df: DataFrame with equity curve data
            metrics: Performance metrics
        """
        try:
            # Skip if no data
            if trades_df.empty or equity_df.empty:
                self.logger.warning("No data to generate plots")
                return

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Plot equity curve
            self._plot_equity_curve(equity_df, timestamp)

            # Plot drawdown
            self._plot_drawdown(equity_df, timestamp)

            # Plot trade outcomes
            self._plot_trade_outcomes(trades_df, timestamp)

            # Plot monthly returns
            if "monthly_returns" in metrics:
                self._plot_monthly_returns(metrics["monthly_returns"], timestamp)

            # Plot symbol performance
            if "symbol_performance" in metrics:
                self._plot_symbol_performance(metrics["symbol_performance"], timestamp)

            self.logger.info("Performance plots generated")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")

    def _plot_equity_curve(self, equity_df: pd.DataFrame, timestamp: str):
        """
        Plot equity curve

        Args:
            equity_df: DataFrame with equity curve data
            timestamp: Timestamp for filename
        """
        try:
            plt.figure(figsize=(12, 6))

            # Convert datetime if it's a string
            if (
                "datetime" in equity_df.columns
                and equity_df["datetime"].dtype == "object"
            ):
                equity_df["datetime"] = pd.to_datetime(equity_df["datetime"])

            # Plot equity curve
            plt.plot(
                equity_df["datetime"], equity_df["equity"], label="Equity", color="blue"
            )

            # Add position markers if available
            if "position" in equity_df.columns:
                # Long entries
                long_entries = equity_df[
                    (equity_df["position"] == "long")
                    & (equity_df["position"].shift() != "long")
                ]
                if not long_entries.empty:
                    plt.scatter(
                        long_entries["datetime"],
                        long_entries["equity"],
                        color="green",
                        marker="^",
                        s=50,
                        label="Long Entry",
                    )

                # Short entries
                short_entries = equity_df[
                    (equity_df["position"] == "short")
                    & (equity_df["position"].shift() != "short")
                ]
                if not short_entries.empty:
                    plt.scatter(
                        short_entries["datetime"],
                        short_entries["equity"],
                        color="red",
                        marker="v",
                        s=50,
                        label="Short Entry",
                    )

                # Position exits
                exits = equity_df[
                    (equity_df["position"].isnull())
                    & (equity_df["position"].shift().notnull())
                ]
                if not exits.empty:
                    plt.scatter(
                        exits["datetime"],
                        exits["equity"],
                        color="black",
                        marker="x",
                        s=50,
                        label="Exit",
                    )

            # Format plot
            plt.title("Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Equity")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / f"equity_curve_{timestamp}.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")

    def _plot_drawdown(self, equity_df: pd.DataFrame, timestamp: str):
        """
        Plot drawdown

        Args:
            equity_df: DataFrame with equity curve data
            timestamp: Timestamp for filename
        """
        try:
            # Calculate drawdown
            equity_df["peak"] = equity_df["equity"].cummax()
            equity_df["drawdown"] = (
                (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
            )

            plt.figure(figsize=(12, 6))

            # Convert datetime if it's a string
            if (
                "datetime" in equity_df.columns
                and equity_df["datetime"].dtype == "object"
            ):
                equity_df["datetime"] = pd.to_datetime(equity_df["datetime"])

            # Plot drawdown
            plt.fill_between(
                equity_df["datetime"], equity_df["drawdown"], 0, color="red", alpha=0.3
            )
            plt.plot(
                equity_df["datetime"],
                equity_df["drawdown"],
                color="red",
                label="Drawdown",
            )

            # Format plot
            plt.title("Drawdown")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / f"drawdown_{timestamp}.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting drawdown: {e}")

    def _plot_trade_outcomes(self, trades_df: pd.DataFrame, timestamp: str):
        """
        Plot trade outcomes

        Args:
            trades_df: DataFrame with trade data
            timestamp: Timestamp for filename
        """
        try:
            plt.figure(figsize=(12, 6))

            # Plot trade PnL
            if "pnl" in trades_df.columns:
                colors = ["green" if pnl > 0 else "red" for pnl in trades_df["pnl"]]
                plt.bar(range(len(trades_df)), trades_df["pnl"], color=colors)

                # Add horizontal line at zero
                plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

                # Format plot
                plt.title("Trade Outcomes")
                plt.xlabel("Trade Number")
                plt.ylabel("Profit/Loss")
                plt.grid(True, alpha=0.3)

                # Save plot
                plt.tight_layout()
                plt.savefig(self.output_dir / f"trade_outcomes_{timestamp}.png")
                plt.close()

            # Plot trade PnL distribution
            plt.figure(figsize=(10, 6))
            plt.hist(trades_df["pnl"], bins=50, alpha=0.7, color="blue")
            plt.axvline(x=0, color="red", linestyle="--")
            plt.title("Trade PnL Distribution")
            plt.xlabel("Profit/Loss")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / f"pnl_distribution_{timestamp}.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting trade outcomes: {e}")

    def _plot_monthly_returns(self, monthly_returns: Dict, timestamp: str):
        """
        Plot monthly returns

        Args:
            monthly_returns: Dictionary with monthly returns
            timestamp: Timestamp for filename
        """
        try:
            # Convert to Series
            monthly_series = pd.Series(monthly_returns)

            # Sort by date
            monthly_series = monthly_series.sort_index()

            plt.figure(figsize=(12, 6))

            # Plot monthly returns
            colors = ["green" if ret > 0 else "red" for ret in monthly_series]
            plt.bar(range(len(monthly_series)), monthly_series, color=colors)

            # Add horizontal line at zero
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            # Format plot
            plt.title("Monthly Returns")
            plt.xlabel("Month")
            plt.ylabel("Return")
            plt.grid(True, alpha=0.3)

            # Format x-axis labels
            plt.xticks(
                range(len(monthly_series)),
                [str(idx) for idx in monthly_series.index],
                rotation=45,
            )

            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / f"monthly_returns_{timestamp}.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting monthly returns: {e}")

    def _plot_symbol_performance(self, symbol_performance: Dict, timestamp: str):
        """
        Plot symbol performance

        Args:
            symbol_performance: Dictionary with symbol performance
            timestamp: Timestamp for filename
        """
        try:
            # Convert to DataFrame
            symbols_df = pd.DataFrame.from_dict(symbol_performance, orient="index")

            # Sort by total PnL
            symbols_df = symbols_df.sort_values("total_pnl", ascending=False)

            # Plot total PnL by symbol
            plt.figure(figsize=(12, 6))

            colors = ["green" if pnl > 0 else "red" for pnl in symbols_df["total_pnl"]]
            plt.bar(range(len(symbols_df)), symbols_df["total_pnl"], color=colors)

            # Add horizontal line at zero
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            # Format plot
            plt.title("Total PnL by Symbol")
            plt.xlabel("Symbol")
            plt.ylabel("Total PnL")
            plt.grid(True, alpha=0.3)

            # Format x-axis labels
            plt.xticks(range(len(symbols_df)), symbols_df.index, rotation=45)

            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / f"symbol_pnl_{timestamp}.png")
            plt.close()

            # Plot win rate by symbol
            plt.figure(figsize=(12, 6))

            plt.bar(range(len(symbols_df)), symbols_df["win_rate"], color="blue")

            # Add horizontal line at 50%
            plt.axhline(y=50, color="red", linestyle="--", alpha=0.7)

            # Format plot
            plt.title("Win Rate by Symbol")
            plt.xlabel("Symbol")
            plt.ylabel("Win Rate (%)")
            plt.grid(True, alpha=0.3)

            # Format x-axis labels
            plt.xticks(range(len(symbols_df)), symbols_df.index, rotation=45)

            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / f"symbol_win_rate_{timestamp}.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting symbol performance: {e}")

    def compare_strategies(self, results_list: List[Dict], names: List[str]) -> Dict:
        """
        Compare multiple strategies

        Args:
            results_list: List of backtest results dictionaries
            names: List of strategy names

        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info("Comparing strategies")

            if len(results_list) != len(names):
                self.logger.error("Number of results and names must match")
                return {"error": "Number of results and names must match"}

            if len(results_list) < 2:
                self.logger.error("Need at least two strategies to compare")
                return {"error": "Need at least two strategies to compare"}

            # Extract equity curves
            equity_curves = []

            for i, results in enumerate(results_list):
                equity_curve = results.get("equity_curve", [])

                if not equity_curve:
                    self.logger.warning(f"No equity curve for strategy {names[i]}")
                    continue

                equity_df = pd.DataFrame(equity_curve)
                equity_df["strategy"] = names[i]

                equity_curves.append(equity_df)

            if not equity_curves:
                self.logger.error("No equity curves to compare")
                return {"error": "No equity curves to compare"}

            # Combine equity curves
            combined_equity = pd.concat(equity_curves)

            # Generate comparison plots
            self._plot_equity_comparison(combined_equity)

            # Extract metrics
            metrics_comparison = {}

            for i, results in enumerate(results_list):
                metrics = results.get("metrics", {})

                if not metrics:
                    self.logger.warning(f"No metrics for strategy {names[i]}")
                    continue

                metrics_comparison[names[i]] = metrics

            if not metrics_comparison:
                self.logger.error("No metrics to compare")
                return {"error": "No metrics to compare"}

            # Generate metrics comparison
            comparison_table = self._generate_metrics_comparison(metrics_comparison)

            return {
                "equity_comparison": combined_equity.to_dict(orient="records"),
                "metrics_comparison": metrics_comparison,
                "comparison_table": comparison_table,
            }
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}")
            return {"error": f"Error comparing strategies: {e}"}

    def _plot_equity_comparison(self, combined_equity: pd.DataFrame):
        """
        Plot equity comparison

        Args:
            combined_equity: DataFrame with combined equity curves
        """
        try:
            # Convert datetime if it's a string
            if (
                "datetime" in combined_equity.columns
                and combined_equity["datetime"].dtype == "object"
            ):
                combined_equity["datetime"] = pd.to_datetime(
                    combined_equity["datetime"]
                )

            # Plot equity curves
            plt.figure(figsize=(12, 6))

            for strategy, group in combined_equity.groupby("strategy"):
                plt.plot(group["datetime"], group["equity"], label=strategy)

            # Format plot
            plt.title("Equity Curve Comparison")
            plt.xlabel("Date")
            plt.ylabel("Equity")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"equity_comparison_{timestamp}.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting equity comparison: {e}")

    def _generate_metrics_comparison(self, metrics_comparison: Dict) -> pd.DataFrame:
        """
        Generate metrics comparison table

        Args:
            metrics_comparison: Dictionary with metrics for each strategy

        Returns:
            DataFrame with metrics comparison
        """
        try:
            # Select key metrics for comparison
            key_metrics = [
                "total_trades",
                "win_rate",
                "profit_factor",
                "net_profit",
                "return_percent",
                "max_drawdown_percent",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "avg_trade",
                "avg_profit",
                "avg_loss",
            ]

            # Create comparison table
            comparison_data = {}

            for strategy, metrics in metrics_comparison.items():
                comparison_data[strategy] = {
                    metric: metrics.get(metric, None) for metric in key_metrics
                }

            comparison_df = pd.DataFrame.from_dict(comparison_data)

            # Save comparison table
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_df.to_csv(
                self.output_dir / f"metrics_comparison_{timestamp}.csv"
            )

            return comparison_df
        except Exception as e:
            self.logger.error(f"Error generating metrics comparison: {e}")
            return pd.DataFrame()
