"""
Visualization module for backtesting results

This module provides visualization tools for analyzing backtesting results,
including equity curves, drawdown charts, trade distributions, and more.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

# Set Seaborn style
sns.set(style="darkgrid")


class BacktestVisualizer:
    """
    Visualizer for backtesting results

    Features:
    - Equity curve visualization
    - Drawdown analysis
    - Trade distribution charts
    - Performance metrics visualization
    - Strategy comparison charts
    """

    def __init__(self, config: Dict):
        """
        Initialize the backtest visualizer

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Set output directory
        output_dir = config.get("backtesting", {}).get("output_dir", "backtest_results")
        self.output_dir = Path(output_dir) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set figure size and DPI
        self.figsize = (12, 8)
        self.dpi = 100

        # Set color palette
        self.colors = sns.color_palette("viridis", 10)

        self.logger.info("Backtest visualizer initialized")

    def plot_equity_curve(self, results: Dict, save: bool = True) -> plt.Figure:
        """
        Plot equity curve

        Args:
            results: Backtest results dictionary
            save: Whether to save the plot to file

        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting equity curve")

        # Extract equity curve data
        equity_curve = results.get("equity_curve", [])

        if not equity_curve:
            self.logger.warning("No equity curve data found")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(equity_curve)

        # Check if timestamp column exists
        if "timestamp" not in df.columns:
            self.logger.warning("No timestamp column in equity curve data")
            return None

        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot equity curve
        ax.plot(
            df["datetime"],
            df["equity"],
            linewidth=2,
            color=self.colors[0],
            label="Equity",
        )

        # Plot initial capital
        initial_capital = results.get("metrics", {}).get("initial_capital", 0)
        ax.axhline(
            y=initial_capital,
            linestyle="--",
            color="gray",
            alpha=0.7,
            label="Initial Capital",
        )

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Equity Curve")

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save figure
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"equity_curve_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            self.logger.info(f"Saved equity curve plot to {filepath}")

        return fig

    def plot_drawdown(self, results: Dict, save: bool = True) -> plt.Figure:
        """
        Plot drawdown chart

        Args:
            results: Backtest results dictionary
            save: Whether to save the plot to file

        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting drawdown chart")

        # Extract equity curve data
        equity_curve = results.get("equity_curve", [])

        if not equity_curve:
            self.logger.warning("No equity curve data found")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(equity_curve)

        # Check if timestamp and equity columns exist
        if "timestamp" not in df.columns or "equity" not in df.columns:
            self.logger.warning("Missing required columns in equity curve data")
            return None

        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Calculate drawdown
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot drawdown
        ax.fill_between(
            df["datetime"], df["drawdown"], 0, color=self.colors[1], alpha=0.3
        )
        ax.plot(df["datetime"], df["drawdown"], linewidth=1, color=self.colors[1])

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown Chart")

        # Invert y-axis
        ax.invert_yaxis()

        # Add grid
        ax.grid(True, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save figure
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawdown_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            self.logger.info(f"Saved drawdown plot to {filepath}")

        return fig

    def plot_trade_distribution(self, results: Dict, save: bool = True) -> plt.Figure:
        """
        Plot trade distribution

        Args:
            results: Backtest results dictionary
            save: Whether to save the plot to file

        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting trade distribution")

        # Extract trades data
        trades = results.get("trades", [])

        if not trades:
            self.logger.warning("No trades data found")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Check if profit_percent column exists
        if "profit_percent" not in df.columns:
            self.logger.warning("No profit_percent column in trades data")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot histogram
        sns.histplot(
            df["profit_percent"], bins=50, kde=True, ax=ax, color=self.colors[2]
        )

        # Add vertical line at zero
        ax.axvline(x=0, linestyle="--", color="gray", alpha=0.7)

        # Set labels and title
        ax.set_xlabel("Profit (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Trade Profit Distribution")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save figure
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_distribution_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            self.logger.info(f"Saved trade distribution plot to {filepath}")

        return fig

    def plot_monthly_returns(self, results: Dict, save: bool = True) -> plt.Figure:
        """
        Plot monthly returns heatmap

        Args:
            results: Backtest results dictionary
            save: Whether to save the plot to file

        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting monthly returns heatmap")

        # Extract equity curve data
        equity_curve = results.get("equity_curve", [])

        if not equity_curve:
            self.logger.warning("No equity curve data found")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(equity_curve)

        # Check if timestamp and equity columns exist
        if "timestamp" not in df.columns or "equity" not in df.columns:
            self.logger.warning("Missing required columns in equity curve data")
            return None

        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Set datetime as index
        df.set_index("datetime", inplace=True)

        # Calculate daily returns
        df["daily_return"] = df["equity"].pct_change()

        # Calculate monthly returns
        monthly_returns = (
            df["daily_return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        )

        # Create monthly returns matrix
        monthly_returns_matrix = (
            monthly_returns.groupby(
                [monthly_returns.index.year, monthly_returns.index.month]
            )
            .first()
            .unstack()
        )

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot heatmap
        sns.heatmap(
            monthly_returns_matrix,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            center=0,
            ax=ax,
        )

        # Set labels and title
        ax.set_title("Monthly Returns (%)")
        ax.set_ylabel("Year")
        ax.set_xlabel("Month")

        # Tight layout
        plt.tight_layout()

        # Save figure
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monthly_returns_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            self.logger.info(f"Saved monthly returns plot to {filepath}")

        return fig

    def plot_performance_metrics(self, results: Dict, save: bool = True) -> plt.Figure:
        """
        Plot performance metrics

        Args:
            results: Backtest results dictionary
            save: Whether to save the plot to file

        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting performance metrics")

        # Extract metrics
        metrics = results.get("metrics", {})

        if not metrics:
            self.logger.warning("No metrics data found")
            return None

        # Select metrics to display
        selected_metrics = [
            "return_percent",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
        ]

        # Filter metrics
        filtered_metrics = {k: v for k, v in metrics.items() if k in selected_metrics}

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot bar chart
        bars = ax.bar(
            filtered_metrics.keys(), filtered_metrics.values(), color=self.colors
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        # Set labels and title
        ax.set_title("Performance Metrics")
        ax.set_ylabel("Value")

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Add grid
        ax.grid(True, alpha=0.3, axis="y")

        # Tight layout
        plt.tight_layout()

        # Save figure
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            self.logger.info(f"Saved performance metrics plot to {filepath}")

        return fig

    def plot_trade_analysis(self, results: Dict, save: bool = True) -> plt.Figure:
        """
        Plot trade analysis charts

        Args:
            results: Backtest results dictionary
            save: Whether to save the plot to file

        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting trade analysis")

        # Extract trades data
        trades = results.get("trades", [])

        if not trades:
            self.logger.warning("No trades data found")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Check if required columns exist
        required_columns = ["timestamp", "side", "profit_percent"]
        if not all(col in df.columns for col in required_columns):
            self.logger.warning("Missing required columns in trades data")
            return None

        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)

        # Plot 1: Profit by trade
        axes[0, 0].scatter(
            range(len(df)), df["profit_percent"], alpha=0.6, color=self.colors[0]
        )
        axes[0, 0].axhline(y=0, linestyle="--", color="gray", alpha=0.7)
        axes[0, 0].set_title("Profit by Trade")
        axes[0, 0].set_xlabel("Trade #")
        axes[0, 0].set_ylabel("Profit (%)")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cumulative profit
        df["cumulative_profit"] = df["profit_percent"].cumsum()
        axes[0, 1].plot(
            range(len(df)), df["cumulative_profit"], linewidth=2, color=self.colors[1]
        )
        axes[0, 1].set_title("Cumulative Profit")
        axes[0, 1].set_xlabel("Trade #")
        axes[0, 1].set_ylabel("Cumulative Profit (%)")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Profit by side
        profit_by_side = df.groupby("side")["profit_percent"].mean()
        axes[1, 0].bar(
            profit_by_side.index, profit_by_side.values, color=self.colors[2:4]
        )
        axes[1, 0].set_title("Average Profit by Side")
        axes[1, 0].set_xlabel("Side")
        axes[1, 0].set_ylabel("Average Profit (%)")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # Plot 4: Trade duration histogram
        if "duration" in df.columns:
            sns.histplot(
                df["duration"] / 60,
                bins=30,
                kde=True,
                ax=axes[1, 1],
                color=self.colors[4],
            )
            axes[1, 1].set_title("Trade Duration Distribution")
            axes[1, 1].set_xlabel("Duration (minutes)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5, 0.5, "No duration data available", ha="center", va="center"
            )
            axes[1, 1].set_title("Trade Duration Distribution")

        # Tight layout
        plt.tight_layout()

        # Save figure
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_analysis_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            self.logger.info(f"Saved trade analysis plot to {filepath}")

        return fig

    def plot_strategy_comparison(
        self, results_list: List[Dict], strategy_names: List[str], save: bool = True
    ) -> plt.Figure:
        """
        Plot strategy comparison

        Args:
            results_list: List of backtest results dictionaries
            strategy_names: List of strategy names
            save: Whether to save the plot to file

        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting strategy comparison")

        if not results_list or not strategy_names:
            self.logger.warning("No results or strategy names provided")
            return None

        if len(results_list) != len(strategy_names):
            self.logger.warning("Number of results and strategy names do not match")
            return None

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)

        # Plot 1: Equity curves
        for i, (results, name) in enumerate(zip(results_list, strategy_names)):
            equity_curve = results.get("equity_curve", [])

            if not equity_curve:
                continue

            df = pd.DataFrame(equity_curve)

            if "timestamp" not in df.columns or "equity" not in df.columns:
                continue

            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Normalize equity to start at 100
            initial_equity = df["equity"].iloc[0]
            df["normalized_equity"] = df["equity"] / initial_equity * 100

            axes[0, 0].plot(
                df["datetime"],
                df["normalized_equity"],
                linewidth=2,
                label=name,
                color=self.colors[i % len(self.colors)],
            )

        axes[0, 0].set_title("Normalized Equity Curves")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Normalized Equity")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axes[0, 0].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Drawdowns
        for i, (results, name) in enumerate(zip(results_list, strategy_names)):
            equity_curve = results.get("equity_curve", [])

            if not equity_curve:
                continue

            df = pd.DataFrame(equity_curve)

            if "timestamp" not in df.columns or "equity" not in df.columns:
                continue

            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["peak"] = df["equity"].cummax()
            df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100

            axes[0, 1].plot(
                df["datetime"],
                df["drawdown"],
                linewidth=2,
                label=name,
                color=self.colors[i % len(self.colors)],
            )

        axes[0, 1].set_title("Drawdown Comparison")
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Drawdown (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_yaxis()
        axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axes[0, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

        # Plot 3: Performance metrics comparison
        metrics_to_compare = [
            "return_percent",
            "sharpe_ratio",
            "max_drawdown_percent",
            "win_rate",
        ]

        metrics_data = []

        for results, name in zip(results_list, strategy_names):
            metrics = results.get("metrics", {})

            if not metrics:
                continue

            row = {"Strategy": name}

            for metric in metrics_to_compare:
                row[metric] = metrics.get(metric, 0)

            metrics_data.append(row)

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.set_index("Strategy", inplace=True)

            df_metrics.plot(
                kind="bar", ax=axes[1, 0], color=self.colors[: len(metrics_to_compare)]
            )
            axes[1, 0].set_title("Performance Metrics Comparison")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend(loc="best")
            axes[1, 0].grid(True, alpha=0.3, axis="y")
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1, 0].text(
                0.5, 0.5, "No metrics data available", ha="center", va="center"
            )
            axes[1, 0].set_title("Performance Metrics Comparison")

        # Plot 4: Trade count and win rate
        trade_data = []

        for results, name in zip(results_list, strategy_names):
            metrics = results.get("metrics", {})

            if not metrics:
                continue

            trade_data.append(
                {
                    "Strategy": name,
                    "Total Trades": metrics.get("total_trades", 0),
                    "Win Rate (%)": metrics.get("win_rate", 0),
                }
            )

        if trade_data:
            df_trades = pd.DataFrame(trade_data)

            ax1 = axes[1, 1]
            ax2 = ax1.twinx()

            bars1 = ax1.bar(
                df_trades["Strategy"],
                df_trades["Total Trades"],
                color=self.colors[0],
                alpha=0.7,
                label="Total Trades",
            )
            bars2 = ax2.bar(
                df_trades["Strategy"],
                df_trades["Win Rate (%)"],
                color=self.colors[1],
                alpha=0.7,
                label="Win Rate (%)",
                width=0.5,
            )

            ax1.set_title("Trade Count and Win Rate")
            ax1.set_ylabel("Total Trades")
            ax2.set_ylabel("Win Rate (%)")

            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            ax1.grid(True, alpha=0.3, axis="y")
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1, 1].text(
                0.5, 0.5, "No trade data available", ha="center", va="center"
            )
            axes[1, 1].set_title("Trade Count and Win Rate")

        # Tight layout
        plt.tight_layout()

        # Save figure
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_comparison_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            self.logger.info(f"Saved strategy comparison plot to {filepath}")

        return fig

    def create_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Create HTML report with all visualizations

        Args:
            results: Backtest results dictionary
            output_path: Output path for the report

        Returns:
            Path to the generated report
        """
        self.logger.info("Creating HTML report")

        # Generate plots
        equity_fig = self.plot_equity_curve(results, save=False)
        drawdown_fig = self.plot_drawdown(results, save=False)
        trade_dist_fig = self.plot_trade_distribution(results, save=False)
        monthly_returns_fig = self.plot_monthly_returns(results, save=False)
        metrics_fig = self.plot_performance_metrics(results, save=False)
        trade_analysis_fig = self.plot_trade_analysis(results, save=False)

        # Create output path
        if output_path:
            report_path = Path(output_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"backtest_report_{timestamp}.html"

        # Create report directory
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Save plots to temporary files
        plot_paths = []

        for i, fig in enumerate(
            [
                equity_fig,
                drawdown_fig,
                trade_dist_fig,
                monthly_returns_fig,
                metrics_fig,
                trade_analysis_fig,
            ]
        ):
            if fig:
                plot_path = self.output_dir / f"temp_plot_{i}.png"
                fig.savefig(plot_path)
                plot_paths.append(plot_path)
                plt.close(fig)

        # Extract metrics
        metrics = results.get("metrics", {})

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: #f2f2f2;
                }}
                .plot-container {{
                    margin-bottom: 30px;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Backtest Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Performance Metrics</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """

        # Add metrics to HTML
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                formatted_value = (
                    f"{metric_value:.2f}"
                    if isinstance(metric_value, float)
                    else f"{metric_value}"
                )
                html_content += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{formatted_value}</td>
                    </tr>
                """

        html_content += """
                </table>
                
                <h2>Visualizations</h2>
        """

        # Add plots to HTML
        plot_titles = [
            "Equity Curve",
            "Drawdown Chart",
            "Trade Profit Distribution",
            "Monthly Returns",
            "Performance Metrics",
            "Trade Analysis",
        ]

        for i, plot_path in enumerate(plot_paths):
            if i < len(plot_titles):
                title = plot_titles[i]
            else:
                title = f"Plot {i+1}"

            html_content += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <img src="data:image/png;base64,{self._encode_image(plot_path)}" alt="{title}">
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        # Write HTML to file
        with open(report_path, "w") as f:
            f.write(html_content)

        # Remove temporary plot files
        for plot_path in plot_paths:
            try:
                os.remove(plot_path)
            except Exception as e:
                self.logger.warning(f"Error removing temporary file {plot_path}: {e}")

        self.logger.info(f"Created HTML report at {report_path}")

        return str(report_path)

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image as base64 string

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        import base64

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
