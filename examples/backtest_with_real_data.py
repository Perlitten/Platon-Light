#!/usr/bin/env python
"""
Example script demonstrating how to use the Platon Light backtesting module with real market data.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Platon Light modules
from platon_light.backtesting.strategies.moving_average_crossover import (
    MovingAverageCrossover,
)
from launch_live_backtest import SecureCredentialManager, fetch_market_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a backtest with real market data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to backtest")
    parser.add_argument("--timeframe", default="1h", help="Timeframe to backtest")
    parser.add_argument(
        "--start-date",
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument("--fast-period", type=int, default=10, help="Fast MA period")
    parser.add_argument("--slow-period", type=int, default=40, help="Slow MA period")
    parser.add_argument(
        "--fast-ma-type", default="EMA", choices=["SMA", "EMA"], help="Fast MA type"
    )
    parser.add_argument(
        "--slow-ma-type", default="SMA", choices=["SMA", "EMA"], help="Slow MA type"
    )
    parser.add_argument(
        "--use-filters", action="store_true", help="Use RSI and Bollinger Bands filters"
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save results to file"
    )
    parser.add_argument("--plot", action="store_true", help="Plot results")

    return parser.parse_args()


def run_backtest(data, strategy_params):
    """Run a backtest with the specified strategy parameters."""
    logger.info(f"Running backtest with strategy parameters: {strategy_params}")

    # Create strategy instance
    strategy = MovingAverageCrossover(**strategy_params)

    # Run strategy
    result = strategy.run(data)

    logger.info(f"Backtest completed with {len(result[result['signal'] != 0])} signals")

    return result


def calculate_performance_metrics(result_df):
    """Calculate performance metrics for the backtest."""
    # Copy dataframe to avoid modifying the original
    df = result_df.copy()

    # Calculate returns
    df["returns"] = df["close"].pct_change()

    # Calculate strategy returns (only when we have a position)
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]

    # Calculate cumulative returns
    df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1
    df["strategy_cumulative_returns"] = (1 + df["strategy_returns"]).cumprod() - 1

    # Calculate drawdowns
    df["strategy_cumulative_returns_peak"] = df["strategy_cumulative_returns"].cummax()
    df["drawdown"] = (
        df["strategy_cumulative_returns_peak"] - df["strategy_cumulative_returns"]
    )
    df["drawdown_pct"] = df["drawdown"] / (1 + df["strategy_cumulative_returns_peak"])

    # Calculate trade statistics
    trades = df[df["signal"] != 0].copy()
    trades["trade_type"] = trades["signal"].apply(lambda x: "buy" if x > 0 else "sell")

    # Calculate metrics
    total_trades = len(trades)
    winning_trades = len(df[df["trade_returns"] > 0])
    losing_trades = len(df[df["trade_returns"] < 0])

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = (
        df[df["trade_returns"] > 0]["trade_returns"].mean() if winning_trades > 0 else 0
    )
    avg_loss = (
        df[df["trade_returns"] < 0]["trade_returns"].mean() if losing_trades > 0 else 0
    )

    profit_factor = (
        abs(
            df[df["trade_returns"] > 0]["trade_returns"].sum()
            / df[df["trade_returns"] < 0]["trade_returns"].sum()
        )
        if losing_trades > 0
        else float("inf")
    )

    # Calculate annualized metrics
    days = (df.index[-1] - df.index[0]).days
    years = days / 365

    total_return = df["strategy_cumulative_returns"].iloc[-1]
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    max_drawdown = df["drawdown_pct"].max()

    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = (
        (df["strategy_returns"].mean() / df["strategy_returns"].std() * (252**0.5))
        if df["strategy_returns"].std() > 0
        else 0
    )

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

    return metrics, df


def plot_results(df, metrics, symbol, timeframe):
    """Plot backtest results."""
    plt.figure(figsize=(12, 8))

    # Plot price and moving averages
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["close"], label="Close Price")
    plt.plot(
        df.index,
        df["fast_ma"],
        label=f"{df['fast_ma_type'].iloc[0]} {df['fast_period'].iloc[0]}",
    )
    plt.plot(
        df.index,
        df["slow_ma"],
        label=f"{df['slow_ma_type'].iloc[0]} {df['slow_period'].iloc[0]}",
    )

    # Plot buy and sell signals
    plt.plot(
        df[df["signal"] == 1].index,
        df[df["signal"] == 1]["close"],
        "^",
        markersize=10,
        color="g",
        label="Buy Signal",
    )
    plt.plot(
        df[df["signal"] == -1].index,
        df[df["signal"] == -1]["close"],
        "v",
        markersize=10,
        color="r",
        label="Sell Signal",
    )

    plt.title(f"{symbol} {timeframe} - Moving Average Crossover Strategy")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()

    # Plot equity curve
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["strategy_cumulative_returns"], label="Strategy Returns")
    plt.plot(df.index, df["cumulative_returns"], label="Buy and Hold Returns")

    # Add drawdown
    plt.fill_between(
        df.index, 0, -df["drawdown_pct"], color="red", alpha=0.3, label="Drawdown"
    )

    plt.title("Equity Curve and Drawdown")
    plt.ylabel("Returns")
    plt.grid(True)
    plt.legend()

    # Add performance metrics as text
    plt.figtext(
        0.01,
        0.01,
        f"""
    Total Return: {metrics['total_return']:.2%}
    Annualized Return: {metrics['annualized_return']:.2%}
    Max Drawdown: {metrics['max_drawdown']:.2%}
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Win Rate: {metrics['win_rate']:.2%}
    Profit Factor: {metrics['profit_factor']:.2f}
    """,
        fontsize=10,
    )

    plt.tight_layout()

    return plt


def main():
    """Main function."""
    args = parse_arguments()

    logger.info(
        f"Starting backtest for {args.symbol} {args.timeframe} from {args.start_date} to {args.end_date}"
    )

    # Initialize credential manager
    credential_manager = SecureCredentialManager()

    # Load credentials
    credentials = credential_manager.load_credentials()

    if not credentials:
        logger.warning(
            "No credentials found. Run 'python launch_live_backtest.py --setup-credentials' to set up API credentials."
        )

    # Fetch market data
    data = fetch_market_data(
        args.symbol,
        args.timeframe,
        args.start_date,
        args.end_date,
        credentials.get("BINANCE_API_KEY"),
        credentials.get("BINANCE_API_SECRET"),
    )

    if data is None or len(data) == 0:
        logger.error(
            "Failed to fetch market data or no data available for the specified period"
        )
        return

    logger.info(
        f"Fetched {len(data)} data points from {data.index[0]} to {data.index[-1]}"
    )

    # Set up strategy parameters
    strategy_params = {
        "fast_ma_type": args.fast_ma_type,
        "slow_ma_type": args.slow_ma_type,
        "fast_period": args.fast_period,
        "slow_period": args.slow_period,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "use_filters": args.use_filters,
    }

    # Run backtest
    result = run_backtest(data, strategy_params)

    # Calculate performance metrics
    metrics, result_with_metrics = calculate_performance_metrics(result)

    # Log metrics
    logger.info("Performance metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save results if requested
    if args.save_results:
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            output_dir / f"backtest_{args.symbol}_{args.timeframe}_{timestamp}.csv"
        )

        result_with_metrics.to_csv(output_file)
        logger.info(f"Results saved to {output_file}")

    # Plot results if requested
    if args.plot:
        plt_obj = plot_results(
            result_with_metrics, metrics, args.symbol, args.timeframe
        )

        if args.save_results:
            plot_file = (
                output_dir / f"backtest_{args.symbol}_{args.timeframe}_{timestamp}.png"
            )
            plt_obj.savefig(plot_file)
            logger.info(f"Plot saved to {plot_file}")

        plt_obj.show()

    logger.info("Backtest completed")


if __name__ == "__main__":
    main()
