#!/usr/bin/env python
"""
Platon Light Enhanced Trading Dashboard

A real-time trading dashboard with advanced features:
1. Select between real trading and dry run mode
2. Monitor trading activity with detailed metrics and risk indicators
3. View performance metrics and charts
4. Telegram notifications for important events

Usage:
    python enhanced_trading_dashboard.py [--mode {dry_run,real}] [--port PORT]
"""

import os
import sys
import logging
import threading
import time
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import Platon Light modules
try:
    from platon_light.backtesting.strategies.moving_average_crossover import (
        MovingAverageCrossover,
    )
    from platon_light.integrations.telegram_bot import TelegramBot
except ImportError:
    logger.warning("Could not import Platon Light modules. Running in standalone mode.")
    TelegramBot = None

# Global variables
trading_active = False
trading_mode = "dry_run"  # 'dry_run' or 'real'
trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
selected_pair = "BTCUSDT"
timeframe = "1h"
trading_balance = 10000.0  # Starting balance in USDT
trading_history = []
last_update_time = datetime.now()
telegram_notifications_enabled = True
risk_level = "medium"  # 'low', 'medium', 'high'
max_drawdown = 0.0
win_rate = 0.0
profit_factor = 0.0

# Performance metrics
performance_metrics = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "win_rate": 0.0,
    "profit_factor": 0.0,
    "max_drawdown": 0.0,
    "sharpe_ratio": 0.0,
    "daily_return": 0.0,
    "monthly_return": 0.0,
}

# Risk levels and their parameters
risk_levels = {
    "low": {
        "position_size": 0.02,  # 2% of balance per trade
        "stop_loss": 0.02,  # 2% stop loss
        "take_profit": 0.04,  # 4% take profit
    },
    "medium": {
        "position_size": 0.05,  # 5% of balance per trade
        "stop_loss": 0.05,  # 5% stop loss
        "take_profit": 0.1,  # 10% take profit
    },
    "high": {
        "position_size": 0.1,  # 10% of balance per trade
        "stop_loss": 0.1,  # 10% stop loss
        "take_profit": 0.2,  # 20% take profit
    },
}

# Telegram bot instance
telegram_bot = None


# Sample data for demonstration
def generate_sample_data(symbol, days=30):
    """Generate sample price data for demonstration."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    # Generate timestamps (hourly data)
    timestamps = pd.date_range(start=start_time, end=end_time, freq="h")

    # Generate price data with random walk
    np.random.seed(42)  # For reproducibility

    # Initial price
    if symbol == "BTCUSDT":
        initial_price = 50000.0
    elif symbol == "ETHUSDT":
        initial_price = 3000.0
    elif symbol == "ADAUSDT":
        initial_price = 1.2
    elif symbol == "SOLUSDT":
        initial_price = 150.0
    else:
        initial_price = 100.0

    # Random walk parameters
    drift = 0.0001  # Small upward drift
    volatility = 0.01  # Volatility

    # Generate log returns
    log_returns = np.random.normal(drift, volatility, len(timestamps))

    # Calculate price series
    price_series = initial_price * np.exp(np.cumsum(log_returns))

    # Generate OHLCV data
    data = pd.DataFrame(index=timestamps)
    data["close"] = price_series

    # Generate open, high, low based on close
    data["open"] = data["close"].shift(1)
    data.loc[data.index[0], "open"] = data["close"].iloc[0] * (
        1 - np.random.normal(0, volatility)
    )

    # High is the maximum of open and close, plus a random amount
    data["high"] = data[["open", "close"]].max(axis=1) * (
        1 + np.abs(np.random.normal(0, volatility, len(data)))
    )

    # Low is the minimum of open and close, minus a random amount
    data["low"] = data[["open", "close"]].min(axis=1) * (
        1 - np.abs(np.random.normal(0, volatility, len(data)))
    )

    # Generate volume
    data["volume"] = np.random.lognormal(10, 1, len(data)) * (
        1 + 0.1 * np.sin(np.linspace(0, 20 * np.pi, len(data)))
    )

    return data


# Generate sample data for each pair
sample_data = {pair: generate_sample_data(pair) for pair in trading_pairs}

# Add technical indicators to sample data
for pair in trading_pairs:
    # Calculate fast EMA (10-period)
    sample_data[pair]["fast_ma"] = sample_data[pair]["close"].ewm(span=10).mean()

    # Calculate slow SMA (40-period)
    sample_data[pair]["slow_ma"] = sample_data[pair]["close"].rolling(window=40).mean()

    # Calculate RSI (14-period)
    delta = sample_data[pair]["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    sample_data[pair]["rsi"] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands (20-period, 2 standard deviations)
    sma = sample_data[pair]["close"].rolling(window=20).mean()
    std = sample_data[pair]["close"].rolling(window=20).std()
    sample_data[pair]["bb_upper"] = sma + 2 * std
    sample_data[pair]["bb_middle"] = sma
    sample_data[pair]["bb_lower"] = sma - 2 * std

    # Generate signals
    sample_data[pair]["signal"] = 0

    # Buy signal when fast MA crosses above slow MA and RSI > 30
    sample_data[pair].loc[
        (sample_data[pair]["fast_ma"] > sample_data[pair]["slow_ma"])
        & (
            sample_data[pair]["fast_ma"].shift(1)
            <= sample_data[pair]["slow_ma"].shift(1)
        )
        & (sample_data[pair]["rsi"] > 30),
        "signal",
    ] = 1

    # Sell signal when fast MA crosses below slow MA or RSI > 70
    sample_data[pair].loc[
        (sample_data[pair]["fast_ma"] < sample_data[pair]["slow_ma"])
        & (
            sample_data[pair]["fast_ma"].shift(1)
            >= sample_data[pair]["slow_ma"].shift(1)
        )
        | (sample_data[pair]["rsi"] > 70),
        "signal",
    ] = -1


# Enhanced Trading Engine with risk management
class EnhancedTradingEngine:
    """Enhanced trading engine with risk management and performance tracking."""

    def __init__(self, mode="dry_run", risk_level="medium"):
        """Initialize the trading engine."""
        self.mode = mode
        self.risk_level = risk_level
        self.running = False
        self.thread = None
        self.positions = {}  # Current open positions
        self.performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "peak_balance": trading_balance,
            "daily_profits": [],
        }

    def start(self):
        """Start the trading engine."""
        if self.running:
            logger.warning("Trading engine is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_trading_loop)
        self.thread.daemon = True
        self.thread.start()

        logger.info(
            f"Trading engine started in {self.mode} mode with {self.risk_level} risk level"
        )

        # Send notification via Telegram if enabled
        if telegram_bot and telegram_notifications_enabled:
            asyncio.run(
                self._send_telegram_notification(
                    f"🚀 Trading bot started in {self.mode.upper()} mode with {self.risk_level.upper()} risk level"
                )
            )

    def stop(self):
        """Stop the trading engine."""
        if not self.running:
            logger.warning("Trading engine is not running")
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

        logger.info("Trading engine stopped")

        # Send notification via Telegram if enabled
        if telegram_bot and telegram_notifications_enabled:
            asyncio.run(self._send_telegram_notification("🛑 Trading bot stopped"))

    async def _send_telegram_notification(self, message):
        """Send a notification via Telegram."""
        try:
            if telegram_bot:
                await telegram_bot.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def _send_trade_notification(self, trade_data):
        """Send a trade notification via Telegram."""
        try:
            if telegram_bot:
                await telegram_bot.send_trade_notification(trade_data)
        except Exception as e:
            logger.error(f"Failed to send trade notification: {e}")

    def _run_trading_loop(self):
        """Main trading loop."""
        global trading_history, last_update_time, performance_metrics

        while self.running:
            try:
                current_time = datetime.now()

                # Simulate trading activity every 10 seconds
                if (current_time - last_update_time).total_seconds() >= 10:
                    # Process each trading pair
                    for pair in trading_pairs:
                        # Get current price and signals
                        current_price = sample_data[pair]["close"].iloc[-1]
                        current_signal = sample_data[pair]["signal"].iloc[-1]

                        # Add some random variation to price
                        price = current_price * (1 + np.random.normal(0, 0.005))

                        # Check if we have an open position for this pair
                        if pair in self.positions:
                            position = self.positions[pair]

                            # Check for stop loss or take profit
                            entry_price = position["entry_price"]
                            position_type = position["type"]

                            # Calculate profit/loss percentage
                            if position_type == "buy":
                                pnl_pct = (price - entry_price) / entry_price
                            else:  # sell
                                pnl_pct = (entry_price - price) / entry_price

                            # Get risk parameters
                            stop_loss = risk_levels[self.risk_level]["stop_loss"]
                            take_profit = risk_levels[self.risk_level]["take_profit"]

                            # Check if we should close the position
                            should_close = False
                            close_reason = ""

                            if pnl_pct <= -stop_loss:
                                should_close = True
                                close_reason = "Stop Loss"
                            elif pnl_pct >= take_profit:
                                should_close = True
                                close_reason = "Take Profit"
                            elif (position_type == "buy" and current_signal == -1) or (
                                position_type == "sell" and current_signal == 1
                            ):
                                should_close = True
                                close_reason = "Signal Reversal"

                            if should_close:
                                # Close the position
                                position_size = position["size"]
                                pnl = position_size * pnl_pct

                                # Record trade
                                trade = {
                                    "timestamp": current_time.isoformat(),
                                    "pair": pair,
                                    "type": (
                                        "sell" if position_type == "buy" else "buy"
                                    ),  # Closing action
                                    "price": float(price),
                                    "amount": position_size / entry_price,
                                    "pnl": float(pnl),
                                    "pnl_pct": float(pnl_pct * 100),
                                    "reason": close_reason,
                                    "mode": self.mode,
                                }
                                trading_history.append(trade)

                                # Update performance metrics
                                self.performance["total_trades"] += 1
                                if pnl > 0:
                                    self.performance["winning_trades"] += 1
                                    self.performance["total_profit"] += pnl
                                else:
                                    self.performance["losing_trades"] += 1
                                    self.performance["total_loss"] += pnl

                                # Update global performance metrics
                                self._update_performance_metrics()

                                # Remove the position
                                del self.positions[pair]

                                logger.info(
                                    f"Closed {position_type.upper()} position for {pair} at {price:.2f} with PnL: {pnl:.2f} ({pnl_pct*100:.2f}%) - Reason: {close_reason}"
                                )

                                # Send trade notification via Telegram
                                if telegram_bot and telegram_notifications_enabled:
                                    asyncio.run(self._send_trade_notification(trade))

                        # Check for new entry signals if we don't have a position
                        elif current_signal != 0 and pair not in self.positions:
                            # Determine position type
                            position_type = "buy" if current_signal == 1 else "sell"

                            # Calculate position size based on risk level
                            position_size_pct = risk_levels[self.risk_level][
                                "position_size"
                            ]
                            position_size = trading_balance * position_size_pct

                            # Open new position
                            self.positions[pair] = {
                                "type": position_type,
                                "entry_price": price,
                                "size": position_size,
                                "entry_time": current_time.isoformat(),
                            }

                            # Record trade
                            trade = {
                                "timestamp": current_time.isoformat(),
                                "pair": pair,
                                "type": position_type,
                                "price": float(price),
                                "amount": position_size / price,
                                "mode": self.mode,
                            }
                            trading_history.append(trade)

                            logger.info(
                                f"Opened {position_type.upper()} position for {pair} at {price:.2f}"
                            )

                            # Send trade notification via Telegram
                            if telegram_bot and telegram_notifications_enabled:
                                asyncio.run(self._send_trade_notification(trade))

                    # Update last update time
                    last_update_time = current_time

                    # Calculate current balance and drawdown
                    self._calculate_current_metrics()

                # Sleep to avoid excessive CPU usage
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)  # Sleep longer on error

    def _calculate_current_metrics(self):
        """Calculate current balance, drawdown, and other metrics."""
        global trading_balance, max_drawdown

        # Calculate current balance including unrealized P&L
        current_balance = trading_balance
        unrealized_pnl = 0.0

        for pair, position in self.positions.items():
            current_price = sample_data[pair]["close"].iloc[-1]
            entry_price = position["entry_price"]
            position_type = position["type"]
            position_size = position["size"]

            # Calculate unrealized P&L
            if position_type == "buy":
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # sell
                pnl_pct = (entry_price - current_price) / entry_price

            unrealized_pnl += position_size * pnl_pct

        # Update current balance with unrealized P&L
        current_balance += unrealized_pnl

        # Update peak balance if current balance is higher
        if current_balance > self.performance["peak_balance"]:
            self.performance["peak_balance"] = current_balance

        # Calculate current drawdown
        if self.performance["peak_balance"] > 0:
            current_drawdown = (
                self.performance["peak_balance"] - current_balance
            ) / self.performance["peak_balance"]
            self.performance["current_drawdown"] = current_drawdown

            # Update max drawdown if current drawdown is higher
            if current_drawdown > self.performance["max_drawdown"]:
                self.performance["max_drawdown"] = current_drawdown
                max_drawdown = current_drawdown

        # Update trading balance
        trading_balance = current_balance

    def _update_performance_metrics(self):
        """Update global performance metrics."""
        global performance_metrics, win_rate, profit_factor

        # Calculate win rate
        total_trades = self.performance["total_trades"]
        winning_trades = self.performance["winning_trades"]

        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
        else:
            win_rate = 0.0

        # Calculate profit factor
        total_profit = self.performance["total_profit"]
        total_loss = abs(self.performance["total_loss"])

        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = total_profit if total_profit > 0 else 0.0

        # Update global performance metrics
        performance_metrics["total_trades"] = total_trades
        performance_metrics["winning_trades"] = winning_trades
        performance_metrics["losing_trades"] = self.performance["losing_trades"]
        performance_metrics["win_rate"] = win_rate
        performance_metrics["profit_factor"] = profit_factor
        performance_metrics["max_drawdown"] = (
            self.performance["max_drawdown"] * 100
        )  # Convert to percentage

        # Calculate daily return (simplified)
        if trading_balance > 0:
            performance_metrics["daily_return"] = (
                trading_balance / 10000.0 - 1
            ) * 100  # Assuming 10000 initial balance
            performance_metrics["monthly_return"] = (
                performance_metrics["daily_return"] * 30
            )  # Simplified

        # Calculate Sharpe ratio (simplified)
        if performance_metrics["daily_return"] > 0:
            performance_metrics["sharpe_ratio"] = performance_metrics[
                "daily_return"
            ] / (
                performance_metrics["max_drawdown"] + 0.01
            )  # Avoid division by zero
        else:
            performance_metrics["sharpe_ratio"] = 0.0


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Platon Light Trading Dashboard"

# Define app layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Platon Light Trading Dashboard",
                            className="text-center mb-4",
                        ),
                        html.H5(
                            f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            id="last-update-time",
                            className="text-center text-muted mb-4",
                        ),
                    ],
                    width=12,
                )
            ],
            className="mt-4 mb-2",
        ),
        # Trading Controls and Status
        dbc.Row(
            [
                # Trading Controls
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Trading Controls", className="text-center")
                                ),
                                dbc.CardBody(
                                    [
                                        # Trading Mode Selection
                                        html.Div(
                                            [
                                                html.Label("Trading Mode:"),
                                                dcc.RadioItems(
                                                    id="trading-mode-select",
                                                    options=[
                                                        {
                                                            "label": "Dry Run Mode",
                                                            "value": "dry_run",
                                                        },
                                                        {
                                                            "label": "Real Trading Mode",
                                                            "value": "real",
                                                        },
                                                    ],
                                                    value="dry_run",
                                                    className="mb-3",
                                                    inputStyle={
                                                        "margin-right": "10px",
                                                        "margin-left": "10px",
                                                    },
                                                ),
                                            ]
                                        ),
                                        # Initial Balance Input
                                        html.Div(
                                            [
                                                html.Label("Initial Balance (USDT):"),
                                                dcc.Input(
                                                    id="initial-balance-input",
                                                    type="number",
                                                    min=100,
                                                    max=1000000,
                                                    step=100,
                                                    value=10000,
                                                    className="form-control mb-3",
                                                ),
                                            ]
                                        ),
                                        # Risk Level Selection
                                        html.Div(
                                            [
                                                html.Label("Risk Level:"),
                                                dcc.RadioItems(
                                                    id="risk-level-select",
                                                    options=[
                                                        {
                                                            "label": "Low Risk",
                                                            "value": "low",
                                                        },
                                                        {
                                                            "label": "Medium Risk",
                                                            "value": "medium",
                                                        },
                                                        {
                                                            "label": "High Risk",
                                                            "value": "high",
                                                        },
                                                    ],
                                                    value="medium",
                                                    className="mb-3",
                                                    inputStyle={
                                                        "margin-right": "10px",
                                                        "margin-left": "10px",
                                                    },
                                                ),
                                            ]
                                        ),
                                        # Telegram Notifications Toggle
                                        html.Div(
                                            [
                                                html.Label("Telegram Notifications:"),
                                                dcc.RadioItems(
                                                    id="telegram-notifications-toggle",
                                                    options=[
                                                        {
                                                            "label": "Enabled",
                                                            "value": True,
                                                        },
                                                        {
                                                            "label": "Disabled",
                                                            "value": False,
                                                        },
                                                    ],
                                                    value=True,
                                                    className="mb-3",
                                                    inputStyle={
                                                        "margin-right": "10px",
                                                        "margin-left": "10px",
                                                    },
                                                ),
                                            ]
                                        ),
                                        # Trading Pair Selection
                                        html.Div(
                                            [
                                                html.Label("Trading Pair:"),
                                                dcc.Dropdown(
                                                    id="trading-pair-select",
                                                    options=[
                                                        {"label": pair, "value": pair}
                                                        for pair in trading_pairs
                                                    ],
                                                    value=trading_pairs[0],
                                                    clearable=False,
                                                    className="mb-3",
                                                ),
                                            ]
                                        ),
                                        # Start/Stop Buttons
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Start Trading",
                                                    id="start-trading-button",
                                                    color="success",
                                                    className="me-2",
                                                ),
                                                dbc.Button(
                                                    "Stop Trading",
                                                    id="stop-trading-button",
                                                    color="danger",
                                                    disabled=True,
                                                ),
                                            ],
                                            className="d-grid gap-2 d-md-flex justify-content-md-center",
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        ),
                    ],
                    width=4,
                ),
                # Trading Status
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Account Overview", className="text-center")
                                ),
                                dbc.CardBody(
                                    [
                                        # Balance Display
                                        html.Div(
                                            [
                                                html.H5("Current Balance:"),
                                                html.H3(
                                                    f"{trading_balance:.2f} USDT",
                                                    id="current-balance",
                                                    className="text-success",
                                                ),
                                            ],
                                            className="mb-3 text-center",
                                        ),
                                        # Trading Status
                                        html.Div(
                                            [
                                                html.H5("Trading Status:"),
                                                html.H4(
                                                    "Inactive",
                                                    id="trading-status",
                                                    className="text-warning",
                                                ),
                                            ],
                                            className="mb-3 text-center",
                                        ),
                                        # Open Positions Count
                                        html.Div(
                                            [
                                                html.H5("Open Positions:"),
                                                html.H4(
                                                    "0",
                                                    id="open-positions-count",
                                                    className="text-info",
                                                ),
                                            ],
                                            className="mb-3 text-center",
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        ),
                    ],
                    width=4,
                ),
                # Risk Metrics
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Risk Metrics", className="text-center")
                                ),
                                dbc.CardBody(
                                    [
                                        # Risk Metrics Table
                                        dash_table.DataTable(
                                            id="risk-metrics-table",
                                            columns=[
                                                {"name": "Metric", "id": "metric"},
                                                {"name": "Value", "id": "value"},
                                            ],
                                            data=[
                                                {
                                                    "metric": "Max Drawdown",
                                                    "value": f"{max_drawdown*100:.2f}%",
                                                },
                                                {
                                                    "metric": "Win Rate",
                                                    "value": f"{win_rate:.2f}%",
                                                },
                                                {
                                                    "metric": "Profit Factor",
                                                    "value": f"{profit_factor:.2f}",
                                                },
                                                {
                                                    "metric": "Sharpe Ratio",
                                                    "value": f"{performance_metrics['sharpe_ratio']:.2f}",
                                                },
                                                {
                                                    "metric": "Daily Return",
                                                    "value": f"{performance_metrics['daily_return']:.2f}%",
                                                },
                                                {
                                                    "metric": "Monthly Return",
                                                    "value": f"{performance_metrics['monthly_return']:.2f}%",
                                                },
                                            ],
                                            style_header={
                                                "backgroundColor": "rgb(30, 30, 30)",
                                                "color": "white",
                                                "fontWeight": "bold",
                                            },
                                            style_cell={
                                                "backgroundColor": "rgb(50, 50, 50)",
                                                "color": "white",
                                                "border": "1px solid grey",
                                                "textAlign": "left",
                                                "padding": "8px",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {
                                                        "row_index": 0
                                                    },  # Max Drawdown
                                                    "color": (
                                                        "red"
                                                        if max_drawdown > 0.1
                                                        else (
                                                            "orange"
                                                            if max_drawdown > 0.05
                                                            else "green"
                                                        )
                                                    ),
                                                },
                                                {
                                                    "if": {"row_index": 1},  # Win Rate
                                                    "color": (
                                                        "green"
                                                        if win_rate > 60
                                                        else (
                                                            "orange"
                                                            if win_rate > 40
                                                            else "red"
                                                        )
                                                    ),
                                                },
                                                {
                                                    "if": {
                                                        "row_index": 2
                                                    },  # Profit Factor
                                                    "color": (
                                                        "green"
                                                        if profit_factor > 1.5
                                                        else (
                                                            "orange"
                                                            if profit_factor > 1
                                                            else "red"
                                                        )
                                                    ),
                                                },
                                            ],
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        # Price Chart and Trading History
        dbc.Row(
            [
                # Price Chart
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Price Chart", className="text-center")
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="price-chart",
                                            config={"displayModeBar": False},
                                            style={"height": "500px"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    width=8,
                ),
                # Trading History
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Trading History", className="text-center")
                                ),
                                dbc.CardBody(
                                    [
                                        dash_table.DataTable(
                                            id="trading-history-table",
                                            columns=[
                                                {"name": "Time", "id": "time"},
                                                {"name": "Pair", "id": "pair"},
                                                {"name": "Type", "id": "type"},
                                                {"name": "Price", "id": "price"},
                                                {"name": "PnL", "id": "pnl"},
                                            ],
                                            data=[],
                                            style_header={
                                                "backgroundColor": "rgb(30, 30, 30)",
                                                "color": "white",
                                                "fontWeight": "bold",
                                            },
                                            style_cell={
                                                "backgroundColor": "rgb(50, 50, 50)",
                                                "color": "white",
                                                "border": "1px solid grey",
                                                "textAlign": "left",
                                                "padding": "8px",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"filter_query": "{pnl} > 0"},
                                                    "color": "green",
                                                },
                                                {
                                                    "if": {"filter_query": "{pnl} < 0"},
                                                    "color": "red",
                                                },
                                                {
                                                    "if": {
                                                        "filter_query": "{type} = 'buy'"
                                                    },
                                                    "color": "lightgreen",
                                                },
                                                {
                                                    "if": {
                                                        "filter_query": "{type} = 'sell'"
                                                    },
                                                    "color": "lightcoral",
                                                },
                                            ],
                                            page_size=10,
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        # Performance Metrics and Open Positions
        dbc.Row(
            [
                # Performance Metrics
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4(
                                        "Performance Metrics", className="text-center"
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="performance-chart",
                                            config={"displayModeBar": False},
                                            style={"height": "300px"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    width=6,
                ),
                # Open Positions
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Open Positions", className="text-center")
                                ),
                                dbc.CardBody(
                                    [
                                        dash_table.DataTable(
                                            id="open-positions-table",
                                            columns=[
                                                {"name": "Pair", "id": "pair"},
                                                {"name": "Type", "id": "type"},
                                                {
                                                    "name": "Entry Price",
                                                    "id": "entry_price",
                                                },
                                                {
                                                    "name": "Current Price",
                                                    "id": "current_price",
                                                },
                                                {"name": "PnL %", "id": "pnl_pct"},
                                            ],
                                            data=[],
                                            style_header={
                                                "backgroundColor": "rgb(30, 30, 30)",
                                                "color": "white",
                                                "fontWeight": "bold",
                                            },
                                            style_cell={
                                                "backgroundColor": "rgb(50, 50, 50)",
                                                "color": "white",
                                                "border": "1px solid grey",
                                                "textAlign": "left",
                                                "padding": "8px",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {
                                                        "filter_query": "{pnl_pct} > 0"
                                                    },
                                                    "color": "green",
                                                },
                                                {
                                                    "if": {
                                                        "filter_query": "{pnl_pct} < 0"
                                                    },
                                                    "color": "red",
                                                },
                                            ],
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        ),
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        # Interval component for updating the dashboard
        dcc.Interval(
            id="interval-component",
            interval=2000,  # in milliseconds (2 seconds)
            n_intervals=0,
        ),
    ],
    fluid=True,
    className="bg-dark text-white",
)

# Initialize trading engine
trading_engine = None


# Callback for updating the price chart
@app.callback(
    Output("price-chart", "figure"),
    [Input("interval-component", "n_intervals"), Input("trading-pair-select", "value")],
)
def update_price_chart(n_intervals, pair_value):
    global selected_pair
    """Update the price chart with the latest data."""

    # Update selected pair
    selected_pair = pair_value

    # Get data for the selected pair
    df = sample_data[selected_pair].copy()

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{selected_pair} Price", "Volume"),
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["fast_ma"],
            name="Fast MA",
            line=dict(color="rgba(255, 165, 0, 0.8)", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["slow_ma"],
            name="Slow MA",
            line=dict(color="rgba(46, 139, 87, 0.8)", width=2),
        ),
        row=1,
        col=1,
    )

    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["bb_upper"],
            name="BB Upper",
            line=dict(color="rgba(173, 216, 230, 0.7)", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["bb_middle"],
            name="BB Middle",
            line=dict(color="rgba(173, 216, 230, 0.7)", width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["bb_lower"],
            name="BB Lower",
            line=dict(color="rgba(173, 216, 230, 0.7)", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Add buy/sell signals
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals["low"] * 0.99,  # Place markers below the candles
            name="Buy Signal",
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=15,
                color="green",
                line=dict(width=2, color="darkgreen"),
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals["high"] * 1.01,  # Place markers above the candles
            name="Sell Signal",
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=15,
                color="red",
                line=dict(width=2, color="darkred"),
            ),
        ),
        row=1,
        col=1,
    )

    # Add volume bar chart
    colors = [
        "green" if row["close"] >= row["open"] else "red" for _, row in df.iterrows()
    ]

    fig.add_trace(
        go.Bar(
            x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.8
        ),
        row=2,
        col=1,
    )

    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["rsi"], name="RSI", line=dict(color="purple", width=1)
        ),
        row=2,
        col=1,
    )

    # Add RSI overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[70, 70],
            name="Overbought",
            line=dict(color="red", width=1, dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[30, 30],
            name="Oversold",
            line=dict(color="green", width=1, dash="dash"),
        ),
        row=2,
        col=1,
    )

    # Add trades to chart if available
    if trading_history:
        for trade in trading_history:
            if trade["pair"] == selected_pair:
                trade_time = datetime.fromisoformat(trade["timestamp"])

                if "type" in trade and trade["type"] == "buy":
                    fig.add_trace(
                        go.Scatter(
                            x=[trade_time],
                            y=[trade["price"] * 0.98],  # Place slightly below the price
                            name="Buy",
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                size=12,
                                color="lightgreen",
                                line=dict(width=2, color="green"),
                            ),
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                elif "type" in trade and trade["type"] == "sell":
                    # Add PnL annotation if available
                    pnl_text = (
                        f" ({trade['pnl_pct']:.2f}%)" if "pnl_pct" in trade else ""
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[trade_time],
                            y=[trade["price"] * 1.02],  # Place slightly above the price
                            name=f"Sell{pnl_text}",
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                size=12,
                                color="lightcoral",
                                line=dict(width=2, color="red"),
                            ),
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

    # Update layout
    fig.update_layout(
        title=f"{selected_pair} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        plot_bgcolor="rgba(25, 25, 25, 1)",
        paper_bgcolor="rgba(25, 25, 25, 1)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=85, b=50),
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume / RSI", row=2, col=1)

    return fig


# Callback for updating the performance chart
@app.callback(
    Output("performance-chart", "figure"), [Input("interval-component", "n_intervals")]
)
def update_performance_chart(n_intervals):
    """Update the performance metrics chart."""
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Calculate cumulative returns from trading history
    if trading_history:
        # Extract timestamps and PnL values
        timestamps = []
        cumulative_pnl = []
        cumulative_sum = 0

        for trade in sorted(trading_history, key=lambda x: x["timestamp"]):
            if "pnl" in trade:
                timestamps.append(datetime.fromisoformat(trade["timestamp"]))
                cumulative_sum += trade["pnl"]
                cumulative_pnl.append(cumulative_sum)

        if timestamps:
            # Add cumulative PnL trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cumulative_pnl,
                    name="Cumulative PnL",
                    line=dict(color="green" if cumulative_sum >= 0 else "red", width=2),
                ),
                secondary_y=False,
            )

            # Add win/loss markers
            win_timestamps = []
            win_values = []
            loss_timestamps = []
            loss_values = []

            prev_sum = 0
            for i, trade in enumerate(
                sorted(trading_history, key=lambda x: x["timestamp"])
            ):
                if "pnl" in trade:
                    timestamp = datetime.fromisoformat(trade["timestamp"])
                    pnl = trade["pnl"]
                    current_sum = prev_sum + pnl

                    if pnl > 0:
                        win_timestamps.append(timestamp)
                        win_values.append(current_sum)
                    else:
                        loss_timestamps.append(timestamp)
                        loss_values.append(current_sum)

                    prev_sum = current_sum

            # Add win markers
            if win_timestamps:
                fig.add_trace(
                    go.Scatter(
                        x=win_timestamps,
                        y=win_values,
                        name="Winning Trades",
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=8,
                            color="green",
                        ),
                        showlegend=True,
                    ),
                    secondary_y=False,
                )

            # Add loss markers
            if loss_timestamps:
                fig.add_trace(
                    go.Scatter(
                        x=loss_timestamps,
                        y=loss_values,
                        name="Losing Trades",
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=8,
                            color="red",
                        ),
                        showlegend=True,
                    ),
                    secondary_y=False,
                )

            # Add balance trace
            balance_timestamps = timestamps.copy()
            balance_values = [
                10000 + pnl for pnl in cumulative_pnl
            ]  # Assuming 10000 initial balance

            fig.add_trace(
                go.Scatter(
                    x=balance_timestamps,
                    y=balance_values,
                    name="Account Balance",
                    line=dict(color="lightblue", width=2, dash="dot"),
                ),
                secondary_y=True,
            )
    else:
        # If no trading history, show empty chart with message
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No trading data available yet",
            showarrow=False,
            font=dict(size=20, color="white"),
        )

    # Update layout
    fig.update_layout(
        title="Trading Performance",
        template="plotly_dark",
        plot_bgcolor="rgba(25, 25, 25, 1)",
        paper_bgcolor="rgba(25, 25, 25, 1)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=85, b=50),
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Cumulative PnL (USDT)", secondary_y=False)
    fig.update_yaxes(title_text="Account Balance (USDT)", secondary_y=True)

    return fig


# Callback for updating the trading history table
@app.callback(
    Output("trading-history-table", "data"),
    [Input("interval-component", "n_intervals")],
)
def update_trading_history(n_intervals):
    """Update the trading history table."""
    history_data = []

    for trade in trading_history:
        trade_data = {
            "time": datetime.fromisoformat(trade["timestamp"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "pair": trade["pair"],
            "type": trade["type"].upper(),
            "price": f"{trade['price']:.2f}",
        }

        if "pnl" in trade:
            trade_data["pnl"] = f"{trade['pnl']:.2f}"
        else:
            trade_data["pnl"] = "-"

        history_data.append(trade_data)

    # Sort by time (newest first)
    history_data.sort(key=lambda x: x["time"], reverse=True)

    return history_data


# Callback for updating the open positions table
@app.callback(
    Output("open-positions-table", "data"), [Input("interval-component", "n_intervals")]
)
def update_open_positions(n_intervals):
    """Update the open positions table."""
    positions_data = []

    if trading_engine and hasattr(trading_engine, "positions"):
        for pair, position in trading_engine.positions.items():
            # Get current price
            current_price = sample_data[pair]["close"].iloc[-1]

            # Calculate PnL percentage
            if position["type"] == "buy":
                pnl_pct = (
                    (current_price - position["entry_price"]) / position["entry_price"]
                ) * 100
            else:  # sell
                pnl_pct = (
                    (position["entry_price"] - current_price) / position["entry_price"]
                ) * 100

            positions_data.append(
                {
                    "pair": pair,
                    "type": position["type"].upper(),
                    "entry_price": f"{position['entry_price']:.2f}",
                    "current_price": f"{current_price:.2f}",
                    "pnl_pct": f"{pnl_pct:.2f}",
                }
            )

    return positions_data


# Callback for updating the risk metrics table
@app.callback(
    Output("risk-metrics-table", "data"), [Input("interval-component", "n_intervals")]
)
def update_risk_metrics(n_intervals):
    """Update the risk metrics table."""
    return [
        {"metric": "Max Drawdown", "value": f"{max_drawdown*100:.2f}%"},
        {"metric": "Win Rate", "value": f"{win_rate:.2f}%"},
        {"metric": "Profit Factor", "value": f"{profit_factor:.2f}"},
        {
            "metric": "Sharpe Ratio",
            "value": f"{performance_metrics['sharpe_ratio']:.2f}",
        },
        {
            "metric": "Daily Return",
            "value": f"{performance_metrics['daily_return']:.2f}%",
        },
        {
            "metric": "Monthly Return",
            "value": f"{performance_metrics['monthly_return']:.2f}%",
        },
    ]


# Callback for updating the account overview
@app.callback(
    [
        Output("current-balance", "children"),
        Output("current-balance", "className"),
        Output("trading-status", "children"),
        Output("trading-status", "className"),
        Output("open-positions-count", "children"),
        Output("last-update-time", "children"),
    ],
    [Input("interval-component", "n_intervals")],
)
def update_account_overview(n_intervals):
    """Update the account overview section."""
    # Update balance
    balance_text = f"{trading_balance:.2f} USDT"
    balance_class = "text-success" if trading_balance >= 10000 else "text-danger"

    # Update trading status
    if trading_active:
        status_text = f"Active ({trading_mode.upper()})"
        status_class = "text-success" if trading_mode == "dry_run" else "text-danger"
    else:
        status_text = "Inactive"
        status_class = "text-warning"

    # Update open positions count
    positions_count = len(trading_engine.positions) if trading_engine else 0

    # Update last update time
    update_time = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return (
        balance_text,
        balance_class,
        status_text,
        status_class,
        str(positions_count),
        update_time,
    )


# Callback for starting/stopping trading
@app.callback(
    [
        Output("start-trading-button", "disabled"),
        Output("stop-trading-button", "disabled"),
    ],
    [
        Input("start-trading-button", "n_clicks"),
        Input("stop-trading-button", "n_clicks"),
    ],
    [
        State("trading-mode-select", "value"),
        State("initial-balance-input", "value"),
        State("risk-level-select", "value"),
        State("telegram-notifications-toggle", "value"),
    ],
    prevent_initial_call=True,
)
def toggle_trading(
    start_clicks, stop_clicks, mode, initial_balance, risk_level_value, telegram_enabled
):
    global trading_active, trading_mode, trading_engine, trading_balance, risk_level, telegram_notifications_enabled
    """Start or stop trading."""

    ctx = dash.callback_context
    if not ctx.triggered:
        return not trading_active, trading_active

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "start-trading-button" and not trading_active:
        trading_active = True
        trading_mode = mode

        # Update initial balance if provided
        if initial_balance is not None and initial_balance > 0:
            trading_balance = float(initial_balance)

        # Update risk level
        risk_level = risk_level_value

        # Update Telegram notifications setting
        telegram_notifications_enabled = telegram_enabled

        # Initialize and start trading engine
        trading_engine = EnhancedTradingEngine(mode=trading_mode, risk_level=risk_level)
        trading_engine.start()

        # Initialize Telegram bot if enabled
        if telegram_notifications_enabled and TelegramBot:
            try:
                # Get Telegram configuration
                config = {
                    "telegram": {
                        "token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
                        "chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
                    }
                }

                # Initialize Telegram bot
                global telegram_bot
                telegram_bot = TelegramBot(config, [int(config["telegram"]["chat_id"])])

                # Start bot in a separate thread
                threading.Thread(
                    target=lambda: asyncio.run(telegram_bot.start()), daemon=True
                ).start()

                # Send startup notification
                threading.Thread(
                    target=lambda: asyncio.run(
                        telegram_bot.send_message(
                            f"🚀 Trading bot started in {trading_mode.upper()} mode with {risk_level.upper()} risk level"
                        )
                    ),
                    daemon=True,
                ).start()

            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")

        return True, False

    elif button_id == "stop-trading-button" and trading_active:
        trading_active = False

        # Stop trading engine
        if trading_engine:
            trading_engine.stop()

        # Stop Telegram bot
        if telegram_bot:
            try:
                # Send shutdown notification
                threading.Thread(
                    target=lambda: asyncio.run(
                        telegram_bot.send_message("🛑 Trading bot stopped")
                    ),
                    daemon=True,
                ).start()

                # Stop bot
                threading.Thread(
                    target=lambda: asyncio.run(telegram_bot.stop()), daemon=True
                ).start()
            except Exception as e:
                logger.error(f"Failed to stop Telegram bot: {e}")

        return False, True

    return not trading_active, trading_active


# Main entry point
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Platon Light Enhanced Trading Dashboard"
    )
    parser.add_argument(
        "--mode", choices=["dry_run", "real"], default="dry_run", help="Trading mode"
    )
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    args = parser.parse_args()

    # Set initial trading mode
    trading_mode = args.mode

    # Print startup message
    print(
        f"Starting Platon Light Enhanced Trading Dashboard in {trading_mode.upper()} mode"
    )
    print(f"Dashboard URL: http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to exit")

    # Start the dashboard
    app.run_server(debug=True, port=args.port)
