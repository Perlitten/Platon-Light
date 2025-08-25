#!/usr/bin/env python
"""
Platon Light Simple Trading Dashboard

A simplified version of the trading dashboard that focuses on core functionality:
1. Select between real trading and dry run mode
2. Monitor trading activity in real-time
3. View performance metrics and charts

Usage:
    python simple_trading_dashboard.py
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Platon Light modules
try:
    from platon_light.backtesting.strategies.moving_average_crossover import (
        MovingAverageCrossover,
    )
    from launch_live_backtest import SecureCredentialManager
except ImportError:
    logger.warning("Could not import Platon Light modules. Running in standalone mode.")

# Global variables
trading_active = False
trading_mode = "dry_run"  # 'dry_run' or 'real'
trading_pairs = ["BTCUSDT", "ETHUSDT"]
selected_pair = "BTCUSDT"
timeframe = "1h"
trading_balance = 10000.0  # Starting balance in USDT
trading_history = []
last_update_time = datetime.now()


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
sample_data = {
    "BTCUSDT": generate_sample_data("BTCUSDT"),
    "ETHUSDT": generate_sample_data("ETHUSDT"),
}

# Add moving averages to sample data
for pair in trading_pairs:
    # Calculate fast EMA (10-period)
    sample_data[pair]["fast_ma"] = sample_data[pair]["close"].ewm(span=10).mean()

    # Calculate slow SMA (40-period)
    sample_data[pair]["slow_ma"] = sample_data[pair]["close"].rolling(window=40).mean()

    # Generate some sample signals
    sample_data[pair]["signal"] = 0

    # Buy signal when fast MA crosses above slow MA
    sample_data[pair].loc[
        (sample_data[pair]["fast_ma"] > sample_data[pair]["slow_ma"])
        & (
            sample_data[pair]["fast_ma"].shift(1)
            <= sample_data[pair]["slow_ma"].shift(1)
        ),
        "signal",
    ] = 1

    # Sell signal when fast MA crosses below slow MA
    sample_data[pair].loc[
        (sample_data[pair]["fast_ma"] < sample_data[pair]["slow_ma"])
        & (
            sample_data[pair]["fast_ma"].shift(1)
            >= sample_data[pair]["slow_ma"].shift(1)
        ),
        "signal",
    ] = -1


# Simulated trading engine
class SimpleTradingEngine:
    """Simplified trading engine for demonstration."""

    def __init__(self, mode="dry_run"):
        """Initialize the trading engine."""
        self.mode = mode
        self.running = False
        self.thread = None

    def start(self):
        """Start the trading engine."""
        if self.running:
            logger.warning("Trading engine is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_trading_loop)
        self.thread.daemon = True
        self.thread.start()

        logger.info(f"Trading engine started in {self.mode} mode")

    def stop(self):
        """Stop the trading engine."""
        if not self.running:
            logger.warning("Trading engine is not running")
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

        logger.info("Trading engine stopped")

    def _run_trading_loop(self):
        """Main trading loop."""
        global trading_history, last_update_time

        while self.running:
            try:
                current_time = datetime.now()

                # Simulate trading activity every 10 seconds
                if (current_time - last_update_time).total_seconds() >= 10:
                    # Generate a random trade for demonstration
                    pair = np.random.choice(trading_pairs)
                    trade_type = np.random.choice(["buy", "sell"])

                    # Get current price from sample data
                    current_price = sample_data[pair]["close"].iloc[-1]

                    # Add some random variation
                    price = current_price * (1 + np.random.normal(0, 0.005))

                    # Record trade
                    trade = {
                        "timestamp": current_time.isoformat(),
                        "pair": pair,
                        "type": trade_type,
                        "price": float(price),
                        "amount": 0.1,  # Fixed amount for simplicity
                        "mode": self.mode,
                    }
                    trading_history.append(trade)

                    logger.info(
                        f"Simulated {trade_type.upper()} signal for {pair} at {price:.2f}"
                    )

                    # Update last update time
                    last_update_time = current_time

                # Sleep to avoid excessive CPU usage
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)  # Sleep longer on error


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Define app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Platon Light Trading Dashboard",
                            className="text-center my-4",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Trading Controls"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Trading Mode"),
                                                        dbc.RadioItems(
                                                            id="trading-mode-select",
                                                            options=[
                                                                {
                                                                    "label": "Dry Run",
                                                                    "value": "dry_run",
                                                                },
                                                                {
                                                                    "label": "Real Trading",
                                                                    "value": "real",
                                                                },
                                                            ],
                                                            value=trading_mode,
                                                            inline=True,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Trading Pair"),
                                                        dbc.Select(
                                                            id="trading-pair-select",
                                                            options=[
                                                                {
                                                                    "label": pair,
                                                                    "value": pair,
                                                                }
                                                                for pair in trading_pairs
                                                            ],
                                                            value=selected_pair,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Initial Balance (USDT)"
                                                        ),
                                                        dbc.Input(
                                                            id="initial-balance-input",
                                                            type="number",
                                                            min=100,
                                                            max=1000000,
                                                            step=100,
                                                            value=trading_balance,
                                                            style={
                                                                "marginBottom": "10px"
                                                            },
                                                        ),
                                                    ],
                                                    width=12,
                                                )
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
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
                                                    width=12,
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Account Overview"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H4("Balance"),
                                                                html.H2(
                                                                    id="account-balance",
                                                                    children=f"${trading_balance:.2f}",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Trading Status"
                                                                ),
                                                                html.H2(
                                                                    id="trading-status",
                                                                    children="Inactive",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        )
                    ],
                    width=8,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Price Chart"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="price-chart", style={"height": "400px"}
                                        )
                                    ]
                                ),
                            ]
                        )
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Trading History"),
                                dbc.CardBody([html.Div(id="trading-history-table")]),
                            ]
                        )
                    ],
                    width=12,
                )
            ]
        ),
        dcc.Interval(
            id="update-interval", interval=5000, n_intervals=0  # Update every 5 seconds
        ),
    ],
    fluid=True,
)


# Define callbacks
@app.callback(
    [
        Output("account-balance", "children"),
        Output("trading-status", "children"),
        Output("trading-history-table", "children"),
        Output("price-chart", "figure"),
    ],
    [Input("update-interval", "n_intervals"), Input("trading-pair-select", "value")],
)
def update_dashboard(n_intervals, selected_pair):
    """Update dashboard with latest data."""
    # Update account balance (simulate some fluctuation)
    balance = (
        trading_balance + np.random.normal(0, 50) if trading_active else trading_balance
    )
    balance_text = f"${balance:.2f}"

    # Trading status
    status_text = "Active" if trading_active else "Inactive"
    status_color = "text-success" if trading_active else "text-danger"
    status_html = html.Span(status_text, className=status_color)

    # Trading history table
    if trading_history:
        # Format the trading history as a table
        history_rows = []

        # Table header
        history_rows.append(
            html.Tr(
                [
                    html.Th("Time"),
                    html.Th("Pair"),
                    html.Th("Type"),
                    html.Th("Price"),
                    html.Th("Amount"),
                    html.Th("Mode"),
                ]
            )
        )

        # Table rows (last 10 trades)
        for trade in trading_history[-10:]:
            trade_time = datetime.fromisoformat(trade["timestamp"]).strftime("%H:%M:%S")
            trade_type_class = (
                "text-success" if trade["type"] == "buy" else "text-danger"
            )

            history_rows.append(
                html.Tr(
                    [
                        html.Td(trade_time),
                        html.Td(trade["pair"]),
                        html.Td(
                            html.Span(trade["type"].upper(), className=trade_type_class)
                        ),
                        html.Td(f"${trade['price']:.2f}"),
                        html.Td(f"{trade['amount']:.4f}"),
                        html.Td(trade["mode"]),
                    ]
                )
            )

        history_table = html.Table(
            history_rows, className="table table-striped table-dark"
        )
    else:
        history_table = html.P("No trading history yet")

    # Price chart
    df = sample_data[selected_pair].copy()

    # Create figure
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    # Add moving averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df["fast_ma"], mode="lines", name="Fast EMA (10)")
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["slow_ma"], mode="lines", name="Slow SMA (40)")
    )

    # Add buy signals
    buy_signals = df[df["signal"] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals["close"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=15, color="green"),
                name="Buy Signal",
            )
        )

    # Add sell signals
    sell_signals = df[df["signal"] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["close"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=15, color="red"),
                name="Sell Signal",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{selected_pair} Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return balance_text, status_html, history_table, fig


@app.callback(
    [
        Output("start-trading-button", "disabled"),
        Output("stop-trading-button", "disabled"),
    ],
    [
        Input("start-trading-button", "n_clicks"),
        Input("stop-trading-button", "n_clicks"),
    ],
    [State("trading-mode-select", "value"), State("initial-balance-input", "value")],
    prevent_initial_call=True,
)
def toggle_trading(start_clicks, stop_clicks, mode, initial_balance):
    """Start or stop trading."""
    global trading_active, trading_mode, trading_engine, trading_balance

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

        # Initialize and start trading engine
        trading_engine = SimpleTradingEngine(mode=trading_mode)
        trading_engine.start()

        return True, False

    elif button_id == "stop-trading-button" and trading_active:
        trading_active = False

        # Stop trading engine
        if "trading_engine" in globals():
            trading_engine.stop()

        return False, True

    return not trading_active, trading_active


if __name__ == "__main__":
    logger.info("Starting Platon Light Simple Trading Dashboard")
    app.run(debug=False, port=8050)
    logger.info("Dashboard is running at http://localhost:8050")
