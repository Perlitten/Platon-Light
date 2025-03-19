#!/usr/bin/env python
"""
Platon Light Trading Dashboard

This script launches a real-time trading dashboard that allows you to:
1. Select between real trading and dry run mode
2. Monitor trading activity in real-time
3. View performance metrics and charts
4. Control trading parameters on the fly

Usage:
    python launch_trading_dashboard.py [options]

Options:
    --port: Port to run the dashboard on (default: 8050)
    --debug: Run in debug mode (default: False)
"""

import os
import sys
import logging
import argparse
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Platon Light modules
from platon_light.backtesting.strategies.moving_average_crossover import MovingAverageCrossover
from launch_live_backtest import SecureCredentialManager, fetch_market_data


# Global variables
trading_active = False
trading_mode = 'dry_run'  # 'dry_run' or 'real'
trading_pairs = ['BTCUSDT', 'ETHUSDT']
selected_pair = 'BTCUSDT'
timeframe = '1m'
strategy_params = {
    'fast_ma_type': 'EMA',
    'slow_ma_type': 'SMA',
    'fast_period': 10,
    'slow_period': 40,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'use_filters': False
}
trading_data = {
    'BTCUSDT': pd.DataFrame(),
    'ETHUSDT': pd.DataFrame()
}
trading_signals = {
    'BTCUSDT': pd.DataFrame(),
    'ETHUSDT': pd.DataFrame()
}
trading_positions = {
    'BTCUSDT': 0,
    'ETHUSDT': 0
}
trading_balance = 10000.0  # Starting balance in USDT
trading_history = []
last_update_time = datetime.now()


class TradingEngine:
    """Trading engine for executing trades in real-time."""
    
    def __init__(self, mode='dry_run'):
        """
        Initialize the trading engine.
        
        Args:
            mode: Trading mode ('dry_run' or 'real')
        """
        self.mode = mode
        self.credential_manager = SecureCredentialManager()
        self.credentials = self.credential_manager.load_credentials()
        self.strategies = {}
        self.running = False
        self.thread = None
        
        # Initialize strategies
        for pair in trading_pairs:
            self.strategies[pair] = MovingAverageCrossover(**strategy_params)
    
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
        global trading_data, trading_signals, trading_positions, trading_balance, trading_history, last_update_time
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Update market data every minute
                if (current_time - last_update_time).total_seconds() >= 60:
                    for pair in trading_pairs:
                        # Fetch latest market data
                        new_data = self._fetch_market_data(pair, timeframe)
                        
                        if new_data is not None and not new_data.empty:
                            # Update trading data
                            trading_data[pair] = new_data
                            
                            # Run strategy
                            result = self.strategies[pair].run(new_data)
                            trading_signals[pair] = result
                            
                            # Check for new signals
                            latest_signal = result['signal'].iloc[-1]
                            current_position = trading_positions[pair]
                            
                            if latest_signal == 1 and current_position <= 0:  # Buy signal
                                # Execute buy order
                                if self.mode == 'real' and self.credentials:
                                    self._execute_order(pair, 'buy')
                                
                                # Update position (in dry run mode or after real execution)
                                trading_positions[pair] = 1
                                
                                # Record trade
                                trade = {
                                    'timestamp': current_time.isoformat(),
                                    'pair': pair,
                                    'type': 'buy',
                                    'price': float(result['close'].iloc[-1]),
                                    'amount': 0.1,  # Fixed amount for simplicity
                                    'mode': self.mode
                                }
                                trading_history.append(trade)
                                
                                logger.info(f"BUY signal for {pair} at {trade['price']}")
                            
                            elif latest_signal == -1 and current_position >= 0:  # Sell signal
                                # Execute sell order
                                if self.mode == 'real' and self.credentials:
                                    self._execute_order(pair, 'sell')
                                
                                # Update position (in dry run mode or after real execution)
                                trading_positions[pair] = -1
                                
                                # Record trade
                                trade = {
                                    'timestamp': current_time.isoformat(),
                                    'pair': pair,
                                    'type': 'sell',
                                    'price': float(result['close'].iloc[-1]),
                                    'amount': 0.1,  # Fixed amount for simplicity
                                    'mode': self.mode
                                }
                                trading_history.append(trade)
                                
                                logger.info(f"SELL signal for {pair} at {trade['price']}")
                    
                    # Update last update time
                    last_update_time = current_time
                
                # Sleep to avoid excessive CPU usage
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)  # Sleep longer on error
    
    def _fetch_market_data(self, symbol, timeframe):
        """
        Fetch latest market data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)  # Get 1 day of data
            
            # Format dates
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data
            if self.mode == 'real' and self.credentials:
                # Use real market data
                data = fetch_market_data(
                    symbol,
                    timeframe,
                    start_date_str,
                    end_date_str,
                    self.credentials.get('BINANCE_API_KEY'),
                    self.credentials.get('BINANCE_API_SECRET')
                )
            else:
                # Use sample data or generate synthetic data
                data_file = Path(__file__).parent / 'data' / 'sample' / f"{symbol}_{timeframe}.csv"
                
                if data_file.exists():
                    data = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
                else:
                    # Generate synthetic data
                    data = self._generate_synthetic_data(symbol, timeframe)
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def _generate_synthetic_data(self, symbol, timeframe):
        """
        Generate synthetic market data for dry run mode.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Number of data points
        n = 1000
        
        # Generate timestamps
        end_time = datetime.now()
        
        if timeframe == '1m':
            start_time = end_time - timedelta(minutes=n)
            freq = 'T'  # Minute frequency
        elif timeframe == '5m':
            start_time = end_time - timedelta(minutes=5*n)
            freq = '5T'
        elif timeframe == '15m':
            start_time = end_time - timedelta(minutes=15*n)
            freq = '15T'
        elif timeframe == '1h':
            start_time = end_time - timedelta(hours=n)
            freq = 'H'
        elif timeframe == '4h':
            start_time = end_time - timedelta(hours=4*n)
            freq = '4H'
        elif timeframe == '1d':
            start_time = end_time - timedelta(days=n)
            freq = 'D'
        else:
            start_time = end_time - timedelta(minutes=n)
            freq = 'T'
        
        timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Generate price data with random walk
        np.random.seed(42)  # For reproducibility
        
        # Initial price
        if symbol == 'BTCUSDT':
            initial_price = 50000.0
        elif symbol == 'ETHUSDT':
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
        data['close'] = price_series
        
        # Generate open, high, low based on close
        data['open'] = data['close'].shift(1)
        data.loc[data.index[0], 'open'] = data['close'].iloc[0] * (1 - np.random.normal(0, volatility))
        
        # High is the maximum of open and close, plus a random amount
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, volatility, len(data))))
        
        # Low is the minimum of open and close, minus a random amount
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, volatility, len(data))))
        
        # Generate volume
        data['volume'] = np.random.lognormal(10, 1, len(data)) * (1 + 0.1 * np.sin(np.linspace(0, 20*np.pi, len(data))))
        
        return data
    
    def _execute_order(self, symbol, order_type):
        """
        Execute a real order on the exchange.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type ('buy' or 'sell')
        """
        try:
            # Import ccxt only when needed
            import ccxt
            
            # Initialize exchange
            exchange_params = {}
            if self.credentials:
                exchange_params['apiKey'] = self.credentials.get('BINANCE_API_KEY')
                exchange_params['secret'] = self.credentials.get('BINANCE_API_SECRET')
            
            exchange = ccxt.binance(exchange_params)
            
            # Prepare order parameters
            amount = 0.1  # Fixed amount for simplicity
            
            # Execute order
            if order_type == 'buy':
                order = exchange.create_market_buy_order(symbol, amount)
            else:
                order = exchange.create_market_sell_order(symbol, amount)
            
            logger.info(f"Order executed: {order}")
            
            return order
        
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return None


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

server = app.server

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Platon Light Trading Dashboard", className="text-center my-4"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Trading Mode"),
                            dbc.RadioItems(
                                id="trading-mode-select",
                                options=[
                                    {"label": "Dry Run", "value": "dry_run"},
                                    {"label": "Real Trading", "value": "real"}
                                ],
                                value=trading_mode,
                                inline=True
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Trading Pair"),
                            dbc.Select(
                                id="trading-pair-select",
                                options=[{"label": pair, "value": pair} for pair in trading_pairs],
                                value=selected_pair
                            )
                        ], width=6)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Start Trading",
                                id="start-trading-button",
                                color="success",
                                className="me-2"
                            ),
                            dbc.Button(
                                "Stop Trading",
                                id="stop-trading-button",
                                color="danger",
                                disabled=not trading_active
                            )
                        ], width=12)
                    ])
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Account Overview"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("Balance"),
                                html.H2(id="account-balance", children=f"${trading_balance:.2f}")
                            ])
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H4("Open Positions"),
                                html.H2(id="open-positions", children="0")
                            ])
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H4("Trading Status"),
                                html.H2(id="trading-status", children="Inactive")
                            ])
                        ], width=4)
                    ])
                ])
            ])
        ], width=8)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Chart"),
                dbc.CardBody([
                    dcc.Graph(id="price-chart", style={"height": "400px"})
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading History"),
                dbc.CardBody([
                    html.Div(id="trading-history-table")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Strategy Parameters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Fast MA Type"),
                            dbc.Select(
                                id="fast-ma-type-select",
                                options=[
                                    {"label": "SMA", "value": "SMA"},
                                    {"label": "EMA", "value": "EMA"}
                                ],
                                value=strategy_params["fast_ma_type"]
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Slow MA Type"),
                            dbc.Select(
                                id="slow-ma-type-select",
                                options=[
                                    {"label": "SMA", "value": "SMA"},
                                    {"label": "EMA", "value": "EMA"}
                                ],
                                value=strategy_params["slow_ma_type"]
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Fast Period"),
                            dbc.Input(
                                id="fast-period-input",
                                type="number",
                                min=5,
                                max=50,
                                step=1,
                                value=strategy_params["fast_period"]
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Slow Period"),
                            dbc.Input(
                                id="slow-period-input",
                                type="number",
                                min=10,
                                max=200,
                                step=1,
                                value=strategy_params["slow_period"]
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Use Filters"),
                            dbc.Checkbox(
                                id="use-filters-checkbox",
                                value=strategy_params["use_filters"]
                            )
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Update Parameters",
                                id="update-params-button",
                                color="primary",
                                className="mt-3"
                            )
                        ], width=12)
                    ])
                ])
            ])
        ], width=6)
    ]),
    
    dcc.Interval(
        id="update-interval",
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    )
], fluid=True)


# Define callbacks
@app.callback(
    [
        Output("account-balance", "children"),
        Output("open-positions", "children"),
        Output("trading-status", "children"),
        Output("trading-history-table", "children"),
        Output("price-chart", "figure")
    ],
    [
        Input("update-interval", "n_intervals"),
        Input("trading-pair-select", "value")
    ]
)
def update_dashboard(n_intervals, selected_pair):
    """Update dashboard with latest data."""
    global trading_data, trading_signals, trading_positions, trading_balance, trading_history
    
    # Update account balance
    balance_text = f"${trading_balance:.2f}"
    
    # Count open positions
    open_positions = sum(1 for pos in trading_positions.values() if pos != 0)
    
    # Trading status
    status_text = "Active" if trading_active else "Inactive"
    status_color = "text-success" if trading_active else "text-danger"
    status_html = html.Span(status_text, className=status_color)
    
    # Trading history table
    if trading_history:
        # Create a DataFrame from trading history
        history_df = pd.DataFrame(trading_history)
        
        # Format the table
        history_table = dbc.Table.from_dataframe(
            history_df.tail(10),  # Show only the last 10 trades
            striped=True,
            bordered=True,
            hover=True,
            responsive=True
        )
    else:
        history_table = html.P("No trading history yet")
    
    # Price chart
    if selected_pair in trading_data and not trading_data[selected_pair].empty:
        df = trading_data[selected_pair].copy()
        
        # Create figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ))
        
        # Add moving averages if available
        if selected_pair in trading_signals and not trading_signals[selected_pair].empty:
            signals_df = trading_signals[selected_pair]
            
            if 'fast_ma' in signals_df.columns:
                fig.add_trace(go.Scatter(
                    x=signals_df.index,
                    y=signals_df['fast_ma'],
                    mode='lines',
                    name=f"{signals_df['fast_ma_type'].iloc[0]} {signals_df['fast_period'].iloc[0]}"
                ))
            
            if 'slow_ma' in signals_df.columns:
                fig.add_trace(go.Scatter(
                    x=signals_df.index,
                    y=signals_df['slow_ma'],
                    mode='lines',
                    name=f"{signals_df['slow_ma_type'].iloc[0]} {signals_df['slow_period'].iloc[0]}"
                ))
            
            # Add buy signals
            buy_signals = signals_df[signals_df['signal'] == 1]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green'
                    ),
                    name="Buy Signal"
                ))
            
            # Add sell signals
            sell_signals = signals_df[signals_df['signal'] == -1]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red'
                    ),
                    name="Sell Signal"
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{selected_pair} Price Chart",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
    else:
        # Empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title=f"{selected_pair} Price Chart - No Data",
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
    
    return balance_text, str(open_positions), status_html, history_table, fig


@app.callback(
    [
        Output("start-trading-button", "disabled"),
        Output("stop-trading-button", "disabled")
    ],
    [
        Input("start-trading-button", "n_clicks"),
        Input("stop-trading-button", "n_clicks")
    ],
    [
        State("trading-mode-select", "value")
    ],
    prevent_initial_call=True
)
def toggle_trading(start_clicks, stop_clicks, mode):
    """Start or stop trading."""
    global trading_active, trading_mode, trading_engine
    
    ctx = callback_context
    if not ctx.triggered:
        return not trading_active, trading_active
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "start-trading-button" and not trading_active:
        trading_active = True
        trading_mode = mode
        
        # Initialize and start trading engine
        trading_engine = TradingEngine(mode=trading_mode)
        trading_engine.start()
        
        return True, False
    
    elif button_id == "stop-trading-button" and trading_active:
        trading_active = False
        
        # Stop trading engine
        if 'trading_engine' in globals():
            trading_engine.stop()
        
        return False, True
    
    return not trading_active, trading_active


@app.callback(
    Output("update-params-button", "disabled"),
    [
        Input("update-params-button", "n_clicks")
    ],
    [
        State("fast-ma-type-select", "value"),
        State("slow-ma-type-select", "value"),
        State("fast-period-input", "value"),
        State("slow-period-input", "value"),
        State("use-filters-checkbox", "value")
    ],
    prevent_initial_call=True
)
def update_strategy_parameters(n_clicks, fast_ma_type, slow_ma_type, fast_period, slow_period, use_filters):
    """Update strategy parameters."""
    global strategy_params
    
    # Update strategy parameters
    strategy_params.update({
        'fast_ma_type': fast_ma_type,
        'slow_ma_type': slow_ma_type,
        'fast_period': fast_period,
        'slow_period': slow_period,
        'use_filters': use_filters
    })
    
    # If trading engine is running, update its strategies
    if trading_active and 'trading_engine' in globals():
        for pair in trading_pairs:
            trading_engine.strategies[pair] = MovingAverageCrossover(**strategy_params)
    
    logger.info(f"Strategy parameters updated: {strategy_params}")
    
    # Flash the button (disable briefly)
    return True


@app.callback(
    Output("update-params-button", "disabled", allow_duplicate=True),
    [Input("update-params-button", "disabled")],
    prevent_initial_call=True
)
def reset_update_button(disabled):
    """Reset the update button after a brief delay."""
    if disabled:
        time.sleep(0.5)  # Wait for 0.5 seconds
        return False
    return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Launch trading dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting Platon Light Trading Dashboard")
    
    # Run the app
    app.run_server(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
