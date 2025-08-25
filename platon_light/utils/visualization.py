"""
Console visualization utilities for displaying trading data and performance metrics
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import plotext as plt
from tabulate import tabulate
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init()


class ConsoleVisualizer:
    """
    Console-based visualizer for displaying trading data and performance metrics
    """

    def __init__(self, config: Dict):
        """
        Initialize the console visualizer

        Args:
            config: Bot configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.last_update = 0
        self.update_interval = config.get("visualization", {}).get(
            "update_interval", 2
        )  # seconds
        self.max_candles = config.get("visualization", {}).get("max_candles", 50)
        self.chart_width = config.get("visualization", {}).get("chart_width", 80)
        self.chart_height = config.get("visualization", {}).get("chart_height", 15)

        self.logger.info("Console visualizer initialized")

    def update_display(
        self, trading_data: Dict, performance_data: Dict, active_positions: List[Dict]
    ):
        """
        Update the console display with trading data and performance metrics

        Args:
            trading_data: Dictionary of trading data
            performance_data: Dictionary of performance metrics
            active_positions: List of active positions
        """
        current_time = time.time()

        # Only update at specified interval
        if current_time - self.last_update < self.update_interval:
            return

        self.last_update = current_time

        # Clear console
        print("\033c", end="")

        # Print header
        self._print_header()

        # Print price chart
        self._print_price_chart(trading_data)

        # Print active positions
        self._print_active_positions(active_positions)

        # Print performance metrics
        self._print_performance_metrics(performance_data)

        # Print order book
        self._print_order_book(trading_data)

    def _print_header(self):
        """Print header with bot name and time"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.CYAN}╔{'═' * 78}╗{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.WHITE}{Style.BRIGHT}PLATON LIGHT - BINANCE SCALPING BOT{Style.RESET_ALL}{' ' * 44}{Fore.CYAN}║{Style.RESET_ALL}"
        )
        print(
            f"{Fore.CYAN}║{Style.RESET_ALL} {current_time}{' ' * (78 - len(current_time))}{Fore.CYAN}║{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}╚{'═' * 78}╝{Style.RESET_ALL}")
        print()

    def _print_price_chart(self, trading_data: Dict):
        """
        Print price chart with indicators

        Args:
            trading_data: Dictionary of trading data
        """
        if (
            not trading_data
            or "symbol" not in trading_data
            or "ohlcv" not in trading_data
        ):
            print(f"{Fore.YELLOW}No price data available{Style.RESET_ALL}")
            return

        symbol = trading_data["symbol"]
        ohlcv = trading_data["ohlcv"]

        # Limit number of candles to display
        if len(ohlcv) > self.max_candles:
            ohlcv = ohlcv[-self.max_candles :]

        # Extract OHLCV data
        timestamps = [
            datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S")
            for ts in [x[0] for x in ohlcv]
        ]
        opens = [float(x[1]) for x in ohlcv]
        highs = [float(x[2]) for x in ohlcv]
        lows = [float(x[3]) for x in ohlcv]
        closes = [float(x[4]) for x in ohlcv]
        volumes = [float(x[5]) for x in ohlcv]

        # Calculate indicators if available
        indicators = trading_data.get("indicators", {})

        # Set up plot
        plt.clf()
        plt.plotsize(self.chart_width, self.chart_height)

        # Plot candlesticks
        plt.candlestick(opens, highs, lows, closes)

        # Plot indicators if available
        if "ema_fast" in indicators:
            plt.plot(indicators["ema_fast"], color="red", label="EMA Fast")

        if "ema_slow" in indicators:
            plt.plot(indicators["ema_slow"], color="blue", label="EMA Slow")

        if "upper_band" in indicators and "lower_band" in indicators:
            plt.plot(indicators["upper_band"], color="green", label="Upper BB")
            plt.plot(indicators["lower_band"], color="green", label="Lower BB")

        # Set title and labels
        plt.title(f"{symbol} - {trading_data.get('timeframe', '1m')}")
        plt.xlabel("Time")
        plt.ylabel("Price")

        # Show plot
        plt.show()

        # Print current price and indicators
        current_price = closes[-1] if closes else None
        if current_price:
            price_change = current_price - opens[-1] if opens else 0
            price_change_pct = (price_change / opens[-1]) * 100 if opens else 0

            price_color = Fore.GREEN if price_change >= 0 else Fore.RED
            print(
                f"Current Price: {price_color}{current_price:.8f}{Style.RESET_ALL} ({price_color}{price_change_pct:+.2f}%{Style.RESET_ALL})"
            )

            if "rsi" in indicators:
                rsi = indicators["rsi"][-1] if indicators["rsi"] else None
                if rsi is not None:
                    rsi_color = (
                        Fore.GREEN
                        if rsi < 30
                        else (Fore.RED if rsi > 70 else Fore.WHITE)
                    )
                    print(f"RSI: {rsi_color}{rsi:.2f}{Style.RESET_ALL}", end="  ")

            if "macd" in indicators and "macd_signal" in indicators:
                macd = indicators["macd"][-1] if indicators["macd"] else None
                macd_signal = (
                    indicators["macd_signal"][-1] if indicators["macd_signal"] else None
                )
                if macd is not None and macd_signal is not None:
                    macd_color = Fore.GREEN if macd > macd_signal else Fore.RED
                    print(f"MACD: {macd_color}{macd:.8f}{Style.RESET_ALL}", end="  ")

            print()

    def _print_active_positions(self, active_positions: List[Dict]):
        """
        Print active positions

        Args:
            active_positions: List of active positions
        """
        if not active_positions:
            print(f"\n{Fore.YELLOW}No active positions{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}Active Positions:{Style.RESET_ALL}")

        # Prepare table data
        table_data = []
        for pos in active_positions:
            # Calculate profit/loss
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", 0)
            quantity = pos.get("quantity", 0)
            side = pos.get("side", "")

            if side.lower() == "buy":
                pnl = (current_price - entry_price) * quantity
                pnl_pct = ((current_price / entry_price) - 1) * 100
            else:  # sell
                pnl = (entry_price - current_price) * quantity
                pnl_pct = ((entry_price / current_price) - 1) * 100

            # Format PnL with color
            pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
            pnl_str = f"{pnl_color}{pnl:.8f} ({pnl_pct:+.2f}%){Style.RESET_ALL}"

            # Add row to table
            table_data.append(
                [
                    pos.get("symbol", ""),
                    pos.get("side", ""),
                    pos.get("quantity", 0),
                    entry_price,
                    current_price,
                    pnl_str,
                    pos.get("tp_price", ""),
                    pos.get("sl_price", ""),
                    pos.get("open_time", ""),
                ]
            )

        # Print table
        headers = [
            "Symbol",
            "Side",
            "Quantity",
            "Entry Price",
            "Current Price",
            "PnL",
            "TP",
            "SL",
            "Open Time",
        ]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

    def _print_performance_metrics(self, performance_data: Dict):
        """
        Print performance metrics

        Args:
            performance_data: Dictionary of performance metrics
        """
        if not performance_data:
            print(f"\n{Fore.YELLOW}No performance data available{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}Performance Metrics:{Style.RESET_ALL}")

        # Extract metrics
        total_trades = performance_data.get("total_trades", 0)
        winning_trades = performance_data.get("winning_trades", 0)
        losing_trades = performance_data.get("losing_trades", 0)

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_profit = performance_data.get("total_profit", 0)
        total_loss = performance_data.get("total_loss", 0)
        net_profit = total_profit + total_loss  # total_loss is negative

        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0

        profit_factor = (
            abs(total_profit / total_loss) if total_loss != 0 else float("inf")
        )

        # Format with colors
        win_rate_color = Fore.GREEN if win_rate >= 50 else Fore.RED
        net_profit_color = Fore.GREEN if net_profit >= 0 else Fore.RED

        # Print metrics
        print(
            f"Total Trades: {total_trades}  |  "
            f"Win Rate: {win_rate_color}{win_rate:.2f}%{Style.RESET_ALL}  |  "
            f"Net Profit: {net_profit_color}{net_profit:.8f}{Style.RESET_ALL}"
        )

        print(
            f"Avg Win: {Fore.GREEN}{avg_win:.8f}{Style.RESET_ALL}  |  "
            f"Avg Loss: {Fore.RED}{avg_loss:.8f}{Style.RESET_ALL}  |  "
            f"Profit Factor: {profit_factor:.2f}"
        )

        # Print daily stats if available
        if "daily_stats" in performance_data:
            daily_stats = performance_data["daily_stats"]
            print(f"\n{Fore.CYAN}Daily Performance:{Style.RESET_ALL}")

            table_data = []
            for date, stats in daily_stats.items():
                daily_profit = stats.get("profit", 0)
                daily_trades = stats.get("trades", 0)
                daily_win_rate = stats.get("win_rate", 0)

                profit_color = Fore.GREEN if daily_profit >= 0 else Fore.RED
                win_rate_color = Fore.GREEN if daily_win_rate >= 50 else Fore.RED

                table_data.append(
                    [
                        date,
                        f"{profit_color}{daily_profit:.8f}{Style.RESET_ALL}",
                        daily_trades,
                        f"{win_rate_color}{daily_win_rate:.2f}%{Style.RESET_ALL}",
                    ]
                )

            headers = ["Date", "Profit", "Trades", "Win Rate"]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))

    def _print_order_book(self, trading_data: Dict):
        """
        Print order book visualization

        Args:
            trading_data: Dictionary of trading data
        """
        if not trading_data or "order_book" not in trading_data:
            return

        order_book = trading_data["order_book"]
        if not order_book or "bids" not in order_book or "asks" not in order_book:
            return

        bids = order_book["bids"][:5]  # Top 5 bids
        asks = order_book["asks"][:5]  # Top 5 asks

        print(f"\n{Fore.CYAN}Order Book:{Style.RESET_ALL}")

        # Print asks (in reverse order, highest first)
        for price, quantity in reversed(asks):
            bar_length = min(int(quantity * 100), 40)  # Scale the bar
            bar = "█" * bar_length
            print(f"{Fore.RED}ASK: {price:.8f} | {quantity:.6f} {bar}{Style.RESET_ALL}")

        # Print current price if available
        if "current_price" in trading_data:
            current_price = trading_data["current_price"]
            print(
                f"{Fore.YELLOW}{'─' * 30} {current_price:.8f} {'─' * 30}{Style.RESET_ALL}"
            )

        # Print bids
        for price, quantity in bids:
            bar_length = min(int(quantity * 100), 40)  # Scale the bar
            bar = "█" * bar_length
            print(
                f"{Fore.GREEN}BID: {price:.8f} | {quantity:.6f} {bar}{Style.RESET_ALL}"
            )

    def print_trade_notification(self, trade: Dict):
        """
        Print trade notification

        Args:
            trade: Dictionary with trade details
        """
        side = trade.get("side", "")
        symbol = trade.get("symbol", "")
        quantity = trade.get("quantity", 0)
        price = trade.get("price", 0)
        pnl = trade.get("pnl", 0)

        side_color = Fore.GREEN if side.lower() == "buy" else Fore.RED
        pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED

        print(f"\n{Fore.YELLOW}{'═' * 40} TRADE EXECUTED {'═' * 40}{Style.RESET_ALL}")
        print(
            f"{side_color}{side.upper()}{Style.RESET_ALL} {symbol} | "
            f"Quantity: {quantity} | "
            f"Price: {price:.8f} | "
            f"PnL: {pnl_color}{pnl:.8f}{Style.RESET_ALL}"
        )
        print(f"{Fore.YELLOW}{'═' * 100}{Style.RESET_ALL}")

    def print_error(self, error_message: str):
        """
        Print error message

        Args:
            error_message: Error message to print
        """
        print(f"\n{Fore.RED}ERROR: {error_message}{Style.RESET_ALL}")

    def print_warning(self, warning_message: str):
        """
        Print warning message

        Args:
            warning_message: Warning message to print
        """
        print(f"\n{Fore.YELLOW}WARNING: {warning_message}{Style.RESET_ALL}")

    def print_info(self, info_message: str):
        """
        Print info message

        Args:
            info_message: Info message to print
        """
        print(f"\n{Fore.CYAN}INFO: {info_message}{Style.RESET_ALL}")

    def print_strategy_signals(self, signals: Dict):
        """
        Print strategy signals

        Args:
            signals: Dictionary of strategy signals
        """
        if not signals:
            return

        print(f"\n{Fore.CYAN}Strategy Signals:{Style.RESET_ALL}")

        for indicator, value in signals.items():
            if isinstance(value, bool):
                color = Fore.GREEN if value else Fore.RED
                print(f"{indicator}: {color}{value}{Style.RESET_ALL}")
            elif isinstance(value, (int, float)):
                if indicator.lower() in ["rsi", "stoch"]:
                    color = (
                        Fore.GREEN
                        if value < 30
                        else (Fore.RED if value > 70 else Fore.WHITE)
                    )
                elif "momentum" in indicator.lower():
                    color = Fore.GREEN if value > 0 else Fore.RED
                else:
                    color = Fore.WHITE
                print(f"{indicator}: {color}{value:.4f}{Style.RESET_ALL}")
            else:
                print(f"{indicator}: {value}")

        print()

    def print_risk_assessment(self, risk_data: Dict):
        """
        Print risk assessment

        Args:
            risk_data: Dictionary of risk assessment data
        """
        if not risk_data:
            return

        print(f"\n{Fore.CYAN}Risk Assessment:{Style.RESET_ALL}")

        # Extract risk metrics
        max_position_size = risk_data.get("max_position_size", 0)
        current_exposure = risk_data.get("current_exposure", 0)
        exposure_pct = (
            (current_exposure / max_position_size) * 100 if max_position_size > 0 else 0
        )

        daily_loss_limit = risk_data.get("daily_loss_limit", 0)
        current_daily_loss = risk_data.get("current_daily_loss", 0)
        loss_limit_pct = (
            (current_daily_loss / daily_loss_limit) * 100
            if daily_loss_limit != 0
            else 0
        )

        volatility = risk_data.get("volatility", 0)
        market_risk = risk_data.get("market_risk", "normal")

        # Format with colors
        exposure_color = (
            Fore.GREEN
            if exposure_pct < 50
            else (Fore.YELLOW if exposure_pct < 80 else Fore.RED)
        )
        loss_color = (
            Fore.GREEN
            if loss_limit_pct < 50
            else (Fore.YELLOW if loss_limit_pct < 80 else Fore.RED)
        )

        market_risk_color = Fore.GREEN
        if market_risk.lower() == "high":
            market_risk_color = Fore.RED
        elif market_risk.lower() == "medium":
            market_risk_color = Fore.YELLOW

        # Print metrics
        print(
            f"Exposure: {exposure_color}{exposure_pct:.2f}%{Style.RESET_ALL} of max ({current_exposure:.8f}/{max_position_size:.8f})"
        )
        print(
            f"Daily Loss: {loss_color}{loss_limit_pct:.2f}%{Style.RESET_ALL} of limit ({current_daily_loss:.8f}/{daily_loss_limit:.8f})"
        )
        print(
            f"Volatility: {volatility:.4f}  |  Market Risk: {market_risk_color}{market_risk}{Style.RESET_ALL}"
        )

        # Print risk warnings if any
        if "warnings" in risk_data and risk_data["warnings"]:
            print(f"\n{Fore.YELLOW}Risk Warnings:{Style.RESET_ALL}")
            for warning in risk_data["warnings"]:
                print(f"- {warning}")

        print()
