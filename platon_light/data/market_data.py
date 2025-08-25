"""
Market data manager for collecting and processing market data
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from collections import deque

from platon_light.core.base_exchange import BaseExchangeClient


class MarketDataManager:
    """
    Market data manager for collecting and processing real-time market data
    including price, volume, order book, and other market metrics
    """

    def __init__(self, config: Dict, exchange: BaseExchangeClient):
        """
        Initialize the market data manager

        Args:
            config: Bot configuration
            exchange: Exchange client instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchange = exchange

        # Data storage
        self.ohlcv_data = {}  # symbol -> timeframe -> data
        self.order_books = {}  # symbol -> order_book
        self.latest_data = {}  # symbol -> latest_data
        self.trade_history = {}  # symbol -> trade_history

        # Timeframes to collect data for
        self.timeframes = config["trading"]["timeframes"]

        # Convert string timeframes to seconds
        self.timeframe_seconds = {}
        for tf in self.timeframes:
            self.timeframe_seconds[tf] = self._timeframe_to_seconds(tf)

        # Order book settings
        self.order_book_depth = (
            config.get("advanced", {})
            .get("market_data", {})
            .get("order_book_depth", 20)
        )

        # Performance metrics
        self.win_rates = {}  # symbol -> win_rate

        # Market data update tasks
        self.tasks = []
        self.is_running = False

        # Custom timeframes for high-frequency data
        self.custom_candles = {}  # symbol -> timeframe -> deque of candles
        self.last_tick = {}  # symbol -> last_tick_time

        self.logger.info("Market data manager initialized")

    async def initialize(self, symbols: List[str]):
        """
        Initialize market data collection for a list of symbols

        Args:
            symbols: List of trading pair symbols
        """
        self.logger.info(f"Initializing market data for {len(symbols)} symbols")
        self.is_running = True

        # Initialize data structures
        for symbol in symbols:
            self.ohlcv_data[symbol] = {}
            self.order_books[symbol] = None
            self.latest_data[symbol] = {}
            self.trade_history[symbol] = []
            self.last_tick[symbol] = 0
            self.custom_candles[symbol] = {}

            # Initialize custom timeframes
            for tf in self.timeframes:
                if self._is_custom_timeframe(tf):
                    self.custom_candles[symbol][tf] = deque(maxlen=500)

        # Start data collection tasks
        for symbol in symbols:
            # Initial data load
            await self._load_initial_data(symbol)

            # Start continuous data collection
            self.tasks.append(asyncio.create_task(self._collect_market_data(symbol)))

        self.logger.info("Market data initialization complete")

    async def stop(self):
        """Stop all market data collection tasks"""
        self.logger.info("Stopping market data collection")
        self.is_running = False

        for task in self.tasks:
            if not task.done():
                task.cancel()

        self.logger.info("Market data collection stopped")

    async def _load_initial_data(self, symbol: str):
        """
        Load initial historical data for a symbol

        Args:
            symbol: Trading pair symbol
        """
        self.logger.info(f"Loading initial data for {symbol}")

        # Load OHLCV data for each timeframe
        for tf in self.timeframes:
            if not self._is_custom_timeframe(tf):
                # Convert to Binance interval format
                interval = self._to_binance_interval(tf)

                # Load historical klines
                klines = await self.exchange.get_klines(symbol, interval, limit=500)

                if klines:
                    # Format data
                    formatted_data = []
                    for k in klines:
                        formatted_data.append(
                            [
                                k[0],  # timestamp
                                k[1],  # open
                                k[2],  # high
                                k[3],  # low
                                k[4],  # close
                                k[5],  # volume
                            ]
                        )

                    self.ohlcv_data[symbol][tf] = formatted_data
                    self.logger.debug(
                        f"Loaded {len(formatted_data)} {tf} candles for {symbol}"
                    )
                else:
                    self.logger.warning(f"Failed to load {tf} data for {symbol}")

        # Load initial order book
        order_book = await self.exchange.get_order_book(symbol, self.order_book_depth)
        if order_book:
            self.order_books[symbol] = order_book
            self.logger.debug(f"Loaded order book for {symbol}")

        # Get current price
        price = await self.exchange.get_ticker_price(symbol)
        if price:
            self.latest_data[symbol]["price"] = price
            self.latest_data[symbol]["timestamp"] = int(time.time() * 1000)

    async def _collect_market_data(self, symbol: str):
        """
        Continuously collect market data for a symbol

        Args:
            symbol: Trading pair symbol
        """
        self.logger.info(f"Starting market data collection for {symbol}")

        update_intervals = {
            "price": 1,  # 1 second
            "order_book": 5,  # 5 seconds
            "ohlcv": 60,  # 60 seconds
        }

        last_updates = {"price": 0, "order_book": 0, "ohlcv": 0}

        while self.is_running:
            try:
                current_time = time.time()

                # Update price
                if current_time - last_updates["price"] >= update_intervals["price"]:
                    price = await self.exchange.get_ticker_price(symbol)
                    if price:
                        # Update latest data
                        self.latest_data[symbol]["price"] = price
                        self.latest_data[symbol]["timestamp"] = int(current_time * 1000)

                        # Update custom timeframe candles
                        await self._update_custom_candles(
                            symbol, price, int(current_time * 1000)
                        )

                    last_updates["price"] = current_time

                # Update order book
                if (
                    current_time - last_updates["order_book"]
                    >= update_intervals["order_book"]
                ):
                    order_book = await self.exchange.get_order_book(
                        symbol, self.order_book_depth
                    )
                    if order_book:
                        self.order_books[symbol] = order_book
                        self.latest_data[symbol]["order_book"] = order_book

                    last_updates["order_book"] = current_time

                # Update OHLCV data
                if current_time - last_updates["ohlcv"] >= update_intervals["ohlcv"]:
                    for tf in self.timeframes:
                        if not self._is_custom_timeframe(tf):
                            # Convert to Binance interval format
                            interval = self._to_binance_interval(tf)

                            # Load latest klines
                            klines = await self.exchange.get_klines(
                                symbol, interval, limit=100
                            )

                            if klines:
                                # Format data
                                formatted_data = []
                                for k in klines:
                                    formatted_data.append(
                                        [
                                            k[0],  # timestamp
                                            k[1],  # open
                                            k[2],  # high
                                            k[3],  # low
                                            k[4],  # close
                                            k[5],  # volume
                                        ]
                                    )

                                self.ohlcv_data[symbol][tf] = formatted_data

                    last_updates["ohlcv"] = current_time

                # Small delay to prevent high CPU usage
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                self.logger.info(f"Market data collection for {symbol} cancelled")
                break

            except Exception as e:
                self.logger.error(f"Error collecting market data for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _update_custom_candles(self, symbol: str, price: float, timestamp: int):
        """
        Update custom timeframe candles with new price data

        Args:
            symbol: Trading pair symbol
            price: Current price
            timestamp: Current timestamp in milliseconds
        """
        # Skip if no custom timeframes
        if not any(self._is_custom_timeframe(tf) for tf in self.timeframes):
            return

        # Get last tick time
        last_tick = self.last_tick.get(symbol, 0)

        # Skip if no price change
        if last_tick == timestamp:
            return

        self.last_tick[symbol] = timestamp

        # Update each custom timeframe
        for tf in self.timeframes:
            if self._is_custom_timeframe(tf):
                tf_seconds = self.timeframe_seconds[tf]
                candles = self.custom_candles[symbol][tf]

                # Create new candle if needed
                if not candles or timestamp - candles[-1][0] >= tf_seconds * 1000:
                    # New candle
                    candle = [timestamp, price, price, price, price, 0]
                    candles.append(candle)
                else:
                    # Update existing candle
                    candle = candles[-1]
                    candle[2] = max(candle[2], price)  # high
                    candle[3] = min(candle[3], price)  # low
                    candle[4] = price  # close
                    candle[5] += 1  # increment volume (tick count)

    def get_latest_data(self, symbol: str) -> Dict:
        """
        Get latest market data for a symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with latest market data
        """
        if symbol not in self.latest_data:
            return {}

        data = self.latest_data[symbol].copy()

        # Add order book
        if symbol in self.order_books and self.order_books[symbol]:
            data["order_book"] = self.order_books[symbol]

        # Add latest candle
        for tf in self.timeframes:
            if (
                symbol in self.ohlcv_data
                and tf in self.ohlcv_data[symbol]
                and self.ohlcv_data[symbol][tf]
            ):
                data[f"ohlcv_{tf}"] = self.ohlcv_data[symbol][tf][-1]

        return data

    def get_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> Optional[List]:
        """
        Get OHLCV data for a symbol and timeframe

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., "1m", "5m", "15s")
            limit: Maximum number of candles to return

        Returns:
            List of OHLCV candles or None if not available
        """
        # Check if data is available
        if symbol not in self.ohlcv_data:
            return None

        # For custom timeframes, get from custom candles
        if self._is_custom_timeframe(timeframe):
            if timeframe not in self.custom_candles.get(symbol, {}):
                return None

            # Convert deque to list and limit
            candles = list(self.custom_candles[symbol][timeframe])
            return candles[-limit:] if limit < len(candles) else candles

        # For standard timeframes, get from OHLCV data
        if timeframe not in self.ohlcv_data[symbol]:
            return None

        candles = self.ohlcv_data[symbol][timeframe]
        return candles[-limit:] if limit < len(candles) else candles

    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Get order book for a symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Order book dictionary or None if not available
        """
        return self.order_books.get(symbol)

    def get_win_rate(self, symbol: str) -> Optional[float]:
        """
        Get historical win rate for a symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Win rate as a float between 0 and 1, or None if not available
        """
        return self.win_rates.get(symbol)

    def update_win_rate(self, symbol: str, win: bool):
        """
        Update win rate for a symbol

        Args:
            symbol: Trading pair symbol
            win: Whether the trade was a win
        """
        if symbol not in self.win_rates:
            self.win_rates[symbol] = 0.5  # Initial win rate

        # Simple exponential moving average
        alpha = 0.1  # Weight for new data
        self.win_rates[symbol] = (
            self.win_rates[symbol] * (1 - alpha) + (1 if win else 0) * alpha
        )

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to seconds

        Args:
            timeframe: Timeframe string (e.g., "1m", "5m", "15s")

        Returns:
            Number of seconds
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            self.logger.warning(f"Unknown timeframe unit: {unit}")
            return 60  # Default to 1 minute

    def _is_custom_timeframe(self, timeframe: str) -> bool:
        """
        Check if a timeframe is a custom timeframe (not supported by Binance API)

        Args:
            timeframe: Timeframe string

        Returns:
            True if custom timeframe, False otherwise
        """
        # Binance supports 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        # Anything else is custom
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit == "s":
            return True  # All second-based timeframes are custom

        if unit == "m":
            return value not in [1, 3, 5, 15, 30]

        if unit == "h":
            return value not in [1, 2, 4, 6, 8, 12]

        return False

    def _to_binance_interval(self, timeframe: str) -> str:
        """
        Convert timeframe to Binance interval format

        Args:
            timeframe: Timeframe string

        Returns:
            Binance interval string
        """
        # If custom timeframe, use closest standard timeframe
        if self._is_custom_timeframe(timeframe):
            seconds = self._timeframe_to_seconds(timeframe)

            if seconds < 60:
                return "1m"  # Use 1m for anything less than 1 minute
            elif seconds < 180:
                return "1m"
            elif seconds < 300:
                return "3m"
            elif seconds < 900:
                return "5m"
            elif seconds < 1800:
                return "15m"
            elif seconds < 3600:
                return "30m"
            else:
                return "1h"  # Use 1h for anything 1 hour or more

        # For standard timeframes, just return as is
        return timeframe
