"""
Data loader module for backtesting

This module provides functionality to load historical market data for backtesting purposes.
It supports loading data from various sources including:
- Binance API (direct historical data retrieval)
- CSV files (pre-downloaded data)
- SQLite database (cached data)

The data is normalized to a standard format for use in the backtesting engine.
"""

import os
import logging
import pandas as pd
import numpy as np
import ccxt
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path


class DataLoader:
    """
    Data loader for backtesting

    This class handles loading historical market data from various sources
    and preparing it for use in the backtesting engine.

    Features:
    - Load data from Binance API
    - Load data from CSV files
    - Load data from SQLite database
    - Cache data for faster subsequent loading
    - Resample data to different timeframes
    - Normalize data format
    """

    def __init__(self, config: Dict):
        """
        Initialize the data loader

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Get data directory
        self.data_dir = Path(
            config.get("backtesting", {}).get("data_dir", "data/historical")
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache database
        self.cache_db = self.data_dir / "cache.db"
        self._init_cache_db()

        # Initialize exchange API if needed
        self._init_exchange()

        self.logger.info("Data loader initialized")

    def _init_cache_db(self):
        """Initialize the cache database"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            # Create cache table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_time INTEGER NOT NULL,
                end_time INTEGER NOT NULL,
                data BLOB NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
            )

            # Create index
            cursor.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_cache_lookup 
            ON ohlcv_cache (symbol, timeframe, start_time, end_time)
            """
            )

            conn.commit()
            conn.close()

            self.logger.debug("Cache database initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize cache database: {e}")

    def _init_exchange(self):
        """Initialize exchange API client"""
        try:
            # Check if we need to use exchange API
            use_api = self.config.get("backtesting", {}).get("use_api", False)

            if use_api:
                exchange_name = self.config.get("exchange", {}).get("name", "binance")

                # Initialize exchange client
                self.exchange = ccxt.binance(
                    {
                        "enableRateLimit": True,
                        "options": {
                            "defaultType": (
                                "future"
                                if self.config.get("trading", {}).get("mode")
                                == "futures"
                                else "spot"
                            )
                        },
                    }
                )

                self.logger.debug(f"Exchange API client initialized: {exchange_name}")
            else:
                self.exchange = None
                self.logger.debug(
                    "Exchange API client not initialized (using local data only)"
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange API client: {e}")
            self.exchange = None

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load historical market data

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info(
            f"Loading data for {symbol} ({timeframe}) from {start_date} to {end_date}"
        )

        # Convert dates to timestamps
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        # Try to load from cache first
        if use_cache:
            data = self._load_from_cache(symbol, timeframe, start_ts, end_ts)
            if data is not None and not data.empty:
                self.logger.debug(f"Loaded data from cache: {len(data)} candles")
                return data

        # Try to load from CSV
        data = self._load_from_csv(symbol, timeframe, start_date, end_date)
        if data is not None and not data.empty:
            # Cache the data for future use
            if use_cache:
                self._save_to_cache(symbol, timeframe, start_ts, end_ts, data)

            self.logger.debug(f"Loaded data from CSV: {len(data)} candles")
            return data

        # Try to load from exchange API
        if self.exchange is not None:
            data = self._load_from_api(symbol, timeframe, start_date, end_date)
            if data is not None and not data.empty:
                # Cache the data for future use
                if use_cache:
                    self._save_to_cache(symbol, timeframe, start_ts, end_ts, data)

                self.logger.debug(f"Loaded data from API: {len(data)} candles")
                return data

        # No data found
        self.logger.warning(
            f"No data found for {symbol} ({timeframe}) from {start_date} to {end_date}"
        )
        return pd.DataFrame()

    def _load_from_cache(
        self, symbol: str, timeframe: str, start_ts: int, end_ts: int
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_ts: Start timestamp (milliseconds)
            end_ts: End timestamp (milliseconds)

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            # Query cache
            cursor.execute(
                """
            SELECT data FROM ohlcv_cache
            WHERE symbol = ? AND timeframe = ? AND start_time <= ? AND end_time >= ?
            ORDER BY (ABS(start_time - ?) + ABS(end_time - ?)) ASC
            LIMIT 1
            """,
                (symbol, timeframe, start_ts, end_ts, start_ts, end_ts),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                # Deserialize data
                data_bytes = row[0]
                df = pd.read_pickle(data_bytes)

                # Filter to requested date range
                df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

                return df
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to load data from cache: {e}")
            return None

    def _save_to_cache(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        data: pd.DataFrame,
    ):
        """
        Save data to cache

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_ts: Start timestamp (milliseconds)
            end_ts: End timestamp (milliseconds)
            data: DataFrame with OHLCV data
        """
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            # Serialize data
            data_bytes = data.to_pickle()

            # Insert into cache
            cursor.execute(
                """
            INSERT INTO ohlcv_cache (
                symbol, timeframe, start_time, end_time, data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    timeframe,
                    start_ts,
                    end_ts,
                    data_bytes,
                    int(datetime.now().timestamp()),
                ),
            )

            conn.commit()
            conn.close()

            self.logger.debug(
                f"Saved data to cache: {symbol} {timeframe} {start_ts}-{end_ts}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save data to cache: {e}")

    def _load_from_csv(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            # Check if CSV file exists
            csv_path = self.data_dir / f"{symbol.lower()}_{timeframe}.csv"

            if not csv_path.exists():
                return None

            # Load CSV
            df = pd.read_csv(csv_path)

            # Check if required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                self.logger.warning(f"CSV file missing required columns: {csv_path}")
                return None

            # Convert timestamp to milliseconds if needed
            if df["timestamp"].max() < 1e12:
                df["timestamp"] = df["timestamp"] * 1000

            # Filter to requested date range
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

            # Sort by timestamp
            df = df.sort_values("timestamp")

            return df
        except Exception as e:
            self.logger.error(f"Failed to load data from CSV: {e}")
            return None

    def _load_from_api(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load data from exchange API

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            if self.exchange is None:
                return None

            # Convert timeframe to milliseconds
            timeframe_ms = self._timeframe_to_ms(timeframe)

            # Calculate number of candles
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            # Binance has a limit of 1000 candles per request
            # We need to make multiple requests if the date range is too large
            all_candles = []
            current_start = start_ts

            while current_start < end_ts:
                # Calculate end time for this batch
                batch_end = min(current_start + (1000 * timeframe_ms), end_ts)

                # Fetch candles
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_start, limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Update start time for next batch
                current_start = candles[-1][0] + timeframe_ms

                # Add a small delay to avoid rate limiting
                import time

                time.sleep(0.5)

            # Convert to DataFrame
            if all_candles:
                df = pd.DataFrame(
                    all_candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                # Filter to requested date range
                df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

                # Sort by timestamp
                df = df.sort_values("timestamp")

                # Save to CSV for future use
                csv_path = self.data_dir / f"{symbol.lower()}_{timeframe}.csv"
                df.to_csv(csv_path, index=False)

                return df
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to load data from API: {e}")
            return None

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """
        Convert timeframe to milliseconds

        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h')

        Returns:
            Timeframe in milliseconds
        """
        # Extract number and unit
        import re

        match = re.match(r"(\d+)([smhdwM])", timeframe)

        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        number, unit = match.groups()
        number = int(number)

        # Convert to milliseconds
        if unit == "s":
            return number * 1000
        elif unit == "m":
            return number * 60 * 1000
        elif unit == "h":
            return number * 60 * 60 * 1000
        elif unit == "d":
            return number * 24 * 60 * 60 * 1000
        elif unit == "w":
            return number * 7 * 24 * 60 * 60 * 1000
        elif unit == "M":
            return number * 30 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Invalid timeframe unit: {unit}")

    def resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe

        Args:
            data: DataFrame with OHLCV data
            target_timeframe: Target timeframe (e.g., '5m', '1h')

        Returns:
            Resampled DataFrame
        """
        try:
            # Convert timestamp to datetime
            df = data.copy()
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")

            # Extract timeframe components
            import re

            match = re.match(r"(\d+)([smhdwM])", target_timeframe)

            if not match:
                raise ValueError(f"Invalid timeframe format: {target_timeframe}")

            number, unit = match.groups()

            # Convert to pandas frequency string
            freq_map = {"s": "S", "m": "T", "h": "H", "d": "D", "w": "W", "M": "M"}
            freq = f"{number}{freq_map[unit]}"

            # Resample
            resampled = df.resample(freq).agg(
                {
                    "timestamp": "first",
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )

            # Reset index
            resampled = resampled.reset_index(drop=True)

            return resampled
        except Exception as e:
            self.logger.error(f"Failed to resample data: {e}")
            return data

    def download_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical data from exchange API

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            save_csv: Whether to save data to CSV

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.exchange is None:
                self.logger.error("Exchange API client not initialized")
                return pd.DataFrame()

            self.logger.info(
                f"Downloading data for {symbol} ({timeframe}) from {start_date} to {end_date}"
            )

            # Download data
            data = self._load_from_api(symbol, timeframe, start_date, end_date)

            if data is None or data.empty:
                self.logger.warning(f"No data downloaded for {symbol} ({timeframe})")
                return pd.DataFrame()

            # Save to cache
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            self._save_to_cache(symbol, timeframe, start_ts, end_ts, data)

            # Save to CSV
            if save_csv:
                csv_path = self.data_dir / f"{symbol.lower()}_{timeframe}.csv"
                data.to_csv(csv_path, index=False)
                self.logger.info(f"Saved data to {csv_path}")

            return data
        except Exception as e:
            self.logger.error(f"Failed to download data: {e}")
            return pd.DataFrame()

    def prepare_data(
        self, data: pd.DataFrame, add_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting

        Args:
            data: DataFrame with OHLCV data
            add_indicators: Whether to add technical indicators

        Returns:
            Prepared DataFrame
        """
        try:
            if data.empty:
                return data

            df = data.copy()

            # Ensure data is sorted by timestamp
            df = df.sort_values("timestamp")

            # Add datetime column
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Add basic price change columns
            df["price_change"] = df["close"].pct_change()
            df["price_change_abs"] = df["close"].diff()

            # Add basic volume change columns
            df["volume_change"] = df["volume"].pct_change()

            # Add indicators if requested
            if add_indicators:
                # Import indicators module
                import sys
                import os

                # Add parent directory to path if needed
                parent_dir = str(Path(__file__).parent.parent)
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

                from platon_light.utils.indicators import calculate_indicators

                # Get indicator settings from config
                indicator_config = self.config.get("strategy", {})

                # Calculate indicators
                df = calculate_indicators(df, indicator_config)

            return df
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            return data
