#!/usr/bin/env python
"""
Live backtest launcher with secure credential handling for Platon Light.

This script securely loads API credentials and launches backtests with real market data.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import dotenv
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from platon_light.backtesting.strategies.moving_average_crossover import (
    MovingAverageCrossover,
)


class SecureCredentialManager:
    """Secure credential manager for API keys."""

    def __init__(self, key_file=None, env_file=None):
        """
        Initialize the secure credential manager.

        Args:
            key_file: Path to the encryption key file
            env_file: Path to the encrypted environment file
        """
        self.key_file = key_file or Path.home() / ".platon" / "key.key"
        self.env_file = (
            env_file or Path.home() / ".platon" / "credentials.env.encrypted"
        )

        # Ensure directory exists
        self.key_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or generate key
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self):
        """Load existing key or generate a new one."""
        if self.key_file.exists():
            with open(self.key_file, "rb") as f:
                return f.read()
        else:
            # Generate a new key
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
            logger.info(f"Generated new encryption key at {self.key_file}")
            return key

    def encrypt_and_save_credentials(self, credentials):
        """
        Encrypt and save credentials.

        Args:
            credentials: Dictionary with credentials
        """
        # Convert credentials to .env format
        env_content = "\n".join([f"{k}={v}" for k, v in credentials.items()])

        # Encrypt
        encrypted_data = self.cipher.encrypt(env_content.encode())

        # Save
        with open(self.env_file, "wb") as f:
            f.write(encrypted_data)

        logger.info(f"Credentials encrypted and saved to {self.env_file}")

    def load_credentials(self):
        """
        Load and decrypt credentials.

        Returns:
            Dictionary with credentials
        """
        if not self.env_file.exists():
            logger.warning(f"Credentials file not found: {self.env_file}")
            return {}

        try:
            # Load encrypted data
            with open(self.env_file, "rb") as f:
                encrypted_data = f.read()

            # Decrypt
            decrypted_data = self.cipher.decrypt(encrypted_data).decode()

            # Parse .env format
            credentials = {}
            for line in decrypted_data.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    credentials[key] = value

            return credentials
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}

    def setup_credentials_interactive(self):
        """Set up credentials interactively."""
        print("\n=== Secure Credential Setup ===")
        print("Please enter your API credentials:")

        credentials = {}

        # Binance API credentials
        credentials["BINANCE_API_KEY"] = input("Binance API Key: ").strip()
        credentials["BINANCE_API_SECRET"] = input("Binance API Secret: ").strip()

        # Telegram credentials (optional)
        use_telegram = (
            input("Do you want to set up Telegram notifications? (y/n): ")
            .strip()
            .lower()
            == "y"
        )
        if use_telegram:
            credentials["TELEGRAM_BOT_TOKEN"] = input("Telegram Bot Token: ").strip()
            credentials["TELEGRAM_CHAT_ID"] = input("Telegram Chat ID: ").strip()

        # Encrypt and save
        self.encrypt_and_save_credentials(credentials)

        print(f"\nCredentials securely saved to {self.env_file}")
        print(f"Encryption key stored at {self.key_file}")
        print("Keep these files secure and do not share them!")


def fetch_market_data(
    symbol, timeframe, start_date, end_date, api_key=None, api_secret=None
):
    """
    Fetch market data from Binance.

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe (e.g., '1h', '4h', '1d')
        start_date: Start date
        end_date: End date
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Import ccxt only when needed
        import ccxt

        logger.info(
            f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}"
        )

        # Initialize exchange
        exchange_params = {}
        if api_key and api_secret:
            exchange_params["apiKey"] = api_key
            exchange_params["secret"] = api_secret

        exchange = ccxt.binance(exchange_params)

        # Convert timeframe to milliseconds
        timeframe_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }

        # Convert dates to timestamps
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        until = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        # Fetch data in chunks
        all_candles = []
        current = since

        while current < until:
            candles = exchange.fetch_ohlcv(symbol, timeframe, current)
            if not candles:
                break

            all_candles.extend(candles)

            # Update current timestamp
            current = candles[-1][0] + timeframe_ms.get(timeframe, 60 * 60 * 1000)

            # Respect rate limits
            exchange.sleep(exchange.rateLimit / 1000)

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        logger.info(f"Fetched {len(df)} data points")

        return df

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return None


def load_optimized_parameters(strategy_name="moving_average_crossover"):
    """
    Load optimized parameters for a strategy.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Dictionary with optimized parameters
    """
    # Find the most recent optimization results
    results_dir = Path(__file__).parent / "results"
    json_files = list(results_dir.glob("top_parameters_*.json"))

    if not json_files:
        logger.warning("No optimization results found, using default parameters")
        return {
            "fast_ma_type": "EMA",
            "slow_ma_type": "SMA",
            "fast_period": 20,
            "slow_period": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "use_filters": True,
        }

    # Sort by modification time (most recent first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    json_file = json_files[0]

    logger.info(f"Loading optimized parameters from {json_file}")

    try:
        # Load parameters from JSON
        with open(json_file, "r") as f:
            results = json.load(f)

        # Get parameters from the best result
        if results and "params" in results[0]:
            params = results[0]["params"]
            logger.info(f"Loaded optimized parameters: {params}")
            return params
        else:
            logger.warning(
                "Invalid optimization results format, using default parameters"
            )
            return {
                "fast_ma_type": "EMA",
                "slow_ma_type": "SMA",
                "fast_period": 20,
                "slow_period": 50,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "use_filters": True,
            }

    except Exception as e:
        logger.error(f"Error loading optimized parameters: {e}")
        logger.warning("Using default parameters")
        return {
            "fast_ma_type": "EMA",
            "slow_ma_type": "SMA",
            "fast_period": 20,
            "slow_period": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "use_filters": True,
        }


def run_backtest(data, strategy_params):
    """
    Run a backtest using the specified strategy parameters.

    Args:
        data: DataFrame with OHLCV data
        strategy_params: Dictionary with strategy parameters

    Returns:
        DataFrame with backtest results
    """
    logger.info(f"Running backtest with strategy parameters: {strategy_params}")

    # Create strategy instance
    strategy = MovingAverageCrossover(**strategy_params)

    # Run strategy
    result = strategy.run(data)

    logger.info(f"Backtest completed with {len(result[result['signal'] != 0])} signals")

    return result


def main():
    """Main function to launch live backtests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Launch live backtests with real market data"
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to backtest")
    parser.add_argument("--timeframe", default="1h", help="Timeframe to backtest")
    parser.add_argument(
        "--start-date",
        default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--setup-credentials", action="store_true", help="Set up API credentials"
    )
    args = parser.parse_args()

    logger.info("Starting live backtest")

    # Initialize credential manager
    credential_manager = SecureCredentialManager()

    # Set up credentials if requested
    if args.setup_credentials:
        credential_manager.setup_credentials_interactive()
        return

    # Load credentials
    credentials = credential_manager.load_credentials()

    if not credentials:
        logger.warning(
            "No credentials found. Run with --setup-credentials to set up API credentials."
        )

    # Load optimized parameters
    params = load_optimized_parameters()

    # Fetch market data
    data = fetch_market_data(
        args.symbol,
        args.timeframe,
        args.start_date,
        args.end_date,
        credentials.get("BINANCE_API_KEY"),
        credentials.get("BINANCE_API_SECRET"),
    )

    if data is None:
        logger.error("Failed to fetch market data")
        return

    # Run backtest
    result = run_backtest(data, params)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        output_dir / f"live_backtest_{args.symbol}_{args.timeframe}_{timestamp}.csv"
    )

    result.to_csv(output_file)
    logger.info(f"Backtest results saved to {output_file}")

    # Run performance analysis
    logger.info("Running performance analysis")

    # Import locally to avoid circular imports
    from analyze_backtest_performance import (
        calculate_performance_metrics,
        visualize_performance,
        generate_performance_report,
    )

    metrics, data_with_metrics = calculate_performance_metrics(result)

    # Log metrics
    logger.info("Performance metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

    # Visualize performance
    visualize_performance(data_with_metrics, metrics, output_dir)

    # Generate performance report
    report_file = generate_performance_report(data_with_metrics, metrics, output_dir)

    logger.info(f"Performance report saved to {report_file}")
    logger.info("Live backtest completed")


if __name__ == "__main__":
    main()
