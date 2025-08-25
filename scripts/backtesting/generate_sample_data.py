#!/usr/bin/env python
"""
Generate sample OHLCV data for backtesting.

This script generates synthetic price data that resembles cryptocurrency market data
with realistic price movements, volatility, and volume patterns.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_sample_data(symbol="BTCUSDT", timeframe="1h", days=365, start_price=10000):
    """
    Generate synthetic OHLCV data for backtesting.

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe (e.g., '1h', '4h', '1d')
        days: Number of days of data to generate
        start_price: Starting price

    Returns:
        DataFrame with OHLCV data
    """
    # Determine number of periods based on timeframe
    if timeframe == "1h":
        periods = days * 24
        freq = "H"
    elif timeframe == "4h":
        periods = days * 6
        freq = "4H"
    elif timeframe == "1d":
        periods = days
        freq = "D"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Generate price data
    np.random.seed(42)  # For reproducibility

    # Parameters for the random walk
    drift = 0.0001  # Small upward drift
    volatility = 0.02  # Daily volatility

    # Adjust volatility based on timeframe
    if timeframe == "1h":
        volatility /= np.sqrt(24)
    elif timeframe == "4h":
        volatility /= np.sqrt(6)

    # Generate log returns
    log_returns = np.random.normal(drift, volatility, periods)

    # Add some autocorrelation and trends
    for i in range(1, len(log_returns)):
        # Add some momentum (autocorrelation)
        log_returns[i] += 0.1 * log_returns[i - 1]

        # Add some longer trends
        if i % 100 == 0:
            trend_direction = np.random.choice([-1, 1])
            trend_strength = np.random.uniform(0.001, 0.003)
            for j in range(i, min(i + 100, len(log_returns))):
                log_returns[j] += trend_direction * trend_strength

    # Convert log returns to price series
    prices = start_price * np.exp(np.cumsum(log_returns))

    # Generate OHLCV data
    data = []
    for i, timestamp in enumerate(timestamps[: len(prices)]):
        price = prices[i]

        # Generate intraperiod volatility
        intraperiod_vol = volatility * price * 2

        # Generate OHLC
        open_price = price
        high_price = price + np.random.uniform(0, intraperiod_vol)
        low_price = price - np.random.uniform(0, intraperiod_vol)

        # Ensure low <= open <= high
        low_price = min(low_price, open_price)
        high_price = max(high_price, open_price)

        # Generate close with some mean reversion
        close_price = price + np.random.uniform(
            -intraperiod_vol / 2, intraperiod_vol / 2
        )

        # Ensure low <= close <= high
        close_price = max(low_price, min(close_price, high_price))

        # Generate volume with some correlation to price movement
        price_change = abs(close_price - open_price)
        base_volume = np.random.uniform(100, 1000)
        volume = base_volume * (1 + 5 * price_change / price)

        # Add some volume spikes
        if np.random.random() < 0.05:  # 5% chance of volume spike
            volume *= np.random.uniform(3, 10)

        data.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


def save_sample_data(df, symbol="BTCUSDT", timeframe="1h"):
    """
    Save sample data to CSV file.

    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        timeframe: Timeframe
    """
    # Create output directory
    output_dir = Path(__file__).parent / "data" / "sample"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create filename
    filename = f"{symbol}_{timeframe}.csv"
    output_path = output_dir / filename

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Sample data saved to {output_path}")
    print(f"Generated {len(df)} data points for {symbol} ({timeframe})")


def main():
    """Main function to generate sample data."""
    print("Generating sample data for backtesting...")

    # Generate data for different timeframes
    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["1h", "4h", "1d"]

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"Generating {symbol} {timeframe} data...")
            df = generate_sample_data(symbol=symbol, timeframe=timeframe)
            save_sample_data(df, symbol=symbol, timeframe=timeframe)

    print("Sample data generation completed.")


if __name__ == "__main__":
    main()
