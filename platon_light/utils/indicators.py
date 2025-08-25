"""
Technical indicators for trading strategy
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def calculate_rsi(prices: pd.Series, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI)

    Args:
        prices: Series of prices
        period: RSI period

    Returns:
        Array of RSI values
    """
    # Convert to numpy array if pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculate price changes
    deltas = np.diff(prices)
    seed = deltas[: period + 1]

    # Calculate gains and losses
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period

    if down == 0:
        return np.ones_like(prices) * 100

    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    # Calculate RSI
    for i in range(period, len(prices)):
        delta = deltas[i - 1]

        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        rs = up / down if down != 0 else float("inf")
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD)

    Args:
        prices: Series of prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period

    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    # Convert to numpy array if pandas Series
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculate EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA)

    Args:
        prices: Array of prices
        period: EMA period

    Returns:
        Array of EMA values
    """
    ema = np.zeros_like(prices)

    # Start with SMA
    ema[:period] = np.mean(prices[:period])

    # Calculate multiplier
    multiplier = 2.0 / (period + 1)

    # Calculate EMA
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

    return ema


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA)

    Args:
        prices: Array of prices
        period: SMA period

    Returns:
        Array of SMA values
    """
    sma = np.zeros_like(prices)

    # Calculate SMA
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1 : i + 1])

    return sma


def calculate_bollinger_bands(
    prices: np.ndarray, period: int = 20, num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands

    Args:
        prices: Array of prices
        period: SMA period
        num_std: Number of standard deviations

    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    # Calculate middle band (SMA)
    middle_band = calculate_sma(prices, period)

    # Calculate standard deviation
    std = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1 : i + 1])

    # Calculate upper and lower bands
    upper_band = middle_band + num_std * std
    lower_band = middle_band - num_std * std

    return upper_band, middle_band, lower_band


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        k_period: %K period
        d_period: %D period
        slowing: Slowing period

    Returns:
        Tuple of (%K, %D)
    """
    # Convert to numpy arrays if pandas Series
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Calculate %K
    k = np.zeros_like(close)

    for i in range(k_period - 1, len(close)):
        highest_high = np.max(high[i - k_period + 1 : i + 1])
        lowest_low = np.min(low[i - k_period + 1 : i + 1])

        if highest_high == lowest_low:
            k[i] = 50.0
        else:
            k[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)

    # Apply slowing if specified
    if slowing > 1:
        k_slowed = np.zeros_like(close)
        for i in range(k_period + slowing - 2, len(close)):
            k_slowed[i] = np.mean(k[i - slowing + 1 : i + 1])
    else:
        k_slowed = k

    # Calculate %D (SMA of %K)
    d = calculate_sma(k_slowed, d_period)

    return k_slowed, d


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR)

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: ATR period

    Returns:
        Array of ATR values
    """
    # Convert to numpy arrays if pandas Series
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Calculate true range
    tr = np.zeros_like(close)

    # First TR value is high - low
    tr[0] = high[0] - low[0]

    # Calculate subsequent TR values
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    # Calculate ATR
    atr = np.zeros_like(close)

    # First ATR value is average of first 'period' TR values
    atr[period - 1] = np.mean(tr[:period])

    # Calculate subsequent ATR values using smoothing
    for i in range(period, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Average Directional Index (ADX)

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: ADX period

    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    # Convert to numpy arrays if pandas Series
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Calculate true range
    tr = np.zeros_like(close)

    # First TR value is high - low
    tr[0] = high[0] - low[0]

    # Calculate subsequent TR values
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    # Calculate directional movement
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)

    for i in range(1, len(close)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Calculate smoothed values
    tr_period = calculate_ema(tr, period)
    plus_di_period = 100 * calculate_ema(plus_dm, period) / tr_period
    minus_di_period = 100 * calculate_ema(minus_dm, period) / tr_period

    # Calculate directional index
    dx = (
        100
        * np.abs(plus_di_period - minus_di_period)
        / (plus_di_period + minus_di_period)
    )

    # Calculate ADX
    adx = calculate_ema(dx, period)

    return adx, plus_di_period, minus_di_period


def calculate_volume_profile(
    prices: np.ndarray, volumes: np.ndarray, num_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Volume Profile

    Args:
        prices: Array of prices
        volumes: Array of volumes
        num_bins: Number of price bins

    Returns:
        Tuple of (price levels, volume at each level)
    """
    # Calculate price range
    min_price = np.min(prices)
    max_price = np.max(prices)

    # Create price bins
    price_bins = np.linspace(min_price, max_price, num_bins + 1)

    # Initialize volume profile
    volume_profile = np.zeros(num_bins)

    # Calculate volume for each price bin
    for i in range(len(prices)):
        # Find bin index for current price
        bin_index = np.digitize(prices[i], price_bins) - 1

        # Ensure bin index is valid
        if 0 <= bin_index < num_bins:
            volume_profile[bin_index] += volumes[i]

    # Calculate price levels (middle of each bin)
    price_levels = (price_bins[:-1] + price_bins[1:]) / 2

    return price_levels, volume_profile


def calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price (VWAP)

    Args:
        prices: Array of prices
        volumes: Array of volumes

    Returns:
        Array of VWAP values
    """
    # Calculate cumulative price * volume
    cum_pv = np.cumsum(prices * volumes)

    # Calculate cumulative volume
    cum_vol = np.cumsum(volumes)

    # Calculate VWAP
    vwap = cum_pv / cum_vol

    return vwap


def calculate_ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Ichimoku Cloud

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        tenkan_period: Tenkan-sen (Conversion Line) period
        kijun_period: Kijun-sen (Base Line) period
        senkou_span_b_period: Senkou Span B period
        displacement: Displacement for Senkou Span A and B

    Returns:
        Tuple of (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
    """
    # Convert to numpy arrays if pandas Series
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Initialize arrays
    tenkan_sen = np.zeros_like(close)
    kijun_sen = np.zeros_like(close)
    senkou_span_a = np.zeros_like(close)
    senkou_span_b = np.zeros_like(close)
    chikou_span = np.zeros_like(close)

    # Calculate Tenkan-sen (Conversion Line)
    for i in range(tenkan_period - 1, len(close)):
        tenkan_sen[i] = (
            np.max(high[i - tenkan_period + 1 : i + 1])
            + np.min(low[i - tenkan_period + 1 : i + 1])
        ) / 2

    # Calculate Kijun-sen (Base Line)
    for i in range(kijun_period - 1, len(close)):
        kijun_sen[i] = (
            np.max(high[i - kijun_period + 1 : i + 1])
            + np.min(low[i - kijun_period + 1 : i + 1])
        ) / 2

    # Calculate Senkou Span A (Leading Span A)
    for i in range(kijun_period - 1, len(close)):
        senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2

    # Calculate Senkou Span B (Leading Span B)
    for i in range(senkou_span_b_period - 1, len(close)):
        senkou_span_b[i] = (
            np.max(high[i - senkou_span_b_period + 1 : i + 1])
            + np.min(low[i - senkou_span_b_period + 1 : i + 1])
        ) / 2

    # Calculate Chikou Span (Lagging Span)
    chikou_span[:-displacement] = close[displacement:]

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
