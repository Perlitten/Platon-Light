#!/usr/bin/env python
"""
Unit tests for advanced technical indicators.

This module contains comprehensive tests for advanced indicators such as
Ichimoku Cloud, VWAP, adaptive indicators, and multi-timeframe indicators.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import indicators to test
from platon_light.indicators.base import BaseIndicator
from platon_light.indicators.advanced import (
    IchimokuCloud, VWAP, HeikinAshi, MultiTimeframeIndicator, 
    VolatilityAdjustedSMA, MarketRegimeIndicator
)
from platon_light.indicators.basic import SMA, RSI


class TestIndicatorBase(unittest.TestCase):
    """Base class for indicator tests with common setup."""
    
    def setUp(self):
        """Set up test data for indicator testing."""
        # Create date range
        self.dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Create sample price data with known patterns
        # This creates a sine wave pattern with a trend for easier validation
        t = np.linspace(0, 4*np.pi, 200)
        trend = np.linspace(0, 30, 200)
        close_prices = 100 + trend + 10 * np.sin(t)
        
        # Add some noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 1, 200)
        close_prices = close_prices + noise
        
        # Create OHLC data
        self.data = pd.DataFrame({
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
            'volume': np.random.normal(1000000, 200000, 200)
        }, index=self.dates)
        
        # Create some missing values for edge case testing
        self.data_with_gaps = self.data.copy()
        self.data_with_gaps.loc[self.dates[10:15], :] = np.nan
        
        # Create empty dataset for edge case testing
        self.empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Create multi-timeframe data
        # This simulates data at different timeframes
        self.hourly_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 24*30),
            'high': np.random.normal(102, 5, 24*30),
            'low': np.random.normal(98, 5, 24*30),
            'close': np.random.normal(101, 5, 24*30),
            'volume': np.random.normal(1000000, 200000, 24*30)
        }, index=pd.date_range(start='2023-01-01', periods=24*30, freq='H'))


class TestIchimokuCloud(TestIndicatorBase):
    """Test cases for the Ichimoku Cloud indicator."""
    
    def test_ichimoku_calculation(self):
        """Test Ichimoku Cloud calculation with various parameters."""
        # Test with default parameters
        ichimoku = IchimokuCloud(tenkan_period=9, kijun_period=26, 
                                senkou_b_period=52, displacement=26)
        result = ichimoku(self.data)
        
        # Check that output columns exist
        self.assertIn('tenkan_sen', result.columns)
        self.assertIn('kijun_sen', result.columns)
        self.assertIn('senkou_span_a', result.columns)
        self.assertIn('senkou_span_b', result.columns)
        self.assertIn('chikou_span', result.columns)
        
        # Check that values are not NaN after sufficient data
        self.assertTrue(result['tenkan_sen'].iloc[9:].notna().all())
        self.assertTrue(result['kijun_sen'].iloc[26:].notna().all())
        
        # Manually calculate Tenkan-sen for verification
        high_9 = self.data['high'].rolling(window=9).max()
        low_9 = self.data['low'].rolling(window=9).min()
        expected_tenkan = (high_9 + low_9) / 2
        
        # Check that values match expected
        pd.testing.assert_series_equal(result['tenkan_sen'], expected_tenkan)
        
        # Test with different parameters
        ichimoku = IchimokuCloud(tenkan_period=7, kijun_period=22, 
                                senkou_b_period=44, displacement=22)
        result = ichimoku(self.data)
        
        # Check that values are calculated correctly
        high_7 = self.data['high'].rolling(window=7).max()
        low_7 = self.data['low'].rolling(window=7).min()
        expected_tenkan = (high_7 + low_7) / 2
        
        pd.testing.assert_series_equal(result['tenkan_sen'], expected_tenkan)
    
    def test_ichimoku_properties(self):
        """Test Ichimoku Cloud properties."""
        ichimoku = IchimokuCloud()
        result = ichimoku(self.data)
        
        # Check that Senkou Span A and B are shifted forward by displacement
        tenkan = result['tenkan_sen']
        kijun = result['kijun_sen']
        senkou_a = result['senkou_span_a']
        
        # Calculate expected Senkou Span A (without displacement)
        expected_senkou_a_no_shift = (tenkan + kijun) / 2
        
        # Check that Senkou Span A is shifted version of (Tenkan + Kijun) / 2
        # Compare the values that overlap
        pd.testing.assert_series_equal(
            senkou_a.iloc[26:-26],
            expected_senkou_a_no_shift.iloc[:-26].shift(26)
        )


class TestVWAP(TestIndicatorBase):
    """Test cases for the Volume Weighted Average Price indicator."""
    
    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        # Test with default parameters
        vwap = VWAP()
        result = vwap(self.data)
        
        # Check that output column exists
        self.assertIn('VWAP', result.columns)
        
        # Check that values are not NaN
        self.assertTrue(result['VWAP'].notna().all())
        
        # Manually calculate VWAP for verification
        cumulative_pv = ((self.data['high'] + self.data['low'] + self.data['close']) / 3 * 
                         self.data['volume']).cumsum()
        cumulative_volume = self.data['volume'].cumsum()
        expected_vwap = cumulative_pv / cumulative_volume
        
        # Check that values match expected
        pd.testing.assert_series_equal(result['VWAP'], expected_vwap)
    
    def test_vwap_with_reset(self):
        """Test VWAP calculation with daily reset."""
        # Create data with multiple days
        dates = pd.date_range(start='2023-01-01', periods=48, freq='H')
        hourly_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 48),
            'high': np.random.normal(102, 5, 48),
            'low': np.random.normal(98, 5, 48),
            'close': np.random.normal(101, 5, 48),
            'volume': np.random.normal(1000000, 200000, 48)
        }, index=dates)
        
        # Test with daily reset
        vwap = VWAP(reset_period='D')
        result = vwap(hourly_data)
        
        # Check that VWAP resets at the start of each day
        # The first value of each day should be equal to the typical price
        day_starts = hourly_data.groupby(hourly_data.index.date).first().index
        for day in day_starts:
            day_str = day.strftime('%Y-%m-%d')
            first_hour = hourly_data.loc[day_str].index[0]
            typical_price = ((hourly_data.loc[first_hour, 'high'] + 
                             hourly_data.loc[first_hour, 'low'] + 
                             hourly_data.loc[first_hour, 'close']) / 3)
            self.assertAlmostEqual(
                result.loc[first_hour, 'VWAP'], 
                typical_price,
                places=10
            )


class TestHeikinAshi(TestIndicatorBase):
    """Test cases for the Heikin-Ashi indicator."""
    
    def test_heikin_ashi_calculation(self):
        """Test Heikin-Ashi calculation."""
        # Test with default parameters
        ha = HeikinAshi()
        result = ha(self.data)
        
        # Check that output columns exist
        self.assertIn('open', result.columns)
        self.assertIn('high', result.columns)
        self.assertIn('low', result.columns)
        self.assertIn('close', result.columns)
        
        # Check that values are not NaN
        self.assertTrue(result['open'].notna().all())
        self.assertTrue(result['high'].notna().all())
        self.assertTrue(result['low'].notna().all())
        self.assertTrue(result['close'].notna().all())
        
        # Check first candle
        # First HA candle should be the same as the regular candle
        self.assertEqual(result['open'].iloc[0], self.data['open'].iloc[0])
        self.assertEqual(result['high'].iloc[0], self.data['high'].iloc[0])
        self.assertEqual(result['low'].iloc[0], self.data['low'].iloc[0])
        self.assertEqual(result['close'].iloc[0], self.data['close'].iloc[0])
        
        # Check second candle
        # HA Close = (Open + High + Low + Close) / 4
        expected_close = (self.data['open'].iloc[1] + 
                         self.data['high'].iloc[1] + 
                         self.data['low'].iloc[1] + 
                         self.data['close'].iloc[1]) / 4
        self.assertEqual(result['close'].iloc[1], expected_close)
        
        # HA Open = (Previous HA Open + Previous HA Close) / 2
        expected_open = (result['open'].iloc[0] + result['close'].iloc[0]) / 2
        self.assertEqual(result['open'].iloc[1], expected_open)
    
    def test_heikin_ashi_properties(self):
        """Test Heikin-Ashi properties."""
        ha = HeikinAshi()
        result = ha(self.data)
        
        # Check that high is always the maximum of (high, open, close)
        for i in range(len(result)):
            self.assertEqual(
                result['high'].iloc[i],
                max(self.data['high'].iloc[i], result['open'].iloc[i], result['close'].iloc[i])
            )
        
        # Check that low is always the minimum of (low, open, close)
        for i in range(len(result)):
            self.assertEqual(
                result['low'].iloc[i],
                min(self.data['low'].iloc[i], result['open'].iloc[i], result['close'].iloc[i])
            )


class TestMultiTimeframeIndicator(TestIndicatorBase):
    """Test cases for the Multi-Timeframe Indicator."""
    
    def test_mtf_indicator_calculation(self):
        """Test Multi-Timeframe Indicator calculation."""
        # Create a multi-timeframe RSI indicator
        base_rsi = RSI(period=14)
        mtf_rsi = MultiTimeframeIndicator(
            base_indicator=base_rsi,
            timeframes=['1h', '4h', '1d']
        )
        
        # Test with hourly data
        result = mtf_rsi(self.hourly_data)
        
        # Check that output columns exist
        self.assertIn('1h', result.columns)
        self.assertIn('4h', result.columns)
        self.assertIn('1d', result.columns)
        
        # Check that values are not all NaN after sufficient data
        self.assertTrue(result['1h'].iloc[14:].notna().any())
        self.assertTrue(result['4h'].iloc[14:].notna().any())
        self.assertTrue(result['1d'].iloc[24:].notna().any())
        
        # Check that 1h values match the base indicator applied to 1h data
        base_result = base_rsi(self.hourly_data)
        pd.testing.assert_series_equal(result['1h'], base_result['RSI_14'])


class TestVolatilityAdjustedSMA(TestIndicatorBase):
    """Test cases for the Volatility-Adjusted SMA indicator."""
    
    def test_volatility_adjusted_sma_calculation(self):
        """Test Volatility-Adjusted SMA calculation."""
        # Test with default parameters
        vol_sma = VolatilityAdjustedSMA(
            base_period=20,
            volatility_window=100,
            min_period=5,
            max_period=50
        )
        result = vol_sma(self.data)
        
        # Check that output column exists
        self.assertIn('VolAdjSMA_20', result.columns)
        
        # Check that values are not all NaN after sufficient data
        self.assertTrue(result['VolAdjSMA_20'].iloc[100:].notna().all())
        
        # Check that periods adapt to volatility
        # In periods of high volatility, the period should be shorter
        # In periods of low volatility, the period should be longer
        
        # Calculate historical volatility
        returns = self.data['close'].pct_change()
        historical_vol = returns.rolling(window=100).std()
        
        # Find periods of high and low volatility
        high_vol_periods = historical_vol.nlargest(10).index
        low_vol_periods = historical_vol.nsmallest(10).index
        
        # Get the SMA values for these periods
        high_vol_sma = result.loc[high_vol_periods, 'VolAdjSMA_20']
        low_vol_sma = result.loc[low_vol_periods, 'VolAdjSMA_20']
        
        # Calculate regular SMAs with different periods for comparison
        sma_5 = self.data['close'].rolling(window=5).mean()
        sma_50 = self.data['close'].rolling(window=50).mean()
        
        # Check that high volatility SMA is closer to short-period SMA
        high_vol_diff_short = (high_vol_sma - sma_5.loc[high_vol_periods]).abs().mean()
        high_vol_diff_long = (high_vol_sma - sma_50.loc[high_vol_periods]).abs().mean()
        self.assertLess(high_vol_diff_short, high_vol_diff_long)
        
        # Check that low volatility SMA is closer to long-period SMA
        low_vol_diff_short = (low_vol_sma - sma_5.loc[low_vol_periods]).abs().mean()
        low_vol_diff_long = (low_vol_sma - sma_50.loc[low_vol_periods]).abs().mean()
        self.assertLess(low_vol_diff_long, low_vol_diff_short)


class TestMarketRegimeIndicator(TestIndicatorBase):
    """Test cases for the Market Regime Indicator."""
    
    def test_market_regime_calculation(self):
        """Test Market Regime Indicator calculation."""
        # Test with default parameters
        regime = MarketRegimeIndicator(lookback_period=100)
        result = regime(self.data)
        
        # Check that output columns exist
        self.assertIn('trending', result.columns)
        self.assertIn('ranging', result.columns)
        self.assertIn('volatile', result.columns)
        
        # Check that values are not all NaN after sufficient data
        self.assertTrue(result['trending'].iloc[100:].notna().all())
        self.assertTrue(result['ranging'].iloc[100:].notna().all())
        self.assertTrue(result['volatile'].iloc[100:].notna().all())
        
        # Check that probabilities sum to 1
        sums = result['trending'] + result['ranging'] + result['volatile']
        np.testing.assert_allclose(sums.dropna().values, 1.0, rtol=1e-10)
        
        # Check that probabilities are between 0 and 1
        self.assertTrue((result['trending'].dropna() >= 0).all())
        self.assertTrue((result['trending'].dropna() <= 1).all())
        self.assertTrue((result['ranging'].dropna() >= 0).all())
        self.assertTrue((result['ranging'].dropna() <= 1).all())
        self.assertTrue((result['volatile'].dropna() >= 0).all())
        self.assertTrue((result['volatile'].dropna() <= 1).all())


if __name__ == '__main__':
    unittest.main()
