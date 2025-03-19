#!/usr/bin/env python
"""
Edge case tests for technical indicators.

This module contains tests for handling various edge cases in indicator calculations,
including empty datasets, missing values, extreme values, and boundary conditions.
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
from platon_light.indicators.basic import SMA, EMA, RSI, BollingerBands, MACD
from platon_light.indicators.advanced import IchimokuCloud, VWAP, HeikinAshi


class TestEmptyDatasets(unittest.TestCase):
    """Test cases for handling empty datasets."""
    
    def setUp(self):
        """Set up empty test data."""
        # Create empty dataset
        self.empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Create dataset with only one row
        self.single_row_data = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000000]
        }, index=[pd.Timestamp('2023-01-01')])
    
    def test_empty_data_sma(self):
        """Test SMA with empty dataset."""
        sma = SMA(period=20)
        result = sma(self.empty_data)
        
        # Check that output column exists
        self.assertIn('SMA_20', result.columns)
        
        # Check that output is empty
        self.assertEqual(len(result), 0)
    
    def test_empty_data_rsi(self):
        """Test RSI with empty dataset."""
        rsi = RSI(period=14)
        result = rsi(self.empty_data)
        
        # Check that output column exists
        self.assertIn('RSI_14', result.columns)
        
        # Check that output is empty
        self.assertEqual(len(result), 0)
    
    def test_empty_data_bollinger(self):
        """Test Bollinger Bands with empty dataset."""
        bb = BollingerBands(period=20, std_dev=2)
        result = bb(self.empty_data)
        
        # Check that output columns exist
        self.assertIn('BB_20_2_middle', result.columns)
        self.assertIn('BB_20_2_upper', result.columns)
        self.assertIn('BB_20_2_lower', result.columns)
        
        # Check that output is empty
        self.assertEqual(len(result), 0)
    
    def test_empty_data_macd(self):
        """Test MACD with empty dataset."""
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd(self.empty_data)
        
        # Check that output columns exist
        self.assertIn('MACD_12_26_9_line', result.columns)
        self.assertIn('MACD_12_26_9_signal', result.columns)
        self.assertIn('MACD_12_26_9_histogram', result.columns)
        
        # Check that output is empty
        self.assertEqual(len(result), 0)
    
    def test_empty_data_ichimoku(self):
        """Test Ichimoku Cloud with empty dataset."""
        ichimoku = IchimokuCloud()
        result = ichimoku(self.empty_data)
        
        # Check that output columns exist
        self.assertIn('tenkan_sen', result.columns)
        self.assertIn('kijun_sen', result.columns)
        self.assertIn('senkou_span_a', result.columns)
        self.assertIn('senkou_span_b', result.columns)
        self.assertIn('chikou_span', result.columns)
        
        # Check that output is empty
        self.assertEqual(len(result), 0)
    
    def test_single_row_data(self):
        """Test indicators with single row dataset."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.single_row_data)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result['SMA_20'].iloc[0]))
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.single_row_data)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result['RSI_14'].iloc[0]))
        
        # Test Bollinger Bands
        bb = BollingerBands(period=20, std_dev=2)
        result = bb(self.single_row_data)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result['BB_20_2_middle'].iloc[0]))
        
        # Test MACD
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd(self.single_row_data)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result['MACD_12_26_9_line'].iloc[0]))
        
        # Test Ichimoku Cloud
        ichimoku = IchimokuCloud()
        result = ichimoku(self.single_row_data)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result['tenkan_sen'].iloc[0]))


class TestMissingValues(unittest.TestCase):
    """Test cases for handling missing values."""
    
    def setUp(self):
        """Set up test data with missing values."""
        # Create date range
        self.dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create sample price data
        np.random.seed(42)  # For reproducibility
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        # Create OHLC data
        self.data = pd.DataFrame({
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
            'volume': np.random.normal(1000000, 200000, 100)
        }, index=self.dates)
        
        # Create missing values in different patterns
        
        # 1. Single missing value
        self.single_missing = self.data.copy()
        self.single_missing.loc[self.dates[50], 'close'] = np.nan
        
        # 2. Block of missing values
        self.block_missing = self.data.copy()
        self.block_missing.loc[self.dates[30:40], 'close'] = np.nan
        
        # 3. Random missing values
        self.random_missing = self.data.copy()
        random_indices = np.random.choice(100, 10, replace=False)
        self.random_missing.loc[self.dates[random_indices], 'close'] = np.nan
        
        # 4. Missing values in multiple columns
        self.multi_col_missing = self.data.copy()
        self.multi_col_missing.loc[self.dates[20:25], ['open', 'high', 'low', 'close']] = np.nan
        
        # 5. All NaN column
        self.all_nan_column = self.data.copy()
        self.all_nan_column['close'] = np.nan
    
    def test_single_missing_value(self):
        """Test indicators with a single missing value."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.single_missing)
        
        # Check that NaN propagates through the calculation window
        self.assertTrue(pd.isna(result['SMA_20'].iloc[50:70]).any())
        self.assertTrue(pd.notna(result['SMA_20'].iloc[70:]).all())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.single_missing)
        
        # Check that NaN propagates through the calculation window
        self.assertTrue(pd.isna(result['RSI_14'].iloc[50:65]).any())
        self.assertTrue(pd.notna(result['RSI_14'].iloc[65:]).all())
        
        # Test Bollinger Bands
        bb = BollingerBands(period=20, std_dev=2)
        result = bb(self.single_missing)
        
        # Check that NaN propagates through the calculation window
        self.assertTrue(pd.isna(result['BB_20_2_middle'].iloc[50:70]).any())
        self.assertTrue(pd.notna(result['BB_20_2_middle'].iloc[70:]).all())
    
    def test_block_missing_values(self):
        """Test indicators with a block of missing values."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.block_missing)
        
        # Check that NaN propagates through the calculation window
        self.assertTrue(pd.isna(result['SMA_20'].iloc[30:60]).all())
        self.assertTrue(pd.notna(result['SMA_20'].iloc[60:]).all())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.block_missing)
        
        # Check that NaN propagates through the calculation window
        self.assertTrue(pd.isna(result['RSI_14'].iloc[30:54]).all())
        self.assertTrue(pd.notna(result['RSI_14'].iloc[54:]).all())
    
    def test_random_missing_values(self):
        """Test indicators with random missing values."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.random_missing)
        
        # Check that some values are NaN
        self.assertTrue(pd.isna(result['SMA_20']).any())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.random_missing)
        
        # Check that some values are NaN
        self.assertTrue(pd.isna(result['RSI_14']).any())
    
    def test_multi_column_missing(self):
        """Test indicators with missing values in multiple columns."""
        # Test Ichimoku Cloud
        ichimoku = IchimokuCloud()
        result = ichimoku(self.multi_col_missing)
        
        # Check that NaN propagates to output
        self.assertTrue(pd.isna(result['tenkan_sen'].iloc[20:29]).any())
        self.assertTrue(pd.isna(result['kijun_sen'].iloc[20:45]).any())
        
        # Test VWAP
        vwap = VWAP()
        result = vwap(self.multi_col_missing)
        
        # Check that NaN propagates to output
        self.assertTrue(pd.isna(result['VWAP'].iloc[20:25]).all())
    
    def test_all_nan_column(self):
        """Test indicators with an all-NaN column."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.all_nan_column)
        
        # Check that all values are NaN
        self.assertTrue(pd.isna(result['SMA_20']).all())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.all_nan_column)
        
        # Check that all values are NaN
        self.assertTrue(pd.isna(result['RSI_14']).all())


class TestExtremeValues(unittest.TestCase):
    """Test cases for handling extreme values."""
    
    def setUp(self):
        """Set up test data with extreme values."""
        # Create date range
        self.dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create normal price data
        np.random.seed(42)  # For reproducibility
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        # Create OHLC data
        self.normal_data = pd.DataFrame({
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
            'volume': np.random.normal(1000000, 200000, 100)
        }, index=self.dates)
        
        # Create data with extreme values
        
        # 1. Very large values
        self.large_values = self.normal_data.copy()
        self.large_values.loc[self.dates[50], 'close'] = 1e9
        
        # 2. Very small values
        self.small_values = self.normal_data.copy()
        self.small_values.loc[self.dates[50], 'close'] = 1e-9
        
        # 3. Negative values
        self.negative_values = self.normal_data.copy()
        self.negative_values.loc[self.dates[50], 'close'] = -100
        
        # 4. Zero values
        self.zero_values = self.normal_data.copy()
        self.zero_values.loc[self.dates[50], 'close'] = 0
        
        # 5. Infinite values
        self.inf_values = self.normal_data.copy()
        self.inf_values.loc[self.dates[50], 'close'] = np.inf
    
    def test_large_values(self):
        """Test indicators with very large values."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.large_values)
        
        # Check that calculation completes without error
        self.assertIn('SMA_20', result.columns)
        
        # Check that large value propagates through the window
        self.assertTrue((result['SMA_20'].iloc[50:70] > 1e7).any())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.large_values)
        
        # Check that calculation completes without error
        self.assertIn('RSI_14', result.columns)
        
        # Check that RSI stays within bounds despite large value
        self.assertTrue((result['RSI_14'].dropna() >= 0).all())
        self.assertTrue((result['RSI_14'].dropna() <= 100).all())
    
    def test_small_values(self):
        """Test indicators with very small values."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.small_values)
        
        # Check that calculation completes without error
        self.assertIn('SMA_20', result.columns)
        
        # Check that small value propagates through the window
        self.assertTrue((result['SMA_20'].iloc[50:70] < 5).any())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.small_values)
        
        # Check that calculation completes without error
        self.assertIn('RSI_14', result.columns)
        
        # Check that RSI stays within bounds despite small value
        self.assertTrue((result['RSI_14'].dropna() >= 0).all())
        self.assertTrue((result['RSI_14'].dropna() <= 100).all())
    
    def test_negative_values(self):
        """Test indicators with negative values."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.negative_values)
        
        # Check that calculation completes without error
        self.assertIn('SMA_20', result.columns)
        
        # Check that negative value propagates through the window
        self.assertTrue((result['SMA_20'].iloc[50:70] < 0).any())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.negative_values)
        
        # Check that calculation completes without error
        self.assertIn('RSI_14', result.columns)
        
        # Check that RSI stays within bounds despite negative value
        self.assertTrue((result['RSI_14'].dropna() >= 0).all())
        self.assertTrue((result['RSI_14'].dropna() <= 100).all())
        
        # Test Bollinger Bands
        bb = BollingerBands(period=20, std_dev=2)
        result = bb(self.negative_values)
        
        # Check that calculation completes without error
        self.assertIn('BB_20_2_middle', result.columns)
        
        # Check that bands maintain their relationship
        # Upper band should still be above middle band
        self.assertTrue((result['BB_20_2_upper'] >= result['BB_20_2_middle']).all())
        
        # Lower band should still be below middle band
        self.assertTrue((result['BB_20_2_lower'] <= result['BB_20_2_middle']).all())
    
    def test_zero_values(self):
        """Test indicators with zero values."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.zero_values)
        
        # Check that calculation completes without error
        self.assertIn('SMA_20', result.columns)
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.zero_values)
        
        # Check that calculation completes without error
        self.assertIn('RSI_14', result.columns)
        
        # Check that RSI stays within bounds despite zero value
        self.assertTrue((result['RSI_14'].dropna() >= 0).all())
        self.assertTrue((result['RSI_14'].dropna() <= 100).all())
    
    def test_infinite_values(self):
        """Test indicators with infinite values."""
        # Test SMA
        sma = SMA(period=20)
        result = sma(self.inf_values)
        
        # Check that calculation completes without error
        self.assertIn('SMA_20', result.columns)
        
        # Check that infinite value propagates as expected
        self.assertTrue(np.isinf(result['SMA_20'].iloc[50:70]).any())
        
        # Test RSI
        rsi = RSI(period=14)
        result = rsi(self.inf_values)
        
        # Check that calculation completes without error
        self.assertIn('RSI_14', result.columns)
        
        # Infinite values should result in NaN or extreme RSI values
        # This is implementation-dependent, so we just check that the calculation completes


class TestBoundaryConditions(unittest.TestCase):
    """Test cases for handling boundary conditions."""
    
    def setUp(self):
        """Set up test data for boundary condition testing."""
        # Create date range
        self.dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create sample price data
        np.random.seed(42)  # For reproducibility
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        # Create OHLC data
        self.data = pd.DataFrame({
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
            'volume': np.random.normal(1000000, 200000, 100)
        }, index=self.dates)
    
    def test_minimum_period(self):
        """Test indicators with minimum valid period."""
        # Test SMA with period=1
        sma = SMA(period=1)
        result = sma(self.data)
        
        # Check that calculation completes without error
        self.assertIn('SMA_1', result.columns)
        
        # Check that values match input (SMA with period=1 is just the input)
        pd.testing.assert_series_equal(result['SMA_1'], self.data['close'])
        
        # Test RSI with period=2 (minimum valid period)
        rsi = RSI(period=2)
        result = rsi(self.data)
        
        # Check that calculation completes without error
        self.assertIn('RSI_2', result.columns)
        
        # Check that values are within bounds
        self.assertTrue((result['RSI_2'].dropna() >= 0).all())
        self.assertTrue((result['RSI_2'].dropna() <= 100).all())
    
    def test_zero_period(self):
        """Test indicators with invalid zero period."""
        # This should raise a ValueError or similar
        with self.assertRaises(Exception):
            sma = SMA(period=0)
            result = sma(self.data)
        
        with self.assertRaises(Exception):
            rsi = RSI(period=0)
            result = rsi(self.data)
    
    def test_negative_period(self):
        """Test indicators with invalid negative period."""
        # This should raise a ValueError or similar
        with self.assertRaises(Exception):
            sma = SMA(period=-5)
            result = sma(self.data)
        
        with self.assertRaises(Exception):
            rsi = RSI(period=-10)
            result = rsi(self.data)
    
    def test_very_large_period(self):
        """Test indicators with very large period."""
        # Test SMA with period larger than dataset
        sma = SMA(period=200)
        result = sma(self.data)
        
        # Check that calculation completes without error
        self.assertIn('SMA_200', result.columns)
        
        # Check that all values are NaN (not enough data)
        self.assertTrue(result['SMA_200'].isna().all())
        
        # Test RSI with period larger than dataset
        rsi = RSI(period=200)
        result = rsi(self.data)
        
        # Check that calculation completes without error
        self.assertIn('RSI_200', result.columns)
        
        # Check that all values are NaN (not enough data)
        self.assertTrue(result['RSI_200'].isna().all())


if __name__ == '__main__':
    unittest.main()
