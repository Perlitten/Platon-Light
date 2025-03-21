#!/usr/bin/env python
"""
Data Pipeline Validation Tests

This script provides comprehensive tests for validating the data pipeline
in the Platon Light backtesting module, ensuring data integrity and correctness.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import tempfile
import json

# Add parent directory to path to import Platon Light modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_data_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestDataPipeline(unittest.TestCase):
    """Test suite for validating the backtesting data pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_dir = cls.temp_dir.name
        
        # Create sample data files
        cls._create_sample_data()
        
        # Create basic configuration
        cls.config = {
            'data': {
                'source': 'csv',
                'directory': cls.data_dir
            }
        }
        
        # Create data loader
        cls.data_loader = DataLoader(cls.config)
        
        logger.info("Test environment set up successfully")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are done"""
        cls.temp_dir.cleanup()
        logger.info("Test environment cleaned up")
    
    @classmethod
    def _create_sample_data(cls):
        """Create sample data files for testing"""
        # Create directory structure
        os.makedirs(os.path.join(cls.data_dir, 'binance'), exist_ok=True)
        
        # Generate sample OHLCV data
        symbols = ['BTCUSDT', 'ETHUSDT']
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Determine time delta based on timeframe
                if timeframe == '1m':
                    delta = timedelta(minutes=1)
                elif timeframe == '5m':
                    delta = timedelta(minutes=5)
                elif timeframe == '15m':
                    delta = timedelta(minutes=15)
                elif timeframe == '1h':
                    delta = timedelta(hours=1)
                elif timeframe == '4h':
                    delta = timedelta(hours=4)
                else:  # '1d'
                    delta = timedelta(days=1)
                
                # Generate timestamps
                current_date = start_date
                timestamps = []
                while current_date <= end_date:
                    timestamps.append(current_date)
                    current_date += delta
                
                # Generate OHLCV data
                n = len(timestamps)
                
                # Start with a base price and add random walk
                base_price = 100 if symbol == 'ETHUSDT' else 40000
                price_volatility = base_price * 0.01  # 1% volatility
                
                # Generate price series with random walk
                np.random.seed(42)  # For reproducibility
                random_walk = np.random.normal(0, price_volatility, n).cumsum()
                prices = base_price + random_walk
                
                # Ensure prices are positive
                prices = np.maximum(prices, base_price * 0.1)
                
                # Generate OHLCV data
                opens = prices.copy()
                highs = prices * (1 + np.random.uniform(0, 0.02, n))
                lows = prices * (1 - np.random.uniform(0, 0.02, n))
                closes = prices * (1 + np.random.normal(0, 0.01, n))
                volumes = np.random.uniform(base_price * 10, base_price * 100, n)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'timestamp': [int(t.timestamp() * 1000) for t in timestamps],
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                })
                
                # Save to CSV
                filename = f"{symbol}_{timeframe}.csv"
                filepath = os.path.join(cls.data_dir, 'binance', filename)
                df.to_csv(filepath, index=False)
                
                logger.info(f"Created sample data file: {filepath}")
        
        # Create a file with missing values
        df_missing = pd.DataFrame({
            'timestamp': [int(t.timestamp() * 1000) for t in timestamps],
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Introduce missing values
        df_missing.loc[10:20, 'close'] = np.nan
        df_missing.loc[30:35, 'volume'] = np.nan
        
        # Save to CSV
        filepath = os.path.join(cls.data_dir, 'binance', 'BTCUSDT_1h_missing.csv')
        df_missing.to_csv(filepath, index=False)
        logger.info(f"Created sample data file with missing values: {filepath}")
        
        # Create a file with duplicate timestamps
        df_duplicate = df.copy()
        duplicate_rows = df.iloc[5:10].copy()
        df_duplicate = pd.concat([df_duplicate, duplicate_rows])
        
        # Save to CSV
        filepath = os.path.join(cls.data_dir, 'binance', 'BTCUSDT_1h_duplicate.csv')
        df_duplicate.to_csv(filepath, index=False)
        logger.info(f"Created sample data file with duplicate timestamps: {filepath}")
        
        # Create a file with out-of-order timestamps
        df_unordered = df.copy()
        df_unordered = df_unordered.sample(frac=1, random_state=42)  # Shuffle rows
        
        # Save to CSV
        filepath = os.path.join(cls.data_dir, 'binance', 'BTCUSDT_1h_unordered.csv')
        df_unordered.to_csv(filepath, index=False)
        logger.info(f"Created sample data file with unordered timestamps: {filepath}")
    
    def test_data_loading_basic(self):
        """Test basic data loading functionality"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 5)
        end_date = datetime(2022, 1, 15)
        
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that data is not empty
        self.assertFalse(data.empty)
        
        # Check that data has the correct columns
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(data.columns), expected_columns)
        
        # Check that data is within the specified date range
        min_date = pd.to_datetime(data['timestamp'].min(), unit='ms')
        max_date = pd.to_datetime(data['timestamp'].max(), unit='ms')
        
        self.assertGreaterEqual(min_date, start_date)
        self.assertLessEqual(max_date, end_date)
        
        # Check that data is sorted by timestamp
        self.assertTrue(data['timestamp'].is_monotonic_increasing)
        
        logger.info("Basic data loading test passed")
    
    def test_data_loading_all_timeframes(self):
        """Test loading data for all timeframes"""
        symbol = 'BTCUSDT'
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        start_date = datetime(2022, 1, 5)
        end_date = datetime(2022, 1, 15)
        
        for timeframe in timeframes:
            data = self.data_loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            # Check that data is not empty
            self.assertFalse(data.empty)
            
            # Check that data has the correct columns
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            self.assertListEqual(list(data.columns), expected_columns)
            
            # Check that data is sorted by timestamp
            self.assertTrue(data['timestamp'].is_monotonic_increasing)
            
            logger.info(f"Data loading test for {timeframe} timeframe passed")
    
    def test_data_loading_all_symbols(self):
        """Test loading data for all symbols"""
        symbols = ['BTCUSDT', 'ETHUSDT']
        timeframe = '1h'
        start_date = datetime(2022, 1, 5)
        end_date = datetime(2022, 1, 15)
        
        for symbol in symbols:
            data = self.data_loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            # Check that data is not empty
            self.assertFalse(data.empty)
            
            # Check that data has the correct columns
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            self.assertListEqual(list(data.columns), expected_columns)
            
            # Check that data is sorted by timestamp
            self.assertTrue(data['timestamp'].is_monotonic_increasing)
            
            logger.info(f"Data loading test for {symbol} symbol passed")
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        # Load data with missing values
        data = self.data_loader.load_data(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 31),
            filename='BTCUSDT_1h_missing.csv'  # Use file with missing values
        )
        
        # Check that data has no missing values (should be filled by preprocessing)
        self.assertFalse(data.isnull().any().any())
        
        logger.info("Data preprocessing test passed")
    
    def test_data_deduplication(self):
        """Test data deduplication functionality"""
        # Load data with duplicate timestamps
        data = self.data_loader.load_data(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 31),
            filename='BTCUSDT_1h_duplicate.csv'  # Use file with duplicate timestamps
        )
        
        # Check that there are no duplicate timestamps
        self.assertEqual(len(data), len(data['timestamp'].unique()))
        
        logger.info("Data deduplication test passed")
    
    def test_data_sorting(self):
        """Test data sorting functionality"""
        # Load data with unordered timestamps
        data = self.data_loader.load_data(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 31),
            filename='BTCUSDT_1h_unordered.csv'  # Use file with unordered timestamps
        )
        
        # Check that data is sorted by timestamp
        self.assertTrue(data['timestamp'].is_monotonic_increasing)
        
        logger.info("Data sorting test passed")
    
    def test_data_integrity(self):
        """Test data integrity checks"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 5)
        end_date = datetime(2022, 1, 15)
        
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that high is always >= low
        self.assertTrue((data['high'] >= data['low']).all())
        
        # Check that high is always >= open and close
        self.assertTrue((data['high'] >= data['open']).all())
        self.assertTrue((data['high'] >= data['close']).all())
        
        # Check that low is always <= open and close
        self.assertTrue((data['low'] <= data['open']).all())
        self.assertTrue((data['low'] <= data['close']).all())
        
        # Check that volume is always >= 0
        self.assertTrue((data['volume'] >= 0).all())
        
        logger.info("Data integrity test passed")
    
    def test_data_continuity(self):
        """Test data continuity (no gaps in timestamps)"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 5)
        end_date = datetime(2022, 1, 15)
        
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert timestamps to datetime
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Calculate time differences between consecutive timestamps
        time_diffs = data['datetime'].diff().dropna()
        
        # Check that all time differences are equal to the timeframe
        expected_diff = pd.Timedelta(hours=1)  # For 1h timeframe
        self.assertTrue((time_diffs == expected_diff).all())
        
        logger.info("Data continuity test passed")
    
    def test_data_aggregation(self):
        """Test data aggregation to higher timeframes"""
        symbol = 'BTCUSDT'
        source_timeframe = '1h'
        target_timeframe = '4h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        # Load source data
        source_data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=source_timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Aggregate to target timeframe
        aggregated_data = self.data_loader.aggregate_timeframe(
            data=source_data,
            source_timeframe=source_timeframe,
            target_timeframe=target_timeframe
        )
        
        # Load target data directly
        target_data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=target_timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that aggregated data has the correct number of rows
        expected_ratio = 4  # 4 hours in 1 hour
        self.assertAlmostEqual(
            len(source_data) / len(aggregated_data),
            expected_ratio,
            delta=1  # Allow for small differences due to rounding
        )
        
        # Check that aggregated data has the correct columns
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(aggregated_data.columns), expected_columns)
        
        # Check that aggregated data is sorted by timestamp
        self.assertTrue(aggregated_data['timestamp'].is_monotonic_increasing)
        
        logger.info("Data aggregation test passed")
    
    def test_data_slicing(self):
        """Test data slicing by date range"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        full_start_date = datetime(2022, 1, 1)
        full_end_date = datetime(2022, 1, 31)
        
        # Load full data
        full_data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=full_start_date,
            end_date=full_end_date
        )
        
        # Define slice range
        slice_start_date = datetime(2022, 1, 10)
        slice_end_date = datetime(2022, 1, 20)
        
        # Slice data
        sliced_data = self.data_loader.slice_data_by_date(
            data=full_data,
            start_date=slice_start_date,
            end_date=slice_end_date
        )
        
        # Check that sliced data is within the specified date range
        min_date = pd.to_datetime(sliced_data['timestamp'].min(), unit='ms')
        max_date = pd.to_datetime(sliced_data['timestamp'].max(), unit='ms')
        
        self.assertGreaterEqual(min_date, slice_start_date)
        self.assertLessEqual(max_date, slice_end_date)
        
        # Check that sliced data is a subset of full data
        self.assertLessEqual(len(sliced_data), len(full_data))
        
        logger.info("Data slicing test passed")
    
    def test_data_indicators(self):
        """Test adding technical indicators to data"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        # Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add SMA indicator
        data_with_sma = self.data_loader.add_indicator(
            data=data,
            indicator='sma',
            params={'window': 20}
        )
        
        # Check that indicator column was added
        self.assertIn('sma_20', data_with_sma.columns)
        
        # Add EMA indicator
        data_with_ema = self.data_loader.add_indicator(
            data=data_with_sma,
            indicator='ema',
            params={'window': 20}
        )
        
        # Check that indicator column was added
        self.assertIn('ema_20', data_with_ema.columns)
        
        # Add RSI indicator
        data_with_rsi = self.data_loader.add_indicator(
            data=data_with_ema,
            indicator='rsi',
            params={'window': 14}
        )
        
        # Check that indicator column was added
        self.assertIn('rsi_14', data_with_rsi.columns)
        
        logger.info("Data indicators test passed")
    
    def test_data_export(self):
        """Test exporting data to CSV"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        # Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Export data to CSV
        export_path = os.path.join(self.data_dir, 'exported_data.csv')
        self.data_loader.export_data(
            data=data,
            filepath=export_path
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(export_path))
        
        # Load exported data
        exported_data = pd.read_csv(export_path)
        
        # Check that exported data has the same shape as original data
        self.assertEqual(exported_data.shape, data.shape)
        
        logger.info("Data export test passed")
    
    def test_data_caching(self):
        """Test data caching functionality"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        # Enable caching
        self.data_loader.enable_caching(cache_dir=self.data_dir)
        
        # Load data (first time, should read from file)
        start_time = datetime.now()
        data1 = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        first_load_time = (datetime.now() - start_time).total_seconds()
        
        # Load data again (should read from cache)
        start_time = datetime.now()
        data2 = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        second_load_time = (datetime.now() - start_time).total_seconds()
        
        # Check that data is the same
        pd.testing.assert_frame_equal(data1, data2)
        
        # Check that second load was faster (or at least not significantly slower)
        self.assertLessEqual(second_load_time, first_load_time * 1.5)
        
        # Disable caching
        self.data_loader.disable_caching()
        
        logger.info("Data caching test passed")
    
    def test_data_validation(self):
        """Test data validation functionality"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        # Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate data
        validation_result = self.data_loader.validate_data(data)
        
        # Check that validation passed
        self.assertTrue(validation_result['valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # Create invalid data (negative volume)
        invalid_data = data.copy()
        invalid_data.loc[10, 'volume'] = -100
        
        # Validate invalid data
        validation_result = self.data_loader.validate_data(invalid_data)
        
        # Check that validation failed
        self.assertFalse(validation_result['valid'])
        self.assertGreater(len(validation_result['errors']), 0)
        
        logger.info("Data validation test passed")
    
    def test_data_merge(self):
        """Test merging data from multiple sources"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 15)
        
        # Load data from first source
        data1 = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=datetime(2022, 1, 7)
        )
        
        # Load data from second source
        data2 = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime(2022, 1, 8),
            end_date=end_date
        )
        
        # Merge data
        merged_data = self.data_loader.merge_data([data1, data2])
        
        # Check that merged data has the correct number of rows
        self.assertEqual(len(merged_data), len(data1) + len(data2))
        
        # Check that merged data is sorted by timestamp
        self.assertTrue(merged_data['timestamp'].is_monotonic_increasing)
        
        # Check that merged data spans the full date range
        min_date = pd.to_datetime(merged_data['timestamp'].min(), unit='ms')
        max_date = pd.to_datetime(merged_data['timestamp'].max(), unit='ms')
        
        self.assertGreaterEqual(min_date, start_date)
        self.assertLessEqual(max_date, end_date)
        
        logger.info("Data merge test passed")
    
    def test_data_resampling(self):
        """Test data resampling functionality"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        # Load data
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Resample to daily
        resampled_data = self.data_loader.resample_data(
            data=data,
            timeframe='1d'
        )
        
        # Check that resampled data has fewer rows
        self.assertLess(len(resampled_data), len(data))
        
        # Check that resampled data has the correct columns
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(resampled_data.columns), expected_columns)
        
        # Check that resampled data is sorted by timestamp
        self.assertTrue(resampled_data['timestamp'].is_monotonic_increasing)
        
        logger.info("Data resampling test passed")
    
    def test_multi_symbol_data_loading(self):
        """Test loading data for multiple symbols simultaneously"""
        symbols = ['BTCUSDT', 'ETHUSDT']
        timeframe = '1h'
        start_date = datetime(2022, 1, 5)
        end_date = datetime(2022, 1, 15)
        
        # Load data for multiple symbols
        multi_data = self.data_loader.load_multi_symbol_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that data was loaded for all symbols
        self.assertEqual(len(multi_data), len(symbols))
        
        # Check that each symbol's data is not empty
        for symbol in symbols:
            self.assertIn(symbol, multi_data)
            self.assertFalse(multi_data[symbol].empty)
            
            # Check that data has the correct columns
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            self.assertListEqual(list(multi_data[symbol].columns), expected_columns)
            
            # Check that data is sorted by timestamp
            self.assertTrue(multi_data[symbol]['timestamp'].is_monotonic_increasing)
        
        logger.info("Multi-symbol data loading test passed")
    
    def test_data_pipeline_end_to_end(self):
        """Test the entire data pipeline end-to-end"""
        symbol = 'BTCUSDT'
        timeframe = '1h'
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        # 1. Load raw data
        raw_data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            preprocess=False  # Skip preprocessing
        )
        
        # 2. Preprocess data
        preprocessed_data = self.data_loader.preprocess_data(raw_data)
        
        # 3. Add technical indicators
        data_with_indicators = preprocessed_data.copy()
        
        # Add SMA
        data_with_indicators = self.data_loader.add_indicator(
            data=data_with_indicators,
            indicator='sma',
            params={'window': 20}
        )
        
        # Add EMA
        data_with_indicators = self.data_loader.add_indicator(
            data=data_with_indicators,
            indicator='ema',
            params={'window': 50}
        )
        
        # Add RSI
        data_with_indicators = self.data_loader.add_indicator(
            data=data_with_indicators,
            indicator='rsi',
            params={'window': 14}
        )
        
        # 4. Validate final data
        validation_result = self.data_loader.validate_data(data_with_indicators)
        
        # Check that validation passed
        self.assertTrue(validation_result['valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # 5. Export final data
        export_path = os.path.join(self.data_dir, 'final_data.csv')
        self.data_loader.export_data(
            data=data_with_indicators,
            filepath=export_path
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(export_path))
        
        logger.info("End-to-end data pipeline test passed")


if __name__ == '__main__':
    unittest.main()
