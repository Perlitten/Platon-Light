#!/usr/bin/env python
"""
Performance benchmarking tests for technical indicators.

This module contains tests for measuring the performance of indicators,
including calculation speed, memory usage, and scaling with data size.
"""

import os
import sys
import time
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from memory_profiler import profile

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import indicators to test
from platon_light.indicators.base import BaseIndicator
from platon_light.indicators.basic import SMA, EMA, RSI, BollingerBands, MACD
from platon_light.indicators.advanced import IchimokuCloud, VWAP, HeikinAshi


class PerformanceTest(unittest.TestCase):
    """Base class for performance testing."""
    
    def setUp(self):
        """Set up test data for performance testing."""
        # Create datasets of different sizes
        self.sizes = [100, 1000, 10000, 100000]
        self.datasets = {}
        
        for size in self.sizes:
            # Create date range
            dates = pd.date_range(start='2020-01-01', periods=size, freq='5min')
            
            # Create sample price data
            np.random.seed(42)  # For reproducibility
            close_prices = 100 + np.cumsum(np.random.normal(0, 0.1, size))
            
            # Create OHLC data
            self.datasets[size] = pd.DataFrame({
                'open': close_prices - np.random.normal(0, 0.05, size),
                'high': close_prices + np.random.normal(0.1, 0.05, size),
                'low': close_prices - np.random.normal(0.1, 0.05, size),
                'close': close_prices,
                'volume': np.random.normal(1000000, 200000, size)
            }, index=dates)
    
    def measure_execution_time(self, indicator, data):
        """Measure execution time of an indicator calculation."""
        start_time = time.time()
        result = indicator(data)
        end_time = time.time()
        return end_time - start_time


class SpeedTest(PerformanceTest):
    """Test cases for measuring calculation speed."""
    
    def test_sma_speed(self):
        """Test SMA calculation speed with different data sizes."""
        print("\nSMA Speed Test")
        print("==============")
        
        times = []
        for size in self.sizes:
            sma = SMA(period=20)
            execution_time = self.measure_execution_time(sma, self.datasets[size])
            times.append(execution_time)
            print(f"SMA with {size} data points: {execution_time:.6f} seconds")
        
        # Check that execution time scales approximately linearly with data size
        # This is a rough check - actual scaling may vary
        if len(times) > 2:
            scaling_factor = times[-1] / times[0]
            data_factor = self.sizes[-1] / self.sizes[0]
            print(f"Scaling factor: {scaling_factor:.2f}, Data factor: {data_factor:.2f}")
            # Linear scaling would mean scaling_factor ≈ data_factor
            # Allow for some overhead
            self.assertLess(scaling_factor, data_factor * 2)
    
    def test_ema_speed(self):
        """Test EMA calculation speed with different data sizes."""
        print("\nEMA Speed Test")
        print("==============")
        
        times = []
        for size in self.sizes:
            ema = EMA(period=20)
            execution_time = self.measure_execution_time(ema, self.datasets[size])
            times.append(execution_time)
            print(f"EMA with {size} data points: {execution_time:.6f} seconds")
    
    def test_rsi_speed(self):
        """Test RSI calculation speed with different data sizes."""
        print("\nRSI Speed Test")
        print("==============")
        
        times = []
        for size in self.sizes:
            rsi = RSI(period=14)
            execution_time = self.measure_execution_time(rsi, self.datasets[size])
            times.append(execution_time)
            print(f"RSI with {size} data points: {execution_time:.6f} seconds")
    
    def test_bollinger_bands_speed(self):
        """Test Bollinger Bands calculation speed with different data sizes."""
        print("\nBollinger Bands Speed Test")
        print("=========================")
        
        times = []
        for size in self.sizes:
            bb = BollingerBands(period=20, std_dev=2)
            execution_time = self.measure_execution_time(bb, self.datasets[size])
            times.append(execution_time)
            print(f"Bollinger Bands with {size} data points: {execution_time:.6f} seconds")
    
    def test_macd_speed(self):
        """Test MACD calculation speed with different data sizes."""
        print("\nMACD Speed Test")
        print("==============")
        
        times = []
        for size in self.sizes:
            macd = MACD(fast_period=12, slow_period=26, signal_period=9)
            execution_time = self.measure_execution_time(macd, self.datasets[size])
            times.append(execution_time)
            print(f"MACD with {size} data points: {execution_time:.6f} seconds")
    
    def test_ichimoku_speed(self):
        """Test Ichimoku Cloud calculation speed with different data sizes."""
        print("\nIchimoku Cloud Speed Test")
        print("========================")
        
        times = []
        for size in self.sizes:
            ichimoku = IchimokuCloud()
            execution_time = self.measure_execution_time(ichimoku, self.datasets[size])
            times.append(execution_time)
            print(f"Ichimoku Cloud with {size} data points: {execution_time:.6f} seconds")
    
    def test_vwap_speed(self):
        """Test VWAP calculation speed with different data sizes."""
        print("\nVWAP Speed Test")
        print("==============")
        
        times = []
        for size in self.sizes:
            vwap = VWAP()
            execution_time = self.measure_execution_time(vwap, self.datasets[size])
            times.append(execution_time)
            print(f"VWAP with {size} data points: {execution_time:.6f} seconds")
    
    def test_heikin_ashi_speed(self):
        """Test Heikin-Ashi calculation speed with different data sizes."""
        print("\nHeikin-Ashi Speed Test")
        print("=====================")
        
        times = []
        for size in self.sizes:
            ha = HeikinAshi()
            execution_time = self.measure_execution_time(ha, self.datasets[size])
            times.append(execution_time)
            print(f"Heikin-Ashi with {size} data points: {execution_time:.6f} seconds")
    
    def test_compare_indicators(self):
        """Compare calculation speed of different indicators."""
        print("\nIndicator Speed Comparison")
        print("=========================")
        
        # Use a fixed data size for comparison
        size = 10000
        data = self.datasets[size]
        
        indicators = [
            ("SMA", SMA(period=20)),
            ("EMA", EMA(period=20)),
            ("RSI", RSI(period=14)),
            ("Bollinger Bands", BollingerBands(period=20, std_dev=2)),
            ("MACD", MACD(fast_period=12, slow_period=26, signal_period=9)),
            ("Ichimoku Cloud", IchimokuCloud()),
            ("VWAP", VWAP()),
            ("Heikin-Ashi", HeikinAshi())
        ]
        
        results = {}
        for name, indicator in indicators:
            execution_time = self.measure_execution_time(indicator, data)
            results[name] = execution_time
            print(f"{name}: {execution_time:.6f} seconds")
        
        # Sort indicators by execution time
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        print("\nIndicators sorted by speed (fastest first):")
        for name, time in sorted_results:
            print(f"{name}: {time:.6f} seconds")


class MemoryTest(PerformanceTest):
    """Test cases for measuring memory usage."""
    
    @profile
    def test_sma_memory(self):
        """Test SMA memory usage with large dataset."""
        print("\nSMA Memory Test")
        print("===============")
        
        # Use the largest dataset
        size = self.sizes[-1]
        data = self.datasets[size]
        
        sma = SMA(period=20)
        result = sma(data)
        
        # Force garbage collection to get accurate memory usage
        import gc
        gc.collect()
    
    @profile
    def test_rsi_memory(self):
        """Test RSI memory usage with large dataset."""
        print("\nRSI Memory Test")
        print("===============")
        
        # Use the largest dataset
        size = self.sizes[-1]
        data = self.datasets[size]
        
        rsi = RSI(period=14)
        result = rsi(data)
        
        # Force garbage collection to get accurate memory usage
        import gc
        gc.collect()
    
    @profile
    def test_bollinger_bands_memory(self):
        """Test Bollinger Bands memory usage with large dataset."""
        print("\nBollinger Bands Memory Test")
        print("==========================")
        
        # Use the largest dataset
        size = self.sizes[-1]
        data = self.datasets[size]
        
        bb = BollingerBands(period=20, std_dev=2)
        result = bb(data)
        
        # Force garbage collection to get accurate memory usage
        import gc
        gc.collect()
    
    @profile
    def test_ichimoku_memory(self):
        """Test Ichimoku Cloud memory usage with large dataset."""
        print("\nIchimoku Cloud Memory Test")
        print("=========================")
        
        # Use the largest dataset
        size = self.sizes[-1]
        data = self.datasets[size]
        
        ichimoku = IchimokuCloud()
        result = ichimoku(data)
        
        # Force garbage collection to get accurate memory usage
        import gc
        gc.collect()


class ScalingTest(PerformanceTest):
    """Test cases for measuring scaling with data size."""
    
    def test_scaling_visualization(self):
        """Visualize how indicators scale with data size."""
        print("\nScaling Visualization Test")
        print("=========================")
        
        indicators = [
            ("SMA", SMA(period=20)),
            ("EMA", EMA(period=20)),
            ("RSI", RSI(period=14)),
            ("Bollinger Bands", BollingerBands(period=20, std_dev=2)),
            ("MACD", MACD(fast_period=12, slow_period=26, signal_period=9))
        ]
        
        results = {name: [] for name, _ in indicators}
        
        for size in self.sizes:
            data = self.datasets[size]
            for name, indicator in indicators:
                execution_time = self.measure_execution_time(indicator, data)
                results[name].append(execution_time)
                print(f"{name} with {size} data points: {execution_time:.6f} seconds")
        
        # Create scaling visualization
        plt.figure(figsize=(10, 6))
        for name, times in results.items():
            plt.plot(self.sizes, times, marker='o', label=name)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Data Size (log scale)')
        plt.ylabel('Execution Time (seconds, log scale)')
        plt.title('Indicator Scaling with Data Size')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'indicator_scaling.png'))
        print(f"Scaling visualization saved to {os.path.join(output_dir, 'indicator_scaling.png')}")
    
    def test_relative_scaling(self):
        """Test relative scaling of indicators with data size."""
        print("\nRelative Scaling Test")
        print("====================")
        
        indicators = [
            ("SMA", SMA(period=20)),
            ("EMA", EMA(period=20)),
            ("RSI", RSI(period=14)),
            ("Bollinger Bands", BollingerBands(period=20, std_dev=2)),
            ("MACD", MACD(fast_period=12, slow_period=26, signal_period=9))
        ]
        
        # Calculate scaling factors
        scaling_factors = {}
        
        for name, indicator in indicators:
            times = []
            for size in self.sizes:
                data = self.datasets[size]
                execution_time = self.measure_execution_time(indicator, data)
                times.append(execution_time)
            
            # Calculate scaling factor between smallest and largest dataset
            if times[0] > 0:  # Avoid division by zero
                scaling_factor = times[-1] / times[0]
                data_factor = self.sizes[-1] / self.sizes[0]
                relative_scaling = scaling_factor / data_factor
                scaling_factors[name] = relative_scaling
                print(f"{name} - Scaling factor: {scaling_factor:.2f}, "
                      f"Data factor: {data_factor:.2f}, "
                      f"Relative scaling: {relative_scaling:.2f}")
        
        # Sort indicators by relative scaling
        sorted_scaling = sorted(scaling_factors.items(), key=lambda x: x[1])
        print("\nIndicators sorted by relative scaling (best first):")
        for name, factor in sorted_scaling:
            print(f"{name}: {factor:.2f}")


class OptimizationTest(PerformanceTest):
    """Test cases for measuring optimization techniques."""
    
    def test_numba_optimization(self):
        """Test performance improvement with Numba optimization."""
        print("\nNumba Optimization Test")
        print("======================")
        
        try:
            from platon_light.indicators.optimized import NumbaAcceleratedRSI
            
            # Use a fixed data size for comparison
            size = 100000
            data = self.datasets[size]
            
            # Standard RSI
            standard_rsi = RSI(period=14)
            standard_time = self.measure_execution_time(standard_rsi, data)
            print(f"Standard RSI: {standard_time:.6f} seconds")
            
            # Numba-accelerated RSI
            numba_rsi = NumbaAcceleratedRSI(period=14)
            numba_time = self.measure_execution_time(numba_rsi, data)
            print(f"Numba RSI: {numba_time:.6f} seconds")
            
            # Calculate speedup
            speedup = standard_time / numba_time
            print(f"Speedup: {speedup:.2f}x")
            
            # Check that results are the same
            standard_result = standard_rsi(data)['RSI_14']
            numba_result = numba_rsi(data)['RSI_14']
            
            # Allow for small numerical differences
            np.testing.assert_allclose(
                standard_result.dropna().values,
                numba_result.dropna().values,
                rtol=1e-10
            )
            
            # Check that Numba version is faster
            self.assertGreater(speedup, 1.0)
        
        except ImportError:
            print("NumbaAcceleratedRSI not found, skipping test")
    
    def test_caching_optimization(self):
        """Test performance improvement with result caching."""
        print("\nCaching Optimization Test")
        print("========================")
        
        try:
            from platon_light.indicators.optimized import CachedIndicator
            
            # Use a fixed data size for comparison
            size = 10000
            data = self.datasets[size]
            
            # Standard SMA
            standard_sma = SMA(period=20)
            
            # First run
            standard_time_1 = self.measure_execution_time(standard_sma, data)
            print(f"Standard SMA (first run): {standard_time_1:.6f} seconds")
            
            # Second run (should be similar to first)
            standard_time_2 = self.measure_execution_time(standard_sma, data)
            print(f"Standard SMA (second run): {standard_time_2:.6f} seconds")
            
            # Cached SMA
            cached_sma = CachedIndicator(SMA(period=20))
            
            # First run (should be similar to standard)
            cached_time_1 = self.measure_execution_time(cached_sma, data)
            print(f"Cached SMA (first run): {cached_time_1:.6f} seconds")
            
            # Second run (should be much faster)
            cached_time_2 = self.measure_execution_time(cached_sma, data)
            print(f"Cached SMA (second run): {cached_time_2:.6f} seconds")
            
            # Calculate speedup for second run
            speedup = standard_time_2 / cached_time_2
            print(f"Caching speedup: {speedup:.2f}x")
            
            # Check that cached version is faster on second run
            self.assertLess(cached_time_2, standard_time_2)
        
        except ImportError:
            print("CachedIndicator not found, skipping test")


def run_performance_tests():
    """Run all performance tests and generate a report."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add speed tests
    suite.addTest(SpeedTest('test_sma_speed'))
    suite.addTest(SpeedTest('test_ema_speed'))
    suite.addTest(SpeedTest('test_rsi_speed'))
    suite.addTest(SpeedTest('test_bollinger_bands_speed'))
    suite.addTest(SpeedTest('test_macd_speed'))
    suite.addTest(SpeedTest('test_ichimoku_speed'))
    suite.addTest(SpeedTest('test_vwap_speed'))
    suite.addTest(SpeedTest('test_heikin_ashi_speed'))
    suite.addTest(SpeedTest('test_compare_indicators'))
    
    # Add memory tests
    suite.addTest(MemoryTest('test_sma_memory'))
    suite.addTest(MemoryTest('test_rsi_memory'))
    suite.addTest(MemoryTest('test_bollinger_bands_memory'))
    suite.addTest(MemoryTest('test_ichimoku_memory'))
    
    # Add scaling tests
    suite.addTest(ScalingTest('test_scaling_visualization'))
    suite.addTest(ScalingTest('test_relative_scaling'))
    
    # Add optimization tests
    suite.addTest(OptimizationTest('test_numba_optimization'))
    suite.addTest(OptimizationTest('test_caching_optimization'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate a report
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
        f.write("Platon Light Indicator Performance Report\n")
        f.write("=======================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Test Summary\n")
        f.write("------------\n")
        f.write(f"Tests run: {result.testsRun}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Failures: {len(result.failures)}\n\n")
        
        if result.errors:
            f.write("Errors\n")
            f.write("------\n")
            for test, error in result.errors:
                f.write(f"{test}: {error}\n\n")
        
        if result.failures:
            f.write("Failures\n")
            f.write("--------\n")
            for test, failure in result.failures:
                f.write(f"{test}: {failure}\n\n")
    
    print(f"Performance report saved to {os.path.join(output_dir, 'performance_report.txt')}")


if __name__ == '__main__':
    run_performance_tests()
