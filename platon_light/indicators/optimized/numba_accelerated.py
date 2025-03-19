#!/usr/bin/env python
"""
Numba-accelerated indicator implementations for improved performance.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True)
    def _calculate_rsi_numba(prices, period):
        """
        Calculate RSI using Numba for acceleration.
        
        Args:
            prices: Array of price values
            period: RSI period
            
        Returns:
            Array of RSI values
        """
        # Initialize output array
        rsi = np.empty_like(prices)
        rsi[:] = np.nan
        
        # Calculate price changes
        delta = np.zeros_like(prices)
        delta[1:] = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gains = np.copy(delta)
        losses = np.copy(delta)
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average is simple average
        if len(prices) > period:
            avg_gain[period] = np.mean(gains[1:period+1])
            avg_loss[period] = np.mean(losses[1:period+1])
            
            # Subsequent averages use smoothing formula
            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
            
            # Calculate RS and RSI
            for i in range(period, len(prices)):
                if avg_loss[i] == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain[i] / avg_loss[i]
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi


class NumbaAcceleratedRSI(BaseIndicator):
    """
    Numba-accelerated Relative Strength Index (RSI) indicator.
    
    This implementation uses Numba JIT compilation to improve performance.
    """
    
    def __init__(self, period=14, input_column='close', output_column=None):
        """
        Initialize the RSI indicator.
        
        Args:
            period: Number of periods for RSI calculation
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
        
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba is required for NumbaAcceleratedRSI")
    
    @property
    def name(self):
        """Get the indicator name with period."""
        return f"RSI_{self.period}"
    
    def calculate(self, data):
        """
        Calculate the Relative Strength Index using Numba.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with RSI values
        """
        # Extract price data as numpy array
        prices = data[self.input_column].values
        
        # Calculate RSI using Numba
        rsi_values = _calculate_rsi_numba(prices, self.period)
        
        # Convert back to Series
        return pd.Series(rsi_values, index=data.index)
else:
    # Fallback implementation if Numba is not available
    class NumbaAcceleratedRSI(BaseIndicator):
        """Placeholder class when Numba is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError("Numba is required for NumbaAcceleratedRSI")
