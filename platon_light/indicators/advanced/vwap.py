#!/usr/bin/env python
"""
Volume Weighted Average Price (VWAP) indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price (VWAP) indicator.
    
    VWAP is calculated by adding up the dollars traded for every transaction
    (price multiplied by the number of shares traded) and then dividing
    by the total shares traded.
    """
    
    def __init__(self, input_columns=None, output_column=None):
        """
        Initialize the VWAP indicator.
        
        Args:
            input_columns: Dict with column names for high, low, close, volume
            output_column: Column name for the output
        """
        # Default input columns
        if input_columns is None:
            input_columns = {
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        super().__init__(input_column='close', output_column=output_column)
        self.input_columns = input_columns
    
    @property
    def name(self):
        """Get the indicator name."""
        return "VWAP"
    
    def calculate(self, data):
        """
        Calculate the Volume Weighted Average Price.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with VWAP values
        """
        # Extract required columns
        high = data[self.input_columns['high']]
        low = data[self.input_columns['low']]
        close = data[self.input_columns['close']]
        volume = data[self.input_columns['volume']]
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP components
        price_volume = typical_price * volume
        cumulative_price_volume = price_volume.cumsum()
        cumulative_volume = volume.cumsum()
        
        # Calculate VWAP
        vwap = cumulative_price_volume / cumulative_volume
        
        return vwap
