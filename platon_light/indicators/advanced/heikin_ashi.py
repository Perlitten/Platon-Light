#!/usr/bin/env python
"""
Heikin-Ashi indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator


class HeikinAshi(BaseIndicator):
    """
    Heikin-Ashi indicator.
    
    Heikin-Ashi candlesticks are a visual technique that eliminates market noise
    and helps identify trends more easily.
    """
    
    def __init__(self, input_columns=None, output_columns=None):
        """
        Initialize the Heikin-Ashi indicator.
        
        Args:
            input_columns: Dict with column names for open, high, low, close
            output_columns: Dict with column names for HA open, high, low, close
        """
        # Default input columns
        if input_columns is None:
            input_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close'
            }
        
        # Default output columns
        if output_columns is None:
            output_columns = {
                'open': 'HA_open',
                'high': 'HA_high',
                'low': 'HA_low',
                'close': 'HA_close'
            }
        
        super().__init__(input_column='close', output_column=None)
        self.input_columns = input_columns
        self.output_columns = output_columns
    
    @property
    def name(self):
        """Get the indicator name."""
        return "HeikinAshi"
    
    def calculate(self, data):
        """
        Calculate the Heikin-Ashi candlesticks.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with Heikin-Ashi values
        """
        # Extract required columns
        open_price = data[self.input_columns['open']]
        high_price = data[self.input_columns['high']]
        low_price = data[self.input_columns['low']]
        close_price = data[self.input_columns['close']]
        
        # Initialize Heikin-Ashi DataFrame
        ha_data = pd.DataFrame(index=data.index)
        
        # Calculate Heikin-Ashi close
        ha_close = (open_price + high_price + low_price + close_price) / 4
        ha_data[self.output_columns['close']] = ha_close
        
        # Calculate Heikin-Ashi open (requires previous values)
        ha_open = pd.Series(index=data.index)
        ha_open.iloc[0] = (open_price.iloc[0] + close_price.iloc[0]) / 2  # First value
        
        for i in range(1, len(data)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
        ha_data[self.output_columns['open']] = ha_open
        
        # Calculate Heikin-Ashi high and low
        ha_high = pd.Series(index=data.index)
        ha_low = pd.Series(index=data.index)
        
        for i in range(len(data)):
            ha_high.iloc[i] = max(high_price.iloc[i], ha_open.iloc[i], ha_close.iloc[i])
            ha_low.iloc[i] = min(low_price.iloc[i], ha_open.iloc[i], ha_close.iloc[i])
        
        ha_data[self.output_columns['high']] = ha_high
        ha_data[self.output_columns['low']] = ha_low
        
        return ha_data
    
    def __call__(self, data):
        """
        Apply the indicator to the data.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with indicator values added as new columns
        """
        # Make a copy to avoid modifying the original data
        result = data.copy()
        
        # Calculate the Heikin-Ashi values
        ha_data = self.calculate(data)
        
        # Add the values to the result
        for column in ha_data.columns:
            result[column] = ha_data[column]
        
        return result
