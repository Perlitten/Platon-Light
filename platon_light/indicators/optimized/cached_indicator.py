#!/usr/bin/env python
"""
Cached indicator implementation for improved performance.
"""

import pandas as pd
import numpy as np
import hashlib
from ..base import BaseIndicator


class CachedIndicator:
    """
    Wrapper class that adds caching to any indicator.
    
    This class improves performance by caching the results of indicator calculations
    to avoid redundant calculations when the same data is processed multiple times.
    """
    
    def __init__(self, indicator):
        """
        Initialize the cached indicator.
        
        Args:
            indicator: The indicator instance to cache
        """
        self.indicator = indicator
        self.cache = {}
    
    def _get_cache_key(self, data):
        """
        Generate a cache key for the given data.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            String hash that uniquely identifies the data
        """
        # Use a hash of the data to create a cache key
        # For simplicity, we'll use a hash of the relevant columns
        if isinstance(self.indicator.input_column, str):
            columns = [self.indicator.input_column]
        else:
            columns = self.indicator.input_column
        
        # Create a string representation of the data
        data_str = str(data[columns].values)
        
        # Create a hash of the data string
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def __call__(self, data):
        """
        Apply the indicator to the data, using cached results if available.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with indicator values added as new columns
        """
        # Generate a cache key for the data
        cache_key = self._get_cache_key(data)
        
        # Check if the result is already in the cache
        if cache_key in self.cache:
            # Get the cached result
            cached_result = self.cache[cache_key]
            
            # Make a copy of the input data
            result = data.copy()
            
            # Add the cached indicator values to the result
            for column in cached_result.columns:
                if column not in data.columns:
                    result[column] = cached_result[column]
            
            return result
        
        # Calculate the indicator values
        result = self.indicator(data)
        
        # Store the result in the cache
        # We only need to cache the columns added by the indicator
        added_columns = set(result.columns) - set(data.columns)
        self.cache[cache_key] = result[list(added_columns)]
        
        return result
