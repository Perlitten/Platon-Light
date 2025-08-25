#!/usr/bin/env python
"""
Simple Moving Average (SMA) indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SMA(BaseIndicator):
    """
    Simple Moving Average (SMA) indicator.

    The Simple Moving Average calculates the average price over a specified
    number of periods.
    """

    def __init__(self, period=20, input_column="close", output_column=None):
        """
        Initialize the SMA indicator.

        Args:
            period: Number of periods for the moving average
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        if output_column is None:
            output_column = f"SMA_{period}"
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period

    @property
    def name(self):
        """Get the indicator name with period."""
        return f"SMA_{self.period}"

    def calculate(self, data):
        """
        Calculate the Simple Moving Average.

        Args:
            data: DataFrame containing price data

        Returns:
            Series with SMA values
        """
        # Calculate SMA using pandas rolling window
        sma = data[self.input_column].rolling(window=self.period).mean()

        # Ensure the Series has the correct name
        sma.name = self.output_column

        return sma

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

        # Calculate the indicator
        sma = self.calculate(data)

        # Add the indicator values to the result
        result[self.output_column] = sma

        return result
