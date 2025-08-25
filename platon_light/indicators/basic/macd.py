#!/usr/bin/env python
"""
Moving Average Convergence Divergence (MACD) indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator
from .ema import EMA


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.

    The MACD is calculated by subtracting the long-term EMA from the short-term EMA.
    The signal line is an EMA of the MACD line. The histogram is the difference
    between the MACD line and the signal line.
    """

    def __init__(
        self,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        input_column="close",
        output_column=None,
    ):
        """
        Initialize the MACD indicator.

        Args:
            fast_period: Number of periods for the fast EMA
            slow_period: Number of periods for the slow EMA
            signal_period: Number of periods for the signal line EMA
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        # Default output column is the MACD line
        if output_column is None:
            output_column = f"MACD_{fast_period}_{slow_period}_{signal_period}"

        super().__init__(input_column=input_column, output_column=output_column)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @property
    def macd_column(self):
        """Get the MACD line column name."""
        return f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"

    @property
    def signal_column(self):
        """Get the signal line column name."""
        return f"MACD_Signal_{self.fast_period}_{self.slow_period}_{self.signal_period}"

    @property
    def histogram_column(self):
        """Get the histogram column name."""
        return f"MACD_Hist_{self.fast_period}_{self.slow_period}_{self.signal_period}"

    def calculate(self, data):
        """
        Calculate the MACD indicator.

        Args:
            data: DataFrame containing price data

        Returns:
            DataFrame with MACD line, signal line, and histogram
        """
        # Use the EMA indicator to calculate fast and slow EMAs
        fast_ema_indicator = EMA(
            period=self.fast_period, input_column=self.input_column
        )
        slow_ema_indicator = EMA(
            period=self.slow_period, input_column=self.input_column
        )

        # Get the EMA values
        fast_ema_result = fast_ema_indicator(data)
        slow_ema_result = slow_ema_indicator(data)

        fast_ema = fast_ema_result[f"EMA_{self.fast_period}"]
        slow_ema = slow_ema_result[f"EMA_{self.slow_period}"]

        # Calculate the MACD line
        macd_line = fast_ema - slow_ema

        # Calculate the signal line (EMA of MACD line)
        # Create a temporary DataFrame with MACD line as a column
        temp_df = pd.DataFrame({self.input_column: macd_line})
        signal_indicator = EMA(
            period=self.signal_period, input_column=self.input_column
        )
        signal_result = signal_indicator(temp_df)
        signal_line = signal_result[f"EMA_{self.signal_period}"]

        # Calculate the histogram
        histogram = macd_line - signal_line

        # Create a DataFrame with all components
        macd_data = pd.DataFrame(
            {
                self.macd_column: macd_line,
                self.signal_column: signal_line,
                self.histogram_column: histogram,
            },
            index=data.index,
        )

        return macd_data

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
        macd_data = self.calculate(data)

        # Add the indicator values to the result
        result[self.macd_column] = macd_data[self.macd_column]
        result[self.signal_column] = macd_data[self.signal_column]
        result[self.histogram_column] = macd_data[self.histogram_column]

        return result
