#!/usr/bin/env python
"""
Exponential Moving Average (EMA) indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator


class EMA(BaseIndicator):
    """
    Exponential Moving Average (EMA) indicator.

    The Exponential Moving Average gives more weight to recent prices while
    still accounting for older prices with an exponentially decreasing weight.
    """

    def __init__(self, period=20, input_column="close", output_column=None):
        """
        Initialize the EMA indicator.

        Args:
            period: Number of periods for the moving average
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        if output_column is None:
            output_column = f"EMA_{period}"
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period

    @property
    def name(self):
        """Get the indicator name with period."""
        return f"EMA_{self.period}"

    def calculate(self, data):
        """
        Calculate the Exponential Moving Average.

        Args:
            data: DataFrame containing price data

        Returns:
            Series with EMA values
        """
        # Get the input data
        input_data = data[self.input_column]

        # Initialize with NaN values
        ema = pd.Series(np.nan, index=data.index)

        # Set the first valid value as the SMA for the first period
        first_valid_idx = input_data.first_valid_index()
        if first_valid_idx is not None:
            # Calculate SMA for the first period
            start_idx = data.index.get_loc(first_valid_idx)
            if start_idx + self.period <= len(data):
                sma_start = input_data.iloc[start_idx : start_idx + self.period].mean()
                ema.iloc[start_idx + self.period - 1] = sma_start

                # Calculate EMA for the rest of the series
                alpha = 2 / (self.period + 1)
                for i in range(start_idx + self.period, len(data)):
                    if pd.isna(input_data.iloc[i]):
                        ema.iloc[i] = np.nan
                    else:
                        ema.iloc[i] = input_data.iloc[i] * alpha + ema.iloc[i - 1] * (
                            1 - alpha
                        )

        # Ensure the Series has the correct name
        ema.name = self.output_column

        return ema

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
        ema = self.calculate(data)

        # Add the indicator values to the result
        result[self.output_column] = ema

        return result
