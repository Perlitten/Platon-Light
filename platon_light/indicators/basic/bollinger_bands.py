#!/usr/bin/env python
"""
Bollinger Bands indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.

    Bollinger Bands consist of a middle band (SMA) with upper and lower bands
    at a specified number of standard deviations away from the middle band.
    """

    def __init__(self, period=20, std_dev=2, input_column="close", output_column=None):
        """
        Initialize the Bollinger Bands indicator.

        Args:
            period: Number of periods for the moving average
            std_dev: Number of standard deviations for the bands
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        # Convert std_dev to string for column names
        std_dev_str = str(std_dev)
        if isinstance(std_dev, float) and std_dev == int(std_dev):
            std_dev_str = str(int(std_dev))
        elif isinstance(std_dev, float):
            std_dev_str = str(std_dev).replace(".0", "")

        # Default output column is the middle band
        if output_column is None:
            output_column = f"BB_{period}_{std_dev_str}_middle"

        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period
        self.std_dev = std_dev
        self.std_dev_str = std_dev_str

    @property
    def middle_band_column(self):
        """Get the middle band column name."""
        return f"BB_{self.period}_{self.std_dev_str}_middle"

    @property
    def upper_band_column(self):
        """Get the upper band column name."""
        return f"BB_{self.period}_{self.std_dev_str}_upper"

    @property
    def lower_band_column(self):
        """Get the lower band column name."""
        return f"BB_{self.period}_{self.std_dev_str}_lower"

    def calculate(self, data):
        """
        Calculate the Bollinger Bands.

        Args:
            data: DataFrame containing price data

        Returns:
            DataFrame with middle, upper, and lower bands
        """
        # Calculate the middle band (SMA)
        middle_band = data[self.input_column].rolling(window=self.period).mean()

        # Calculate the standard deviation
        std = data[self.input_column].rolling(window=self.period).std()

        # Ensure std is not zero to avoid division by zero
        std = std.replace(0, np.nan)

        # Calculate the upper and lower bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)

        # Ensure the bands have the correct names
        middle_band.name = self.middle_band_column
        upper_band.name = self.upper_band_column
        lower_band.name = self.lower_band_column

        # Create a DataFrame with all bands
        bands = pd.DataFrame(
            {
                self.middle_band_column: middle_band,
                self.upper_band_column: upper_band,
                self.lower_band_column: lower_band,
            },
            index=data.index,
        )

        return bands

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
        bands = self.calculate(data)

        # Add the indicator values to the result
        result[self.middle_band_column] = bands[self.middle_band_column]
        result[self.upper_band_column] = bands[self.upper_band_column]
        result[self.lower_band_column] = bands[self.lower_band_column]

        return result
