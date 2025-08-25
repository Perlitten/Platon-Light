#!/usr/bin/env python
"""
Relative Strength Index (RSI) indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator


class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator.

    The RSI is a momentum oscillator that measures the speed and change of price movements.
    It oscillates between 0 and 100, with values above 70 typically considered overbought
    and values below 30 considered oversold.
    """

    def __init__(self, period=14, input_column="close", output_column=None):
        """
        Initialize the RSI indicator.

        Args:
            period: Number of periods for RSI calculation
            input_column: Column name to use as input
            output_column: Column name for the output
        """
        if output_column is None:
            output_column = f"RSI_{period}"
        super().__init__(input_column=input_column, output_column=output_column)
        self.period = period

    @property
    def name(self):
        """Get the indicator name with period."""
        return f"RSI_{self.period}"

    def calculate(self, data):
        """
        Calculate the Relative Strength Index.

        Args:
            data: DataFrame containing price data

        Returns:
            Series with RSI values
        """
        # Extract price data
        price = data[self.input_column]

        # Calculate price changes
        delta = price.diff()

        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gains and losses
        avg_gain = pd.Series(index=price.index)
        avg_loss = pd.Series(index=price.index)

        # First average is simple average
        avg_gain.iloc[self.period] = gains.iloc[1 : self.period + 1].mean()
        avg_loss.iloc[self.period] = losses.iloc[1 : self.period + 1].mean()

        # Subsequent averages use smoothing formula
        for i in range(self.period + 1, len(price)):
            avg_gain.iloc[i] = (
                avg_gain.iloc[i - 1] * (self.period - 1) + gains.iloc[i]
            ) / self.period
            avg_loss.iloc[i] = (
                avg_loss.iloc[i - 1] * (self.period - 1) + losses.iloc[i]
            ) / self.period

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
