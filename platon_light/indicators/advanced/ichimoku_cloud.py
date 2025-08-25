#!/usr/bin/env python
"""
Ichimoku Cloud indicator implementation.
"""

import pandas as pd
import numpy as np
from ..base import BaseIndicator


class IchimokuCloud(BaseIndicator):
    """
    Ichimoku Cloud indicator.

    The Ichimoku Cloud is a comprehensive indicator that shows support and resistance,
    momentum, and trend direction.
    """

    def __init__(
        self,
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52,
        displacement=26,
        input_columns=None,
        output_column=None,
    ):
        """
        Initialize the Ichimoku Cloud indicator.

        Args:
            tenkan_period: Period for Tenkan-sen (Conversion Line)
            kijun_period: Period for Kijun-sen (Base Line)
            senkou_b_period: Period for Senkou Span B
            displacement: Displacement for Senkou Span A and B (Kumo/Cloud)
            input_columns: Dict with column names for high, low, close
            output_column: Column name for the output
        """
        # Default input columns
        if input_columns is None:
            input_columns = {"high": "high", "low": "low", "close": "close"}

        super().__init__(input_column="close", output_column=output_column)
        self.input_columns = input_columns
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    @property
    def name(self):
        """Get the indicator name."""
        return "Ichimoku"

    @property
    def tenkan_column(self):
        """Get the Tenkan-sen column name."""
        return f"Ichimoku_Tenkan_{self.tenkan_period}"

    @property
    def kijun_column(self):
        """Get the Kijun-sen column name."""
        return f"Ichimoku_Kijun_{self.kijun_period}"

    @property
    def senkou_a_column(self):
        """Get the Senkou Span A column name."""
        return f"Ichimoku_Senkou_A_{self.tenkan_period}_{self.kijun_period}"

    @property
    def senkou_b_column(self):
        """Get the Senkou Span B column name."""
        return f"Ichimoku_Senkou_B_{self.senkou_b_period}"

    @property
    def chikou_column(self):
        """Get the Chikou Span column name."""
        return f"Ichimoku_Chikou_{self.displacement}"

    def _donchian(self, high, low, period):
        """
        Calculate the Donchian channel middle line.

        Args:
            high: Series of high prices
            low: Series of low prices
            period: Period for the Donchian channel

        Returns:
            Series with Donchian channel middle line values
        """
        # Calculate the highest high and lowest low over the period
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        # Calculate the middle line
        middle = (highest_high + lowest_low) / 2

        return middle

    def calculate(self, data):
        """
        Calculate the Ichimoku Cloud components.

        Args:
            data: DataFrame containing price data

        Returns:
            DataFrame with Ichimoku Cloud components
        """
        # Extract high, low, and close prices
        high = data[self.input_columns["high"]]
        low = data[self.input_columns["low"]]
        close = data[self.input_columns["close"]]

        # Calculate Tenkan-sen (Conversion Line)
        tenkan_sen = self._donchian(high, low, self.tenkan_period)

        # Calculate Kijun-sen (Base Line)
        kijun_sen = self._donchian(high, low, self.kijun_period)

        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)

        # Calculate Senkou Span B (Leading Span B)
        senkou_span_b = self._donchian(high, low, self.senkou_b_period).shift(
            self.displacement
        )

        # Calculate Chikou Span (Lagging Span)
        chikou_span = close.shift(-self.displacement)

        # Create a DataFrame with all components
        ichimoku_data = pd.DataFrame(
            {
                self.tenkan_column: tenkan_sen,
                self.kijun_column: kijun_sen,
                self.senkou_a_column: senkou_span_a,
                self.senkou_b_column: senkou_span_b,
                self.chikou_column: chikou_span,
            }
        )

        return ichimoku_data

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

        # Calculate the Ichimoku components
        ichimoku_data = self.calculate(data)

        # Add the components to the result
        for column in ichimoku_data.columns:
            result[column] = ichimoku_data[column]

        return result
