#!/usr/bin/env python
"""
Base indicator module for Platon Light.

This module defines the BaseIndicator class that all indicators should inherit from.
"""

import pandas as pd
import numpy as np


class BaseIndicator:
    """
    Base class for all technical indicators.

    All custom indicators should inherit from this class and implement
    the calculate method.
    """

    def __init__(self, input_column="close", output_column=None):
        """
        Initialize the indicator.

        Args:
            input_column: Column name to use as input for the indicator
            output_column: Column name for the output. If None, will use self.name
        """
        self.input_column = input_column
        self._output_column = output_column

    @property
    def name(self):
        """
        Get the indicator name.

        This should be overridden by child classes to provide a descriptive name.
        """
        return self.__class__.__name__

    @property
    def output_column(self):
        """Get the output column name."""
        if self._output_column is not None:
            return self._output_column
        return self.name

    def calculate(self, data):
        """
        Calculate the indicator.

        This method should be implemented by all child classes.

        Args:
            data: DataFrame containing price data

        Returns:
            Series with indicator values
        """
        raise NotImplementedError("Subclasses must implement calculate()")

    def __call__(self, data):
        """
        Apply the indicator to the data.

        Args:
            data: DataFrame containing price data

        Returns:
            DataFrame with indicator values added as a new column
        """
        # Make a copy to avoid modifying the original data
        result = data.copy()

        # Calculate the indicator
        indicator_values = self.calculate(data)

        # Add the indicator values to the result
        if isinstance(indicator_values, pd.DataFrame):
            # If the indicator returns a DataFrame, add all columns
            for column in indicator_values.columns:
                result[column] = indicator_values[column]
        else:
            # If the indicator returns a Series, add it with the output column name
            result[self.output_column] = indicator_values

        return result
