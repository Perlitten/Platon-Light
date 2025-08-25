#!/usr/bin/env python
"""
Strategy module for Platon Light backtesting framework.

This module defines the BaseStrategy class that all trading strategies should inherit from.
"""

import pandas as pd
import numpy as np


class BaseStrategy:
    """
    Base class for all trading strategies.

    All custom strategies should inherit from this class and implement
    the prepare_data and generate_signals methods.
    """

    def __init__(self):
        """Initialize the strategy."""
        pass

    def prepare_data(self, data):
        """
        Prepare data for the strategy by adding indicators.

        Args:
            data: DataFrame containing price data

        Returns:
            DataFrame with added indicators
        """
        # This method should be implemented by child classes
        return data

    def generate_signals(self, data):
        """
        Generate trading signals.

        Args:
            data: DataFrame containing price data and indicators

        Returns:
            DataFrame with added signal column
        """
        # This method should be implemented by child classes
        # Should add a 'signal' column with values:
        # 1 for buy, -1 for sell, 0 for no action
        return data

    def run(self, data):
        """
        Run the strategy on the given data.

        Args:
            data: DataFrame containing price data

        Returns:
            DataFrame with added indicators and signals
        """
        # Prepare data by adding indicators
        prepared_data = self.prepare_data(data)

        # Generate signals
        result = self.generate_signals(prepared_data)

        return result
