#!/usr/bin/env python
"""
Moving Average Crossover Strategy for Platon Light backtesting framework.

This strategy generates buy signals when the fast moving average crosses above the slow moving average,
and sell signals when the fast moving average crosses below the slow moving average.
"""

import pandas as pd
import numpy as np
from ..strategy import BaseStrategy
from ...indicators.basic import SMA, EMA, BollingerBands, RSI, MACD


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy uses two moving averages (fast and slow) to generate trading signals.
    A buy signal is generated when the fast MA crosses above the slow MA,
    and a sell signal is generated when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, fast_ma_type='SMA', slow_ma_type='SMA', 
                 fast_period=20, slow_period=50, 
                 rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                 use_filters=True):
        """
        Initialize the strategy.
        
        Args:
            fast_ma_type: Type of fast moving average ('SMA' or 'EMA')
            slow_ma_type: Type of slow moving average ('SMA' or 'EMA')
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            rsi_period: Period for the RSI indicator
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            use_filters: Whether to use additional filters (RSI, Bollinger Bands)
        """
        super().__init__()
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_filters = use_filters
        
        # Initialize indicators
        if fast_ma_type == 'SMA':
            self.fast_ma = SMA(period=fast_period, output_column=f'fast_ma')
        else:
            self.fast_ma = EMA(period=fast_period, output_column=f'fast_ma')
            
        if slow_ma_type == 'SMA':
            self.slow_ma = SMA(period=slow_period, output_column=f'slow_ma')
        else:
            self.slow_ma = EMA(period=slow_period, output_column=f'slow_ma')
        
        # Additional indicators for filtering
        self.rsi = RSI(period=rsi_period)
        self.bb = BollingerBands(period=20, std_dev=2)
        self.macd = MACD(fast_period=12, slow_period=26, signal_period=9)
    
    def prepare_data(self, data):
        """
        Prepare data for the strategy by adding indicators.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with added indicators
        """
        # Add moving averages
        data = self.fast_ma(data)
        data = self.slow_ma(data)
        
        # Add additional indicators if filters are enabled
        if self.use_filters:
            data = self.rsi(data)
            data = self.bb(data)
            data = self.macd(data)
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals.
        
        Args:
            data: DataFrame containing price data and indicators
            
        Returns:
            DataFrame with added signal column
        """
        # Initialize signal column with zeros (no action)
        data['signal'] = 0
        
        # Calculate crossovers
        data['crossover'] = 0
        
        # Get the column names
        fast_ma_col = 'fast_ma'
        slow_ma_col = 'slow_ma'
        rsi_col = f'RSI_{self.rsi_period}'
        
        # Calculate crossovers (1 for golden cross, -1 for death cross)
        for i in range(1, len(data)):
            # Check if fast MA crossed above slow MA (golden cross)
            if (data[fast_ma_col].iloc[i-1] <= data[slow_ma_col].iloc[i-1] and 
                data[fast_ma_col].iloc[i] > data[slow_ma_col].iloc[i]):
                data.loc[data.index[i], 'crossover'] = 1
            
            # Check if fast MA crossed below slow MA (death cross)
            elif (data[fast_ma_col].iloc[i-1] >= data[slow_ma_col].iloc[i-1] and 
                  data[fast_ma_col].iloc[i] < data[slow_ma_col].iloc[i]):
                data.loc[data.index[i], 'crossover'] = -1
        
        # Generate signals based on crossovers and filters
        for i in range(0, len(data)):
            # Skip if we don't have enough data for all indicators
            if pd.isna(data[fast_ma_col].iloc[i]) or pd.isna(data[slow_ma_col].iloc[i]):
                continue
                
            # Check for buy signal (golden cross)
            if data['crossover'].iloc[i] == 1:
                # Apply filters if enabled
                if self.use_filters:
                    # Check if RSI is not overbought and price is not above upper Bollinger Band
                    if (pd.isna(data[rsi_col].iloc[i]) or 
                        data[rsi_col].iloc[i] < self.rsi_overbought):
                        data.loc[data.index[i], 'signal'] = 1
                else:
                    data.loc[data.index[i], 'signal'] = 1
            
            # Check for sell signal (death cross)
            elif data['crossover'].iloc[i] == -1:
                # Apply filters if enabled
                if self.use_filters:
                    # Check if RSI is not oversold and price is not below lower Bollinger Band
                    if (pd.isna(data[rsi_col].iloc[i]) or 
                        data[rsi_col].iloc[i] > self.rsi_oversold):
                        data.loc[data.index[i], 'signal'] = -1
                else:
                    data.loc[data.index[i], 'signal'] = -1
        
        return data
