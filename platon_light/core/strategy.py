"""
Scalping strategy implementation for the trading bot
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from platon_light.data.market_data import MarketDataManager
from platon_light.utils.indicators import calculate_rsi, calculate_macd, calculate_stochastic
from platon_light.utils.order_book import analyze_order_book_imbalance


class ScalpingStrategy:
    """
    Advanced scalping strategy implementation utilizing momentum indicators,
    volume analysis, and order book imbalance detection
    """
    
    def __init__(self, config: Dict, market_data: MarketDataManager):
        """
        Initialize the scalping strategy
        
        Args:
            config: Strategy configuration
            market_data: Market data manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.market_data = market_data
        
        # Extract strategy parameters from config
        self.strategy_config = config["trading"]
        self.entry_config = self.strategy_config["entry"]
        self.exit_config = self.strategy_config["exit"]
        
        # Timeframes to analyze
        self.timeframes = self.strategy_config["timeframes"]
        
        # Signal cache to avoid recalculating signals too frequently
        self.signal_cache = {}
        self.signal_cache_expiry = 5  # seconds
        
    async def generate_signal(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """
        Generate trading signal for a symbol based on current market data
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            market_data: Current market data for the symbol
            
        Returns:
            Signal dictionary or None if no signal
        """
        # Check cache first
        cache_key = f"{symbol}_{int(time.time() / self.signal_cache_expiry)}"
        if cache_key in self.signal_cache:
            return self.signal_cache[cache_key]
            
        # Get current price
        current_price = market_data.get("close", 0)
        if not current_price:
            self.logger.warning(f"No current price available for {symbol}")
            return None
            
        # Get order book
        order_book = market_data.get("order_book", {})
        
        # Analyze each timeframe
        signals = []
        for timeframe in self.timeframes:
            # Get OHLCV data for the current timeframe
            ohlcv = self.market_data.get_ohlcv(symbol, timeframe)
            if ohlcv is None or len(ohlcv) < 30:  # Need enough data for indicators
                continue
                
            # Calculate indicators
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # RSI
            rsi_period = self.entry_config["rsi"]["period"]
            rsi = calculate_rsi(df["close"], rsi_period)
            
            # MACD
            fast_period = self.entry_config["macd"]["fast_period"]
            slow_period = self.entry_config["macd"]["slow_period"]
            signal_period = self.entry_config["macd"]["signal_period"]
            macd, macd_signal, macd_hist = calculate_macd(
                df["close"], fast_period, slow_period, signal_period
            )
            
            # Stochastic
            k_period = self.entry_config["stochastic"]["k_period"]
            d_period = self.entry_config["stochastic"]["d_period"]
            slowing = self.entry_config["stochastic"]["slowing"]
            stoch_k, stoch_d = calculate_stochastic(
                df["high"], df["low"], df["close"], k_period, d_period, slowing
            )
            
            # Volume analysis
            volume = df["volume"].values
            volume_sma = np.mean(volume[-20:])
            volume_percentile = np.percentile(volume[-50:], self.entry_config["min_volume_percentile"])
            high_volume = volume[-1] > volume_percentile
            
            # Order book imbalance
            if order_book and "bids" in order_book and "asks" in order_book:
                imbalance_ratio = analyze_order_book_imbalance(order_book)
                strong_imbalance = imbalance_ratio > self.entry_config["min_order_book_imbalance"]
            else:
                imbalance_ratio = 1.0
                strong_imbalance = False
                
            # Determine signal direction based on indicators
            signal_strength = 0
            signal_direction = None
            
            # Check for long signal
            if (rsi[-1] < self.entry_config["rsi"]["oversold"] and 
                macd_hist[-1] > macd_hist[-2] and 
                stoch_k[-1] < 30 and stoch_k[-1] > stoch_d[-1]):
                signal_direction = "long"
                signal_strength += 1
                
            # Check for short signal
            elif (rsi[-1] > self.entry_config["rsi"]["overbought"] and 
                  macd_hist[-1] < macd_hist[-2] and 
                  stoch_k[-1] > 70 and stoch_k[-1] < stoch_d[-1]):
                signal_direction = "short"
                signal_strength += 1
                
            # Add volume confirmation
            if high_volume and signal_direction:
                signal_strength += 1
                
            # Add order book confirmation
            if strong_imbalance:
                if (signal_direction == "long" and imbalance_ratio > 1) or \
                   (signal_direction == "short" and imbalance_ratio < 1):
                    signal_strength += 1
                    
            # Store timeframe signal
            if signal_direction and signal_strength >= 2:
                signals.append({
                    "timeframe": timeframe,
                    "direction": signal_direction,
                    "strength": signal_strength,
                    "indicators": {
                        "rsi": rsi[-1],
                        "macd_hist": macd_hist[-1],
                        "stoch_k": stoch_k[-1],
                        "stoch_d": stoch_d[-1],
                        "volume_ratio": volume[-1] / volume_sma,
                        "order_book_imbalance": imbalance_ratio
                    }
                })
                
        # Aggregate signals across timeframes
        if not signals:
            self.signal_cache[cache_key] = None
            return None
            
        # Count signals by direction
        long_signals = [s for s in signals if s["direction"] == "long"]
        short_signals = [s for s in signals if s["direction"] == "short"]
        
        # Determine final signal direction
        if len(long_signals) > len(short_signals) and len(long_signals) >= 2:
            direction = "long"
            side = "BUY"
            signals_used = long_signals
        elif len(short_signals) > len(long_signals) and len(short_signals) >= 2:
            direction = "short"
            side = "SELL"
            signals_used = short_signals
        else:
            self.signal_cache[cache_key] = None
            return None
            
        # Calculate entry price
        entry_price = current_price
        
        # Calculate profit target based on historical volatility
        volatility = self._calculate_volatility(symbol)
        if self.exit_config["profit_target"]["type"] == "dynamic":
            profit_target_pct = volatility * self.exit_config["profit_target"]["volatility_multiplier"]
        else:
            profit_target_pct = self.exit_config["profit_target"]["fixed_percentage"]
            
        # Calculate stop loss
        if self.exit_config["stop_loss"]["type"] == "fixed":
            stop_loss_pct = self.exit_config["stop_loss"]["initial_percentage"]
        else:
            # Adaptive stop loss based on volatility
            stop_loss_pct = min(
                volatility * 0.5,  # Half of volatility
                self.exit_config["stop_loss"]["initial_percentage"]  # But not more than configured
            )
            
        # Calculate target and stop prices
        if direction == "long":
            target_price = entry_price * (1 + profit_target_pct / 100)
            stop_price = entry_price * (1 - stop_loss_pct / 100)
        else:
            target_price = entry_price * (1 - profit_target_pct / 100)
            stop_price = entry_price * (1 + stop_loss_pct / 100)
            
        # Create signal
        signal = {
            "symbol": symbol,
            "direction": direction,
            "side": side,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "profit_target_pct": profit_target_pct,
            "stop_loss_pct": stop_loss_pct,
            "timeframes": [s["timeframe"] for s in signals_used],
            "strength": sum(s["strength"] for s in signals_used),
            "timestamp": int(time.time() * 1000),
            "indicators": {tf: s["indicators"] for tf, s in zip([s["timeframe"] for s in signals_used], signals_used)}
        }
        
        # Log signal
        self.logger.info(f"Generated {direction.upper()} signal for {symbol} at {entry_price}")
        self.logger.debug(f"Signal details: {signal}")
        
        # Cache the signal
        self.signal_cache[cache_key] = signal
        
        return signal
        
    def _calculate_volatility(self, symbol: str) -> float:
        """
        Calculate historical volatility for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Volatility as a percentage
        """
        # Get 1-minute data for the last 24 hours
        ohlcv = self.market_data.get_ohlcv(symbol, "1m", limit=1440)
        if not ohlcv or len(ohlcv) < 60:
            # Default volatility if not enough data
            return 0.5
            
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Calculate returns
        df["returns"] = df["close"].pct_change() * 100
        
        # Calculate volatility (standard deviation of returns)
        volatility = df["returns"].std()
        
        # Annualize volatility and scale down for short-term trading
        scaled_volatility = volatility * np.sqrt(1440) / 100  # Annualized
        scaled_volatility = scaled_volatility * 0.1  # Scale for short timeframe
        
        return max(0.1, min(scaled_volatility, 2.0))  # Clamp between 0.1% and 2%
        
    def calculate_position_size(self, symbol: str, signal: Dict, balance: float) -> float:
        """
        Calculate position size based on strategy configuration
        
        Args:
            symbol: Trading pair symbol
            signal: Trading signal
            balance: Available balance
            
        Returns:
            Position size
        """
        position_config = self.strategy_config["position_sizing"]
        method = position_config["method"]
        
        if method == "fixed":
            # Fixed position size
            return position_config.get("fixed_amount", 100)
            
        elif method == "percentage":
            # Percentage of balance
            percentage = position_config.get("percentage", 5.0)
            return balance * (percentage / 100)
            
        elif method == "kelly":
            # Kelly Criterion
            win_rate = self.market_data.get_win_rate(symbol) or 0.5
            profit_ratio = signal["profit_target_pct"] / signal["stop_loss_pct"]
            
            # Kelly formula: f = (bp - q) / b
            # where: b = odds received on wager, p = win probability, q = loss probability
            kelly_fraction = (profit_ratio * win_rate - (1 - win_rate)) / profit_ratio
            
            # Apply Kelly fraction and conservative multiplier
            kelly_fraction = max(0.01, min(kelly_fraction, 0.5))  # Clamp between 1% and 50%
            adjusted_kelly = kelly_fraction * position_config.get("kelly_fraction", 0.5)
            
            # Apply maximum position size limit
            max_percentage = position_config.get("max_position_size_percentage", 10.0)
            final_percentage = min(adjusted_kelly * 100, max_percentage)
            
            return balance * (final_percentage / 100)
            
        else:
            # Default to 5% of balance
            self.logger.warning(f"Unknown position sizing method: {method}, using default 5%")
            return balance * 0.05
