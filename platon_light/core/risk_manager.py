"""
Risk management system for the trading bot
"""

import logging
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class RiskManager:
    """
    Risk management system that implements various risk controls:
    - Per-trade risk limits
    - Daily drawdown circuit breakers
    - Position count limits
    - Correlation-based exposure limitations
    - Dynamic leverage adjustments
    - Liquidation risk calculator
    - Abnormal market condition detection
    """

    def __init__(self, config: Dict):
        """
        Initialize the risk manager

        Args:
            config: Bot configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.risk_config = config.get("risk", {})

        # Risk limits
        self.max_risk_per_trade = self.risk_config.get(
            "max_risk_per_trade_percentage", 1.0
        )
        self.max_open_positions = self.risk_config.get("max_open_positions", 3)
        self.correlation_limit = self.risk_config.get("correlation_limit", 0.7)

        # Circuit breakers
        self.daily_drawdown_warning = self.risk_config.get(
            "daily_drawdown_limits", {}
        ).get("warning", 3.0)
        self.daily_drawdown_soft_stop = self.risk_config.get(
            "daily_drawdown_limits", {}
        ).get("soft_stop", 5.0)
        self.daily_drawdown_hard_stop = self.risk_config.get(
            "daily_drawdown_limits", {}
        ).get("hard_stop", 10.0)

        # State variables
        self.daily_pnl = 0.0
        self.daily_pnl_percentage = 0.0
        self.starting_balance = 0.0
        self.current_balance = 0.0
        self.trading_suspended = False
        self.suspension_reason = None
        self.last_balance_update = 0

        # Abnormal market detection
        self.volatility_history = {}
        self.price_history = {}
        self.abnormal_market_detection = self.risk_config.get(
            "abnormal_market_detection", True
        )

        # Liquidation safety
        self.liquidation_safety_margin = self.risk_config.get(
            "liquidation_safety_margin_percentage", 20.0
        )

        self.logger.info("Risk manager initialized")

    def update_config(self, risk_config: Dict):
        """
        Update risk configuration

        Args:
            risk_config: New risk configuration
        """
        self.risk_config = risk_config

        # Update risk limits
        self.max_risk_per_trade = self.risk_config.get(
            "max_risk_per_trade_percentage", 1.0
        )
        self.max_open_positions = self.risk_config.get("max_open_positions", 3)
        self.correlation_limit = self.risk_config.get("correlation_limit", 0.7)

        # Update circuit breakers
        self.daily_drawdown_warning = self.risk_config.get(
            "daily_drawdown_limits", {}
        ).get("warning", 3.0)
        self.daily_drawdown_soft_stop = self.risk_config.get(
            "daily_drawdown_limits", {}
        ).get("soft_stop", 5.0)
        self.daily_drawdown_hard_stop = self.risk_config.get(
            "daily_drawdown_limits", {}
        ).get("hard_stop", 10.0)

        # Update other settings
        self.abnormal_market_detection = self.risk_config.get(
            "abnormal_market_detection", True
        )
        self.liquidation_safety_margin = self.risk_config.get(
            "liquidation_safety_margin_percentage", 20.0
        )

        self.logger.info("Risk configuration updated")

    def update_balance(self, balance: float):
        """
        Update current balance and calculate daily P&L

        Args:
            balance: Current account balance
        """
        current_time = time.time()

        # Initialize starting balance if not set
        if self.starting_balance == 0:
            self.starting_balance = balance
            self.logger.info(f"Initial balance set to {balance}")

        # Update current balance
        self.current_balance = balance

        # Calculate daily P&L
        self.daily_pnl = self.current_balance - self.starting_balance
        self.daily_pnl_percentage = (
            (self.daily_pnl / self.starting_balance) * 100
            if self.starting_balance > 0
            else 0
        )

        # Check for circuit breakers
        if self.daily_pnl_percentage <= -self.daily_drawdown_hard_stop:
            if not self.trading_suspended:
                self.logger.warning(
                    f"HARD STOP: Daily drawdown ({self.daily_pnl_percentage:.2f}%) exceeded hard limit ({self.daily_drawdown_hard_stop}%)"
                )
                self.trading_suspended = True
                self.suspension_reason = f"Daily drawdown ({self.daily_pnl_percentage:.2f}%) exceeded hard limit ({self.daily_drawdown_hard_stop}%)"

        elif self.daily_pnl_percentage <= -self.daily_drawdown_soft_stop:
            if not self.trading_suspended:
                self.logger.warning(
                    f"SOFT STOP: Daily drawdown ({self.daily_pnl_percentage:.2f}%) exceeded soft limit ({self.daily_drawdown_soft_stop}%)"
                )

        elif self.daily_pnl_percentage <= -self.daily_drawdown_warning:
            self.logger.warning(
                f"WARNING: Daily drawdown ({self.daily_pnl_percentage:.2f}%) exceeded warning level ({self.daily_drawdown_warning}%)"
            )

        self.last_balance_update = current_time

    def reset_daily_stats(self):
        """Reset daily statistics (called at the start of a new trading day)"""
        self.logger.info(
            f"Resetting daily stats. Previous day P&L: {self.daily_pnl:.2f} ({self.daily_pnl_percentage:.2f}%)"
        )
        self.starting_balance = self.current_balance
        self.daily_pnl = 0.0
        self.daily_pnl_percentage = 0.0
        self.trading_suspended = False
        self.suspension_reason = None

    async def validate_trade(
        self, symbol: str, signal: Dict, current_positions: List[Dict]
    ) -> bool:
        """
        Validate if a trade should be executed based on risk parameters

        Args:
            symbol: Trading pair symbol
            signal: Trading signal
            current_positions: List of current open positions

        Returns:
            True if trade is valid, False otherwise
        """
        # Check if trading is suspended
        if self.trading_suspended:
            self.logger.warning(
                f"Trade validation failed: Trading suspended due to {self.suspension_reason}"
            )
            return False

        # Check max open positions
        if len(current_positions) >= self.max_open_positions:
            self.logger.warning(
                f"Trade validation failed: Maximum open positions ({self.max_open_positions}) reached"
            )
            return False

        # Check if we already have a position for this symbol
        for position in current_positions:
            if position["symbol"] == symbol:
                self.logger.warning(
                    f"Trade validation failed: Position already exists for {symbol}"
                )
                return False

        # Check risk per trade
        risk_percentage = signal["stop_loss_pct"]
        if risk_percentage > self.max_risk_per_trade:
            self.logger.warning(
                f"Trade validation failed: Risk per trade ({risk_percentage:.2f}%) exceeds maximum ({self.max_risk_per_trade}%)"
            )
            return False

        # Check for abnormal market conditions
        if self.abnormal_market_detection and self.is_market_abnormal(symbol):
            self.logger.warning(
                f"Trade validation failed: Abnormal market conditions detected for {symbol}"
            )
            return False

        # Check correlation with existing positions
        if len(current_positions) > 0 and not self.check_correlation(
            symbol, current_positions
        ):
            self.logger.warning(
                f"Trade validation failed: {symbol} is too correlated with existing positions"
            )
            return False

        # Check liquidation risk for futures
        if self.config["general"][
            "market_type"
        ] == "futures" and not self.check_liquidation_risk(symbol, signal):
            self.logger.warning(
                f"Trade validation failed: Liquidation risk too high for {symbol}"
            )
            return False

        self.logger.info(f"Trade validation passed for {symbol}")
        return True

    async def check_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed based on risk parameters

        Returns:
            True if trading is allowed, False otherwise
        """
        # Check if trading is suspended
        if self.trading_suspended:
            self.logger.debug(f"Trading suspended due to {self.suspension_reason}")
            return False

        # Check if daily drawdown exceeds soft stop (allow manual override)
        if self.daily_pnl_percentage <= -self.daily_drawdown_soft_stop:
            self.logger.debug(
                f"Trading restricted: Daily drawdown ({self.daily_pnl_percentage:.2f}%) exceeded soft limit ({self.daily_drawdown_soft_stop}%)"
            )
            return False

        return True

    def check_correlation(self, symbol: str, current_positions: List[Dict]) -> bool:
        """
        Check if a symbol is too correlated with existing positions

        Args:
            symbol: Trading pair symbol
            current_positions: List of current open positions

        Returns:
            True if correlation is acceptable, False if too correlated
        """
        # If no correlation limit set, always allow
        if self.correlation_limit <= 0:
            return True

        # If no positions, no correlation
        if not current_positions:
            return True

        # Get price history for the symbol
        symbol_prices = self.price_history.get(symbol, [])
        if len(symbol_prices) < 30:
            # Not enough data to calculate correlation
            return True

        # Check correlation with each existing position
        for position in current_positions:
            pos_symbol = position["symbol"]
            pos_prices = self.price_history.get(pos_symbol, [])

            if len(pos_prices) < 30:
                continue

            # Ensure both price arrays are the same length
            min_length = min(len(symbol_prices), len(pos_prices))
            if min_length < 30:
                continue

            # Calculate correlation
            symbol_returns = (
                np.diff(symbol_prices[-min_length:]) / symbol_prices[-min_length:-1]
            )
            pos_returns = np.diff(pos_prices[-min_length:]) / pos_prices[-min_length:-1]

            correlation = np.corrcoef(symbol_returns, pos_returns)[0, 1]

            # Check if correlation exceeds limit
            if abs(correlation) > self.correlation_limit:
                self.logger.debug(
                    f"Correlation between {symbol} and {pos_symbol} is {correlation:.2f}, exceeding limit {self.correlation_limit}"
                )
                return False

        return True

    def check_liquidation_risk(self, symbol: str, signal: Dict) -> bool:
        """
        Check if a trade has acceptable liquidation risk for futures trading

        Args:
            symbol: Trading pair symbol
            signal: Trading signal

        Returns:
            True if liquidation risk is acceptable, False otherwise
        """
        if self.config["general"]["market_type"] != "futures":
            return True

        # Get leverage
        leverage = self.config["general"]["leverage"]

        # Calculate liquidation price
        entry_price = signal["entry_price"]
        direction = signal["direction"]
        stop_price = signal["stop_price"]

        # Calculate distance to stop as percentage
        stop_distance_pct = abs(entry_price - stop_price) / entry_price * 100

        # Calculate liquidation distance based on leverage
        # Simplified formula: liquidation occurs at approximately 1/leverage distance
        liquidation_distance_pct = (
            100 / leverage * (1 - 1 / self.liquidation_safety_margin)
        )

        # Check if stop is too close to potential liquidation
        if stop_distance_pct < liquidation_distance_pct:
            self.logger.warning(
                f"Liquidation risk: Stop at {stop_distance_pct:.2f}% is too close to liquidation threshold {liquidation_distance_pct:.2f}% "
                f"with {leverage}x leverage"
            )
            return False

        return True

    def is_market_abnormal(self, symbol: str) -> bool:
        """
        Detect abnormal market conditions based on volatility and price action

        Args:
            symbol: Trading pair symbol

        Returns:
            True if market conditions are abnormal, False otherwise
        """
        if not self.abnormal_market_detection:
            return False

        # Get volatility history
        volatility = self.volatility_history.get(symbol, [])
        if len(volatility) < 10:
            return False

        # Get price history
        prices = self.price_history.get(symbol, [])
        if len(prices) < 30:
            return False

        # Check for abnormal volatility (3x the average)
        avg_volatility = np.mean(volatility[:-1])
        current_volatility = volatility[-1]

        if current_volatility > avg_volatility * 3:
            self.logger.warning(
                f"Abnormal volatility detected for {symbol}: {current_volatility:.2f}% (avg: {avg_volatility:.2f}%)"
            )
            return True

        # Check for abnormal price movement (5% in last 5 minutes)
        if len(prices) >= 5:
            price_change = abs(prices[-1] - prices[-5]) / prices[-5] * 100
            if price_change > 5:
                self.logger.warning(
                    f"Abnormal price movement detected for {symbol}: {price_change:.2f}% in 5 minutes"
                )
                return True

        return False

    def update_market_data(self, symbol: str, price: float, volatility: float):
        """
        Update market data for risk calculations

        Args:
            symbol: Trading pair symbol
            price: Current price
            volatility: Current volatility estimate
        """
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(price)

        # Keep only the last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

        # Update volatility history
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []

        self.volatility_history[symbol].append(volatility)

        # Keep only the last 20 volatility measurements
        if len(self.volatility_history[symbol]) > 20:
            self.volatility_history[symbol] = self.volatility_history[symbol][-20:]

    def get_status(self) -> Dict:
        """
        Get current risk management status

        Returns:
            Dictionary with risk status information
        """
        return {
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percentage": self.daily_pnl_percentage,
            "trading_suspended": self.trading_suspended,
            "suspension_reason": self.suspension_reason,
            "max_open_positions": self.max_open_positions,
            "max_risk_per_trade": self.max_risk_per_trade,
            "circuit_breakers": {
                "warning": self.daily_drawdown_warning,
                "soft_stop": self.daily_drawdown_soft_stop,
                "hard_stop": self.daily_drawdown_hard_stop,
            },
        }

    def calculate_optimal_leverage(self, symbol: str, volatility: float) -> int:
        """
        Calculate optimal leverage based on market volatility

        Args:
            symbol: Trading pair symbol
            volatility: Current volatility estimate (percentage)

        Returns:
            Optimal leverage value
        """
        if not self.risk_config.get("volatility_adjustment", True):
            return self.config["general"]["leverage"]

        # Base leverage from config
        base_leverage = self.config["general"]["leverage"]

        # Adjust based on volatility
        # Higher volatility = lower leverage
        if volatility <= 0.5:
            # Low volatility
            leverage = base_leverage
        elif volatility <= 1.0:
            # Medium volatility
            leverage = max(1, base_leverage - 1)
        elif volatility <= 2.0:
            # High volatility
            leverage = max(1, base_leverage - 2)
        else:
            # Very high volatility
            leverage = max(1, base_leverage - 3)

        return leverage
