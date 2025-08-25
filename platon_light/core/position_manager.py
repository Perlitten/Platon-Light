"""
Position manager for handling trade execution and position management
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import asyncio

from platon_light.core.base_exchange import BaseExchangeClient
from platon_light.core.risk_manager import RiskManager


class Position:
    """Class representing a trading position"""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        target_price: float,
        stop_price: float,
        entry_time: int,
        order_id: str = None,
    ):
        """
        Initialize a new position

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            direction: Position direction (long or short)
            entry_price: Entry price
            quantity: Position size
            target_price: Take profit price
            stop_price: Stop loss price
            entry_time: Entry timestamp in milliseconds
            order_id: Exchange order ID
        """
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.quantity = quantity
        self.initial_target_price = target_price
        self.target_price = target_price
        self.initial_stop_price = stop_price
        self.stop_price = stop_price
        self.entry_time = entry_time
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_percentage = 0.0
        self.status = "open"
        self.order_id = order_id
        self.tp_order_id = None
        self.sl_order_id = None
        self.trailing_activated = False
        self.max_price = entry_price if direction == "long" else 0
        self.min_price = entry_price if direction == "short" else float("inf")

    def update(self, current_price: float):
        """
        Update position with current market price

        Args:
            current_price: Current market price
        """
        # Update max/min price seen
        if self.direction == "long":
            self.max_price = max(self.max_price, current_price)
        else:
            self.min_price = min(self.min_price, current_price)

        # Calculate unrealized P&L
        if self.direction == "long":
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price

        self.pnl = price_diff * self.quantity
        self.pnl_percentage = (price_diff / self.entry_price) * 100

    def close(self, exit_price: float, exit_time: int):
        """
        Close the position

        Args:
            exit_price: Exit price
            exit_time: Exit timestamp in milliseconds
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = "closed"

        # Calculate realized P&L
        if self.direction == "long":
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price

        self.pnl = price_diff * self.quantity
        self.pnl_percentage = (price_diff / self.entry_price) * 100

    def to_dict(self) -> Dict:
        """
        Convert position to dictionary

        Returns:
            Position as dictionary
        """
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage,
            "status": self.status,
            "duration": (
                (self.exit_time - self.entry_time) / 1000 if self.exit_time else None
            ),
        }


class PositionManager:
    """
    Position manager for handling trade execution and position management
    """

    def __init__(
        self, config: Dict, exchange: BaseExchangeClient, risk_manager: RiskManager
    ):
        """
        Initialize the position manager

        Args:
            config: Bot configuration
            exchange: Exchange client instance
            risk_manager: Risk manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchange = exchange
        self.risk_manager = risk_manager

        # Trading settings
        self.market_type = config["general"]["market_type"]
        self.leverage = config["general"]["leverage"]
        self.is_dry_run = config["general"]["mode"] == "dry-run"

        # Exit settings
        self.exit_config = config["trading"]["exit"]
        self.trailing_stop = self.exit_config["stop_loss"]["type"] == "trailing"
        self.trailing_delta = self.exit_config["stop_loss"].get("trailing_delta", 0.1)
        self.max_trade_duration = self.exit_config.get(
            "max_trade_duration_seconds", 300
        )

        # Active positions
        self.positions = {}  # symbol -> Position
        self.closed_positions = []  # List of closed positions

        self.logger.info("Position manager initialized")

    async def execute_trade(self, symbol: str, signal: Dict, position_size: float) -> Optional[Dict]:
        """
        Execute a trade based on a signal

        Args:
            symbol: Trading pair symbol
            signal: Trading signal
            position_size: The calculated size of the position in the base currency.

        Returns:
            Trade result dictionary or None if execution failed
        """
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            self.logger.warning(f"Position already exists for {symbol}")
            return None

        # Get current price
        current_price = await self.exchange.get_ticker_price(symbol)
        if not current_price:
            self.logger.error(f"Failed to get current price for {symbol}")
            return None

        # Set up leverage for futures
        if self.market_type == "futures" and not self.is_dry_run:
            # Calculate optimal leverage based on volatility
            volatility = strategy._calculate_volatility(symbol)
            optimal_leverage = self.risk_manager.calculate_optimal_leverage(
                symbol, volatility
            )

            # Set leverage on exchange
            await self.exchange.set_leverage(symbol, optimal_leverage)

            # Set margin type (isolated)
            await self.exchange.set_margin_type(symbol, "ISOLATED")

        # Calculate quantity based on price
        quantity = position_size / current_price

        # Prepare order parameters
        direction = signal["direction"]
        side = "BUY" if direction == "long" else "SELL"

        # Execute the order
        order = await self.exchange.create_order(
            symbol=symbol, side=side, order_type="MARKET", quantity=quantity
        )

        if not order:
            self.logger.error(f"Failed to execute {side} order for {symbol}")
            return None

        # Get actual execution price
        if "fills" in order and order["fills"]:
            # Calculate weighted average price from fills
            total_qty = 0
            total_cost = 0
            for fill in order["fills"]:
                fill_qty = float(fill["qty"])
                fill_price = float(fill["price"])
                total_qty += fill_qty
                total_cost += fill_qty * fill_price

            entry_price = (
                total_cost / total_qty if total_qty > 0 else float(order["price"])
            )
        else:
            # Use order price if no fills available
            entry_price = float(order["price"]) if order.get("price") else current_price

        # Create position object
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=float(order["executedQty"]),
            target_price=signal["target_price"],
            stop_price=signal["stop_price"],
            entry_time=int(time.time() * 1000),
            order_id=order["orderId"],
        )

        # Store position
        self.positions[symbol] = position

        # Place take profit and stop loss orders for futures
        if self.market_type == "futures" and not self.is_dry_run:
            # Place take profit order
            tp_side = "SELL" if direction == "long" else "BUY"
            tp_order = await self.exchange.create_order(
                symbol=symbol,
                side=tp_side,
                order_type="LIMIT",
                quantity=position.quantity,
                price=position.target_price,
                reduce_only=True,
            )

            if tp_order:
                position.tp_order_id = tp_order["orderId"]

            # Place stop loss order
            sl_side = "SELL" if direction == "long" else "BUY"
            sl_order = await self.exchange.create_order(
                symbol=symbol,
                side=sl_side,
                order_type="STOP_MARKET",
                quantity=position.quantity,
                stop_price=position.stop_price,
                reduce_only=True,
            )

            if sl_order:
                position.sl_order_id = sl_order["orderId"]

        # Log the trade
        self.logger.info(
            f"Executed {direction.upper()} trade for {symbol}: {position.quantity} @ {entry_price} "
            f"(Target: {position.target_price}, Stop: {position.stop_price})"
        )

        # Return trade result
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "quantity": position.quantity,
            "target_price": position.target_price,
            "stop_price": position.stop_price,
            "entry_time": position.entry_time,
        }

    async def manage_position(self, position: Position, market_data: Dict) -> bool:
        """
        Manage an open position based on current market data

        Args:
            position: Position to manage
            market_data: Current market data

        Returns:
            True if position was closed, False otherwise
        """
        if position.status != "open":
            return False

        symbol = position.symbol
        current_price = market_data.get("close", 0)
        current_time = int(time.time() * 1000)

        if not current_price:
            self.logger.warning(f"No current price available for {symbol}")
            return False

        # Update position with current price
        position.update(current_price)

        # Check for take profit
        take_profit_hit = (
            position.direction == "long" and current_price >= position.target_price
        ) or (position.direction == "short" and current_price <= position.target_price)

        # Check for stop loss
        stop_loss_hit = (
            position.direction == "long" and current_price <= position.stop_price
        ) or (position.direction == "short" and current_price >= position.stop_price)

        # Check for max trade duration
        max_duration_exceeded = (
            current_time - position.entry_time
        ) / 1000 > self.max_trade_duration

        # Update trailing stop if enabled
        if self.trailing_stop and not position.trailing_activated:
            # Activate trailing stop when profit reaches half of target
            half_target_pnl = abs(position.entry_price - position.target_price) / 2

            if position.direction == "long" and current_price >= (
                position.entry_price + half_target_pnl
            ):
                position.trailing_activated = True
                self.logger.info(
                    f"Trailing stop activated for {symbol} at {current_price}"
                )

            elif position.direction == "short" and current_price <= (
                position.entry_price - half_target_pnl
            ):
                position.trailing_activated = True
                self.logger.info(
                    f"Trailing stop activated for {symbol} at {current_price}"
                )

        # Update trailing stop price if activated
        if self.trailing_stop and position.trailing_activated:
            if position.direction == "long":
                # Calculate new stop price based on trailing delta
                new_stop = position.max_price * (1 - self.trailing_delta / 100)

                # Only move stop up, never down
                if new_stop > position.stop_price:
                    old_stop = position.stop_price
                    position.stop_price = new_stop
                    self.logger.info(
                        f"Updated trailing stop for {symbol}: {old_stop:.8f} -> {new_stop:.8f}"
                    )

                    # Update stop loss order if using futures
                    if (
                        self.market_type == "futures"
                        and position.sl_order_id
                        and not self.is_dry_run
                    ):
                        # Cancel existing stop order
                        await self.exchange.cancel_order(symbol, position.sl_order_id)

                        # Place new stop order
                        sl_order = await self.exchange.create_order(
                            symbol=symbol,
                            side="SELL",
                            order_type="STOP_MARKET",
                            quantity=position.quantity,
                            stop_price=position.stop_price,
                            reduce_only=True,
                        )

                        if sl_order:
                            position.sl_order_id = sl_order["orderId"]

            else:  # short position
                # Calculate new stop price based on trailing delta
                new_stop = position.min_price * (1 + self.trailing_delta / 100)

                # Only move stop down, never up
                if new_stop < position.stop_price:
                    old_stop = position.stop_price
                    position.stop_price = new_stop
                    self.logger.info(
                        f"Updated trailing stop for {symbol}: {old_stop:.8f} -> {new_stop:.8f}"
                    )

                    # Update stop loss order if using futures
                    if (
                        self.market_type == "futures"
                        and position.sl_order_id
                        and not self.is_dry_run
                    ):
                        # Cancel existing stop order
                        await self.exchange.cancel_order(symbol, position.sl_order_id)

                        # Place new stop order
                        sl_order = await self.exchange.create_order(
                            symbol=symbol,
                            side="BUY",
                            order_type="STOP_MARKET",
                            quantity=position.quantity,
                            stop_price=position.stop_price,
                            reduce_only=True,
                        )

                        if sl_order:
                            position.sl_order_id = sl_order["orderId"]

        # Close position if any exit condition is met
        if take_profit_hit or stop_loss_hit or max_duration_exceeded:
            reason = (
                "take profit"
                if take_profit_hit
                else "stop loss" if stop_loss_hit else "max duration"
            )
            await self.close_position(symbol, reason)
            return True

        return False

    async def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """
        Close a position

        Args:
            symbol: Trading pair symbol
            reason: Reason for closing the position

        Returns:
            True if position was closed, False otherwise
        """
        if symbol not in self.positions:
            self.logger.warning(f"No position found for {symbol}")
            return False

        position = self.positions[symbol]

        # Get current price
        current_price = await self.exchange.get_ticker_price(symbol)
        if not current_price:
            self.logger.error(f"Failed to get current price for {symbol}")
            return False

        # Cancel any existing orders for futures
        if self.market_type == "futures" and not self.is_dry_run:
            if position.tp_order_id:
                await self.exchange.cancel_order(symbol, position.tp_order_id)

            if position.sl_order_id:
                await self.exchange.cancel_order(symbol, position.sl_order_id)

        # Execute closing order
        side = "SELL" if position.direction == "long" else "BUY"

        order = await self.exchange.create_order(
            symbol=symbol, side=side, order_type="MARKET", quantity=position.quantity
        )

        if not order:
            self.logger.error(f"Failed to close position for {symbol}")
            return False

        # Get actual execution price
        if "fills" in order and order["fills"]:
            # Calculate weighted average price from fills
            total_qty = 0
            total_cost = 0
            for fill in order["fills"]:
                fill_qty = float(fill["qty"])
                fill_price = float(fill["price"])
                total_qty += fill_qty
                total_cost += fill_qty * fill_price

            exit_price = (
                total_cost / total_qty if total_qty > 0 else float(order["price"])
            )
        else:
            # Use order price if no fills available
            exit_price = float(order["price"]) if order.get("price") else current_price

        # Close position
        position.close(exit_price, int(time.time() * 1000))

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]

        # Log the trade
        self.logger.info(
            f"Closed {position.direction.upper()} position for {symbol} at {exit_price} "
            f"(Reason: {reason}, P&L: {position.pnl:.2f} {self.config['general']['base_currency']} / {position.pnl_percentage:.2f}%)"
        )

        return True

    async def close_all_positions(self) -> int:
        """
        Close all open positions

        Returns:
            Number of positions closed
        """
        self.logger.info("Closing all open positions")

        symbols = list(self.positions.keys())
        closed_count = 0

        for symbol in symbols:
            if await self.close_position(symbol, "bot_shutdown"):
                closed_count += 1

            # Small delay to avoid rate limits
            await asyncio.sleep(0.2)

        return closed_count

    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Get all open positions or a specific position

        Args:
            symbol: Trading pair symbol or None for all positions

        Returns:
            List of position dictionaries
        """
        if symbol:
            if symbol in self.positions:
                return [self.positions[symbol].to_dict()]
            return []

        return [position.to_dict() for position in self.positions.values()]

    def get_closed_positions(self, limit: int = None) -> List[Dict]:
        """
        Get closed positions

        Args:
            limit: Maximum number of positions to return (most recent first)

        Returns:
            List of position dictionaries
        """
        positions = sorted(
            self.closed_positions, key=lambda p: p.exit_time or 0, reverse=True
        )

        if limit:
            positions = positions[:limit]

        return [position.to_dict() for position in positions]

    def get_position_summary(self) -> Dict:
        """
        Get summary of all positions

        Returns:
            Dictionary with position summary
        """
        open_positions = self.get_positions()
        closed_positions = self.get_closed_positions()

        # Calculate total P&L
        total_pnl = sum(p["pnl"] for p in closed_positions)
        total_pnl_percentage = (
            sum(p["pnl_percentage"] for p in closed_positions) / len(closed_positions)
            if closed_positions
            else 0
        )

        # Calculate win rate
        winning_trades = sum(1 for p in closed_positions if p["pnl"] > 0)
        win_rate = winning_trades / len(closed_positions) if closed_positions else 0

        # Calculate average trade duration
        durations = [
            p["duration"] for p in closed_positions if p["duration"] is not None
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "open_positions_count": len(open_positions),
            "closed_positions_count": len(closed_positions),
            "total_pnl": total_pnl,
            "total_pnl_percentage": total_pnl_percentage,
            "win_rate": win_rate,
            "avg_duration": avg_duration,
        }
