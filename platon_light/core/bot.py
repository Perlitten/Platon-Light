"""
Core Trading Bot implementation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from platon_light.core.config_models import BotConfig
from platon_light.core.exchange_factory import get_exchange_client
from platon_light.core.strategy import ScalpingStrategy
from platon_light.core.risk_manager import RiskManager
from platon_light.core.position_manager import PositionManager
from platon_light.data.market_data import MarketDataManager
from platon_light.utils.performance import PerformanceTracker
from platon_light.integrations.telegram_bot import TelegramBot
from platon_light.utils.visualization import ConsoleVisualizer


class TradingBot:
    """Main trading bot class that coordinates all components"""

    def __init__(self, config: BotConfig):
        """
        Initialize the trading bot with the provided configuration

        Args:
            config: Pydantic model containing all configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.is_running = False
        self.start_time = None

        # Convert Pydantic config to dict for components that are not yet updated
        config_dict = config.model_dump()

        # Initialize components
        self.exchange = get_exchange_client(config_dict)
        self.market_data = MarketDataManager(config_dict, self.exchange)
        self.strategy = ScalpingStrategy(config_dict, self.market_data)
        self.risk_manager = RiskManager(config_dict)
        self.position_manager = PositionManager(
            config_dict, self.exchange, self.risk_manager
        )
        self.performance_tracker = PerformanceTracker(config_dict)

        # Initialize optional components
        self.telegram = None
        if self.config.telegram.enabled:
            # The TelegramBot expects a list of authorized user IDs, which we get from the config.
            # This assumes the config structure has `command_access.admin_users`.
            auth_users = self.config.telegram.command_access.get("admin_users", [])
            self.telegram = TelegramBot(config_dict, auth_users)

        self.visualizer = None
        if self.config.visualization.console.enabled:
            self.visualizer = ConsoleVisualizer(
                config_dict, self.market_data, self.position_manager
            )

        # Trading pairs
        self.base_currency = self.config.general.base_currency
        self.quote_currencies = self.config.general.quote_currencies
        self.trading_pairs = [
            f"{quote}{self.base_currency}" for quote in self.quote_currencies
        ]

        # Trading state
        self.active_pairs = set()
        self.tasks = []

    async def start(self):
        """Start the trading bot and all its components"""
        self.logger.info("Starting Platon Light Trading Bot")
        self.is_running = True
        self.start_time = time.time()

        # Connect to exchange
        await self.exchange.connect()

        # Initialize market data
        await self.market_data.initialize(self.trading_pairs)

        # Start telegram bot if enabled
        if self.telegram:
            self.tasks.append(asyncio.create_task(self.telegram.start()))

        # Start visualizer if enabled
        if self.visualizer:
            self.tasks.append(asyncio.create_task(self.visualizer.start()))

        # Start trading tasks
        self.tasks.append(asyncio.create_task(self._trading_loop()))

        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            self.logger.info("Bot tasks cancelled")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading bot and all its components"""
        self.logger.info("Stopping Platon Light Trading Bot")
        self.is_running = False

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Close positions if configured (assuming a default if not in risk config)
        # Note: This part of the config is not yet in the Pydantic model.
        # A default can be assumed or added to the model.
        # For now, we'll assume a default of True.
        if self.config.risk and hasattr(self.config.risk, 'close_positions_on_stop') and self.config.risk.close_positions_on_stop:
             await self.position_manager.close_all_positions()
        else: # Default behavior
             await self.position_manager.close_all_positions()


        # Disconnect from exchange
        await self.exchange.disconnect()

        # Stop telegram bot if enabled
        if self.telegram:
            await self.telegram.stop()

        # Stop visualizer if enabled
        if self.visualizer:
            await self.visualizer.stop()

        # Final performance report
        self.performance_tracker.generate_summary_report()

    async def _trading_loop(self):
        """Main trading loop that processes market data and executes the strategy"""
        self.logger.info(
            f"Starting trading loop for pairs: {', '.join(self.trading_pairs)}"
        )

        while self.is_running:
            try:
                # Check for risk management circuit breakers
                if not await self.risk_manager.check_trading_allowed():
                    self.logger.warning(
                        "Trading halted by risk management circuit breaker"
                    )
                    await asyncio.sleep(60)  # Check again after a minute
                    continue

                # Process each trading pair
                for pair in self.trading_pairs:
                    # Skip if pair is already being processed
                    if pair in self.active_pairs:
                        continue

                    self.active_pairs.add(pair)
                    try:
                        # Get latest market data
                        market_data = self.market_data.get_latest_data(pair)
                        if not market_data:
                            self.logger.warning(f"No market data available for {pair}")
                            continue

                        # Check if we should enter a position
                        signal = await self.strategy.generate_signal(pair, market_data)

                        if signal:
                            # Validate with risk manager
                            if await self.risk_manager.validate_trade(
                                pair, signal, self.position_manager.get_positions()
                            ):
                                # Calculate position size
                                balance = await self.exchange.get_account_balance(self.base_currency)
                                position_size = self.strategy.calculate_position_size(
                                    pair, signal, balance
                                )

                                # Execute the trade
                                execution_start = time.time()
                                trade_result = (
                                    await self.position_manager.execute_trade(
                                        pair, signal, position_size
                                    )
                                )
                                execution_time = (
                                    time.time() - execution_start
                                ) * 1000  # Convert to ms

                                # Check execution latency
                                max_latency = self.config.general.execution_timeout_ms
                                if execution_time > max_latency:
                                    self.logger.warning(
                                        f"Trade execution latency ({execution_time:.2f}ms) exceeded limit ({max_latency}ms)"
                                    )

                                # Track performance
                                self.performance_tracker.record_trade(trade_result)

                                # Notify via telegram if enabled
                                if self.telegram and trade_result:
                                    await self.telegram.send_trade_notification(
                                        trade_result
                                    )

                        # Manage existing positions
                        positions = self.position_manager.get_positions(pair)
                        for position in positions:
                            await self.position_manager.manage_position(
                                position, market_data
                            )

                    finally:
                        self.active_pairs.remove(pair)

                # Small delay to prevent high CPU usage
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.exception(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    def get_status(self) -> Dict:
        """Get the current status of the trading bot"""
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "running": self.is_running,
            "uptime": uptime,
            "mode": self.config.general.mode,
            "trading_pairs": self.trading_pairs,
            "positions": self.position_manager.get_positions(),
            "performance": self.performance_tracker.get_summary(),
            "risk_status": self.risk_manager.get_status(),
        }

    async def execute_command(self, command: str, params: Dict = None) -> Dict:
        """
        Execute a command from an external source (e.g., Telegram)

        Args:
            command: Command to execute
            params: Optional parameters for the command

        Returns:
            Dictionary with command result
        """
        if not params:
            params = {}

        if command == "status":
            return self.get_status()

        elif command == "stop":
            asyncio.create_task(self.stop())
            return {"success": True, "message": "Bot is shutting down"}

        elif command == "restart":
            asyncio.create_task(self.stop())
            asyncio.create_task(self.start())
            return {"success": True, "message": "Bot is restarting"}

        elif command == "close_position":
            pair = params.get("pair")
            if not pair:
                return {"success": False, "message": "Missing pair parameter"}

            result = await self.position_manager.close_position(pair)
            return {
                "success": result,
                "message": (
                    f"Position for {pair} closed"
                    if result
                    else f"No position found for {pair}"
                ),
            }

        elif command == "close_all_positions":
            result = await self.position_manager.close_all_positions()
            return {"success": True, "message": f"Closed {result} positions"}

        elif command == "update_risk_params":
            # This is more complex with Pydantic and requires careful implementation
            # For now, we'll log a warning that it's not fully supported.
            self.logger.warning("Runtime risk parameter updates are not fully supported with Pydantic model.")
            # A proper implementation would update the model and re-initialize components.
            return {"success": False, "message": "Runtime updates not supported in this version."}

        else:
            return {"success": False, "message": f"Unknown command: {command}"}
