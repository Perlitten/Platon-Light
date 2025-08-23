"""
Core Trading Bot implementation
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional

from platon_light.core.exchange_factory import get_exchange_client
from platon_light.core.strategy import ScalpingStrategy
from platon_light.core.risk_manager import RiskManager
from platon_light.core.position_manager import PositionManager
from platon_light.data.market_data import MarketDataManager
from platon_light.utils.performance import PerformanceTracker
from platon_light.integrations.telegram import TelegramManager
from platon_light.visualization.console import ConsoleVisualizer


class TradingBot:
    """Main trading bot class that coordinates all components"""
    
    def __init__(self, config: Dict):
        """
        Initialize the trading bot with the provided configuration
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.is_running = False
        self.start_time = None
        
        # Initialize components
        self.exchange = get_exchange_client(config)
        self.market_data = MarketDataManager(config, self.exchange)
        self.strategy = ScalpingStrategy(config, self.market_data)
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager(config, self.exchange, self.risk_manager)
        self.performance_tracker = PerformanceTracker(config)
        
        # Initialize optional components
        self.telegram = None
        if config.get("telegram", {}).get("enabled", False):
            self.telegram = TelegramManager(config, self)
        
        self.visualizer = None
        if config.get("visualization", {}).get("console", {}).get("enabled", False):
            self.visualizer = ConsoleVisualizer(config, self.market_data, self.position_manager)
        
        # Trading pairs
        self.base_currency = config["general"]["base_currency"]
        self.quote_currencies = config["general"]["quote_currencies"]
        self.trading_pairs = [f"{quote}{self.base_currency}" for quote in self.quote_currencies]
        
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
        
        # Close positions if configured
        if self.config.get("risk", {}).get("close_positions_on_stop", True):
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
        self.logger.info(f"Starting trading loop for pairs: {', '.join(self.trading_pairs)}")
        
        while self.is_running:
            try:
                # Check for risk management circuit breakers
                if not await self.risk_manager.check_trading_allowed():
                    self.logger.warning("Trading halted by risk management circuit breaker")
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
                            if await self.risk_manager.validate_trade(pair, signal, self.position_manager.get_positions()):
                                # Execute the trade
                                execution_start = time.time()
                                trade_result = await self.position_manager.execute_trade(pair, signal)
                                execution_time = (time.time() - execution_start) * 1000  # Convert to ms
                                
                                # Check execution latency
                                max_latency = self.config["general"].get("execution_timeout_ms", 300)
                                if execution_time > max_latency:
                                    self.logger.warning(f"Trade execution latency ({execution_time:.2f}ms) exceeded limit ({max_latency}ms)")
                                
                                # Track performance
                                self.performance_tracker.record_trade(trade_result)
                                
                                # Notify via telegram if enabled
                                if self.telegram and trade_result:
                                    await self.telegram.send_trade_notification(trade_result)
                        
                        # Manage existing positions
                        positions = self.position_manager.get_positions(pair)
                        for position in positions:
                            await self.position_manager.manage_position(position, market_data)
                    
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
            "mode": self.config["general"]["mode"],
            "trading_pairs": self.trading_pairs,
            "positions": self.position_manager.get_positions(),
            "performance": self.performance_tracker.get_summary(),
            "risk_status": self.risk_manager.get_status()
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
            return {"success": result, "message": f"Position for {pair} closed" if result else f"No position found for {pair}"}
        
        elif command == "close_all_positions":
            result = await self.position_manager.close_all_positions()
            return {"success": True, "message": f"Closed {result} positions"}
        
        elif command == "update_risk_params":
            for key, value in params.items():
                if key in self.config["risk"]:
                    self.config["risk"][key] = value
            
            self.risk_manager.update_config(self.config["risk"])
            return {"success": True, "message": "Risk parameters updated"}
        
        else:
            return {"success": False, "message": f"Unknown command: {command}"}
