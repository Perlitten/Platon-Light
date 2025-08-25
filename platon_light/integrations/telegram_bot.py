"""
Telegram bot integration for remote monitoring and control
"""

import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


class TelegramBot:
    """
    Telegram bot for remote monitoring and control of the trading bot
    """

    def __init__(self, config: Dict, auth_users: List[int]):
        """
        Initialize the Telegram bot

        Args:
            config: Bot configuration
            auth_users: List of authorized user IDs
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.auth_users = auth_users
        self.token = config.get("telegram", {}).get("token", "")
        self.chat_ids = []

        # Callback functions
        self.callbacks = {
            "start_trading": None,
            "stop_trading": None,
            "get_status": None,
            "get_positions": None,
            "get_performance": None,
            "update_config": None,
            "close_position": None,
        }

        # Initialize bot
        self.app = None
        self.is_running = False

        self.logger.info("Telegram bot initialized")

    async def start(self):
        """Start the Telegram bot"""
        if not self.token:
            self.logger.warning(
                "Telegram bot token not provided, skipping initialization"
            )
            return

        try:
            # Create application
            self.app = Application.builder().token(self.token).build()

            # Add handlers
            self.app.add_handler(CommandHandler("start", self._handle_start))
            self.app.add_handler(CommandHandler("help", self._handle_help))
            self.app.add_handler(CommandHandler("status", self._handle_status))
            self.app.add_handler(CommandHandler("positions", self._handle_positions))
            self.app.add_handler(
                CommandHandler("performance", self._handle_performance)
            )
            self.app.add_handler(
                CommandHandler("start_trading", self._handle_start_trading)
            )
            self.app.add_handler(
                CommandHandler("stop_trading", self._handle_stop_trading)
            )
            self.app.add_handler(CommandHandler("config", self._handle_config))
            self.app.add_handler(CallbackQueryHandler(self._handle_callback))

            # Start the bot
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()

            self.is_running = True
            self.logger.info("Telegram bot started")

        except Exception as e:
            self.logger.error(f"Failed to start Telegram bot: {e}")

    async def stop(self):
        """Stop the Telegram bot"""
        if self.app and self.is_running:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                self.is_running = False
                self.logger.info("Telegram bot stopped")
            except Exception as e:
                self.logger.error(f"Failed to stop Telegram bot: {e}")

    def register_callback(self, name: str, callback: Callable):
        """
        Register a callback function

        Args:
            name: Callback name
            callback: Callback function
        """
        if name in self.callbacks:
            self.callbacks[name] = callback
            self.logger.debug(f"Registered callback: {name}")
        else:
            self.logger.warning(f"Unknown callback name: {name}")

    async def send_message(self, message: str, chat_id: Optional[int] = None):
        """
        Send a message to a chat

        Args:
            message: Message to send
            chat_id: Chat ID to send to, or None to send to all chats
        """
        if not self.is_running:
            return

        try:
            if chat_id:
                await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=telegram.constants.ParseMode.MARKDOWN,
                )
            else:
                for cid in self.chat_ids:
                    try:
                        await self.app.bot.send_message(
                            chat_id=cid,
                            text=message,
                            parse_mode=telegram.constants.ParseMode.MARKDOWN,
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to send message to chat {cid}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    async def send_trade_notification(self, trade_data: Dict):
        """
        Send a trade notification

        Args:
            trade_data: Trade data
        """
        if not self.is_running:
            return

        try:
            symbol = trade_data.get("symbol", "")
            side = trade_data.get("side", "")
            quantity = trade_data.get("quantity", 0)
            price = trade_data.get("price", 0)
            pnl = trade_data.get("pnl", 0)

            emoji = "🟢" if side.lower() == "buy" else "🔴"
            pnl_emoji = "✅" if pnl >= 0 else "❌"

            message = f"{emoji} *TRADE EXECUTED*\n\n"
            message += f"*Symbol:* {symbol}\n"
            message += f"*Side:* {side.upper()}\n"
            message += f"*Quantity:* {quantity}\n"
            message += f"*Price:* {price:.8f}\n"

            if "pnl" in trade_data:
                message += f"*PnL:* {pnl_emoji} {pnl:.8f}\n"

            if "reason" in trade_data:
                message += f"*Reason:* {trade_data['reason']}\n"

            await self.send_message(message)

        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {e}")

    async def send_performance_update(self, performance_data: Dict):
        """
        Send a performance update

        Args:
            performance_data: Performance data
        """
        if not self.is_running:
            return

        try:
            total_trades = performance_data.get("total_trades", 0)
            winning_trades = performance_data.get("winning_trades", 0)
            losing_trades = performance_data.get("losing_trades", 0)

            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

            total_profit = performance_data.get("total_profit", 0)
            total_loss = performance_data.get("total_loss", 0)
            net_profit = total_profit + total_loss  # total_loss is negative

            message = f"📊 *PERFORMANCE UPDATE*\n\n"
            message += f"*Total Trades:* {total_trades}\n"
            message += f"*Win Rate:* {win_rate:.2f}%\n"
            message += f"*Net Profit:* {net_profit:.8f}\n"

            if "daily_profit" in performance_data:
                message += f"*Daily Profit:* {performance_data['daily_profit']:.8f}\n"

            await self.send_message(message)

        except Exception as e:
            self.logger.error(f"Failed to send performance update: {e}")

    async def send_error_notification(self, error_message: str):
        """
        Send an error notification

        Args:
            error_message: Error message
        """
        if not self.is_running:
            return

        try:
            message = f"⚠️ *ERROR*\n\n{error_message}"
            await self.send_message(message)
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {e}")

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)

        await update.message.reply_text(
            f"👋 Welcome to Platon Light Trading Bot!\n\n"
            f"Use /help to see available commands."
        )

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        await update.message.reply_text(
            "*Available Commands:*\n\n"
            "/status - Get bot status\n"
            "/positions - View active positions\n"
            "/performance - View performance metrics\n"
            "/start_trading - Start trading\n"
            "/stop_trading - Stop trading\n"
            "/config - View and update configuration\n",
            parse_mode=telegram.constants.ParseMode.MARKDOWN,
        )

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        if self.callbacks["get_status"]:
            status = await self.callbacks["get_status"]()

            message = f"📈 *BOT STATUS*\n\n"
            message += f"*Running:* {'Yes' if status.get('running', False) else 'No'}\n"
            message += f"*Trading Mode:* {status.get('mode', 'Unknown')}\n"
            message += f"*Active Pairs:* {', '.join(status.get('symbols', []))}\n"
            message += f"*Uptime:* {status.get('uptime', 'Unknown')}\n"

            if "balance" in status:
                message += f"\n*Balance:* {status['balance']:.8f}\n"

            if "current_price" in status:
                message += f"*Current Price:* {status['current_price']:.8f}\n"

            await update.message.reply_text(
                message, parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text("Status callback not registered.")

    async def _handle_positions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /positions command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        if self.callbacks["get_positions"]:
            positions = await self.callbacks["get_positions"]()

            if not positions:
                await update.message.reply_text("No active positions.")
                return

            message = f"🔍 *ACTIVE POSITIONS*\n\n"

            for pos in positions:
                side_emoji = "🟢" if pos.get("side", "").lower() == "buy" else "🔴"

                message += f"{side_emoji} *{pos.get('symbol', '')}*\n"
                message += f"  Side: {pos.get('side', '').upper()}\n"
                message += f"  Quantity: {pos.get('quantity', 0)}\n"
                message += f"  Entry: {pos.get('entry_price', 0):.8f}\n"
                message += f"  Current: {pos.get('current_price', 0):.8f}\n"

                # Calculate PnL
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", 0)
                quantity = pos.get("quantity", 0)
                side = pos.get("side", "")

                if side.lower() == "buy":
                    pnl = (current_price - entry_price) * quantity
                    pnl_pct = ((current_price / entry_price) - 1) * 100
                else:  # sell
                    pnl = (entry_price - current_price) * quantity
                    pnl_pct = ((entry_price / current_price) - 1) * 100

                pnl_emoji = "✅" if pnl >= 0 else "❌"
                message += f"  PnL: {pnl_emoji} {pnl:.8f} ({pnl_pct:+.2f}%)\n\n"

                # Add close position button
                keyboard = [
                    [
                        InlineKeyboardButton(
                            f"Close {pos.get('symbol', '')}",
                            callback_data=f"close_{pos.get('id', '')}",
                        )
                    ],
                ]

            await update.message.reply_text(
                message,
                parse_mode=telegram.constants.ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard) if positions else None,
            )
        else:
            await update.message.reply_text("Positions callback not registered.")

    async def _handle_performance(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /performance command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        if self.callbacks["get_performance"]:
            performance = await self.callbacks["get_performance"]()

            message = f"📊 *PERFORMANCE METRICS*\n\n"

            total_trades = performance.get("total_trades", 0)
            winning_trades = performance.get("winning_trades", 0)
            losing_trades = performance.get("losing_trades", 0)

            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

            total_profit = performance.get("total_profit", 0)
            total_loss = performance.get("total_loss", 0)
            net_profit = total_profit + total_loss  # total_loss is negative

            message += f"*Total Trades:* {total_trades}\n"
            message += f"*Winning Trades:* {winning_trades}\n"
            message += f"*Losing Trades:* {losing_trades}\n"
            message += f"*Win Rate:* {win_rate:.2f}%\n\n"

            message += f"*Net Profit:* {net_profit:.8f}\n"
            message += f"*Total Profit:* {total_profit:.8f}\n"
            message += f"*Total Loss:* {total_loss:.8f}\n"

            if "daily_stats" in performance:
                message += f"\n*Daily Performance:*\n"

                for date, stats in performance["daily_stats"].items():
                    daily_profit = stats.get("profit", 0)
                    daily_trades = stats.get("trades", 0)
                    daily_win_rate = stats.get("win_rate", 0)

                    message += f"*{date}:* {daily_profit:.8f} ({daily_trades} trades, {daily_win_rate:.2f}% win)\n"

            await update.message.reply_text(
                message, parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text("Performance callback not registered.")

    async def _handle_start_trading(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /start_trading command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        if self.callbacks["start_trading"]:
            result = await self.callbacks["start_trading"]()

            if result.get("success", False):
                await update.message.reply_text("✅ Trading started successfully.")
            else:
                await update.message.reply_text(
                    f"❌ Failed to start trading: {result.get('error', 'Unknown error')}"
                )
        else:
            await update.message.reply_text("Start trading callback not registered.")

    async def _handle_stop_trading(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /stop_trading command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        if self.callbacks["stop_trading"]:
            result = await self.callbacks["stop_trading"]()

            if result.get("success", False):
                await update.message.reply_text("✅ Trading stopped successfully.")
            else:
                await update.message.reply_text(
                    f"❌ Failed to stop trading: {result.get('error', 'Unknown error')}"
                )
        else:
            await update.message.reply_text("Stop trading callback not registered.")

    async def _handle_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command"""
        user_id = update.effective_user.id

        if user_id not in self.auth_users:
            await update.message.reply_text("⚠️ You are not authorized to use this bot.")
            return

        # Get arguments
        args = context.args

        if not args:
            # Show config options
            keyboard = [
                [InlineKeyboardButton("Risk Settings", callback_data="config_risk")],
                [InlineKeyboardButton("Trading Pairs", callback_data="config_pairs")],
                [
                    InlineKeyboardButton(
                        "Strategy Parameters", callback_data="config_strategy"
                    )
                ],
            ]

            await update.message.reply_text(
                "Select configuration to view/update:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
        else:
            # Parse config update command
            # Format: /config parameter value
            if len(args) < 2:
                await update.message.reply_text(
                    "Invalid format. Use: /config parameter value"
                )
                return

            param = args[0]
            value = " ".join(args[1:])

            if self.callbacks["update_config"]:
                result = await self.callbacks["update_config"](param, value)

                if result.get("success", False):
                    await update.message.reply_text(
                        f"✅ Configuration updated: {param} = {value}"
                    )
                else:
                    await update.message.reply_text(
                        f"❌ Failed to update configuration: {result.get('error', 'Unknown error')}"
                    )
            else:
                await update.message.reply_text(
                    "Update config callback not registered."
                )

    async def _handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle callback queries"""
        query = update.callback_query
        user_id = query.from_user.id

        if user_id not in self.auth_users:
            await query.answer("You are not authorized to use this bot.")
            return

        await query.answer()

        callback_data = query.data

        if callback_data.startswith("close_"):
            # Close position
            position_id = callback_data[6:]

            if self.callbacks["close_position"]:
                result = await self.callbacks["close_position"](position_id)

                if result.get("success", False):
                    await query.edit_message_text(f"✅ Position closed successfully.")
                else:
                    await query.edit_message_text(
                        f"❌ Failed to close position: {result.get('error', 'Unknown error')}"
                    )
            else:
                await query.edit_message_text("Close position callback not registered.")

        elif callback_data.startswith("config_"):
            # Show config section
            section = callback_data[7:]

            if self.callbacks["get_status"]:
                status = await self.callbacks["get_status"]()
                config = status.get("config", {})

                if section == "risk":
                    message = "*Risk Management Settings*\n\n"

                    risk_config = config.get("risk_management", {})

                    message += f"*Max Position Size:* {risk_config.get('max_position_size', 'N/A')}\n"
                    message += f"*Max Open Positions:* {risk_config.get('max_open_positions', 'N/A')}\n"
                    message += f"*Daily Loss Limit:* {risk_config.get('daily_loss_limit', 'N/A')}\n"
                    message += (
                        f"*Max Drawdown:* {risk_config.get('max_drawdown', 'N/A')}\n"
                    )

                    await query.edit_message_text(
                        message, parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )

                elif section == "pairs":
                    message = "*Trading Pairs*\n\n"

                    pairs = config.get("trading", {}).get("symbols", [])

                    for pair in pairs:
                        message += f"• {pair}\n"

                    await query.edit_message_text(
                        message, parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )

                elif section == "strategy":
                    message = "*Strategy Parameters*\n\n"

                    strategy_config = config.get("strategy", {})

                    message += (
                        f"*RSI Period:* {strategy_config.get('rsi_period', 'N/A')}\n"
                    )
                    message += f"*RSI Overbought:* {strategy_config.get('rsi_overbought', 'N/A')}\n"
                    message += f"*RSI Oversold:* {strategy_config.get('rsi_oversold', 'N/A')}\n"
                    message += (
                        f"*MACD Fast:* {strategy_config.get('macd_fast', 'N/A')}\n"
                    )
                    message += (
                        f"*MACD Slow:* {strategy_config.get('macd_slow', 'N/A')}\n"
                    )
                    message += (
                        f"*MACD Signal:* {strategy_config.get('macd_signal', 'N/A')}\n"
                    )

                    await query.edit_message_text(
                        message, parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )
            else:
                await query.edit_message_text("Status callback not registered.")
