"""
CCXT-based Exchange client for interacting with Bybit API
"""
import logging
from typing import Dict, List, Optional, Union

import ccxt.async_support as ccxt
from .base_exchange import BaseExchangeClient

class BybitClient(BaseExchangeClient):
    """Client for interacting with Bybit exchange API using CCXT."""

    def __init__(self, config: Dict, api_key: str, api_secret: str):
        """
        Initialize the Bybit exchange client.
        Args:
            config: Bot configuration dictionary.
            api_key: The API key for the exchange.
            api_secret: The API secret for the exchange.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        # Per user request, this client connects to the Bybit Demo Trading account
        self.mode = "demo"
        self.market_type = config["general"].get("market_type", "spot")

        options = {'defaultType': self.market_type}

        self.ccxt_exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'options': options,
            'urls': {
                'api': 'https://api-demo.bybit.com',
            }
        })
        self.logger.info("Bybit client configured for Demo Trading account.")


    async def connect(self):
        """Connect to the exchange by loading markets."""
        self.logger.info(f"Connecting to Bybit ({self.market_type}) in {self.mode} mode via CCXT")
        try:
            await self.ccxt_exchange.load_markets()
            self.logger.info("Successfully connected to Bybit and loaded markets.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the exchange and clean up resources."""
        self.logger.info("Disconnecting from Bybit")
        await self.ccxt_exchange.close()
        self.logger.info("Successfully disconnected from Bybit")

    async def get_account_balance(self, asset: str = None) -> Union[Dict, float]:
        """Get account balance for a specific asset or all assets."""
        try:
            # For Bybit Unified account, we specify the account type
            balance = await self.ccxt_exchange.fetch_balance(params={'accountType': 'UNIFIED'})
            if asset:
                return balance.get(asset, {}).get('free', 0.0)
            else:
                return {currency: data['free'] for currency, data in balance['total'].items() if data['free'] > 0}
        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            return {} if not asset else 0.0

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            ticker = await self.ccxt_exchange.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            self.logger.error(f"Failed to get ticker price for {symbol}: {e}")
            return None

    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book for a symbol."""
        try:
            return await self.ccxt_exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            return None

    async def create_order(self, symbol: str, side: str, order_type: str,
                           quantity: float, price: float = None,
                           stop_price: float = None, reduce_only: bool = False) -> Optional[Dict]:
        """Create a new order."""
        try:
            params = {}
            if stop_price:
                params['stopPrice'] = stop_price
            if reduce_only:
                params['reduceOnly'] = True

            return await self.ccxt_exchange.create_order(symbol, order_type, side, quantity, price, params)
        except Exception as e:
            self.logger.error(f"Failed to create order for {symbol}: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await self.ccxt_exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Successfully cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
            return False

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders for a symbol or all symbols."""
        try:
            return await self.ccxt_exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol (futures only)."""
        if self.market_type != "futures":
            self.logger.warning("Position information is only available for futures")
            return None
        try:
            if self.ccxt_exchange.has['fetchPositions']:
                positions = await self.ccxt_exchange.fetch_positions([symbol])
                return positions[0] if positions else None
            else:
                self.logger.warning("Exchange does not support fetchPositions.")
                return None
        except Exception as e:
            self.logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol (futures only)."""
        if self.market_type != "futures":
            return False
        try:
            await self.ccxt_exchange.set_leverage(leverage, symbol)
            self.logger.info(f"Successfully set leverage for {symbol} to {leverage}x")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Set margin type for a symbol (futures only)."""
        if self.market_type != "futures":
            return False
        try:
            await self.ccxt_exchange.set_margin_mode(margin_type.lower(), symbol)
            self.logger.info(f"Successfully set margin mode for {symbol} to {margin_type}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set margin mode for {symbol}: {e}")
            return False

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get klines/candlestick data for a symbol (OHLCV)."""
        try:
            return await self.ccxt_exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        except Exception as e:
            self.logger.error(f"Failed to get klines for {symbol} ({interval}): {e}")
            return []
