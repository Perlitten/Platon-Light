"""
Mock Exchange Client for testing purposes.
"""
import asyncio
from typing import Dict, List, Optional, Union

from platon_light.core.base_exchange import BaseExchangeClient


class MockExchangeClient(BaseExchangeClient):
    """
    A mock exchange client that simulates the behavior of a real exchange
    for testing the bot's logic without making real API calls.
    """

    def __init__(self, config: Dict, api_key: str, api_secret: str):
        """Initializes the mock client and a log to record method calls."""
        self.calls = []
        self.config = config
        self._is_connected = False
        self.balance = {"USDT": 10000.0, "BTC": 0.0}
        self.ticker_price = {"BTCUSDT": 50000.0}
        self.open_orders_data = {}
        self.positions = {}

    async def connect(self):
        """Simulates connecting to the exchange."""
        self.calls.append({"method": "connect", "params": {}})
        self._is_connected = True
        await asyncio.sleep(0.01)

    async def disconnect(self):
        """Simulates disconnecting from the exchange."""
        self.calls.append({"method": "disconnect", "params": {}})
        self._is_connected = False
        await asyncio.sleep(0.01)

    async def get_account_balance(self, asset: str = None) -> Union[Dict, float]:
        """Returns a mock account balance."""
        self.calls.append({"method": "get_account_balance", "params": {"asset": asset}})
        if asset:
            return self.balance.get(asset, 0.0)
        return self.balance

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Returns a mock ticker price."""
        self.calls.append({"method": "get_ticker_price", "params": {"symbol": symbol}})
        return self.ticker_price.get(symbol)

    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Returns a mock order book."""
        self.calls.append({"method": "get_order_book", "params": {"symbol": symbol, "limit": limit}})
        price = self.ticker_price.get(symbol, 50000.0)
        return {
            "bids": [[price - i, 1.0] for i in range(1, limit + 1)],
            "asks": [[price + i, 1.0] for i in range(1, limit + 1)],
        }

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
        stop_price: float = None,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Simulates creating an order and returns a mock order dictionary."""
        self.calls.append({
            "method": "create_order",
            "params": {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
            },
        })
        # Simulate order execution
        order_id = f"mock_order_{len(self.calls)}"
        self.balance["USDT"] -= quantity * (price or self.ticker_price.get(symbol, 50000))
        self.balance["BTC"] += quantity
        return {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "status": "closed",
            "price": price,
            "amount": quantity,
        }

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Simulates cancelling an order."""
        self.calls.append({"method": "cancel_order", "params": {"symbol": symbol, "order_id": order_id}})
        return True

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Returns a list of mock open orders."""
        self.calls.append({"method": "get_open_orders", "params": {"symbol": symbol}})
        if symbol:
            return self.open_orders_data.get(symbol, [])

        all_orders = []
        for orders in self.open_orders_data.values():
            all_orders.extend(orders)
        return all_orders

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Returns a mock position."""
        self.calls.append({"method": "get_position", "params": {"symbol": symbol}})
        return self.positions.get(symbol)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Simulates setting leverage."""
        self.calls.append({"method": "set_leverage", "params": {"symbol": symbol, "leverage": leverage}})
        return True

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Simulates setting margin type."""
        self.calls.append({"method": "set_margin_type", "params": {"symbol": symbol, "margin_type": margin_type}})
        return True

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[List]:
        """Returns mock kline data."""
        self.calls.append({"method": "get_klines", "params": {"symbol": symbol, "interval": interval, "limit": limit}})
        price = self.ticker_price.get(symbol, 50000.0)
        # Return a list of klines [timestamp, open, high, low, close, volume]
        return [
            [1672531200000 + i*60000, price, price+10, price-10, price, 100]
            for i in range(limit)
        ]
