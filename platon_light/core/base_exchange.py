"""
Abstract Base Class for an exchange client.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class BaseExchangeClient(ABC):
    """
    Abstract interface for an exchange client, defining the common methods
    required for the trading bot to interact with different exchanges.
    """

    @abstractmethod
    def __init__(self, config: Dict, api_key: str, api_secret: str):
        """
        Initialize the exchange client.
        Args:
            config: Bot configuration dictionary.
            api_key: The API key for the exchange.
            api_secret: The API secret for the exchange.
        """
        raise NotImplementedError

    @abstractmethod
    async def connect(self):
        """Connect to the exchange and initialize clients."""
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the exchange and clean up resources."""
        raise NotImplementedError

    @abstractmethod
    async def get_account_balance(self, asset: str = None) -> Union[Dict, float]:
        """
        Get account balance for a specific asset or all assets.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Get order book for a symbol.
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        Create a new order.
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all open orders for a symbol or all symbols.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for a symbol (futures only).
        """
        raise NotImplementedError

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol (futures only).
        """
        raise NotImplementedError

    @abstractmethod
    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Set margin type for a symbol (futures only).
        """
        raise NotImplementedError

    @abstractmethod
    async def get_klines(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[List]:
        """
        Get klines/candlestick data for a symbol.
        """
        raise NotImplementedError
