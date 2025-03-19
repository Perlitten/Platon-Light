"""
Exchange client for interacting with Binance API
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import hmac
import hashlib
from urllib.parse import urlencode

import aiohttp
from binance.client import Client
from binance.exceptions import BinanceAPIException

class ExchangeClient:
    """Client for interacting with Binance exchange API"""
    
    def __init__(self, config: Dict):
        """
        Initialize the exchange client
        
        Args:
            config: Bot configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.mode = config["general"]["mode"]
        self.is_dry_run = self.mode == "dry-run"
        self.market_type = config["general"]["market_type"]
        
        # API credentials
        self.api_key = config.get("api_key") or self._get_env_var("BINANCE_API_KEY")
        self.api_secret = config.get("api_secret") or self._get_env_var("BINANCE_API_SECRET")
        
        # Client instances
        self.rest_client = None
        self.session = None
        
        # Execution settings
        self.max_retries = config.get("advanced", {}).get("execution", {}).get("retry_attempts", 3)
        self.retry_delay = config.get("advanced", {}).get("execution", {}).get("retry_delay_ms", 100) / 1000
        
        # Rate limiting
        self.last_request_time = 0
        self.request_weight = 0
        self.request_count = 0
        
        # Exchange info
        self.exchange_info = {}
        self.symbol_info = {}
        self.leverage_brackets = {}
        
    def _get_env_var(self, name: str) -> Optional[str]:
        """Get environment variable safely"""
        import os
        return os.environ.get(name)
        
    async def connect(self):
        """Connect to the exchange and initialize clients"""
        self.logger.info(f"Connecting to Binance ({self.market_type}) in {self.mode} mode")
        
        # Initialize REST client
        self.rest_client = Client(self.api_key, self.api_secret)
        
        # Initialize HTTP session for async requests
        self.session = aiohttp.ClientSession()
        
        # Load exchange information
        await self._load_exchange_info()
        
        if self.market_type == "futures" and not self.is_dry_run:
            # Load leverage brackets for futures
            await self._load_leverage_brackets()
        
        self.logger.info("Successfully connected to Binance")
        
    async def disconnect(self):
        """Disconnect from the exchange and clean up resources"""
        self.logger.info("Disconnecting from Binance")
        
        if self.session:
            await self.session.close()
            self.session = None
            
        self.logger.info("Successfully disconnected from Binance")
        
    async def _load_exchange_info(self):
        """Load exchange information including trading rules"""
        try:
            if self.market_type == "futures":
                exchange_info = self.rest_client.futures_exchange_info()
            else:
                exchange_info = self.rest_client.get_exchange_info()
                
            self.exchange_info = exchange_info
            
            # Process symbol info
            for symbol in exchange_info["symbols"]:
                self.symbol_info[symbol["symbol"]] = symbol
                
            self.logger.info(f"Loaded exchange info for {len(self.symbol_info)} symbols")
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to load exchange info: {e}")
            raise
            
    async def _load_leverage_brackets(self):
        """Load leverage brackets for futures trading"""
        try:
            brackets = self.rest_client.futures_leverage_bracket()
            
            for item in brackets:
                symbol = item["symbol"]
                self.leverage_brackets[symbol] = item["brackets"]
                
            self.logger.info(f"Loaded leverage brackets for {len(self.leverage_brackets)} symbols")
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to load leverage brackets: {e}")
            
    async def get_account_balance(self, asset: str = None) -> Union[Dict, float]:
        """
        Get account balance for a specific asset or all assets
        
        Args:
            asset: Asset symbol (e.g., USDT). If None, returns all balances.
            
        Returns:
            Dictionary of all balances or float value for specific asset
        """
        if self.is_dry_run:
            # Return simulated balance from config
            sim_balance = self.config.get("simulation", {}).get("balance", 10000)
            if asset:
                return float(sim_balance)
            else:
                return {self.config["general"]["base_currency"]: float(sim_balance)}
        
        try:
            if self.market_type == "futures":
                account = await self._make_request("GET", "/fapi/v2/account")
                balances = {asset["asset"]: float(asset["balance"]) for asset in account["assets"]}
            else:
                account = await self._make_request("GET", "/api/v3/account")
                balances = {asset["asset"]: float(asset["free"]) for asset in account["balances"]}
            
            if asset:
                return balances.get(asset, 0.0)
            return balances
            
        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            if asset:
                return 0.0
            return {}
            
    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            
        Returns:
            Current price as float or None if error
        """
        try:
            endpoint = "/fapi/v1/ticker/price" if self.market_type == "futures" else "/api/v3/ticker/price"
            params = {"symbol": symbol}
            
            ticker = await self._make_request("GET", endpoint, params)
            return float(ticker["price"])
            
        except Exception as e:
            self.logger.error(f"Failed to get ticker price for {symbol}: {e}")
            return None
            
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Get order book for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            limit: Number of price levels to return (max 1000)
            
        Returns:
            Order book dictionary or None if error
        """
        try:
            endpoint = "/fapi/v1/depth" if self.market_type == "futures" else "/api/v3/depth"
            params = {"symbol": symbol, "limit": limit}
            
            order_book = await self._make_request("GET", endpoint, params)
            
            # Convert strings to floats
            for side in ["bids", "asks"]:
                order_book[side] = [[float(price), float(qty)] for price, qty in order_book[side]]
                
            return order_book
            
        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            return None
            
    async def create_order(self, symbol: str, side: str, order_type: str, 
                          quantity: float, price: float = None, 
                          stop_price: float = None, reduce_only: bool = False) -> Optional[Dict]:
        """
        Create a new order
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT, MARKET, STOP_MARKET, etc.)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)
            reduce_only: Whether the order should only reduce position (futures only)
            
        Returns:
            Order information dictionary or None if error
        """
        if self.is_dry_run:
            # Simulate order creation in dry run mode
            order_id = f"dry_run_{int(time.time() * 1000)}"
            
            simulated_order = {
                "symbol": symbol,
                "orderId": order_id,
                "clientOrderId": f"simulated_{order_id}",
                "transactTime": int(time.time() * 1000),
                "price": str(price) if price else "0",
                "origQty": str(quantity),
                "executedQty": str(quantity),
                "status": "FILLED",
                "timeInForce": "GTC",
                "type": order_type,
                "side": side,
                "fills": [
                    {
                        "price": str(price if price else await self.get_ticker_price(symbol)),
                        "qty": str(quantity),
                        "commission": "0",
                        "commissionAsset": symbol[-4:]
                    }
                ]
            }
            
            self.logger.info(f"[DRY RUN] Created {side} {order_type} order for {symbol}: {quantity} @ {price if price else 'MARKET'}")
            return simulated_order
            
        # Prepare order parameters
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": self._format_quantity(symbol, quantity)
        }
        
        if order_type == "LIMIT":
            params["timeInForce"] = "GTC"
            params["price"] = self._format_price(symbol, price)
            
        if stop_price:
            params["stopPrice"] = self._format_price(symbol, stop_price)
            
        if reduce_only and self.market_type == "futures":
            params["reduceOnly"] = "true"
            
        # Execute the order with retry logic
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.market_type == "futures":
                    endpoint = "/fapi/v1/order"
                else:
                    endpoint = "/api/v3/order"
                    
                order = await self._make_request("POST", endpoint, params)
                self.logger.info(f"Created {side} {order_type} order for {symbol}: {quantity} @ {price if price else 'MARKET'}")
                return order
                
            except BinanceAPIException as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Order attempt {attempt} failed: {e}. Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to create order after {self.max_retries} attempts: {e}")
                    return None
                    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if self.is_dry_run:
            self.logger.info(f"[DRY RUN] Cancelled order {order_id} for {symbol}")
            return True
            
        try:
            endpoint = "/fapi/v1/order" if self.market_type == "futures" else "/api/v3/order"
            params = {
                "symbol": symbol,
                "orderId": order_id
            }
            
            await self._make_request("DELETE", endpoint, params)
            self.logger.info(f"Cancelled order {order_id} for {symbol}")
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
            return False
            
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all open orders for a symbol or all symbols
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT). If None, returns all open orders.
            
        Returns:
            List of open orders
        """
        if self.is_dry_run:
            return []
            
        try:
            endpoint = "/fapi/v1/openOrders" if self.market_type == "futures" else "/api/v3/openOrders"
            params = {}
            
            if symbol:
                params["symbol"] = symbol
                
            orders = await self._make_request("GET", endpoint, params)
            return orders
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
            
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for a symbol (futures only)
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            
        Returns:
            Position information or None if no position or error
        """
        if self.market_type != "futures":
            self.logger.warning("Position information is only available for futures")
            return None
            
        if self.is_dry_run:
            # Return empty position in dry run mode
            return {
                "symbol": symbol,
                "positionAmt": "0",
                "entryPrice": "0",
                "markPrice": "0",
                "unRealizedProfit": "0",
                "liquidationPrice": "0",
                "leverage": str(self.config["general"]["leverage"]),
                "marginType": "isolated"
            }
            
        try:
            endpoint = "/fapi/v2/positionRisk"
            params = {"symbol": symbol}
            
            positions = await self._make_request("GET", endpoint, params)
            
            # Find the position for the specified symbol
            for position in positions:
                if position["symbol"] == symbol:
                    return position
                    
            return None
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get position for {symbol}: {e}")
            return None
            
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol (futures only)
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            leverage: Leverage value (1-125)
            
        Returns:
            True if successful, False otherwise
        """
        if self.market_type != "futures":
            self.logger.warning("Leverage setting is only available for futures")
            return False
            
        if self.is_dry_run:
            self.logger.info(f"[DRY RUN] Set leverage for {symbol} to {leverage}x")
            return True
            
        try:
            endpoint = "/fapi/v1/leverage"
            params = {
                "symbol": symbol,
                "leverage": leverage
            }
            
            await self._make_request("POST", endpoint, params)
            self.logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False
            
    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Set margin type for a symbol (futures only)
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            margin_type: Margin type (ISOLATED or CROSSED)
            
        Returns:
            True if successful, False otherwise
        """
        if self.market_type != "futures":
            self.logger.warning("Margin type setting is only available for futures")
            return False
            
        if self.is_dry_run:
            self.logger.info(f"[DRY RUN] Set margin type for {symbol} to {margin_type}")
            return True
            
        try:
            endpoint = "/fapi/v1/marginType"
            params = {
                "symbol": symbol,
                "marginType": margin_type
            }
            
            await self._make_request("POST", endpoint, params)
            self.logger.info(f"Set margin type for {symbol} to {margin_type}")
            return True
            
        except BinanceAPIException as e:
            # If already in the desired margin type, consider it successful
            if "Already" in str(e):
                return True
                
            self.logger.error(f"Failed to set margin type for {symbol}: {e}")
            return False
            
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """
        Get klines/candlestick data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            interval: Kline interval (e.g., 1m, 5m, 1h)
            limit: Number of klines to return (max 1000)
            
        Returns:
            List of klines
        """
        try:
            endpoint = "/fapi/v1/klines" if self.market_type == "futures" else "/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            klines = await self._make_request("GET", endpoint, params)
            
            # Format klines
            formatted_klines = []
            for k in klines:
                formatted_klines.append([
                    k[0],  # Open time
                    float(k[1]),  # Open
                    float(k[2]),  # High
                    float(k[3]),  # Low
                    float(k[4]),  # Close
                    float(k[5]),  # Volume
                    k[6],  # Close time
                    float(k[7]),  # Quote asset volume
                    k[8],  # Number of trades
                    float(k[9]),  # Taker buy base asset volume
                    float(k[10]),  # Taker buy quote asset volume
                    k[11]  # Ignore
                ])
                
            return formatted_klines
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get klines for {symbol} ({interval}): {e}")
            return []
            
    async def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """
        Make an authenticated request to the Binance API
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data
        """
        if not self.session:
            raise RuntimeError("HTTP session not initialized. Call connect() first.")
            
        # Rate limiting: ensure we don't exceed API limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 0.05:  # Limit to 20 requests per second
            await asyncio.sleep(0.05 - time_since_last)
            
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Prepare request
        url = f"https://{'fapi' if self.market_type == 'futures' else 'api'}.binance.com{endpoint}"
        
        # Add timestamp for authenticated endpoints
        if params is None:
            params = {}
            
        if method != "GET" or endpoint.startswith("/api/v3/account") or endpoint.startswith("/fapi/v"):
            params["timestamp"] = int(time.time() * 1000)
            
        # Generate signature
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        params["signature"] = signature
        
        headers = {
            "X-MBX-APIKEY": self.api_key
        }
        
        # Make request
        async with getattr(self.session, method.lower())(
            url, params=params, headers=headers
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise BinanceAPIException(f"API request failed: {response.status} - {text}")
                
            return await response.json()
            
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol's tick size"""
        if symbol not in self.symbol_info:
            return str(price)
            
        # Find the price filter
        filters = self.symbol_info[symbol]["filters"]
        price_filter = next((f for f in filters if f["filterType"] == "PRICE_FILTER"), None)
        
        if not price_filter:
            return str(price)
            
        tick_size = float(price_filter["tickSize"])
        
        # Round to the nearest tick size
        rounded_price = round(price / tick_size) * tick_size
        
        # Convert to string with appropriate precision
        precision = len(price_filter["tickSize"].rstrip("0").split(".")[1]) if "." in price_filter["tickSize"] else 0
        return f"{rounded_price:.{precision}f}"
        
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol's lot size"""
        if symbol not in self.symbol_info:
            return str(quantity)
            
        # Find the lot size filter
        filters = self.symbol_info[symbol]["filters"]
        lot_filter = next((f for f in filters if f["filterType"] == "LOT_SIZE"), None)
        
        if not lot_filter:
            return str(quantity)
            
        step_size = float(lot_filter["stepSize"])
        
        # Round to the nearest step size
        rounded_qty = round(quantity / step_size) * step_size
        
        # Convert to string with appropriate precision
        precision = len(lot_filter["stepSize"].rstrip("0").split(".")[1]) if "." in lot_filter["stepSize"] else 0
        return f"{rounded_qty:.{precision}f}"
