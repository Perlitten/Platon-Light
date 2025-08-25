"""
Factory for creating exchange client instances.
"""

import os
from typing import Dict
from tests.mocks.mock_exchange_client import MockExchangeClient
from platon_light.core.base_exchange import BaseExchangeClient
from platon_light.core.binance_client import BinanceClient
from platon_light.core.bybit_client import BybitClient


def get_exchange_client(config: Dict) -> BaseExchangeClient:
    """
    Creates and returns an exchange client instance based on the configuration.

    Args:
        config: The bot's configuration dictionary.

    Returns:
        An instance of a class that inherits from BaseExchangeClient.

    Raises:
        ValueError: If the specified exchange is not supported or keys are missing.
    """
    exchange_name = config.get("general", {}).get("exchange", "").lower()

    if not exchange_name:
        raise ValueError("Exchange not specified in configuration")

    # For mock exchange, we don't need real keys
    if exchange_name == "mock":
        return MockExchangeClient(config, "mock_key", "mock_secret")

    # Get API keys from environment variables
    api_key_env_var = f"{exchange_name.upper()}_API_KEY"
    api_secret_env_var = f"{exchange_name.upper()}_API_SECRET"

    api_key = os.environ.get(api_key_env_var)
    api_secret = os.environ.get(api_secret_env_var)

    # In dry-run or testnet mode for Bybit, we might not need keys for all operations,
    # but ccxt requires them for authentication on private endpoints.
    # We will let the client classes handle dummy keys if needed.
    if not api_key or not api_secret:
        # Allow missing keys for binance dry-run, but bybit testnet needs them.
        if (
            exchange_name == "binance"
            and config.get("general", {}).get("mode") == "dry-run"
        ):
            pass  # It's ok for binance dry-run
        else:
            raise ValueError(
                f"API keys ({api_key_env_var}, {api_secret_env_var}) not found in environment variables."
            )

    if exchange_name == "binance":
        return BinanceClient(config, api_key, api_secret)
    elif exchange_name == "bybit":
        return BybitClient(config, api_key, api_secret)
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")
