import pytest
import asyncio
from unittest.mock import AsyncMock

from platon_light.core.bot import TradingBot
from platon_light.core.config_models import BotConfig
from tests.mocks.mock_exchange_client import MockExchangeClient

# A minimal, valid config for testing
MOCK_CONFIG_DICT = {
    "general": {
        "exchange": "mock",
        "quote_currencies": ["BTC"],
        "base_currency": "USDT",
        "market_type": "spot",
    },
    "trading": {
        "timeframes": ["1m"],
    },
    "risk": {"max_risk_per_trade_percentage": 2.0},
    "telegram": {"enabled": False},
    "visualization": {"console": {"enabled": False}},
}


@pytest.mark.asyncio
async def test_bot_creates_order_on_signal():
    """
    Tests that the TradingBot, when given a buy signal, correctly calculates
    position size, and calls the position manager, which in turn calls
    the exchange client's create_order method.
    """
    # 1. Setup
    config = BotConfig(**MOCK_CONFIG_DICT)
    bot = TradingBot(config)
    mock_exchange: MockExchangeClient = bot.exchange

    # 2. Mock dependencies
    mock_signal = {
        "direction": "long",
        "side": "buy",
        "entry_price": 50000,
        "target_price": 51000,
        "stop_price": 49500,
        "stop_loss_pct": 1.0,
        "profit_target_pct": 2.0,
    }
    # Mock the bot's own strategy instance
    bot.strategy.generate_signal = AsyncMock(return_value=mock_signal)
    bot.strategy.calculate_position_size = lambda symbol, signal, balance: 100.0 # Return a fixed size of 100 USDT

    # Mock data and other helpers
    bot.market_data.get_latest_data = lambda symbol: {"close": 50000.0}
    bot.position_manager.get_positions = lambda symbol=None: []

    # 3. Run bot
    main_task = asyncio.create_task(bot.start())
    await asyncio.sleep(0.5)
    main_task.cancel()
    try:
        await main_task
    except asyncio.CancelledError:
        pass

    # 4. Assertions
    create_order_call = next((c for c in mock_exchange.calls if c['method'] == 'create_order'), None)

    assert create_order_call is not None, "create_order was not called on the mock exchange"
    assert create_order_call['params']['symbol'] == 'BTCUSDT'
    # The quantity is position_size / current_price -> 100.0 / 50000.0
    assert create_order_call['params']['quantity'] == (100.0 / 50000.0)

    assert any(call["method"] == "connect" for call in mock_exchange.calls), "connect was not called"
    assert any(call["method"] == "disconnect" for call in mock_exchange.calls), "disconnect was not called"
