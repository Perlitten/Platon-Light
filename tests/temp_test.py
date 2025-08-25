import pytest
import asyncio
from platon_light.core.risk_manager import RiskManager

@pytest.mark.asyncio
async def test_risk_manager_validation():
    """Directly test the RiskManager's validate_trade method."""

    # 1. Setup a minimal config and the RiskManager
    config = {
        "risk": {
            "max_risk_per_trade_percentage": 2.0,
            "max_open_positions": 5,
            "abnormal_market_detection": False, # Disable complex checks
            "correlation_limit": 0
        },
        "general": {
            "market_type": "spot"
        }
    }
    risk_manager = RiskManager(config)

    # 2. Create the mock signal that was causing issues
    mock_signal = {
        "type": "buy",
        "side": "buy",
        "price": 50000,
        "quantity": 0.01,
        "stop_loss_pct": 1.0,
        "profit_target_pct": 2.0,
        "direction": "long"
    }

    # 3. Call the method directly and assert it passes
    is_valid = await risk_manager.validate_trade(
        symbol='BTCUSDT',
        signal=mock_signal,
        current_positions=[]
    )

    assert is_valid is True, "Trade validation should have passed"
