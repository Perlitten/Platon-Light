# Configuration Guide

This guide explains how to configure Platon Light for your specific trading needs.

## Configuration Files

Platon Light uses several configuration files:

1. **`.env`**: Contains sensitive information like API keys and credentials
2. **`config.yaml`**: Main configuration file for trading parameters and settings

## Environment Variables (`.env`)

The `.env` file should be created based on the provided `.env.example` file. This file contains sensitive information and should never be committed to version control.

Example `.env` file:

```
# Binance API credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Telegram Bot settings
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
TELEGRAM_2FA_SECRET=your_2fa_secret_key

# Database settings
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=your_organization
INFLUXDB_BUCKET=platon_light

# Logging settings
LOG_LEVEL=INFO
LOG_FILE=logs/platon_light.log

# Performance settings
MAX_WORKERS=4
```

## Trading Configuration

### Dashboard Settings

The trading dashboard allows you to configure the following settings through the UI:

- **Trading Mode**: Choose between "Dry Run" (simulated trading) and "Real" trading
- **Initial Balance**: Set the starting balance for your trading session
- **Risk Level**: Select the risk level (Low, Medium, High) for your trading strategy
- **Telegram Notifications**: Enable or disable Telegram notifications
- **Trading Pair**: Select the cryptocurrency pair to trade

### Strategy Parameters

Strategy parameters can be configured through the dashboard or by modifying the configuration files:

#### Moving Average Crossover Strategy

```yaml
strategy:
  name: MovingAverageCrossover
  params:
    fast_ma_type: EMA  # SMA or EMA
    slow_ma_type: SMA  # SMA or EMA
    fast_period: 9
    slow_period: 21
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    use_filters: true
```

## Risk Management Configuration

Risk management settings control how the bot manages risk:

```yaml
risk_management:
  max_drawdown_pct: 10.0  # Maximum allowed drawdown percentage
  max_open_positions: 3   # Maximum number of open positions
  position_size_pct: 5.0  # Percentage of portfolio per position
  stop_loss_pct: 2.0      # Stop loss percentage
  take_profit_pct: 4.0    # Take profit percentage
  risk_reward_ratio: 2.0  # Minimum risk-reward ratio
```

## Exchange Configuration

Exchange-specific settings:

```yaml
exchange:
  name: binance
  testnet: true  # Use testnet for testing
  futures: false  # Use futures market
  leverage: 1     # Leverage for futures trading
  margin_type: isolated  # isolated or cross
```

## Telegram Configuration

Configure Telegram notifications:

```yaml
telegram:
  enabled: true
  send_trade_notifications: true
  send_performance_updates: true
  update_interval_minutes: 60
  send_error_alerts: true
```

## Advanced Configuration

For advanced users, additional configuration options are available:

```yaml
advanced:
  use_multi_timeframe_analysis: false
  timeframes: [1m, 5m, 15m, 1h]
  use_order_book_data: false
  order_book_depth: 10
  use_sentiment_analysis: false
  sentiment_sources: [twitter, reddit]
```

## Saving and Loading Configurations

You can save and load different configurations for different trading strategies or market conditions:

```bash
# Save current configuration
python run_platon_light.py --save-config my_strategy

# Load saved configuration
python run_platon_light.py --load-config my_strategy
```

## Troubleshooting

If you encounter issues with your configuration:

1. Check for syntax errors in your YAML files
2. Verify that all required fields are present
3. Ensure API keys have the correct permissions
4. Check that the values are within acceptable ranges

For more assistance, refer to the [troubleshooting guide](troubleshooting.md) or open an issue on GitHub.
