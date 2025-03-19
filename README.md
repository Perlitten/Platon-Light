# Platon Light - Advanced Cryptocurrency Trading Bot

A high-performance trading bot for cryptocurrency exchanges with advanced features for both spot and futures markets, comprehensive backtesting, and an interactive dashboard.

## Features

- **Advanced Trading Strategies**: Utilizes momentum indicators (RSI, MACD, Stochastic), volume analysis, and order book imbalance detection
- **Multi-Market Support**: Works with both spot and futures markets with configurable leverage
- **Real-time Analysis**: Analyzes price movements with configurable timeframes (5s to 5m)
- **Risk Management**: Implements Kelly Criterion, dynamic stop-loss, and various risk limiters
- **Dry Run Mode**: Full simulation environment with historical data replay
- **Telegram Integration**: Real-time notifications and remote command capabilities
- **Interactive Dashboard**: Real-time monitoring of trades, performance metrics, and risk indicators
- **Comprehensive Backtesting**: Complete backtesting framework with performance analysis and strategy optimization

## Dashboard Features

- **Real-time Trading Controls**: Start/stop trading, switch between dry run and real trading modes
- **Performance Metrics**: Track key metrics like win rate, profit factor, Sharpe ratio, and drawdown
- **Risk Indicators**: Visual indicators for risk levels based on performance metrics
- **Price Charts**: Interactive price charts with technical indicators and buy/sell signals
- **Trading History**: Detailed view of executed trades with entry/exit points and performance
- **Telegram Notifications**: Configurable alerts for trade events and status updates

## Installation

1. Clone the repository
   ```
   git clone https://github.com/Perlitten/Platon-Light.git
   cd Platon-Light
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your API keys in `.env` file (use `.env.example` as template)

## Docker Installation

You can also run Platon Light using Docker:

1. Clone the repository
   ```
   git clone https://github.com/Perlitten/Platon-Light.git
   cd Platon-Light
   ```

2. Create a `.env` file based on `.env.example`

3. Build and run using Docker Compose
   ```
   docker-compose up -d
   ```

4. Access the dashboard at http://localhost:8050

## Usage

### Running the Interactive Dashboard

```bash
python run_platon_light.py
```

Select "Launch Trading Dashboard" from the menu to start the interactive dashboard.

### Backtesting

```bash
python run_platon_light.py
```

Select "Run Backtest Workflow" or "Optimize Strategy Parameters" from the menu.

## Backtesting Strategies

The backtesting module includes the following strategies:

### Moving Average Crossover

A strategy that generates buy and sell signals based on the crossover of fast and slow moving averages. Optional filters include RSI and Bollinger Bands.

Parameters:
- `fast_ma_type`: Type of fast moving average ('SMA' or 'EMA')
- `slow_ma_type`: Type of slow moving average ('SMA' or 'EMA')
- `fast_period`: Period for fast moving average
- `slow_period`: Period for slow moving average
- `rsi_period`: Period for RSI calculation
- `rsi_overbought`: RSI overbought threshold
- `rsi_oversold`: RSI oversold threshold
- `use_filters`: Whether to use additional filters (RSI, Bollinger Bands)

## Security

Platon Light implements secure credential handling to protect your API keys. Credentials are encrypted and stored securely.

## License

MIT

## Disclaimer

This software is for educational purposes only. Use at your own risk. The developers are not responsible for any financial losses incurred while using this software.
