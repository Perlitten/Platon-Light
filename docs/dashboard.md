# Trading Dashboard Guide

This guide explains how to use the Platon Light Trading Dashboard, which provides real-time monitoring and control of your trading activities.

## Starting the Dashboard

To start the trading dashboard, run the following command:

```bash
python run_platon_light.py
```

Then select "Launch Trading Dashboard" from the main menu.

## Dashboard Features

The dashboard is divided into several sections, each providing different functionality:

### Trading Controls

This section allows you to control the trading bot:

- **Trading Mode**: Switch between "Dry Run" (simulated trading) and "Real" trading modes
- **Initial Balance**: Set the starting balance for your trading session
- **Risk Level**: Select the risk level (Low, Medium, High) for your trading strategy
- **Telegram Notifications**: Toggle Telegram notifications on/off
- **Trading Pair**: Select the cryptocurrency pair to trade
- **Start/Stop Trading**: Buttons to start or stop the trading bot

### Account Overview

This section displays key information about your trading account:

- **Current Balance**: Your current account balance
- **Trading Status**: Whether trading is active or inactive
- **Open Positions**: Number of currently open positions

### Risk Metrics

This table displays important risk and performance metrics:

- **Max Drawdown**: Maximum observed loss from a peak to a trough
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Sharpe Ratio**: Measure of risk-adjusted return
- **Daily Return**: Average daily return percentage
- **Monthly Return**: Average monthly return percentage

Each metric is color-coded based on its value (green for good, yellow for moderate, red for concerning).

### Price Chart

This interactive chart displays:

- Price candlesticks
- Moving averages
- Bollinger Bands
- Buy and sell signals

You can hover over elements to see more details and use the toolbar to zoom, pan, or download the chart.

### Trading History

This table shows a history of executed trades with the following information:

- Entry time
- Exit time
- Entry price
- Exit price
- Position size
- Profit/Loss
- Return percentage

### Performance Metrics

This chart displays:

- Cumulative profit and loss
- Account balance over time
- Distribution of winning and losing trades

## Using the Dashboard Effectively

1. **Start in Dry Run Mode**: Always begin with dry run mode to test your strategy without risking real funds
2. **Monitor Risk Metrics**: Pay close attention to the risk metrics, especially max drawdown and Sharpe ratio
3. **Set Appropriate Risk Level**: Adjust the risk level based on your risk tolerance and market conditions
4. **Enable Telegram Notifications**: For important alerts when you're away from the dashboard
5. **Analyze Trading History**: Regularly review your trading history to identify patterns and improve your strategy

## Troubleshooting

If you encounter issues with the dashboard:

- Check your internet connection
- Verify that your API keys are correctly configured
- Ensure you have the latest version of Platon Light
- Check the console for any error messages

For persistent issues, please [open an issue](https://github.com/Perlitten/Platon-Light/issues) on GitHub.
