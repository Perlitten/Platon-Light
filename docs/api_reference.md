# API Reference

This document provides a reference for the key classes and functions in the Platon Light API.

## Backtesting Module

### BaseStrategy

```python
class BaseStrategy:
    """Base class for all trading strategies."""
    
    def prepare_data(self, data):
        """
        Prepare data by adding indicators.
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Data with added indicators
        """
        pass
        
    def generate_signals(self, data):
        """
        Generate buy and sell signals.
        
        Args:
            data (pd.DataFrame): Prepared data with indicators
            
        Returns:
            pd.DataFrame: Data with added 'signal' column (1 for buy, -1 for sell, 0 for hold)
        """
        pass
```

### BacktestEngine

```python
class BacktestEngine:
    """Engine for running backtests."""
    
    def __init__(self, strategy, data_loader, initial_capital=10000.0):
        """
        Initialize the backtest engine.
        
        Args:
            strategy (BaseStrategy): Trading strategy to test
            data_loader (DataLoader): Data loader for historical data
            initial_capital (float): Initial capital for the backtest
        """
        pass
        
    def run(self, symbol, timeframe, start_date, end_date):
        """
        Run the backtest.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            start_date (datetime): Start date for the backtest
            end_date (datetime): End date for the backtest
            
        Returns:
            dict: Backtest results including trades and performance metrics
        """
        pass
        
    def optimize(self, symbol, timeframe, start_date, end_date, param_grid):
        """
        Optimize strategy parameters.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            start_date (datetime): Start date for the backtest
            end_date (datetime): End date for the backtest
            param_grid (dict): Grid of parameters to test
            
        Returns:
            dict: Optimization results including best parameters and performance
        """
        pass
```

### DataLoader

```python
class DataLoader:
    """Loader for historical market data."""
    
    def __init__(self, data_dir=None, exchange=None):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory for storing data
            exchange (Exchange): Exchange instance for fetching data
        """
        pass
        
    def load_data(self, symbol, timeframe, start_date, end_date):
        """
        Load historical data.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pd.DataFrame: Historical market data
        """
        pass
        
    def prepare_data(self, data):
        """
        Prepare data by adding common indicators.
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Data with added indicators
        """
        pass
```

## Trading Module

### TradingEngine

```python
class TradingEngine:
    """Engine for executing trades."""
    
    def __init__(self, exchange, strategy, initial_capital=10000.0, mode='dry_run'):
        """
        Initialize the trading engine.
        
        Args:
            exchange (Exchange): Exchange instance for executing trades
            strategy (BaseStrategy): Trading strategy
            initial_capital (float): Initial capital
            mode (str): Trading mode ('dry_run' or 'real')
        """
        pass
        
    def start(self):
        """Start the trading engine."""
        pass
        
    def stop(self):
        """Stop the trading engine."""
        pass
        
    def get_status(self):
        """
        Get the current status of the trading engine.
        
        Returns:
            dict: Trading engine status
        """
        pass
        
    def get_positions(self):
        """
        Get current open positions.
        
        Returns:
            list: Open positions
        """
        pass
        
    def get_balance(self):
        """
        Get current account balance.
        
        Returns:
            float: Account balance
        """
        pass
```

## Exchange Module

### Exchange

```python
class Exchange:
    """Base class for exchange integrations."""
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """
        Initialize the exchange.
        
        Args:
            api_key (str): API key
            api_secret (str): API secret
            testnet (bool): Whether to use testnet
        """
        pass
        
    def get_balance(self):
        """
        Get account balance.
        
        Returns:
            dict: Account balance
        """
        pass
        
    def get_ticker(self, symbol):
        """
        Get current ticker.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Current ticker
        """
        pass
        
    def get_historical_data(self, symbol, timeframe, limit=100):
        """
        Get historical market data.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            limit (int): Number of candles
            
        Returns:
            pd.DataFrame: Historical market data
        """
        pass
        
    def create_order(self, symbol, order_type, side, amount, price=None):
        """
        Create a new order.
        
        Args:
            symbol (str): Trading symbol
            order_type (str): Order type ('limit', 'market')
            side (str): Order side ('buy', 'sell')
            amount (float): Order amount
            price (float): Order price (for limit orders)
            
        Returns:
            dict: Order details
        """
        pass
```

## Telegram Integration

### TelegramBot

```python
class TelegramBot:
    """Telegram bot for notifications and control."""
    
    def __init__(self, token, chat_id):
        """
        Initialize the Telegram bot.
        
        Args:
            token (str): Telegram bot token
            chat_id (str): Telegram chat ID
        """
        pass
        
    def send_message(self, message):
        """
        Send a message.
        
        Args:
            message (str): Message to send
        """
        pass
        
    def send_trade_notification(self, trade):
        """
        Send a trade notification.
        
        Args:
            trade (dict): Trade details
        """
        pass
        
    def send_error_notification(self, error):
        """
        Send an error notification.
        
        Args:
            error (str): Error message
        """
        pass
        
    def send_performance_update(self, performance):
        """
        Send a performance update.
        
        Args:
            performance (dict): Performance metrics
        """
        pass
```

## Dashboard Module

### Dashboard

The dashboard is built using Dash and provides a web interface for monitoring and controlling the trading bot. The main components are:

- `app`: Dash application instance
- `layout`: Dashboard layout
- Callback functions for handling user interactions and updating the UI

## Utility Functions

### Risk Management

```python
def calculate_position_size(account_balance, risk_per_trade, stop_loss_pct):
    """
    Calculate position size based on risk management rules.
    
    Args:
        account_balance (float): Account balance
        risk_per_trade (float): Risk per trade as percentage
        stop_loss_pct (float): Stop loss percentage
        
    Returns:
        float: Position size
    """
    pass
    
def calculate_risk_metrics(trades, initial_capital):
    """
    Calculate risk metrics from trade history.
    
    Args:
        trades (list): List of trades
        initial_capital (float): Initial capital
        
    Returns:
        dict: Risk metrics
    """
    pass
```

### Technical Indicators

```python
def calculate_sma(data, period):
    """
    Calculate Simple Moving Average.
    
    Args:
        data (pd.Series): Price data
        period (int): Period
        
    Returns:
        pd.Series: SMA values
    """
    pass
    
def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average.
    
    Args:
        data (pd.Series): Price data
        period (int): Period
        
    Returns:
        pd.Series: EMA values
    """
    pass
    
def calculate_rsi(data, period):
    """
    Calculate Relative Strength Index.
    
    Args:
        data (pd.Series): Price data
        period (int): Period
        
    Returns:
        pd.Series: RSI values
    """
    pass
    
def calculate_bollinger_bands(data, period, std_dev):
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.Series): Price data
        period (int): Period
        std_dev (float): Standard deviation multiplier
        
    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    pass
```

## Configuration Management

```python
def load_config(config_file):
    """
    Load configuration from file.
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration
    """
    pass
    
def save_config(config, config_file):
    """
    Save configuration to file.
    
    Args:
        config (dict): Configuration
        config_file (str): Path to configuration file
    """
    pass
```

## Secure Credential Handling

```python
def encrypt_credentials(credentials, key):
    """
    Encrypt credentials.
    
    Args:
        credentials (dict): Credentials
        key (str): Encryption key
        
    Returns:
        bytes: Encrypted credentials
    """
    pass
    
def decrypt_credentials(encrypted_credentials, key):
    """
    Decrypt credentials.
    
    Args:
        encrypted_credentials (bytes): Encrypted credentials
        key (str): Encryption key
        
    Returns:
        dict: Decrypted credentials
    """
    pass
```
