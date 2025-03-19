# Secure API Credentials Management

This guide explains how to securely manage API credentials in the Platon Light backtesting framework.

## Table of Contents

1. [Introduction](#introduction)
2. [Security Best Practices](#security-best-practices)
3. [Using the ConfigManager](#using-the-configmanager)
4. [Environment Variables](#environment-variables)
5. [Encrypted Storage](#encrypted-storage)
6. [Integration with Backtesting](#integration-with-backtesting)
7. [Example Implementation](#example-implementation)

## Introduction

Trading applications often require API credentials for accessing exchange data, sending notifications, or executing trades. Properly securing these credentials is critical to protect your trading accounts and sensitive information.

The Platon Light framework provides a secure `ConfigManager` class that helps you manage API credentials safely through:

- Environment variables
- Encrypted storage
- Secure memory handling

## Security Best Practices

When working with API credentials, always follow these security best practices:

1. **Never hardcode credentials** in your source code
2. **Never commit credentials** to version control systems
3. **Use environment variables** or encrypted storage for sensitive information
4. **Restrict file permissions** for credential files
5. **Use read-only API keys** when possible
6. **Rotate API keys** periodically
7. **Use IP restrictions** when supported by the API provider
8. **Implement the principle of least privilege** - only request permissions your application needs

## Using the ConfigManager

The `ConfigManager` class provides a secure way to manage API credentials:

```python
from platon_light.core.config_manager import ConfigManager

# Create config manager
config_manager = ConfigManager()

# Initialize encryption
config_manager.initialize_encryption('config/encryption.key')

# Set API credentials
config_manager.set_api_credentials('binance', {
    'api_key': 'your_api_key',
    'api_secret': 'your_api_secret'
})

# Save configuration
config_manager.save_config('config/config.json')

# Retrieve API credentials when needed
credentials = config_manager.get_api_credentials('binance')
if credentials:
    api_key = credentials['api_key']
    api_secret = credentials['api_secret']
```

### Key Features

- **Encryption**: Sensitive data is encrypted using Fernet symmetric encryption
- **Multiple Sources**: Credentials can be loaded from environment variables or configuration files
- **Secure Storage**: Credentials are never saved in plain text
- **Validation**: Built-in validation ensures all required credentials are available

## Environment Variables

Using environment variables is one of the safest ways to manage credentials:

### Setting Environment Variables

#### Linux/macOS:

```bash
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_api_secret
```

#### Windows (Command Prompt):

```cmd
set BINANCE_API_KEY=your_api_key
set BINANCE_API_SECRET=your_api_secret
```

#### Windows (PowerShell):

```powershell
$env:BINANCE_API_KEY="your_api_key"
$env:BINANCE_API_SECRET="your_api_secret"
```

### Using .env Files

For development, you can use a `.env` file:

1. Create a `.env` file in your project root:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

2. Add `.env` to your `.gitignore` file to prevent accidental commits
3. The `ConfigManager` will automatically load variables from this file

## Encrypted Storage

For more secure storage, the `ConfigManager` supports encrypted credential files:

1. Initialize encryption with a secure key:

```python
config_manager.initialize_encryption('config/encryption.key')
```

2. Store your credentials securely:

```python
config_manager.set_api_credentials('binance', {
    'api_key': 'your_api_key',
    'api_secret': 'your_api_secret'
}, encrypt=True)

config_manager.save_config('config/config.json')
```

3. Keep the encryption key secure and separate from the configuration file
4. Set restrictive file permissions:

```bash
chmod 600 config/encryption.key
chmod 600 config/config.json
```

## Integration with Backtesting

When running backtests that require API access (e.g., for loading historical data), use the `ConfigManager` to securely access credentials:

```python
from platon_light.core.config_manager import ConfigManager
from platon_light.backtesting.data_loader import DataLoader

# Initialize config manager
config_manager = ConfigManager('config/config.json')
config_manager.initialize_encryption('config/encryption.key')

# Get API credentials
binance_credentials = config_manager.get_api_credentials('binance')

# Initialize data loader with credentials
data_loader = DataLoader(
    exchange='binance',
    api_key=binance_credentials['api_key'],
    api_secret=binance_credentials['api_secret']
)

# Load historical data
data = data_loader.load_data(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

## Example Implementation

Here's a complete example of securely managing API credentials for a backtesting project:

```python
import os
import logging
from pathlib import Path
from platon_light.core.config_manager import ConfigManager
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.strategies.moving_average_strategy import MovingAverageStrategy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_config():
    """Set up configuration and API credentials"""
    # Create config directories if they don't exist
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Initialize encryption
    key_path = config_dir / 'encryption.key'
    config_manager.initialize_encryption(str(key_path))
    
    # Check if credentials are already in environment variables
    binance_key = os.environ.get('BINANCE_API_KEY')
    binance_secret = os.environ.get('BINANCE_API_SECRET')
    
    if not (binance_key and binance_secret):
        # Prompt for credentials if not in environment
        logger.info("Binance API credentials not found in environment variables")
        binance_key = input("Enter Binance API key: ")
        binance_secret = input("Enter Binance API secret: ")
    
    # Set and save credentials
    config_manager.set_api_credentials('binance', {
        'api_key': binance_key,
        'api_secret': binance_secret
    }, encrypt=True)
    
    # Set other configuration values
    config_manager.set_config_value('backtesting.initial_capital', 10000)
    config_manager.set_config_value('backtesting.fee_rate', 0.001)
    
    # Save configuration
    config_path = config_dir / 'config.json'
    config_manager.save_config(str(config_path))
    
    logger.info(f"Configuration saved to {config_path}")
    return config_manager

def run_backtest(config_manager):
    """Run backtest using securely stored credentials"""
    # Get API credentials
    binance_credentials = config_manager.get_api_credentials('binance')
    if not binance_credentials:
        logger.error("Failed to retrieve Binance API credentials")
        return
    
    # Initialize data loader with credentials
    data_loader = DataLoader(
        exchange='binance',
        api_key=binance_credentials['api_key'],
        api_secret=binance_credentials['api_secret']
    )
    
    # Load historical data
    data = data_loader.load_data(
        symbol='BTCUSDT',
        timeframe='1h',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Create strategy
    strategy = MovingAverageStrategy(
        short_window=20,
        long_window=50
    )
    
    # Get backtest configuration
    initial_capital = config_manager.get_config_value('backtesting.initial_capital', 10000)
    fee_rate = config_manager.get_config_value('backtesting.fee_rate', 0.001)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        strategy=strategy,
        initial_capital=initial_capital,
        fee_rate=fee_rate
    )
    
    # Run backtest
    results = backtest_engine.run_with_data(data)
    
    # Analyze results
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(results)
    
    # Print key metrics
    logger.info(f"Total Return: {metrics['return_percent']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
    
    return results, metrics

if __name__ == "__main__":
    # Set up configuration
    config_manager = setup_config()
    
    # Run backtest
    results, metrics = run_backtest(config_manager)
```

## Best Practices for Production

When deploying to production, consider these additional security measures:

1. **Use a secrets management service** like HashiCorp Vault or AWS Secrets Manager
2. **Implement key rotation** to periodically update API credentials
3. **Use separate API keys** for different environments (development, testing, production)
4. **Monitor API usage** to detect unauthorized access
5. **Implement IP whitelisting** when supported by the API provider
6. **Use hardware security modules (HSMs)** for storing encryption keys in high-security environments
7. **Implement access controls** to limit who can access credentials

By following these guidelines, you can ensure that your API credentials remain secure while using the Platon Light backtesting framework.
