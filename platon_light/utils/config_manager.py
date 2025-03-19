"""
Configuration manager for loading and validating bot configuration
"""
import os
import yaml
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv


class ConfigManager:
    """
    Configuration manager for loading and validating bot configuration
    
    Features:
    - Load configuration from YAML files
    - Load environment variables
    - Validate configuration
    - Provide defaults for missing values
    - Update configuration at runtime
    """
    
    def __init__(self, config_path: str, env_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file
            env_path: Path to the .env file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        
        # Load configuration
        self._load_config()
        
        # Load environment variables
        if env_path:
            self._load_env_vars()
            
        # Validate configuration
        self._validate_config()
        
        self.logger.info("Configuration loaded and validated")
        
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.logger.debug(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            # Initialize with default configuration
            self.config = self._get_default_config()
            
    def _load_env_vars(self):
        """Load environment variables from .env file"""
        try:
            # Load .env file
            if os.path.exists(self.env_path):
                load_dotenv(self.env_path)
                self.logger.debug(f"Loaded environment variables from {self.env_path}")
                
            # Update configuration with environment variables
            self._update_config_from_env()
        except Exception as e:
            self.logger.error(f"Failed to load environment variables from {self.env_path}: {e}")
            
    def _update_config_from_env(self):
        """Update configuration with environment variables"""
        # API keys
        if "BINANCE_API_KEY" in os.environ:
            self.config.setdefault("exchange", {})["api_key"] = os.environ["BINANCE_API_KEY"]
            
        if "BINANCE_API_SECRET" in os.environ:
            self.config.setdefault("exchange", {})["api_secret"] = os.environ["BINANCE_API_SECRET"]
            
        # Telegram bot token
        if "TELEGRAM_BOT_TOKEN" in os.environ:
            self.config.setdefault("telegram", {})["token"] = os.environ["TELEGRAM_BOT_TOKEN"]
            
        # Authorized Telegram users
        if "TELEGRAM_AUTH_USERS" in os.environ:
            auth_users = os.environ["TELEGRAM_AUTH_USERS"].split(",")
            self.config.setdefault("telegram", {})["authorized_users"] = [int(user.strip()) for user in auth_users if user.strip()]
            
        # Database settings
        if "DB_HOST" in os.environ:
            self.config.setdefault("database", {})["host"] = os.environ["DB_HOST"]
            
        if "DB_PORT" in os.environ:
            self.config.setdefault("database", {})["port"] = int(os.environ["DB_PORT"])
            
        if "DB_NAME" in os.environ:
            self.config.setdefault("database", {})["name"] = os.environ["DB_NAME"]
            
        if "DB_USER" in os.environ:
            self.config.setdefault("database", {})["user"] = os.environ["DB_USER"]
            
        if "DB_PASSWORD" in os.environ:
            self.config.setdefault("database", {})["password"] = os.environ["DB_PASSWORD"]
            
    def _validate_config(self):
        """Validate configuration and set defaults for missing values"""
        # Ensure required sections exist
        required_sections = ["exchange", "trading", "risk_management", "strategy"]
        
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"Missing configuration section: {section}")
                self.config[section] = {}
                
        # Validate exchange configuration
        exchange_config = self.config["exchange"]
        
        if "api_key" not in exchange_config or not exchange_config["api_key"]:
            self.logger.warning("Missing Binance API key")
            
        if "api_secret" not in exchange_config or not exchange_config["api_secret"]:
            self.logger.warning("Missing Binance API secret")
            
        if "testnet" not in exchange_config:
            exchange_config["testnet"] = True
            self.logger.info("Using testnet by default")
            
        # Validate trading configuration
        trading_config = self.config["trading"]
        
        if "symbols" not in trading_config or not trading_config["symbols"]:
            trading_config["symbols"] = ["BTCUSDT"]
            self.logger.warning(f"No trading symbols specified, using default: {trading_config['symbols']}")
            
        if "timeframes" not in trading_config or not trading_config["timeframes"]:
            trading_config["timeframes"] = ["1m", "5m"]
            self.logger.warning(f"No timeframes specified, using default: {trading_config['timeframes']}")
            
        if "mode" not in trading_config:
            trading_config["mode"] = "spot"
            self.logger.info(f"Trading mode not specified, using default: {trading_config['mode']}")
            
        if "dry_run" not in trading_config:
            trading_config["dry_run"] = True
            self.logger.info("Dry run mode enabled by default")
            
        # Validate risk management configuration
        risk_config = self.config["risk_management"]
        
        if "max_position_size" not in risk_config:
            risk_config["max_position_size"] = 0.01
            self.logger.info(f"Max position size not specified, using default: {risk_config['max_position_size']}")
            
        if "max_open_positions" not in risk_config:
            risk_config["max_open_positions"] = 3
            self.logger.info(f"Max open positions not specified, using default: {risk_config['max_open_positions']}")
            
        if "daily_loss_limit" not in risk_config:
            risk_config["daily_loss_limit"] = 0.05
            self.logger.info(f"Daily loss limit not specified, using default: {risk_config['daily_loss_limit']}")
            
        if "max_drawdown" not in risk_config:
            risk_config["max_drawdown"] = 0.1
            self.logger.info(f"Max drawdown not specified, using default: {risk_config['max_drawdown']}")
            
        # Validate strategy configuration
        strategy_config = self.config["strategy"]
        
        if "name" not in strategy_config:
            strategy_config["name"] = "scalping"
            self.logger.info(f"Strategy name not specified, using default: {strategy_config['name']}")
            
        # Set defaults for strategy parameters
        if strategy_config["name"] == "scalping":
            if "rsi_period" not in strategy_config:
                strategy_config["rsi_period"] = 14
                
            if "rsi_overbought" not in strategy_config:
                strategy_config["rsi_overbought"] = 70
                
            if "rsi_oversold" not in strategy_config:
                strategy_config["rsi_oversold"] = 30
                
            if "macd_fast" not in strategy_config:
                strategy_config["macd_fast"] = 12
                
            if "macd_slow" not in strategy_config:
                strategy_config["macd_slow"] = 26
                
            if "macd_signal" not in strategy_config:
                strategy_config["macd_signal"] = 9
                
            if "profit_target" not in strategy_config:
                strategy_config["profit_target"] = 0.005
                
            if "stop_loss" not in strategy_config:
                strategy_config["stop_loss"] = 0.003
                
        # Validate logging configuration
        if "logging" not in self.config:
            self.config["logging"] = {}
            
        logging_config = self.config["logging"]
        
        if "level" not in logging_config:
            logging_config["level"] = "INFO"
            
        if "log_dir" not in logging_config:
            logging_config["log_dir"] = "logs"
            
        # Validate visualization configuration
        if "visualization" not in self.config:
            self.config["visualization"] = {}
            
        viz_config = self.config["visualization"]
        
        if "enabled" not in viz_config:
            viz_config["enabled"] = True
            
        if "update_interval" not in viz_config:
            viz_config["update_interval"] = 2
            
        # Validate Telegram configuration
        if "telegram" not in self.config:
            self.config["telegram"] = {}
            
        telegram_config = self.config["telegram"]
        
        if "enabled" not in telegram_config:
            telegram_config["enabled"] = False
            
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            "exchange": {
                "name": "binance",
                "testnet": True
            },
            "trading": {
                "symbols": ["BTCUSDT"],
                "timeframes": ["1m", "5m"],
                "mode": "spot",
                "dry_run": True
            },
            "risk_management": {
                "max_position_size": 0.01,
                "max_open_positions": 3,
                "daily_loss_limit": 0.05,
                "max_drawdown": 0.1
            },
            "strategy": {
                "name": "scalping",
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "profit_target": 0.005,
                "stop_loss": 0.003
            },
            "logging": {
                "level": "INFO",
                "log_dir": "logs"
            },
            "visualization": {
                "enabled": True,
                "update_interval": 2
            },
            "telegram": {
                "enabled": False
            }
        }
        
    def get_config(self) -> Dict:
        """
        Get the full configuration
        
        Returns:
            Configuration dictionary
        """
        return self.config
        
    def get_section(self, section: str) -> Dict:
        """
        Get a configuration section
        
        Args:
            section: Section name
            
        Returns:
            Section dictionary
        """
        return self.config.get(section, {})
        
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            section: Section name
            key: Key name
            default: Default value
            
        Returns:
            Configuration value
        """
        return self.config.get(section, {}).get(key, default)
        
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """
        Update a configuration value
        
        Args:
            section: Section name
            key: Key name
            value: New value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if section not in self.config:
                self.config[section] = {}
                
            self.config[section][key] = value
            self.logger.info(f"Updated configuration: {section}.{key} = {value}")
            
            # Save configuration to file
            self._save_config()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
            
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
                self.logger.debug(f"Saved configuration to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {self.config_path}: {e}")
            
    def export_config(self, format: str = "yaml") -> str:
        """
        Export configuration to a string
        
        Args:
            format: Export format (yaml, json)
            
        Returns:
            Configuration string
        """
        if format.lower() == "yaml":
            return yaml.dump(self.config, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(self.config, indent=2)
        else:
            self.logger.warning(f"Invalid export format: {format}")
            return ""
