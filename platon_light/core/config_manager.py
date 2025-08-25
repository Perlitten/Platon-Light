#!/usr/bin/env python
"""
Configuration Manager

This module provides secure handling of configuration settings and API credentials
for the Platon Light backtesting framework. It supports loading credentials from
environment variables, encrypted files, or secure credential stores.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import dotenv
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration settings and API credentials securely.

    This class provides methods to:
    - Load configuration from various sources
    - Securely store and retrieve API credentials
    - Encrypt sensitive information
    - Validate configuration settings
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager.

        Args:
            config_path: Optional path to configuration file
        """
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        self.encryption_key = None

        # Load environment variables from .env file if it exists
        dotenv_path = Path(".env")
        if dotenv_path.exists():
            dotenv.load_dotenv(dotenv_path)
            logger.info(f"Loaded environment variables from {dotenv_path}")

        # Load configuration if path provided
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration settings
        """
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return {}

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to a file.

        Args:
            config_path: Optional path to save configuration file

        Returns:
            True if successful, False otherwise
        """
        path = config_path or self.config_path
        if not path:
            logger.error("No configuration path specified")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Remove sensitive data before saving
            safe_config = self.config.copy()
            if "api_credentials" in safe_config:
                del safe_config["api_credentials"]

            with open(path, "w") as f:
                json.dump(safe_config, f, indent=4)
            logger.info(f"Saved configuration to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")
            return False

    def initialize_encryption(self, key_path: Optional[str] = None) -> bool:
        """
        Initialize encryption for sensitive data.

        Args:
            key_path: Optional path to encryption key file

        Returns:
            True if successful, False otherwise
        """
        try:
            if key_path and os.path.exists(key_path):
                # Load existing key
                with open(key_path, "rb") as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()

                # Save key if path provided
                if key_path:
                    os.makedirs(os.path.dirname(key_path), exist_ok=True)
                    with open(key_path, "wb") as f:
                        f.write(self.encryption_key)

                    # Set restrictive permissions
                    os.chmod(key_path, 0o600)

            logger.info("Encryption initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.encryption_key = None
            return False

    def encrypt_value(self, value: str) -> Optional[bytes]:
        """
        Encrypt a sensitive value.

        Args:
            value: Value to encrypt

        Returns:
            Encrypted value as bytes, or None if encryption fails
        """
        if not self.encryption_key:
            logger.error("Encryption not initialized")
            return None

        try:
            cipher = Fernet(self.encryption_key)
            return cipher.encrypt(value.encode())
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None

    def decrypt_value(self, encrypted_value: bytes) -> Optional[str]:
        """
        Decrypt an encrypted value.

        Args:
            encrypted_value: Encrypted value to decrypt

        Returns:
            Decrypted value as string, or None if decryption fails
        """
        if not self.encryption_key:
            logger.error("Encryption not initialized")
            return None

        try:
            cipher = Fernet(self.encryption_key)
            return cipher.decrypt(encrypted_value).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

    def set_api_credentials(
        self, service: str, credentials: Dict[str, str], encrypt: bool = True
    ) -> bool:
        """
        Set API credentials for a service.

        Args:
            service: Service name (e.g., 'binance', 'telegram')
            credentials: Dictionary of credential key-value pairs
            encrypt: Whether to encrypt the credentials

        Returns:
            True if successful, False otherwise
        """
        if encrypt and not self.encryption_key:
            logger.warning(
                "Encryption not initialized, storing credentials unencrypted"
            )
            encrypt = False

        try:
            if "api_credentials" not in self.config:
                self.config["api_credentials"] = {}

            if encrypt:
                encrypted_creds = {}
                for key, value in credentials.items():
                    encrypted_value = self.encrypt_value(value)
                    if encrypted_value:
                        encrypted_creds[key] = encrypted_value.decode()
                    else:
                        return False

                self.config["api_credentials"][service] = {
                    "encrypted": True,
                    "credentials": encrypted_creds,
                }
            else:
                self.config["api_credentials"][service] = {
                    "encrypted": False,
                    "credentials": credentials,
                }

            logger.info(f"API credentials for {service} set successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set API credentials for {service}: {e}")
            return False

    def get_api_credentials(self, service: str) -> Optional[Dict[str, str]]:
        """
        Get API credentials for a service.

        Args:
            service: Service name (e.g., 'binance', 'telegram')

        Returns:
            Dictionary of credential key-value pairs, or None if not found
        """
        try:
            # Check if credentials exist in config
            if (
                "api_credentials" not in self.config
                or service not in self.config["api_credentials"]
            ):

                # Check environment variables as fallback
                env_creds = self._get_credentials_from_env(service)
                if env_creds:
                    return env_creds

                logger.warning(f"No API credentials found for {service}")
                return None

            service_config = self.config["api_credentials"][service]

            if service_config["encrypted"]:
                if not self.encryption_key:
                    logger.error(
                        "Cannot decrypt credentials: encryption not initialized"
                    )
                    return None

                decrypted_creds = {}
                for key, value in service_config["credentials"].items():
                    decrypted_value = self.decrypt_value(value.encode())
                    if decrypted_value:
                        decrypted_creds[key] = decrypted_value
                    else:
                        return None

                return decrypted_creds
            else:
                return service_config["credentials"]
        except Exception as e:
            logger.error(f"Failed to get API credentials for {service}: {e}")
            return None

    def _get_credentials_from_env(self, service: str) -> Optional[Dict[str, str]]:
        """
        Get API credentials from environment variables.

        Args:
            service: Service name (e.g., 'binance', 'telegram')

        Returns:
            Dictionary of credential key-value pairs, or None if not found
        """
        try:
            if service.upper() == "BINANCE":
                api_key = os.environ.get("BINANCE_API_KEY")
                api_secret = os.environ.get("BINANCE_API_SECRET")

                if api_key and api_secret:
                    return {"api_key": api_key, "api_secret": api_secret}

            elif service.upper() == "TELEGRAM":
                bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
                chat_id = os.environ.get("TELEGRAM_CHAT_ID")

                if bot_token and chat_id:
                    return {"bot_token": bot_token, "chat_id": chat_id}

            # Add more services as needed

            return None
        except Exception as e:
            logger.error(f"Failed to get credentials from environment: {e}")
            return None

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            # Handle nested keys with dot notation
            if "." in key:
                parts = key.split(".")
                value = self.config
                for part in parts:
                    if part not in value:
                        return default
                    value = value[part]
                return value

            return self.config.get(key, default)
        except Exception as e:
            logger.error(f"Failed to get configuration value for {key}: {e}")
            return default

    def set_config_value(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set

        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle nested keys with dot notation
            if "." in key:
                parts = key.split(".")
                config = self.config
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                config[parts[-1]] = value
            else:
                self.config[key] = value

            return True
        except Exception as e:
            logger.error(f"Failed to set configuration value for {key}: {e}")
            return False

    def validate_config(self, required_keys: Optional[list] = None) -> bool:
        """
        Validate configuration by checking required keys.

        Args:
            required_keys: List of required configuration keys

        Returns:
            True if all required keys exist, False otherwise
        """
        if not required_keys:
            return True

        missing_keys = []
        for key in required_keys:
            if self.get_config_value(key) is None:
                missing_keys.append(key)

        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False

        return True


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create config manager
    config_manager = ConfigManager()

    # Initialize encryption
    config_manager.initialize_encryption("config/encryption.key")

    # Set API credentials
    config_manager.set_api_credentials(
        "binance", {"api_key": "your_api_key", "api_secret": "your_api_secret"}
    )

    # Save configuration
    config_manager.save_config("config/config.json")

    # Get API credentials
    credentials = config_manager.get_api_credentials("binance")
    if credentials:
        print(f"Retrieved API credentials: {credentials}")
