"""
Configuration manager for loading and validating bot configuration using Pydantic.
"""
import os
import yaml
import logging
from typing import Optional
from pydantic import ValidationError

from platon_light.core.config_models import BotConfig

class ConfigManager:
    """
    Loads, validates, and provides access to the bot's configuration.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config: Optional[BotConfig] = None
        self._load_and_validate()

    def _load_and_validate(self):
        """
        Load the YAML configuration file and validate it with Pydantic.
        """
        try:
            self.logger.info(f"Loading configuration from {self.config_path}...")
            with open(self.config_path, "r") as file:
                raw_config = yaml.safe_load(file) or {}
            
            self.config = BotConfig(**raw_config)
            self.logger.info("Configuration loaded and validated successfully.")

        except FileNotFoundError:
            self.logger.warning(
                f"Configuration file not found at {self.config_path}. "
                "Using default configuration."
            )
            self.config = BotConfig() # Create config with all defaults
        except ValidationError as e:
            self.logger.error("Configuration validation failed!")
            self.logger.error(e)
            raise ValueError("Invalid configuration. Please check config.yaml.") from e
        except Exception as e:
            self.logger.error(f"Failed to load or parse configuration: {e}")
            raise

    def get_config(self) -> BotConfig:
        """
        Get the validated configuration object.

        Returns:
            The BotConfig object.
        """
        if not self.config:
            raise RuntimeError("Configuration has not been loaded.")
        return self.config
