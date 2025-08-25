"""
Logging setup utility for the Platon Light trading bot.
"""
import logging.config
import os
from typing import Dict
import yaml

def setup_logging(config_path: str = 'config.yaml', default_level=logging.INFO):
    """
    Set up logging configuration for the application from a YAML file.

    Args:
        config_path (str): Path to the logging configuration file.
        default_level (int): The logging level to use if configuration fails.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'rt') as f:
                config = yaml.safe_load(f)
            if 'logging' in config:
                # Ensure log directory exists
                log_config = config['logging']
                if 'file' in log_config.get('handlers', {}):
                    log_filename = log_config['handlers']['file'].get('filename')
                    if log_filename:
                        log_dir = os.path.dirname(log_filename)
                        os.makedirs(log_dir, exist_ok=True)
                
                logging.config.dictConfig(log_config)
                logging.getLogger(__name__).info("Logging configured successfully from %s.", config_path)
            else:
                logging.basicConfig(level=default_level, format='%(asctime)s - %(levelname)s - %(message)s')
                logging.getLogger(__name__).info("No 'logging' section in config file. Using basicConfig.")
        except Exception as e:
            logging.basicConfig(level=default_level, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.getLogger(__name__).error("Error configuring logging: %s. Falling back to basicConfig.", e, exc_info=True)
    else:
        logging.basicConfig(level=default_level, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger(__name__).info("Configuration file not found. Using basicConfig.")
