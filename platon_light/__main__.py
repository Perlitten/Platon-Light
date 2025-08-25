"""
Main entry point for the Platon Light trading bot
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from platon_light.core.bot import TradingBot
from platon_light.utils.logger import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Platon Light - Advanced Binance Scalping Trading Bot"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in simulation mode without real trading",
    )
    parser.add_argument("--pair", type=str, help="Override trading pair in config")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--version", action="store_true", help="Show version information and exit"
    )

    return parser.parse_args()


def load_configuration(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


async def main():
    """Main function to run the trading bot"""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    args = parse_arguments()

    # Show version and exit if requested
    if args.version:
        from platon_light import __version__

        print(f"Platon Light v{__version__}")
        sys.exit(0)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    # Load configuration
    config = load_configuration(args.config)

    # Override configuration with command line arguments
    if args.dry_run:
        config["general"]["mode"] = "dry-run"

    if args.pair:
        # Override the first trading pair
        if "quote_currencies" in config["general"]:
            base = config["general"]["base_currency"]
            config["general"]["quote_currencies"] = [args.pair.replace(base, "")]

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Initialize and run the trading bot
    bot = TradingBot(config)
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.exception(f"Unhandled exception: {e}")
        sys.exit(1)
