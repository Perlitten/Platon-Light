#!/usr/bin/env python
"""
Platon Light Trading Bot - Main Entry Point

This script serves as the main entry point for the Platon Light trading bot,
allowing users to easily launch different components of the system through
an interactive menu interface.

Usage:
    python run_platon_light.py
"""

import os
import sys
import logging
import time
from pathlib import Path

from platon_light.utils.logging_utils import setup_logging

# Configure logging from config file
setup_logging()
logger = logging.getLogger(__name__)


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    """Print the application header."""
    clear_screen()
    print("=" * 60)
    print("                PLATON LIGHT TRADING BOT")
    print("=" * 60)
    print("A cryptocurrency trading bot for backtesting and live trading")
    print("-" * 60)
    print()


def print_menu():
    """Print the main menu options."""
    print("MAIN MENU")
    print("-" * 60)
    print("[1] Run Core Trading Bot")
    print("[2] Launch Trading Dashboard (Simulation only)")
    print("[3] Run Backtest Workflow")
    print("[4] Optimize Strategy Parameters")
    print("[5] View Documentation")
    print("[6] Exit")
    print()
    return input("Select an option (1-6): ")


def print_trading_mode_menu():
    """Print the trading mode selection menu."""
    print("\nTRADING MODE")
    print("-" * 60)
    print("[1] Dry Run Mode (Simulated trading, no real funds used)")
    print("[2] Real Trading Mode (CAUTION: Will use real funds!)")
    print("[3] Back to Main Menu")
    print()
    return input("Select trading mode (1-3): ")


def print_dashboard_type_menu():
    """Print the dashboard type selection menu."""
    print("\nDASHBOARD TYPE")
    print("-" * 60)
    print("[1] Simple Dashboard")
    print("[2] Enhanced Dashboard with Advanced Visualization")
    print("[3] Back to Main Menu")
    print()
    return input("Select dashboard type (1-3): ")


def run_dashboard(
    mode="dry_run",
    dashboard_type="simple",
    port=8050,
    debug=False,
    initial_balance=10000,
):
    """Launch the trading dashboard.

    Args:
        mode (str): Trading mode ('dry_run' or 'real')
        dashboard_type (str): Dashboard type ('simple' or 'enhanced')
        port (int): Port to run the dashboard on
        debug (bool): Whether to run in debug mode
        initial_balance (float): Initial balance for dry run mode
    """
    try:
        print(f"\nStarting Platon Light Trading Dashboard in {mode.upper()} mode...")
        print(f"Dashboard will be available at http://127.0.0.1:{port}")
        print("Press Ctrl+C to stop the dashboard and return to the menu.\n")

        # Give user time to read the message
        time.sleep(2)

        # Determine which dashboard script to use
        if dashboard_type == "enhanced":
            dashboard_script = "enhanced_trading_dashboard.py"
        else:
            dashboard_script = "simple_trading_dashboard.py"

        # Check if dashboard script exists in new location
        dashboard_path = os.path.join("scripts", "dashboard", dashboard_script)
        if not os.path.exists(dashboard_path):
            # Try the root directory as fallback
            dashboard_path = dashboard_script
            if not os.path.exists(dashboard_path):
                raise FileNotFoundError(
                    f"Dashboard script {dashboard_script} not found"
                )

        # Run the dashboard script with appropriate parameters
        command = f"python {dashboard_path}"
        if mode:
            command += f" --mode {mode}"
        if initial_balance:
            command += f" --initial_balance {initial_balance}"

        # Run the dashboard script
        os.system(command)

    except FileNotFoundError as e:
        logger.error(f"Dashboard script not found: {e}")
        print("\nERROR: Dashboard script not found.")
        print("Make sure the file exists at either:")
        print(f"  - scripts/dashboard/{dashboard_script}")
        print(f"  - {dashboard_script}")
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        print(f"\nERROR: Failed to launch dashboard: {e}")
        input("\nPress Enter to continue...")


def run_backtest():
    """Run the backtest workflow."""
    try:
        print("\nRunning Backtest Workflow...")

        # Check if backtest script exists in new location
        backtest_path = os.path.join(
            "scripts", "backtesting", "run_backtest_workflow.py"
        )
        if not os.path.exists(backtest_path):
            # Try the root directory as fallback
            backtest_path = "run_backtest_workflow.py"
            if not os.path.exists(backtest_path):
                raise FileNotFoundError("Backtest script not found")

        # Run the backtest script
        os.system(f"python {backtest_path}")

        print("\nBacktest completed.")
        input("\nPress Enter to continue...")
    except FileNotFoundError as e:
        logger.error(f"Backtest script not found: {e}")
        print("\nERROR: Backtest script not found.")
        print("Make sure the file exists at either:")
        print("  - scripts/backtesting/run_backtest_workflow.py")
        print("  - run_backtest_workflow.py")
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        print(f"\nERROR: Failed to run backtest: {e}")
        input("\nPress Enter to continue...")


def run_optimize():
    """Optimize strategy parameters."""
    try:
        print("\nOptimizing Strategy Parameters...")

        # Check if optimize script exists in new location
        optimize_path = os.path.join(
            "scripts", "backtesting", "optimize_strategy_parameters.py"
        )
        if not os.path.exists(optimize_path):
            # Try the root directory as fallback
            optimize_path = "optimize_strategy_parameters.py"
            if not os.path.exists(optimize_path):
                raise FileNotFoundError("Optimization script not found")

        # Run the optimization script
        os.system(f"python {optimize_path}")

        print("\nOptimization completed.")
        input("\nPress Enter to continue...")
    except FileNotFoundError as e:
        logger.error(f"Optimization script not found: {e}")
        print("\nERROR: Optimization script not found.")
        print("Make sure the file exists at either:")
        print("  - scripts/backtesting/optimize_strategy_parameters.py")
        print("  - optimize_strategy_parameters.py")
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error optimizing parameters: {e}")
        print(f"\nERROR: Failed to optimize parameters: {e}")
        input("\nPress Enter to continue...")


import asyncio
from platon_light.utils.config_manager import ConfigManager
from platon_light.core.bot import TradingBot


def run_bot():
    """Initializes and runs the core TradingBot."""
    try:
        print("\nInitializing Core Trading Bot...")
        # Load and validate configuration
        config_manager = ConfigManager(config_path='config.yaml')
        config = config_manager.get_config()

        print(f"Starting bot for exchange: {config.general.exchange} in {config.general.mode} mode.")
        if config.general.mode == 'live':
            print("\n" + "="*60)
            print("!! CAUTION: RUNNING IN LIVE MODE. REAL FUNDS WILL BE USED. !!")
            print("="*60 + "\n")
            confirm = input("Are you sure you want to continue? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                print("Aborting live run.")
                return

        bot = TradingBot(config)
        asyncio.run(bot.start())

    except Exception as e:
        logger.error(f"Critical error running the bot: {e}", exc_info=True)
        print(f"\nERROR: A critical error occurred: {e}")
        input("\nPress Enter to continue...")


def view_documentation():
    """View the documentation."""
    try:
        print("\nOpening Documentation...")

        # Check if README exists
        readme_path = "README.md"
        if os.path.exists(readme_path):
            # Display the README content
            with open(readme_path, "r") as f:
                content = f.read()
                print("\n" + content)
        else:
            print("\nDocumentation not found.")

        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error viewing documentation: {e}")
        print(f"\nERROR: Failed to view documentation: {e}")
        input("\nPress Enter to continue...")


def main():
    """Main entry point with interactive menu."""
    while True:
        print_header()
        choice = print_menu()

        if choice == "1":
            # Run Core Trading Bot
            run_bot()

        elif choice == "2":
            # Launch Trading Dashboard
            mode_choice = print_trading_mode_menu()

            if mode_choice == "1" or mode_choice == "2":
                # Get dashboard type
                dashboard_choice = print_dashboard_type_menu()

                # Get initial balance for dry run mode
                initial_balance = 10000
                if mode_choice == "1":  # Dry run mode
                    try:
                        balance_input = input(
                            "\nEnter initial balance (USDT) [default: 10000]: "
                        )
                        if balance_input.strip():
                            initial_balance = float(balance_input)
                    except ValueError:
                        print("Invalid input. Using default balance of 10000 USDT.")
                        initial_balance = 10000

                # Launch appropriate dashboard
                if dashboard_choice == "1":
                    if mode_choice == "1":
                        run_dashboard(
                            mode="dry_run",
                            dashboard_type="simple",
                            initial_balance=initial_balance,
                        )
                    elif mode_choice == "2":
                        print("\nCAUTION: You are about to use REAL trading mode!")
                        print(
                            "This is a simulation and does not affect the Core Trading Bot."
                        )
                        confirm = input("Are you sure you want to continue? (yes/no): ")
                        if confirm.lower() in ["yes", "y"]:
                            run_dashboard(mode="real", dashboard_type="simple")
                elif dashboard_choice == "2":
                    if mode_choice == "1":
                        run_dashboard(
                            mode="dry_run",
                            dashboard_type="enhanced",
                            initial_balance=initial_balance,
                        )
                    elif mode_choice == "2":
                        print("\nCAUTION: You are about to use REAL trading mode!")
                        print(
                            "This is a simulation and does not affect the Core Trading Bot."
                        )
                        confirm = input("Are you sure you want to continue? (yes/no): ")
                        if confirm.lower() in ["yes", "y"]:
                            run_dashboard(mode="real", dashboard_type="enhanced")
            # If mode_choice is '3' or anything else, just go back to main menu

        elif choice == "3":
            # Run Backtest Workflow
            run_backtest()

        elif choice == "4":
            # Optimize Strategy Parameters
            run_optimize()

        elif choice == "5":
            # View Documentation
            view_documentation()

        elif choice == "6":
            # Exit
            print("\nExiting Platon Light Trading Bot. Goodbye!")
            break

        else:
            print("\nInvalid option. Please try again.")
            time.sleep(1)


if __name__ == "__main__":
    main()
