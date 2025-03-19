#!/usr/bin/env python
"""
Monte Carlo Simulation for Backtesting

This script provides functionality to perform Monte Carlo simulations on backtest results
to assess the robustness of trading strategies and estimate the range of possible outcomes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import logging
from scipy import stats

# Add parent directory to path to import Platon Light modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monte_carlo_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Class for performing Monte Carlo simulations on backtest results"""
    
    def __init__(self, output_dir="monte_carlo_results"):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            output_dir (str): Directory to save simulation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logger
    
    def run_simulation(self, backtest_results, num_simulations=1000, confidence_level=0.95, 
                      save_results=True):
        """
        Run Monte Carlo simulation on backtest results.
        
        Args:
            backtest_results (dict): Results from a backtest
            num_simulations (int): Number of simulations to run
            confidence_level (float): Confidence level for intervals (0-1)
            save_results (bool): Whether to save results to disk
            
        Returns:
            dict: Dictionary containing simulation results
        """
        self.logger.info(f"Running Monte Carlo simulation with {num_simulations} iterations")
        self.logger.info(f"Confidence level: {confidence_level * 100}%")
        
        # Extract trade data
        trades = pd.DataFrame(backtest_results['trades']) if 'trades' in backtest_results else None
        
        if trades is None or trades.empty or 'profit_loss_percent' not in trades.columns:
            self.logger.error("No valid trade data found in backtest results")
            return None
        
        # Run simulation based on trade outcomes
        simulation_results = self.simulate_trade_outcomes(
            trades, num_simulations, confidence_level)
        
        # Run simulation based on daily returns
        equity_curve = pd.DataFrame(backtest_results['equity_curve'])
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'], unit='ms')
        equity_curve.set_index('timestamp', inplace=True)
        equity_curve['daily_return'] = equity_curve['equity'].pct_change()
        
        returns_simulation = self.simulate_returns(
            equity_curve, num_simulations, confidence_level)
        
        # Combine results
        all_results = {
            'trade_based_simulation': simulation_results,
            'returns_based_simulation': returns_simulation,
            'backtest_metrics': backtest_results['metrics']
        }
        
        # Save results if requested
        if save_results:
            self.save_simulation_results(all_results)
        
        return all_results
    
    def simulate_trade_outcomes(self, trades, num_simulations=1000, confidence_level=0.95):
        """
        Simulate trade outcomes by bootstrapping from historical trades.
        
        Args:
            trades (DataFrame): DataFrame of historical trades
            num_simulations (int): Number of simulations to run
            confidence_level (float): Confidence level for intervals
            
        Returns:
            dict: Dictionary containing simulation results
        """
        # Extract profit/loss percentages from trades
        pnl_values = trades['profit_loss_percent'].values
        
        # Get number of trades
        num_trades = len(pnl_values)
        
        # Initialize array to store final equity for each simulation
        final_equities = np.zeros(num_simulations)
        
        # Initialize array to store equity curves for each simulation
        equity_curves = np.zeros((num_simulations, num_trades + 1))
        equity_curves[:, 0] = 100  # Start with 100% (normalized)
        
        # Run simulations
        for i in range(num_simulations):
            # Resample trades with replacement
            sampled_pnl = np.random.choice(pnl_values, size=num_trades, replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + sampled_pnl / 100)
            
            # Store equity curve
            equity_curves[i, 1:] = 100 * cumulative_returns
            
            # Store final equity
            final_equities[i] = equity_curves[i, -1]
        
        # Calculate statistics
        mean_final_equity = np.mean(final_equities)
        median_final_equity = np.median(final_equities)
        std_final_equity = np.std(final_equities)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(final_equities, lower_percentile)
        upper_bound = np.percentile(final_equities, upper_percentile)
        
        # Calculate probability of profit
        prob_profit = np.mean(final_equities > 100)
        
        # Calculate worst-case scenarios
        worst_case_5pct = np.percentile(final_equities, 5)
        worst_case_1pct = np.percentile(final_equities, 1)
        
        # Calculate maximum drawdowns for each simulation
        max_drawdowns = np.zeros(num_simulations)
        for i in range(num_simulations):
            # Calculate drawdowns
            peak = np.maximum.accumulate(equity_curves[i])
            drawdown = (equity_curves[i] - peak) / peak * 100
            max_drawdowns[i] = np.min(drawdown)
        
        # Calculate drawdown statistics
        mean_max_drawdown = np.mean(max_drawdowns)
        median_max_drawdown = np.median(max_drawdowns)
        worst_drawdown = np.min(max_drawdowns)
        
        # Calculate drawdown confidence interval
        drawdown_lower_bound = np.percentile(max_drawdowns, lower_percentile)
        drawdown_upper_bound = np.percentile(max_drawdowns, upper_percentile)
        
        # Plot equity curves
        self.plot_equity_curves(equity_curves, confidence_level, 'trade_based')
        
        # Plot final equity distribution
        self.plot_final_equity_distribution(final_equities, confidence_level, 'trade_based')
        
        # Plot drawdown distribution
        self.plot_drawdown_distribution(max_drawdowns, confidence_level, 'trade_based')
        
        # Compile results
        results = {
            'num_simulations': num_simulations,
            'confidence_level': confidence_level,
            'mean_final_equity': mean_final_equity,
            'median_final_equity': median_final_equity,
            'std_final_equity': std_final_equity,
            'confidence_interval': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'probability_of_profit': prob_profit,
            'worst_case_scenarios': {
                '5_percent': worst_case_5pct,
                '1_percent': worst_case_1pct
            },
            'drawdown_statistics': {
                'mean_max_drawdown': mean_max_drawdown,
                'median_max_drawdown': median_max_drawdown,
                'worst_drawdown': worst_drawdown,
                'confidence_interval': {
                    'lower_bound': drawdown_lower_bound,
                    'upper_bound': drawdown_upper_bound
                }
            }
        }
        
        # Log key results
        self.logger.info("\nTrade-Based Monte Carlo Simulation Results:")
        self.logger.info(f"Mean Final Equity: {mean_final_equity:.2f}%")
        self.logger.info(f"Median Final Equity: {median_final_equity:.2f}%")
        self.logger.info(f"{confidence_level*100}% Confidence Interval: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
        self.logger.info(f"Probability of Profit: {prob_profit*100:.2f}%")
        self.logger.info(f"Mean Maximum Drawdown: {mean_max_drawdown:.2f}%")
        self.logger.info(f"Worst Drawdown (across all simulations): {worst_drawdown:.2f}%")
        
        return results
    
    def simulate_returns(self, equity_curve, num_simulations=1000, confidence_level=0.95):
        """
        Simulate equity curves by bootstrapping from historical returns.
        
        Args:
            equity_curve (DataFrame): DataFrame with equity curve data
            num_simulations (int): Number of simulations to run
            confidence_level (float): Confidence level for intervals
            
        Returns:
            dict: Dictionary containing simulation results
        """
        # Extract daily returns
        daily_returns = equity_curve['daily_return'].dropna().values
        
        # Get number of days
        num_days = len(daily_returns)
        
        # Initialize array to store final equity for each simulation
        final_equities = np.zeros(num_simulations)
        
        # Initialize array to store equity curves for each simulation
        equity_curves = np.zeros((num_simulations, num_days + 1))
        equity_curves[:, 0] = 100  # Start with 100% (normalized)
        
        # Run simulations
        for i in range(num_simulations):
            # Resample daily returns with replacement
            sampled_returns = np.random.choice(daily_returns, size=num_days, replace=True)
            
            # Calculate cumulative returns
            for j in range(num_days):
                equity_curves[i, j+1] = equity_curves[i, j] * (1 + sampled_returns[j])
            
            # Store final equity
            final_equities[i] = equity_curves[i, -1]
        
        # Calculate statistics
        mean_final_equity = np.mean(final_equities)
        median_final_equity = np.median(final_equities)
        std_final_equity = np.std(final_equities)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(final_equities, lower_percentile)
        upper_bound = np.percentile(final_equities, upper_percentile)
        
        # Calculate probability of profit
        prob_profit = np.mean(final_equities > 100)
        
        # Calculate worst-case scenarios
        worst_case_5pct = np.percentile(final_equities, 5)
        worst_case_1pct = np.percentile(final_equities, 1)
        
        # Calculate maximum drawdowns for each simulation
        max_drawdowns = np.zeros(num_simulations)
        for i in range(num_simulations):
            # Calculate drawdowns
            peak = np.maximum.accumulate(equity_curves[i])
            drawdown = (equity_curves[i] - peak) / peak * 100
            max_drawdowns[i] = np.min(drawdown)
        
        # Calculate drawdown statistics
        mean_max_drawdown = np.mean(max_drawdowns)
        median_max_drawdown = np.median(max_drawdowns)
        worst_drawdown = np.min(max_drawdowns)
        
        # Calculate drawdown confidence interval
        drawdown_lower_bound = np.percentile(max_drawdowns, lower_percentile)
        drawdown_upper_bound = np.percentile(max_drawdowns, upper_percentile)
        
        # Plot equity curves
        self.plot_equity_curves(equity_curves, confidence_level, 'returns_based')
        
        # Plot final equity distribution
        self.plot_final_equity_distribution(final_equities, confidence_level, 'returns_based')
        
        # Plot drawdown distribution
        self.plot_drawdown_distribution(max_drawdowns, confidence_level, 'returns_based')
        
        # Compile results
        results = {
            'num_simulations': num_simulations,
            'confidence_level': confidence_level,
            'mean_final_equity': mean_final_equity,
            'median_final_equity': median_final_equity,
            'std_final_equity': std_final_equity,
            'confidence_interval': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'probability_of_profit': prob_profit,
            'worst_case_scenarios': {
                '5_percent': worst_case_5pct,
                '1_percent': worst_case_1pct
            },
            'drawdown_statistics': {
                'mean_max_drawdown': mean_max_drawdown,
                'median_max_drawdown': median_max_drawdown,
                'worst_drawdown': worst_drawdown,
                'confidence_interval': {
                    'lower_bound': drawdown_lower_bound,
                    'upper_bound': drawdown_upper_bound
                }
            }
        }
        
        # Log key results
        self.logger.info("\nReturns-Based Monte Carlo Simulation Results:")
        self.logger.info(f"Mean Final Equity: {mean_final_equity:.2f}%")
        self.logger.info(f"Median Final Equity: {median_final_equity:.2f}%")
        self.logger.info(f"{confidence_level*100}% Confidence Interval: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
        self.logger.info(f"Probability of Profit: {prob_profit*100:.2f}%")
        self.logger.info(f"Mean Maximum Drawdown: {mean_max_drawdown:.2f}%")
        self.logger.info(f"Worst Drawdown (across all simulations): {worst_drawdown:.2f}%")
        
        return results
    
    def plot_equity_curves(self, equity_curves, confidence_level, simulation_type):
        """
        Plot equity curves from Monte Carlo simulation.
        
        Args:
            equity_curves (ndarray): Array of equity curves from simulation
            confidence_level (float): Confidence level for intervals
            simulation_type (str): Type of simulation ('trade_based' or 'returns_based')
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate percentiles for each time step
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        num_simulations, num_steps = equity_curves.shape
        
        # Calculate percentiles for each time step
        median_curve = np.median(equity_curves, axis=0)
        lower_bound = np.percentile(equity_curves, lower_percentile, axis=0)
        upper_bound = np.percentile(equity_curves, upper_percentile, axis=0)
        
        # Plot a subset of simulations for visual clarity
        subset_size = min(100, num_simulations)
        indices = np.random.choice(num_simulations, subset_size, replace=False)
        
        for i in indices:
            plt.plot(equity_curves[i], color='lightgray', alpha=0.3)
        
        # Plot median and confidence intervals
        plt.plot(median_curve, color='blue', linewidth=2, label='Median')
        plt.plot(lower_bound, color='red', linewidth=2, 
                label=f'{lower_percentile:.1f}th Percentile')
        plt.plot(upper_bound, color='green', linewidth=2, 
                label=f'{upper_percentile:.1f}th Percentile')
        
        plt.title(f'{simulation_type.capitalize()} Monte Carlo Simulation - Equity Curves')
        plt.xlabel('Trade Number' if simulation_type == 'trade_based' else 'Day')
        plt.ylabel('Equity (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.output_dir}/{simulation_type}_equity_curves.png"
        plt.savefig(filename)
        plt.close()
    
    def plot_final_equity_distribution(self, final_equities, confidence_level, simulation_type):
        """
        Plot distribution of final equity values.
        
        Args:
            final_equities (ndarray): Array of final equity values
            confidence_level (float): Confidence level for intervals
            simulation_type (str): Type of simulation ('trade_based' or 'returns_based')
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(final_equities, lower_percentile)
        upper_bound = np.percentile(final_equities, upper_percentile)
        
        # Plot histogram
        plt.hist(final_equities, bins=50, alpha=0.7, color='skyblue')
        
        # Plot confidence interval lines
        plt.axvline(lower_bound, color='red', linestyle='--', 
                   label=f'{lower_percentile:.1f}th Percentile: {lower_bound:.2f}%')
        plt.axvline(upper_bound, color='green', linestyle='--', 
                   label=f'{upper_percentile:.1f}th Percentile: {upper_bound:.2f}%')
        
        # Plot mean and median
        mean_equity = np.mean(final_equities)
        median_equity = np.median(final_equities)
        
        plt.axvline(mean_equity, color='blue', linestyle='-', 
                   label=f'Mean: {mean_equity:.2f}%')
        plt.axvline(median_equity, color='purple', linestyle='-', 
                   label=f'Median: {median_equity:.2f}%')
        
        # Plot initial equity line
        plt.axvline(100, color='black', linestyle='-', label='Initial Equity (100%)')
        
        plt.title(f'{simulation_type.capitalize()} Monte Carlo Simulation - Final Equity Distribution')
        plt.xlabel('Final Equity (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.output_dir}/{simulation_type}_final_equity_distribution.png"
        plt.savefig(filename)
        plt.close()
    
    def plot_drawdown_distribution(self, max_drawdowns, confidence_level, simulation_type):
        """
        Plot distribution of maximum drawdowns.
        
        Args:
            max_drawdowns (ndarray): Array of maximum drawdowns
            confidence_level (float): Confidence level for intervals
            simulation_type (str): Type of simulation ('trade_based' or 'returns_based')
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(max_drawdowns, lower_percentile)
        upper_bound = np.percentile(max_drawdowns, upper_percentile)
        
        # Plot histogram
        plt.hist(max_drawdowns, bins=50, alpha=0.7, color='salmon')
        
        # Plot confidence interval lines
        plt.axvline(lower_bound, color='red', linestyle='--', 
                   label=f'{lower_percentile:.1f}th Percentile: {lower_bound:.2f}%')
        plt.axvline(upper_bound, color='green', linestyle='--', 
                   label=f'{upper_percentile:.1f}th Percentile: {upper_bound:.2f}%')
        
        # Plot mean and median
        mean_drawdown = np.mean(max_drawdowns)
        median_drawdown = np.median(max_drawdowns)
        worst_drawdown = np.min(max_drawdowns)
        
        plt.axvline(mean_drawdown, color='blue', linestyle='-', 
                   label=f'Mean: {mean_drawdown:.2f}%')
        plt.axvline(median_drawdown, color='purple', linestyle='-', 
                   label=f'Median: {median_drawdown:.2f}%')
        plt.axvline(worst_drawdown, color='black', linestyle='-', 
                   label=f'Worst: {worst_drawdown:.2f}%')
        
        plt.title(f'{simulation_type.capitalize()} Monte Carlo Simulation - Maximum Drawdown Distribution')
        plt.xlabel('Maximum Drawdown (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        filename = f"{self.output_dir}/{simulation_type}_drawdown_distribution.png"
        plt.savefig(filename)
        plt.close()
    
    def save_simulation_results(self, results):
        """
        Save simulation results to disk.
        
        Args:
            results (dict): Simulation results
            
        Returns:
            str: Path to saved results
        """
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.output_dir}/monte_carlo_simulation_{timestamp}.json"
        
        # Convert results to JSON
        results_json = json.dumps(results, default=str, indent=2)
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(results_json)
        
        self.logger.info(f"Simulation results saved to {filename}")
        
        return filename


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation for Backtesting')
    
    parser.add_argument('--backtest-results', type=str, required=True,
                        help='Path to backtest results JSON file')
    parser.add_argument('--num-simulations', type=int, default=1000,
                        help='Number of simulations to run')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level for intervals (0-1)')
    parser.add_argument('--output-dir', type=str, default='monte_carlo_results',
                        help='Directory to save simulation results')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Load backtest results
    with open(args.backtest_results, 'r') as f:
        backtest_results = json.load(f)
    
    # Create Monte Carlo simulator
    simulator = MonteCarloSimulator(output_dir=args.output_dir)
    
    # Run simulation
    simulator.run_simulation(
        backtest_results=backtest_results,
        num_simulations=args.num_simulations,
        confidence_level=args.confidence_level
    )


if __name__ == '__main__':
    main()
