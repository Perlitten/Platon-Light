# Evaluating Backtest Results

This guide explains how to properly evaluate and interpret backtesting results using the Platon Light backtesting module.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Performance Metrics](#key-performance-metrics)
3. [Statistical Significance](#statistical-significance)
4. [Avoiding Overfitting](#avoiding-overfitting)
5. [Robustness Testing](#robustness-testing)
6. [Comparative Analysis](#comparative-analysis)
7. [Visualization Techniques](#visualization-techniques)
8. [Implementation in Platon Light](#implementation-in-platon-light)
9. [Best Practices](#best-practices)
10. [Example Analysis](#example-analysis)

## Introduction

Backtesting is only as valuable as your ability to interpret the results correctly. A thorough evaluation helps:

- Distinguish between genuine edge and statistical noise
- Identify potential issues before deploying capital
- Understand a strategy's strengths and weaknesses
- Make data-driven improvements to trading systems

This guide covers comprehensive methods for evaluating backtest results to ensure your trading strategies are robust and reliable.

## Key Performance Metrics

### Return Metrics

1. **Total Return**: Overall percentage gain/loss
2. **Annualized Return**: Return normalized to yearly performance
3. **Compound Annual Growth Rate (CAGR)**: Smoothed annual growth rate
4. **Monthly/Quarterly Returns**: Performance broken down by time periods

### Risk Metrics

1. **Maximum Drawdown**: Largest peak-to-trough decline
2. **Value at Risk (VaR)**: Potential loss at a given confidence level
3. **Expected Shortfall (CVaR)**: Average loss beyond VaR threshold
4. **Volatility**: Standard deviation of returns
5. **Downside Deviation**: Standard deviation of negative returns only

### Risk-Adjusted Return Metrics

1. **Sharpe Ratio**: Return per unit of risk (volatility)
   ```
   Sharpe Ratio = (Strategy Return - Risk-Free Rate) / Strategy Volatility
   ```

2. **Sortino Ratio**: Return per unit of downside risk
   ```
   Sortino Ratio = (Strategy Return - Risk-Free Rate) / Downside Deviation
   ```

3. **Calmar Ratio**: Return relative to maximum drawdown
   ```
   Calmar Ratio = Annualized Return / Maximum Drawdown
   ```

4. **Omega Ratio**: Probability-weighted ratio of gains versus losses
5. **Information Ratio**: Excess return per unit of tracking risk

### Trade Statistics

1. **Win Rate**: Percentage of winning trades
2. **Profit Factor**: Gross profit divided by gross loss
3. **Average Win/Loss**: Average profit of winning trades vs. losing trades
4. **Win/Loss Ratio**: Ratio of average win to average loss
5. **Expectancy**: Average amount you can expect to win (or lose) per trade
   ```
   Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
   ```

6. **Average Holding Period**: Average time in trades
7. **Maximum Consecutive Wins/Losses**: Longest streaks

### Exposure Metrics

1. **Percent Time in Market**: Proportion of time with open positions
2. **Average Exposure**: Average capital deployed in positions
3. **Maximum Leverage**: Highest leverage used during testing period

## Statistical Significance

### Assessing Significance

1. **T-Test**: Determine if returns are significantly different from zero
2. **Monte Carlo Simulation**: Randomize trade sequence to assess robustness
3. **Bootstrap Analysis**: Resample trades to create confidence intervals
4. **Minimum Sample Size**: Ensure enough trades for statistical validity

### Implementation Example

```python
from scipy import stats
import numpy as np

def assess_statistical_significance(trade_returns, confidence_level=0.95):
    """
    Assess if strategy returns are statistically significant.
    
    Args:
        trade_returns: Array of individual trade returns
        confidence_level: Statistical confidence level (default: 0.95)
        
    Returns:
        Dictionary with t-statistic, p-value, and significance assessment
    """
    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(trade_returns, 0)
    
    # Calculate confidence interval
    n = len(trade_returns)
    std_error = np.std(trade_returns, ddof=1) / np.sqrt(n)
    margin_error = std_error * stats.t.ppf((1 + confidence_level) / 2, n - 1)
    mean_return = np.mean(trade_returns)
    conf_interval = (mean_return - margin_error, mean_return + margin_error)
    
    # Assess significance
    is_significant = p_value < (1 - confidence_level)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence_interval': conf_interval,
        'is_significant': is_significant,
        'sample_size': n
    }
```

## Avoiding Overfitting

### Warning Signs of Overfitting

1. **Too Many Parameters**: Complex strategies with many parameters
2. **Excessive Optimization**: Extensive parameter tuning
3. **Perfect Performance**: Suspiciously high returns with minimal drawdowns
4. **Parameter Sensitivity**: Performance collapses with small parameter changes
5. **Poor Out-of-Sample Performance**: Strategy fails on unseen data

### Prevention Techniques

1. **Train/Test Split**: Divide data into in-sample and out-of-sample periods
2. **Cross-Validation**: Use multiple train/test splits
3. **Walk-Forward Analysis**: Continuously retrain on expanding windows
4. **Parameter Robustness Testing**: Test performance across parameter ranges
5. **Complexity Penalties**: Apply information criteria (AIC, BIC) to penalize complexity

## Robustness Testing

### Stress Testing Methods

1. **Market Regime Testing**: Test across bull, bear, and sideways markets
2. **Volatility Regime Testing**: Test across high and low volatility periods
3. **Parameter Sensitivity**: Vary parameters to test stability
4. **Monte Carlo Simulation**: Randomize aspects of the backtest
5. **Synthetic Market Data**: Test on artificially generated data

### Implementation Example

```python
def perform_robustness_testing(strategy, data, base_params, param_ranges, metric='sharpe_ratio'):
    """
    Test strategy robustness by varying parameters.
    
    Args:
        strategy: Strategy object
        data: Market data
        base_params: Base parameters that work well
        param_ranges: Dictionary of parameter ranges to test
        metric: Performance metric to track
        
    Returns:
        DataFrame with parameter variations and resulting metrics
    """
    results = []
    
    # For each parameter to test
    for param_name, param_range in param_ranges.items():
        for param_value in param_range:
            # Create modified parameters
            test_params = base_params.copy()
            test_params[param_name] = param_value
            
            # Run backtest with modified parameters
            backtest_result = run_backtest(strategy, data, test_params)
            performance = calculate_metrics(backtest_result)
            
            # Store results
            results.append({
                'parameter': param_name,
                'value': param_value,
                'base_value': base_params[param_name],
                'deviation': (param_value / base_params[param_name]) - 1,
                metric: performance[metric],
                'change': (performance[metric] / base_performance[metric]) - 1
            })
    
    return pd.DataFrame(results)
```

## Comparative Analysis

### Benchmarking Approaches

1. **Market Index Comparison**: Compare to relevant market indices
2. **Buy-and-Hold Comparison**: Compare to simple buy-and-hold strategy
3. **Risk-Free Rate Comparison**: Compare to risk-free investment
4. **Strategy Variants Comparison**: Compare to variations of your strategy
5. **Peer Strategy Comparison**: Compare to other strategies in same asset class

### Implementation Example

```python
def compare_strategies(strategy_results, benchmark_results, risk_free_rate=0.02):
    """
    Compare strategy performance against benchmarks.
    
    Args:
        strategy_results: Dictionary of strategy backtest results
        benchmark_results: Dictionary of benchmark backtest results
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with comparative metrics
    """
    comparison = {}
    
    # Calculate metrics for each strategy and benchmark
    for name, results in {**strategy_results, **benchmark_results}.items():
        metrics = calculate_metrics(results, risk_free_rate)
        comparison[name] = metrics
    
    # Calculate relative metrics
    for strategy_name in strategy_results.keys():
        for benchmark_name in benchmark_results.keys():
            # Calculate alpha (excess return over benchmark)
            alpha = comparison[strategy_name]['annualized_return'] - comparison[benchmark_name]['annualized_return']
            
            # Calculate beta (sensitivity to benchmark)
            beta = calculate_beta(strategy_results[strategy_name]['returns'], benchmark_results[benchmark_name]['returns'])
            
            # Calculate information ratio
            tracking_error = calculate_tracking_error(strategy_results[strategy_name]['returns'], benchmark_results[benchmark_name]['returns'])
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            # Store relative metrics
            comparison[f"{strategy_name}_vs_{benchmark_name}"] = {
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error
            }
    
    return pd.DataFrame(comparison)
```

## Visualization Techniques

### Essential Visualizations

1. **Equity Curve**: Plot of cumulative returns over time
2. **Drawdown Chart**: Visualization of drawdowns over time
3. **Monthly/Yearly Returns**: Heatmap of returns by time period
4. **Return Distribution**: Histogram of return distribution
5. **Rolling Performance**: Moving window of performance metrics
6. **Trade Analysis**: Scatter plot of individual trade performance
7. **Regime Analysis**: Performance across different market regimes

### Advanced Visualizations

1. **Monte Carlo Simulations**: Probability cones for future performance
2. **Parameter Sensitivity**: Heatmaps of parameter combinations
3. **Correlation Matrix**: Relationships between strategy and benchmarks
4. **Underwater Plot**: Time spent in drawdowns
5. **Trade Clustering**: Temporal clustering of winning/losing trades

## Implementation in Platon Light

The Platon Light backtesting module provides built-in tools for evaluating backtest results:

```python
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer

# Create analyzer and visualizer
analyzer = PerformanceAnalyzer(config)
visualizer = BacktestVisualizer(config)

# Analyze backtest results
metrics = analyzer.analyze(results)

# Print key metrics
print(f"Total Return: {metrics['return_percent']:.2f}%")
print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")

# Visualize results
visualizer.plot_equity_curve(results)
visualizer.plot_drawdown_curve(results)
visualizer.plot_monthly_returns(results)
visualizer.plot_return_distribution(results)
visualizer.plot_rolling_performance(results, window=20)

# Compare to benchmark
benchmark_results = run_benchmark_backtest(data)
comparison = analyzer.compare_strategies({
    'Strategy': results,
    'Benchmark': benchmark_results
})
visualizer.plot_strategy_comparison({
    'Strategy': results,
    'Benchmark': benchmark_results
})

# Assess statistical significance
significance = analyzer.assess_statistical_significance(results['trade_returns'])
print(f"Statistically Significant: {significance['is_significant']}")
print(f"p-value: {significance['p_value']:.4f}")

# Perform Monte Carlo simulation
mc_results = analyzer.monte_carlo_simulation(results, num_simulations=1000)
visualizer.plot_monte_carlo_simulations(mc_results)
```

## Best Practices

1. **Use Multiple Metrics**: Don't rely on a single performance metric
2. **Compare to Benchmarks**: Always compare to relevant benchmarks
3. **Test Statistical Significance**: Ensure results aren't due to chance
4. **Validate Out-of-Sample**: Test on unseen data
5. **Stress Test Parameters**: Ensure robustness across parameter ranges
6. **Consider Transaction Costs**: Include realistic trading costs
7. **Analyze Drawdowns**: Understand worst-case scenarios
8. **Check for Survivorship Bias**: Use point-in-time data when possible
9. **Beware of Look-Ahead Bias**: Ensure strategy only uses data available at the time
10. **Document Assumptions**: Record all assumptions made in the backtest

## Example Analysis

Here's a comprehensive example of evaluating backtest results:

```python
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy import stats

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.backtesting.data_loader import DataLoader
from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer
from platon_light.backtesting.visualization import BacktestVisualizer
from platon_light.core.strategy_factory import StrategyFactory


def evaluate_backtest_results():
    """Comprehensive evaluation of backtest results"""
    # Load backtest results
    results = load_backtest_results()
    
    # Create analyzer and visualizer
    config = {
        'backtesting': {
            'initial_capital': 10000,
            'risk_free_rate': 0.02  # 2% annual risk-free rate
        }
    }
    analyzer = PerformanceAnalyzer(config)
    visualizer = BacktestVisualizer(config)
    
    # 1. Calculate performance metrics
    metrics = analyzer.analyze(results)
    print_performance_metrics(metrics)
    
    # 2. Visualize performance
    visualize_performance(results, visualizer)
    
    # 3. Compare to benchmarks
    benchmark_results = load_benchmark_results()
    compare_to_benchmarks(results, benchmark_results, analyzer, visualizer)
    
    # 4. Assess statistical significance
    assess_significance(results, analyzer)
    
    # 5. Test robustness
    test_robustness(results, analyzer, visualizer)
    
    # 6. Analyze market regimes
    analyze_market_regimes(results, analyzer, visualizer)
    
    # 7. Perform Monte Carlo simulation
    perform_monte_carlo(results, analyzer, visualizer)
    
    # 8. Generate comprehensive report
    generate_report(results, metrics, benchmark_results, analyzer)


def print_performance_metrics(metrics):
    """Print key performance metrics"""
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Total Return: {metrics['return_percent']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Expectancy: ${metrics['expectancy']:.2f}")
    print(f"Average Trade: ${metrics['average_trade']:.2f}")
    print(f"Average Win: ${metrics['average_win']:.2f}")
    print(f"Average Loss: ${metrics['average_loss']:.2f}")
    print(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")
    print(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")
    print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
    print(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
    print(f"Risk-Adjusted Return: {metrics['risk_adjusted_return']:.2f}%")
    print(f"Percent Time in Market: {metrics['percent_time_in_market']:.2f}%")


def visualize_performance(results, visualizer):
    """Create standard performance visualizations"""
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Equity curve
    visualizer.plot_equity_curve(results, title="Strategy Equity Curve")
    
    # Drawdown chart
    visualizer.plot_drawdown_curve(results, title="Strategy Drawdowns")
    
    # Monthly returns heatmap
    visualizer.plot_monthly_returns(results, title="Monthly Returns")
    
    # Return distribution
    visualizer.plot_return_distribution(results, title="Return Distribution")
    
    # Rolling performance
    visualizer.plot_rolling_performance(results, window=20, title="Rolling Performance Metrics")
    
    # Trade analysis
    visualizer.plot_trade_analysis(results, title="Trade Analysis")
    
    # Underwater plot
    visualizer.plot_underwater(results, title="Underwater Plot")
    
    print("Visualizations generated successfully")


def compare_to_benchmarks(results, benchmark_results, analyzer, visualizer):
    """Compare strategy to benchmarks"""
    print("\n=== BENCHMARK COMPARISON ===")
    
    # Calculate comparative metrics
    comparison = analyzer.compare_strategies({
        'Strategy': results,
        'Buy-and-Hold': benchmark_results['buy_and_hold'],
        'Market Index': benchmark_results['market_index']
    })
    
    # Print comparison table
    print(comparison)
    
    # Visualize comparison
    visualizer.plot_strategy_comparison({
        'Strategy': results,
        'Buy-and-Hold': benchmark_results['buy_and_hold'],
        'Market Index': benchmark_results['market_index']
    }, title="Strategy vs. Benchmarks")
    
    # Calculate alpha and beta
    alpha_beta = analyzer.calculate_alpha_beta(
        results['returns'], 
        benchmark_results['market_index']['returns']
    )
    
    print(f"Alpha: {alpha_beta['alpha']:.4f}")
    print(f"Beta: {alpha_beta['beta']:.4f}")
    print(f"Information Ratio: {alpha_beta['information_ratio']:.4f}")


def assess_significance(results, analyzer):
    """Assess statistical significance of results"""
    print("\n=== STATISTICAL SIGNIFICANCE ===")
    
    # Perform t-test on trade returns
    significance = analyzer.assess_statistical_significance(
        results['trade_returns']
    )
    
    print(f"Number of Trades: {significance['sample_size']}")
    print(f"Mean Return per Trade: {np.mean(results['trade_returns']):.4f}")
    print(f"t-statistic: {significance['t_statistic']:.4f}")
    print(f"p-value: {significance['p_value']:.4f}")
    print(f"Confidence Interval (95%): {significance['confidence_interval']}")
    print(f"Statistically Significant: {significance['is_significant']}")
    
    # Minimum required sample size
    min_sample = analyzer.calculate_minimum_sample_size(
        results['trade_returns'],
        margin_of_error=0.01,
        confidence_level=0.95
    )
    
    print(f"Minimum Required Sample Size: {min_sample}")
    print(f"Sample Size Sufficient: {significance['sample_size'] >= min_sample}")


def test_robustness(results, analyzer, visualizer):
    """Test strategy robustness"""
    print("\n=== ROBUSTNESS TESTING ===")
    
    # Parameter sensitivity analysis
    sensitivity = analyzer.parameter_sensitivity_analysis(
        results['strategy_config'],
        results['parameter_ranges']
    )
    
    print("Parameter Sensitivity:")
    print(sensitivity)
    
    # Visualize parameter sensitivity
    visualizer.plot_parameter_sensitivity(
        sensitivity,
        title="Parameter Sensitivity Analysis"
    )
    
    # Walk-forward analysis results
    wfa_results = analyzer.get_walk_forward_results(results)
    
    print("Walk-Forward Analysis:")
    print(f"In-Sample Sharpe: {wfa_results['in_sample_sharpe']:.4f}")
    print(f"Out-of-Sample Sharpe: {wfa_results['out_of_sample_sharpe']:.4f}")
    print(f"Robustness Ratio: {wfa_results['robustness_ratio']:.4f}")
    
    # Visualize walk-forward results
    visualizer.plot_walk_forward_results(
        wfa_results,
        title="Walk-Forward Analysis"
    )


def analyze_market_regimes(results, analyzer, visualizer):
    """Analyze performance across market regimes"""
    print("\n=== MARKET REGIME ANALYSIS ===")
    
    # Identify market regimes
    regimes = analyzer.identify_market_regimes(results['market_data'])
    
    # Calculate performance by regime
    regime_performance = analyzer.analyze_performance_by_regime(
        results,
        regimes
    )
    
    print("Performance by Market Regime:")
    print(regime_performance)
    
    # Visualize regime performance
    visualizer.plot_regime_performance(
        regime_performance,
        title="Performance by Market Regime"
    )


def perform_monte_carlo(results, analyzer, visualizer):
    """Perform Monte Carlo simulation"""
    print("\n=== MONTE CARLO SIMULATION ===")
    
    # Run Monte Carlo simulation
    mc_results = analyzer.monte_carlo_simulation(
        results,
        num_simulations=1000,
        method='trade_resample'
    )
    
    print(f"Expected Return: {mc_results['expected_return']:.2f}%")
    print(f"5th Percentile Return: {mc_results['percentiles'][5]:.2f}%")
    print(f"95th Percentile Return: {mc_results['percentiles'][95]:.2f}%")
    print(f"Value at Risk (95%): {mc_results['var_95']:.2f}%")
    print(f"Expected Shortfall (95%): {mc_results['cvar_95']:.2f}%")
    print(f"Probability of Profit: {mc_results['probability_of_profit']:.2f}%")
    
    # Visualize Monte Carlo results
    visualizer.plot_monte_carlo_simulations(
        mc_results,
        title="Monte Carlo Simulation"
    )


def generate_report(results, metrics, benchmark_results, analyzer):
    """Generate comprehensive backtest report"""
    print("\n=== GENERATING COMPREHENSIVE REPORT ===")
    
    # Create report dataframe
    report = {
        'Strategy Name': results['strategy_name'],
        'Test Period': f"{results['start_date']} to {results['end_date']}",
        'Initial Capital': f"${results['initial_capital']:.2f}",
        'Final Capital': f"${results['final_capital']:.2f}",
        'Total Return': f"{metrics['return_percent']:.2f}%",
        'Annualized Return': f"{metrics['annualized_return']:.2f}%",
        'Benchmark Return': f"{benchmark_results['market_index']['return_percent']:.2f}%",
        'Alpha': f"{metrics['alpha']:.4f}",
        'Beta': f"{metrics['beta']:.4f}",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
        'Sortino Ratio': f"{metrics['sortino_ratio']:.2f}",
        'Calmar Ratio': f"{metrics['calmar_ratio']:.2f}",
        'Max Drawdown': f"{metrics['max_drawdown_percent']:.2f}%",
        'Win Rate': f"{metrics['win_rate']:.2f}%",
        'Profit Factor': f"{metrics['profit_factor']:.2f}",
        'Expectancy': f"${metrics['expectancy']:.2f}",
        'Recovery Factor': f"{metrics['recovery_factor']:.2f}",
        'Statistically Significant': str(metrics['is_significant']),
        'p-value': f"{metrics['p_value']:.4f}",
        'Robustness Score': f"{metrics['robustness_score']:.2f}/10"
    }
    
    # Save report to file
    report_df = pd.DataFrame([report])
    report_df.to_csv('backtest_report.csv', index=False)
    
    print("Comprehensive report generated and saved to 'backtest_report.csv'")


if __name__ == "__main__":
    evaluate_backtest_results()
```

## Conclusion

Properly evaluating backtest results is essential for developing robust trading strategies. By applying the techniques in this guide, you can:

1. Accurately assess strategy performance
2. Identify and avoid overfitted strategies
3. Understand how a strategy will likely perform in different market conditions
4. Make data-driven improvements to your trading systems
5. Build confidence in strategies before deploying real capital

Remember that backtesting is inherently limited by historical data and assumptions. Even the most thorough evaluation cannot guarantee future performance. Always start with small position sizes when transitioning a strategy from backtesting to live trading, and continuously monitor performance to ensure it aligns with expectations.

By combining rigorous statistical analysis, comprehensive performance metrics, and thorough robustness testing, you can develop trading strategies that stand a better chance of success in live markets.
