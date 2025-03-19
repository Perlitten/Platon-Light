# Platon Light Backtesting Performance Metrics Reference

This document provides a comprehensive reference for all performance metrics calculated by the Platon Light backtesting module. Understanding these metrics is crucial for properly evaluating trading strategy performance.

## Table of Contents

1. [Return Metrics](#return-metrics)
2. [Risk Metrics](#risk-metrics)
3. [Risk-Adjusted Return Metrics](#risk-adjusted-return-metrics)
4. [Trade Metrics](#trade-metrics)
5. [Drawdown Metrics](#drawdown-metrics)
6. [Consistency Metrics](#consistency-metrics)
7. [Custom Metrics](#custom-metrics)
8. [Interpreting Metrics](#interpreting-metrics)
9. [Comparing Strategies](#comparing-strategies)
10. [References](#references)

## Return Metrics

### Total Return

**Definition:** The total percentage return of the strategy over the entire backtest period.

**Formula:**
```
Total Return (%) = ((Final Equity - Initial Capital) / Initial Capital) * 100
```

**Interpretation:**
- Higher is better
- Benchmark against market returns during the same period
- Consider alongside risk metrics

**Example:**
```python
initial_capital = 10000
final_equity = 12500
total_return = ((final_equity - initial_capital) / initial_capital) * 100
# Result: 25.0%
```

### Annualized Return

**Definition:** The return normalized to a one-year period.

**Formula:**
```
Annualized Return (%) = ((1 + Total Return / 100) ^ (365 / Days)) - 1) * 100
```
Where `Days` is the number of days in the backtest period.

**Interpretation:**
- Allows comparison of strategies tested over different time periods
- Higher is better, but consider risk

**Example:**
```python
total_return = 25.0  # 25%
days = 180  # 6 months
annualized_return = (((1 + total_return / 100) ** (365 / days)) - 1) * 100
# Result: 56.2%
```

### Daily/Monthly/Yearly Returns

**Definition:** Returns broken down by time period.

**Formula:**
```
Period Return (%) = ((End Equity - Start Equity) / Start Equity) * 100
```

**Interpretation:**
- Useful for analyzing performance consistency
- Look for patterns or seasonality

**Example:**
```python
# Monthly returns
monthly_returns = {
    "Jan 2023": 2.5,
    "Feb 2023": -1.2,
    "Mar 2023": 3.7,
    # ...
}
```

### Compound Annual Growth Rate (CAGR)

**Definition:** The mean annual growth rate of an investment over a specified time period longer than one year.

**Formula:**
```
CAGR (%) = ((Final Equity / Initial Capital) ^ (1 / Years)) - 1) * 100
```
Where `Years` is the number of years in the backtest period.

**Interpretation:**
- Smooths returns over time
- Better than simple average for long-term performance

**Example:**
```python
initial_capital = 10000
final_equity = 14000
years = 2.5
cagr = (((final_equity / initial_capital) ** (1 / years)) - 1) * 100
# Result: 13.8%
```

## Risk Metrics

### Volatility (Standard Deviation)

**Definition:** The standard deviation of returns, measuring the dispersion of returns around the mean.

**Formula:**
```
Volatility (%) = Standard Deviation of Period Returns * √(Trading Periods Per Year)
```

**Interpretation:**
- Lower is better (less risk)
- Typical values: 10-20% for balanced strategies
- High volatility (>30%) indicates high risk

**Example:**
```python
daily_returns = [0.1, -0.2, 0.3, -0.1, 0.2]  # in percent
std_dev = np.std(daily_returns)
annualized_volatility = std_dev * np.sqrt(252)  # 252 trading days per year
# Result: If std_dev = 0.2, then annualized_volatility = 3.17%
```

### Maximum Drawdown

**Definition:** The maximum observed loss from a peak to a trough of the equity curve, before a new peak is attained.

**Formula:**
```
Maximum Drawdown (%) = ((Trough Value - Peak Value) / Peak Value) * 100
```

**Interpretation:**
- Lower is better (less risk)
- Typical values: 10-30% for balanced strategies
- >50% indicates high risk

**Example:**
```python
equity_curve = [10000, 10500, 10200, 9800, 10100, 10700, 10300]
max_drawdown_pct = calculate_max_drawdown(equity_curve)
# Result: -6.7% (from 10500 to 9800)
```

### Downside Deviation

**Definition:** Similar to standard deviation, but only considers returns below a specified threshold (usually 0 or the risk-free rate).

**Formula:**
```
Downside Deviation = √(Sum of Squared Negative Returns / Number of Periods)
```

**Interpretation:**
- Lower is better
- More relevant than standard deviation for most investors
- Focuses on harmful volatility (downside risk)

**Example:**
```python
daily_returns = [0.1, -0.2, 0.3, -0.1, 0.2]  # in percent
negative_returns = [r for r in daily_returns if r < 0]
downside_deviation = np.sqrt(sum([r**2 for r in negative_returns]) / len(daily_returns))
# Result: 0.1
```

### Value at Risk (VaR)

**Definition:** The maximum loss expected over a given time period at a given confidence level.

**Formula:**
```
VaR = Mean Return - (Z-score * Standard Deviation)
```
Where Z-score is 1.645 for 95% confidence, 2.33 for 99% confidence.

**Interpretation:**
- Lower is better
- Represents the worst-case scenario with a given probability
- 95% VaR of -2% means there's a 95% chance the daily loss won't exceed 2%

**Example:**
```python
mean_return = 0.05  # daily mean return in percent
std_dev = 0.2  # daily standard deviation in percent
var_95 = mean_return - (1.645 * std_dev)
# Result: -0.28% (95% VaR)
```

## Risk-Adjusted Return Metrics

### Sharpe Ratio

**Definition:** The average return earned in excess of the risk-free rate per unit of volatility.

**Formula:**
```
Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
```

**Interpretation:**
- Higher is better
- <1: Poor
- 1-2: Acceptable
- 2-3: Very good
- >3: Excellent

**Example:**
```python
annualized_return = 15.0  # in percent
risk_free_rate = 2.0  # in percent
annualized_volatility = 10.0  # in percent
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
# Result: 1.3
```

### Sortino Ratio

**Definition:** Similar to the Sharpe ratio, but uses downside deviation instead of standard deviation.

**Formula:**
```
Sortino Ratio = (Annualized Return - Risk-Free Rate) / Annualized Downside Deviation
```

**Interpretation:**
- Higher is better
- Better than Sharpe for asymmetric return distributions
- Same general thresholds as Sharpe, but values are typically higher

**Example:**
```python
annualized_return = 15.0  # in percent
risk_free_rate = 2.0  # in percent
annualized_downside_deviation = 6.0  # in percent
sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_deviation
# Result: 2.17
```

### Calmar Ratio

**Definition:** The ratio of annualized return to maximum drawdown.

**Formula:**
```
Calmar Ratio = Annualized Return / |Maximum Drawdown|
```

**Interpretation:**
- Higher is better
- <1: Poor risk-adjusted return
- 1-3: Good
- >3: Excellent
- Focuses on worst-case scenario

**Example:**
```python
annualized_return = 15.0  # in percent
max_drawdown = -20.0  # in percent
calmar_ratio = annualized_return / abs(max_drawdown)
# Result: 0.75
```

### Omega Ratio

**Definition:** The probability-weighted ratio of gains versus losses for a threshold return.

**Formula:**
```
Omega Ratio = (Sum of Returns Above Threshold) / |Sum of Returns Below Threshold|
```

**Interpretation:**
- Higher is better
- >1: More gain than loss potential
- <1: More loss than gain potential
- More comprehensive than Sharpe/Sortino

**Example:**
```python
returns = [0.1, -0.2, 0.3, -0.1, 0.2]  # in percent
threshold = 0  # threshold return
returns_above = [r for r in returns if r > threshold]
returns_below = [r for r in returns if r <= threshold]
omega_ratio = sum(returns_above) / abs(sum(returns_below))
# Result: 2.0
```

## Trade Metrics

### Win Rate

**Definition:** The percentage of trades that are profitable.

**Formula:**
```
Win Rate (%) = (Number of Winning Trades / Total Number of Trades) * 100
```

**Interpretation:**
- Higher is better, but not in isolation
- 50-60% is typical for many strategies
- Consider alongside average win/loss ratio

**Example:**
```python
winning_trades = 60
total_trades = 100
win_rate = (winning_trades / total_trades) * 100
# Result: 60%
```

### Profit Factor

**Definition:** The ratio of gross profit to gross loss.

**Formula:**
```
Profit Factor = Gross Profit / |Gross Loss|
```

**Interpretation:**
- Higher is better
- <1: Losing strategy
- 1-1.5: Marginally profitable
- 1.5-2: Good
- >2: Excellent

**Example:**
```python
gross_profit = 5000  # sum of all profitable trades
gross_loss = -3000  # sum of all losing trades
profit_factor = gross_profit / abs(gross_loss)
# Result: 1.67
```

### Average Trade

**Definition:** The average profit or loss per trade.

**Formula:**
```
Average Trade = Net Profit / Total Number of Trades
```

**Interpretation:**
- Higher is better
- Should be positive
- Consider alongside win rate and number of trades

**Example:**
```python
net_profit = 2000
total_trades = 100
average_trade = net_profit / total_trades
# Result: $20 per trade
```

### Average Win/Loss Ratio

**Definition:** The ratio of the average winning trade to the average losing trade.

**Formula:**
```
Average Win/Loss Ratio = Average Winning Trade / |Average Losing Trade|
```

**Interpretation:**
- Higher is better
- >1: Winners are larger than losers
- <1: Losers are larger than winners
- Ideally >2 for strategies with win rates <50%

**Example:**
```python
avg_winning_trade = 100
avg_losing_trade = -50
avg_win_loss_ratio = avg_winning_trade / abs(avg_losing_trade)
# Result: 2.0
```

### Expectancy

**Definition:** The expected return per dollar risked on each trade.

**Formula:**
```
Expectancy = (Win Rate * Average Win) - ((1 - Win Rate) * Average Loss)
```

**Interpretation:**
- Higher is better
- >0: Profitable in the long run
- <0: Losing in the long run
- Comprehensive metric combining win rate and win/loss ratio

**Example:**
```python
win_rate = 0.6  # 60%
avg_win = 100
avg_loss = 50
expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
# Result: $40 per trade
```

### Average Holding Period

**Definition:** The average time a position is held.

**Formula:**
```
Average Holding Period = Sum of All Trade Durations / Total Number of Trades
```

**Interpretation:**
- Neither higher nor lower is inherently better
- Shorter for scalping strategies (minutes to hours)
- Longer for trend-following strategies (days to weeks)
- Should align with strategy design

**Example:**
```python
# For trades with durations in hours
trade_durations = [2, 5, 1, 3, 4]  # hours
avg_holding_period = sum(trade_durations) / len(trade_durations)
# Result: 3 hours
```

## Drawdown Metrics

### Average Drawdown

**Definition:** The average of all drawdowns during the backtest period.

**Formula:**
```
Average Drawdown = Sum of All Drawdowns / Number of Drawdowns
```

**Interpretation:**
- Lower is better
- Provides a sense of typical drawdowns
- Consider alongside maximum drawdown

**Example:**
```python
drawdowns = [-5, -10, -3, -7, -2]  # in percent
avg_drawdown = sum(drawdowns) / len(drawdowns)
# Result: -5.4%
```

### Drawdown Duration

**Definition:** The time it takes to recover from a drawdown.

**Formula:**
```
Drawdown Duration = Recovery Date - Drawdown Start Date
```

**Interpretation:**
- Shorter is better
- Long durations can be psychologically challenging
- Consider maximum and average durations

**Example:**
```python
# For a drawdown that started on day 10 and recovered on day 25
drawdown_duration = 25 - 10
# Result: 15 days
```

### Ulcer Index

**Definition:** A measure of downside risk that considers both the depth and duration of drawdowns.

**Formula:**
```
Ulcer Index = √(Sum of Squared Drawdowns / Number of Periods)
```

**Interpretation:**
- Lower is better
- More comprehensive than maximum drawdown
- Penalizes deep and prolonged drawdowns

**Example:**
```python
# For a series of drawdowns at each period
drawdowns = [0, -2, -5, -3, -1, 0]  # in percent
squared_drawdowns = [d**2 for d in drawdowns]
ulcer_index = np.sqrt(sum(squared_drawdowns) / len(drawdowns))
# Result: 2.16
```

## Consistency Metrics

### Monthly/Yearly Win Rate

**Definition:** The percentage of months or years with positive returns.

**Formula:**
```
Monthly Win Rate (%) = (Number of Positive Months / Total Number of Months) * 100
```

**Interpretation:**
- Higher is better
- >50% indicates consistency
- >70% is excellent
- Look for patterns in losing months

**Example:**
```python
monthly_returns = [1.2, -0.5, 0.8, 1.5, -0.3, 0.9]  # in percent
positive_months = sum(1 for r in monthly_returns if r > 0)
monthly_win_rate = (positive_months / len(monthly_returns)) * 100
# Result: 66.7%
```

### Return Consistency

**Definition:** The standard deviation of period returns, indicating how consistent returns are.

**Formula:**
```
Return Consistency = Standard Deviation of Period Returns
```

**Interpretation:**
- Lower is better
- Indicates predictability of returns
- Consider alongside average return

**Example:**
```python
monthly_returns = [1.2, -0.5, 0.8, 1.5, -0.3, 0.9]  # in percent
return_consistency = np.std(monthly_returns)
# Result: 0.76
```

### Longest Winning/Losing Streak

**Definition:** The maximum number of consecutive winning or losing trades.

**Formula:**
```
# Calculated by counting consecutive wins or losses
```

**Interpretation:**
- Longer winning streaks are better
- Shorter losing streaks are better
- Long losing streaks can be psychologically challenging

**Example:**
```python
trade_results = [1, 1, -1, -1, -1, 1, 1, 1, 1, -1]  # 1 for win, -1 for loss
longest_winning_streak = 4
longest_losing_streak = 3
```

## Custom Metrics

### Strategy-Specific Metrics

Depending on the strategy type, additional custom metrics may be relevant:

#### For Mean Reversion Strategies:
- **Reversion Efficiency**: How efficiently the strategy captures mean reversion moves
- **Overbought/Oversold Accuracy**: Success rate of trades taken at extreme levels

#### For Trend Following Strategies:
- **Trend Capture Ratio**: Percentage of the trend captured by the strategy
- **Trend Detection Lag**: Average time to detect and enter a trend

#### For Volatility-Based Strategies:
- **Volatility Capture**: How well the strategy profits from volatility
- **Volatility Prediction Accuracy**: Accuracy of volatility forecasts

### Risk Management Metrics

- **Risk-Reward Ratio**: Average risk taken versus reward gained per trade
- **Maximum Consecutive Risk**: Highest cumulative risk during a sequence of trades
- **Risk-Adjusted Win Rate**: Win rate weighted by risk taken on each trade

## Interpreting Metrics

### Key Metric Combinations

Single metrics in isolation can be misleading. Consider these combinations:

1. **Return + Risk**: Always evaluate returns in the context of risk taken
   - Total Return + Maximum Drawdown
   - CAGR + Volatility
   - Sharpe/Sortino Ratio (combines both)

2. **Win Rate + Average Win/Loss Ratio**: These metrics complement each other
   - High win rate can offset small average win/loss ratio
   - High average win/loss ratio can offset low win rate
   - Expectancy combines both

3. **Consistency + Magnitude**: Consider both how often you win and how much
   - Monthly Win Rate + Average Monthly Return
   - Longest Losing Streak + Maximum Drawdown

### Red Flags

Watch for these warning signs in your metrics:

1. **Too Good To Be True**
   - Sharpe ratios >5
   - Win rates >90%
   - No significant drawdowns
   - These often indicate look-ahead bias or overfitting

2. **Inconsistent Performance**
   - Large discrepancies between in-sample and out-of-sample performance
   - Extreme sensitivity to parameter changes
   - Performance concentrated in a few lucky trades

3. **Poor Risk Management**
   - Maximum drawdown >50% of returns
   - Average losing trade much larger than average winning trade
   - Long recovery periods from drawdowns

### Benchmark Comparison

Always compare your strategy metrics to appropriate benchmarks:

1. **Market Benchmarks**
   - S&P 500 for US equities
   - Relevant crypto index for crypto strategies
   - Risk-free rate (e.g., Treasury yields)

2. **Strategy-Type Benchmarks**
   - Typical Sharpe ratios for your strategy type
   - Typical drawdowns for your strategy type
   - Typical win rates for your strategy type

3. **Improvement Benchmarks**
   - Your strategy's previous versions
   - Simple baseline strategies (e.g., buy and hold, moving average crossover)

## Comparing Strategies

### Ranking Methodologies

Different ways to rank multiple strategies:

1. **Risk-Adjusted Return Ranking**
   - Rank by Sharpe or Sortino ratio
   - Balances return and risk
   - Most common approach

2. **Drawdown-Adjusted Ranking**
   - Rank by Calmar ratio
   - Emphasizes downside protection
   - Good for risk-averse investors

3. **Consistency Ranking**
   - Rank by monthly win rate or return consistency
   - Emphasizes predictability
   - Good for steady income strategies

4. **Composite Ranking**
   - Create a weighted score of multiple metrics
   - Customize weights based on priorities
   - Most comprehensive approach

### Visualization Techniques

Effective ways to compare strategies visually:

1. **Equity Curves**
   - Plot equity curves on the same chart
   - Visualize performance differences over time
   - Identify periods of outperformance/underperformance

2. **Drawdown Comparison**
   - Plot drawdown charts side by side
   - Compare depth and duration of drawdowns
   - Identify which strategies recover faster

3. **Return Distribution**
   - Plot histograms of returns for each strategy
   - Compare distribution shapes
   - Identify outliers and skewness

4. **Risk-Return Scatter Plot**
   - Plot return (y-axis) versus risk (x-axis)
   - Strategies in the upper-left quadrant are superior
   - Visualize the efficient frontier

5. **Monthly/Yearly Heatmap**
   - Color-code returns by period
   - Identify seasonal patterns
   - Compare consistency across time

## References

1. Bacon, C. R. (2008). Practical Portfolio Performance Measurement and Attribution. Wiley.
2. Chan, E. P. (2013). Algorithmic Trading: Winning Strategies and Their Rationale. Wiley.
3. Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies. Wiley.
4. Tomasini, E., & Jaekle, U. (2009). Trading Systems: A New Approach to System Development and Portfolio Optimisation. Harriman House.
5. Zakamulin, V. (2017). Market Timing with Moving Averages: The Anatomy and Performance of Trading Rules. Palgrave Macmillan.
