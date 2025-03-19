"""
Backtesting engine for testing trading strategies on historical data

This module provides a backtesting engine for testing trading strategies on historical data.
It simulates trading with historical data, calculates performance metrics, and generates reports.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path

from platon_light.backtesting.data_loader import DataLoader
from platon_light.core.strategy import ScalpingStrategy
from platon_light.utils.indicators import calculate_indicators


class BacktestEngine:
    """
    Backtesting engine for testing trading strategies on historical data
    
    Features:
    - Load historical data
    - Apply trading strategies
    - Simulate trade execution
    - Calculate performance metrics
    - Generate reports
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the backtesting engine
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize data loader
        self.data_loader = DataLoader(config)
        
        # Initialize strategy
        strategy_name = config.get("strategy", {}).get("name", "scalping")
        if strategy_name == "scalping":
            self.strategy = ScalpingStrategy(config)
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")
            
        # Initialize results storage
        self.results = {
            "trades": [],
            "equity_curve": [],
            "metrics": {}
        }
        
        # Set default parameters
        self.commission = config.get("backtesting", {}).get("commission", 0.001)
        self.slippage = config.get("backtesting", {}).get("slippage", 0.0005)
        self.initial_capital = config.get("backtesting", {}).get("initial_capital", 10000)
        
        self.logger.info("Backtesting engine initialized")
        
    def run(self, symbol: str, timeframe: str, start_date: datetime, 
           end_date: datetime) -> Dict:
        """
        Run backtest
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Backtest results
        """
        self.logger.info(f"Running backtest for {symbol} ({timeframe}) from {start_date} to {end_date}")
        
        # Load data
        data = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            self.logger.error("No data available for backtest")
            return {"error": "No data available"}
            
        # Prepare data (add indicators)
        data = self.data_loader.prepare_data(data)
        
        # Run simulation
        results = self._simulate_trading(data, symbol, timeframe)
        
        # Calculate metrics
        self._calculate_metrics(results)
        
        self.logger.info(f"Backtest completed: {len(results['trades'])} trades")
        
        return results
        
    def _simulate_trading(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Simulate trading on historical data
        
        Args:
            data: DataFrame with OHLCV data and indicators
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            Simulation results
        """
        # Initialize results
        trades = []
        equity_curve = []
        
        # Initialize trading variables
        capital = self.initial_capital
        position = None
        position_size = 0
        entry_price = 0
        entry_time = None
        
        # Get risk parameters
        risk_params = self.config.get("risk_management", {})
        max_position_size = risk_params.get("max_position_size", 0.01)
        
        # Get strategy parameters
        strategy_params = self.config.get("strategy", {})
        profit_target = strategy_params.get("profit_target", 0.005)
        stop_loss = strategy_params.get("stop_loss", 0.003)
        
        # Simulate trading
        for i, row in data.iterrows():
            timestamp = row['timestamp']
            current_time = datetime.fromtimestamp(timestamp / 1000)
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Calculate position size based on capital and max position size
            position_value = capital * max_position_size
            potential_position_size = position_value / close_price
            
            # Update equity curve
            if position:
                # Calculate unrealized PnL
                if position == "long":
                    unrealized_pnl = (close_price - entry_price) * position_size
                else:  # short
                    unrealized_pnl = (entry_price - close_price) * position_size
                    
                # Apply commission
                unrealized_pnl -= (position_size * entry_price * self.commission)
                
                equity = capital + unrealized_pnl
            else:
                equity = capital
                
            equity_curve.append({
                "timestamp": timestamp,
                "datetime": current_time,
                "equity": equity,
                "position": position
            })
            
            # Check for exit conditions if in a position
            if position:
                # Calculate price movement
                if position == "long":
                    price_movement = (high_price - entry_price) / entry_price
                    stop_price = entry_price * (1 - stop_loss)
                    target_price = entry_price * (1 + profit_target)
                    
                    # Check if stop loss hit
                    if low_price <= stop_price:
                        # Exit at stop loss
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                        
                        # Calculate PnL
                        pnl = (exit_price - entry_price) * position_size
                        pnl -= (position_size * entry_price * self.commission)
                        pnl -= (position_size * exit_price * self.commission)
                        
                        # Record trade
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "position": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "position_size": position_size,
                            "pnl": pnl,
                            "pnl_percent": pnl / capital * 100,
                            "exit_reason": exit_reason
                        })
                        
                        # Update capital
                        capital += pnl
                        
                        # Reset position
                        position = None
                        position_size = 0
                        entry_price = 0
                        entry_time = None
                        
                    # Check if take profit hit
                    elif high_price >= target_price:
                        # Exit at take profit
                        exit_price = target_price
                        exit_reason = "take_profit"
                        
                        # Calculate PnL
                        pnl = (exit_price - entry_price) * position_size
                        pnl -= (position_size * entry_price * self.commission)
                        pnl -= (position_size * exit_price * self.commission)
                        
                        # Record trade
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "position": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "position_size": position_size,
                            "pnl": pnl,
                            "pnl_percent": pnl / capital * 100,
                            "exit_reason": exit_reason
                        })
                        
                        # Update capital
                        capital += pnl
                        
                        # Reset position
                        position = None
                        position_size = 0
                        entry_price = 0
                        entry_time = None
                else:  # short
                    price_movement = (entry_price - low_price) / entry_price
                    stop_price = entry_price * (1 + stop_loss)
                    target_price = entry_price * (1 - profit_target)
                    
                    # Check if stop loss hit
                    if high_price >= stop_price:
                        # Exit at stop loss
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                        
                        # Calculate PnL
                        pnl = (entry_price - exit_price) * position_size
                        pnl -= (position_size * entry_price * self.commission)
                        pnl -= (position_size * exit_price * self.commission)
                        
                        # Record trade
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "position": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "position_size": position_size,
                            "pnl": pnl,
                            "pnl_percent": pnl / capital * 100,
                            "exit_reason": exit_reason
                        })
                        
                        # Update capital
                        capital += pnl
                        
                        # Reset position
                        position = None
                        position_size = 0
                        entry_price = 0
                        entry_time = None
                        
                    # Check if take profit hit
                    elif low_price <= target_price:
                        # Exit at take profit
                        exit_price = target_price
                        exit_reason = "take_profit"
                        
                        # Calculate PnL
                        pnl = (entry_price - exit_price) * position_size
                        pnl -= (position_size * entry_price * self.commission)
                        pnl -= (position_size * exit_price * self.commission)
                        
                        # Record trade
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "position": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "position_size": position_size,
                            "pnl": pnl,
                            "pnl_percent": pnl / capital * 100,
                            "exit_reason": exit_reason
                        })
                        
                        # Update capital
                        capital += pnl
                        
                        # Reset position
                        position = None
                        position_size = 0
                        entry_price = 0
                        entry_time = None
            
            # Check for entry signals if not in a position
            if not position:
                # Get signals from strategy
                signal = self.strategy.generate_signal(row)
                
                if signal == "buy":
                    # Enter long position
                    position = "long"
                    entry_price = close_price
                    position_size = potential_position_size
                    entry_time = current_time
                    
                    self.logger.debug(f"Entered long position at {entry_price}")
                    
                elif signal == "sell":
                    # Enter short position
                    position = "short"
                    entry_price = close_price
                    position_size = potential_position_size
                    entry_time = current_time
                    
                    self.logger.debug(f"Entered short position at {entry_price}")
        
        # Close any open position at the end of the backtest
        if position:
            # Get last price
            last_row = data.iloc[-1]
            last_price = last_row['close']
            last_time = datetime.fromtimestamp(last_row['timestamp'] / 1000)
            
            # Calculate PnL
            if position == "long":
                pnl = (last_price - entry_price) * position_size
            else:  # short
                pnl = (entry_price - last_price) * position_size
                
            pnl -= (position_size * entry_price * self.commission)
            pnl -= (position_size * last_price * self.commission)
            
            # Record trade
            trades.append({
                "entry_time": entry_time,
                "exit_time": last_time,
                "symbol": symbol,
                "timeframe": timeframe,
                "position": position,
                "entry_price": entry_price,
                "exit_price": last_price,
                "position_size": position_size,
                "pnl": pnl,
                "pnl_percent": pnl / capital * 100,
                "exit_reason": "end_of_backtest"
            })
            
            # Update capital
            capital += pnl
            
        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "final_capital": capital,
            "profit_loss": capital - self.initial_capital,
            "profit_loss_percent": (capital - self.initial_capital) / self.initial_capital * 100
        }
        
    def _calculate_metrics(self, results: Dict):
        """
        Calculate performance metrics
        
        Args:
            results: Simulation results
        """
        trades = results["trades"]
        equity_curve = results["equity_curve"]
        
        if not trades:
            self.logger.warning("No trades to calculate metrics")
            results["metrics"] = {}
            return
            
        # Convert to DataFrame for easier calculations
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # PnL metrics
        total_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        total_loss = trades_df[trades_df["pnl"] <= 0]["pnl"].sum()
        net_profit = total_profit + total_loss
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Average trade metrics
        avg_profit = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if losing_trades > 0 else 0
        avg_trade = trades_df["pnl"].mean()
        
        # Risk metrics
        if equity_df.empty:
            max_drawdown = 0
            max_drawdown_percent = 0
        else:
            # Calculate drawdown
            equity_df["peak"] = equity_df["equity"].cummax()
            equity_df["drawdown"] = equity_df["peak"] - equity_df["equity"]
            equity_df["drawdown_percent"] = equity_df["drawdown"] / equity_df["peak"] * 100
            
            max_drawdown = equity_df["drawdown"].max()
            max_drawdown_percent = equity_df["drawdown_percent"].max()
        
        # Calculate Sharpe ratio
        if len(equity_df) > 1:
            equity_df["return"] = equity_df["equity"].pct_change()
            sharpe_ratio = equity_df["return"].mean() / equity_df["return"].std() * np.sqrt(252) if equity_df["return"].std() != 0 else 0
        else:
            sharpe_ratio = 0
            
        # Store metrics
        results["metrics"] = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "max_drawdown": max_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "sharpe_ratio": sharpe_ratio,
            "initial_capital": self.initial_capital,
            "final_capital": results["final_capital"],
            "return_percent": results["profit_loss_percent"]
        }
        
    def generate_report(self, results: Dict, output_dir: Optional[str] = None) -> str:
        """
        Generate backtest report
        
        Args:
            results: Backtest results
            output_dir: Output directory for report files
            
        Returns:
            Report summary
        """
        try:
            # Create output directory if needed
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path("backtest_reports")
                output_path.mkdir(parents=True, exist_ok=True)
                
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract data
            trades = results.get("trades", [])
            equity_curve = results.get("equity_curve", [])
            metrics = results.get("metrics", {})
            
            if not trades:
                return "No trades to generate report"
                
            # Convert to DataFrames
            trades_df = pd.DataFrame(trades)
            equity_df = pd.DataFrame(equity_curve)
            
            # Save trades to CSV
            trades_file = output_path / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            
            # Save equity curve to CSV
            equity_file = output_path / f"equity_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=False)
            
            # Generate HTML report
            report_file = output_path / f"report_{timestamp}.html"
            
            # Generate report content
            report_content = self._generate_html_report(results, timestamp)
            
            # Write report to file
            with open(report_file, "w") as f:
                f.write(report_content)
                
            self.logger.info(f"Backtest report generated: {report_file}")
            
            # Return summary
            summary = f"Backtest Report Summary:\n"
            summary += f"Total Trades: {metrics.get('total_trades', 0)}\n"
            summary += f"Win Rate: {metrics.get('win_rate', 0):.2f}%\n"
            summary += f"Net Profit: {metrics.get('net_profit', 0):.2f}\n"
            summary += f"Return: {metrics.get('return_percent', 0):.2f}%\n"
            summary += f"Max Drawdown: {metrics.get('max_drawdown_percent', 0):.2f}%\n"
            summary += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            summary += f"Report saved to: {report_file}"
            
            return summary
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return f"Failed to generate report: {e}"
            
    def _generate_html_report(self, results: Dict, timestamp: str) -> str:
        """
        Generate HTML report content
        
        Args:
            results: Backtest results
            timestamp: Report timestamp
            
        Returns:
            HTML report content
        """
        # Extract data
        trades = results.get("trades", [])
        metrics = results.get("metrics", {})
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Initial Capital</td>
                        <td>{metrics.get('initial_capital', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Final Capital</td>
                        <td>{metrics.get('final_capital', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Net Profit</td>
                        <td class="{'positive' if metrics.get('net_profit', 0) > 0 else 'negative'}">{metrics.get('net_profit', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Return</td>
                        <td class="{'positive' if metrics.get('return_percent', 0) > 0 else 'negative'}">{metrics.get('return_percent', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Total Trades</td>
                        <td>{metrics.get('total_trades', 0)}</td>
                    </tr>
                    <tr>
                        <td>Winning Trades</td>
                        <td>{metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0):.2f}%)</td>
                    </tr>
                    <tr>
                        <td>Losing Trades</td>
                        <td>{metrics.get('losing_trades', 0)} ({100 - metrics.get('win_rate', 0):.2f}%)</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{metrics.get('profit_factor', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Profit</td>
                        <td class="positive">{metrics.get('avg_profit', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td class="negative">{metrics.get('avg_loss', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Trade</td>
                        <td class="{'positive' if metrics.get('avg_trade', 0) > 0 else 'negative'}">{metrics.get('avg_trade', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td class="negative">{metrics.get('max_drawdown', 0):.2f} ({metrics.get('max_drawdown_percent', 0):.2f}%)</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                    </tr>
                </table>
            </div>
            
            <h2>Trade List</h2>
            <table>
                <tr>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Symbol</th>
                    <th>Position</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Size</th>
                    <th>PnL</th>
                    <th>PnL %</th>
                    <th>Exit Reason</th>
                </tr>
        """
        
        # Add trade rows
        for trade in trades:
            pnl = trade.get("pnl", 0)
            pnl_class = "positive" if pnl > 0 else "negative"
            
            html += f"""
                <tr>
                    <td>{trade.get("entry_time").strftime("%Y-%m-%d %H:%M:%S") if trade.get("entry_time") else ""}</td>
                    <td>{trade.get("exit_time").strftime("%Y-%m-%d %H:%M:%S") if trade.get("exit_time") else ""}</td>
                    <td>{trade.get("symbol", "")}</td>
                    <td>{trade.get("position", "")}</td>
                    <td>{trade.get("entry_price", 0):.6f}</td>
                    <td>{trade.get("exit_price", 0):.6f}</td>
                    <td>{trade.get("position_size", 0):.6f}</td>
                    <td class="{pnl_class}">{pnl:.2f}</td>
                    <td class="{pnl_class}">{trade.get("pnl_percent", 0):.2f}%</td>
                    <td>{trade.get("exit_reason", "")}</td>
                </tr>
            """
            
        # Close HTML
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
