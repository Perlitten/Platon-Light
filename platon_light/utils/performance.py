"""
Performance tracking and reporting for the Platon Light trading bot
"""
import logging
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd


class PerformanceTracker:
    """
    Manages performance metrics, trade history, and reporting for the trading bot.
    """

    def __init__(self, config: Dict):
        """
        Initialize the performance tracker.

        Args:
            config: Bot configuration dictionary.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.log_dir = config.get("logging", {}).get("log_dir", "logs")

        # Performance metrics
        self.performance_metrics = {
            "start_time": time.time(),
            "execution_times": [],
            "api_calls": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "response_times": [],
            },
            "trades": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "execution_times": [],
            },
            "errors": [],
            "warnings": [],
        }

        # Trade history
        self.trade_history = []
        self.logger.info("Performance tracker initialized")

    def record_trade(self, trade_result: Optional[Dict]):
        """Records a trade and its outcome."""
        if not trade_result:
            return

        # This is a simplified version of the original log_trade
        self.performance_metrics["trades"]["total"] += 1

        # Assuming PnL calculation happens elsewhere and is part of trade_result
        if trade_result.get("pnl", 0) >= 0:
             self.performance_metrics["trades"]["successful"] += 1
        else:
             self.performance_metrics["trades"]["failed"] += 1

        self.trade_history.append(trade_result)
        self.logger.debug(f"Recorded trade: {trade_result.get('symbol')}")

    def get_summary(self) -> Dict:
        """
        Get a summary of performance metrics.
        """
        # Calculate derived metrics
        uptime = time.time() - self.performance_metrics["start_time"]

        api_calls = self.performance_metrics["api_calls"]
        avg_response_time = sum(api_calls["response_times"]) / len(api_calls["response_times"]) if api_calls["response_times"] else 0

        trades = self.performance_metrics["trades"]
        avg_execution_time = sum(trades["execution_times"]) / len(trades["execution_times"]) if trades["execution_times"] else 0

        win_rate = (trades["successful"] / trades["total"]) * 100 if trades["total"] > 0 else 0

        # Create metrics dictionary
        metrics = {
            "uptime": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "api_calls": {
                "total": api_calls["total"],
                "successful": api_calls["successful"],
                "failed": api_calls["failed"],
                "success_rate": (api_calls["successful"] / api_calls["total"]) * 100 if api_calls["total"] > 0 else 0,
                "avg_response_time": avg_response_time,
            },
            "trades": {
                "total": trades["total"],
                "successful": trades["successful"],
                "failed": trades["failed"],
                "win_rate": win_rate,
                "avg_execution_time": avg_execution_time,
            },
            "errors": len(self.performance_metrics["errors"]),
            "warnings": len(self.performance_metrics["warnings"]),
        }

        return metrics

    def generate_summary_report(self):
        """Generates and logs a final summary report."""
        summary = self.get_summary()
        self.logger.info("--- Trading Session Summary ---")
        self.logger.info(f"Duration: {summary['uptime_formatted']}")
        self.logger.info(f"Total Trades: {summary['trades']['total']}")
        self.logger.info(f"Win Rate: {summary['trades']['win_rate']:.2f}%")
        self.logger.info("--- End of Summary ---")

    def log_error(self, source: str, error: str, details: Optional[Dict] = None):
        """Logs an error for tracking."""
        error_record = {
            "timestamp": time.time(),
            "source": source,
            "error": error,
            "details": details or {},
        }
        self.performance_metrics["errors"].append(error_record)
        self.logger.error(f"Logged error from {source}: {error}")

    def log_warning(self, source: str, warning: str, details: Optional[Dict] = None):
        """Logs a warning for tracking."""
        warning_record = {
            "timestamp": time.time(),
            "source": source,
            "warning": warning,
            "details": details or {},
        }
        self.performance_metrics["warnings"].append(warning_record)
        self.logger.warning(f"Logged warning from {source}: {warning}")
