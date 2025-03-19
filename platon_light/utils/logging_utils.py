"""
Logging and diagnostics utilities for the Platon Light trading bot
"""
import logging
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path


class LoggingManager:
    """
    Manages logging and diagnostics for the trading bot
    
    Features:
    - Configurable logging levels
    - Log rotation
    - Performance metrics tracking
    - Trade history logging
    - Error and warning tracking
    - Automated report generation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the logging manager
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.log_dir = config.get("logging", {}).get("log_dir", "logs")
        self.log_level = self._get_log_level(config.get("logging", {}).get("level", "INFO"))
        self.max_log_files = config.get("logging", {}).get("max_files", 10)
        self.log_rotation_days = config.get("logging", {}).get("rotation_days", 1)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Performance metrics
        self.performance_metrics = {
            "start_time": time.time(),
            "execution_times": [],
            "api_calls": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "response_times": []
            },
            "trades": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "execution_times": []
            },
            "errors": [],
            "warnings": []
        }
        
        # Trade history
        self.trade_history = []
        
        # Initialize logger
        self._setup_logger()
        
        # Log initialization
        self.logger.info("Logging manager initialized")
        
    def _get_log_level(self, level_str: str) -> int:
        """
        Convert string log level to logging level
        
        Args:
            level_str: String log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Logging level
        """
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        return levels.get(level_str.upper(), logging.INFO)
        
    def _setup_logger(self):
        """Set up the logger with appropriate handlers and formatters"""
        # Create logger
        self.logger = logging.getLogger("platon_light")
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # Create formatters
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        
        # Create file handler
        log_file = os.path.join(self.log_dir, f"platon_light_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Rotate logs if needed
        self._rotate_logs()
        
    def _rotate_logs(self):
        """Rotate log files to maintain the maximum number of log files"""
        log_files = sorted(
            [f for f in os.listdir(self.log_dir) if f.startswith("platon_light_") and f.endswith(".log")],
            reverse=True
        )
        
        # Delete old log files
        if len(log_files) > self.max_log_files:
            for file in log_files[self.max_log_files:]:
                try:
                    os.remove(os.path.join(self.log_dir, file))
                    self.logger.debug(f"Deleted old log file: {file}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete old log file {file}: {e}")
                    
    def log_api_call(self, endpoint: str, success: bool, response_time: float, error: Optional[str] = None):
        """
        Log an API call
        
        Args:
            endpoint: API endpoint
            success: Whether the call was successful
            response_time: Response time in seconds
            error: Error message if the call failed
        """
        # Update metrics
        self.performance_metrics["api_calls"]["total"] += 1
        
        if success:
            self.performance_metrics["api_calls"]["successful"] += 1
        else:
            self.performance_metrics["api_calls"]["failed"] += 1
            
            if error:
                self.performance_metrics["errors"].append({
                    "timestamp": time.time(),
                    "source": "api_call",
                    "endpoint": endpoint,
                    "error": error
                })
                
        self.performance_metrics["api_calls"]["response_times"].append(response_time)
        
        # Log the call
        if success:
            self.logger.debug(f"API call to {endpoint} succeeded in {response_time:.3f}s")
        else:
            self.logger.warning(f"API call to {endpoint} failed in {response_time:.3f}s: {error}")
            
    def log_trade(self, trade_data: Dict, success: bool, execution_time: float, error: Optional[str] = None):
        """
        Log a trade
        
        Args:
            trade_data: Trade data
            success: Whether the trade was successful
            execution_time: Execution time in seconds
            error: Error message if the trade failed
        """
        # Update metrics
        self.performance_metrics["trades"]["total"] += 1
        
        if success:
            self.performance_metrics["trades"]["successful"] += 1
        else:
            self.performance_metrics["trades"]["failed"] += 1
            
            if error:
                self.performance_metrics["errors"].append({
                    "timestamp": time.time(),
                    "source": "trade",
                    "trade_data": trade_data,
                    "error": error
                })
                
        self.performance_metrics["trades"]["execution_times"].append(execution_time)
        
        # Add to trade history
        trade_record = {
            "timestamp": time.time(),
            "success": success,
            "execution_time": execution_time,
            **trade_data
        }
        
        if error:
            trade_record["error"] = error
            
        self.trade_history.append(trade_record)
        
        # Log the trade
        symbol = trade_data.get("symbol", "unknown")
        side = trade_data.get("side", "unknown")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        
        if success:
            self.logger.info(f"Trade executed: {side.upper()} {quantity} {symbol} @ {price:.8f}")
        else:
            self.logger.warning(f"Trade failed: {side.upper()} {quantity} {symbol} @ {price:.8f}: {error}")
            
    def log_execution_time(self, operation: str, execution_time: float):
        """
        Log execution time for an operation
        
        Args:
            operation: Operation name
            execution_time: Execution time in seconds
        """
        self.performance_metrics["execution_times"].append({
            "timestamp": time.time(),
            "operation": operation,
            "execution_time": execution_time
        })
        
        self.logger.debug(f"Execution time for {operation}: {execution_time:.6f}s")
        
    def log_error(self, source: str, error: str, details: Optional[Dict] = None):
        """
        Log an error
        
        Args:
            source: Error source
            error: Error message
            details: Additional error details
        """
        error_record = {
            "timestamp": time.time(),
            "source": source,
            "error": error
        }
        
        if details:
            error_record["details"] = details
            
        self.performance_metrics["errors"].append(error_record)
        
        # Log the error
        if details:
            self.logger.error(f"{source}: {error} - {json.dumps(details)}")
        else:
            self.logger.error(f"{source}: {error}")
            
    def log_warning(self, source: str, warning: str, details: Optional[Dict] = None):
        """
        Log a warning
        
        Args:
            source: Warning source
            warning: Warning message
            details: Additional warning details
        """
        warning_record = {
            "timestamp": time.time(),
            "source": source,
            "warning": warning
        }
        
        if details:
            warning_record["details"] = details
            
        self.performance_metrics["warnings"].append(warning_record)
        
        # Log the warning
        if details:
            self.logger.warning(f"{source}: {warning} - {json.dumps(details)}")
        else:
            self.logger.warning(f"{source}: {warning}")
            
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate derived metrics
        uptime = time.time() - self.performance_metrics["start_time"]
        
        api_calls = self.performance_metrics["api_calls"]
        avg_response_time = sum(api_calls["response_times"]) / len(api_calls["response_times"]) if api_calls["response_times"] else 0
        
        trades = self.performance_metrics["trades"]
        avg_execution_time = sum(trades["execution_times"]) / len(trades["execution_times"]) if trades["execution_times"] else 0
        
        # Create metrics dictionary
        metrics = {
            "uptime": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "api_calls": {
                "total": api_calls["total"],
                "successful": api_calls["successful"],
                "failed": api_calls["failed"],
                "success_rate": (api_calls["successful"] / api_calls["total"]) * 100 if api_calls["total"] > 0 else 0,
                "avg_response_time": avg_response_time
            },
            "trades": {
                "total": trades["total"],
                "successful": trades["successful"],
                "failed": trades["failed"],
                "success_rate": (trades["successful"] / trades["total"]) * 100 if trades["total"] > 0 else 0,
                "avg_execution_time": avg_execution_time
            },
            "errors": len(self.performance_metrics["errors"]),
            "warnings": len(self.performance_metrics["warnings"])
        }
        
        return metrics
        
    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get trade history
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trade records
        """
        if limit:
            return self.trade_history[-limit:]
        else:
            return self.trade_history
            
    def get_errors(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get error history
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error records
        """
        errors = self.performance_metrics["errors"]
        
        if limit:
            return errors[-limit:]
        else:
            return errors
            
    def get_warnings(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get warning history
        
        Args:
            limit: Maximum number of warnings to return
            
        Returns:
            List of warning records
        """
        warnings = self.performance_metrics["warnings"]
        
        if limit:
            return warnings[-limit:]
        else:
            return warnings
            
    def generate_report(self, report_type: str = "daily") -> str:
        """
        Generate a performance report
        
        Args:
            report_type: Report type (daily, weekly, monthly)
            
        Returns:
            Path to the generated report file
        """
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(self.log_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Determine report period
        now = datetime.now()
        
        if report_type == "daily":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            report_name = f"daily_report_{now.strftime('%Y%m%d')}.json"
        elif report_type == "weekly":
            start_time = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            report_name = f"weekly_report_{start_time.strftime('%Y%m%d')}.json"
        elif report_type == "monthly":
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            report_name = f"monthly_report_{now.strftime('%Y%m')}.json"
        else:
            self.logger.warning(f"Invalid report type: {report_type}")
            return ""
            
        start_timestamp = start_time.timestamp()
        
        # Filter data for the report period
        trades = [t for t in self.trade_history if t["timestamp"] >= start_timestamp]
        errors = [e for e in self.performance_metrics["errors"] if e["timestamp"] >= start_timestamp]
        warnings = [w for w in self.performance_metrics["warnings"] if w["timestamp"] >= start_timestamp]
        
        # Calculate trade metrics
        successful_trades = [t for t in trades if t["success"]]
        failed_trades = [t for t in trades if not t["success"]]
        
        # Calculate profit/loss
        total_profit = sum([t.get("pnl", 0) for t in successful_trades if "pnl" in t])
        
        # Create report data
        report_data = {
            "report_type": report_type,
            "generated_at": time.time(),
            "period_start": start_timestamp,
            "period_end": time.time(),
            "trades": {
                "total": len(trades),
                "successful": len(successful_trades),
                "failed": len(failed_trades),
                "success_rate": (len(successful_trades) / len(trades)) * 100 if trades else 0,
                "total_profit": total_profit
            },
            "errors": {
                "total": len(errors),
                "sources": {}
            },
            "warnings": {
                "total": len(warnings),
                "sources": {}
            },
            "performance": self.get_performance_metrics()
        }
        
        # Count errors by source
        for error in errors:
            source = error.get("source", "unknown")
            if source not in report_data["errors"]["sources"]:
                report_data["errors"]["sources"][source] = 0
            report_data["errors"]["sources"][source] += 1
            
        # Count warnings by source
        for warning in warnings:
            source = warning.get("source", "unknown")
            if source not in report_data["warnings"]["sources"]:
                report_data["warnings"]["sources"][source] = 0
            report_data["warnings"]["sources"][source] += 1
            
        # Write report to file
        report_path = os.path.join(reports_dir, report_name)
        
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        self.logger.info(f"Generated {report_type} report: {report_path}")
        
        return report_path
        
    def export_trade_history(self, format: str = "csv") -> str:
        """
        Export trade history to a file
        
        Args:
            format: Export format (csv, json)
            
        Returns:
            Path to the exported file
        """
        # Create exports directory if it doesn't exist
        exports_dir = os.path.join(self.log_dir, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        
        # Generate filename
        now = datetime.now()
        filename = f"trade_history_{now.strftime('%Y%m%d_%H%M%S')}"
        
        if format.lower() == "csv":
            # Convert to DataFrame
            df = pd.DataFrame(self.trade_history)
            
            # Convert timestamps to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                
            # Export to CSV
            file_path = os.path.join(exports_dir, f"{filename}.csv")
            df.to_csv(file_path, index=False)
            
        elif format.lower() == "json":
            # Export to JSON
            file_path = os.path.join(exports_dir, f"{filename}.json")
            
            with open(file_path, "w") as f:
                json.dump(self.trade_history, f, indent=2)
                
        else:
            self.logger.warning(f"Invalid export format: {format}")
            return ""
            
        self.logger.info(f"Exported trade history to {file_path}")
        
        return file_path
        
    def benchmark_performance(self, operation: str, iterations: int = 100) -> Dict:
        """
        Benchmark performance of an operation
        
        Args:
            operation: Operation name
            iterations: Number of iterations
            
        Returns:
            Dictionary of benchmark results
        """
        # Filter execution times for the operation
        execution_times = [
            et["execution_time"] 
            for et in self.performance_metrics["execution_times"] 
            if et["operation"] == operation
        ]
        
        if not execution_times:
            return {
                "operation": operation,
                "samples": 0,
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "p95_time": 0,
                "p99_time": 0
            }
            
        # Calculate statistics
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Calculate percentiles
        sorted_times = sorted(execution_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)
        
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_time
        p99_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_time
        
        # Create benchmark results
        results = {
            "operation": operation,
            "samples": len(execution_times),
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "p95_time": p95_time,
            "p99_time": p99_time
        }
        
        self.logger.info(f"Benchmark results for {operation}: avg={avg_time:.6f}s, min={min_time:.6f}s, max={max_time:.6f}s")
        
        return results
