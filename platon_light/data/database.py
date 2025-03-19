"""
Database module for storing trade history and performance data
"""
import logging
import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class DatabaseManager:
    """
    Database manager for storing trade history and performance data
    
    Features:
    - Store trade history
    - Store performance metrics
    - Store bot configuration
    - Query historical data
    - Export data to various formats
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the database manager
        
        Args:
            config: Bot configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Get database path
        db_dir = config.get("database", {}).get("dir", "data")
        db_name = config.get("database", {}).get("name", "platon_light.db")
        self.db_path = Path(db_dir) / db_name
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info("Database manager initialized")
        
    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                order_id TEXT,
                trade_id TEXT,
                fee REAL,
                fee_asset TEXT,
                pnl REAL,
                strategy TEXT,
                timeframe TEXT,
                execution_time REAL,
                status TEXT,
                error TEXT,
                notes TEXT,
                extra JSON
            )
            ''')
            
            # Create performance table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                symbol TEXT,
                timeframe TEXT,
                notes TEXT
            )
            ''')
            
            # Create config table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                config JSON NOT NULL,
                notes TEXT
            )
            ''')
            
            # Create errors table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                source TEXT NOT NULL,
                error TEXT NOT NULL,
                details JSON,
                resolved INTEGER DEFAULT 0
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.debug("Database tables initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            
    def store_trade(self, trade_data: Dict) -> int:
        """
        Store a trade in the database
        
        Args:
            trade_data: Trade data
            
        Returns:
            Trade ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract trade data
            timestamp = trade_data.get("timestamp", int(time.time()))
            symbol = trade_data.get("symbol", "")
            side = trade_data.get("side", "")
            quantity = trade_data.get("quantity", 0)
            price = trade_data.get("price", 0)
            order_id = trade_data.get("order_id", "")
            trade_id = trade_data.get("trade_id", "")
            fee = trade_data.get("fee", 0)
            fee_asset = trade_data.get("fee_asset", "")
            pnl = trade_data.get("pnl", 0)
            strategy = trade_data.get("strategy", "")
            timeframe = trade_data.get("timeframe", "")
            execution_time = trade_data.get("execution_time", 0)
            status = trade_data.get("status", "")
            error = trade_data.get("error", "")
            notes = trade_data.get("notes", "")
            
            # Extract extra data
            extra_keys = ["entry_price", "exit_price", "stop_loss", "take_profit", "position_id", "leverage", "margin_type"]
            extra = {k: trade_data[k] for k in extra_keys if k in trade_data}
            
            # Insert trade
            cursor.execute('''
            INSERT INTO trades (
                timestamp, symbol, side, quantity, price, order_id, trade_id,
                fee, fee_asset, pnl, strategy, timeframe, execution_time,
                status, error, notes, extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, symbol, side, quantity, price, order_id, trade_id,
                fee, fee_asset, pnl, strategy, timeframe, execution_time,
                status, error, notes, json.dumps(extra)
            ))
            
            trade_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Stored trade: {trade_id}")
            
            return trade_id
        except Exception as e:
            self.logger.error(f"Failed to store trade: {e}")
            return -1
            
    def store_performance(self, metric: str, value: float, symbol: Optional[str] = None, 
                         timeframe: Optional[str] = None, notes: Optional[str] = None) -> int:
        """
        Store a performance metric in the database
        
        Args:
            metric: Metric name
            value: Metric value
            symbol: Symbol (optional)
            timeframe: Timeframe (optional)
            notes: Notes (optional)
            
        Returns:
            Performance ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert performance metric
            cursor.execute('''
            INSERT INTO performance (
                timestamp, metric, value, symbol, timeframe, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                int(time.time()), metric, value, symbol, timeframe, notes
            ))
            
            performance_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Stored performance metric: {metric} = {value}")
            
            return performance_id
        except Exception as e:
            self.logger.error(f"Failed to store performance metric: {e}")
            return -1
            
    def store_config(self, config: Dict, notes: Optional[str] = None) -> int:
        """
        Store a configuration snapshot in the database
        
        Args:
            config: Configuration dictionary
            notes: Notes (optional)
            
        Returns:
            Config ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert config
            cursor.execute('''
            INSERT INTO config (
                timestamp, config, notes
            ) VALUES (?, ?, ?)
            ''', (
                int(time.time()), json.dumps(config), notes
            ))
            
            config_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Stored configuration snapshot: {config_id}")
            
            return config_id
        except Exception as e:
            self.logger.error(f"Failed to store configuration: {e}")
            return -1
            
    def store_error(self, source: str, error: str, details: Optional[Dict] = None) -> int:
        """
        Store an error in the database
        
        Args:
            source: Error source
            error: Error message
            details: Error details (optional)
            
        Returns:
            Error ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert error
            cursor.execute('''
            INSERT INTO errors (
                timestamp, source, error, details
            ) VALUES (?, ?, ?, ?)
            ''', (
                int(time.time()), source, error, json.dumps(details) if details else None
            ))
            
            error_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Stored error: {source} - {error}")
            
            return error_id
        except Exception as e:
            self.logger.error(f"Failed to store error: {e}")
            return -1
            
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100, 
                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict]:
        """
        Get trades from the database
        
        Args:
            symbol: Symbol filter (optional)
            limit: Maximum number of trades to return
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of trade dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM trades"
            params = []
            
            # Add filters
            filters = []
            
            if symbol:
                filters.append("symbol = ?")
                params.append(symbol)
                
            if start_time:
                filters.append("timestamp >= ?")
                params.append(start_time)
                
            if end_time:
                filters.append("timestamp <= ?")
                params.append(end_time)
                
            if filters:
                query += " WHERE " + " AND ".join(filters)
                
            # Add order and limit
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            
            # Convert rows to dictionaries
            trades = []
            for row in cursor.fetchall():
                trade = dict(row)
                
                # Parse extra JSON
                if trade["extra"]:
                    trade["extra"] = json.loads(trade["extra"])
                    
                trades.append(trade)
                
            conn.close()
            
            return trades
        except Exception as e:
            self.logger.error(f"Failed to get trades: {e}")
            return []
            
    def get_performance(self, metric: Optional[str] = None, symbol: Optional[str] = None, 
                       limit: int = 100, start_time: Optional[int] = None, 
                       end_time: Optional[int] = None) -> List[Dict]:
        """
        Get performance metrics from the database
        
        Args:
            metric: Metric filter (optional)
            symbol: Symbol filter (optional)
            limit: Maximum number of metrics to return
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of performance dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM performance"
            params = []
            
            # Add filters
            filters = []
            
            if metric:
                filters.append("metric = ?")
                params.append(metric)
                
            if symbol:
                filters.append("symbol = ?")
                params.append(symbol)
                
            if start_time:
                filters.append("timestamp >= ?")
                params.append(start_time)
                
            if end_time:
                filters.append("timestamp <= ?")
                params.append(end_time)
                
            if filters:
                query += " WHERE " + " AND ".join(filters)
                
            # Add order and limit
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            
            # Convert rows to dictionaries
            metrics = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return []
            
    def get_latest_config(self) -> Optional[Dict]:
        """
        Get the latest configuration from the database
        
        Returns:
            Configuration dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get latest config
            cursor.execute("SELECT * FROM config ORDER BY timestamp DESC LIMIT 1")
            
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                config = dict(row)
                config["config"] = json.loads(config["config"])
                return config
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to get latest config: {e}")
            return None
            
    def get_errors(self, source: Optional[str] = None, resolved: Optional[bool] = None, 
                  limit: int = 100, start_time: Optional[int] = None, 
                  end_time: Optional[int] = None) -> List[Dict]:
        """
        Get errors from the database
        
        Args:
            source: Source filter (optional)
            resolved: Resolved filter (optional)
            limit: Maximum number of errors to return
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of error dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM errors"
            params = []
            
            # Add filters
            filters = []
            
            if source:
                filters.append("source = ?")
                params.append(source)
                
            if resolved is not None:
                filters.append("resolved = ?")
                params.append(1 if resolved else 0)
                
            if start_time:
                filters.append("timestamp >= ?")
                params.append(start_time)
                
            if end_time:
                filters.append("timestamp <= ?")
                params.append(end_time)
                
            if filters:
                query += " WHERE " + " AND ".join(filters)
                
            # Add order and limit
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            
            # Convert rows to dictionaries
            errors = []
            for row in cursor.fetchall():
                error = dict(row)
                
                # Parse details JSON
                if error["details"]:
                    error["details"] = json.loads(error["details"])
                    
                errors.append(error)
                
            conn.close()
            
            return errors
        except Exception as e:
            self.logger.error(f"Failed to get errors: {e}")
            return []
            
    def mark_error_resolved(self, error_id: int) -> bool:
        """
        Mark an error as resolved
        
        Args:
            error_id: Error ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update error
            cursor.execute("UPDATE errors SET resolved = 1 WHERE id = ?", (error_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Marked error {error_id} as resolved")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to mark error as resolved: {e}")
            return False
            
    def get_daily_performance(self, days: int = 7) -> List[Dict]:
        """
        Get daily performance summary
        
        Args:
            days: Number of days to include
            
        Returns:
            List of daily performance dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate start timestamp (midnight of days ago)
            now = datetime.now()
            start_date = datetime(now.year, now.month, now.day) - datetime.timedelta(days=days-1)
            start_timestamp = int(start_date.timestamp())
            
            # Get daily trade counts and PnL
            cursor.execute('''
            SELECT 
                date(datetime(timestamp, 'unixepoch')) as day,
                COUNT(*) as trade_count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl
            FROM trades
            WHERE timestamp >= ?
            GROUP BY day
            ORDER BY day DESC
            ''', (start_timestamp,))
            
            # Convert rows to dictionaries
            daily_performance = []
            for row in cursor.fetchall():
                day, trade_count, winning_trades, losing_trades, total_pnl = row
                
                win_rate = (winning_trades / trade_count) * 100 if trade_count > 0 else 0
                
                daily_performance.append({
                    "day": day,
                    "trade_count": trade_count,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "total_pnl": total_pnl
                })
                
            conn.close()
            
            return daily_performance
        except Exception as e:
            self.logger.error(f"Failed to get daily performance: {e}")
            return []
            
    def get_symbol_performance(self, limit: int = 10) -> List[Dict]:
        """
        Get performance by symbol
        
        Args:
            limit: Maximum number of symbols to return
            
        Returns:
            List of symbol performance dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get performance by symbol
            cursor.execute('''
            SELECT 
                symbol,
                COUNT(*) as trade_count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl
            FROM trades
            GROUP BY symbol
            ORDER BY total_pnl DESC
            LIMIT ?
            ''', (limit,))
            
            # Convert rows to dictionaries
            symbol_performance = []
            for row in cursor.fetchall():
                symbol, trade_count, winning_trades, losing_trades, total_pnl = row
                
                win_rate = (winning_trades / trade_count) * 100 if trade_count > 0 else 0
                
                symbol_performance.append({
                    "symbol": symbol,
                    "trade_count": trade_count,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "total_pnl": total_pnl
                })
                
            conn.close()
            
            return symbol_performance
        except Exception as e:
            self.logger.error(f"Failed to get symbol performance: {e}")
            return []
            
    def export_trades(self, format: str = "csv", file_path: Optional[str] = None) -> Optional[str]:
        """
        Export trades to a file
        
        Args:
            format: Export format (csv, json)
            file_path: File path (optional)
            
        Returns:
            File path or None if export failed
        """
        try:
            # Get all trades
            trades = self.get_trades(limit=10000)
            
            if not trades:
                self.logger.warning("No trades to export")
                return None
                
            # Generate file path if not provided
            if not file_path:
                export_dir = Path("exports")
                export_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = str(export_dir / f"trades_{timestamp}.{format}")
                
            # Export to CSV
            if format.lower() == "csv":
                import csv
                
                with open(file_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                    writer.writeheader()
                    writer.writerows(trades)
                    
            # Export to JSON
            elif format.lower() == "json":
                with open(file_path, "w") as f:
                    json.dump(trades, f, indent=2)
                    
            else:
                self.logger.warning(f"Invalid export format: {format}")
                return None
                
            self.logger.info(f"Exported {len(trades)} trades to {file_path}")
            
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to export trades: {e}")
            return None
