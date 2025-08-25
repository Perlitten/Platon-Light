"""
Pydantic models for configuration validation.
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

# Pydantic models based on config.example.yaml

class GeneralConfig(BaseModel):
    mode: Literal['live', 'dry-run'] = 'dry-run'
    exchange: Literal['binance', 'bybit', 'mock'] = 'binance'
    base_currency: str = 'USDT'
    quote_currencies: List[str] = ['BTC', 'ETH']
    market_type: Literal['spot', 'futures'] = 'futures'
    leverage: int = Field(5, ge=1, le=20)
    execution_timeout_ms: int = 300

class PositionSizingConfig(BaseModel):
    method: Literal['fixed', 'percentage', 'kelly'] = 'kelly'
    kelly_fraction: float = Field(0.5, ge=0.1, le=1.0)
    max_position_size_percentage: float = 10.0

class RsiConfig(BaseModel):
    timeframe: str = '1m'
    period: int = 14
    oversold: int = 30
    overbought: int = 70

class MacdConfig(BaseModel):
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

class StochasticConfig(BaseModel):
    k_period: int = 14
    d_period: int = 3
    slowing: int = 3

class EntryConfig(BaseModel):
    min_volume_percentile: int = 70
    min_order_book_imbalance: float = 1.5
    rsi: RsiConfig = Field(default_factory=RsiConfig)
    macd: MacdConfig = Field(default_factory=MacdConfig)
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)

class ProfitTargetConfig(BaseModel):
    type: Literal['fixed', 'dynamic'] = 'dynamic'
    fixed_percentage: float = 0.5
    volatility_multiplier: float = 1.5

class StopLossConfig(BaseModel):
    type: Literal['fixed', 'trailing'] = 'trailing'
    initial_percentage: float = 0.5
    trailing_delta: float = 0.1

class ExitConfig(BaseModel):
    profit_target: ProfitTargetConfig = Field(default_factory=ProfitTargetConfig)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    max_trade_duration_seconds: int = 300

class TradingConfig(BaseModel):
    timeframes: List[str] = ['15s', '1m', '5m']
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)

class DrawdownLimitsConfig(BaseModel):
    warning: float = 3.0
    soft_stop: float = 5.0
    hard_stop: float = 10.0

class RiskConfig(BaseModel):
    max_risk_per_trade_percentage: float = 1.0
    daily_drawdown_limits: DrawdownLimitsConfig = Field(default_factory=DrawdownLimitsConfig)
    max_open_positions: int = 3
    correlation_limit: float = 0.7
    volatility_adjustment: bool = True
    liquidation_safety_margin_percentage: float = 20.0
    abnormal_market_detection: bool = True

class TelegramNotificationsConfig(BaseModel):
    trade_entry: bool = True
    trade_exit: bool = True
    position_update: bool = True
    performance_summary: bool = True
    risk_alerts: bool = True

class TelegramReportScheduleConfig(BaseModel):
    hourly: bool = False
    daily: bool = True
    weekly: bool = True

class TelegramConfig(BaseModel):
    enabled: bool = False
    notification_levels: TelegramNotificationsConfig = Field(default_factory=TelegramNotificationsConfig)
    performance_report_schedule: TelegramReportScheduleConfig = Field(default_factory=TelegramReportScheduleConfig)
    command_access: Dict[str, List[str]] = {'admin_users': [], 'require_2fa': 'true'}

class ConsoleChartsConfig(BaseModel):
    price: bool = True
    indicators: bool = True
    order_book: bool = True
    positions: bool = True

class ConsoleConfig(BaseModel):
    enabled: bool = True
    mode: Literal['minimal', 'standard', 'advanced'] = 'standard'
    update_interval_ms: int = 1000
    color_theme: Literal['dark', 'light', 'high_contrast'] = 'dark'
    charts: ConsoleChartsConfig = Field(default_factory=ConsoleChartsConfig)
    language: str = 'en'

class VisualizationConfig(BaseModel):
    console: ConsoleConfig = Field(default_factory=ConsoleConfig)

class LoggingConfig(BaseModel):
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    handlers: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    loggers: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    root: Dict[str, str] = Field(default_factory=dict)

class MarketDataConfig(BaseModel):
    order_book_depth: int = 20
    websocket_buffer_size: int = 1000
    reconnect_attempts: int = 5
    reconnect_delay_ms: int = 1000

class ExecutionConfig(BaseModel):
    retry_attempts: int = 3
    retry_delay_ms: int = 100
    order_types: List[str] = ['LIMIT', 'MARKET', 'STOP_MARKET']

class CustomIndicatorsConfig(BaseModel):
    enabled: bool = False
    path: str = "custom_indicators/"

class DistributedConfig(BaseModel):
    enabled: bool = False
    nodes: int = 1
    coordinator_url: str = "http://localhost:8000"

class AdvancedConfig(BaseModel):
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    custom_indicators: CustomIndicatorsConfig = Field(default_factory=CustomIndicatorsConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)


class BotConfig(BaseModel):
    """Top-level Pydantic model for the entire bot configuration."""
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
