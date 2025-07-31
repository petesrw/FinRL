#!/usr/bin/env python3
"""
Configuration Manager for Forex Trading System
Handles .env file loading and validation
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import warnings

# Load environment variables from .env file
load_dotenv()

@dataclass
class MT5Config:
    """MT5 Broker Configuration with Cross-Platform Path Support"""
    login: int
    password: str
    server: str
    path: str
    data_path: str
    auto_detect: bool
    timeout: int
    
    @classmethod
    def from_env(cls):
        return cls(
            login=int(os.getenv('MT5_LOGIN', 0)),
            password=os.getenv('MT5_PASSWORD', ''),
            server=os.getenv('MT5_SERVER', 'MetaQuotes-Demo'),
            path=cls._detect_mt5_path(),
            data_path=cls._detect_mt5_data_path(),
            auto_detect=os.getenv('MT5_AUTO_DETECT_PATH', 'true').lower() == 'true',
            timeout=int(os.getenv('MT5_TIMEOUT_SECONDS', 30))
        )
    
    @classmethod
    def _detect_mt5_path(cls) -> str:
        """Detect MT5 installation path based on operating system"""
        import platform
        import os.path
        
        system = platform.system().lower()
        
        if system == 'windows':
            return cls._detect_windows_mt5_path()
        elif system == 'darwin':  # macOS
            return cls._detect_macos_mt5_path()
        elif system == 'linux':
            return cls._detect_linux_mt5_path()
        else:
            return ""
    
    @classmethod
    def _detect_windows_mt5_path(cls) -> str:
        """Detect MT5 path on Windows"""
        # Check custom path first
        custom_path = os.getenv('MT5_PATH_WINDOWS_CUSTOM', '')
        if custom_path and os.path.exists(custom_path):
            return custom_path
        
        # Common Windows paths
        paths_to_check = [
            os.getenv('MT5_PATH_WINDOWS_DEFAULT', r'C:\Program Files\MetaTrader 5\terminal64.exe'),
            os.path.expandvars(os.getenv('MT5_PATH_WINDOWS_APPDATA', r'%LOCALAPPDATA%\Programs\MetaTrader 5\terminal64.exe')),
            r'C:\Program Files (x86)\MetaTrader 5\terminal64.exe',
            r'C:\Users\%USERNAME%\AppData\Local\Programs\MetaTrader 5\terminal64.exe',
            r'D:\MetaTrader 5\terminal64.exe',
            r'E:\MetaTrader 5\terminal64.exe'
        ]
        
        for path in paths_to_check:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        return ""
    
    @classmethod
    def _detect_macos_mt5_path(cls) -> str:
        """Detect MT5 path on macOS (Wine/CrossOver)"""
        # Check custom path first
        custom_path = os.getenv('MT5_PATH_MACOS_CUSTOM', '')
        if custom_path and os.path.exists(os.path.expanduser(custom_path)):
            return os.path.expanduser(custom_path)
        
        # Common macOS paths
        paths_to_check = [
            os.getenv('MT5_PATH_MACOS_WINE', '~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'),
            os.getenv('MT5_PATH_MACOS_CROSSOVER', '~/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine'),
            '~/.wine/drive_c/Program Files (x86)/MetaTrader 5/terminal64.exe',
            '~/Library/Application Support/CrossOver/Bottles/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe'
        ]
        
        for path in paths_to_check:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        return ""
    
    @classmethod
    def _detect_linux_mt5_path(cls) -> str:
        """Detect MT5 path on Linux (Wine)"""
        # Check custom path first
        custom_path = os.getenv('MT5_PATH_LINUX_CUSTOM', '')
        if custom_path and os.path.exists(os.path.expanduser(custom_path)):
            return os.path.expanduser(custom_path)
        
        # Common Linux paths
        paths_to_check = [
            os.getenv('MT5_PATH_LINUX_WINE', '~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'),
            os.getenv('MT5_PATH_LINUX_PLAYONLINUX', '~/.PlayOnLinux/wineprefix/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe'),
            '~/.wine/drive_c/Program Files (x86)/MetaTrader 5/terminal64.exe',
            '/opt/wine-stable/bin/wine ~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'
        ]
        
        for path in paths_to_check:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        return ""
    
    @classmethod
    def _detect_mt5_data_path(cls) -> str:
        """Detect MT5 data path based on operating system"""
        import platform
        
        system = platform.system().lower()
        
        if system == 'windows':
            return os.path.expandvars(os.getenv('MT5_DATA_PATH_WINDOWS', r'%APPDATA%\MetaQuotes\Terminal'))
        elif system == 'darwin':  # macOS
            return os.path.expanduser(os.getenv('MT5_DATA_PATH_MACOS', '~/.wine/drive_c/users/crossover/Application Data/MetaQuotes/Terminal'))
        elif system == 'linux':
            return os.path.expanduser(os.getenv('MT5_DATA_PATH_LINUX', '~/.wine/drive_c/users/$USER/Application Data/MetaQuotes/Terminal'))
        else:
            return ""
    
    def is_valid(self) -> bool:
        return self.login > 0 and len(self.password) > 0 and len(self.server) > 0
    
    def is_path_valid(self) -> bool:
        return len(self.path) > 0 and os.path.exists(self.path)

@dataclass
class TradingConfig:
    """Trading Parameters Configuration"""
    default_symbol: str
    risk_per_trade: float
    max_drawdown: float
    max_daily_trades: int
    max_concurrent_positions: int
    target_win_rate: float
    target_profit_factor: float
    target_sharpe_ratio: float
    
    @classmethod
    def from_env(cls):
        return cls(
            default_symbol=os.getenv('DEFAULT_SYMBOL', 'EURUSD'),
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.02)),
            max_drawdown=float(os.getenv('MAX_DRAWDOWN', 0.20)),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', 100)),
            max_concurrent_positions=int(os.getenv('MAX_CONCURRENT_POSITIONS', 3)),
            target_win_rate=float(os.getenv('TARGET_WIN_RATE', 0.65)),
            target_profit_factor=float(os.getenv('TARGET_PROFIT_FACTOR', 1.5)),
            target_sharpe_ratio=float(os.getenv('TARGET_SHARPE_RATIO', 1.2))
        )

@dataclass
class ModelConfig:
    """Model and Training Configuration"""
    model_type: str
    training_timesteps: int
    lookback_window: int
    initial_balance: float
    model_save_path: str
    model_name_prefix: str
    timeframe: str
    
    @classmethod
    def from_env(cls):
        return cls(
            model_type=os.getenv('MODEL_TYPE', 'PPO'),
            training_timesteps=int(os.getenv('TRAINING_TIMESTEPS', 100000)),
            lookback_window=int(os.getenv('LOOKBACK_WINDOW', 100)),
            initial_balance=float(os.getenv('INITIAL_BALANCE', 10000.0)),
            model_save_path=os.getenv('MODEL_SAVE_PATH', './models/'),
            model_name_prefix=os.getenv('MODEL_NAME_PREFIX', 'forex_model'),
            timeframe=os.getenv('TIMEFRAME', '5m')
        )

@dataclass
class ScheduleConfig:
    """Trading Schedule Configuration"""
    trading_start_hour: int
    trading_end_hour: int
    trading_days: List[int]
    trading_timezone: str
    
    @classmethod
    def from_env(cls):
        trading_days_str = os.getenv('TRADING_DAYS', '0,1,2,3,4')
        trading_days = [int(day.strip()) for day in trading_days_str.split(',')]
        
        return cls(
            trading_start_hour=int(os.getenv('TRADING_START_HOUR', 8)),
            trading_end_hour=int(os.getenv('TRADING_END_HOUR', 18)),
            trading_days=trading_days,
            trading_timezone=os.getenv('TRADING_TIMEZONE', 'UTC')
        )

@dataclass
class IndicatorConfig:
    """Technical Indicators Configuration with Automatic Selection Support"""
    # Automatic Selection Settings
    auto_select_indicators: bool
    indicator_optimization_method: str
    allow_manual_override: bool
    auto_indicator_count: int
    auto_optimization_period: int
    auto_performance_threshold: float
    
    # Manual/Fallback Indicator Settings
    atr_period: int
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    sma_fast: int
    sma_slow: int
    ema_fast: int
    ema_slow: int
    rsi_period: int
    stoch_k_period: int
    stoch_d_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_period: int
    bb_std: float
    williams_r_period: int
    cci_period: int
    
    # Force Override Settings (Optional)
    force_rsi_period: Optional[int]
    force_macd_fast: Optional[int]
    force_macd_slow: Optional[int]
    force_sma_fast: Optional[int]
    force_sma_slow: Optional[int]
    force_bb_period: Optional[int]
    force_atr_period: Optional[int]
    
    @classmethod
    def from_env(cls):
        return cls(
            # Automatic Selection Settings
            auto_select_indicators=os.getenv('AUTO_SELECT_INDICATORS', 'true').lower() == 'true',
            indicator_optimization_method=os.getenv('INDICATOR_OPTIMIZATION_METHOD', 'smart_defaults'),
            allow_manual_override=os.getenv('ALLOW_MANUAL_OVERRIDE', 'true').lower() == 'true',
            auto_indicator_count=int(os.getenv('AUTO_INDICATOR_COUNT', 8)),
            auto_optimization_period=int(os.getenv('AUTO_OPTIMIZATION_PERIOD', 1000)),
            auto_performance_threshold=float(os.getenv('AUTO_PERFORMANCE_THRESHOLD', 0.65)),
            
            # Manual/Fallback Settings
            atr_period=int(os.getenv('ATR_PERIOD', 14)),
            sl_atr_multiplier=float(os.getenv('SL_ATR_MULTIPLIER', 1.5)),
            tp_atr_multiplier=float(os.getenv('TP_ATR_MULTIPLIER', 2.5)),
            sma_fast=int(os.getenv('SMA_FAST', 20)),
            sma_slow=int(os.getenv('SMA_SLOW', 50)),
            ema_fast=int(os.getenv('EMA_FAST', 12)),
            ema_slow=int(os.getenv('EMA_SLOW', 26)),
            rsi_period=int(os.getenv('RSI_PERIOD', 14)),
            stoch_k_period=int(os.getenv('STOCH_K_PERIOD', 14)),
            stoch_d_period=int(os.getenv('STOCH_D_PERIOD', 3)),
            macd_fast=int(os.getenv('MACD_FAST', 12)),
            macd_slow=int(os.getenv('MACD_SLOW', 26)),
            macd_signal=int(os.getenv('MACD_SIGNAL', 9)),
            bb_period=int(os.getenv('BB_PERIOD', 20)),
            bb_std=float(os.getenv('BB_STD', 2.0)),
            williams_r_period=int(os.getenv('WILLIAMS_R_PERIOD', 14)),
            cci_period=int(os.getenv('CCI_PERIOD', 20)),
            
            # Force Override Settings
            force_rsi_period=int(os.getenv('FORCE_RSI_PERIOD')) if os.getenv('FORCE_RSI_PERIOD') else None,
            force_macd_fast=int(os.getenv('FORCE_MACD_FAST')) if os.getenv('FORCE_MACD_FAST') else None,
            force_macd_slow=int(os.getenv('FORCE_MACD_SLOW')) if os.getenv('FORCE_MACD_SLOW') else None,
            force_sma_fast=int(os.getenv('FORCE_SMA_FAST')) if os.getenv('FORCE_SMA_FAST') else None,
            force_sma_slow=int(os.getenv('FORCE_SMA_SLOW')) if os.getenv('FORCE_SMA_SLOW') else None,
            force_bb_period=int(os.getenv('FORCE_BB_PERIOD')) if os.getenv('FORCE_BB_PERIOD') else None,
            force_atr_period=int(os.getenv('FORCE_ATR_PERIOD')) if os.getenv('FORCE_ATR_PERIOD') else None
        )

@dataclass
class LoggingConfig:
    """Logging Configuration"""
    log_level: str
    log_file: str
    log_max_size: int
    log_backup_count: int
    save_trades_to_db: bool
    db_file: str
    
    @classmethod
    def from_env(cls):
        return cls(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'forex_trading.log'),
            log_max_size=int(os.getenv('LOG_MAX_SIZE', 10485760)),
            log_backup_count=int(os.getenv('LOG_BACKUP_COUNT', 5)),
            save_trades_to_db=os.getenv('SAVE_TRADES_TO_DB', 'true').lower() == 'true',
            db_file=os.getenv('DB_FILE', 'trades.db')
        )

@dataclass
class NotificationConfig:
    """Notification Configuration"""
    telegram_bot_token: str
    telegram_chat_id: str
    email_smtp_server: str
    email_smtp_port: int
    email_username: str
    email_password: str
    email_to: str
    
    @classmethod
    def from_env(cls):
        return cls(
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
            email_smtp_server=os.getenv('EMAIL_SMTP_SERVER', ''),
            email_smtp_port=int(os.getenv('EMAIL_SMTP_PORT', 587)),
            email_username=os.getenv('EMAIL_USERNAME', ''),
            email_password=os.getenv('EMAIL_PASSWORD', ''),
            email_to=os.getenv('EMAIL_TO', '')
        )
    
    def has_telegram(self) -> bool:
        return len(self.telegram_bot_token) > 0 and len(self.telegram_chat_id) > 0
    
    def has_email(self) -> bool:
        return (len(self.email_smtp_server) > 0 and 
                len(self.email_username) > 0 and 
                len(self.email_password) > 0 and 
                len(self.email_to) > 0)

@dataclass
class SafetyConfig:
    """Safety Settings Configuration"""
    enable_emergency_stop: bool
    max_consecutive_losses: int
    emergency_stop_loss_amount: float
    enable_sl_cooldown: bool
    sl_cooldown_minutes: int
    avoid_news_trading: bool
    news_buffer_minutes: int
    max_slippage_pips: int
    max_spread_pips: int
    
    @classmethod
    def from_env(cls):
        return cls(
            enable_emergency_stop=os.getenv('ENABLE_EMERGENCY_STOP', 'true').lower() == 'true',
            max_consecutive_losses=int(os.getenv('MAX_CONSECUTIVE_LOSSES', 5)),
            emergency_stop_loss_amount=float(os.getenv('EMERGENCY_STOP_LOSS_AMOUNT', 1000.0)),
            enable_sl_cooldown=os.getenv('ENABLE_SL_COOLDOWN', 'true').lower() == 'true',
            sl_cooldown_minutes=int(os.getenv('SL_COOLDOWN_MINUTES', 30)),
            avoid_news_trading=os.getenv('AVOID_NEWS_TRADING', 'true').lower() == 'true',
            news_buffer_minutes=int(os.getenv('NEWS_BUFFER_MINUTES', 30)),
            max_slippage_pips=int(os.getenv('MAX_SLIPPAGE_PIPS', 3)),
            max_spread_pips=int(os.getenv('MAX_SPREAD_PIPS', 5))
        )

@dataclass
class DevelopmentConfig:
    """Development Settings Configuration"""
    demo_mode: bool
    backtest_start_date: str
    backtest_end_date: str
    backtest_initial_balance: float
    debug_mode: bool
    save_debug_data: bool
    debug_data_path: str
    
    @classmethod
    def from_env(cls):
        return cls(
            demo_mode=os.getenv('DEMO_MODE', 'true').lower() == 'true',
            backtest_start_date=os.getenv('BACKTEST_START_DATE', '2020-01-01'),
            backtest_end_date=os.getenv('BACKTEST_END_DATE', '2023-12-31'),
            backtest_initial_balance=float(os.getenv('BACKTEST_INITIAL_BALANCE', 10000.0)),
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            save_debug_data=os.getenv('SAVE_DEBUG_DATA', 'false').lower() == 'true',
            debug_data_path=os.getenv('DEBUG_DATA_PATH', './debug/')
        )

@dataclass
class APIConfig:
    """API Settings Configuration"""
    enable_web_dashboard: bool
    dashboard_host: str
    dashboard_port: int
    dashboard_secret_key: str
    enable_rest_api: bool
    api_host: str
    api_port: int
    api_secret_key: str
    
    @classmethod
    def from_env(cls):
        return cls(
            enable_web_dashboard=os.getenv('ENABLE_WEB_DASHBOARD', 'false').lower() == 'true',
            dashboard_host=os.getenv('DASHBOARD_HOST', 'localhost'),
            dashboard_port=int(os.getenv('DASHBOARD_PORT', 8000)),
            dashboard_secret_key=os.getenv('DASHBOARD_SECRET_KEY', 'your_secret_key_here'),
            enable_rest_api=os.getenv('ENABLE_REST_API', 'false').lower() == 'true',
            api_host=os.getenv('API_HOST', 'localhost'),
            api_port=int(os.getenv('API_PORT', 8001)),
            api_secret_key=os.getenv('API_SECRET_KEY', 'your_api_secret_here')
        )

@dataclass
class AdvancedConfig:
    """Advanced Settings Configuration"""
    enable_multi_symbol: bool
    trading_symbols: List[str]
    portfolio_allocation: Dict[str, float]
    enable_correlation_filter: bool
    max_correlation_threshold: float
    position_sizing_method: str
    kelly_lookback_trades: int
    kelly_max_fraction: float
    
    @classmethod
    def from_env(cls):
        # Parse trading symbols
        symbols_str = os.getenv('TRADING_SYMBOLS', 'EURUSD,GBPUSD,USDJPY,AUDUSD')
        trading_symbols = [symbol.strip() for symbol in symbols_str.split(',')]
        
        # Parse portfolio allocation
        portfolio_allocation = {}
        for symbol in trading_symbols:
            key = f'PORTFOLIO_ALLOCATION_{symbol}'
            allocation = float(os.getenv(key, 0.25))  # Default equal allocation
            portfolio_allocation[symbol] = allocation
        
        return cls(
            enable_multi_symbol=os.getenv('ENABLE_MULTI_SYMBOL', 'false').lower() == 'true',
            trading_symbols=trading_symbols,
            portfolio_allocation=portfolio_allocation,
            enable_correlation_filter=os.getenv('ENABLE_CORRELATION_FILTER', 'false').lower() == 'true',
            max_correlation_threshold=float(os.getenv('MAX_CORRELATION_THRESHOLD', 0.7)),
            position_sizing_method=os.getenv('POSITION_SIZING_METHOD', 'PERCENT_RISK'),
            kelly_lookback_trades=int(os.getenv('KELLY_LOOKBACK_TRADES', 100)),
            kelly_max_fraction=float(os.getenv('KELLY_MAX_FRACTION', 0.25))
        )

class ConfigManager:
    """Main Configuration Manager"""
    
    def __init__(self):
        self.mt5 = MT5Config.from_env()
        self.trading = TradingConfig.from_env()
        self.model = ModelConfig.from_env()
        self.schedule = ScheduleConfig.from_env()
        self.indicators = IndicatorConfig.from_env()
        self.logging = LoggingConfig.from_env()
        self.notifications = NotificationConfig.from_env()
        self.safety = SafetyConfig.from_env()
        self.development = DevelopmentConfig.from_env()
        self.api = APIConfig.from_env()
        self.advanced = AdvancedConfig.from_env()
        
        self._setup_logging()
        self._validate_config()
        self._create_directories()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.logging.log_level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.logging.log_file,
            maxBytes=self.logging.log_max_size,
            backupCount=self.logging.log_backup_count
        )
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuration loaded successfully")
    
    def _validate_config(self):
        """Validate configuration settings"""
        errors = []
        warnings_list = []
        
        # Validate MT5 config
        if not self.development.demo_mode and not self.mt5.is_valid():
            errors.append("MT5 credentials are required for live trading")
        
        # Validate trading parameters
        if not 0 < self.trading.risk_per_trade <= 0.1:
            warnings_list.append(f"Risk per trade {self.trading.risk_per_trade:.1%} seems unusual (recommended: 1-5%)")
        
        if not 0.5 <= self.trading.target_win_rate <= 0.9:
            warnings_list.append(f"Target win rate {self.trading.target_win_rate:.1%} may be unrealistic")
        
        # Validate model config
        if self.model.model_type not in ['PPO', 'SAC', 'A2C']:
            errors.append(f"Invalid model type: {self.model.model_type}")
        
        # Validate schedule
        if not 0 <= self.schedule.trading_start_hour <= 23:
            errors.append(f"Invalid trading start hour: {self.schedule.trading_start_hour}")
        
        if not 0 <= self.schedule.trading_end_hour <= 23:
            errors.append(f"Invalid trading end hour: {self.schedule.trading_end_hour}")
        
        # Validate portfolio allocation (if multi-symbol enabled)
        if self.advanced.enable_multi_symbol:
            total_allocation = sum(self.advanced.portfolio_allocation.values())
            if abs(total_allocation - 1.0) > 0.01:
                warnings_list.append(f"Portfolio allocation sums to {total_allocation:.2f}, not 1.0")
        
        # Log errors and warnings
        for error in errors:
            self.logger.error(error)
        
        for warning in warnings_list:
            self.logger.warning(warning)
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.model.model_save_path,
            os.path.dirname(self.logging.log_file),
            os.path.dirname(self.logging.db_file) if self.logging.save_trades_to_db else None,
            self.development.debug_data_path if self.development.save_debug_data else None
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    self.logger.info(f"Created directory: {directory}")
                except Exception as e:
                    self.logger.error(f"Failed to create directory {directory}: {e}")
    
    def get_model_path(self, symbol: str) -> str:
        """Get full model file path"""
        filename = f"{self.model.model_name_prefix}_{self.model.model_type}_{symbol}.zip"
        return os.path.join(self.model.model_save_path, filename)
    
    def is_trading_time(self) -> bool:
        """Check if current time is within trading hours"""
        from datetime import datetime
        import pytz
        
        try:
            tz = pytz.timezone(self.schedule.trading_timezone)
            now = datetime.now(tz)
            
            # Check day of week
            if now.weekday() not in self.schedule.trading_days:
                return False
            
            # Check hour
            if not self.schedule.trading_start_hour <= now.hour < self.schedule.trading_end_hour:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking trading time: {e}")
            return True  # Default to allow trading if check fails
    
    def get_symbols_list(self) -> List[str]:
        """Get list of symbols to trade"""
        if self.advanced.enable_multi_symbol:
            return self.advanced.trading_symbols
        else:
            return [self.trading.default_symbol]
    
    def get_symbol_allocation(self, symbol: str) -> float:
        """Get portfolio allocation for a symbol"""
        if self.advanced.enable_multi_symbol:
            return self.advanced.portfolio_allocation.get(symbol, 0.0)
        else:
            return 1.0 if symbol == self.trading.default_symbol else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'mt5': self.mt5.__dict__,
            'trading': self.trading.__dict__,
            'model': self.model.__dict__,
            'schedule': self.schedule.__dict__,
            'indicators': self.indicators.__dict__,
            'logging': self.logging.__dict__,
            'notifications': self.notifications.__dict__,
            'safety': self.safety.__dict__,
            'development': self.development.__dict__,
            'api': self.api.__dict__,
            'advanced': self.advanced.__dict__
        }
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("🔧 FOREX TRADING SYSTEM CONFIGURATION")
        print("="*60)
        
        print(f"\n📊 TRADING SETTINGS:")
        print(f"   Symbol(s): {', '.join(self.get_symbols_list())}")
        print(f"   Risk per trade: {self.trading.risk_per_trade:.1%}")
        print(f"   Target win rate: {self.trading.target_win_rate:.1%}")
        print(f"   Max drawdown: {self.trading.max_drawdown:.1%}")
        
        print(f"\n🤖 MODEL SETTINGS:")
        print(f"   Algorithm: {self.model.model_type}")
        print(f"   Training steps: {self.model.training_timesteps:,}")
        print(f"   Initial balance: ${self.model.initial_balance:,.2f}")
        
        print(f"\n⏰ SCHEDULE:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        trading_days = [days[i] for i in self.schedule.trading_days]
        print(f"   Trading days: {', '.join(trading_days)}")
        print(f"   Trading hours: {self.schedule.trading_start_hour:02d}:00 - {self.schedule.trading_end_hour:02d}:00 {self.schedule.trading_timezone}")
        
        print(f"\n🛡️ SAFETY:")
        print(f"   Emergency stop: {'Enabled' if self.safety.enable_emergency_stop else 'Disabled'}")
        print(f"   Max consecutive losses: {self.safety.max_consecutive_losses}")
        print(f"   Max spread: {self.safety.max_spread_pips} pips")
        
        print(f"\n🔗 CONNECTION:")
        if self.development.demo_mode:
            print(f"   Mode: DEMO (simulated data)")
        else:
            print(f"   Mode: LIVE (MT5 connection)")
            print(f"   Server: {self.mt5.server}")
            print(f"   Login: {self.mt5.login}")
        
        print(f"\n📝 LOGGING:")
        print(f"   Level: {self.logging.log_level}")
        print(f"   File: {self.logging.log_file}")
        print(f"   Database: {'Enabled' if self.logging.save_trades_to_db else 'Disabled'}")
        
        if self.notifications.has_telegram() or self.notifications.has_email():
            print(f"\n📱 NOTIFICATIONS:")
            if self.notifications.has_telegram():
                print(f"   Telegram: Enabled")
            if self.notifications.has_email():
                print(f"   Email: Enabled")
        
        print("="*60)

# Global configuration instance
config = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload configuration from .env file"""
    global config
    load_dotenv(override=True)  # Reload .env file
    config = ConfigManager()
    return config

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    config.print_summary()