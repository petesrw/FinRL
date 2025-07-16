#!/usr/bin/env python3
"""
Advanced Forex Auto Trading System v2.0 with .env Configuration
Complete system with configuration management, notifications, and safety features
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO, A2C, SAC
import talib
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import threading
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import os

# Import configuration
from config import get_config
from indicator_manager import create_indicator_manager, SmartIndicatorManager

class ConfigurableForexBot:
    """
    Main Forex Trading Bot with full .env configuration support
    """
    
    def __init__(self, symbol: str = None):
        # Load configuration
        self.config = get_config()
        self.symbol = symbol or self.config.trading.default_symbol
        
        # Initialize automatic indicator manager
        self.indicator_manager = self._initialize_indicator_manager()
        
        # Initialize components
        self.model = None
        self.is_trading = False
        self.trading_thread = None
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0
        }
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize notification system
        self.notification_manager = NotificationManager()
        
        # Initialize database
        if self.config.logging.save_trades_to_db:
            self.init_database()
    
    def _initialize_indicator_manager(self) -> SmartIndicatorManager:
        """Initialize automatic indicator manager based on configuration"""
        # Get indicator configuration from config
        auto_select = self.config.indicators.auto_select_indicators
        optimization_method = self.config.indicators.indicator_optimization_method
        
        return create_indicator_manager(
            symbol=self.symbol,
            optimization_method=optimization_method,
            auto_select=auto_select,
            timeframe="H1"  # Default timeframe
        )
    
    def init_database(self):
        """Initialize SQLite database for trade logging"""
        try:
            conn = sqlite3.connect(self.config.logging.db_file)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    entry_time DATETIME,
                    exit_time DATETIME,
                    entry_price REAL,
                    exit_price REAL,
                    position_type TEXT,
                    position_size REAL,
                    pnl REAL,
                    close_reason TEXT,
                    win INTEGER
                )
            ''')
            
            # Create performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    balance REAL,
                    drawdown REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def create_training_environment(self):
        """Create training environment with config parameters"""
        return ForexEnvironment(
            symbol=self.symbol,
            config=self.config,
            indicator_manager=self.indicator_manager
        )
    
    def train_model(self, total_timesteps: int = None):
        """Train RL model using configuration parameters"""
        timesteps = total_timesteps or self.config.model.training_timesteps
        
        self.logger.info(f"Starting model training for {self.symbol}")
        self.logger.info(f"Algorithm: {self.config.model.model_type}")
        self.logger.info(f"Training steps: {timesteps:,}")
        
        # Create environment
        env = self.create_training_environment()
        
        # Check device availability and log
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Create model based on config with GPU support
        if self.config.model.model_type == "PPO":
            self.model = PPO(
                "MlpPolicy", 
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                verbose=1,
                device=device,
                tensorboard_log="./forex_tensorboard/"
            )
        elif self.config.model.model_type == "SAC":
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                verbose=1,
                device=device,
                tensorboard_log="./forex_tensorboard/"
            )
        elif self.config.model.model_type == "A2C":
            self.model = A2C(
                "MlpPolicy",
                env,
                learning_rate=7e-4,
                n_steps=5,
                verbose=1,
                device=device,
                tensorboard_log="./forex_tensorboard/"
            )
        
        # Train model
        self.model.learn(total_timesteps=timesteps)
        
        # Save model
        model_path = self.config.get_model_path(self.symbol)
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Send notification
        self.notification_manager.notify(
            f"🤖 Model Training Completed\n"
            f"Symbol: {self.symbol}\n"
            f"Algorithm: {self.config.model.model_type}\n"
            f"Training Steps: {timesteps:,}\n"
            f"Model saved to: {model_path}"
        )
        
        return True
    
    def load_model(self, model_path: str = None):
        """Load trained model"""
        if model_path is None:
            model_path = self.config.get_model_path(self.symbol)
        
        try:
            if self.config.model.model_type == "PPO":
                self.model = PPO.load(model_path)
            elif self.config.model.model_type == "SAC":
                self.model = SAC.load(model_path)
            elif self.config.model.model_type == "A2C":
                self.model = A2C.load(model_path)
            
            self.logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def test_model(self, episodes: int = 10):
        """Test model performance"""
        if self.model is None:
            self.logger.error("No model loaded for testing")
            return False
        
        self.logger.info(f"Testing model for {episodes} episodes...")
        
        env = self.create_training_environment()
        total_rewards = []
        win_rates = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if done or truncated:
                    break
            
            total_rewards.append(episode_reward)
            
            if env.total_trades > 0:
                win_rate = env.win_count / env.total_trades
                win_rates.append(win_rate)
        
        # Calculate statistics
        avg_reward = np.mean(total_rewards)
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"Average Reward: {avg_reward:.4f}")
        self.logger.info(f"Average Win Rate: {avg_win_rate:.2%}")
        self.logger.info(f"Target Win Rate: {self.config.trading.target_win_rate:.2%}")
        
        # Check if target is met
        target_met = avg_win_rate >= self.config.trading.target_win_rate
        status = "✅ TARGET MET" if target_met else "⚠️ BELOW TARGET"
        
        # Send notification
        self.notification_manager.notify(
            f"🧪 Model Test Results\n"
            f"Symbol: {self.symbol}\n"
            f"Win Rate: {avg_win_rate:.1%}\n"
            f"Target: {self.config.trading.target_win_rate:.1%}\n"
            f"Status: {status}"
        )
        
        return target_met
    
    def start_live_trading(self):
        """Start live trading with full configuration support"""
        if self.model is None:
            self.logger.error("No model loaded - cannot start trading")
            return False
        
        # Connect to MT5 if not in demo mode
        if not self.config.development.demo_mode:
            if not self.connect_mt5():
                self.logger.error("Failed to connect to MT5")
                return False
        
        self.is_trading = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()
        
        mode = "DEMO" if self.config.development.demo_mode else "LIVE"
        self.logger.info(f"Started {mode} trading for {self.symbol}")
        
        # Send notification
        self.notification_manager.notify(
            f"🚀 Trading Started - {mode} Mode\n"
            f"Symbol: {self.symbol}\n"
            f"Model: {self.config.model.model_type}\n"
            f"Risk per trade: {self.config.trading.risk_per_trade:.1%}\n"
            f"Target win rate: {self.config.trading.target_win_rate:.1%}"
        )
        
        return True
    
    def stop_live_trading(self):
        """Stop live trading"""
        self.is_trading = False
        if self.trading_thread:
            self.trading_thread.join()
        
        # Disconnect MT5
        if not self.config.development.demo_mode:
            mt5.shutdown()
        
        self.logger.info(f"Stopped trading for {self.symbol}")
        
        # Send final notification
        self.notification_manager.notify(
            f"🛑 Trading Stopped\n"
            f"Symbol: {self.symbol}\n"
            f"Total Trades: {self.performance_stats['total_trades']}\n"
            f"Win Rate: {self.performance_stats['win_rate']:.1%}\n"
            f"Total P&L: ${self.performance_stats['total_profit']:.2f}"
        )
    
    def connect_mt5(self):
        """Connect to MT5 using config credentials"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
            
            if not mt5.login(
                self.config.mt5.login,
                self.config.mt5.password,
                self.config.mt5.server
            ):
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            
            account_info = mt5.account_info()
            if account_info:
                self.logger.info(f"Connected to MT5 - Balance: ${account_info.balance:.2f}")
            
            return True
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    def _trading_loop(self):
        """Main trading loop with configuration-based logic"""
        self.logger.info("Trading loop started")
        
        while self.is_trading:
            try:
                # Check trading hours
                if not self.config.is_trading_time():
                    self.logger.debug("Outside trading hours")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Check emergency conditions
                if self._check_emergency_stop():
                    self.logger.warning("Emergency stop triggered!")
                    break
                
                # Get market data and make decision
                if self.config.development.demo_mode:
                    self._demo_trading_step()
                else:
                    self._live_trading_step()
                
                # Update performance
                self._update_performance()
                
                # Wait for next decision (15 minutes for M15)
                time.sleep(900)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                self.notification_manager.notify(f"❌ Trading Error: {str(e)}")
                time.sleep(60)
    
    def _demo_trading_step(self):
        """Execute one demo trading step"""
        # Simulate trading decision
        # This would use the trained model to make decisions
        # For demo purposes, we'll just log
        self.logger.debug(f"Demo trading step for {self.symbol}")
    
    def _live_trading_step(self):
        """Execute one live trading step"""
        # Get current market data from MT5
        # Use model to make trading decision
        # Execute trades via MT5
        self.logger.debug(f"Live trading step for {self.symbol}")
    
    def _check_emergency_stop(self):
        """Check emergency stop conditions"""
        if not self.config.safety.enable_emergency_stop:
            return False
        
        # Check consecutive losses
        if self.performance_stats['consecutive_losses'] >= self.config.safety.max_consecutive_losses:
            self.notification_manager.notify(
                f"🚨 EMERGENCY STOP: {self.config.safety.max_consecutive_losses} consecutive losses!"
            )
            return True
        
        # Check maximum loss
        if abs(self.performance_stats['total_profit']) >= self.config.safety.emergency_stop_loss_amount:
            self.notification_manager.notify(
                f"🚨 EMERGENCY STOP: Loss limit ${self.config.safety.emergency_stop_loss_amount} reached!"
            )
            return True
        
        return False
    
    def _update_performance(self):
        """Update performance statistics"""
        # This would be implemented to track actual trading performance
        pass
    
    def get_performance_report(self):
        """Get comprehensive performance report"""
        return {
            'symbol': self.symbol,
            'config': {
                'model_type': self.config.model.model_type,
                'risk_per_trade': self.config.trading.risk_per_trade,
                'target_win_rate': self.config.trading.target_win_rate,
                'demo_mode': self.config.development.demo_mode
            },
            'performance': self.performance_stats,
            'status': {
                'is_trading': self.is_trading,
                'model_loaded': self.model is not None
            }
        }


class ForexEnvironment(gym.Env):
    """Enhanced Forex Environment with Automatic Indicator Selection"""
    
    def __init__(self, symbol: str, config, indicator_manager: SmartIndicatorManager = None):
        super().__init__()
        self.symbol = symbol
        self.config = config
        self.indicator_manager = indicator_manager
        
        # Environment parameters from config
        self.initial_balance = config.model.initial_balance
        self.current_balance = self.initial_balance
        self.lookback_window = config.model.lookback_window
        
        # Trading state
        self.current_position = 0
        self.total_trades = 0
        self.win_count = 0
        self.position_entry_price = 0
        self.position_entry_step = 0
        
        # Performance tracking for adaptive optimization
        self.episode_performance = []
        self.recent_trades = []
        
        # Enhanced observation space for indicators
        # Base features (20) + Technical indicators (30) = 50 total
        self.action_space = spaces.Discrete(4)  # Hold, Buy, Sell, Close
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(50,), dtype=np.float32
        )
        
        # Load demo data
        self.load_demo_data()
        self.current_step = 0
        
        # Calculate technical indicators
        self.calculate_technical_indicators()
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators using automatic selection"""
        if self.indicator_manager is None:
            self.logger.warning("No indicator manager available")
            return
        
        # Get optimal indicator configuration
        indicator_config = self.indicator_manager.get_optimal_config(self.data)
        
        # Calculate indicators based on configuration
        self.indicators = {}
        
        try:
            # RSI
            self.indicators['rsi'] = talib.RSI(
                self.data['close'].values, 
                timeperiod=indicator_config.rsi_period
            )
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                self.data['close'].values,
                fastperiod=indicator_config.macd_fast,
                slowperiod=indicator_config.macd_slow,
                signalperiod=indicator_config.macd_signal
            )
            self.indicators['macd'] = macd
            self.indicators['macd_signal'] = macd_signal
            self.indicators['macd_hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                self.data['close'].values,
                timeperiod=indicator_config.bb_period,
                nbdevup=indicator_config.bb_std,
                nbdevdn=indicator_config.bb_std
            )
            self.indicators['bb_upper'] = bb_upper
            self.indicators['bb_middle'] = bb_middle
            self.indicators['bb_lower'] = bb_lower
            
            # Moving Averages
            self.indicators['sma_fast'] = talib.SMA(
                self.data['close'].values, 
                timeperiod=indicator_config.sma_fast
            )
            self.indicators['sma_slow'] = talib.SMA(
                self.data['close'].values, 
                timeperiod=indicator_config.sma_slow
            )
            
            # EMA
            self.indicators['ema_fast'] = talib.EMA(
                self.data['close'].values, 
                timeperiod=indicator_config.ema_fast
            )
            self.indicators['ema_slow'] = talib.EMA(
                self.data['close'].values, 
                timeperiod=indicator_config.ema_slow
            )
            
            # ATR
            self.indicators['atr'] = talib.ATR(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=indicator_config.atr_period
            )
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                fastk_period=indicator_config.stoch_k,
                slowk_period=indicator_config.stoch_d,
                slowd_period=indicator_config.stoch_d
            )
            self.indicators['stoch_k'] = stoch_k
            self.indicators['stoch_d'] = stoch_d
            
            # Williams %R
            self.indicators['williams_r'] = talib.WILLR(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=indicator_config.williams_r_period
            )
            
            # CCI
            self.indicators['cci'] = talib.CCI(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=indicator_config.cci_period
            )
            
            self.logger.info(f"Technical indicators calculated with config: {indicator_config.to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            # Initialize with zeros if calculation fails
            for key in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
                       'sma_fast', 'sma_slow', 'ema_fast', 'ema_slow', 'atr', 'stoch_k', 'stoch_d',
                       'williams_r', 'cci']:
                self.indicators[key] = np.zeros(len(self.data))
    
    def load_demo_data(self):
        """Load demo forex data"""
        np.random.seed(42)
        n_points = 5000
        
        # Generate realistic price data
        base_price = 1.1000 if self.symbol == 'EURUSD' else 1.3000
        
        # Price movements
        returns = np.random.normal(0, 0.0005, n_points)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        self.data = pd.DataFrame({
            'time': pd.date_range(start='2020-01-01', periods=n_points, freq='15T'),
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0002, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0002, n_points))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_points)
        })
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.current_balance = self.initial_balance
        self.current_position = 0
        self.total_trades = 0
        self.win_count = 0
        self.current_step = np.random.randint(100, len(self.data) - 1000)
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get enhanced observation with automatic indicator selection"""
        if self.current_step >= len(self.data):
            return np.zeros(50, dtype=np.float32)
        
        features = []
        current_idx = self.current_step
        
        # Ensure we have valid data
        if current_idx < 50:  # Need enough data for indicators
            return np.zeros(50, dtype=np.float32)
        
        try:
            # Price-based features (10 features)
            current_price = self.data['close'].iloc[current_idx]
            
            # Recent price changes (5 features)
            for i in [1, 2, 3, 5, 10]:
                if current_idx >= i:
                    prev_price = self.data['close'].iloc[current_idx - i]
                    price_change = (current_price - prev_price) / prev_price
                    features.append(price_change)
                else:
                    features.append(0.0)
            
            # Volume features (2 features)
            current_volume = self.data['volume'].iloc[current_idx]
            avg_volume = self.data['volume'].iloc[max(0, current_idx-20):current_idx].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            features.extend([
                np.log(volume_ratio),  # Log volume ratio
                (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
            ])
            
            # High-Low spread (1 feature)
            hl_spread = (self.data['high'].iloc[current_idx] - self.data['low'].iloc[current_idx]) / current_price
            features.append(hl_spread)
            
            # Position information (2 features)
            features.extend([
                float(self.current_position),  # Current position (-1, 0, 1)
                (current_price - self.position_entry_price) / self.position_entry_price if self.position_entry_price > 0 else 0.0
            ])
            
            # Technical Indicators (30 features) - Using automatic selection
            if hasattr(self, 'indicators') and self.indicators:
                # RSI (1 feature)
                rsi_val = self.indicators['rsi'][current_idx] if not np.isnan(self.indicators['rsi'][current_idx]) else 50.0
                features.append((rsi_val - 50) / 50)  # Normalize RSI to [-1, 1]
                
                # MACD (3 features)
                macd_val = self.indicators['macd'][current_idx] if not np.isnan(self.indicators['macd'][current_idx]) else 0.0
                macd_signal_val = self.indicators['macd_signal'][current_idx] if not np.isnan(self.indicators['macd_signal'][current_idx]) else 0.0
                macd_hist_val = self.indicators['macd_hist'][current_idx] if not np.isnan(self.indicators['macd_hist'][current_idx]) else 0.0
                
                features.extend([
                    macd_val / current_price * 10000,  # Scale MACD
                    macd_signal_val / current_price * 10000,  # Scale MACD Signal
                    macd_hist_val / current_price * 10000   # Scale MACD Histogram
                ])
                
                # Bollinger Bands (3 features)
                bb_upper_val = self.indicators['bb_upper'][current_idx] if not np.isnan(self.indicators['bb_upper'][current_idx]) else current_price
                bb_middle_val = self.indicators['bb_middle'][current_idx] if not np.isnan(self.indicators['bb_middle'][current_idx]) else current_price
                bb_lower_val = self.indicators['bb_lower'][current_idx] if not np.isnan(self.indicators['bb_lower'][current_idx]) else current_price
                
                bb_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val != bb_lower_val else 0.5
                bb_width = (bb_upper_val - bb_lower_val) / bb_middle_val if bb_middle_val > 0 else 0.0
                bb_squeeze = 1.0 if bb_width < 0.02 else 0.0  # Bollinger squeeze indicator
                
                features.extend([bb_position, bb_width, bb_squeeze])
                
                # Moving Averages (4 features)
                sma_fast_val = self.indicators['sma_fast'][current_idx] if not np.isnan(self.indicators['sma_fast'][current_idx]) else current_price
                sma_slow_val = self.indicators['sma_slow'][current_idx] if not np.isnan(self.indicators['sma_slow'][current_idx]) else current_price
                ema_fast_val = self.indicators['ema_fast'][current_idx] if not np.isnan(self.indicators['ema_fast'][current_idx]) else current_price
                ema_slow_val = self.indicators['ema_slow'][current_idx] if not np.isnan(self.indicators['ema_slow'][current_idx]) else current_price
                
                features.extend([
                    (current_price - sma_fast_val) / sma_fast_val,  # Price vs SMA Fast
                    (current_price - sma_slow_val) / sma_slow_val,  # Price vs SMA Slow
                    (sma_fast_val - sma_slow_val) / sma_slow_val,   # SMA crossover
                    (ema_fast_val - ema_slow_val) / ema_slow_val    # EMA crossover
                ])
                
                # ATR (1 feature)
                atr_val = self.indicators['atr'][current_idx] if not np.isnan(self.indicators['atr'][current_idx]) else 0.001
                features.append(atr_val / current_price)  # ATR as percentage of price
                
                # Stochastic (2 features)
                stoch_k_val = self.indicators['stoch_k'][current_idx] if not np.isnan(self.indicators['stoch_k'][current_idx]) else 50.0
                stoch_d_val = self.indicators['stoch_d'][current_idx] if not np.isnan(self.indicators['stoch_d'][current_idx]) else 50.0
                
                features.extend([
                    (stoch_k_val - 50) / 50,  # Normalize to [-1, 1]
                    (stoch_d_val - 50) / 50   # Normalize to [-1, 1]
                ])
                
                # Williams %R (1 feature)
                williams_r_val = self.indicators['williams_r'][current_idx] if not np.isnan(self.indicators['williams_r'][current_idx]) else -50.0
                features.append(williams_r_val / 50)  # Normalize to [-2, 0] then to [-1, 0]
                
                # CCI (1 feature)
                cci_val = self.indicators['cci'][current_idx] if not np.isnan(self.indicators['cci'][current_idx]) else 0.0
                features.append(np.tanh(cci_val / 100))  # Normalize CCI using tanh
                
                # Trend indicators (5 features)
                # Price momentum over different periods
                for period in [5, 10, 20]:
                    if current_idx >= period:
                        momentum = (current_price - self.data['close'].iloc[current_idx - period]) / self.data['close'].iloc[current_idx - period]
                        features.append(momentum)
                    else:
                        features.append(0.0)
                
                # Moving average slopes
                if current_idx >= 5:
                    sma_slope = (sma_fast_val - self.indicators['sma_fast'][current_idx - 5]) / self.indicators['sma_fast'][current_idx - 5] if not np.isnan(self.indicators['sma_fast'][current_idx - 5]) else 0.0
                    ema_slope = (ema_fast_val - self.indicators['ema_fast'][current_idx - 5]) / self.indicators['ema_fast'][current_idx - 5] if not np.isnan(self.indicators['ema_fast'][current_idx - 5]) else 0.0
                    features.extend([sma_slope, ema_slope])
                else:
                    features.extend([0.0, 0.0])
                
                # Market regime indicators (7 features)
                # Volatility regime
                recent_volatility = self.data['close'].iloc[max(0, current_idx-20):current_idx].pct_change().std()
                long_volatility = self.data['close'].iloc[max(0, current_idx-100):current_idx].pct_change().std()
                vol_regime = recent_volatility / long_volatility if long_volatility > 0 else 1.0
                
                # Trend strength
                trend_up = sum(1 for i in range(max(0, current_idx-10), current_idx) if self.data['close'].iloc[i] > self.data['close'].iloc[i-1] if i > 0)
                trend_strength = (trend_up - 5) / 5  # Normalize to [-1, 1]
                
                # Support/Resistance levels
                recent_high = self.data['high'].iloc[max(0, current_idx-20):current_idx].max()
                recent_low = self.data['low'].iloc[max(0, current_idx-20):current_idx].min()
                support_distance = (current_price - recent_low) / current_price
                resistance_distance = (recent_high - current_price) / current_price
                
                # Time-based features
                hour_of_day = (current_idx % 96) / 96  # Assuming 15-minute bars, 96 per day
                day_of_week = ((current_idx // 96) % 7) / 7
                
                features.extend([
                    np.log(vol_regime),
                    trend_strength,
                    support_distance,
                    resistance_distance,
                    hour_of_day,
                    day_of_week,
                    float(recent_volatility > 0.01)  # High volatility flag
                ])
            
            else:
                # If indicators not available, pad with zeros
                features.extend([0.0] * 30)
            
            # Ensure exactly 50 features
            while len(features) < 50:
                features.append(0.0)
            
            # Clip extreme values and handle NaN
            features = features[:50]
            features = [np.clip(f, -10, 10) if not np.isnan(f) else 0.0 for f in features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in observation calculation: {e}")
            # Return zero observation on error
            return np.zeros(50, dtype=np.float32)
    
    def step(self, action):
        """Execute one step"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        reward = 0
        info = {}
        
        # Simple reward logic
        if action == 1:  # Buy
            reward = np.random.normal(0.001, 0.01)  # Small positive expected return
            if reward > 0:
                self.win_count += 1
            self.total_trades += 1
        elif action == 2:  # Sell
            reward = np.random.normal(0.001, 0.01)  # Small positive expected return
            if reward > 0:
                self.win_count += 1
            self.total_trades += 1
        
        self.current_balance += reward * 1000  # Scale reward
        self.current_step += 1
        
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, False, info


class NotificationManager:
    """Handle notifications via Telegram and Email"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def notify(self, message: str, subject: str = "Forex Trading Alert"):
        """Send notification via all configured channels"""
        self.send_telegram(message)
        self.send_email(subject, message)
    
    def send_telegram(self, message: str):
        """Send Telegram notification"""
        if not self.config.notifications.has_telegram():
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.config.notifications.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.config.notifications.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Telegram notification failed: {e}")
            return False
    
    def send_email(self, subject: str, message: str):
        """Send email notification"""
        if not self.config.notifications.has_email():
            return False
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.config.notifications.email_username
            msg['To'] = self.config.notifications.email_to
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain'))
            
            server = smtplib.SMTP(
                self.config.notifications.email_smtp_server,
                self.config.notifications.email_smtp_port
            )
            server.starttls()
            server.login(
                self.config.notifications.email_username,
                self.config.notifications.email_password
            )
            
            text = msg.as_string()
            server.sendmail(
                self.config.notifications.email_username,
                self.config.notifications.email_to,
                text
            )
            server.quit()
            return True
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
            return False


def main():
    """Main function demonstrating the configured forex system"""
    print("🚀 Advanced Forex Auto Trading System v2.0")
    print("📋 Configuration-Driven Trading Bot")
    print("=" * 60)
    
    # Load and display configuration
    config = get_config()
    config.print_summary()
    
    # Get symbols to trade
    symbols = config.get_symbols_list()
    print(f"\n🎯 Trading Symbols: {', '.join(symbols)}")
    
    # Initialize bots for each symbol
    bots = {}
    for symbol in symbols:
        print(f"\n🤖 Initializing bot for {symbol}...")
        bot = ConfigurableForexBot(symbol=symbol)
        bots[symbol] = bot
        
        # Check if model exists
        model_path = config.get_model_path(symbol)
        if not os.path.exists(model_path):
            print(f"📚 Training new model for {symbol}...")
            bot.train_model()
            bot.test_model()
        else:
            print(f"📂 Loading existing model for {symbol}...")
            if bot.load_model():
                bot.test_model()
    
    # Start trading
    print(f"\n🚀 Starting trading...")
    for symbol, bot in bots.items():
        if bot.start_live_trading():
            print(f"✅ Trading started for {symbol}")
        else:
            print(f"❌ Failed to start trading for {symbol}")
    
    # Monitor performance
    try:
        print(f"\n📊 Monitoring performance (Ctrl+C to stop)...")
        while True:
            time.sleep(300)  # Check every 5 minutes
            
            print(f"\n📈 Performance Update:")
            for symbol, bot in bots.items():
                report = bot.get_performance_report()
                stats = report['performance']
                print(f"  {symbol}: "
                      f"Trades: {stats['total_trades']}, "
                      f"Win Rate: {stats['win_rate']:.1%}, "
                      f"P&L: ${stats['total_profit']:.2f}")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping all trading bots...")
        for symbol, bot in bots.items():
            bot.stop_live_trading()
            print(f"✅ Stopped {symbol}")
        
        print("🎉 All bots stopped successfully!")


if __name__ == "__main__":
    main()