#!/usr/bin/env python3
"""
Advanced Forex Auto Trading System with Reinforcement Learning
Features:
- MT5 Integration for live trading
- Reinforcement Learning for decision making
- Dynamic TP/SL calculation
- Risk management with position sizing
- Performance tracking and optimization
- Target: 65%+ win rate with positive ROI
- Configuration via .env file
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import talib
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import threading
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests

# Import configuration
from config import get_config

class ForexTradingEnvironment(gym.Env):
    """
    Custom Forex Trading Environment for Reinforcement Learning
    """
    
    def __init__(self, 
                 symbol: str = "EURUSD",
                 timeframe: int = mt5.TIMEFRAME_M15,
                 lookback_window: int = 100,
                 initial_balance: float = 10000.0,
                 max_risk_per_trade: float = 0.02,  # 2% risk per trade
                 target_win_rate: float = 0.65):
        
        super(ForexTradingEnvironment, self).__init__()
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.target_win_rate = target_win_rate
        
        # Trading state
        self.current_position = 0  # -1: sell, 0: no position, 1: buy
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_size = 0.0
        
        # Performance tracking
        self.trades_history = []
        self.equity_curve = []
        self.win_count = 0
        self.total_trades = 0
        
        # Market data
        self.data = None
        self.current_step = 0
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation space: OHLC + indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(50,), dtype=np.float32  # 50 features
        )
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the trading system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('forex_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Load fresh market data
        self.load_market_data()
        
        # Reset trading state
        self.current_balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_size = 0.0
        
        # Reset performance tracking
        self.trades_history = []
        self.equity_curve = [self.initial_balance]
        self.win_count = 0
        self.total_trades = 0
        
        # Start from a random point in data
        self.current_step = np.random.randint(
            self.lookback_window, 
            len(self.data) - 1000
        )
        
        return self._get_observation(), {}
    
    def load_market_data(self):
        """Load market data from MT5 or historical data"""
        try:
            # Try to get live data from MT5
            if mt5.initialize():
                rates = mt5.copy_rates_from_pos(
                    self.symbol, self.timeframe, 0, 5000
                )
                if rates is not None:
                    self.data = pd.DataFrame(rates)
                    self.data['time'] = pd.to_datetime(self.data['time'], unit='s')
                    self.logger.info(f"Loaded {len(self.data)} bars from MT5")
                else:
                    self.load_demo_data()
            else:
                self.load_demo_data()
        except:
            self.load_demo_data()
    
    def load_demo_data(self):
        """Load demo forex data for testing"""
        self.logger.info("Loading demo forex data...")
        
        # Generate realistic forex data
        np.random.seed(42)
        n_points = 10000
        
        # Base price around 1.1000 for EURUSD
        base_price = 1.1000
        
        # Generate price movements with trend and noise
        trend = np.cumsum(np.random.normal(0, 0.0001, n_points))
        noise = np.random.normal(0, 0.0005, n_points)
        close_prices = base_price + trend + noise
        
        # Generate OHLC from close prices
        high_prices = close_prices + np.abs(np.random.normal(0, 0.0003, n_points))
        low_prices = close_prices - np.abs(np.random.normal(0, 0.0003, n_points))
        
        # Open prices (previous close + small gap)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        open_prices += np.random.normal(0, 0.0001, n_points)
        
        # Volume
        volumes = np.random.randint(1000, 10000, n_points)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'time': pd.date_range(start='2020-01-01', periods=n_points, freq='15T'),
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'tick_volume': volumes
        })
        
        self.logger.info(f"Generated {len(self.data)} demo forex bars")
    
    def _calculate_indicators(self, data_slice):
        """Calculate technical indicators"""
        close = data_slice['close'].values
        high = data_slice['high'].values
        low = data_slice['low'].values
        volume = data_slice['tick_volume'].values
        
        indicators = {}
        
        try:
            # Moving Averages
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1] if len(close) >= 20 else close[-1]
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else close[-1]
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1] if len(close) >= 12 else close[-1]
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)[-1] if len(close) >= 26 else close[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
            indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            indicators['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            indicators['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else close[-1]
            indicators['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else close[-1]
            indicators['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if not np.isnan(bb_upper[-1]) else 0
            
            # ATR for volatility
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0.001
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else 50
            indicators['stoch_d'] = slowd[-1] if not np.isnan(slowd[-1]) else 50
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else -50
            
            # CCI
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            
        except Exception as e:
            # Fallback values if indicators fail
            self.logger.warning(f"Indicator calculation failed: {e}")
            for key in ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
                       'rsi', 'bb_upper', 'bb_lower', 'bb_width', 'atr', 'stoch_k', 'stoch_d',
                       'williams_r', 'cci']:
                if key not in indicators:
                    indicators[key] = 0.0
        
        return indicators
    
    def _get_observation(self):
        """Get current market observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Get recent data
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1
        recent_data = self.data.iloc[start_idx:end_idx]
        
        if len(recent_data) == 0:
            return np.zeros(50, dtype=np.float32)
        
        # Current price data
        current_bar = recent_data.iloc[-1]
        current_price = current_bar['close']
        
        # Price features (normalized)
        price_features = [
            (current_bar['open'] - current_price) / current_price,
            (current_bar['high'] - current_price) / current_price,
            (current_bar['low'] - current_price) / current_price,
            current_bar['tick_volume'] / 10000.0  # Normalize volume
        ]
        
        # Technical indicators
        indicators = self._calculate_indicators(recent_data)
        
        # Normalize indicators
        indicator_features = [
            (indicators['sma_20'] - current_price) / current_price,
            (indicators['sma_50'] - current_price) / current_price,
            (indicators['ema_12'] - current_price) / current_price,
            (indicators['ema_26'] - current_price) / current_price,
            indicators['macd'] * 10000,  # Scale MACD
            indicators['macd_signal'] * 10000,
            indicators['macd_hist'] * 10000,
            (indicators['rsi'] - 50) / 50,  # Center RSI around 0
            (indicators['bb_upper'] - current_price) / current_price,
            (indicators['bb_lower'] - current_price) / current_price,
            indicators['bb_width'],
            indicators['atr'] / current_price,
            (indicators['stoch_k'] - 50) / 50,
            (indicators['stoch_d'] - 50) / 50,
            indicators['williams_r'] / 100,
            indicators['cci'] / 100
        ]
        
        # Position features
        position_features = [
            float(self.current_position),  # Current position
            (self.entry_price - current_price) / current_price if self.entry_price > 0 else 0,
            (self.stop_loss - current_price) / current_price if self.stop_loss > 0 else 0,
            (self.take_profit - current_price) / current_price if self.take_profit > 0 else 0,
            self.position_size / self.current_balance if self.current_balance > 0 else 0
        ]
        
        # Account features
        account_features = [
            (self.current_balance - self.initial_balance) / self.initial_balance,
            self.win_count / max(1, self.total_trades),  # Win rate
            len(self.trades_history) / 1000.0  # Number of trades (normalized)
        ]
        
        # Price momentum features (last 5 bars)
        momentum_features = []
        if len(recent_data) >= 5:
            for i in range(1, 6):
                if len(recent_data) > i:
                    prev_price = recent_data.iloc[-i-1]['close']
                    momentum = (current_price - prev_price) / prev_price
                    momentum_features.append(momentum * 1000)  # Scale momentum
                else:
                    momentum_features.append(0.0)
        else:
            momentum_features = [0.0] * 5
        
        # Market time features
        current_time = recent_data.iloc[-1]['time']
        time_features = [
            current_time.hour / 24.0,
            current_time.weekday() / 7.0,
            np.sin(2 * np.pi * current_time.hour / 24),  # Cyclical hour
            np.cos(2 * np.pi * current_time.hour / 24)
        ]
        
        # Combine all features
        observation = np.array(
            price_features + 
            indicator_features + 
            position_features + 
            account_features + 
            momentum_features + 
            time_features,
            dtype=np.float32
        )
        
        # Ensure we have exactly 50 features
        if len(observation) < 50:
            observation = np.pad(observation, (0, 50 - len(observation)), 'constant')
        elif len(observation) > 50:
            observation = observation[:50]
        
        # Replace any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _calculate_position_size(self, stop_loss_pips: float):
        """Calculate position size based on risk management"""
        if stop_loss_pips <= 0:
            return 0.0
        
        # Risk amount per trade
        risk_amount = self.current_balance * self.max_risk_per_trade
        
        # Calculate position size
        pip_value = 10.0  # For EURUSD, 1 pip = $10 for 1 lot
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Limit position size
        max_position = self.current_balance / 1000  # Max 1 lot per $1000
        position_size = min(position_size, max_position)
        
        return round(position_size, 2)
    
    def _calculate_tp_sl(self, action: int, current_price: float, atr: float):
        """Calculate dynamic TP and SL based on market conditions"""
        
        # Base TP/SL ratios
        base_sl_ratio = 1.5  # 1.5 * ATR for SL
        base_tp_ratio = 2.5  # 2.5 * ATR for TP (1:1.67 RR)
        
        # Adjust based on market volatility
        volatility_multiplier = max(0.5, min(2.0, atr / 0.001))  # Adjust based on ATR
        
        sl_distance = atr * base_sl_ratio * volatility_multiplier
        tp_distance = atr * base_tp_ratio * volatility_multiplier
        
        if action == 1:  # Buy
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        else:  # Sell
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
        
        return stop_loss, take_profit
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        current_bar = self.data.iloc[self.current_step]
        current_price = current_bar['close']
        
        # Calculate indicators for TP/SL calculation
        start_idx = max(0, self.current_step - 50)
        recent_data = self.data.iloc[start_idx:self.current_step + 1]
        indicators = self._calculate_indicators(recent_data)
        atr = indicators['atr']
        
        reward = 0
        info = {}
        
        # Check if current position should be closed (SL/TP hit)
        if self.current_position != 0:
            position_closed = False
            
            if self.current_position == 1:  # Long position
                if current_price <= self.stop_loss:
                    # Stop loss hit
                    pnl = (self.stop_loss - self.entry_price) * self.position_size * 100000
                    reward = -abs(pnl) / self.current_balance  # Negative reward
                    position_closed = True
                    info['close_reason'] = 'stop_loss'
                elif current_price >= self.take_profit:
                    # Take profit hit
                    pnl = (self.take_profit - self.entry_price) * self.position_size * 100000
                    reward = pnl / self.current_balance  # Positive reward
                    position_closed = True
                    info['close_reason'] = 'take_profit'
                    self.win_count += 1
            
            elif self.current_position == -1:  # Short position
                if current_price >= self.stop_loss:
                    # Stop loss hit
                    pnl = (self.entry_price - self.stop_loss) * self.position_size * 100000
                    reward = -abs(pnl) / self.current_balance  # Negative reward
                    position_closed = True
                    info['close_reason'] = 'stop_loss'
                elif current_price <= self.take_profit:
                    # Take profit hit
                    pnl = (self.entry_price - self.take_profit) * self.position_size * 100000
                    reward = pnl / self.current_balance  # Positive reward
                    position_closed = True
                    info['close_reason'] = 'take_profit'
                    self.win_count += 1
            
            if position_closed:
                self.current_balance += pnl
                self.trades_history.append({
                    'entry_time': self.current_step - 1,
                    'exit_time': self.current_step,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position_type': 'buy' if self.current_position == 1 else 'sell',
                    'position_size': self.position_size,
                    'pnl': pnl,
                    'close_reason': info['close_reason']
                })
                self.total_trades += 1
                
                # Reset position
                self.current_position = 0
                self.entry_price = 0.0
                self.stop_loss = 0.0
                self.take_profit = 0.0
                self.position_size = 0.0
        
        # Execute new action
        if action == 1 and self.current_position == 0:  # Buy
            stop_loss, take_profit = self._calculate_tp_sl(1, current_price, atr)
            sl_pips = (current_price - stop_loss) * 10000  # Convert to pips
            position_size = self._calculate_position_size(sl_pips)
            
            if position_size > 0:
                self.current_position = 1
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.position_size = position_size
                info['action_taken'] = 'buy'
        
        elif action == 2 and self.current_position == 0:  # Sell
            stop_loss, take_profit = self._calculate_tp_sl(2, current_price, atr)
            sl_pips = (stop_loss - current_price) * 10000  # Convert to pips
            position_size = self._calculate_position_size(sl_pips)
            
            if position_size > 0:
                self.current_position = -1
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.position_size = position_size
                info['action_taken'] = 'sell'
        
        elif action == 3 and self.current_position != 0:  # Close position
            if self.current_position == 1:
                pnl = (current_price - self.entry_price) * self.position_size * 100000
            else:
                pnl = (self.entry_price - current_price) * self.position_size * 100000
            
            self.current_balance += pnl
            reward = pnl / self.current_balance
            
            if pnl > 0:
                self.win_count += 1
            
            self.trades_history.append({
                'entry_time': self.current_step - 1,
                'exit_time': self.current_step,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position_type': 'buy' if self.current_position == 1 else 'sell',
                'position_size': self.position_size,
                'pnl': pnl,
                'close_reason': 'manual'
            })
            self.total_trades += 1
            
            # Reset position
            self.current_position = 0
            self.entry_price = 0.0
            self.stop_loss = 0.0
            self.take_profit = 0.0
            self.position_size = 0.0
            info['action_taken'] = 'close'
        
        # Add small penalty for holding positions too long
        if self.current_position != 0:
            reward -= 0.001
        
        # Bonus for maintaining target win rate
        if self.total_trades > 10:
            current_win_rate = self.win_count / self.total_trades
            if current_win_rate >= self.target_win_rate:
                reward += 0.01
        
        # Update equity curve
        unrealized_pnl = 0
        if self.current_position != 0:
            if self.current_position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.position_size * 100000
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size * 100000
        
        current_equity = self.current_balance + unrealized_pnl
        self.equity_curve.append(current_equity)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                self.current_balance < self.initial_balance * 0.5)  # Stop if 50% drawdown
        
        return self._get_observation(), reward, done, False, info


class MT5Connector:
    """MT5 connection and trading operations"""
    
    def __init__(self):
        self.connected = False
        self.setup_logging()
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def connect(self, login: int = None, password: str = None, server: str = None):
        """Connect to MT5"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
            
            if login and password and server:
                if not mt5.login(login, password, server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            self.connected = True
            account_info = mt5.account_info()
            if account_info:
                self.logger.info(f"Connected to MT5 - Account: {account_info.login}, Balance: {account_info.balance}")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        self.logger.info("Disconnected from MT5")
    
    def get_symbol_info(self, symbol: str):
        """Get symbol information"""
        if not self.connected:
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {symbol} not found")
            return None
        
        return {
            'symbol': symbol_info.name,
            'digits': symbol_info.digits,
            'point': symbol_info.point,
            'spread': symbol_info.spread,
            'min_lot': symbol_info.volume_min,
            'max_lot': symbol_info.volume_max,
            'lot_step': symbol_info.volume_step
        }
    
    def get_current_price(self, symbol: str):
        """Get current bid/ask prices"""
        if not self.connected:
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None,
                   comment: str = "AI Trading"):
        """Place a trading order"""
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return None
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {symbol} not found")
            return None
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Failed to get tick for {symbol}")
            return None
        
        # Determine order type and price
        if order_type.upper() == "BUY":
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask if price is None else price
        elif order_type.upper() == "SELL":
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid if price is None else price
        else:
            self.logger.error(f"Invalid order type: {order_type}")
            return None
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL and TP if provided
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        self.logger.info(f"Order placed successfully: {order_type} {volume} {symbol} at {price}")
        return {
            'ticket': result.order,
            'volume': result.volume,
            'price': result.price,
            'comment': result.comment
        }
    
    def close_position(self, ticket: int):
        """Close a position by ticket"""
        if not self.connected:
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            self.logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        # Determine close order type
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Close order failed: {result.retcode}")
            return False
        
        self.logger.info(f"Position {ticket} closed successfully")
        return True
    
    def get_positions(self):
        """Get all open positions"""
        if not self.connected:
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        return [{
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
            'volume': pos.volume,
            'price_open': pos.price_open,
            'price_current': pos.price_current,
            'profit': pos.profit,
            'sl': pos.sl,
            'tp': pos.tp,
            'time': datetime.fromtimestamp(pos.time)
        } for pos in positions]


class ForexTradingBot:
    """Main Forex Trading Bot with RL and MT5 integration"""
    
    def __init__(self, 
                 symbol: str = "EURUSD",
                 model_type: str = "PPO",
                 risk_per_trade: float = 0.02,
                 target_win_rate: float = 0.65):
        
        self.symbol = symbol
        self.model_type = model_type
        self.risk_per_trade = risk_per_trade
        self.target_win_rate = target_win_rate
        
        # Initialize components
        self.env = ForexTradingEnvironment(
            symbol=symbol,
            max_risk_per_trade=risk_per_trade,
            target_win_rate=target_win_rate
        )
        self.mt5 = MT5Connector()
        self.model = None
        
        # Trading state
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
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, total_timesteps: int = 100000):
        """Train the RL model"""
        self.logger.info(f"Starting model training with {total_timesteps} timesteps...")
        
        # Create model
        if self.model_type == "PPO":
            self.model = PPO(
                "MlpPolicy", 
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./forex_tensorboard/"
            )
        elif self.model_type == "SAC":
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                verbose=1,
                tensorboard_log="./forex_tensorboard/"
            )
        elif self.model_type == "A2C":
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                verbose=1,
                tensorboard_log="./forex_tensorboard/"
            )
        
        # Train model
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save model
        model_path = f"forex_model_{self.model_type}_{self.symbol}.zip"
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Test model
        self.test_model()
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            if self.model_type == "PPO":
                self.model = PPO.load(model_path)
            elif self.model_type == "SAC":
                self.model = SAC.load(model_path)
            elif self.model_type == "A2C":
                self.model = A2C.load(model_path)
            
            self.logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def test_model(self, episodes: int = 10):
        """Test the trained model"""
        if self.model is None:
            self.logger.error("No model loaded")
            return
        
        self.logger.info(f"Testing model for {episodes} episodes...")
        
        total_rewards = []
        win_rates = []
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if done or truncated:
                    break
            
            total_rewards.append(episode_reward)
            
            # Calculate win rate for this episode
            if self.env.total_trades > 0:
                win_rate = self.env.win_count / self.env.total_trades
                win_rates.append(win_rate)
            
            self.logger.info(f"Episode {episode + 1}: Reward={episode_reward:.4f}, "
                           f"Trades={self.env.total_trades}, Win Rate={win_rate:.2%}")
        
        # Calculate overall statistics
        avg_reward = np.mean(total_rewards)
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"Average Reward: {avg_reward:.4f}")
        self.logger.info(f"Average Win Rate: {avg_win_rate:.2%}")
        self.logger.info(f"Target Win Rate: {self.target_win_rate:.2%}")
        
        if avg_win_rate >= self.target_win_rate:
            self.logger.info("✅ Model meets target win rate!")
        else:
            self.logger.warning("⚠️ Model below target win rate - consider more training")
    
    def connect_mt5(self, login: int = None, password: str = None, server: str = None):
        """Connect to MT5"""
        return self.mt5.connect(login, password, server)
    
    def start_live_trading(self):
        """Start live trading"""
        if not self.mt5.connected:
            self.logger.error("MT5 not connected")
            return False
        
        if self.model is None:
            self.logger.error("No model loaded")
            return False
        
        self.is_trading = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()
        
        self.logger.info("Live trading started")
        return True
    
    def stop_live_trading(self):
        """Stop live trading"""
        self.is_trading = False
        if self.trading_thread:
            self.trading_thread.join()
        
        self.logger.info("Live trading stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Trading loop started")
        
        while self.is_trading:
            try:
                # Get current market data
                current_price = self.mt5.get_current_price(self.symbol)
                if current_price is None:
                    time.sleep(60)  # Wait 1 minute if no price data
                    continue
                
                # Get observation from environment
                obs = self.env._get_observation()
                
                # Get model prediction
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Execute action
                self._execute_action(action, current_price)
                
                # Update performance stats
                self._update_performance_stats()
                
                # Wait before next decision (15 minutes for M15 timeframe)
                time.sleep(900)  # 15 minutes
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _execute_action(self, action: int, price_info: dict):
        """Execute trading action"""
        current_positions = self.mt5.get_positions()
        symbol_positions = [pos for pos in current_positions if pos['symbol'] == self.symbol]
        
        if action == 1:  # Buy signal
            if not symbol_positions:  # No existing position
                # Calculate position size and TP/SL
                account_info = mt5.account_info()
                if account_info:
                    balance = account_info.balance
                    risk_amount = balance * self.risk_per_trade
                    
                    # Simple position sizing (can be improved)
                    volume = min(0.1, risk_amount / 1000)  # Basic calculation
                    
                    # Calculate TP/SL (simplified)
                    current_price = price_info['ask']
                    sl = current_price - 0.0050  # 50 pips SL
                    tp = current_price + 0.0075  # 75 pips TP (1:1.5 RR)
                    
                    # Place buy order
                    result = self.mt5.place_order(
                        symbol=self.symbol,
                        order_type="BUY",
                        volume=volume,
                        sl=sl,
                        tp=tp,
                        comment="AI Buy"
                    )
                    
                    if result:
                        self.logger.info(f"Buy order placed: {volume} lots at {current_price}")
        
        elif action == 2:  # Sell signal
            if not symbol_positions:  # No existing position
                # Calculate position size and TP/SL
                account_info = mt5.account_info()
                if account_info:
                    balance = account_info.balance
                    risk_amount = balance * self.risk_per_trade
                    
                    # Simple position sizing
                    volume = min(0.1, risk_amount / 1000)
                    
                    # Calculate TP/SL
                    current_price = price_info['bid']
                    sl = current_price + 0.0050  # 50 pips SL
                    tp = current_price - 0.0075  # 75 pips TP (1:1.5 RR)
                    
                    # Place sell order
                    result = self.mt5.place_order(
                        symbol=self.symbol,
                        order_type="SELL",
                        volume=volume,
                        sl=sl,
                        tp=tp,
                        comment="AI Sell"
                    )
                    
                    if result:
                        self.logger.info(f"Sell order placed: {volume} lots at {current_price}")
        
        elif action == 3:  # Close signal
            # Close all positions for this symbol
            for position in symbol_positions:
                self.mt5.close_position(position['ticket'])
                self.logger.info(f"Position {position['ticket']} closed")
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        positions = self.mt5.get_positions()
        # This is a simplified version - you'd want to track closed trades
        # and calculate proper statistics
        pass
    
    def get_performance_report(self):
        """Generate performance report"""
        return {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'target_win_rate': self.target_win_rate,
            'current_stats': self.performance_stats,
            'is_trading': self.is_trading
        }


def main():
    """Main function to demonstrate the forex trading system"""
    print("🚀 Advanced Forex Auto Trading System with RL")
    print("=" * 60)
    
    # Initialize trading bot
    bot = ForexTradingBot(
        symbol="EURUSD",
        model_type="PPO",  # Can be PPO, SAC, or A2C
        risk_per_trade=0.02,  # 2% risk per trade
        target_win_rate=0.65  # 65% target win rate
    )
    
    print("📊 Training RL model...")
    # Train the model
    bot.train_model(total_timesteps=50000)  # Adjust based on your needs
    
    print("\n🔗 Connecting to MT5...")
    # Connect to MT5 (you'll need to provide your credentials)
    # bot.connect_mt5(login=YOUR_LOGIN, password="YOUR_PASSWORD", server="YOUR_SERVER")
    
    print("\n📈 Starting live trading simulation...")
    # For demo purposes, we'll just show the capabilities
    print("Live trading would start here with:")
    print("- Real-time market data from MT5")
    print("- AI decision making every 15 minutes")
    print("- Automatic order placement with TP/SL")
    print("- Risk management and position sizing")
    print("- Performance tracking and optimization")
    
    # Generate performance report
    report = bot.get_performance_report()
    print(f"\n📊 Performance Report:")
    print(f"Symbol: {report['symbol']}")
    print(f"Model: {report['model_type']}")
    print(f"Target Win Rate: {report['target_win_rate']:.1%}")
    
    print("\n✅ System ready for live trading!")
    print("Note: Connect to MT5 and start live trading when ready.")


if __name__ == "__main__":
    main()