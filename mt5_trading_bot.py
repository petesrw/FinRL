#!/usr/bin/env python3
"""
🤖 MT5 Trading Bot with Trained RL Model
Loads trained models and executes real trades on MT5
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from stable_baselines3 import PPO, SAC, A2C
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    """Load and validate trained RL models"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.model_info = None
        
    def list_available_models(self, tier=None):
        """List all available trained models"""
        models = []
        
        tiers = ['diamond', 'gold', 'silver', 'bronze'] if tier is None else [tier]
        
        for tier_name in tiers:
            tier_dir = os.path.join(self.models_dir, tier_name)
            if os.path.exists(tier_dir):
                for file in os.listdir(tier_dir):
                    if file.endswith('.zip'):
                        model_path = os.path.join(tier_dir, file)
                        info_path = model_path.replace('.zip', '_info.json')
                        
                        if os.path.exists(info_path):
                            with open(info_path, 'r') as f:
                                info = json.load(f)
                            
                            models.append({
                                'tier': tier_name,
                                'model_path': model_path,
                                'info': info,
                                'score': info.get('score', 0),
                                'symbol': info.get('symbol', 'Unknown'),
                                'timestamp': info.get('timestamp', '')
                            })
        
        # Sort by score (best first)
        models.sort(key=lambda x: x['score'], reverse=True)
        return models
    
    def load_best_model(self, symbol=None, min_score=60):
        """Load the best available model for a symbol"""
        models = self.list_available_models()
        
        if symbol:
            models = [m for m in models if m['symbol'].upper() == symbol.upper()]
        
        models = [m for m in models if m['score'] >= min_score]
        
        if not models:
            raise ValueError(f"No suitable models found for {symbol} with score >= {min_score}")
        
        best_model = models[0]
        print(f"🏆 Loading best model:")
        print(f"   📈 Symbol: {best_model['symbol']}")
        print(f"   💎 Tier: {best_model['tier'].upper()}")
        print(f"   📊 Score: {best_model['score']:.1f}")
        print(f"   📅 Date: {best_model['timestamp']}")
        
        return self.load_model(best_model['model_path'])
    
    def load_model(self, model_path):
        """Load a specific model from file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        info_path = model_path.replace('.zip', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
        
        # Determine algorithm from filename or info
        if '_ppo_' in model_path.lower() or (self.model_info and self.model_info.get('algorithm') == 'PPO'):
            self.model = PPO.load(model_path)
            print(f"✅ Loaded PPO model from: {model_path}")
        elif '_sac_' in model_path.lower() or (self.model_info and self.model_info.get('algorithm') == 'SAC'):
            self.model = SAC.load(model_path)
            print(f"✅ Loaded SAC model from: {model_path}")
        elif '_a2c_' in model_path.lower() or (self.model_info and self.model_info.get('algorithm') == 'A2C'):
            self.model = A2C.load(model_path)
            print(f"✅ Loaded A2C model from: {model_path}")
        else:
            # Try to load as PPO by default
            try:
                self.model = PPO.load(model_path)
                print(f"✅ Loaded model as PPO from: {model_path}")
            except:
                raise ValueError(f"Unable to determine model algorithm for: {model_path}")
        
        return self.model

class MT5Interface:
    """Interface for MetaTrader 5 operations"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        
    def connect(self, login=None, password=None, server=None):
        """Connect to MT5"""
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        if login and password and server:
            if not mt5.login(login, password=password, server=server):
                print(f"❌ MT5 login failed: {mt5.last_error()}")
                return False
        
        self.account_info = mt5.account_info()
        if self.account_info is None:
            print(f"❌ Failed to get account info: {mt5.last_error()}")
            return False
        
        self.connected = True
        print(f"✅ Connected to MT5")
        print(f"   💰 Account: {self.account_info.login}")
        print(f"   💳 Balance: ${self.account_info.balance:.2f}")
        print(f"   📊 Equity: ${self.account_info.equity:.2f}")
        print(f"   🏢 Server: {self.account_info.server}")
        
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        print("🔌 Disconnected from MT5")
    
    def get_symbol_info(self, symbol):
        """Get symbol information"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Failed to get symbol info for {symbol}")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Failed to select symbol {symbol}")
                return None
        
        return symbol_info
    
    def get_rates(self, symbol, timeframe, count=1000):
        """Get historical rates"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"❌ Failed to get rates for {symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        
        return df
    
    def send_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment="RL Bot"):
        """Send trading order"""
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        if price is None:
            if order_type == mt5.ORDER_TYPE_BUY:
                price = mt5.symbol_info_tick(symbol).ask
            else:
                price = mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result
    
    def close_position(self, ticket):
        """Close specific position"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return None
        
        position = positions[0]
        
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "magic": 123456,
            "comment": "RL Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result
    
    def get_positions(self, symbol=None):
        """Get current positions"""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        return positions

class TradingBot:
    """Main trading bot using trained RL model"""
    
    def __init__(self, symbol, model_path=None, risk_percent=1.0):
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.model_loader = ModelLoader()
        self.mt5 = MT5Interface()
        self.model = None
        self.running = False
        
        # Trading parameters
        self.min_confidence = 0.6  # Minimum confidence for trade execution
        self.max_positions = 1     # Maximum concurrent positions
        self.lookback_window = 50  # Must match training
        
        # Load model
        if model_path:
            self.model = self.model_loader.load_model(model_path)
        else:
            self.model = self.model_loader.load_best_model(symbol)
        
        # Get model parameters from info
        if self.model_loader.model_info:
            self.lookback_window = self.model_loader.model_info.get('lookback_window', 50)
            self.transaction_cost = self.model_loader.model_info.get('transaction_cost', 0.0)
    
    def connect_mt5(self, login=None, password=None, server=None):
        """Connect to MT5"""
        return self.mt5.connect(login, password, server)
    
    def calculate_position_size(self):
        """Calculate position size based on risk management"""
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01  # Minimum lot size
        
        balance = account_info.equity
        risk_amount = balance * (self.risk_percent / 100)
        
        # Get symbol info for pip value calculation
        symbol_info = self.mt5.get_symbol_info(self.symbol)
        if symbol_info is None:
            return 0.01
        
        # Simplified position sizing (you may want to enhance this)
        # This assumes 1% risk with 100 pip stop loss
        pip_value = symbol_info.trade_tick_value
        lot_size = risk_amount / (100 * pip_value)
        
        # Ensure minimum lot size and step
        min_lot = symbol_info.volume_min
        lot_step = symbol_info.volume_step
        
        lot_size = max(min_lot, round(lot_size / lot_step) * lot_step)
        lot_size = min(lot_size, symbol_info.volume_max)
        
        return lot_size
    
    def calculate_indicators(self, df):
        """Calculate technical indicators (must match training)"""
        # Moving Averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
    
    def prepare_observation(self, df, current_position=0):
        """Prepare observation for model (must match training format)"""
        # Get required columns (must match training)
        feature_cols = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi', 'macd', 
                       'macd_signal', 'bb_upper', 'bb_lower', 'atr']
        
        # Get last lookback_window rows
        obs_data = df[feature_cols].tail(self.lookback_window).values
        
        # Handle insufficient data
        if len(obs_data) < self.lookback_window:
            padding = np.tile(obs_data[0], (self.lookback_window - len(obs_data), 1))
            obs_data = np.vstack([padding, obs_data])
        
        # Add position information (must match training)
        position_info = np.full((obs_data.shape[0], 1), current_position)
        obs_data = np.hstack([obs_data, position_info])
        
        # Normalize data (except position)
        obs_data[:, :-1] = (obs_data[:, :-1] - np.mean(obs_data[:, :-1], axis=0)) / (np.std(obs_data[:, :-1], axis=0) + 1e-8)
        
        return obs_data.astype(np.float32)
    
    def predict_action(self, observation):
        """Get action prediction from model"""
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Convert continuous action to discrete
        if isinstance(action, (list, np.ndarray)):
            action_value = float(action[0])
        else:
            action_value = float(action)
        
        # Convert to discrete action (must match training logic)
        if action_value < -0.3:
            discrete_action = 2  # Sell
            confidence = abs(action_value + 0.65) / 0.7  # Scale confidence
        elif action_value < 0.3:
            discrete_action = 0  # Hold
            confidence = 1.0 - abs(action_value) / 0.3
        elif action_value < 0.7:
            discrete_action = 1  # Buy
            confidence = (action_value - 0.3) / 0.4
        else:
            discrete_action = 3  # Close
            confidence = (action_value - 0.7) / 0.3
        
        return discrete_action, confidence, action_value
    
    def execute_trade(self, action, confidence):
        """Execute trade based on model prediction"""
        if confidence < self.min_confidence:
            print(f"   ⚠️ Low confidence ({confidence:.2f}), skipping trade")
            return None
        
        # Check current positions
        positions = self.mt5.get_positions(self.symbol)
        current_position = 0
        if positions:
            if len(positions) >= self.max_positions:
                print(f"   ⚠️ Maximum positions ({self.max_positions}) reached")
                return None
            current_position = 1 if positions[0].type == mt5.POSITION_TYPE_BUY else -1
        
        # Calculate position size and SL/TP
        volume = self.calculate_position_size()
        symbol_info = self.mt5.get_symbol_info(self.symbol)
        current_price = mt5.symbol_info_tick(self.symbol)
        
        if symbol_info is None or current_price is None:
            print(f"   ❌ Failed to get symbol/price info")
            return None
        
        # Calculate SL/TP based on ATR or fixed percentage
        atr_multiplier = 2.0
        sl_distance = symbol_info.point * 100  # Default 100 points
        tp_distance = sl_distance * 1.5  # 1:1.5 risk/reward
        
        result = None
        
        if action == 1 and current_position == 0:  # Buy signal
            sl = current_price.ask - sl_distance
            tp = current_price.ask + tp_distance
            result = self.mt5.send_order(
                self.symbol, 
                mt5.ORDER_TYPE_BUY, 
                volume, 
                sl=sl, 
                tp=tp,
                comment=f"RL Buy C:{confidence:.2f}"
            )
            
        elif action == 2 and current_position == 0:  # Sell signal
            sl = current_price.bid + sl_distance
            tp = current_price.bid - tp_distance
            result = self.mt5.send_order(
                self.symbol, 
                mt5.ORDER_TYPE_SELL, 
                volume,
                sl=sl,
                tp=tp, 
                comment=f"RL Sell C:{confidence:.2f}"
            )
            
        elif action == 3 and current_position != 0:  # Close signal
            if positions:
                result = self.mt5.close_position(positions[0].ticket)
        
        return result
    
    def run_trading_loop(self, interval_minutes=5):
        """Main trading loop"""
        if not self.mt5.connected:
            print("❌ MT5 not connected")
            return
        
        print(f"🤖 Starting trading bot for {self.symbol}")
        print(f"   📊 Model: {self.model_loader.model_info.get('tier', 'Unknown').upper()} tier")
        print(f"   💰 Risk per trade: {self.risk_percent}%")
        print(f"   ⏰ Check interval: {interval_minutes} minutes")
        print(f"   🎯 Min confidence: {self.min_confidence}")
        
        self.running = True
        
        try:
            while self.running:
                # Get latest market data
                df = self.mt5.get_rates(self.symbol, mt5.TIMEFRAME_M5, 200)
                if df is None:
                    print(f"❌ Failed to get market data")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                df = df.dropna()
                
                if len(df) < self.lookback_window:
                    print(f"⚠️ Insufficient data ({len(df)} < {self.lookback_window})")
                    time.sleep(60)
                    continue
                
                # Get current position
                positions = self.mt5.get_positions(self.symbol)
                current_position = 0
                if positions:
                    current_position = 1 if positions[0].type == mt5.POSITION_TYPE_BUY else -1
                
                # Prepare observation
                observation = self.prepare_observation(df, current_position)
                
                # Get model prediction
                action, confidence, raw_action = self.predict_action(observation)
                
                # Log prediction
                action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
                current_time = datetime.now().strftime('%H:%M:%S')
                current_price = mt5.symbol_info_tick(self.symbol).bid
                
                print(f"\n🕐 {current_time} | {self.symbol} @ {current_price:.5f}")
                print(f"   🤖 Prediction: {action_names[action]} (confidence: {confidence:.2f}, raw: {raw_action:.3f})")
                print(f"   📊 Position: {current_position} | Positions: {len(positions) if positions else 0}")
                
                # Execute trade if conditions are met
                if action != 0:  # Not hold
                    result = self.execute_trade(action, confidence)
                    if result is not None:
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"   ✅ Trade executed: {action_names[action]}")
                            print(f"      📊 Order: {result.order}")
                            print(f"      💰 Volume: {result.volume}")
                            print(f"      💵 Price: {result.price:.5f}")
                        else:
                            print(f"   ❌ Trade failed: {result.retcode} - {result.comment}")
                
                # Wait for next check
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Trading bot stopped by user")
        except Exception as e:
            print(f"\n❌ Trading bot error: {e}")
        finally:
            self.running = False
            print("🔌 Trading bot shutdown")
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        self.mt5.disconnect()

def main():
    """Example usage"""
    print("🤖 RL Trading Bot for MT5")
    print("=" * 50)
    
    # Configuration
    SYMBOL = "XAUUSD"  # Change to your symbol
    RISK_PERCENT = 1.0  # Risk per trade in %
    CHECK_INTERVAL = 5  # Minutes between checks
    
    # MT5 Login credentials (set these)
    LOGIN = None  # Your MT5 login
    PASSWORD = None  # Your MT5 password  
    SERVER = None  # Your MT5 server
    
    try:
        # Initialize bot
        bot = TradingBot(SYMBOL, risk_percent=RISK_PERCENT)
        
        # Connect to MT5
        if not bot.connect_mt5(LOGIN, PASSWORD, SERVER):
            print("❌ Failed to connect to MT5")
            return
        
        # Start trading
        bot.run_trading_loop(CHECK_INTERVAL)
        
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        if 'bot' in locals():
            bot.stop()

if __name__ == "__main__":
    main()
