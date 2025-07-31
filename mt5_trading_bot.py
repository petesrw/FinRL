#!/usr/bin/env python3
"""
🤖 MT5 Trading Bot with Trained RL Model
Loads trained models and executes real trades on MT5 - Enhanced Multi-Symbol Support
"""

import os
import json
import time
import random
import warnings
import pandas as pd
import numpy as np,sys
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from stable_baselines3 import PPO, SAC, A2C
import warnings
from multi_symbol_config import multi_symbol_config
from multi_account_mt5 import multi_mt5

# monkey-patch ก่อน import โมเดล
sys.modules['numpy._core']              = np.core
sys.modules['numpy._core.multiarray']   = np.core.multiarray
sys.modules['numpy._core.umath']        = np.core.umath
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
        
        # Try to determine algorithm and load model
        algorithm = None
        
        # Check filename first
        if '_ppo_' in model_path.lower():
            algorithm = 'PPO'
        elif '_sac_' in model_path.lower():
            algorithm = 'SAC'
        elif '_a2c_' in model_path.lower():
            algorithm = 'A2C'
        elif self.model_info and self.model_info.get('algorithm'):
            algorithm = self.model_info.get('algorithm')
        
        # Try to load with specific algorithm first, then try all algorithms
        algorithms_to_try = []
        if algorithm:
            algorithms_to_try.append(algorithm)
        
        # Add all algorithms to try if specific one fails
        for alg in ['PPO', 'SAC', 'A2C']:
            if alg not in algorithms_to_try:
                algorithms_to_try.append(alg)
        
        loaded = False
        for alg in algorithms_to_try:
            try:
                if alg == 'PPO':
                    self.model = PPO.load(model_path)
                elif alg == 'SAC':
                    self.model = SAC.load(model_path)
                elif alg == 'A2C':
                    self.model = A2C.load(model_path)
                
                print(f"✅ Loaded {alg} model from: {model_path}")
                loaded = True
                break
            except Exception as e:
                print(f"⚠️ Failed to load as {alg}: {e}")
                continue
        
        if not loaded:
            # Final fallback - try PPO
            try:
                self.model = PPO.load(model_path)
                print(f"✅ Loaded model as PPO from: {model_path} (fallback)")
                loaded = True
            except Exception as e:
                raise ValueError(f"Unable to load model from {model_path}. Tried all algorithms: {e}")
        
        return self.model

class MT5Interface:
    """Enhanced Interface for MetaTrader 5 operations with Multi-Account Support"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.current_symbol = None
        self.multi_account = multi_mt5  # Use global multi-account manager
        
    def connect(self, login=None, password=None, server=None, symbol=None):
        """Connect to MT5 - now uses multi-account system based on symbol"""
        if symbol:
            self.current_symbol = symbol
            success, config = self.multi_account.get_connection_for_symbol(symbol)
            if success:
                self.connected = True
                self.account_info = self.multi_account.get_account_info_for_symbol(symbol)
                if self.account_info:
                    print(f"✅ Connected to MT5 for {symbol}")
                    print(f"   💰 Account: {self.account_info.login}")
                    print(f"   💳 Balance: ${self.account_info.balance:.2f}")
                    print(f"   📊 Equity: ${self.account_info.equity:.2f}")
                    print(f"   🏢 Server: {self.account_info.server}")
                    return True
                else:
                    print(f"❌ Failed to get account info for {symbol}")
                    return False
            else:
                print(f"❌ Failed to connect to MT5 for {symbol}")
                return False
        else:
            # Fallback to original method for backward compatibility
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
            print(f"✅ Connected to MT5 (default)")
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
        """Get symbol information with format detection"""
        # Try different symbol formats
        symbol_variants = [
            symbol,
            f"{symbol}.m",
            f"{symbol}m", 
            f"{symbol}.c",
            f"{symbol}c",
            f"{symbol}.",
            symbol.replace("USD", "usd"),
            symbol.lower()
        ]
        
        for variant in symbol_variants:
            symbol_info = mt5.symbol_info(variant)
            if symbol_info is not None:
                print(f"✅ Found symbol: {variant}")
                # Make sure symbol is visible in Market Watch
                if not mt5.symbol_select(variant, True):
                    print(f"⚠️ Warning: Could not add {variant} to Market Watch")
                return symbol_info, variant
        
        print(f"❌ Symbol {symbol} not found in any format")
        # List available symbols for debugging
        symbols = mt5.symbols_get()
        if symbols:
            forex_symbols = [s.name for s in symbols if 'USD' in s.name or 'EUR' in s.name][:10]
            print(f"💡 Available forex symbols: {forex_symbols}")
        
        return None, symbol

    def get_rates(self, symbol, timeframe, count=1000):
        """Get historical rates with enhanced error handling"""
        print(f"🔍 Getting rates for {symbol}...")
        
        # First, ensure symbol is available
        symbol_info, actual_symbol = self.get_symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Symbol {symbol} not available")
            return None
        
        print(f"✅ Using symbol: {actual_symbol}")
        
        # Check if market is open
        if not symbol_info.trade_mode:
            print(f"⚠️ Trading disabled for {actual_symbol}")
        
        # Check current time and market session
        import datetime
        current_time = datetime.datetime.now()
        print(f"🕐 Current time: {current_time}")
        
        # Try to get rates
        print(f"📊 Requesting {count} bars on timeframe {timeframe}")
        rates = mt5.copy_rates_from_pos(actual_symbol, timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            print(f"❌ Failed to get rates for {actual_symbol}")
            error = mt5.last_error()
            print(f"   Error code: {error}")
            
            # Try alternative timeframes
            alternative_timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]
            for alt_tf in alternative_timeframes:
                if alt_tf != timeframe:
                    print(f"🔄 Trying alternative timeframe {alt_tf}...")
                    rates = mt5.copy_rates_from_pos(actual_symbol, alt_tf, 0, min(count, 100))
                    if rates is not None and len(rates) > 0:
                        print(f"✅ Got {len(rates)} rates with alternative timeframe {alt_tf}")
                        timeframe = alt_tf  # Use the working timeframe
                        break
            
            if rates is None or len(rates) == 0:
                print(f"❌ No market data available for {actual_symbol}")
                # Try to get at least some historical data
                print(f"🔄 Trying to get minimal historical data...")
                rates = mt5.copy_rates_from_pos(actual_symbol, mt5.TIMEFRAME_H1, 0, 10)
                if rates is None or len(rates) == 0:
                    print(f"❌ Cannot get any data for {actual_symbol}")
                    return None
                else:
                    print(f"✅ Got minimal data: {len(rates)} records")
        
        print(f"✅ Retrieved {len(rates)} price records for {actual_symbol}")
        
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
        
        print(f"📈 Latest price: {df['close'].iloc[-1]:.5f}")
        
        return df
    
    def send_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment="RL Bot"):
        """Send trading order using multi-account system"""
        # Use multi-account system to send order
        if hasattr(self, 'multi_account') and self.multi_account:
            print(f"🏦 Using multi-account system for {symbol}")
            result = self.multi_account.send_order_for_symbol(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                sl=sl,
                tp=tp,
                comment=comment
            )
            if result:
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Order sent via multi-account system for {symbol}: Order #{result.order}")
                    return result
                else:
                    print(f"❌ Order failed via multi-account system for {symbol}: {result.retcode}")
                    return result
            else:
                print(f"❌ Failed to send order via multi-account system for {symbol}")
                # Fall back to old method
        
        # Fallback to original method
        print(f"🔄 Using fallback method for {symbol}")
        symbol_info, actual_symbol = self.get_symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Cannot get symbol info for {symbol}")
            return None
        
        if price is None:
            tick = mt5.symbol_info_tick(actual_symbol)
            if tick is None:
                print(f"❌ Cannot get current price for {actual_symbol}")
                return None
            
            if order_type == mt5.ORDER_TYPE_BUY:
                price = tick.ask
            else:
                price = tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": actual_symbol,
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
    
    def modify_position_sl_tp(self, ticket, new_sl=None, new_tp=None):
        """Modify stop loss and take profit of an existing position"""
        position = None
        positions = mt5.positions_get(ticket=ticket)
        if positions:
            position = positions[0]
        
        if position is None:
            print(f"❌ Position {ticket} not found for modification")
            return None
        
        # Build modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl if new_sl is not None else position.sl,
            "tp": new_tp if new_tp is not None else position.tp,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Position {ticket} SL/TP modified successfully")
            if new_sl is not None:
                print(f"   🛑 New Stop Loss: {new_sl:.5f}")
            if new_tp is not None:
                print(f"   🎯 New Take Profit: {new_tp:.5f}")
        else:
            print(f"❌ Failed to modify position {ticket}: {result.retcode} - {result.comment}")
        
        return result
    
    def get_positions(self, symbol=None):
        """Get current positions with symbol format detection"""
        if symbol:
            # Try to get symbol info to find correct format
            symbol_info, actual_symbol = self.get_symbol_info(symbol)
            if symbol_info is not None:
                positions = mt5.positions_get(symbol=actual_symbol)
            else:
                positions = mt5.positions_get(symbol=symbol)  # Fallback to original
        else:
            positions = mt5.positions_get()
        
        return positions

class TradingBot:
    """Main trading bot using trained RL model"""
    
    def __init__(self, symbol, model_path=None, risk_percent=None):
        self.symbol = symbol
        self.model_loader = ModelLoader()
        self.mt5 = MT5Interface()
        self.model = None
        self.running = False
        
        # Connect to MT5 with credentials from environment
        print("🔗 Connecting to MT5...")
        if not self._connect_mt5():
            raise Exception("Failed to connect to MT5")
        
        # Get symbol-specific configuration
        self.symbol_config = multi_symbol_config.get_symbol_config(symbol)
        
        # Use symbol-specific risk or provided risk
        self.risk_percent = risk_percent if risk_percent is not None else self.symbol_config['risk_percent']
        
        # Trading parameters - MULTI-POSITION SETUP with Symbol Config
        self.min_confidence = self.symbol_config['confidence_threshold']
        self.max_positions = self.symbol_config['max_positions']
        self.lookback_window = 50  # Default, will be updated based on model
        
        base_symbol = symbol.replace('m', '').replace('.c', '').replace('.', '').upper()
        print(f"🚀 MULTI-SYMBOL BOT INITIALIZED for {base_symbol}")
        print(f"   📊 Max Positions: {self.max_positions}")
        print(f"   💰 Risk per trade: {self.risk_percent}%")
        print(f"   🎯 Confidence Threshold: {self.min_confidence}")
        print(f"   � Description: {self.symbol_config['description']}")
        
        # Load model
        if model_path:
            self.model = self.model_loader.load_model(model_path)
        else:
            self.model = self.model_loader.load_best_model(symbol)
        
        # Get model parameters from info and detect observation shape
        if self.model_loader.model_info:
            self.lookback_window = self.model_loader.model_info.get('lookback_window', 50)
            self.transaction_cost = self.model_loader.model_info.get('transaction_cost', 0.0)
        
        # Detect model's expected observation shape
        if hasattr(self.model, 'observation_space'):
            obs_shape = self.model.observation_space.shape
            print(f"🔍 Model expects observation shape: {obs_shape}")
            
            if len(obs_shape) == 2:  # 2D observation like (25, 13)
                self.obs_rows, self.obs_cols = obs_shape
                self.observation_type = '2d'
                print(f"✅ Using 2D observation: {self.obs_rows} x {self.obs_cols}")
            else:  # 1D observation like (50,)
                self.obs_size = obs_shape[0]
                self.observation_type = '1d'
                print(f"✅ Using 1D observation: {self.obs_size} features")
        else:
            # Default to 1D if we can't detect
            self.obs_size = 50
            self.observation_type = '1d'
            print("⚠️ Could not detect model observation space, using 1D default")
        
        # Initialize debug mode and action history for monitoring
        self._debug_mode = True  # Enable debugging by default
        self._recent_actions = []  # Track recent actions for pattern detection
        self._bias_counter = 0  # Count consecutive bias detections
        self._exploration_mode = False  # Enable exploration when bias detected
        self._emergency_mode = False  # Emergency trading mode when model fails
        self._force_retrain = False  # Force retraining recommendation
        
        # 🎯 ADVANCED RISK MANAGEMENT FEATURES
        self.enable_trailing_stop = True  # Enable trailing stop loss
        self.enable_breakeven_stop = True  # Enable break-even stop at $4 profit
        self.breakeven_profit_threshold = 4.0  # $4 USD profit threshold
        self.trailing_stop_distance_pips = 50  # Distance in pips for trailing stop
        self.position_tracking = {}  # Track position states for advanced features
        
        # 🛑 STOP LOSS COOLDOWN SYSTEM
        # Load configuration for SL cooldown
        from config import get_config
        config = get_config()
        
        self.enable_sl_cooldown = config.safety.enable_sl_cooldown  # From configuration
        self.sl_cooldown_minutes = config.safety.sl_cooldown_minutes  # From configuration
        self.sl_cooldown_end_time = None  # When cooldown expires
        self.recent_closed_positions = {}  # Track recently closed positions
        self.sl_hit_count = 0  # Count of SL hits today
        
        print(f"🎯 ADVANCED RISK MANAGEMENT ENABLED")
        print(f"   🏃 Trailing Stop: {'ON' if self.enable_trailing_stop else 'OFF'} ({self.trailing_stop_distance_pips} pips)")
        print(f"   💰 Break-even Stop: {'ON' if self.enable_breakeven_stop else 'OFF'} (${self.breakeven_profit_threshold}+ profit)")
        print(f"   🛑 SL Cooldown: {'ON' if self.enable_sl_cooldown else 'OFF'} ({self.sl_cooldown_minutes} min pause)")
    
    def _connect_mt5(self):
        """Connect to MT5 using multi-account system based on symbol"""
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        print(f"🏦 Connecting to MT5 for symbol: {self.symbol}")
        
        # Use multi-account system to connect based on symbol
        return self.mt5.connect(symbol=self.symbol)

    def enable_emergency_mode(self, enabled=True):
        """Enable emergency trading mode when model fails"""
        self._emergency_mode = enabled
        if enabled:
            print("🚨 EMERGENCY MODE ACTIVATED")
            print("   📊 Using simplified random/rule-based trading")
            print("   ⚠️ Model predictions ignored")
            print("   🛑 RECOMMEND: Stop trading and retrain model")
        else:
            print("✅ Emergency mode deactivated")
    
    def enable_exploration_mode(self, enabled=True):
        """Enable/disable exploration mode to combat bias"""
        self._exploration_mode = enabled
        if enabled:
            print("🎲 Exploration mode ENABLED - Using non-deterministic predictions")
        else:
            print("🎯 Exploration mode DISABLED - Using deterministic predictions")
    
    def reset_bias_detection(self):
        """Reset bias detection counters and history"""
        self._recent_actions = []
        self._recent_raw_actions = []
        self._bias_counter = 0
        print("🔄 Bias detection reset")
    
    def get_bias_report(self):
        """Generate comprehensive bias report with enhanced sample size requirements"""
        sample_count = len(getattr(self, '_recent_actions', []))
        
        # Enhanced minimum requirements
        MIN_SAMPLES_FOR_ANALYSIS = 50   # เพิ่มจาก 10
        MIN_SAMPLES_FOR_CRITICAL = 100  # เพิ่มจาก 20
        
        if sample_count < MIN_SAMPLES_FOR_ANALYSIS:
            return f"📊 INSUFFICIENT DATA: Need at least {MIN_SAMPLES_FOR_ANALYSIS} samples for reliable analysis (have {sample_count})\n💡 Continue trading to collect more data...\n⏳ Early data collection stage"
        
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
        
        # Count actions
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for a in self._recent_actions:
            action_counts[a] += 1
        
        total_predictions = len(self._recent_actions)
        report = []
        report.append(f"📊 BIAS ANALYSIS REPORT ({total_predictions} samples)")
        report.append("=" * 50)
        
        # Enhanced confidence levels
        if total_predictions < MIN_SAMPLES_FOR_CRITICAL:
            confidence = "MEDIUM"
            confidence_icon = "⚠️"
            confidence_msg = f"Need {MIN_SAMPLES_FOR_CRITICAL}+ samples for critical decisions"
        else:
            confidence = "HIGH"
            confidence_icon = "✅"
            confidence_msg = "Sufficient samples for reliable analysis"
        
        report.append(f"{confidence_icon} CONFIDENCE: {confidence} ({confidence_msg})")
        report.append("")
        
        # Action distribution with enhanced sample-size-aware analysis
        for action_id, count in action_counts.items():
            percentage = (count / total_predictions) * 100
            action_name = action_names[action_id]
            
            # Enhanced thresholds based on sample size and confidence
            if confidence == "MEDIUM":
                # Medium confidence - more conservative detection
                if action_id == 0:  # HOLD
                    if percentage == 100.0 and count >= MIN_SAMPLES_FOR_ANALYSIS:
                        # Check raw action variance for 100% HOLD
                        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= MIN_SAMPLES_FOR_ANALYSIS:
                            raw_variance = np.var(self._recent_raw_actions[-min(100, count):])
                            if raw_variance < 0.000001:
                                status = "🚨"  # Extremely low variance - likely stuck
                            elif raw_variance < 0.00001:
                                status = "⚠️"   # Low variance - early warning
                            else:
                                status = "�"   # Conservative but with variance - monitor
                        else:
                            status = "⚠️"  # 100% HOLD without enough raw data
                    elif percentage > 95:
                        status = "⚠️"  # High HOLD percentage - monitor
                    else:
                        status = "📊"  # Normal
                else:  # Trading actions
                    status = "🚨" if percentage > 85 else "⚠️" if percentage > 70 else "📊"
            else:
                # High confidence - normal detection thresholds
                if action_id == 0:  # HOLD action
                    if percentage == 100.0:
                        # Check raw action variance for 100% HOLD
                        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= MIN_SAMPLES_FOR_CRITICAL:
                            raw_variance = np.var(self._recent_raw_actions[-min(100, count):])
                            if raw_variance < 0.000001:
                                status = "🚨"  # Model truly stuck
                            elif raw_variance < 0.00001:
                                status = "⚠️"   # Low variance - monitor
                            else:
                                status = "💤"   # Conservative trading - may be appropriate
                        else:
                            status = "⚠️"  # Need more raw data for variance analysis
                    elif percentage > 90:
                        status = "⚠️"  # High HOLD - monitor
                    else:
                        status = "📊"  # Normal HOLD levels
                else:  # Trading actions (BUY, SELL, CLOSE)
                    status = "🚨" if percentage > 80 else "⚠️" if percentage > 60 else "📊"
            
            report.append(f"{status} {action_name}: {count}/{total_predictions} ({percentage:.1f}%)")
        
        # Enhanced raw action analysis
        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= MIN_SAMPLES_FOR_ANALYSIS:
            raw_actions = self._recent_raw_actions[-min(100, len(self._recent_raw_actions)):]
            avg_raw = np.mean(raw_actions)
            std_raw = np.std(raw_actions)
            
            # Determine if model is stuck based on confidence level
            if confidence == "HIGH" and std_raw < 0.00001:
                status_icon = "🚨"
                status_msg = "CRITICAL: Model stuck with no variance"
            elif confidence == "HIGH" and std_raw < 0.0001:
                status_icon = "⚠️"
                status_msg = "LOW VARIANCE: Model in conservative mode"
            elif confidence == "MEDIUM" and std_raw < 0.000001:
                status_icon = "⚠️"
                status_msg = "EARLY WARNING: Very low variance detected"
            elif confidence == "MEDIUM" and std_raw < 0.00001:
                status_icon = "💤"
                status_msg = "LOW VARIANCE: Model consistently in HOLD zone - may be market conditions"
            else:
                status_icon = "✅"
                status_msg = "HEALTHY: Good variance in predictions"
            
            report.append(f"{status_icon} Raw actions: avg={avg_raw:.3f}, std={std_raw:.6f}")
            report.append(f"💡 {status_msg}")
        else:
            report.append("⚠️ Insufficient raw action data for variance analysis")
        
        # Exploration mode status
        exploration_status = "ACTIVE" if getattr(self, '_exploration_mode', False) else "INACTIVE"
        report.append(f"🎯 Exploration mode: {exploration_status}")
        
        return "\n".join(report)

        # Bias counter with improved context
        bias_count = getattr(self, '_bias_counter', 0)
        if bias_count > 0:
            if bias_count >= 10:
                report.append(f"🚨 SEVERE: {bias_count} bias detections - STOP TRADING")
            elif bias_count >= 5:
                report.append(f"⚠️ ELEVATED: {bias_count} bias detections - increased monitoring")
            else:
                report.append(f"📊 MINOR: {bias_count} bias detections - within acceptable range")
        
        # Exploration mode status
        if getattr(self, '_exploration_mode', False):
            report.append("🎲 Exploration mode: ACTIVE (bias mitigation)")
        else:
            report.append("🎯 Exploration mode: INACTIVE")
        
        # 💡 TRADING CONTEXT SUGGESTIONS
        hold_percentage = (action_counts[0] / total_predictions) * 100
        if hold_percentage >= 80:
            report.append("")
            report.append("💡 HIGH HOLD FREQUENCY SUGGESTIONS:")
            if hasattr(self, '_recent_raw_actions'):
                raw_var = np.var(self._recent_raw_actions[-10:]) if len(self._recent_raw_actions) >= 10 else 0
                if raw_var < 0.001:
                    report.append("   🔄 Consider model retraining - may be stuck")
                else:
                    report.append("   📊 May indicate unfavorable market conditions")
                    report.append("   💤 Conservative approach is often profitable")
                    report.append("   🎯 Monitor for market volatility changes")
        
    def get_bias_report(self):
        """Generate comprehensive bias report with enhanced sample size requirements"""
        sample_count = len(getattr(self, '_recent_actions', []))
        
        # Enhanced minimum requirements (50/100 instead of 20/50)
        MIN_SAMPLES_FOR_ANALYSIS = 50   # เพิ่มจาก 20
        MIN_SAMPLES_FOR_CRITICAL = 100  # เพิ่มจาก 50
        
        if sample_count < MIN_SAMPLES_FOR_ANALYSIS:
            return f"📊 INSUFFICIENT DATA: Need at least {MIN_SAMPLES_FOR_ANALYSIS} samples for reliable analysis (have {sample_count})\n💡 Continue trading to collect more data...\n⏳ Early data collection stage"
        
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
        
        # Count actions
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for a in self._recent_actions:
            action_counts[a] += 1
        
        total_predictions = len(self._recent_actions)
        report = []
        report.append(f"📊 BIAS ANALYSIS REPORT ({total_predictions} samples)")
        report.append("=" * 50)
        
        # Enhanced confidence levels based on new thresholds
        if total_predictions < MIN_SAMPLES_FOR_CRITICAL:
            confidence = "MEDIUM"
            confidence_icon = "⚡"
            confidence_msg = f"Need {MIN_SAMPLES_FOR_CRITICAL}+ samples for critical decisions"
        else:
            confidence = "HIGH"
            confidence_icon = "🔥"
            confidence_msg = "Sufficient samples for highly reliable analysis"
        
        report.append(f"{confidence_icon} CONFIDENCE: {confidence} ({confidence_msg})")
        report.append("")
        
        # Action distribution with enhanced sample-size-aware analysis
        for action_id, count in action_counts.items():
            percentage = (count / total_predictions) * 100
            action_name = action_names[action_id]
            
            # Enhanced thresholds based on sample size and confidence
            if confidence == "MEDIUM":
                # Medium confidence (50-99 samples) - more conservative detection
                if action_id == 0:  # HOLD
                    if percentage == 100.0 and count >= MIN_SAMPLES_FOR_ANALYSIS:
                        # Check raw action variance for 100% HOLD
                        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= MIN_SAMPLES_FOR_ANALYSIS:
                            raw_variance = np.var(self._recent_raw_actions[-min(100, count):])
                            if raw_variance < 0.000001:
                                status = "🚨"  # Extremely low variance - likely stuck
                            elif raw_variance < 0.00001:
                                status = "⚠️"   # Low variance - early warning
                            else:
                                status = "💤"   # Conservative but with variance - monitor
                        else:
                            status = "⚠️"  # 100% HOLD without enough raw data
                    elif percentage > 95:
                        status = "⚠️"  # High HOLD percentage - monitor
                    else:
                        status = "📊"  # Normal
                else:  # Trading actions
                    status = "🚨" if percentage > 85 else "⚠️" if percentage > 70 else "📊"
            else:
                # High confidence (100+ samples) - reliable detection thresholds
                if action_id == 0:  # HOLD action
                    if percentage == 100.0:
                        # Check raw action variance for 100% HOLD
                        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= MIN_SAMPLES_FOR_CRITICAL:
                            raw_variance = np.var(self._recent_raw_actions[-min(100, count):])
                            if raw_variance < 0.000001:
                                status = "🚨"  # Model truly stuck
                            elif raw_variance < 0.00001:
                                status = "⚠️"   # Low variance - monitor
                            else:
                                status = "💤"   # Conservative trading - may be appropriate
                        else:
                            status = "⚠️"  # Need more raw data for variance analysis
                    elif percentage > 90:
                        status = "⚠️"  # High HOLD - monitor
                    else:
                        status = "📊"  # Normal HOLD levels
                else:  # Trading actions (BUY, SELL, CLOSE)
                    status = "🚨" if percentage > 80 else "⚠️" if percentage > 60 else "📊"
            
            report.append(f"{status} {action_name}: {count}/{total_predictions} ({percentage:.1f}%)")
        
        # Enhanced raw action analysis
        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= MIN_SAMPLES_FOR_ANALYSIS:
            raw_actions = self._recent_raw_actions[-min(100, len(self._recent_raw_actions)):]
            avg_raw = np.mean(raw_actions)
            std_raw = np.std(raw_actions)
            
            # Determine if model is stuck based on confidence level
            if confidence == "HIGH" and std_raw < 0.00001:
                status_icon = "🚨"
                status_msg = "CRITICAL: Model stuck with no variance - high confidence"
            elif confidence == "HIGH" and std_raw < 0.0001:
                status_icon = "⚠️"
                status_msg = "LOW VARIANCE: Model in conservative mode - high confidence"
            elif confidence == "MEDIUM" and std_raw < 0.000001:
                status_icon = "⚠️"
                status_msg = "EARLY WARNING: Very low variance detected - medium confidence"
            elif confidence == "MEDIUM" and std_raw < 0.00001:
                status_icon = "💤"
                status_msg = "LOW VARIANCE: Model in HOLD zone - need more samples"
            else:
                status_icon = "✅"
                status_msg = f"HEALTHY: Good variance in predictions ({confidence} confidence)"
            
            report.append(f"{status_icon} Raw actions: avg={avg_raw:.3f}, std={std_raw:.6f}")
            report.append(f"💡 {status_msg}")
        else:
            report.append("⚠️ Insufficient raw action data for variance analysis")
        
        # Bias counter with improved context
        bias_count = getattr(self, '_bias_counter', 0)
        if bias_count > 0:
            if bias_count >= 10:
                report.append(f"🚨 SEVERE: {bias_count} bias detections - STOP TRADING")
            elif bias_count >= 5:
                report.append(f"⚠️ ELEVATED: {bias_count} bias detections - increased monitoring")
            else:
                report.append(f"📊 MINOR: {bias_count} bias detections - within acceptable range")
        
        # Exploration mode status
        exploration_status = "ACTIVE" if getattr(self, '_exploration_mode', False) else "INACTIVE"
        report.append(f"🎯 Exploration mode: {exploration_status}")
        
        # Trading context suggestions
        hold_percentage = (action_counts[0] / total_predictions) * 100
        if hold_percentage >= 80:
            report.append("")
            report.append("💡 HIGH HOLD FREQUENCY SUGGESTIONS:")
            if hasattr(self, '_recent_raw_actions'):
                raw_var = np.var(self._recent_raw_actions[-10:]) if len(self._recent_raw_actions) >= 10 else 0
                if raw_var < 0.001:
                    report.append("   🔄 Consider model retraining - may be stuck")
                else:
                    report.append("   📊 May indicate unfavorable market conditions")
                    report.append("   💤 Conservative approach is often profitable")
                    report.append("   🎯 Monitor for market volatility changes")
        
        return "\n".join(report)
    
    def force_model_retraining_mode(self, enabled=True):
        """Force the bot to recommend model retraining"""
        self._force_retrain = enabled
        if enabled:
            print("🚨 FORCE RETRAIN MODE: Bot will recommend immediate model retraining")
            print("📋 Recommended actions:")
            print("   1. Stop current trading immediately")
            print("   2. Check training data quality")
            print("   3. Retrain model with better parameters")
            print("   4. Validate new model before deployment")
        else:
            print("✅ Force retrain mode disabled")
    
    def configure_risk_management(self, trailing_stop=None, breakeven_stop=None, 
                                 breakeven_threshold=None, trailing_distance_pips=None):
        """🎯 Configure advanced risk management settings"""
        if trailing_stop is not None:
            self.enable_trailing_stop = trailing_stop
        if breakeven_stop is not None:
            self.enable_breakeven_stop = breakeven_stop
        if breakeven_threshold is not None:
            self.breakeven_profit_threshold = breakeven_threshold
        if trailing_distance_pips is not None:
            self.trailing_stop_distance_pips = trailing_distance_pips
        
        print(f"🎯 RISK MANAGEMENT UPDATED:")
        print(f"   🏃 Trailing Stop: {'ON' if self.enable_trailing_stop else 'OFF'} ({self.trailing_stop_distance_pips} pips)")
        print(f"   💰 Break-even Stop: {'ON' if self.enable_breakeven_stop else 'OFF'} (${self.breakeven_profit_threshold}+ profit)")
    
    def get_risk_management_status(self):
        """Get current risk management status and position tracking info"""
        status = {
            'trailing_stop_enabled': self.enable_trailing_stop,
            'breakeven_stop_enabled': self.enable_breakeven_stop,
            'breakeven_threshold_usd': self.breakeven_profit_threshold,
            'trailing_distance_pips': self.trailing_stop_distance_pips,
            'tracked_positions': len(self.position_tracking),
            'position_details': {}
        }
        
        for ticket, data in self.position_tracking.items():
            status['position_details'][ticket] = {
                'breakeven_set': data.get('breakeven_set', False),
                'highest_profit': data.get('highest_profit', 0),
                'best_price': data.get('best_price', 0)
            }
        
        return status
    
    def detect_model_bias(self):
        """ตรวจสอบ bias ของ model ด้วยเกณฑ์ที่เหมาะสม - Enhanced Sample Size Requirements"""
        min_samples_for_analysis = 50   # เพิ่มจาก 20 เป็น 50
        min_samples_for_critical = 100  # เพิ่มจาก 50 เป็น 100
        
        if not hasattr(self, '_recent_raw_actions') or len(self._recent_raw_actions) < 10:
            return "INSUFFICIENT_DATA", f"Need at least 10 samples for basic monitoring (have {len(getattr(self, '_recent_raw_actions', []))})"
        
        sample_count = len(self._recent_raw_actions)
        
        # ถ้ายังไม่ถึง min_samples_for_analysis ให้แสดงข้อความว่าต้องรอ
        if sample_count < min_samples_for_analysis:
            bias_report = []
            bias_report.append(f"📊 INSUFFICIENT DATA: Need at least {min_samples_for_analysis} samples for reliable analysis (have {sample_count})")
            bias_report.append("💡 Continue trading to collect more data...")
            bias_report.append("⏳ Early data collection stage")
            return "INSUFFICIENT_DATA", "\n".join(bias_report)
        
        # ใช้ samples ล่าสุดสำหรับการวิเคราะห์
        recent_raw = self._recent_raw_actions[-min(100, sample_count):]
        variance = np.var(recent_raw)
        mean_raw = np.mean(recent_raw)
        
        # กำหนดเกณฑ์ที่สมเหตุสมผล
        DEAD_THRESHOLD = 0.0000001    # Model completely dead (no variation at all)
        CRITICAL_THRESHOLD = 0.00001  # Model likely stuck (extremely low variation)
        SEVERE_THRESHOLD = 0.0001     # Low variation but may be legitimate
        MODERATE_THRESHOLD = 0.001    # Somewhat low variation
        
        bias_report = []
        bias_report.append(f"📊 Sample Count: {sample_count}, Analyzing: {len(recent_raw)}")
        bias_report.append(f"📊 Variance: {variance:.8f}, Mean: {mean_raw:.4f}")
        
        # กำหนดระดับความมั่นใจ
        if sample_count < min_samples_for_critical:
            confidence = "MEDIUM"
            bias_report.append(f"⚠️ CONFIDENCE: MEDIUM (Need {min_samples_for_critical}+ samples for critical decisions)")
        else:
            confidence = "HIGH"
            bias_report.append(f"✅ CONFIDENCE: HIGH (Sufficient samples for reliable analysis)")
        
        # วิเคราะห์ bias ตามระดับความมั่นใจ
        if variance < DEAD_THRESHOLD:
            bias_report.append("🚨 MODEL DEAD: Zero variance - complete system failure")
            return "DEAD", "\n".join(bias_report)
        
        elif variance < CRITICAL_THRESHOLD:
            # Check if consistently in reasonable HOLD zone
            if -0.2 <= mean_raw <= 0.2:
                if confidence == "MEDIUM":
                    bias_report.append("⚠️ EARLY WARNING: Extremely low variance in HOLD zone")
                    bias_report.append("💡 Monitor closely - may indicate conservative market response")
                    bias_report.append(f"🔍 Need {min_samples_for_critical - sample_count} more samples for critical decision")
                    return "EARLY_WARNING", "\n".join(bias_report)
                else:
                    bias_report.append("⚠️ CRITICAL LOW VARIANCE in HOLD zone")
                    bias_report.append("💡 May indicate conservative market response")
                    bias_report.append("� Monitor: If market volatility increases but variance doesn't = STUCK")
                    return "CRITICAL_WATCH", "\n".join(bias_report)
            else:
                if confidence == "MEDIUM":
                    bias_report.append(f"⚠️ EARLY WARNING: Low variance outside HOLD zone")
                    bias_report.append(f"� Model trending towards stuck at mean {mean_raw:.4f}")
                    bias_report.append(f"🔍 Need {min_samples_for_critical - sample_count} more samples to confirm")
                    return "EARLY_WARNING", "\n".join(bias_report)
                else:
                    bias_report.append(f"🚨 CRITICAL BIAS: Extremely low variance outside HOLD zone")
                    bias_report.append(f"🚨 Model stuck at mean {mean_raw:.4f}")
                    return "CRITICAL", "\n".join(bias_report)
        
        elif variance < SEVERE_THRESHOLD:
            if -0.3 <= mean_raw <= 0.3:
                if confidence == "MEDIUM":
                    bias_report.append("📊 MODERATE variance in HOLD zone - early monitoring")
                    bias_report.append("💡 Conservative behavior may be appropriate")
                    return "EARLY_MONITORING", "\n".join(bias_report)
                else:
                    bias_report.append("⚠️ LOW VARIANCE in HOLD zone - acceptable for stable markets")
                    bias_report.append("💡 Conservative behavior may be appropriate")
                    return "ACCEPTABLE_LOW", "\n".join(bias_report)
            else:
                bias_report.append(f"⚠️ MODERATE BIAS: Low variance outside HOLD zone")
                return "MODERATE" if confidence == "HIGH" else "EARLY_WARNING", "\n".join(bias_report)
        
        elif variance < MODERATE_THRESHOLD:
            bias_report.append("📊 MODERATE variance - continue monitoring")
            return "MODERATE", "\n".join(bias_report)
        
        else:
            bias_report.append("✅ HEALTHY variance - good model exploration")
            return "HEALTHY", "\n".join(bias_report)

    def check_model_health(self):
        """Comprehensive model health check with improved bias detection"""
        health_report = []
        health_report.append("🏥 MODEL HEALTH CHECK")
        health_report.append("=" * 30)
        
        # ใช้ bias detection ใหม่
        bias_status, bias_details = self.detect_model_bias()
        health_report.append("� BIAS ANALYSIS:")
        health_report.append(bias_details)
        
        # จัดการตาม bias status
        if bias_status == "DEAD":
            health_report.append("🚨 ACTION: IMMEDIATE MODEL REPLACEMENT REQUIRED")
            self.force_model_retraining_mode(True)
        elif bias_status == "CRITICAL":
            health_report.append("🚨 ACTION: STOP TRADING - MODEL RESTART NEEDED")
            self.force_model_retraining_mode(True)
        elif bias_status == "CRITICAL_WATCH":
            health_report.append("⚠️ ACTION: ENHANCED MONITORING - May be legitimate")
        elif bias_status == "SEVERE":
            health_report.append("⚠️ ACTION: Increased monitoring required")
        elif bias_status == "ACCEPTABLE_LOW":
            health_report.append("✅ ACTION: Continue with normal monitoring")
        elif bias_status == "HEALTHY":
            health_report.append("✅ ACTION: Model performing well")
        
        # Check bias counter
        bias_count = getattr(self, '_bias_counter', 0)
        if bias_count >= 10:
            health_report.append("🚨 BIAS COUNTER: Critical level reached - STOP TRADING")
            self.force_model_retraining_mode(True)
        elif bias_count >= 5:
            health_report.append("⚠️ BIAS COUNTER: Elevated - increased monitoring")
        elif bias_count > 0:
            health_report.append(f"📊 BIAS COUNTER: {bias_count} - within acceptable range")
        else:
            health_report.append("✅ BIAS COUNTER: No bias detected")
        
        # Check recent action patterns
        if hasattr(self, '_recent_actions') and len(self._recent_actions) >= 10:
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for a in self._recent_actions:
                action_counts[a] += 1
            
            hold_percentage = (action_counts[0] / len(self._recent_actions)) * 100
            
            health_report.append(f"📊 ACTION DISTRIBUTION:")
            health_report.append(f"   HOLD: {action_counts[0]} ({hold_percentage:.1f}%)")
            health_report.append(f"   BUY: {action_counts[1]} ({action_counts[1]*10:.1f}%)")
            health_report.append(f"   SELL: {action_counts[2]} ({action_counts[2]*10:.1f}%)")
            health_report.append(f"   CLOSE: {action_counts[3]} ({action_counts[3]*10:.1f}%)")
            
            if hold_percentage >= 90:
                if bias_status in ["ACCEPTABLE_LOW", "HEALTHY"]:
                    health_report.append("💤 PATTERN: High HOLD - Conservative/waiting behavior")
                    health_report.append("💡 CONTEXT: May indicate unfavorable market conditions")
                else:
                    health_report.append("� PATTERN: High HOLD with problematic bias")
            elif hold_percentage <= 20:
                health_report.append("⚡ PATTERN: High activity - frequent trading")
            else:
                health_report.append("✅ PATTERN: Balanced action distribution")
        
        # Check special modes
        if getattr(self, '_exploration_mode', False):
            health_report.append("🎲 STATUS: Exploration mode active (bias mitigation)")
        
        if getattr(self, '_emergency_mode', False):
            health_report.append("🚨 EMERGENCY: Model in emergency mode - manual intervention required")
        
        return "\n".join(health_report)
    
    def should_retrain_model(self):
        """ตรวจสอบว่าควร retrain model หรือไม่ - with improved bias detection"""
        retrain_reasons = []
        
        print("🔍 CHECKING IF MODEL NEEDS RETRAINING")
        print("=" * 40)
        
        # ใช้ bias detection ใหม่
        bias_status, bias_details = self.detect_model_bias()
        print(f"🔍 BIAS STATUS: {bias_status}")
        print(f"📊 DETAILS: {bias_details}")
        
        # จัดการตาม bias status
        if bias_status == "DEAD":
            retrain_reasons.append("🚨 MODEL DEATH: Complete system failure")
        elif bias_status == "CRITICAL":
            retrain_reasons.append("🚨 CRITICAL BIAS: Model stuck outside HOLD zone")
        elif bias_status == "CRITICAL_WATCH":
            print("   💤 INFO: Low variance in HOLD zone - may be legitimate market response")
            print("   💡 Continuing monitoring before retraining decision")
        elif bias_status == "SEVERE":
            retrain_reasons.append("⚠️ SEVERE BIAS: Low variance outside acceptable range")
        elif bias_status == "ACCEPTABLE_LOW":
            print("   ✅ INFO: Low variance but within acceptable HOLD behavior")
        elif bias_status == "HEALTHY":
            print("   ✅ INFO: Healthy variance - good model exploration")
        
        # ตรวจสอบ bias counter
        bias_count = getattr(self, '_bias_counter', 0)
        if bias_count >= 10:
            retrain_reasons.append("🚨 SEVERE BIAS: Bias counter >= 10")
        elif bias_count >= 7:
            retrain_reasons.append("⚠️ HIGH BIAS: Bias counter >= 7")
        
        # ตรวจสอบ action pattern
        if hasattr(self, '_recent_actions') and len(self._recent_actions) >= 10:
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for a in self._recent_actions:
                action_counts[a] += 1
            
            max_count = max(action_counts.values())
            dominant_action = max(action_counts, key=action_counts.get)
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            
            if dominant_action == 0:  # HOLD is dominant
                if max_count >= 10 and bias_status in ["CRITICAL", "DEAD"]:
                    retrain_reasons.append(f"🚨 MODEL STUCK: 100% HOLD with {bias_status} bias")
                elif max_count >= 10:
                    print(f"   💤 INFO: 100% HOLD with {bias_status} bias - may be appropriate")
            else:  # Non-HOLD action is dominant
                if max_count >= 8:  # 80%+ of one trading action
                    retrain_reasons.append(f"🚨 EXTREME BIAS: {action_names[dominant_action]} dominates {max_count}/10 actions")
        
        # ตรวจสอบ special modes
        if getattr(self, '_exploration_mode', False):
            retrain_reasons.append("⚠️ PERSISTENT EXPLORATION: Model requires exploration mode")
        
        if getattr(self, '_emergency_mode', False):
            retrain_reasons.append("🚨 EMERGENCY MODE: Model completely failed")
        
        # สรุปผลการตรวจสอบ
        if retrain_reasons:
            print("❌ MODEL NEEDS RETRAINING")
            print("📋 Reasons:")
            for reason in retrain_reasons:
                print(f"   {reason}")
            
            print("\n📋 RECOMMENDED ACTIONS:")
            print("   1. 🛑 STOP live trading immediately")
            print("   2. 📦 Backup current model")
            print("   3. 🔄 Start fresh model training")
            print("   4. 📊 Use different hyperparameters")
            print("   5. 🧪 Validate new model thoroughly")
            print("   6. 📈 Consider market conditions in evaluation")
            
            return True
        else:
            print("✅ MODEL APPEARS HEALTHY")
            print("   📊 Conservative behavior may be appropriate for current market")
            print("   💡 HOLD-heavy patterns can indicate smart risk management")
            print("   🔍 Continue monitoring performance")
            return False
    
    def get_retraining_recommendation(self):
        """ให้คำแนะนำการ retrain แบบละเอียด"""
        recommendation = []
        recommendation.append("🔄 MODEL RETRAINING RECOMMENDATIONS")
        recommendation.append("=" * 45)
        
        # วิเคราะห์ปัญหาปัจจุบัน
        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= 5:
            variance = np.var(self._recent_raw_actions)
            mean_action = np.mean(self._recent_raw_actions)
            
            recommendation.append(f"📊 Current Model Analysis:")
            recommendation.append(f"   • Raw action variance: {variance:.6f}")
            recommendation.append(f"   • Mean raw action: {mean_action:.3f}")
            recommendation.append(f"   • Bias counter: {getattr(self, '_bias_counter', 0)}")
        
        recommendation.append("")
        recommendation.append("🚨 DO NOT USE CURRENT MODEL FOR RETRAINING")
        recommendation.append("   ❌ Model has collapsed/biased")
        recommendation.append("   ❌ Weights are corrupted")
        recommendation.append("   ❌ Will propagate existing problems")
        
        recommendation.append("")
        recommendation.append("✅ RECOMMENDED APPROACH:")
        recommendation.append("   1. 🔄 START COMPLETELY FRESH")
        recommendation.append("      • Initialize new neural network")
        recommendation.append("      • Use default hyperparameters")
        recommendation.append("      • Clean training data")
        
        recommendation.append("")
        recommendation.append("   2. 📊 IMPROVE TRAINING DATA")
        recommendation.append("      • Use more diverse market conditions")
        recommendation.append("      • Ensure balanced action distribution")
        recommendation.append("      • Remove extreme outliers")
        
        recommendation.append("")
        recommendation.append("   3. 🔧 ADJUST HYPERPARAMETERS")
        recommendation.append("      • Lower learning rate (3e-4 → 1e-4)")
        recommendation.append("      • Increase entropy coefficient (exploration)")
        recommendation.append("      • Add regularization")
        
        recommendation.append("")
        recommendation.append("   4. 🧪 ENHANCED VALIDATION")
        recommendation.append("      • Longer training period")
        recommendation.append("      • Multiple evaluation metrics")
        recommendation.append("      • Paper trading before live deployment")
        
        recommendation.append("")
        recommendation.append("📋 FILES TO CREATE:")
        recommendation.append("   • retrain_fresh_model.py (already created)")
        recommendation.append("   • new_training_config.json")
        recommendation.append("   • validation_metrics.py")
        
        return "\n".join(recommendation)
    
    def get_positions_summary(self):
        """Get detailed summary of all current positions"""
        positions = self.mt5.get_positions(self.symbol)
        if not positions:
            return {
                'total_positions': 0,
                'buy_positions': 0,
                'sell_positions': 0,
                'net_exposure': 0,
                'total_volume': 0.0,
                'oldest_position': None,
                'newest_position': None
            }
        
        buy_positions = [p for p in positions if p.type == mt5.POSITION_TYPE_BUY]
        sell_positions = [p for p in positions if p.type == mt5.POSITION_TYPE_SELL]
        
        # Calculate net exposure and total volume
        buy_volume = sum(p.volume for p in buy_positions)
        sell_volume = sum(p.volume for p in sell_positions)
        net_exposure = buy_volume - sell_volume
        total_volume = buy_volume + sell_volume
        
        # Find oldest and newest positions
        positions_by_time = sorted(positions, key=lambda p: p.time)
        oldest = positions_by_time[0] if positions_by_time else None
        newest = positions_by_time[-1] if positions_by_time else None
        
        return {
            'total_positions': len(positions),
            'buy_positions': len(buy_positions),
            'sell_positions': len(sell_positions),
            'net_exposure': net_exposure,
            'total_volume': total_volume,
            'oldest_position': oldest,
            'newest_position': newest,
            'positions_list': positions
        }

    def print_positions_report(self):
        """Print detailed positions report"""
        summary = self.get_positions_summary()
        
        print(f"\n📊 POSITIONS REPORT")
        print(f"   Total Positions: {summary['total_positions']}/{self.max_positions}")
        print(f"   BUY Positions: {summary['buy_positions']}")
        print(f"   SELL Positions: {summary['sell_positions']}")
        print(f"   Net Exposure: {summary['net_exposure']:.2f} lots")
        print(f"   Total Volume: {summary['total_volume']:.2f} lots")
        
        if summary['oldest_position']:
            oldest = summary['oldest_position']
            oldest_time = datetime.fromtimestamp(oldest.time)
            print(f"   Oldest Position: #{oldest.ticket} ({oldest_time.strftime('%H:%M:%S')})")
        
        if summary['total_positions'] > 0:
            print(f"   Positions Details:")
            for i, pos in enumerate(summary['positions_list'], 1):
                pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                pos_time = datetime.fromtimestamp(pos.time)
                profit = pos.profit
                print(f"     {i}. #{pos.ticket} {pos_type} {pos.volume} lots @ {pos.price_open:.5f} | P&L: ${profit:.2f} | {pos_time.strftime('%H:%M:%S')}")

    def connect_mt5(self, login=None, password=None, server=None):
        """Connect to MT5"""
        return self.mt5.connect(login, password, server)
    
    def calculate_position_size(self):
        """Calculate position size based on risk management - Enhanced for multiple symbols"""
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01
        
        balance = account_info.equity
        
        # Get symbol info
        symbol_info_data = self.mt5.get_symbol_info(self.symbol)
        if symbol_info_data is None or symbol_info_data[0] is None:
            return 0.01
        
        symbol_info, actual_symbol = symbol_info_data
        
        # Get symbol-specific configuration
        symbol_config = multi_symbol_config.get_symbol_config(actual_symbol)
        
        # Calculate risk amount based on symbol configuration
        risk_amount = balance * (symbol_config['risk_percent'] / 100)
        
        # Calculate lot size based on account balance tiers
        if balance >= 10000:
            lot_multiplier = 2.0
        elif balance >= 5000:
            lot_multiplier = 1.5
        elif balance >= 1000:
            lot_multiplier = 1.0
        else:
            lot_multiplier = 0.5
        
        # Base lot size calculation using config
        calculated_lot = symbol_config['base_lot'] * lot_multiplier
        
        # Apply risk-based adjustment
        if balance > 0:
            risk_ratio = min(risk_amount / (balance * 0.01), 3.0)  # Cap at 3x
            calculated_lot *= risk_ratio
        
        # Ensure compliance with broker requirements
        min_lot = symbol_info.volume_min
        max_lot = min(symbol_info.volume_max, symbol_config['max_lot'])
        lot_step = symbol_info.volume_step
        
        # Round to step size
        calculated_lot = max(min_lot, round(calculated_lot / lot_step) * lot_step)
        calculated_lot = min(calculated_lot, max_lot)
        
        base_symbol = actual_symbol.replace('m', '').replace('.c', '').replace('.', '').upper()
        print(f"   💰 Position Sizing for {base_symbol}:")
        print(f"      Balance: ${balance:.2f}")
        print(f"      Risk: {symbol_config['risk_percent']}% = ${risk_amount:.2f}")
        print(f"      Base Lot: {symbol_config['base_lot']}")
        print(f"      Multiplier: {lot_multiplier}x")
        print(f"      Final Lot: {calculated_lot}")
        print(f"      Limits: {min_lot} - {max_lot}")
        
        return calculated_lot
    
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
        """Prepare observation for model (auto-detects 1D vs 2D format)"""
        print(f"   🔧 Preparing {self.observation_type} observation (position: {current_position})")
        
        if self.observation_type == '2d':
            return self._prepare_2d_observation(df, current_position)
        else:
            return self._prepare_1d_observation(df, current_position)
    
    def _prepare_2d_observation(self, df, current_position=0):
        """Prepare 2D observation for models expecting (rows, cols) shape"""
        rows, cols = self.obs_rows, self.obs_cols
        
        # Get recent price data for sliding window
        window_size = rows
        
        # Extract price data
        price_data = df['close'].tail(window_size).values
        high_data = df['high'].tail(window_size).values
        low_data = df['low'].tail(window_size).values
        volume_data = df['volume'].tail(window_size).values if 'volume' in df.columns else np.ones(len(price_data))
        
        # Pad if insufficient data
        if len(price_data) < window_size:
            padding_size = window_size - len(price_data)
            price_data = np.pad(price_data, (padding_size, 0), 'edge')
            high_data = np.pad(high_data, (padding_size, 0), 'edge')
            low_data = np.pad(low_data, (padding_size, 0), 'edge')
            volume_data = np.pad(volume_data, (padding_size, 0), 'edge')
        
        # Normalize data
        base_price = price_data[-1]
        price_returns = np.diff(price_data) / price_data[:-1]
        price_returns = np.append(price_returns, 0)  # Add current step
        
        # Create feature matrix
        features = np.zeros((rows, cols))
        
        for i in range(rows):
            if i < len(price_data):
                features[i, 0] = price_returns[i] * 100  # Price return
                features[i, 1] = (high_data[i] - price_data[i]) / price_data[i] * 100  # High-Close
                features[i, 2] = (price_data[i] - low_data[i]) / price_data[i] * 100   # Close-Low
                features[i, 3] = volume_data[i] / np.mean(volume_data) if np.mean(volume_data) > 0 else 1.0
                
                # Add technical indicators if available
                idx = len(df) - window_size + i
                if idx >= 0 and idx < len(df):
                    # RSI
                    if 'rsi' in df.columns:
                        rsi_val = df['rsi'].iloc[idx] if not pd.isna(df['rsi'].iloc[idx]) else 50.0
                        features[i, 4] = (rsi_val - 50) / 50
                    
                    # MACD
                    if 'macd' in df.columns:
                        macd_val = df['macd'].iloc[idx] if not pd.isna(df['macd'].iloc[idx]) else 0.0
                        features[i, 5] = macd_val / base_price * 10000
                    
                    # Bollinger Bands
                    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                        bb_upper = df['bb_upper'].iloc[idx] if not pd.isna(df['bb_upper'].iloc[idx]) else price_data[i]
                        bb_lower = df['bb_lower'].iloc[idx] if not pd.isna(df['bb_lower'].iloc[idx]) else price_data[i]
                        features[i, 6] = (bb_upper - price_data[i]) / price_data[i] * 100
                        features[i, 7] = (price_data[i] - bb_lower) / price_data[i] * 100
                    
                    # Stochastic and ATR
                    if 'atr' in df.columns:
                        atr_val = df['atr'].iloc[idx] if not pd.isna(df['atr'].iloc[idx]) else 0.001
                        features[i, 8] = atr_val / price_data[i] * 100
                
                # Position information on last row
                if i == rows - 1:
                    features[i, 9] = float(current_position)
                    features[i, 10] = 0.0  # Entry price difference (simplified)
                    features[i, 11] = 0.0  # Trade count (simplified)
                    if cols > 12:
                        features[i, 12] = 0.0  # Additional feature if needed
        
        # Clip extreme values
        features = np.clip(features, -10, 10)
        print(f"   📊 2D observation prepared: shape {features.shape}, mean={np.mean(features):.3f}")
        return features.astype(np.float32)
    
    def _prepare_1d_observation(self, df, current_position=0):
        """Prepare 1D observation for models expecting (features,) shape"""
        # Get required columns (must match training)
        feature_cols = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi', 'macd', 
                       'macd_signal', 'bb_upper', 'bb_lower', 'atr']
        
        # Filter only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Get last lookback_window rows
        obs_data = df[available_cols].tail(self.lookback_window).values
        
        # Handle insufficient data
        if len(obs_data) < self.lookback_window:
            padding = np.tile(obs_data[0], (self.lookback_window - len(obs_data), 1))
            obs_data = np.vstack([padding, obs_data])
        
        # Add position information (must match training)
        position_info = np.full((obs_data.shape[0], 1), current_position)
        obs_data = np.hstack([obs_data, position_info])
        
        # Normalize data (except position)
        obs_data[:, :-1] = (obs_data[:, :-1] - np.mean(obs_data[:, :-1], axis=0)) / (np.std(obs_data[:, :-1], axis=0) + 1e-8)
        
        # Flatten for 1D models if needed
        obs_flattened = obs_data.flatten().astype(np.float32)
        print(f"   📊 1D observation prepared: shape {obs_flattened.shape}, mean={np.mean(obs_flattened):.3f}")
        
        return obs_flattened
    
    def predict_action(self, observation, current_position=0):
        """Get action prediction from model - EXACTLY MATCH TRAINING LOGIC"""
        # Debug: print observation shape and some stats
        if hasattr(observation, 'shape'):
            print(f"   🔍 Observation shape: {observation.shape}")
            if len(observation.shape) == 2:
                print(f"   📊 Obs stats: mean={np.mean(observation):.3f}, std={np.std(observation):.3f}")
                print(f"   📊 Obs range: [{np.min(observation):.3f}, {np.max(observation):.3f}]")
            elif len(observation.shape) == 3:
                print(f"   📊 Obs stats: mean={np.mean(observation):.3f}, std={np.std(observation):.3f}")
        
        # 🚨 BIAS DETECTION: Check if model is stuck by using non-deterministic first
        action_nd, _states_nd = self.model.predict(observation, deterministic=False)
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Convert continuous action to discrete
        if isinstance(action, (list, np.ndarray)):
            action_value = float(action[0])
            action_nd_value = float(action_nd[0])
        else:
            action_value = float(action)
            action_nd_value = float(action_nd)
        
        # Debug: show raw action values and detect bias
        print(f"   🎯 Raw model output (deterministic): {action_value:.6f}")
        print(f"   🎯 Raw model output (non-deterministic): {action_nd_value:.6f}")
        
        # 🚨 BIAS DETECTION: If deterministic always gives extreme values, use non-deterministic
        bias_detected = False
        
        # Check for extreme bias: both deterministic and non-deterministic are identical
        if abs(action_value - action_nd_value) < 0.001 and abs(action_value) >= 0.99:
            print(f"   🚨 SEVERE BIAS: Both deterministic and non-deterministic identical at extreme value {action_value:.3f}")
            print(f"   🚨 MODEL FAILURE: Model has collapsed - needs immediate retraining!")
            bias_detected = True
            self._bias_counter = getattr(self, '_bias_counter', 0) + 10  # Severe penalty
            
            # Force random action as emergency fallback
            import random
            emergency_actions = [0, 1, 2]  # HOLD, BUY, SELL (avoid CLOSE when no position)
            if current_position == 0:  # No position, use any action except CLOSE
                emergency_action = random.choice(emergency_actions)
            else:  # Has position, include CLOSE as option
                emergency_action = random.choice([0, 1, 2, 3])
            
            # Map emergency action back to continuous value
            if emergency_action == 0:    # HOLD
                action_value = 0.0
            elif emergency_action == 1:  # BUY  
                action_value = 0.5
            elif emergency_action == 2:  # SELL
                action_value = -0.5
            else:                        # CLOSE
                action_value = 0.8
                
            print(f"   🎲 EMERGENCY: Using random action {emergency_action} (value: {action_value:.3f})")
            
        elif abs(action_value) >= 0.99:  # Model stuck at extreme values
            print(f"   ⚠️ BIAS DETECTED: Model stuck at extreme value {action_value:.3f}")
            bias_detected = True
            self._bias_counter = getattr(self, '_bias_counter', 0) + 1
            
            # Auto-enable exploration mode after 5 consecutive bias detections (increased from 3)
            if self._bias_counter >= 5 and not self._exploration_mode:
                print(f"   🎲 Auto-enabling exploration mode after {self._bias_counter} bias detections")
                self.enable_exploration_mode(True)
            
            # Use non-deterministic prediction instead
            action_value = action_nd_value
            print(f"   🔄 Using non-deterministic prediction: {action_value:.6f}")
        else:
            # 🎯 CONTEXTUAL BIAS DETECTION: Only reset bias counter for non-HOLD actions
            # If action is in HOLD zone (-0.3 to 0.3), don't reset bias counter immediately
            if abs(action_value) > 0.3:  # Non-HOLD action
                self._bias_counter = 0  # Reset bias counter for active trading decisions
            # For HOLD actions, bias counter decays slowly instead of immediate reset
            elif self._bias_counter > 0:
                self._bias_counter = max(0, self._bias_counter - 0.1)  # Slow decay for HOLD
            
            # If in exploration mode, still use non-deterministic occasionally
            if self._exploration_mode:
                action_value = action_nd_value
                print(f"   🎲 Exploration mode: Using non-deterministic {action_value:.6f}")
        
        # 🎯 EXACT TRAINING MAPPING: 
        # This logic now perfectly matches the training environment.
        # No confidence check is performed, only action mapping.
        # [-1, -0.3): Sell (Short) = 2
        # [-0.3, 0.3): Hold = 0
        # [0.3, 0.7): Buy = 1
        # [0.7, 1]: Close = 3
        if action_value < -0.3:
            discrete_action = 2  # Sell
        elif action_value < 0.3:
            discrete_action = 0  # Hold (balanced zone)
        elif action_value < 0.7:
            discrete_action = 1  # Buy
        else:
            discrete_action = 3  # Close
        
        # Confidence is no longer used for filtering, set to 1.0 for logging.
        confidence = 1.0
        
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        # Track action patterns to detect if model is stuck
        self._recent_actions.append(discrete_action)
        self._recent_raw_actions = getattr(self, '_recent_raw_actions', [])
        self._recent_raw_actions.append(action_value)
        
        if len(self._recent_actions) > 10:
            self._recent_actions.pop(0)
            self._recent_raw_actions.pop(0)
        
        # 🚨 IMPROVED ADVANCED BIAS DETECTION
        if len(self._recent_actions) >= 5:
            last_5_actions = self._recent_actions[-5:]
            last_5_raw = self._recent_raw_actions[-5:]
            
            # Check for identical actions with improved HOLD handling
            if all(a == last_5_actions[0] for a in last_5_actions):
                action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
                
                if last_5_actions[0] == 0:  # All HOLD
                    # For HOLD, check raw variance to distinguish legitimate from stuck
                    raw_variance = np.var(last_5_raw)
                    raw_mean = np.mean(last_5_raw)
                    
                    if raw_variance < 0.001:
                        print(f"   🚨 MODEL STUCK: HOLD pattern with no variance - raw: {raw_mean:.3f}±{raw_variance:.6f}")
                    elif raw_variance < 0.01:
                        print(f"   ⚠️ CONSERVATIVE: Consistent HOLD decisions - raw: {raw_mean:.3f}±{raw_variance:.6f}")
                    else:
                        print(f"   💤 PATIENT: Legitimate waiting pattern - raw: {raw_mean:.3f}±{raw_variance:.6f}")
                else:
                    # Non-HOLD patterns are concerning
                    print(f"   🚨 BIAS ALERT: Model stuck in pattern - last 5 actions: {last_5_actions}")
                    print(f"   📊 Raw action values: {[f'{r:.3f}' for r in last_5_raw]}")
                    
                    # Check if raw values are also similar (indicating model convergence issue)
                    if all(abs(r - last_5_raw[0]) < 0.1 for r in last_5_raw):
                        print(f"   🚨 SEVERE BIAS: Raw values also identical - Model needs retraining!")
                    
        # Calculate action distribution for the last 10 predictions with improved analysis
        if len(self._recent_actions) >= 10:
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # HOLD, BUY, SELL, CLOSE
            for a in self._recent_actions:
                action_counts[a] += 1
            
            action_names_debug = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            distribution = {action_names_debug[k]: v for k, v in action_counts.items()}
            print(f"   📊 Last 10 actions distribution: {distribution}")
            
            # 🎯 IMPROVED BIAS DETECTION: Different thresholds for HOLD vs trading actions
            max_count = max(action_counts.values())
            dominant_action = max(action_counts, key=action_counts.get)
            
            if dominant_action == 0:  # HOLD is dominant
                if max_count >= 9:  # 90%+ HOLD - check if it's legitimate
                    raw_variance = np.var(self._recent_raw_actions[-10:])
                    if raw_variance < 0.001:
                        print(f"   🚨 EXTREME BIAS: HOLD dominates {max_count}/10 + no variance - MODEL STUCK!")
                    else:
                        print(f"   💤 CONSERVATIVE: HOLD dominates {max_count}/10 - may be market conditions")
            else:  # Trading action is dominant
                if max_count >= 7:  # 70%+ of one trading action is concerning
                    print(f"   🚨 EXTREME BIAS: {action_names_debug[dominant_action]} dominates {max_count}/10 predictions!")
        
        return discrete_action, confidence, action_value
    
    def calculate_stop_levels(self, symbol_info, actual_symbol, current_price, is_buy=True):
        """Calculate proper stop loss and take profit levels to avoid 'Invalid stops' error"""
        
        # Get symbol-specific configuration
        symbol_config = multi_symbol_config.get_symbol_config(actual_symbol)
        base_symbol = actual_symbol.replace('m', '').replace('.c', '').replace('.', '').upper()
        
        # Get minimum stop level from broker
        stops_level = symbol_info.trade_stops_level
        min_stop_distance = max(stops_level, symbol_config['min_stop_points']) * symbol_info.point
        
        # Use configured stop distance, but ensure it meets broker minimum
        sl_distance = max(symbol_config['sl_points'] * symbol_info.point, min_stop_distance)
        tp_distance = sl_distance * symbol_config['tp_ratio']
        
        if is_buy:
            sl = current_price.ask - sl_distance
            tp = current_price.ask + tp_distance
            entry_price = current_price.ask
        else:
            sl = current_price.bid + sl_distance  
            tp = current_price.bid - tp_distance
            entry_price = current_price.bid
        
        print(f"   📏 Stop Levels for {base_symbol}:")
        print(f"      Entry: {entry_price:.5f}")
        print(f"      Stop Distance: {sl_distance/symbol_info.point:.0f} points")
        print(f"      SL: {sl:.5f} (Distance: {abs(entry_price-sl)/symbol_info.point:.0f} points)")
        print(f"      TP: {tp:.5f} (Distance: {abs(tp-entry_price)/symbol_info.point:.0f} points)")
        print(f"      Broker Min: {stops_level} points")
        print(f"      Config Min: {symbol_config['min_stop_points']} points")
        
        return sl, tp

    def execute_trade(self, action, confidence):
        """Execute trade based on model prediction - HANDLES MULTIPLE POSITIONS WITH PROPER STOPS"""
        print(f"   🎯 EXECUTING TRADE (Multi-Position Logic): Action={action}, Confidence={confidence:.3f}")
        
        # 🛑 CHECK STOP LOSS COOLDOWN FIRST
        in_cooldown, remaining_minutes = self.is_in_cooldown()
        if in_cooldown:
            print(f"   🛑 TRADING PAUSED - Stop Loss Cooldown Active")
            print(f"   ⏰ Remaining: {remaining_minutes:.1f} minutes")
            print(f"   📊 SL hits today: {self.sl_hit_count}")
            print(f"   💡 Reason: Protecting from consecutive losses")
            return None  # Skip trading during cooldown
        
        positions = self.mt5.get_positions(self.symbol)
        num_positions = len(positions) if positions else 0
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
        action_name = action_names.get(action, 'UNKNOWN')

        print(f"   📊 Position check: current count={num_positions}, max={self.max_positions}")

        # --- Action: BUY ---
        if action == 1:
            if num_positions < self.max_positions:
                print(f"   💰 Calculating position size for new BUY...")
                volume = self.calculate_position_size()
                symbol_info, actual_symbol = self.mt5.get_symbol_info(self.symbol)
                current_price = mt5.symbol_info_tick(actual_symbol)
                
                if symbol_info and current_price:
                    # Calculate proper stop levels
                    sl, tp = self.calculate_stop_levels(symbol_info, actual_symbol, current_price, is_buy=True)
                    
                    print(f"   💹 Executing new BUY order for {volume} lots...")
                    result = self.mt5.send_order(
                        actual_symbol, mt5.ORDER_TYPE_BUY, volume, sl=sl, tp=tp,
                        comment=f"RL Buy C:{confidence:.2f}"
                    )
                    
                    # 🎯 Initialize position tracking for new positions
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Get the new position ticket (order number from result)
                        if hasattr(result, 'order') and result.order:
                            # Note: We'll track this position once it shows up in positions
                            pass
                    
                    return result
                else:
                    print(f"   ❌ Failed to get symbol/price info for new BUY.")
                    return False  # Return False instead of None
            else:
                print(f"   ⚠️ {action_name} signal ignored, max positions ({self.max_positions}) reached.")
                return False  # Return False instead of None

        # --- Action: SELL ---
        elif action == 2:
            if num_positions < self.max_positions:
                print(f"   📉 Calculating position size for new SELL...")
                volume = self.calculate_position_size()
                symbol_info, actual_symbol = self.mt5.get_symbol_info(self.symbol)
                current_price = mt5.symbol_info_tick(actual_symbol)

                if symbol_info and current_price:
                    # Calculate proper stop levels
                    sl, tp = self.calculate_stop_levels(symbol_info, actual_symbol, current_price, is_buy=False)
                    
                    print(f"   💹 Executing new SELL order for {volume} lots...")
                    result = self.mt5.send_order(
                        actual_symbol, mt5.ORDER_TYPE_SELL, volume, sl=sl, tp=tp,
                        comment=f"RL Sell C:{confidence:.2f}"
                    )
                    
                    # 🎯 Initialize position tracking for new positions  
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Get the new position ticket (order number from result)
                        if hasattr(result, 'order') and result.order:
                            # Note: We'll track this position once it shows up in positions
                            pass
                    
                    return result
                else:
                    print(f"   ❌ Failed to get symbol/price info for new SELL.")
                    return False  # Return False instead of None
            else:
                print(f"   ⚠️ {action_name} signal ignored, max positions ({self.max_positions}) reached.")
                return False  # Return False instead of None

        # --- Action: CLOSE ---
        elif action == 3:
            if positions:
                # Strategy: Close the OLDEST position
                positions_by_time = sorted(positions, key=lambda p: p.time)
                oldest_position = positions_by_time[0]
                
                print(f"   🔄 Closing oldest position: Ticket {oldest_position.ticket}, Type: {oldest_position.type}, Time: {datetime.fromtimestamp(oldest_position.time)}")
                return self.mt5.close_position(oldest_position.ticket)
            else:
                print(f"   ⚠️ {action_name} signal ignored, no position to close.")
                return False  # Return False instead of None
        
        # --- Action: HOLD ---
        elif action == 0:
             print(f"   💤 HOLD signal received. No action taken.")
             return True  # Return True for successful HOLD

        else:
            print(f"   ⚠️ Unknown action {action} - conditions not met.")
            return False  # Return False instead of None
    
    def run_single_iteration(self):
        """Execute one trading iteration (for integration with other systems)"""
        if not self.mt5.connected:
            print("❌ MT5 not connected")
            return None
        
        try:
            # Get latest market data with symbol format detection
            df = self.mt5.get_rates(self.symbol, mt5.TIMEFRAME_M5, 200)
            if df is None:
                print(f"❌ Failed to get market data")
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            df = df.dropna()
            
            if len(df) < self.lookback_window:
                print(f"⚠️ Insufficient data ({len(df)} < {self.lookback_window})")
                return None
            
            # Get current position (try different symbol formats)
            symbol_info, actual_symbol = self.mt5.get_symbol_info(self.symbol)
            if symbol_info is None:
                print(f"❌ Cannot get symbol info for {self.symbol}")
                return None
            
            positions = self.mt5.get_positions(actual_symbol)
            current_position = 0
            if positions:
                buy_positions = [p for p in positions if p.type == mt5.POSITION_TYPE_BUY]
                sell_positions = [p for p in positions if p.type == mt5.POSITION_TYPE_SELL]
                
                if len(buy_positions) > 0 and len(sell_positions) == 0:
                    current_position = 1  # Net long
                elif len(sell_positions) > 0 and len(buy_positions) == 0:
                    current_position = -1 # Net short
                # If mixed, current_position remains 0 (neutral), which is a safe state for the model.
            
            # Prepare observation
            observation = self.prepare_observation(df, current_position)
            
            # Get model prediction
            action, confidence, raw_action = self.predict_action(observation, current_position)
            
            # Log prediction
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Get current price using the correct symbol format
            tick = mt5.symbol_info_tick(actual_symbol)
            if tick is None:
                print(f"⚠️ Cannot get current price for {actual_symbol}")
                current_price = df['close'].iloc[-1]  # Use last close price
            else:
                current_price = tick.bid
            
            print(f"🕐 {current_time} | {actual_symbol} @ {current_price:.5f}")
            print(f"   🤖 Prediction: {action_names[action]} (confidence: {confidence:.2f}, raw: {raw_action:.3f})")
            print(f"   📊 Net Position State: {current_position} | Open Positions: {len(positions) if positions else 0}")
            
            # 🛑 CHECK FOR STOP LOSS HITS AND MANAGE COOLDOWN
            self.check_for_sl_hits()
            
            # Show cooldown status if active
            cooldown_status = self.get_cooldown_status()
            if cooldown_status['active']:
                print(f"   🛑 COOLDOWN: {cooldown_status['remaining_minutes']:.1f} min remaining (SL hits: {cooldown_status['sl_hits_today']})")
            
            # Show detailed positions summary for multi-position trading
            if len(positions) > 0:
                summary = self.get_positions_summary()
                print(f"   💰 Portfolio: {summary['buy_positions']} BUY, {summary['sell_positions']} SELL | Net: {summary['net_exposure']:.2f} lots")
                
                # 🎯 MANAGE ADVANCED RISK FEATURES (trailing stop & break-even)
                self.manage_advanced_risk()
                self._cleanup_closed_positions()
            
            # Show bias report every 5 predictions
            if len(getattr(self, '_recent_actions', [])) % 5 == 0 and len(getattr(self, '_recent_actions', [])) > 0:
                print("\n" + self.get_bias_report() + "\n")
                
            # Show health check every 10 predictions or if severe bias detected
            bias_count = getattr(self, '_bias_counter', 0)
            if (len(getattr(self, '_recent_actions', [])) % 10 == 0 and len(getattr(self, '_recent_actions', [])) > 0) or bias_count >= 10:
                print("\n" + self.check_model_health() + "\n")
                
                # Check if model needs retraining
                if bias_count >= 10 or getattr(self, '_force_retrain', False):
                    print("\n" + "🚨 CRITICAL MODEL FAILURE DETECTED" + "\n")
                    should_retrain = self.should_retrain_model()
                    if should_retrain:
                        print("\n" + self.get_retraining_recommendation() + "\n")
                        print("🛑 STOPPING TRADING - MODEL UNSAFE FOR LIVE TRADING")
                        return None
            
            # Execute trade if conditions are met
            result = None
            executed = False
            if action != 0:  # Not hold
                print(f"   🚀 Non-HOLD action detected, calling execute_trade...")
                result = self.execute_trade(action, confidence)
                print(f"   🔍 DEBUG: execute_trade returned: {type(result)} - {result}")
                if result is not None:
                    if isinstance(result, bool):
                        # Handle boolean result (success/failure)
                        if result:
                            print(f"   ✅ Trade action completed: {action_names[action]}")
                            executed = True
                        else:
                            print(f"   ❌ Trade not executed: {action_names[action]} (conditions not met)")
                        # Show updated positions after trade
                        self.print_positions_report()
                    elif hasattr(result, 'retcode'):
                        # Handle MT5 trade result object
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"   ✅ Trade executed successfully: {action_names[action]}")
                            print(f"      📊 Order: {result.order}, Volume: {result.volume}, Price: {result.price:.5f}")
                            executed = True
                        else:
                            print(f"   ❌ Trade failed: {action_names[action]} - Code: {result.retcode}, Comment: {result.comment}")
                        # Show updated positions after trade
                        self.print_positions_report()
                    else:
                        print(f"   ✅ Trade action completed: {action_names[action]}")
                        executed = True
                        # Show updated positions after trade
                        self.print_positions_report()
                else:
                    print(f"   ⚠️ Trade not executed: {action_names[action]} (conditions not met - see details above)")
            else:
                print(f"   💤 HOLD action - no trade needed")
                result = True  # HOLD is always successful
                executed = True
            
            return {
                'action': action,
                'confidence': confidence,
                'raw_action': raw_action,
                'current_price': current_price,
                'result': executed  # Return boolean result instead of complex object
            }
            
        except Exception as e:
            print(f"❌ Single iteration error: {e}")
            return None
    
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
                action, confidence, raw_action = self.predict_action(observation, current_position)
                
                # Log prediction
                action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
                current_time = datetime.now().strftime('%H:%M:%S')
                current_price = mt5.symbol_info_tick(self.symbol).bid
                
                print(f"\n🕐 {current_time} | {self.symbol} @ {current_price:.5f}")
                print(f"   🤖 Prediction: {action_names[action]} (confidence: {confidence:.2f}, raw: {raw_action:.3f})")
                print(f"   📊 Position: {current_position} | Positions: {len(positions) if positions else 0}")
                
                # 🛑 CHECK FOR STOP LOSS HITS AND MANAGE COOLDOWN
                self.check_for_sl_hits()
                
                # Show cooldown status if active
                cooldown_status = self.get_cooldown_status()
                if cooldown_status['active']:
                    print(f"   🛑 COOLDOWN: {cooldown_status['remaining_minutes']:.1f} min remaining (SL hits: {cooldown_status['sl_hits_today']})")
                
                # 🎯 MANAGE ADVANCED RISK FEATURES (trailing stop & break-even)
                if positions:  # Only manage risk if we have positions
                    self.manage_advanced_risk()
                    self._cleanup_closed_positions()
                
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
    
    def manage_advanced_risk(self):
        """🎯 Manage trailing stops and break-even stops for all positions"""
        if not (self.enable_trailing_stop or self.enable_breakeven_stop):
            return
        
        positions = self.mt5.get_positions(self.symbol)
        if not positions:
            return
        
        symbol_info, actual_symbol = self.mt5.get_symbol_info(self.symbol)
        if not symbol_info:
            return
        
        current_tick = mt5.symbol_info_tick(actual_symbol)
        if not current_tick:
            return
        
        for position in positions:
            self._manage_position_risk(position, symbol_info, current_tick)
    
    def _manage_position_risk(self, position, symbol_info, current_tick):
        """Manage risk for individual position (trailing stop & break-even)"""
        ticket = position.ticket
        pos_type = position.type
        open_price = position.price_open
        current_sl = position.sl
        current_tp = position.tp
        profit_usd = position.profit
        
        # Get current market price
        if pos_type == mt5.POSITION_TYPE_BUY:
            current_price = current_tick.bid
            is_profitable = current_price > open_price
        else:  # SELL
            current_price = current_tick.ask
            is_profitable = current_price < open_price
            
        # 🎯 Initialize tracking for new positions
        if ticket not in self.position_tracking:
            self.position_tracking[ticket] = {
                'breakeven_set': False,
                'highest_profit': profit_usd,
                'best_price': current_price,
                'created_time': position.time
            }
            print(f"🆕 Started tracking position #{ticket} (${profit_usd:.2f} profit)")
        
        tracking = self.position_tracking[ticket]
        new_sl = current_sl
        modified = False
        
        # Always update the highest profit and best price for tracking
        if profit_usd > tracking['highest_profit']:
            tracking['highest_profit'] = profit_usd
            
        if pos_type == mt5.POSITION_TYPE_BUY and current_price > tracking['best_price']:
            tracking['best_price'] = current_price
        elif pos_type == mt5.POSITION_TYPE_SELL and current_price < tracking['best_price']:
            tracking['best_price'] = current_price
        
        # 💰 BREAK-EVEN STOP: Move SL to entry when profit >= $4
        if (self.enable_breakeven_stop and 
            profit_usd >= self.breakeven_profit_threshold and
            not tracking['breakeven_set']):
            
            # Check if we haven't already set break-even
            tolerance = symbol_info.point * 5  # 5 pip tolerance for break-even check
            entry_distance = abs(current_sl - open_price) if current_sl != 0 else float('inf')
            
            if entry_distance > tolerance:  # SL is not at break-even yet
                new_sl = open_price
                tracking['breakeven_set'] = True
                modified = True
                print(f"💰 BREAK-EVEN activated for #{ticket}: Profit ${profit_usd:.2f} >= ${self.breakeven_profit_threshold}")
                print(f"   🛑 Moving SL from {current_sl:.5f} to break-even {new_sl:.5f}")
        
        # 🏃 TRAILING STOP: Move SL with favorable price movement (after break-even is set)
        elif (self.enable_trailing_stop and 
              is_profitable and 
              tracking['breakeven_set'] and
              profit_usd > self.breakeven_profit_threshold):  # Continue trailing above break-even threshold
            
            if pos_type == mt5.POSITION_TYPE_BUY:
                # For BUY: Trail SL upward as price rises
                # Calculate new trailing SL
                trail_distance = symbol_info.point * self.trailing_stop_distance_pips
                potential_sl = current_price - trail_distance
                
                # Only move SL up (never down) and must be above current SL and entry price
                if potential_sl > current_sl and potential_sl >= open_price:
                    new_sl = potential_sl
                    modified = True
                    print(f"🏃 TRAILING STOP (BUY) for #{ticket}: SL {current_sl:.5f} → {new_sl:.5f}")
                    print(f"   📈 Price: {current_price:.5f}, Profit: ${profit_usd:.2f}")
            
            else:  # SELL position
                # For SELL: Trail SL downward as price falls
                # Calculate new trailing SL
                trail_distance = symbol_info.point * self.trailing_stop_distance_pips
                potential_sl = current_price + trail_distance
                
                # Only move SL down (never up) for SELL and must be below current SL and entry price
                if (current_sl == 0 or potential_sl < current_sl) and potential_sl <= open_price:
                    new_sl = potential_sl
                    modified = True
                    print(f"🏃 TRAILING STOP (SELL) for #{ticket}: SL {current_sl:.5f} → {new_sl:.5f}")
                    print(f"   📉 Price: {current_price:.5f}, Profit: ${profit_usd:.2f}")
        
        # 📊 Debug info for positions with significant profit
        if profit_usd >= self.breakeven_profit_threshold:
            be_status = "✅ SET" if tracking['breakeven_set'] else "⏳ PENDING"
            trail_status = "🏃 ACTIVE" if tracking['breakeven_set'] else "⏸️ WAITING"
            print(f"💰 Position #{ticket}: ${profit_usd:.2f} profit | Break-even: {be_status} | Trailing: {trail_status}")
            print(f"   📊 Current: {current_price:.5f} | SL: {current_sl:.5f} | Entry: {open_price:.5f}")
        elif ticket in self.position_tracking:
            print(f"📊 Position #{ticket}: ${profit_usd:.2f} profit (need ${self.breakeven_profit_threshold:.2f} for break-even)")
        
        # Apply the modification if needed
        if modified and new_sl != current_sl:
            result = self.mt5.modify_position_sl_tp(ticket, new_sl=new_sl)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position #{ticket} SL updated successfully")
            else:
                print(f"❌ Failed to update SL for position #{ticket}: {result.retcode if result else 'No result'}")
                if result:
                    print(f"   Error: {result.comment}")
        elif not modified and profit_usd >= self.breakeven_profit_threshold:
            # Show why no modification was made
            if not self.enable_breakeven_stop and not self.enable_trailing_stop:
                print(f"⚠️ Risk management disabled for position #{ticket}")
            elif not tracking['breakeven_set']:
                print(f"⏳ Position #{ticket} waiting for break-even activation")
            else:
                print(f"👀 Position #{ticket} being monitored for trailing opportunities")
    
    def _cleanup_closed_positions(self):
        """Remove tracking data for closed positions"""
        if not self.position_tracking:
            return
        
        current_tickets = set()
        positions = self.mt5.get_positions(self.symbol)
        if positions:
            current_tickets = {pos.ticket for pos in positions}
        
        # Remove tracking for closed positions
        closed_tickets = set(self.position_tracking.keys()) - current_tickets
        for ticket in closed_tickets:
            del self.position_tracking[ticket]
            print(f"🧹 Cleaned tracking data for closed position #{ticket}")
    
    def get_risk_management_report(self):
        """📊 Get detailed risk management report"""
        if not self.position_tracking:
            return "📊 No positions currently being tracked"
        
        report = []
        report.append("🎯 ADVANCED RISK MANAGEMENT REPORT")
        report.append("=" * 60)
        report.append(f"💰 Break-even threshold: ${self.breakeven_profit_threshold} USD")
        report.append(f"🏃 Trailing stop distance: {self.trailing_stop_distance_pips} pips")
        report.append(f"✅ Break-even stop: {'ENABLED' if self.enable_breakeven_stop else 'DISABLED'}")
        report.append(f"🏃 Trailing stop: {'ENABLED' if self.enable_trailing_stop else 'DISABLED'}")
        report.append("")
        
        # Get current positions
        positions = self.mt5.get_positions(self.symbol)
        if not positions:
            report.append("📭 No open positions")
            return "\n".join(report)
            
        report.append("📊 POSITION TRACKING STATUS:")
        report.append("-" * 60)
        
        for position in positions:
            ticket = position.ticket
            profit = position.profit
            
            if ticket in self.position_tracking:
                tracking = self.position_tracking[ticket]
                be_status = "✅ SET" if tracking['breakeven_set'] else "⏳ PENDING"
                
                pos_type = "BUY" if position.type == 0 else "SELL"
                entry_price = position.price_open
                current_sl = position.sl if position.sl != 0 else "None"
                
                report.append(f"#{ticket} | {pos_type} | Entry: {entry_price:.5f} | SL: {current_sl}")
                report.append(f"   💰 Profit: ${profit:.2f} | Break-even: {be_status}")
                report.append(f"   📈 Best Price: {tracking['best_price']:.5f} | Max Profit: ${tracking['highest_profit']:.2f}")
                
                if profit >= self.breakeven_profit_threshold:
                    if not tracking['breakeven_set']:
                        report.append(f"   🚨 ACTION REQUIRED: Position should trigger break-even!")
                    else:
                        report.append(f"   ✅ Break-even protection active")
                
                report.append("")
            else:
                report.append(f"#{ticket} | ⚠️ NOT TRACKED (New position detected)")
                report.append("")
        
        return "\n".join(report)
    
    def force_risk_management_check(self):
        """🔧 Force immediate risk management check (for debugging)"""
        print("🔧 FORCING RISK MANAGEMENT CHECK...")
        print("=" * 50)
        
        positions = self.mt5.get_positions(self.symbol)
        if not positions:
            print("📭 No positions to manage")
            return
            
        print(f"📊 Found {len(positions)} positions to check")
        
        # Force check each position
        for i, position in enumerate(positions, 1):
            print(f"\n🔍 Checking position {i}/{len(positions)}: #{position.ticket}")
            print(f"   💰 Current profit: ${position.profit:.2f}")
            print(f"   🎯 Break-even threshold: ${self.breakeven_profit_threshold}")
            
            if position.profit >= self.breakeven_profit_threshold:
                print(f"   ✅ Position qualifies for break-even stop!")
            else:
                needed = self.breakeven_profit_threshold - position.profit
                print(f"   ⏳ Needs ${needed:.2f} more profit for break-even")
        
        # Run the actual management
        print(f"\n🎯 Running risk management...")
        self.manage_advanced_risk()
        
        # Show final status
        print(f"\n📊 Final tracking status:")
        for ticket, data in self.position_tracking.items():
            be_status = "✅ SET" if data['breakeven_set'] else "❌ NOT SET"
            print(f"   #{ticket}: Break-even {be_status}, Max profit: ${data['highest_profit']:.2f}")
    
    def check_for_sl_hits(self):
        """🛑 Check for Stop Loss hits and manage cooldown"""
        if not self.enable_sl_cooldown:
            return
        
        current_positions = self.mt5.get_positions(self.symbol)
        current_tickets = set()
        if current_positions:
            current_tickets = {pos.ticket for pos in current_positions}
        
        # Check for closed positions (potential SL hits)
        tracked_tickets = set(self.position_tracking.keys())
        recently_closed = tracked_tickets - current_tickets
        
        for ticket in recently_closed:
            if ticket in self.position_tracking:
                tracking_data = self.position_tracking[ticket]
                
                # If position was profitable but suddenly closed, likely SL hit
                if tracking_data['highest_profit'] > 0:
                    # Check if it was closed at a loss from highest profit
                    # This indicates potential SL activation
                    print(f"🛑 POTENTIAL SL HIT DETECTED: Position #{ticket}")
                    print(f"   📊 Highest profit reached: ${tracking_data['highest_profit']:.2f}")
                    
                    self._handle_sl_hit(ticket, tracking_data)
    
    def _handle_sl_hit(self, ticket, tracking_data):
        """Handle Stop Loss hit - activate cooldown"""
        from datetime import datetime, timedelta
        
        self.sl_hit_count += 1
        self.sl_cooldown_end_time = datetime.now() + timedelta(minutes=self.sl_cooldown_minutes)
        
        print(f"🛑 STOP LOSS COOLDOWN ACTIVATED!")
        print(f"   🎫 Position: #{ticket}")
        print(f"   📈 Max profit was: ${tracking_data['highest_profit']:.2f}")
        print(f"   ⏰ Trading paused until: {self.sl_cooldown_end_time.strftime('%H:%M:%S')}")
        print(f"   🕐 Duration: {self.sl_cooldown_minutes} minutes")
        print(f"   📊 SL hits today: {self.sl_hit_count}")
        
        # Store the closed position data
        self.recent_closed_positions[ticket] = {
            'closed_time': datetime.now(),
            'max_profit': tracking_data['highest_profit'],
            'reason': 'potential_sl_hit'
        }
    
    def is_in_cooldown(self):
        """Check if trading is currently in cooldown period"""
        if not self.enable_sl_cooldown or self.sl_cooldown_end_time is None:
            return False, 0  # Always return tuple
        
        from datetime import datetime
        
        if datetime.now() < self.sl_cooldown_end_time:
            remaining = self.sl_cooldown_end_time - datetime.now()
            remaining_minutes = remaining.total_seconds() / 60
            return True, remaining_minutes
        else:
            # Cooldown expired
            self.sl_cooldown_end_time = None
            return False, 0
    
    def get_cooldown_status(self):
        """Get detailed cooldown status"""
        in_cooldown, remaining_minutes = self.is_in_cooldown()
        
        if in_cooldown:
            return {
                'active': True,
                'remaining_minutes': remaining_minutes,
                'end_time': self.sl_cooldown_end_time,
                'sl_hits_today': self.sl_hit_count,
                'reason': 'Stop Loss hit detected'
            }
        else:
            return {
                'active': False,
                'remaining_minutes': 0,
                'end_time': None,
                'sl_hits_today': self.sl_hit_count,
                'reason': None
            }

    def stop(self):
        """Stop the trading bot"""
        self.running = False
        self.mt5.disconnect()

def main():
    """Example usage - MULTI-POSITION TRADING BOT"""
    print("🤖 RL Multi-Position Trading Bot for MT5")
    print("=" * 50)
    print("🚀 FEATURES:")
    print("   • Can hold up to 6 concurrent positions")
    print("   • AI manages each position intelligently")
    print("   • FIFO closing strategy (oldest first)")
    print("   • Supports hedging (BUY + SELL simultaneously)")
    print("=" * 50)
    
    # Configuration
    SYMBOL = "EURUSDm"  # Change to your symbol
    RISK_PERCENT = 1.0  # Risk per trade in %
    CHECK_INTERVAL = 5  # Minutes between checks
    
    # MT5 Login credentials (set these)
    LOGIN = None  # Your MT5 login
    PASSWORD = None  # Your MT5 password  
    SERVER = None  # Your MT5 server
    
    try:
        # Initialize bot
        print(f"\n🔧 Initializing Multi-Position Bot...")
        bot = TradingBot(SYMBOL, risk_percent=RISK_PERCENT)
        
        # Connect to MT5
        print(f"\n🔌 Connecting to MT5...")
        if not bot.connect_mt5(LOGIN, PASSWORD, SERVER):
            print("❌ Failed to connect to MT5")
            return
        
        # Show initial positions (if any)
        bot.print_positions_report()
        
        print(f"\n🚀 Choose your mode:")
        print(f"   1. Single iteration (testing)")
        print(f"   2. Continuous trading loop")
        
        # For demo purposes, let's run single iteration
        # You can modify this to run continuous loop
        
        print(f"\n🧪 Running single iteration for testing...")
        result = bot.run_single_iteration()
        
        if result:
            print(f"\n✅ Iteration completed!")
            print(f"   Action: {result['action']}")
            print(f"   Price: {result['current_price']:.5f}")
            if result['result']:
                print(f"   Trade Result: {result['result']}")
        
        # Uncomment this to run continuous trading:
        # print(f"\n🔄 Starting continuous trading...")
        # bot.run_trading_loop(CHECK_INTERVAL)
        
    except Exception as e:
        print(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'bot' in locals():
            bot.stop()

# Example of how to run the bot for specific scenarios
def example_multi_position_usage():
    """Example of advanced multi-position usage"""
    bot = TradingBot("EURUSDm", risk_percent=0.5)  # Lower risk per trade
    
    # Connect
    if bot.connect_mt5():
        print("🔍 Testing multi-position capabilities...")
        
        # Show current status
        bot.print_positions_report()
        
        # Run a few iterations to see how it handles multiple positions
        for i in range(5):
            print(f"\n--- Iteration {i+1} ---")
            result = bot.run_single_iteration()
            if result:
                print(f"Action taken: {result['action']}")
            
            # Small delay between iterations
            import time
            time.sleep(2)
        
        # Final report
        print(f"\n🏁 Final Positions Report:")
        bot.print_positions_report()
        
    bot.stop()

if __name__ == "__main__":
    main()
    
    # Uncomment to run advanced example:
    # example_multi_position_usage()
