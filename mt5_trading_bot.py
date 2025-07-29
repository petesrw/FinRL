#!/usr/bin/env python3
"""
🤖 MT5 Trading Bot with Trained RL Model
Loads trained models and executes real trades on MT5
"""

import os
import json
import time
import pandas as pd
import numpy as np,sys
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from stable_baselines3 import PPO, SAC, A2C
import warnings

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
        """Send trading order with symbol format detection"""
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
    
    def __init__(self, symbol, model_path=None, risk_percent=1.0):
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.model_loader = ModelLoader()
        self.mt5 = MT5Interface()
        self.model = None
        self.running = False
        
        # Trading parameters
        self.min_confidence = 0.4  # Minimum confidence for trade execution (lowered to match training behavior)
        self.max_positions = 6     # Maximum concurrent positions
        self.lookback_window = 50  # Default, will be updated based on model
        
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
        """Generate a comprehensive bias report"""
        if not hasattr(self, '_recent_actions') or len(self._recent_actions) < 5:
            return "📊 Insufficient data for bias analysis (need at least 5 predictions)"
        
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
        
        # Count actions
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for a in self._recent_actions:
            action_counts[a] += 1
        
        total_predictions = len(self._recent_actions)
        report = []
        report.append(f"📊 BIAS ANALYSIS REPORT ({total_predictions} recent predictions)")
        report.append("=" * 50)
        
        # Action distribution
        for action_id, count in action_counts.items():
            percentage = (count / total_predictions) * 100
            action_name = action_names[action_id]
            status = "🚨" if percentage > 60 else "⚠️" if percentage > 40 else "✅"
            report.append(f"{status} {action_name}: {count}/{total_predictions} ({percentage:.1f}%)")
        
        # Check for patterns
        if len(self._recent_actions) >= 5:
            last_5 = self._recent_actions[-5:]
            if all(a == last_5[0] for a in last_5):
                report.append(f"🚨 CRITICAL: Last 5 actions identical: {action_names[last_5[0]]}")
        
        # Raw action analysis
        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= 5:
            raw_actions = self._recent_raw_actions[-10:]
            avg_raw = np.mean(raw_actions)
            std_raw = np.std(raw_actions)
            report.append(f"📈 Raw actions: avg={avg_raw:.3f}, std={std_raw:.3f}")
            if std_raw < 0.1:
                report.append("🚨 CRITICAL: Raw action variance too low - model not exploring")
        
        # Bias counter
        bias_count = getattr(self, '_bias_counter', 0)
        if bias_count > 0:
            report.append(f"⚠️ Consecutive bias detections: {bias_count}")
        
        # Exploration mode status
        if getattr(self, '_exploration_mode', False):
            report.append("🎲 Exploration mode: ACTIVE")
        else:
            report.append("🎯 Exploration mode: INACTIVE")
        
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
    
    def check_model_health(self):
        """Comprehensive model health check"""
        health_report = []
        health_report.append("🏥 MODEL HEALTH CHECK")
        health_report.append("=" * 30)
        
        # Check recent predictions variance
        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= 5:
            recent_raw = self._recent_raw_actions[-10:]
            variance = np.var(recent_raw)
            mean_raw = np.mean(recent_raw)
            
            health_report.append(f"📊 Raw action variance: {variance:.6f}")
            health_report.append(f"📊 Mean raw action: {mean_raw:.3f}")
            
            if variance < 0.001:
                health_report.append("🚨 CRITICAL: No variance in predictions - MODEL DEAD")
                self.force_model_retraining_mode(True)
            elif variance < 0.01:
                health_report.append("⚠️ WARNING: Very low variance - model may be stuck")
            else:
                health_report.append("✅ Variance within acceptable range")
        
        # Check bias counter
        bias_count = getattr(self, '_bias_counter', 0)
        if bias_count >= 10:
            health_report.append("🚨 CRITICAL: Severe bias detected - STOP TRADING")
            self.force_model_retraining_mode(True)
        elif bias_count >= 5:
            health_report.append("⚠️ WARNING: High bias count")
        else:
            health_report.append("✅ Bias count acceptable")
        
        # Check exploration mode necessity
        if getattr(self, '_exploration_mode', False):
            health_report.append("🎲 INFO: Exploration mode active (bias mitigation)")
        
        return "\n".join(health_report)
    
    def should_retrain_model(self):
        """ตรวจสอบว่าควร retrain model หรือไม่"""
        retrain_reasons = []
        
        print("🔍 CHECKING IF MODEL NEEDS RETRAINING")
        print("=" * 40)
        
        # ตรวจสอบ bias counter
        bias_count = getattr(self, '_bias_counter', 0)
        if bias_count >= 10:
            retrain_reasons.append("🚨 SEVERE BIAS: Bias counter >= 10")
        elif bias_count >= 5:
            retrain_reasons.append("⚠️ HIGH BIAS: Bias counter >= 5")
        
        # ตรวจสอบ variance ของ raw actions
        if hasattr(self, '_recent_raw_actions') and len(self._recent_raw_actions) >= 10:
            variance = np.var(self._recent_raw_actions)
            if variance < 0.001:
                retrain_reasons.append("🚨 MODEL DEATH: No variance in predictions")
            elif variance < 0.01:
                retrain_reasons.append("⚠️ LOW VARIANCE: Model may be stuck")
        
        # ตรวจสอบ action pattern
        if hasattr(self, '_recent_actions') and len(self._recent_actions) >= 10:
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for a in self._recent_actions:
                action_counts[a] += 1
            
            max_count = max(action_counts.values())
            if max_count >= 9:  # 90% of actions are the same
                dominant_action = max(action_counts, key=action_counts.get)
                action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
                retrain_reasons.append(f"🚨 EXTREME BIAS: {action_names[dominant_action]} dominates {max_count}/10 actions")
        
        # ตรวจสอบ exploration mode ที่เปิดอยู่นาน
        if getattr(self, '_exploration_mode', False):
            retrain_reasons.append("⚠️ PERSISTENT EXPLORATION: Model requires exploration mode")
        
        # ตรวจสอบ emergency mode
        if getattr(self, '_emergency_mode', False):
            retrain_reasons.append("🚨 EMERGENCY MODE: Model completely failed")
        
        # สรุปผลการตรวจสอบ
        if retrain_reasons:
            print("❌ MODEL NEEDS RETRAINING")
            print("📋 Reasons:")
            for reason in retrain_reasons:
                print(f"   {reason}")
            
            # แนะนำขั้นตอนต่อไป
            print("\n📋 RECOMMENDED ACTIONS:")
            print("   1. 🛑 STOP live trading immediately")
            print("   2. 📦 Backup current model")
            print("   3. 🔄 Start fresh model training")
            print("   4. 📊 Use different hyperparameters")
            print("   5. 🧪 Validate new model thoroughly")
            
            return True
        else:
            print("✅ MODEL APPEARS HEALTHY")
            print("   Continue monitoring performance")
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
        symbol_info_data = self.mt5.get_symbol_info(self.symbol)
        if symbol_info_data is None or symbol_info_data[0] is None:
            return 0.01
        
        # Extract symbol_info object from tuple
        symbol_info, actual_symbol = symbol_info_data
        
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
            
            # Auto-enable exploration mode after 3 consecutive bias detections
            if self._bias_counter >= 3 and not self._exploration_mode:
                print(f"   🎲 Auto-enabling exploration mode after {self._bias_counter} bias detections")
                self.enable_exploration_mode(True)
            
            # Use non-deterministic prediction instead
            action_value = action_nd_value
            print(f"   🔄 Using non-deterministic prediction: {action_value:.6f}")
        else:
            # Reset bias counter if no bias detected
            self._bias_counter = 0
            # If in exploration mode, still use non-deterministic occasionally
            if self._exploration_mode:
                action_value = action_nd_value
                print(f"   🎲 Exploration mode: Using non-deterministic {action_value:.6f}")
        
        # 🎯 EXACT TRAINING MAPPING: 
        # [-1, -0.3): Sell (35% of action space)
        # [-0.3, 0.3): Hold (30% of action space)  
        # [0.3, 0.7): Buy (40% of action space)
        # [0.7, 1]: Close (30% of action space)
        if action_value < -0.3:
            discrete_action = 2  # Sell
        elif action_value < 0.3:
            discrete_action = 0  # Hold (balanced zone)
        elif action_value < 0.7:
            discrete_action = 1  # Buy
        else:
            discrete_action = 3  # Close
        
        # 🎯 EXACT TRAINING CONFIDENCE CALCULATION
        confidence = 1.0  # Default confidence
        
        if discrete_action == 1 or discrete_action == 2:  # Only for Buy/Sell actions
            if discrete_action == 2:  # Sell
                confidence = abs(action_value + 0.65) / 0.7  # Exact training formula
            elif discrete_action == 1:  # Buy
                confidence = (action_value - 0.3) / 0.4      # Exact training formula
        else:
            # For Hold and Close, use simple confidence
            confidence = 0.5  # Neutral confidence for Hold/Close
        
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        # Track action patterns to detect if model is stuck
        self._recent_actions.append(discrete_action)
        self._recent_raw_actions = getattr(self, '_recent_raw_actions', [])
        self._recent_raw_actions.append(action_value)
        
        if len(self._recent_actions) > 10:
            self._recent_actions.pop(0)
            self._recent_raw_actions.pop(0)
        
        # 🚨 ADVANCED BIAS DETECTION
        if len(self._recent_actions) >= 5:
            last_5_actions = self._recent_actions[-5:]
            last_5_raw = self._recent_raw_actions[-5:]
            
            # Check for identical actions
            if all(a == last_5_actions[0] for a in last_5_actions):
                print(f"   🚨 BIAS ALERT: Model stuck in pattern - last 5 actions: {last_5_actions}")
                print(f"   📊 Raw action values: {[f'{r:.3f}' for r in last_5_raw]}")
                
                # Check if raw values are also similar (indicating model convergence issue)
                if all(abs(r - last_5_raw[0]) < 0.1 for r in last_5_raw):
                    print(f"   🚨 SEVERE BIAS: Raw values also identical - Model needs retraining!")
                    
        # Calculate action distribution for the last 10 predictions
        if len(self._recent_actions) >= 10:
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # HOLD, BUY, SELL, CLOSE
            for a in self._recent_actions:
                action_counts[a] += 1
            
            action_names_debug = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            distribution = {action_names_debug[k]: v for k, v in action_counts.items()}
            print(f"   📊 Last 10 actions distribution: {distribution}")
            
            # Check for extreme bias (>80% of one action)
            max_count = max(action_counts.values())
            if max_count >= 8:
                dominant_action = max(action_counts, key=action_counts.get)
                print(f"   🚨 EXTREME BIAS: {action_names_debug[dominant_action]} dominates {max_count}/10 predictions!")
        
        return discrete_action, confidence, action_value
    
    def execute_trade(self, action, confidence):
        """Execute trade based on model prediction - MATCH TRAINING LOGIC"""
        print(f"   🎯 EXECUTING TRADE: Action={action}, Confidence={confidence:.3f}")
        
        # 🎯 EXACT TRAINING CONFIDENCE CHECK
        # Apply minimum confidence threshold like training (0.4)
        # Only check confidence for Buy/Sell actions, not Hold/Close
        if (action == 1 or action == 2) and confidence < self.min_confidence:
            print(f"   ⚠️ Low confidence ({confidence:.2f}), skipping trade")
            return None
        
        print(f"   ✅ Confidence check passed ({confidence:.2f} >= {self.min_confidence})")
        
        # Check current positions
        positions = self.mt5.get_positions(self.symbol)
        current_position = 0
        if positions:
            if len(positions) >= self.max_positions:
                print(f"   ⚠️ Maximum positions ({self.max_positions}) reached")
                return None
            current_position = 1 if positions[0].type == mt5.POSITION_TYPE_BUY else -1
        
        print(f"   📊 Position check: current={current_position}, max={self.max_positions}")
        
        # Calculate position size and SL/TP
        print(f"   💰 Calculating position size...")
        volume = self.calculate_position_size()
        print(f"   💰 Position size: {volume}")
        
        print(f"   📈 Getting symbol info for {self.symbol}...")
        symbol_info, actual_symbol = self.mt5.get_symbol_info(self.symbol)
        current_price = mt5.symbol_info_tick(actual_symbol)
        
        if symbol_info is None or current_price is None:
            print(f"   ❌ Failed to get symbol/price info (symbol_info: {symbol_info is not None}, price: {current_price is not None})")
            return None
        
        print(f"   ✅ Symbol info obtained for {actual_symbol}")
        
        # Calculate SL/TP based on ATR or fixed percentage
        atr_multiplier = 2.0
        # Fix: symbol_info is the object, not tuple
        sl_distance = symbol_info.point * 100  # Default 100 points
        tp_distance = sl_distance * 1.5  # 1:1.5 risk/reward
        
        result = None
        
        # 🚨 IMPROVED LOGIC: Check current position before executing actions
        if action == 1 and current_position == 0:  # Buy signal - only if no position
            print(f"   💹 Executing BUY order...")
            sl = current_price.ask - sl_distance
            tp = current_price.ask + tp_distance
            result = self.mt5.send_order(
                actual_symbol, 
                mt5.ORDER_TYPE_BUY, 
                volume, 
                sl=sl, 
                tp=tp,
                comment=f"RL Buy C:{confidence:.2f}"
            )
            
        elif action == 2 and current_position == 0:  # Sell signal - only if no position
            print(f"   💹 Executing SELL order...")
            sl = current_price.bid + sl_distance
            tp = current_price.bid - tp_distance
            result = self.mt5.send_order(
                actual_symbol, 
                mt5.ORDER_TYPE_SELL, 
                volume,
                sl=sl,
                tp=tp, 
                comment=f"RL Sell C:{confidence:.2f}"
            )
            
        elif action == 1 and current_position != 0:  # Buy signal but position exists
            print(f"   ⚠️ BUY signal ignored - position already exists (current: {current_position})")
            if current_position == -1:  # Have sell position, could close first
                print(f"   💡 Suggestion: Close SELL position first, then BUY")
            return None
            
        elif action == 2 and current_position != 0:  # Sell signal but position exists
            print(f"   ⚠️ SELL signal ignored - position already exists (current: {current_position})")
            if current_position == 1:  # Have buy position, could close first
                print(f"   💡 Suggestion: Close BUY position first, then SELL")
            return None
            
        elif action == 3:  # Close signal
            if current_position != 0 and positions:
                print(f"   🔄 Closing position {positions[0].ticket}...")
                result = self.mt5.close_position(positions[0].ticket)
                if result is not None:
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"   ✅ Position closed successfully")
                    else:
                        print(f"   ❌ Position close failed: {result.retcode} - {result.comment}")
            else:
                print(f"   ⚠️ CLOSE action ignored - no position to close (current_position: {current_position})")
                return None
        else:
            # Log unhandled action combinations
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            print(f"   ⚠️ Action {action_names.get(action, action)} not executed - conditions not met")
            print(f"      Current position: {current_position}, Action: {action}")
            return None
        
        return result
    
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
                current_position = 1 if positions[0].type == mt5.POSITION_TYPE_BUY else -1
            
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
            print(f"   📊 Position: {current_position} | Positions: {len(positions) if positions else 0}")
            
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
            if action != 0:  # Not hold
                print(f"   🚀 Non-HOLD action detected, calling execute_trade...")
                result = self.execute_trade(action, confidence)
                if result is not None:
                    if hasattr(result, 'retcode'):
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"   ✅ Trade executed successfully: {action_names[action]}")
                            print(f"      📊 Order: {result.order}, Volume: {result.volume}, Price: {result.price:.5f}")
                        else:
                            print(f"   ❌ Trade failed: {action_names[action]} - Code: {result.retcode}, Comment: {result.comment}")
                    else:
                        print(f"   ✅ Trade action completed: {action_names[action]}")
                else:
                    print(f"   ⚠️ Trade not executed: {action_names[action]} (conditions not met - see details above)")
            else:
                print(f"   💤 HOLD action - no trade needed")
            
            return {
                'action': action,
                'confidence': confidence,
                'raw_action': raw_action,
                'current_price': current_price,
                'result': result
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
    SYMBOL = "EURUSDm"  # Change to your symbol
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
