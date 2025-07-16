#!/usr/bin/env python3
"""
🚀 Simple Forex RL System (No talib required)
Reinforcement Learning system with automatic indicator selection
Works without external dependencies for quick testing
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO, A2C, SAC
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import os

# Import our automatic indicator system
from indicator_manager import create_indicator_manager, SmartIndicatorManager
from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleForexEnvironment(gym.Env):
    """
    🚀 Simple Forex Environment with Automatic Indicator Selection
    Uses basic technical indicators without talib dependency
    """
    
    def __init__(self, symbol: str = "EURUSD", config=None, indicator_manager: SmartIndicatorManager = None):
        super().__init__()
        self.symbol = symbol
        self.config = config or get_config()
        self.indicator_manager = indicator_manager
        
        # Environment parameters
        self.initial_balance = self.config.model.initial_balance
        self.current_balance = self.initial_balance
        self.lookback_window = self.config.model.lookback_window
        
        # Trading state
        self.current_position = 0  # -1: Short, 0: Neutral, 1: Long
        self.position_entry_price = 0
        self.position_entry_step = 0
        self.total_trades = 0
        self.win_count = 0
        self.current_step = 0
        
        # Performance tracking
        self.episode_performance = []
        self.recent_trades = []
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # Hold, Buy, Sell, Close
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(30,), dtype=np.float32  # Reduced from 50 to 30 for simplicity
        )
        
        # Generate sample data
        self.generate_sample_data()
        
        # Calculate simple indicators
        self.calculate_simple_indicators()
        
        logger.info(f"Simple Forex Environment initialized for {symbol}")
    
    def generate_sample_data(self, days: int = 365):
        """Generate realistic forex sample data"""
        
        # Base price for different symbols
        base_prices = {
            "EURUSD": 1.1000, "GBPUSD": 1.3000, "USDJPY": 110.00,
            "XAUUSD": 1800.00, "USDCHF": 0.9200, "USDCAD": 1.2500
        }
        
        base_price = base_prices.get(self.symbol, 1.1000)
        
        # Generate time series (15-minute bars)
        periods = days * 96  # 96 bars per day
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=periods, freq='15min')
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results
        
        # Different volatility for different symbols
        volatility = {
            "EURUSD": 0.0008, "GBPUSD": 0.0012, "USDJPY": 0.008,
            "XAUUSD": 0.015,  "USDCHF": 0.0007, "USDCAD": 0.0009
        }.get(self.symbol, 0.0008)
        
        # Generate returns with trend and mean reversion
        returns = []
        trend = 0.0
        
        for i in range(periods):
            # Add trend persistence
            if i > 0:
                trend = 0.95 * trend + 0.05 * np.random.normal(0, volatility)
            else:
                trend = np.random.normal(0, volatility)
            
            # Add noise
            noise = np.random.normal(0, volatility)
            
            # Combine trend and noise
            daily_return = trend + noise
            returns.append(daily_return)
        
        # Convert to prices
        returns = np.array(returns)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        data = []
        for i, price in enumerate(prices):
            # Generate realistic OHLC from close price
            volatility_factor = np.random.uniform(0.5, 1.5)
            spread = volatility * volatility_factor * price
            
            high = price + np.random.uniform(0, spread)
            low = price - np.random.uniform(0, spread)
            
            # Ensure OHLC consistency
            if i == 0:
                open_price = price
            else:
                open_price = data[-1]['close']
            
            # Adjust high and low to include open and close
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            data.append({
                'time': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': np.random.randint(1000, 10000)
            })
        
        self.data = pd.DataFrame(data)
        logger.info(f"Generated {len(self.data)} data points for {self.symbol}")
    
    def calculate_simple_indicators(self):
        """Calculate simple technical indicators without talib"""
        
        # Get optimal configuration from indicator manager
        if self.indicator_manager:
            indicator_config = self.indicator_manager.get_optimal_config(self.data)
            logger.info(f"Using automatic indicator config: RSI={indicator_config.rsi_period}, MACD=({indicator_config.macd_fast},{indicator_config.macd_slow})")
        else:
            # Use default values
            class DefaultConfig:
                rsi_period = 14
                macd_fast = 12
                macd_slow = 26
                sma_fast = 20
                sma_slow = 50
                bb_period = 20
                bb_std = 2.0
                atr_period = 14
            
            indicator_config = DefaultConfig()
        
        self.indicators = {}
        
        try:
            # Simple RSI calculation
            self.indicators['rsi'] = self.calculate_rsi(self.data['close'], indicator_config.rsi_period)
            
            # Simple MACD calculation
            macd, macd_signal = self.calculate_macd(
                self.data['close'], 
                indicator_config.macd_fast, 
                indicator_config.macd_slow
            )
            self.indicators['macd'] = macd
            self.indicators['macd_signal'] = macd_signal
            
            # Simple Moving Averages
            self.indicators['sma_fast'] = self.data['close'].rolling(indicator_config.sma_fast).mean()
            self.indicators['sma_slow'] = self.data['close'].rolling(indicator_config.sma_slow).mean()
            
            # Simple Bollinger Bands
            bb_middle = self.data['close'].rolling(indicator_config.bb_period).mean()
            bb_std = self.data['close'].rolling(indicator_config.bb_period).std()
            self.indicators['bb_upper'] = bb_middle + (bb_std * indicator_config.bb_std)
            self.indicators['bb_lower'] = bb_middle - (bb_std * indicator_config.bb_std)
            self.indicators['bb_middle'] = bb_middle
            
            # Simple ATR calculation
            self.indicators['atr'] = self.calculate_atr(
                self.data['high'], 
                self.data['low'], 
                self.data['close'], 
                indicator_config.atr_period
            )
            
            logger.info("Simple technical indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Initialize with zeros if calculation fails
            for key in ['rsi', 'macd', 'macd_signal', 'sma_fast', 'sma_slow', 
                       'bb_upper', 'bb_lower', 'bb_middle', 'atr']:
                self.indicators[key] = pd.Series(np.zeros(len(self.data)), index=self.data.index)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate simple RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate simple MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate simple ATR"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(0.001)
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.current_balance = self.initial_balance
        self.current_position = 0
        self.position_entry_price = 0
        self.position_entry_step = 0
        self.total_trades = 0
        self.win_count = 0
        self.current_step = np.random.randint(100, len(self.data) - 1000)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation with automatic indicators"""
        if self.current_step >= len(self.data) or self.current_step < 50:
            return np.zeros(30, dtype=np.float32)
        
        features = []
        current_idx = self.current_step
        
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
                np.log(volume_ratio),
                (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
            ])
            
            # High-Low spread (1 feature)
            hl_spread = (self.data['high'].iloc[current_idx] - self.data['low'].iloc[current_idx]) / current_price
            features.append(hl_spread)
            
            # Position information (2 features)
            features.extend([
                float(self.current_position),
                (current_price - self.position_entry_price) / self.position_entry_price if self.position_entry_price > 0 else 0.0
            ])
            
            # Technical Indicators (20 features)
            if hasattr(self, 'indicators') and self.indicators:
                # RSI (1 feature)
                rsi_val = self.indicators['rsi'].iloc[current_idx] if not pd.isna(self.indicators['rsi'].iloc[current_idx]) else 50.0
                features.append((rsi_val - 50) / 50)
                
                # MACD (2 features)
                macd_val = self.indicators['macd'].iloc[current_idx] if not pd.isna(self.indicators['macd'].iloc[current_idx]) else 0.0
                macd_signal_val = self.indicators['macd_signal'].iloc[current_idx] if not pd.isna(self.indicators['macd_signal'].iloc[current_idx]) else 0.0
                
                features.extend([
                    macd_val / current_price * 10000,
                    macd_signal_val / current_price * 10000
                ])
                
                # Bollinger Bands (3 features)
                bb_upper_val = self.indicators['bb_upper'].iloc[current_idx] if not pd.isna(self.indicators['bb_upper'].iloc[current_idx]) else current_price
                bb_lower_val = self.indicators['bb_lower'].iloc[current_idx] if not pd.isna(self.indicators['bb_lower'].iloc[current_idx]) else current_price
                bb_middle_val = self.indicators['bb_middle'].iloc[current_idx] if not pd.isna(self.indicators['bb_middle'].iloc[current_idx]) else current_price
                
                bb_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val != bb_lower_val else 0.5
                bb_width = (bb_upper_val - bb_lower_val) / bb_middle_val if bb_middle_val > 0 else 0.0
                
                features.extend([bb_position, bb_width, float(bb_width < 0.02)])
                
                # Moving Averages (4 features)
                sma_fast_val = self.indicators['sma_fast'].iloc[current_idx] if not pd.isna(self.indicators['sma_fast'].iloc[current_idx]) else current_price
                sma_slow_val = self.indicators['sma_slow'].iloc[current_idx] if not pd.isna(self.indicators['sma_slow'].iloc[current_idx]) else current_price
                
                features.extend([
                    (current_price - sma_fast_val) / sma_fast_val,
                    (current_price - sma_slow_val) / sma_slow_val,
                    (sma_fast_val - sma_slow_val) / sma_slow_val,
                    1.0 if sma_fast_val > sma_slow_val else 0.0  # Trend direction
                ])
                
                # ATR (1 feature)
                atr_val = self.indicators['atr'].iloc[current_idx] if not pd.isna(self.indicators['atr'].iloc[current_idx]) else 0.001
                features.append(atr_val / current_price)
                
                # Market regime indicators (6 features)
                # Volatility
                recent_volatility = self.data['close'].iloc[max(0, current_idx-20):current_idx].pct_change().std()
                features.append(recent_volatility if not pd.isna(recent_volatility) else 0.01)
                
                # Trend strength
                trend_up = sum(1 for i in range(max(0, current_idx-10), current_idx) 
                              if i > 0 and self.data['close'].iloc[i] > self.data['close'].iloc[i-1])
                trend_strength = (trend_up - 5) / 5
                features.append(trend_strength)
                
                # Support/Resistance
                recent_high = self.data['high'].iloc[max(0, current_idx-20):current_idx].max()
                recent_low = self.data['low'].iloc[max(0, current_idx-20):current_idx].min()
                support_distance = (current_price - recent_low) / current_price
                resistance_distance = (recent_high - current_price) / current_price
                
                features.extend([support_distance, resistance_distance])
                
                # Time features
                hour_of_day = (current_idx % 96) / 96
                features.append(hour_of_day)
            
            else:
                # If indicators not available, pad with zeros
                features.extend([0.0] * 20)
            
            # Ensure exactly 30 features
            while len(features) < 30:
                features.append(0.0)
            
            # Clip extreme values and handle NaN
            features = features[:30]
            features = [np.clip(f, -10, 10) if not pd.isna(f) else 0.0 for f in features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error in observation calculation: {e}")
            return np.zeros(30, dtype=np.float32)
    
    def step(self, action):
        """Execute one step"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        reward = 0
        info = {}
        current_price = self.data['close'].iloc[self.current_step]
        
        # Execute action
        if action == 1 and self.current_position == 0:  # Buy
            self.current_position = 1
            self.position_entry_price = current_price
            self.position_entry_step = self.current_step
            
        elif action == 2 and self.current_position == 0:  # Sell
            self.current_position = -1
            self.position_entry_price = current_price
            self.position_entry_step = self.current_step
            
        elif action == 3 and self.current_position != 0:  # Close position
            if self.current_position == 1:  # Close long
                pnl = (current_price - self.position_entry_price) / self.position_entry_price
            else:  # Close short
                pnl = (self.position_entry_price - current_price) / self.position_entry_price
            
            reward = pnl * 100  # Scale reward
            
            if pnl > 0:
                self.win_count += 1
            
            self.total_trades += 1
            self.recent_trades.append(pnl)
            
            # Reset position
            self.current_position = 0
            self.position_entry_price = 0
            self.position_entry_step = 0
        
        # Calculate unrealized PnL for open positions
        if self.current_position != 0:
            if self.current_position == 1:  # Long position
                unrealized_pnl = (current_price - self.position_entry_price) / self.position_entry_price
            else:  # Short position
                unrealized_pnl = (self.position_entry_price - current_price) / self.position_entry_price
            
            # Small reward for unrealized gains, penalty for losses
            reward += unrealized_pnl * 0.1
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Update performance for adaptive optimization
        if self.total_trades > 0:
            current_performance = self.win_count / self.total_trades
            self.episode_performance.append(current_performance)
            
            # Update indicator manager if using adaptive method
            if (self.indicator_manager and 
                self.indicator_manager.optimization_method.value == "adaptive" and
                len(self.episode_performance) % 100 == 0):  # Update every 100 steps
                
                self.indicator_manager.get_optimal_config(self.data, current_performance)
                # Recalculate indicators with new config
                self.calculate_simple_indicators()
        
        return self._get_observation(), reward, done, False, info
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if self.total_trades == 0:
            return {"win_rate": 0.0, "total_trades": 0, "avg_pnl": 0.0}
        
        win_rate = self.win_count / self.total_trades
        avg_pnl = np.mean(self.recent_trades[-100:]) if self.recent_trades else 0.0
        
        return {
            "win_rate": win_rate,
            "total_trades": self.total_trades,
            "avg_pnl": avg_pnl,
            "current_balance": self.current_balance
        }

class SimpleForexBot:
    """
    🚀 Simple Forex Trading Bot with Automatic Indicator Selection
    """
    
    def __init__(self, symbol: str = "EURUSD", optimization_method: str = "smart_defaults"):
        self.symbol = symbol
        self.config = get_config()
        
        # Initialize automatic indicator manager
        self.indicator_manager = create_indicator_manager(
            symbol=symbol,
            optimization_method=optimization_method,
            auto_select=True
        )
        
        # Initialize components
        self.model = None
        self.env = None
        
        logger.info(f"Simple Forex Bot initialized for {symbol} with {optimization_method}")
    
    def create_environment(self):
        """Create training environment"""
        self.env = SimpleForexEnvironment(
            symbol=self.symbol,
            config=self.config,
            indicator_manager=self.indicator_manager
        )
        return self.env
    
    def train_model(self, total_timesteps: int = 10000, algorithm: str = "PPO"):
        """Train RL model"""
        logger.info(f"Starting model training for {self.symbol}")
        logger.info(f"Algorithm: {algorithm}, Steps: {total_timesteps:,}")
        
        # Create environment
        env = self.create_environment()
        
        # Check device availability
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Create model with device specification
        if algorithm == "PPO":
            self.model = PPO("MlpPolicy", env, verbose=1, device=device)
        elif algorithm == "SAC":
            self.model = SAC("MlpPolicy", env, verbose=1, device=device)
        elif algorithm == "A2C":
            self.model = A2C("MlpPolicy", env, verbose=1, device=device)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train model
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save model
        model_path = f"simple_forex_model_{self.symbol}_{algorithm}.zip"
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return True
    
    def test_model(self, episodes: int = 5):
        """Test model performance"""
        if self.model is None:
            logger.error("No model loaded for testing")
            return False
        
        logger.info(f"Testing model for {episodes} episodes...")
        
        env = self.create_environment()
        total_rewards = []
        performance_stats = []
        
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
            stats = env.get_performance_stats()
            performance_stats.append(stats)
            
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
                       f"Win Rate={stats['win_rate']:.1%}, Trades={stats['total_trades']}")
        
        # Calculate overall statistics
        avg_reward = np.mean(total_rewards)
        avg_win_rate = np.mean([s['win_rate'] for s in performance_stats])
        total_trades = sum([s['total_trades'] for s in performance_stats])
        
        logger.info(f"\n📊 Test Results Summary:")
        logger.info(f"Average Reward: {avg_reward:.2f}")
        logger.info(f"Average Win Rate: {avg_win_rate:.1%}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Target Win Rate: {self.config.trading.target_win_rate:.1%}")
        
        return avg_win_rate >= self.config.trading.target_win_rate
    
    def load_model(self, model_path: str, algorithm: str = "PPO"):
        """Load trained model"""
        try:
            if algorithm == "PPO":
                self.model = PPO.load(model_path)
            elif algorithm == "SAC":
                self.model = SAC.load(model_path)
            elif algorithm == "A2C":
                self.model = A2C.load(model_path)
            
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

def main():
    """Main function to run the simple RL system"""
    print("🚀 SIMPLE FOREX RL SYSTEM")
    print("=" * 60)
    print("Reinforcement Learning with Automatic Indicator Selection")
    print("No talib dependency required!")
    print("=" * 60)
    
    # Test different symbols and optimization methods
    test_configs = [
        ("EURUSD", "smart_defaults"),
        ("XAUUSD", "adaptive"),
        ("GBPUSD", "meta_learning")
    ]
    
    for symbol, method in test_configs:
        print(f"\n🎯 Testing {symbol} with {method}")
        print("-" * 40)
        
        try:
            # Create bot
            bot = SimpleForexBot(symbol=symbol, optimization_method=method)
            
            # Train model (small number for quick demo)
            print("🤖 Training model...")
            bot.train_model(total_timesteps=5000, algorithm="PPO")
            
            # Test model
            print("🧪 Testing model...")
            success = bot.test_model(episodes=3)
            
            if success:
                print(f"✅ {symbol} with {method}: TARGET MET!")
            else:
                print(f"⚠️ {symbol} with {method}: Below target, needs more training")
                
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
    
    print("\n🎉 Simple RL System Demo Completed!")
    print("\n💡 Next Steps:")
    print("1. Install talib for full functionality: pip install talib")
    print("2. Run full system: python forex_system_with_config.py")
    print("3. Experiment with different optimization methods")
    print("4. Try different symbols including XAUUSD (Gold)")

if __name__ == "__main__":
    main()