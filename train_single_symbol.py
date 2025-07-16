#!/usr/bin/env python3
"""
Train RL model for a single symbol
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleForexEnv(gym.Env):
    """
    Simple Forex Environment for RL Training
    """
    
    def __init__(self, data, symbol='XAUUSD', initial_balance=10000, lookback_window=50):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Remove NaN values
        self.data = self.data.dropna().reset_index(drop=True)
        
        # Environment state
        self.current_step = self.lookback_window
        self.max_steps = len(self.data) - 1
        
        # Trading state
        self.balance = initial_balance
        self.position = 0  # -1: Short, 0: Neutral, 1: Long
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLC + indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, 8), 
            dtype=np.float32
        )
    
    def _calculate_indicators(self):
        """Calculate technical indicators"""
        # Simple Moving Averages
        self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
        self.data['sma_50'] = self.data['close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = self.data['close'].ewm(span=12).mean()
        ema_26 = self.data['close'].ewm(span=26).mean()
        self.data['macd'] = ema_12 - ema_26
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
    
    def _get_observation(self):
        """Get current observation"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        obs_data = self.data.iloc[start_idx:end_idx][
            ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi', 'macd']
        ].values
        
        # Normalize data
        obs_data = (obs_data - np.mean(obs_data, axis=0)) / (np.std(obs_data, axis=0) + 1e-8)
        
        return obs_data.astype(np.float32)
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step"""
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            
        elif action == 2 and self.position == 1:  # Sell
            profit = current_price - self.entry_price
            self.balance += profit * 100  # Assume 100 units
            
            if profit > 0:
                reward = 1
                self.profitable_trades += 1
            else:
                reward = -1
            
            self.position = 0
            self.total_trades += 1
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Calculate reward based on balance change
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros((self.lookback_window, 8), dtype=np.float32)
            
            # Final reward based on total performance
            total_return = (self.balance - self.initial_balance) / self.initial_balance
            reward += total_return * 10  # Scale reward
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades
        }
        
        return obs, reward, done, False, info

def train_model(symbol='XAUUSD', timesteps=50000):
    """
    Train RL model for a symbol
    """
    print(f"🔄 Training {symbol} model...")
    print("="*50)
    
    # Load data
    data_file = f"train_data/{symbol}/{symbol}_M5_real.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return None
    
    print(f"📊 Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df):,} rows")
    
    # Create environment
    print(f"🏗️ Creating trading environment...")
    env = SimpleForexEnv(df, symbol=symbol)
    env = DummyVecEnv([lambda: env])
    
    # Create model
    print(f"🤖 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Train model
    print(f"🚀 Starting training for {timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(total_timesteps=timesteps)
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"⏱️ Training completed in {training_time}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{symbol.lower()}_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Test model
    print(f"\n🧪 Testing trained model...")
    test_env = SimpleForexEnv(df.tail(5000), symbol=symbol)  # Test on last 5000 rows
    
    obs, _ = test_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        total_reward += reward
    
    # Results
    final_balance = info['balance']
    total_return = (final_balance - test_env.initial_balance) / test_env.initial_balance
    win_rate = info['profitable_trades'] / max(info['total_trades'], 1)
    
    print(f"\n📈 Test Results:")
    print(f"   Initial balance: ${test_env.initial_balance:,.2f}")
    print(f"   Final balance: ${final_balance:,.2f}")
    print(f"   Total return: {total_return:.2%}")
    print(f"   Total trades: {info['total_trades']}")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Total reward: {total_reward:.2f}")
    
    # Save results
    results_file = f"{symbol.lower()}_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"{symbol} RL Training Results\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Training timesteps: {timesteps:,}\n")
        f.write(f"Training time: {training_time}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Final balance: ${final_balance:,.2f}\n")
        f.write(f"Total return: {total_return:.2%}\n")
        f.write(f"Total trades: {info['total_trades']}\n")
        f.write(f"Win rate: {win_rate:.1%}\n")
    
    print(f"📄 Results saved to: {results_file}")
    
    return model_path

def main():
    """Main function"""
    symbol = 'XAUUSD'
    timesteps = 100000
    
    print("🚀 RL Model Training")
    print("="*50)
    print(f"Symbol: {symbol}")
    print(f"Timesteps: {timesteps:,}")
    print()
    
    model_path = train_model(symbol, timesteps)
    
    if model_path:
        print(f"\n✅ Training completed successfully!")
        print(f"Model saved at: {model_path}")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()