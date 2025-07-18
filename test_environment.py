#!/usr/bin/env python3
"""
🧪 Test Environment เพื่อเช็คว่า AI เทรดได้จริงไหม
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

print("🧪 Testing Environment...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Simple test environment
class SimpleForexEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 50
        self.max_steps = len(data) - 1
        self.balance = 10000
        self.position = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Action: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(50, 5), dtype=np.float32
        )
    
    def _get_observation(self):
        start_idx = self.current_step - 50
        end_idx = self.current_step
        obs = self.data.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values
        return obs.astype(np.float32)
    
    def reset(self, seed=None):
        self.current_step = 50
        self.balance = 10000
        self.position = 0
        self.total_trades = 0
        self.profitable_trades = 0
        return self._get_observation(), {}
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Simple trading logic
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            print(f"📈 BUY at {current_price:.2f}")
            
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = current_price
            print(f"📉 SELL at {current_price:.2f}")
            
        elif action == 3 and self.position != 0:  # Close
            if self.position == 1:
                profit = current_price - self.entry_price
            else:
                profit = self.entry_price - current_price
            
            self.balance += profit
            self.total_trades += 1
            
            if profit > 0:
                self.profitable_trades += 1
                reward = 1
                print(f"✅ CLOSE at {current_price:.2f}, Profit: {profit:.2f}")
            else:
                reward = -1
                print(f"❌ CLOSE at {current_price:.2f}, Loss: {profit:.2f}")
            
            self.position = 0
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if done:
            win_rate = self.profitable_trades / max(self.total_trades, 1)
            total_return = (self.balance - 10000) / 10000
            print(f"\n📊 Final Results:")
            print(f"   Balance: {self.balance:.2f}")
            print(f"   Total Return: {total_return:.2%}")
            print(f"   Total Trades: {self.total_trades}")
            print(f"   Win Rate: {win_rate:.2%}")
        
        return self._get_observation(), reward, done, False, {}

# Load data
print("\n📊 Loading data...")
try:
    df = pd.read_csv('train_data/XAUUSD/XAUUSD_M5_real.csv')
    print(f"   Loaded {len(df)} rows")
    
    # Add volume column if missing
    if 'volume' not in df.columns:
        df['volume'] = 1000
    
    # Test environment
    print("\n🧪 Testing Environment...")
    env = SimpleForexEnv(df.head(1000))  # Use first 1000 rows
    
    # Test reset
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    
    # Test a few steps
    print("\n🎮 Testing Actions...")
    for i in range(10):
        action = np.random.randint(0, 4)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    # Test with PPO
    print("\n🤖 Testing PPO...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env_wrapper = DummyVecEnv([lambda: env])
    
    model = PPO(
        "MlpPolicy", 
        env_wrapper,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        device=device,
        verbose=1
    )
    
    print(f"   Model created on device: {model.device}")
    
    # Quick training test
    print("\n🏋️ Quick training test...")
    model.learn(total_timesteps=1000)
    
    print("\n✅ Environment test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()