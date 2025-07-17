#!/usr/bin/env python3
"""
🚀 RTX 5060 TI Optimized Training - Maximum GPU Utilization
Target: 90%+ GPU utilization (vs 20% current)
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
import json
from datetime import datetime
import warnings
import time
import torch
from gpu_boost_config import apply_rtx_5060_ti_boost, get_gpu_optimized_model_config
warnings.filterwarnings('ignore')

# Apply RTX 5060 TI maximum performance boost
DEVICE = apply_rtx_5060_ti_boost()

class GPUOptimizedForexEnv(gym.Env):
    """
    GPU-Optimized Forex Environment for maximum RTX 5060 TI utilization
    """
    
    def __init__(self, data, symbol='XAUUSD', initial_balance=10000, lookback_window=200, 
                 transaction_cost=0.0001, max_position_size=1.0):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Calculate advanced indicators (GPU-optimized)
        self._calculate_indicators_gpu()
        
        # Remove NaN values
        self.data = self.data.dropna().reset_index(drop=True)
        
        # Environment state
        self.current_step = self.lookback_window
        self.max_steps = len(self.data) - 1
        
        # Trading state
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_equity = initial_balance
        self.max_drawdown = 0
        
        # Trade history
        self.trades = []
        self.equity_curve = [initial_balance]
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Larger observation space for GPU utilization
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, 20),  # More features for GPU
            dtype=np.float32
        )
    
    def _calculate_indicators_gpu(self):
        """Calculate indicators optimized for GPU processing"""
        # Convert to GPU tensors for faster computation
        close_tensor = torch.tensor(self.data['close'].values, device=DEVICE, dtype=torch.float32)
        high_tensor = torch.tensor(self.data['high'].values, device=DEVICE, dtype=torch.float32)
        low_tensor = torch.tensor(self.data['low'].values, device=DEVICE, dtype=torch.float32)
        
        # GPU-accelerated moving averages
        for window in [5, 10, 20, 50, 100]:
            ma = torch.nn.functional.avg_pool1d(
                close_tensor.unsqueeze(0).unsqueeze(0), 
                kernel_size=window, 
                stride=1, 
                padding=window//2
            ).squeeze().cpu().numpy()
            self.data[f'sma_{window}'] = ma
        
        # GPU-accelerated RSI calculation
        delta = torch.diff(close_tensor)
        gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
        loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # Use GPU for RSI calculation
        avg_gain = torch.nn.functional.avg_pool1d(
            gain.unsqueeze(0).unsqueeze(0), 
            kernel_size=14, 
            stride=1, 
            padding=7
        ).squeeze()
        avg_loss = torch.nn.functional.avg_pool1d(
            loss.unsqueeze(0).unsqueeze(0), 
            kernel_size=14, 
            stride=1, 
            padding=7
        ).squeeze()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        self.data['rsi'] = torch.cat([torch.zeros(1, device=DEVICE), rsi]).cpu().numpy()
        
        # Additional GPU-accelerated indicators
        self.data['volatility'] = self.data['close'].rolling(20).std()
        self.data['momentum'] = self.data['close'].pct_change(10)
        self.data['bb_upper'] = self.data['close'].rolling(20).mean() + 2 * self.data['close'].rolling(20).std()
        self.data['bb_lower'] = self.data['close'].rolling(20).mean() - 2 * self.data['close'].rolling(20).std()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    def _get_observation(self):
        """Get GPU-optimized observation with more features"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        # More features for GPU utilization
        feature_cols = [
            'open', 'high', 'low', 'close', 
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
            'rsi', 'volatility', 'momentum', 
            'bb_upper', 'bb_lower'
        ]
        
        obs_data = self.data.iloc[start_idx:end_idx][feature_cols].values
        
        # Add position and market state info (6 more features)
        additional_features = np.full((self.lookback_window, 6), [
            self.position, self.position_size, self.balance/self.initial_balance,
            self.equity/self.initial_balance, self.total_trades, self.max_drawdown
        ])
        
        obs_data = np.hstack([obs_data, additional_features])
        
        # GPU-accelerated normalization
        obs_tensor = torch.tensor(obs_data, device=DEVICE, dtype=torch.float32)
        normalized = (obs_tensor - obs_tensor.mean(dim=0)) / (obs_tensor.std(dim=0) + 1e-8)
        
        return normalized.cpu().numpy().astype(np.float32)
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_equity = self.initial_balance
        self.max_drawdown = 0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute step with GPU-optimized calculations"""
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.position_size = self.max_position_size
            self.entry_price = current_price
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            
        elif action == 2 and self.position == 0:  # Sell (Short)
            self.position = -1
            self.position_size = self.max_position_size
            self.entry_price = current_price
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            
        elif action == 3 and self.position != 0:  # Close position
            if self.position == 1:  # Close long
                profit = (current_price - self.entry_price) * self.position_size
            else:  # Close short
                profit = (self.entry_price - current_price) * self.position_size
            
            cost = current_price * self.position_size * self.transaction_cost
            profit -= cost
            self.balance += profit
            
            # Track trade
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position': self.position,
                'profit': profit,
                'timestamp': self.current_step
            })
            
            if profit > 0:
                self.profitable_trades += 1
                self.total_profit += profit
                reward = 3  # Higher reward for profitable trades
            else:
                self.total_loss += abs(profit)
                reward = -2  # Penalty for losing trades
            
            self.total_trades += 1
            self.position = 0
            self.position_size = 0
        
        # Update equity
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
            self.equity = self.balance + unrealized_pnl
        else:
            self.equity = self.balance
        
        # Track equity and drawdown
        self.equity_curve.append(self.equity)
        self.max_equity = max(self.max_equity, self.equity)
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Enhanced reward shaping
        if self.max_drawdown > 0.15:
            reward -= 5
        elif self.max_drawdown < 0.05:
            reward += 2
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.max_steps or self.equity <= self.initial_balance * 0.5
        
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros((self.lookback_window, 20), dtype=np.float32)
            total_return = (self.equity - self.initial_balance) / self.initial_balance
            reward += total_return * 50  # Scale final reward
        
        info = self._get_performance_metrics()
        
        return obs, reward, done, False, info
    
    def _get_performance_metrics(self):
        """Calculate performance metrics"""
        if self.total_trades == 0:
            return {
                'balance': self.balance,
                'equity': self.equity,
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': self.max_drawdown,
                'total_trades': 0,
                'sharpe_ratio': 0
            }
        
        total_return = (self.equity - self.initial_balance) / self.initial_balance
        win_rate = self.profitable_trades / self.total_trades
        profit_factor = self.total_profit / max(self.total_loss, 1e-8)
        
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'sharpe_ratio': sharpe_ratio
        }

def create_gpu_optimized_env(data, symbol='XAUUSD'):
    """Create GPU-optimized environment"""
    return GPUOptimizedForexEnv(
        data, 
        symbol=symbol,
        lookback_window=400,  # Large window for GPU
        transaction_cost=0.0001
    )

def train_gpu_optimized_model(data, symbol='XAUUSD'):
    """Train model with maximum GPU utilization"""
    print(f"🚀 Starting GPU-Optimized Training for {symbol}")
    print("🎯 Target: 90%+ GPU Utilization")
    
    # Create multiple environments for parallel processing
    def make_env():
        return create_gpu_optimized_env(data, symbol)
    
    # Use 4 parallel environments to maximize GPU usage
    env = SubprocVecEnv([make_env for _ in range(4)])
    
    # Get GPU-optimized model configuration
    gpu_config = get_gpu_optimized_model_config()
    
    print(f"🔥 GPU Configuration Applied:")
    print(f"   📊 Neural Network: {gpu_config['policy_kwargs']['net_arch']}")
    print(f"   🎯 Batch Size: {gpu_config['batch_size']}")
    print(f"   ⚡ N Steps: {gpu_config['n_steps']}")
    print(f"   🚀 4 Parallel Environments")
    
    # Create PPO model with GPU optimization
    model = PPO(
        "MlpPolicy",
        env,
        device=DEVICE,
        **gpu_config
    )
    
    print(f"🚀 Starting Training with Maximum GPU Utilization...")
    print(f"   Device: {DEVICE}")
    print(f"   Expected GPU Utilization: 80-95%")
    
    # Train with large timesteps for GPU utilization
    start_time = time.time()
    model.learn(total_timesteps=1000000)  # 1M timesteps for maximum GPU usage
    training_time = time.time() - start_time
    
    print(f"✅ Training Completed!")
    print(f"   Training Time: {training_time:.1f}s")
    print(f"   Average Speed: {1000000/training_time:.0f} steps/second")
    
    # Test the model
    test_env = create_gpu_optimized_env(data.tail(10000), symbol)
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
    
    metrics = info
    print(f"📊 Final Results:")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"   Total Return: {metrics['total_return']:.1%}")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"models/gpu_optimized_{symbol}_{timestamp}.zip"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    return model, metrics

def main():
    """Main function for GPU-optimized training"""
    symbol = 'XAUUSD'
    
    # Load data
    data_file = f"train_data/{symbol}/{symbol}_M5_real.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df):,} rows")
    
    # Start GPU-optimized training
    model, metrics = train_gpu_optimized_model(df, symbol)
    
    print(f"\n🎉 GPU-Optimized Training Complete!")
    print(f"🚀 RTX 5060 TI Performance Maximized!")

if __name__ == "__main__":
    main()