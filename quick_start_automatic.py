#!/usr/bin/env python3
"""
🚀 Quick Start Automatic Training System
Fast training with automatic parameter optimization
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import json
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

class QuickForexEnv(gym.Env):
    """
    Quick Forex Environment for fast training
    """
    
    def __init__(self, data, symbol='XAUUSD', initial_balance=10000, lookback_window=30):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Calculate basic indicators
        self._calculate_indicators()
        
        # Remove NaN values
        self.data = self.data.dropna().reset_index(drop=True)
        
        # Environment state
        self.current_step = self.lookback_window
        self.max_steps = min(len(self.data) - 1, self.lookback_window + 5000)  # Limit for speed
        
        # Trading state
        self.balance = initial_balance
        self.position = 0  # -1: Short, 0: Neutral, 1: Long
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_balance = initial_balance
        self.max_drawdown = 0
        
        # Trade history
        self.trades = []
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLC + indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, 8), 
            dtype=np.float32
        )
    
    def _calculate_indicators(self):
        """Calculate basic technical indicators"""
        # Simple Moving Averages
        self.data['sma_10'] = self.data['close'].rolling(window=10).mean()
        self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
        
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
            ['open', 'high', 'low', 'close', 'sma_10', 'sma_20', 'rsi', 'macd']
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
        self.total_profit = 0
        self.total_loss = 0
        self.max_balance = self.initial_balance
        self.max_drawdown = 0
        self.trades = []
        
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
            profit = (current_price - self.entry_price) * 100  # Assume 100 units
            self.balance += profit
            
            # Track trade
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'profit': profit
            })
            
            if profit > 0:
                self.profitable_trades += 1
                self.total_profit += profit
                reward = 2
            else:
                self.total_loss += abs(profit)
                reward = -1
            
            self.total_trades += 1
            self.position = 0
        
        # Update max balance and drawdown
        self.max_balance = max(self.max_balance, self.balance)
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros((self.lookback_window, 8), dtype=np.float32)
            
            # Final reward based on total performance
            total_return = (self.balance - self.initial_balance) / self.initial_balance
            reward += total_return * 10
        
        info = self._get_performance_metrics()
        
        return obs, reward, done, False, info
    
    def _get_performance_metrics(self):
        """Calculate performance metrics"""
        if self.total_trades == 0:
            return {
                'balance': self.balance,
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': self.max_drawdown,
                'total_trades': 0
            }
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        win_rate = self.profitable_trades / self.total_trades
        profit_factor = self.total_profit / max(self.total_loss, 1e-8)
        
        return {
            'balance': self.balance,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades
        }

def quick_train(symbol='XAUUSD', max_attempts=10):
    """
    Quick training with automatic optimization
    """
    print(f"🚀 QUICK AUTOMATIC TRAINING FOR {symbol}")
    print("="*50)
    
    # Load data
    data_file = f"train_data/{symbol}/{symbol}_M5_real.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df):,} rows")
    
    # Training configurations to try
    configs = [
        {'learning_rate': 0.0003, 'n_steps': 1024, 'batch_size': 64, 'timesteps': 20000},
        {'learning_rate': 0.001, 'n_steps': 2048, 'batch_size': 32, 'timesteps': 30000},
        {'learning_rate': 0.0001, 'n_steps': 1024, 'batch_size': 128, 'timesteps': 25000},
        {'learning_rate': 0.003, 'n_steps': 512, 'batch_size': 64, 'timesteps': 15000},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'timesteps': 40000},
    ]
    
    best_score = 0
    best_model = None
    best_metrics = None
    results = []
    
    for i, config in enumerate(configs[:max_attempts]):
        print(f"\n🔄 Attempt {i+1}/{min(max_attempts, len(configs))}")
        print(f"   Config: {config}")
        
        try:
            # Create environment
            env = QuickForexEnv(df, symbol=symbol)
            env = DummyVecEnv([lambda: env])
            
            # Create model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                verbose=0
            )
            
            # Train model
            start_time = time.time()
            model.learn(total_timesteps=config['timesteps'])
            training_time = time.time() - start_time
            
            # Test model
            test_env = QuickForexEnv(df.tail(2000), symbol=symbol)
            
            obs, _ = test_env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = test_env.step(action)
            
            metrics = info
            
            # Calculate score
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            max_drawdown = metrics.get('max_drawdown', 1)
            
            score = (
                win_rate * 40 +
                min(profit_factor / 3.0, 1.0) * 30 +
                max(0, 1 - max_drawdown * 2) * 20 +
                (metrics.get('total_return', 0) > 0) * 10
            )
            
            # Determine tier
            if win_rate >= 0.80 and profit_factor >= 2.8 and max_drawdown <= 0.10:
                tier, emoji = 'gold', '🥇'
            elif win_rate >= 0.75 and profit_factor >= 2.5 and max_drawdown <= 0.12:
                tier, emoji = 'silver', '🥈'
            elif win_rate >= 0.70 and profit_factor >= 2.0 and max_drawdown <= 0.15:
                tier, emoji = 'bronze', '🥉'
            else:
                tier, emoji = 'none', '❌'
            
            print(f"   {emoji} Tier: {tier.upper()}")
            print(f"   📊 Score: {score:.1f}")
            print(f"   📈 Win Rate: {win_rate:.1%}")
            print(f"   💰 Profit Factor: {profit_factor:.2f}")
            print(f"   📉 Max Drawdown: {max_drawdown:.1%}")
            print(f"   💵 Total Return: {metrics.get('total_return', 0):.1%}")
            print(f"   🔢 Total Trades: {metrics.get('total_trades', 0)}")
            print(f"   ⏱️ Training Time: {training_time:.1f}s")
            
            # Save result
            result = {
                'attempt': i + 1,
                'config': config,
                'metrics': metrics,
                'score': score,
                'tier': tier,
                'training_time': training_time
            }
            results.append(result)
            
            # Check if this is the best
            if score > best_score:
                best_score = score
                best_model = model
                best_metrics = metrics
                
                # Save best model
                os.makedirs('models', exist_ok=True)
                model_path = f"models/{symbol.lower()}_quick_best_{tier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                model.save(model_path)
                print(f"   💾 New best model saved: {model_path}")
            
            # Stop if gold tier reached
            if tier == 'gold':
                print(f"\n🎉 GOLD TIER REACHED! Stopping training.")
                break
                
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
    
    # Final summary
    print("\n" + "="*50)
    print("🏁 QUICK TRAINING SUMMARY")
    print("="*50)
    
    if best_metrics:
        print(f"🏆 Best Performance:")
        print(f"   Score: {best_score:.1f}")
        print(f"   Win Rate: {best_metrics['win_rate']:.1%}")
        print(f"   Profit Factor: {best_metrics['profit_factor']:.2f}")
        print(f"   Max Drawdown: {best_metrics['max_drawdown']:.1%}")
        print(f"   Total Return: {best_metrics['total_return']:.1%}")
        print(f"   Total Trades: {best_metrics['total_trades']}")
    else:
        print("❌ No successful training attempts")
    
    # Save results
    results_file = f"{symbol.lower()}_quick_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {results_file}")
    
    return best_model, best_metrics

def main():
    """Main function"""
    symbol = 'XAUUSD'
    
    print("🎯 Quick Automatic Training System")
    print("Optimized for speed and efficiency")
    print()
    
    best_model, best_metrics = quick_train(symbol, max_attempts=5)
    
    if best_model and best_metrics:
        print(f"\n✅ Training completed successfully!")
        
        # Check if target reached
        win_rate = best_metrics['win_rate']
        profit_factor = best_metrics['profit_factor']
        max_drawdown = best_metrics['max_drawdown']
        
        if win_rate >= 0.80 and profit_factor >= 2.8 and max_drawdown <= 0.10:
            print("🥇 GOLD TARGET ACHIEVED!")
        elif win_rate >= 0.75 and profit_factor >= 2.5 and max_drawdown <= 0.12:
            print("🥈 SILVER TIER ACHIEVED!")
        elif win_rate >= 0.70 and profit_factor >= 2.0 and max_drawdown <= 0.15:
            print("🥉 BRONZE TIER ACHIEVED!")
        else:
            print("📈 Good progress made, continue training for better results")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()