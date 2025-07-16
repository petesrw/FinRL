#!/usr/bin/env python3
"""
🎯 Adaptive RL Training System for Forex
Trains until reaching excellence targets with full tracking
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import json
from datetime import datetime
import warnings
import time
import random
warnings.filterwarnings('ignore')

class AdvancedForexEnv(gym.Env):
    """
    Advanced Forex Environment with comprehensive metrics
    """
    
    def __init__(self, data, symbol='XAUUSD', initial_balance=10000, lookback_window=50, 
                 transaction_cost=0.0001, max_position_size=1.0):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Calculate advanced indicators
        self._calculate_indicators()
        
        # Remove NaN values
        self.data = self.data.dropna().reset_index(drop=True)
        
        # Environment state
        self.current_step = self.lookback_window
        self.max_steps = len(self.data) - 1
        
        # Trading state
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0  # -1: Short, 0: Neutral, 1: Long
        self.position_size = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_equity = initial_balance
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Trade history
        self.trades = []
        self.equity_curve = [initial_balance]
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation space: OHLC + indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, 13), 
            dtype=np.float32
        )
    
    def _calculate_indicators(self):
        """Calculate comprehensive technical indicators"""
        # Moving Averages
        self.data['sma_10'] = self.data['close'].rolling(window=10).mean()
        self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
        self.data['sma_50'] = self.data['close'].rolling(window=50).mean()
        self.data['ema_12'] = self.data['close'].ewm(span=12).mean()
        self.data['ema_26'] = self.data['close'].ewm(span=26).mean()
        
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.data['macd'] = self.data['ema_12'] - self.data['ema_26']
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
        
        # Bollinger Bands
        self.data['bb_middle'] = self.data['close'].rolling(window=20).mean()
        bb_std = self.data['close'].rolling(window=20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + (bb_std * 2)
        self.data['bb_lower'] = self.data['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.data['atr'] = true_range.rolling(14).mean()
    
    def _get_observation(self):
        """Get current observation with position info"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        obs_data = self.data.iloc[start_idx:end_idx][
            ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi', 'macd', 
             'macd_signal', 'bb_upper', 'bb_lower', 'atr']
        ].values
        
        # Add position information to each timestep
        position_info = np.full((self.lookback_window, 1), self.position)
        obs_data = np.hstack([obs_data, position_info])
        
        # Normalize data (except position)
        obs_data[:, :-1] = (obs_data[:, :-1] - np.mean(obs_data[:, :-1], axis=0)) / (np.std(obs_data[:, :-1], axis=0) + 1e-8)
        
        return obs_data.astype(np.float32)
    
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
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step with advanced reward calculation"""
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.position_size = self.max_position_size
            self.entry_price = current_price
            # Transaction cost
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            
        elif action == 2 and self.position == 0:  # Sell (Short)
            self.position = -1
            self.position_size = self.max_position_size
            self.entry_price = current_price
            # Transaction cost
            cost = current_price * self.position_size * self.transaction_cost
            self.balance -= cost
            
        elif action == 3 and self.position != 0:  # Close position
            if self.position == 1:  # Close long
                profit = (current_price - self.entry_price) * self.position_size
            else:  # Close short
                profit = (self.entry_price - current_price) * self.position_size
            
            # Transaction cost
            cost = current_price * self.position_size * self.transaction_cost
            profit -= cost
            
            self.balance += profit
            
            # Track trade
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position': self.position,
                'profit': profit,
                'timestamp': self.data.iloc[self.current_step]['timestamp']
            })
            
            if profit > 0:
                self.profitable_trades += 1
                self.total_profit += profit
                self.consecutive_losses = 0
                reward = 2  # Reward for profitable trade
            else:
                self.total_loss += abs(profit)
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                reward = -1  # Penalty for losing trade
            
            self.total_trades += 1
            self.position = 0
            self.position_size = 0
        
        # Update equity
        if self.position != 0:
            if self.position == 1:  # Long position
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:  # Short position
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
            self.equity = self.balance + unrealized_pnl
        else:
            self.equity = self.balance
        
        # Track equity curve and drawdown
        self.equity_curve.append(self.equity)
        self.max_equity = max(self.max_equity, self.equity)
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Enhanced reward shaping for better performance
        if self.max_drawdown > 0.15:  # Stricter penalty for high drawdown
            reward -= 3
        elif self.max_drawdown < 0.05:  # Reward for low drawdown
            reward += 1
        
        if self.consecutive_losses >= 3:  # Earlier penalty for consecutive losses
            reward -= 2
        
        # Reward for maintaining good win rate
        if self.total_trades > 10:
            current_win_rate = self.profitable_trades / self.total_trades
            if current_win_rate > 0.6:
                reward += 2
            elif current_win_rate > 0.7:
                reward += 3
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.equity <= self.initial_balance * 0.5
        
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros((self.lookback_window, 13), dtype=np.float32)
            
            # Final reward based on total performance
            total_return = (self.equity - self.initial_balance) / self.initial_balance
            reward += total_return * 20  # Scale final reward
        
        info = self._get_performance_metrics()
        
        return obs, reward, done, False, info
    
    def _get_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
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
        
        # Calculate Sharpe ratio
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
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_losses': self.max_consecutive_losses
        }

class AdaptiveTrainer:
    """
    Adaptive trainer that learns from previous attempts
    """
    
    def __init__(self, symbol='XAUUSD'):
        self.symbol = symbol
        self.training_history = []
        self.best_model = None
        self.best_score = 0
        
        # Create organized folder structure
        self.base_dir = "training_logs"
        self.history_dir = f"{self.base_dir}/history"
        self.config_dir = f"{self.base_dir}/configs"
        self.failed_config_dir = f"{self.base_dir}/failed_configs"
        self.successful_config_dir = f"{self.base_dir}/successful_configs"
        
        # Create directories if they don't exist
        for directory in [self.base_dir, self.history_dir, self.config_dir, 
                         self.failed_config_dir, self.successful_config_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # File paths
        self.history_file = f"{self.history_dir}/{symbol.lower()}_training_history.json"
        self.failed_configs_file = f"{self.failed_config_dir}/{symbol.lower()}_failed_configs.json"
        self.successful_configs_file = f"{self.successful_config_dir}/{symbol.lower()}_successful_configs.json"
        
        self.load_history()
        
        # Excellence targets
        self.targets = {
            'bronze': {'win_rate': 0.70, 'profit_factor': 2.0, 'max_drawdown': 0.15, 'score': 70},
            'silver': {'win_rate': 0.75, 'profit_factor': 2.5, 'max_drawdown': 0.12, 'score': 75},
            'gold': {'win_rate': 0.80, 'profit_factor': 2.8, 'max_drawdown': 0.10, 'score': 80},
            'diamond': {'win_rate': 0.85, 'profit_factor': 3.2, 'max_drawdown': 0.08, 'score': 85}
        }
    
    def load_history(self):
        """Load training history"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.training_history = json.load(f)
            print(f"📚 Loaded {len(self.training_history)} previous training attempts")
        else:
            print("🆕 Starting fresh training history")
    
    def save_history(self):
        """Save training history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def save_failed_config(self, config, error_msg, attempt_num):
        """Save failed configuration separately"""
        failed_configs = []
        if os.path.exists(self.failed_configs_file):
            with open(self.failed_configs_file, 'r') as f:
                failed_configs = json.load(f)
        
        failed_config = {
            'attempt': attempt_num,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'error': error_msg,
            'symbol': self.symbol
        }
        
        failed_configs.append(failed_config)
        
        with open(self.failed_configs_file, 'w') as f:
            json.dump(failed_configs, f, indent=2)
        
        print(f"   📝 Failed config saved to: {self.failed_configs_file}")
    
    def save_successful_config(self, config, metrics, score, tier, attempt_num):
        """Save successful configuration separately"""
        successful_configs = []
        if os.path.exists(self.successful_configs_file):
            with open(self.successful_configs_file, 'r') as f:
                successful_configs = json.load(f)
        
        successful_config = {
            'attempt': attempt_num,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'score': score,
            'tier': tier,
            'symbol': self.symbol
        }
        
        successful_configs.append(successful_config)
        
        # Sort by score (best first)
        successful_configs.sort(key=lambda x: x['score'], reverse=True)
        
        with open(self.successful_configs_file, 'w') as f:
            json.dump(successful_configs, f, indent=2)
        
        print(f"   📝 Successful config saved to: {self.successful_configs_file}")
    
    def load_failed_configs(self):
        """Load previously failed configurations"""
        if os.path.exists(self.failed_configs_file):
            with open(self.failed_configs_file, 'r') as f:
                failed_configs = json.load(f)
            return [fc['config'] for fc in failed_configs]
        return []
    
    def _save_model_by_tier(self, model, tier, score, attempt, is_best=True):
        """Save model in organized folder structure by tier"""
        # Create tier-specific directories
        tier_dirs = {
            'bronze': 'models/bronze',
            'silver': 'models/silver', 
            'gold': 'models/gold',
            'diamond': 'models/diamond',
            'none': 'models/successful'
        }
        
        # Create directory if it doesn't exist
        model_dir = tier_dirs.get(tier, 'models/successful')
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if is_best:
            filename = f"{self.symbol.lower()}_best_{tier}_score{score:.0f}_attempt{attempt}_{timestamp}.zip"
        else:
            filename = f"{self.symbol.lower()}_{tier}_score{score:.0f}_attempt{attempt}_{timestamp}.zip"
        
        model_path = f"{model_dir}/{filename}"
        
        # Save model
        model.save(model_path)
        
        # Create model info file
        info_file = model_path.replace('.zip', '_info.json')
        model_info = {
            'symbol': self.symbol,
            'tier': tier,
            'score': score,
            'attempt': attempt,
            'timestamp': timestamp,
            'is_best': is_best,
            'model_path': model_path
        }
        
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        if is_best:
            print(f"   💾 New best {tier.upper()} model saved: {model_path}")
        else:
            print(f"   💾 {tier.upper()} tier model saved: {model_path}")
        
        return model_path
    
    def calculate_score(self, metrics):
        """Calculate overall performance score"""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        # Weighted scoring
        score = (
            win_rate * 40 +  # 40% weight on win rate
            min(profit_factor / 3.0, 1.0) * 30 +  # 30% weight on profit factor (capped at 3.0)
            max(0, 1 - max_drawdown * 2) * 20 +  # 20% weight on drawdown control
            min(sharpe_ratio / 2.0, 1.0) * 10  # 10% weight on Sharpe ratio
        ) * 100
        
        return score
    
    def get_tier(self, metrics):
        """Determine performance tier"""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        
        if (win_rate >= self.targets['diamond']['win_rate'] and 
            profit_factor >= self.targets['diamond']['profit_factor'] and 
            max_drawdown <= self.targets['diamond']['max_drawdown']):
            return 'diamond', '💎'
        elif (win_rate >= self.targets['gold']['win_rate'] and 
              profit_factor >= self.targets['gold']['profit_factor'] and 
              max_drawdown <= self.targets['gold']['max_drawdown']):
            return 'gold', '🥇'
        elif (win_rate >= self.targets['silver']['win_rate'] and 
              profit_factor >= self.targets['silver']['profit_factor'] and 
              max_drawdown <= self.targets['silver']['max_drawdown']):
            return 'silver', '🥈'
        elif (win_rate >= self.targets['bronze']['win_rate'] and 
              profit_factor >= self.targets['bronze']['profit_factor'] and 
              max_drawdown <= self.targets['bronze']['max_drawdown']):
            return 'bronze', '🥉'
        else:
            return 'none', '❌'
    
    def generate_hyperparameters(self):
        """Generate hyperparameters based on history"""
        # Avoid previously failed configurations
        failed_configs = [h['hyperparameters'] for h in self.training_history if h.get('score', 0) < 50]
        
        max_attempts = 100
        for _ in range(max_attempts):
            algorithm = random.choice(['PPO', 'A2C'])  # Remove SAC for now due to action space issues
            config = {
                'algorithm': algorithm,
                'learning_rate': random.choice([0.0001, 0.0003, 0.001, 0.003]),
                'n_steps': random.choice([1024, 2048, 4096]) if algorithm == 'PPO' else None,
                'batch_size': random.choice([32, 64, 128, 256]),
                'gamma': random.choice([0.95, 0.99, 0.995]),
                'lookback_window': random.choice([30, 50, 100, 150]),
                'transaction_cost': random.choice([0.0001, 0.0002, 0.0005]),
                'timesteps': random.choice([50000, 100000, 200000])
            }
            
            # Check if this config was already tried and failed
            config_str = str(sorted(config.items()))
            failed_config_strs = [str(sorted(fc.items())) for fc in failed_configs]
            if config_str not in failed_config_strs:
                return config
        
        # If all configs were tried, generate a random one
        return {
            'algorithm': 'PPO',
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'lookback_window': 50,
            'transaction_cost': 0.0001,
            'timesteps': 100000
        }
    
    def train_model(self, data, hyperparameters):
        """Train a single model with given hyperparameters"""
        print(f"\n🚀 Training with hyperparameters:")
        for key, value in hyperparameters.items():
            if value is not None:
                print(f"   {key}: {value}")
        
        # Create environment
        env = AdvancedForexEnv(
            data, 
            symbol=self.symbol,
            lookback_window=hyperparameters['lookback_window'],
            transaction_cost=hyperparameters['transaction_cost']
        )
        env = DummyVecEnv([lambda: env])
        
        # Create model based on algorithm
        if hyperparameters['algorithm'] == 'PPO':
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=hyperparameters['learning_rate'],
                n_steps=hyperparameters['n_steps'],
                batch_size=hyperparameters['batch_size'],
                gamma=hyperparameters['gamma'],
                verbose=0
            )
        elif hyperparameters['algorithm'] == 'SAC':
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=hyperparameters['learning_rate'],
                batch_size=hyperparameters['batch_size'],
                gamma=hyperparameters['gamma'],
                verbose=0
            )
        else:  # A2C
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=hyperparameters['learning_rate'],
                gamma=hyperparameters['gamma'],
                verbose=0
            )
        
        # Train model
        start_time = time.time()
        model.learn(total_timesteps=hyperparameters['timesteps'])
        training_time = time.time() - start_time
        
        # Test model
        test_env = AdvancedForexEnv(
            data.tail(10000), 
            symbol=self.symbol,
            lookback_window=hyperparameters['lookback_window'],
            transaction_cost=hyperparameters['transaction_cost']
        )
        
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
        
        metrics = info
        score = self.calculate_score(metrics)
        tier, emoji = self.get_tier(metrics)
        
        return model, metrics, score, tier, emoji, training_time
    
    def adaptive_train(self, data, max_attempts=50, target_tier='gold'):
        """Adaptive training until target is reached"""
        print(f"🎯 ADAPTIVE TRAINING FOR {self.symbol}")
        print("="*60)
        print(f"Target: {target_tier.upper()} tier")
        print(f"Max attempts: {max_attempts}")
        print()
        
        attempt = 1
        best_attempt = None
        
        while attempt <= max_attempts:
            print(f"🔄 Attempt {attempt}/{max_attempts}")
            
            # Generate hyperparameters
            hyperparams = self.generate_hyperparameters()
            
            try:
                # Train model
                model, metrics, score, tier, emoji, training_time = self.train_model(data, hyperparams)
                
                # Record attempt
                attempt_record = {
                    'attempt': attempt,
                    'timestamp': datetime.now().isoformat(),
                    'hyperparameters': hyperparams,
                    'metrics': metrics,
                    'score': score,
                    'tier': tier,
                    'training_time': training_time
                }
                
                self.training_history.append(attempt_record)
                
                # Print results
                print(f"   {emoji} Tier: {tier.upper()}")
                print(f"   📊 Score: {score:.1f}")
                print(f"   📈 Win Rate: {metrics['win_rate']:.1%}")
                print(f"   💰 Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.1%}")
                print(f"   ⏱️ Training Time: {training_time:.1f}s")
                
                # Check if this is the best so far
                if score > self.best_score:
                    self.best_score = score
                    best_attempt = attempt_record
                    
                    # Save best model in organized folder structure
                    self._save_model_by_tier(model, tier, score, attempt)
                
                # Also save any model that reaches a tier (not just best)
                if tier in ['bronze', 'silver', 'gold', 'diamond']:
                    self._save_model_by_tier(model, tier, score, attempt, is_best=False)
                
                # Check if target reached
                if tier == target_tier or (target_tier == 'gold' and tier == 'diamond'):
                    print(f"\n🎉 TARGET REACHED! {emoji} {tier.upper()} tier achieved!")
                    break
                
                # Save successful config separately
                self.save_successful_config(hyperparams, metrics, score, tier, attempt)
                
                # Save history after each attempt
                self.save_history()
                
            except Exception as e:
                print(f"   ❌ Training failed: {e}")
                
                # Save failed config separately
                self.save_failed_config(hyperparams, str(e), attempt)
                
                # Record failed attempt
                failed_record = {
                    'attempt': attempt,
                    'timestamp': datetime.now().isoformat(),
                    'hyperparameters': hyperparams,
                    'error': str(e),
                    'score': 0,
                    'tier': 'failed'
                }
                self.training_history.append(failed_record)
            
            attempt += 1
            print()
        
        # Final summary
        print("="*60)
        print("🏁 TRAINING SUMMARY")
        print("="*60)
        
        if best_attempt:
            print(f"🏆 Best Performance:")
            print(f"   Attempt: {best_attempt['attempt']}")
            print(f"   Tier: {best_attempt['tier'].upper()}")
            print(f"   Score: {best_attempt['score']:.1f}")
            print(f"   Win Rate: {best_attempt['metrics']['win_rate']:.1%}")
            print(f"   Profit Factor: {best_attempt['metrics']['profit_factor']:.2f}")
            print(f"   Max Drawdown: {best_attempt['metrics']['max_drawdown']:.1%}")
        else:
            print("❌ No successful training attempts")
        
        # Save final history
        self.save_history()
        
        return best_attempt

def main():
    """Main training function"""
    symbol = 'XAUUSD'
    
    # Load data
    data_file = f"train_data/{symbol}/{symbol}_M5_real.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df):,} rows")
    
    # Initialize trainer
    trainer = AdaptiveTrainer(symbol)
    
    # Start adaptive training
    best_result = trainer.adaptive_train(df, max_attempts=20, target_tier='gold')
    
    if best_result:
        print(f"\n✅ Training completed successfully!")
        print(f"Best model tier: {best_result['tier'].upper()}")
    else:
        print(f"\n❌ Training failed to reach target")

if __name__ == "__main__":
    main()