#!/usr/bin/env python3
"""
🎯 Adaptive RL Training System for Forex - FIXED VERSION
Trains until reaching excellence targets with full tracking and GPU boost
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import json
from datetime import datetime
import warnings
import time
import random
import torch
warnings.filterwarnings('ignore')

# GPU Detection and Setup
def detect_and_setup_gpu():
    """Smart GPU detection with RTX 5060 TI optimization"""
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 GPU Detected: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            # RTX 5060 TI compatibility mode
            if "RTX 5060" in gpu_name or "RTX 50" in gpu_name:
                print("   🔧 RTX 5060 TI detected - applying compatibility mode...")
                
                try:
                    # Test CUDA compatibility first
                    test_tensor = torch.randn(16, 16, device=device, dtype=torch.float32)
                    test_result = torch.matmul(test_tensor, test_tensor.T)
                    test_sum = test_result.sum().item()
                    
                    print("   ✅ CUDA compatibility test passed!")
                    
                    # Conservative settings for RTX 5060 TI compatibility
                    torch.backends.cudnn.benchmark = False  # Disable for compatibility
                    torch.backends.cudnn.deterministic = True  # Enable for stability
                    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for compatibility
                    torch.backends.cudnn.allow_tf32 = False
                    
                    # Conservative memory usage
                    torch.cuda.set_per_process_memory_fraction(0.7)  # Reduced to 70%
                    torch.cuda.empty_cache()
                    torch.set_num_threads(8)  # Reduced threads
                    
                    print("   ✅ RTX 5060 TI compatibility mode applied!")
                    return device
                    
                except Exception as cuda_error:
                    print(f"   ❌ RTX 5060 TI CUDA compatibility failed: {str(cuda_error)[:100]}...")
                    print("   💻 Falling back to CPU training for stability")
                    return torch.device("cpu")
            else:
                # Standard GPU setup
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                return device
                
        except Exception as e:
            print(f"   ⚠️ GPU initialization failed: {str(e)[:100]}...")
            print("   💻 Using CPU training instead")
            return torch.device("cpu")
    else:
        print("💻 No CUDA GPU detected - using CPU")
        return torch.device("cpu")

# Setup device globally
DEVICE = detect_and_setup_gpu()

def get_optimal_batch_size(device, base_batch_size=64):
    """Get optimal batch size for RTX 5060 TI"""
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        
        if "RTX 5060" in gpu_name and gpu_memory_gb >= 15:
            return min(base_batch_size * 8, 2048)  # RTX 5060 TI boost
        elif gpu_memory_gb >= 16:
            return min(base_batch_size * 6, 1536)
        elif gpu_memory_gb >= 8:
            return min(base_batch_size * 4, 1024)
        else:
            return min(base_batch_size * 2, 512)
    return base_batch_size

def get_optimal_timesteps(device, base_timesteps=100000):
    """Get optimal timesteps for RTX 5060 TI"""
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        
        if "RTX 5060" in gpu_name and gpu_memory_gb >= 15:
            return min(base_timesteps * 4, 500000)  # RTX 5060 TI boost
        elif gpu_memory_gb >= 16:
            return min(base_timesteps * 3, 400000)
        elif gpu_memory_gb >= 8:
            return min(base_timesteps * 2, 300000)
        else:
            return min(base_timesteps * 1.5, 200000)
    return base_timesteps

class AdvancedForexEnv(gym.Env):
    """Advanced Forex Environment with comprehensive metrics"""
    
    def __init__(self, data, symbol='XAUUSD', initial_balance=10000, lookback_window=50, 
                 transaction_cost=0.0001, max_position_size=1.0):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Calculate indicators
        self._calculate_indicators()
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
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Trade history
        self.trades = []
        self.equity_curve = [initial_balance]
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, 13), 
            dtype=np.float32
        )
    
    def _calculate_indicators(self):
        """Calculate technical indicators"""
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
        """Get current observation"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        obs_data = self.data.iloc[start_idx:end_idx][
            ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi', 'macd', 
             'macd_signal', 'bb_upper', 'bb_lower', 'atr']
        ].values
        
        # Add position information
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
        """Execute one step"""
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
            if self.position == 1:
                profit = (current_price - self.entry_price) * self.position_size
            else:
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
                'timestamp': self.data.iloc[self.current_step].get('timestamp', self.current_step)
            })
            
            if profit > 0:
                self.profitable_trades += 1
                self.total_profit += profit
                self.consecutive_losses = 0
                reward = 2
            else:
                self.total_loss += abs(profit)
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                reward = -1
            
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
        
        # Track equity curve and drawdown
        self.equity_curve.append(self.equity)
        self.max_equity = max(self.max_equity, self.equity)
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Enhanced reward shaping
        if self.max_drawdown > 0.15:
            reward -= 3
        elif self.max_drawdown < 0.05:
            reward += 1
        
        if self.consecutive_losses >= 3:
            reward -= 2
        
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
            total_return = (self.equity - self.initial_balance) / self.initial_balance
            reward += total_return * 20
        
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
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_losses': self.max_consecutive_losses
        }

class AdaptiveTrainer:
    """Adaptive trainer with simplified config management"""
    
    def __init__(self, symbol='XAUUSD'):
        self.symbol = symbol
        self.training_history = []
        self.best_model = None
        self.best_score = 0
        
        # Create directories
        self.base_dir = "training_logs"
        self.history_dir = f"{self.base_dir}/history"
        self.models_dir = "models"
        
        for directory in [self.base_dir, self.history_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.history_file = f"{self.history_dir}/{symbol.lower()}_training_history.json"
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
    
    def calculate_score(self, metrics):
        """Calculate overall performance score"""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        score = (
            win_rate * 40 +
            min(profit_factor / 3.0, 1.0) * 30 +
            max(0, 1 - max_drawdown * 2) * 20 +
            min(sharpe_ratio / 2.0, 1.0) * 10
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
        """Generate fresh hyperparameters - simplified approach"""
        algorithm = random.choice(['PPO', 'A2C'])
        
        base_batch_size = random.choice([128, 256, 512])
        optimal_batch_size = get_optimal_batch_size(DEVICE, base_batch_size)
        
        base_timesteps = random.choice([200000, 300000, 400000])
        optimal_timesteps = get_optimal_timesteps(DEVICE, base_timesteps)
        
        config = {
            'algorithm': algorithm,
            'learning_rate': random.choice([0.0001, 0.0003, 0.001]),
            'n_steps': random.choice([4096, 8192, 16384]) if algorithm == 'PPO' else None,
            'batch_size': optimal_batch_size,
            'gamma': random.choice([0.95, 0.99, 0.995]),
            'lookback_window': random.choice([100, 200, 300]),
            'transaction_cost': random.choice([0.0001, 0.0002, 0.0005]),
            'timesteps': optimal_timesteps
        }
        
        return config
    
    def train_model(self, data, hyperparameters):
        """Train a single model"""
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
        
        # Model configuration
        model_kwargs = {
            "device": DEVICE,
            "verbose": 0,
            "tensorboard_log": None,
        }
        
        # GPU optimizations
        if DEVICE.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            
            if "RTX 5060" in gpu_name or "RTX 50" in gpu_name:
                # RTX 5060 TI conservative compatibility mode
                gpu_config = {
                    "policy_kwargs": {
                        "net_arch": [1024, 1024, 512, 256],  # Smaller networks for compatibility
                        "activation_fn": torch.nn.ReLU,
                        "ortho_init": False,
                    },
                    "batch_size": min(hyperparameters['batch_size'], 512),  # Smaller batch size
                }
                
                if hyperparameters['algorithm'] == 'PPO':
                    gpu_config.update({
                        "n_steps": min(hyperparameters.get('n_steps', 4096), 4096),  # Smaller n_steps
                        "gae_lambda": 0.95,
                        "clip_range": 0.2,
                        "ent_coef": 0.01,
                        "vf_coef": 0.5,
                        "max_grad_norm": 0.5,
                    })
                
                print(f"   🔧 RTX 5060 TI COMPATIBILITY MODE:")
                print(f"   📈 Neural Networks: {gpu_config['policy_kwargs']['net_arch']}")
                print(f"   🎯 Batch Size: {gpu_config['batch_size']}")
                print(f"   🛡️ Conservative settings for stability")
            else:
                # Standard GPU optimization
                gpu_config = {
                    "policy_kwargs": {
                        "net_arch": [2048, 2048, 1024, 512],
                        "activation_fn": torch.nn.ReLU,
                        "ortho_init": False,
                    },
                    "batch_size": max(hyperparameters['batch_size'], 1024)
                }
                
                if hyperparameters['algorithm'] == 'PPO':
                    gpu_config["n_steps"] = max(hyperparameters.get('n_steps', 4096), 8192)
            
            model_kwargs.update(gpu_config)
        
        # Create model
        if hyperparameters['algorithm'] == 'PPO':
            batch_size = model_kwargs.get('batch_size', hyperparameters['batch_size'])
            n_steps = model_kwargs.get('n_steps', hyperparameters['n_steps'])
            
            ppo_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size', 'n_steps']}
            
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=hyperparameters['learning_rate'],
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=hyperparameters['gamma'],
                **ppo_kwargs
            )
        else:  # A2C
            a2c_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size', 'n_steps']}
            
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=hyperparameters['learning_rate'],
                gamma=hyperparameters['gamma'],
                **a2c_kwargs
            )
        
        print(f"   🖥️ Using device: {DEVICE}")
        
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
                    
                    # Save best model
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_path = f"{self.models_dir}/{self.symbol.lower()}_best_{tier}_score{score:.0f}_attempt{attempt}_{timestamp}.zip"
                    model.save(model_path)
                    print(f"   💾 New best model saved: {model_path}")
                
                # Check if target reached
                if tier == target_tier or (target_tier == 'gold' and tier == 'diamond'):
                    print(f"\n🎉 TARGET REACHED! {emoji} {tier.upper()} tier achieved!")
                    break
                
                # Save history after each attempt
                self.save_history()
                
            except Exception as e:
                print(f"   ❌ Training failed: {e}")
                
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
                self.save_history()
            
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
    best_result = trainer.adaptive_train(df, max_attempts=50, target_tier='gold')
    
    if best_result:
        print(f"\n✅ Training completed successfully!")
        print(f"Best model tier: {best_result['tier'].upper()}")
    else:
        print(f"\n❌ Training failed to reach target")

if __name__ == "__main__":
    main()