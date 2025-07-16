#!/usr/bin/env python3
"""
🎯 Professional RL Training System for Forex
Full data, advanced indicators, serious training for excellence
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import json
from datetime import datetime
import warnings
import time
import random
warnings.filterwarnings('ignore')

class ProgressCallback(BaseCallback):
    """
    Custom callback to show training progress
    """
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            progress = self.n_calls / self.locals.get('total_timesteps', 1)
            print(f"   Progress: {progress:.1%} | Steps: {self.n_calls:,} | Time: {elapsed:.0f}s")
        return True

class ProfessionalForexEnv(gym.Env):
    """
    Professional Forex Environment with full indicators and comprehensive metrics
    """
    
    def __init__(self, data, symbol='XAUUSD', initial_balance=10000, lookback_window=100, 
                 transaction_cost=0.0002, max_position_size=1.0):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Calculate comprehensive indicators
        self._calculate_indicators()
        
        # Remove NaN values
        self.data = self.data.dropna().reset_index(drop=True)
        print(f"   Environment initialized with {len(self.data):,} data points")
        
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
        
        # Advanced metrics
        self.trades = []
        self.equity_curve = [initial_balance]
        self.daily_returns = []
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation space: comprehensive indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback_window, 15), 
            dtype=np.float32
        )
    
    def _calculate_indicators(self):
        """Calculate comprehensive technical indicators"""
        print("   Calculating advanced technical indicators...")
        
        # Price-based indicators
        self.data['returns'] = self.data['close'].pct_change()
        self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Moving Averages (multiple timeframes)
        for period in [5, 10, 20, 50, 100]:
            self.data[f'sma_{period}'] = self.data['close'].rolling(window=period).mean()
            self.data[f'ema_{period}'] = self.data['close'].ewm(span=period).mean()
        
        # RSI (multiple timeframes)
        for period in [14, 21]:
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.data['ema_12'] = self.data['close'].ewm(span=12).mean()
        self.data['ema_26'] = self.data['close'].ewm(span=26).mean()
        self.data['macd'] = self.data['ema_12'] - self.data['ema_26']
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
        
        # Bollinger Bands
        self.data['bb_middle'] = self.data['close'].rolling(window=20).mean()
        bb_std = self.data['close'].rolling(window=20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + (bb_std * 2)
        self.data['bb_lower'] = self.data['bb_middle'] - (bb_std * 2)
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
        self.data['bb_position'] = (self.data['close'] - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])
        
        # ATR (Average True Range)
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.data['atr'] = true_range.rolling(14).mean()
        self.data['atr_percent'] = self.data['atr'] / self.data['close']
        
        # Stochastic Oscillator
        lowest_low = self.data['low'].rolling(window=14).min()
        highest_high = self.data['high'].rolling(window=14).max()
        self.data['stoch_k'] = 100 * (self.data['close'] - lowest_low) / (highest_high - lowest_low)
        self.data['stoch_d'] = self.data['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        self.data['williams_r'] = -100 * (highest_high - self.data['close']) / (highest_high - lowest_low)
        
        # Commodity Channel Index (CCI)
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        self.data['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Volume indicators (using synthetic volume)
        self.data['volume_sma'] = self.data['volume'].rolling(window=20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma']
        
        print(f"   Calculated {len([col for col in self.data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")
    
    def _get_observation(self):
        """Get current observation with selected indicators"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        # Select key indicators for observation
        selected_indicators = [
            'open', 'high', 'low', 'close',
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal',
            'bb_position', 'atr_percent', 'stoch_k', 'williams_r', 'cci', 'volume_ratio'
        ]
        
        obs_data = self.data.iloc[start_idx:end_idx][selected_indicators].values
        
        # Normalize data
        obs_data = (obs_data - np.mean(obs_data, axis=0)) / (np.std(obs_data, axis=0) + 1e-8)
        
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
        self.daily_returns = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step with professional reward calculation"""
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
                reward = 3  # Higher reward for profitable trade
            else:
                self.total_loss += abs(profit)
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                reward = -2  # Penalty for losing trade
            
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
        
        # Advanced reward shaping
        if self.max_drawdown > 0.15:  # Penalty for high drawdown
            reward -= 3
        
        if self.consecutive_losses >= 5:  # Penalty for consecutive losses
            reward -= 2
        
        # Reward for maintaining good risk management
        if self.max_drawdown < 0.05:
            reward += 0.5
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.equity <= self.initial_balance * 0.3
        
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros((self.lookback_window, 15), dtype=np.float32)
            
            # Final reward based on comprehensive performance
            total_return = (self.equity - self.initial_balance) / self.initial_balance
            reward += total_return * 50  # Higher scale for final reward
        
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
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0
            }
        
        total_return = (self.equity - self.initial_balance) / self.initial_balance
        win_rate = self.profitable_trades / self.total_trades
        profit_factor = self.total_profit / max(self.total_loss, 1e-8)
        
        # Calculate advanced ratios
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            
            # Sharpe ratio
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                sortino_ratio = np.mean(returns) / (np.std(downside_returns) + 1e-8) * np.sqrt(252)
            else:
                sortino_ratio = sharpe_ratio
            
            # Calmar ratio
            calmar_ratio = (total_return * 252) / max(self.max_drawdown, 1e-8)
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
        
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
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_consecutive_losses': self.max_consecutive_losses
        }

class ProfessionalTrainer:
    """
    Professional trainer for serious results
    """
    
    def __init__(self, symbol='XAUUSD'):
        self.symbol = symbol
        self.training_history = []
        self.best_model = None
        self.best_score = 0
        self.history_file = f"{symbol.lower()}_professional_history.json"
        self.load_history()
        
        # Professional excellence targets
        self.targets = {
            'bronze': {'win_rate': 0.70, 'profit_factor': 2.0, 'max_drawdown': 0.15, 'sharpe': 1.0},
            'silver': {'win_rate': 0.75, 'profit_factor': 2.5, 'max_drawdown': 0.12, 'sharpe': 1.5},
            'gold': {'win_rate': 0.80, 'profit_factor': 2.8, 'max_drawdown': 0.10, 'sharpe': 2.0},
            'diamond': {'win_rate': 0.85, 'profit_factor': 3.2, 'max_drawdown': 0.08, 'sharpe': 2.5}
        }
    
    def load_history(self):
        """Load training history"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.training_history = json.load(f)
            print(f"📚 Loaded {len(self.training_history)} previous professional training attempts")
        else:
            print("🆕 Starting fresh professional training history")
    
    def save_history(self):
        """Save training history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def calculate_professional_score(self, metrics):
        """Calculate professional performance score"""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        sortino_ratio = metrics.get('sortino_ratio', 0)
        calmar_ratio = metrics.get('calmar_ratio', 0)
        
        # Professional weighted scoring
        score = (
            win_rate * 25 +  # 25% weight on win rate
            min(profit_factor / 4.0, 1.0) * 25 +  # 25% weight on profit factor
            max(0, 1 - max_drawdown * 3) * 20 +  # 20% weight on drawdown control
            min(sharpe_ratio / 3.0, 1.0) * 15 +  # 15% weight on Sharpe ratio
            min(sortino_ratio / 3.0, 1.0) * 10 +  # 10% weight on Sortino ratio
            min(calmar_ratio / 10.0, 1.0) * 5  # 5% weight on Calmar ratio
        ) * 100
        
        return score
    
    def get_professional_tier(self, metrics):
        """Determine professional performance tier"""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        if (win_rate >= self.targets['diamond']['win_rate'] and 
            profit_factor >= self.targets['diamond']['profit_factor'] and 
            max_drawdown <= self.targets['diamond']['max_drawdown'] and
            sharpe_ratio >= self.targets['diamond']['sharpe']):
            return 'diamond', '💎'
        elif (win_rate >= self.targets['gold']['win_rate'] and 
              profit_factor >= self.targets['gold']['profit_factor'] and 
              max_drawdown <= self.targets['gold']['max_drawdown'] and
              sharpe_ratio >= self.targets['gold']['sharpe']):
            return 'gold', '🥇'
        elif (win_rate >= self.targets['silver']['win_rate'] and 
              profit_factor >= self.targets['silver']['profit_factor'] and 
              max_drawdown <= self.targets['silver']['max_drawdown'] and
              sharpe_ratio >= self.targets['silver']['sharpe']):
            return 'silver', '🥈'
        elif (win_rate >= self.targets['bronze']['win_rate'] and 
              profit_factor >= self.targets['bronze']['profit_factor'] and 
              max_drawdown <= self.targets['bronze']['max_drawdown'] and
              sharpe_ratio >= self.targets['bronze']['sharpe']):
            return 'bronze', '🥉'
        else:
            return 'none', '❌'
    
    def generate_professional_hyperparameters(self):
        """Generate professional hyperparameters"""
        # Professional configurations
        configs = [
            # Conservative but effective
            {'algorithm': 'PPO', 'learning_rate': 0.0001, 'n_steps': 2048, 'batch_size': 64, 'gamma': 0.99, 'timesteps': 150000},
            {'algorithm': 'PPO', 'learning_rate': 0.0003, 'n_steps': 4096, 'batch_size': 128, 'gamma': 0.995, 'timesteps': 200000},
            
            # Aggressive learning
            {'algorithm': 'PPO', 'learning_rate': 0.001, 'n_steps': 1024, 'batch_size': 32, 'gamma': 0.99, 'timesteps': 100000},
            {'algorithm': 'A2C', 'learning_rate': 0.0003, 'batch_size': 64, 'gamma': 0.99, 'timesteps': 120000},
            
            # Balanced approach
            {'algorithm': 'PPO', 'learning_rate': 0.0005, 'n_steps': 2048, 'batch_size': 64, 'gamma': 0.995, 'timesteps': 180000},
            {'algorithm': 'A2C', 'learning_rate': 0.0001, 'batch_size': 128, 'gamma': 0.995, 'timesteps': 150000},
        ]
        
        # Avoid failed configurations
        failed_configs = [h['hyperparameters'] for h in self.training_history if h.get('score', 0) < 60]
        
        for config in configs:
            config_str = str(sorted(config.items()))
            failed_config_strs = [str(sorted(fc.items())) for fc in failed_configs]
            if config_str not in failed_config_strs:
                return config
        
        # Return default if all tried
        return configs[0]
    
    def professional_train(self, data, max_attempts=10, target_tier='gold'):
        """Professional training with full data and advanced techniques"""
        print(f"🎯 PROFESSIONAL TRAINING FOR {self.symbol}")
        print("="*60)
        print(f"Target: {target_tier.upper()} tier")
        print(f"Max attempts: {max_attempts}")
        print(f"Using full dataset: {len(data):,} rows")
        print()
        
        attempt = 1
        best_attempt = None
        
        while attempt <= max_attempts:
            print(f"🔄 Professional Attempt {attempt}/{max_attempts}")
            
            # Generate hyperparameters
            hyperparams = self.generate_professional_hyperparameters()
            
            try:
                print(f"🚀 Training with professional configuration:")
                for key, value in hyperparams.items():
                    if value is not None:
                        print(f"   {key}: {value}")
                
                # Create professional environment
                env = ProfessionalForexEnv(
                    data, 
                    symbol=self.symbol,
                    lookback_window=100,
                    transaction_cost=0.0002
                )
                env = DummyVecEnv([lambda: env])
                
                # Create model
                if hyperparams['algorithm'] == 'PPO':
                    model = PPO(
                        "MlpPolicy",
                        env,
                        learning_rate=hyperparams['learning_rate'],
                        n_steps=hyperparams['n_steps'],
                        batch_size=hyperparams['batch_size'],
                        gamma=hyperparams['gamma'],
                        verbose=0
                    )
                else:  # A2C
                    model = A2C(
                        "MlpPolicy",
                        env,
                        learning_rate=hyperparams['learning_rate'],
                        gamma=hyperparams['gamma'],
                        verbose=0
                    )
                
                # Train with progress callback
                print("   🚀 Starting professional training...")
                callback = ProgressCallback(check_freq=20000)
                start_time = time.time()
                model.learn(total_timesteps=hyperparams['timesteps'], callback=callback)
                training_time = time.time() - start_time
                
                print(f"   ✅ Training completed in {training_time:.1f}s")
                
                # Professional testing on larger dataset
                print("   🧪 Professional testing...")
                test_env = ProfessionalForexEnv(
                    data.tail(15000),  # Use more data for testing
                    symbol=self.symbol,
                    lookback_window=100,
                    transaction_cost=0.0002
                )
                
                obs, _ = test_env.reset()
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = test_env.step(action)
                
                metrics = info
                score = self.calculate_professional_score(metrics)
                tier, emoji = self.get_professional_tier(metrics)
                
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
                
                # Print professional results
                print(f"   {emoji} Professional Tier: {tier.upper()}")
                print(f"   📊 Professional Score: {score:.1f}")
                print(f"   📈 Win Rate: {metrics['win_rate']:.1%}")
                print(f"   💰 Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.1%}")
                print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"   📊 Sortino Ratio: {metrics['sortino_ratio']:.2f}")
                print(f"   💵 Total Return: {metrics['total_return']:.1%}")
                print(f"   🔢 Total Trades: {metrics['total_trades']}")
                print(f"   ⏱️ Training Time: {training_time:.1f}s")
                
                # Check if this is the best so far
                if score > self.best_score:
                    self.best_score = score
                    best_attempt = attempt_record
                    
                    # Save best model
                    os.makedirs('models', exist_ok=True)
                    model_path = f"models/{self.symbol.lower()}_professional_{tier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    model.save(model_path)
                    print(f"   💾 New best professional model saved: {model_path}")
                
                # Check if target reached
                if tier == target_tier or (target_tier == 'gold' and tier == 'diamond'):
                    print(f"\n🎉 PROFESSIONAL TARGET REACHED! {emoji} {tier.upper()} tier achieved!")
                    break
                
                # Save history after each attempt
                self.save_history()
                
            except Exception as e:
                print(f"   ❌ Professional training failed: {e}")
                import traceback
                traceback.print_exc()
                
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
        
        # Professional summary
        print("="*60)
        print("🏁 PROFESSIONAL TRAINING SUMMARY")
        print("="*60)
        
        if best_attempt:
            print(f"🏆 Best Professional Performance:")
            print(f"   Attempt: {best_attempt['attempt']}")
            print(f"   Tier: {best_attempt['tier'].upper()}")
            print(f"   Score: {best_attempt['score']:.1f}")
            print(f"   Win Rate: {best_attempt['metrics']['win_rate']:.1%}")
            print(f"   Profit Factor: {best_attempt['metrics']['profit_factor']:.2f}")
            print(f"   Max Drawdown: {best_attempt['metrics']['max_drawdown']:.1%}")
            print(f"   Sharpe Ratio: {best_attempt['metrics']['sharpe_ratio']:.2f}")
            print(f"   Total Return: {best_attempt['metrics']['total_return']:.1%}")
        else:
            print("❌ No successful professional training attempts")
        
        # Save final history
        self.save_history()
        
        return best_attempt

def main():
    """Main professional training function"""
    symbol = 'XAUUSD'
    
    # Load full data
    data_file = f"train_data/{symbol}/{symbol}_M5_real.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading full professional dataset from {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df):,} rows for professional training")
    
    # Initialize professional trainer
    trainer = ProfessionalTrainer(symbol)
    
    # Start professional training
    best_result = trainer.professional_train(df, max_attempts=6, target_tier='gold')
    
    if best_result:
        print(f"\n✅ Professional training completed successfully!")
        print(f"Best professional model tier: {best_result['tier'].upper()}")
        
        # Check achievement
        tier = best_result['tier']
        if tier == 'diamond':
            print("💎 DIAMOND TIER ACHIEVED - EXCEPTIONAL PERFORMANCE!")
        elif tier == 'gold':
            print("🥇 GOLD TIER ACHIEVED - EXCELLENT PERFORMANCE!")
        elif tier == 'silver':
            print("🥈 SILVER TIER ACHIEVED - VERY GOOD PERFORMANCE!")
        elif tier == 'bronze':
            print("🥉 BRONZE TIER ACHIEVED - GOOD PERFORMANCE!")
    else:
        print(f"\n❌ Professional training failed to reach target")

if __name__ == "__main__":
    main()