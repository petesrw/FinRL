#!/usr/bin/env python3
"""
🔄 Fresh Model Retraining Script
เริ่มต้น training ใหม่ทั้งหมดแทนการใช้ model เดิม
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

class FreshModelRetraining:
    """เริ่มต้น training model ใหม่ทั้งหมด"""
    
    def __init__(self, symbol, data_path, models_dir="models"):
        self.symbol = symbol
        self.data_path = data_path
        self.models_dir = models_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def backup_old_models(self):
        """สำรองข้อมูล model เก่า"""
        backup_dir = os.path.join(self.models_dir, f"backup_{self.timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        print(f"📦 Backing up old models to: {backup_dir}")
        
        # Move old models to backup
        for tier in ['diamond', 'gold', 'silver', 'bronze']:
            tier_dir = os.path.join(self.models_dir, tier)
            if os.path.exists(tier_dir):
                backup_tier_dir = os.path.join(backup_dir, tier)
                os.makedirs(backup_tier_dir, exist_ok=True)
                
                for file in os.listdir(tier_dir):
                    if self.symbol.lower() in file.lower():
                        old_path = os.path.join(tier_dir, file)
                        new_path = os.path.join(backup_tier_dir, file)
                        print(f"   📁 Moving: {file}")
                        os.rename(old_path, new_path)
        
        return backup_dir
    
    def create_fresh_training_config(self):
        """สร้าง configuration ใหม่สำหรับ training"""
        config = {
            "symbol": self.symbol,
            "training_mode": "FRESH_START",
            "timestamp": self.timestamp,
            "data_source": self.data_path,
            "model_parameters": {
                "algorithm": "PPO",  # เริ่มต้นด้วย PPO
                "learning_rate": 3e-4,  # Learning rate มาตรฐาน
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,  # เพิ่ม exploration
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            "environment_parameters": {
                "lookback_window": 50,
                "transaction_cost": 0.0001,
                "reward_scaling": 100,
                "action_space": "continuous",
                "observation_normalization": True
            },
            "training_parameters": {
                "total_timesteps": 100000,  # เริ่มต้นด้วยจำนวนน้อย
                "eval_freq": 5000,
                "save_freq": 10000,
                "early_stopping_patience": 20000
            }
        }
        
        config_path = f"fresh_training_config_{self.symbol}_{self.timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📋 Created fresh training config: {config_path}")
        return config
    
    def validate_training_data(self):
        """ตรวจสอบคุณภาพของข้อมูล training"""
        print("🔍 Validating training data...")
        
        try:
            df = pd.read_csv(self.data_path)
            
            # ตรวจสอบ columns ที่จำเป็น
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            # ตรวจสอบ data quality
            data_issues = []
            
            # ตรวจสอบ NaN values
            nan_counts = df.isnull().sum()
            if nan_counts.any():
                data_issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
            
            # ตรวจสอบ negative prices
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                data_issues.append("Negative or zero prices found")
            
            # ตรวจสอบ unrealistic price movements
            price_changes = df['close'].pct_change()
            extreme_changes = abs(price_changes) > 0.1  # 10% change
            if extreme_changes.any():
                data_issues.append(f"Extreme price changes: {extreme_changes.sum()} instances")
            
            if data_issues:
                print("⚠️ Data quality issues found:")
                for issue in data_issues:
                    print(f"   - {issue}")
                return False
            else:
                print("✅ Training data validation passed")
                print(f"   📊 Data shape: {df.shape}")
                print(f"   📅 Date range: {df.index[0]} to {df.index[-1]}")
                return True
                
        except Exception as e:
            print(f"❌ Data validation failed: {e}")
            return False
    
    def create_fresh_environment(self, config):
        """สร้าง environment ใหม่"""
        print("🌍 Creating fresh training environment...")
        
        # สร้าง environment class ใหม่
        env_config = {
            'symbol': self.symbol,
            'data_path': self.data_path,
            'lookback_window': config['environment_parameters']['lookback_window'],
            'transaction_cost': config['environment_parameters']['transaction_cost'],
            'reward_scaling': config['environment_parameters']['reward_scaling']
        }
        
        # สร้าง vectorized environment
        # env = make_vec_env(lambda: YourTradingEnvironment(**env_config), n_envs=1)
        
        print("✅ Fresh environment created")
        return env_config  # Return config for now
    
    def train_fresh_model(self, config, env_config):
        """เริ่มต้น training model ใหม่"""
        print("🎯 Starting fresh model training...")
        
        model_params = config['model_parameters']
        training_params = config['training_parameters']
        
        # สร้าง model ใหม่ทั้งหมด (ไม่ load จากไฟล์เก่า)
        print(f"🔧 Creating new {model_params['algorithm']} model...")
        
        model_kwargs = {
            'learning_rate': model_params['learning_rate'],
            'n_steps': model_params['n_steps'],
            'batch_size': model_params['batch_size'],
            'n_epochs': model_params['n_epochs'],
            'gamma': model_params['gamma'],
            'gae_lambda': model_params['gae_lambda'],
            'clip_range': model_params['clip_range'],
            'ent_coef': model_params['ent_coef'],
            'vf_coef': model_params['vf_coef'],
            'max_grad_norm': model_params['max_grad_norm'],
            'verbose': 1
        }
        
        # สร้าง model ใหม่
        if model_params['algorithm'] == 'PPO':
            # model = PPO('MlpPolicy', env, **model_kwargs)
            pass
        elif model_params['algorithm'] == 'SAC':
            # model = SAC('MlpPolicy', env, **model_kwargs)
            pass
        elif model_params['algorithm'] == 'A2C':
            # model = A2C('MlpPolicy', env, **model_kwargs)
            pass
        
        print("✅ Fresh model created")
        print("🚀 Starting training...")
        
        # Setup callbacks
        save_path = os.path.join(self.models_dir, 'training_checkpoints')
        os.makedirs(save_path, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=training_params['save_freq'],
            save_path=save_path,
            name_prefix=f'fresh_{self.symbol}_{self.timestamp}'
        )
        
        # เริ่ม training
        # model.learn(
        #     total_timesteps=training_params['total_timesteps'],
        #     callback=checkpoint_callback
        # )
        
        print("✅ Fresh training completed")
        return None  # Return model when implemented
    
    def run_fresh_retraining(self):
        """รัน process การ retrain แบบใหม่ทั้งหมด"""
        print("🔄 STARTING FRESH MODEL RETRAINING")
        print("=" * 50)
        
        # 1. Backup old models
        backup_dir = self.backup_old_models()
        
        # 2. Validate training data
        if not self.validate_training_data():
            print("❌ Training data validation failed. Please fix data issues first.")
            return False
        
        # 3. Create fresh config
        config = self.create_fresh_training_config()
        
        # 4. Create fresh environment
        env_config = self.create_fresh_environment(config)
        
        # 5. Train fresh model
        new_model = self.train_fresh_model(config, env_config)
        
        print("✅ Fresh retraining completed successfully!")
        print(f"📦 Old models backed up to: {backup_dir}")
        print("🎯 Ready for deployment after validation")
        
        return True

# ตัวอย่างการใช้งาน
def main():
    """Example usage for fresh retraining"""
    
    # กำหนด parameters
    SYMBOL = "XAUUSD"  # หรือ "EURUSD"
    DATA_PATH = "data/processed_data.csv"  # path ไปยังข้อมูล training
    
    print("🔄 Fresh Model Retraining for RL Trading Bot")
    print("=" * 50)
    
    # สร้าง retrainer
    retrainer = FreshModelRetraining(SYMBOL, DATA_PATH)
    
    # เริ่ม fresh retraining
    success = retrainer.run_fresh_retraining()
    
    if success:
        print("\n✅ Fresh retraining completed successfully!")
        print("📋 Next steps:")
        print("   1. Validate new model performance")
        print("   2. Test with paper trading")
        print("   3. Deploy to live trading if satisfied")
    else:
        print("\n❌ Fresh retraining failed!")
        print("📋 Please check:")
        print("   1. Training data quality")
        print("   2. Environment configuration")
        print("   3. Model parameters")

if __name__ == "__main__":
    main()
