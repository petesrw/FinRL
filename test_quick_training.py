#!/usr/bin/env python3
"""
Quick test to verify that the persistent counter fix works during training
"""
import asyncio
import sys
import pandas as pd
import numpy as np
from train_all_models import AdaptiveTrainer

async def test_quick_training():
    """Test if persistent counters work during actual training"""
    print("🚀 Starting quick training test...")
    
    # Create trainer
    trainer = AdaptiveTrainer(symbol='EURUSD')
    
    # Load data
    data_path = "train_data/EURUSD_M5_data.csv"
    try:
        data = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(data)} rows")
    except:
        print("❌ Failed to load data - using dummy data")
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='5T'),
            'open': np.random.randn(1000).cumsum() + 1.1000,
            'high': np.random.randn(1000).cumsum() + 1.1010,
            'low': np.random.randn(1000).cumsum() + 1.0990,
            'close': np.random.randn(1000).cumsum() + 1.1000,
            'volume': np.random.randint(1000, 10000, 1000)
        })
    
    # Quick training with 1 model only for testing
    config = {
        'algorithm': 'PPO',
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'lookback_window': 50,
        'transaction_cost': 0,
        'timesteps': 5000,  # Very short for testing
        'n_steps': 512,
        'batch_size': 64,
        'n_epochs': 3,
        'clip_range': 0.2,
        'ent_coef': 1.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    }
    
    try:
        print("🎯 Starting quick training...")
        result = await trainer.train_model_async(config, data, model_id=999)
        
        print(f"🏆 Training completed!")
        print(f"📊 Total trades: {result['metrics']['total_trades']}")
        print(f"📊 Win rate: {result['metrics']['win_rate']}")
        
        if result['metrics']['total_trades'] > 0:
            print("🎉 SUCCESS! Persistent counters are working - trades are being recorded!")
        else:
            print("❌ FAILED: Still showing 0 trades")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_quick_training())
