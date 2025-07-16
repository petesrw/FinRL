#!/usr/bin/env python3
"""
Train XAUUSD model directly
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from forex_rl_simple import SimpleForexBot
from config import get_config

def main():
    """
    Train XAUUSD model
    """
    print("🔄 Starting XAUUSD Model Training...")
    print("="*50)
    
    # Get configuration
    config = get_config()
    
    # Check if training data exists
    data_file = "train_data/XAUUSD/XAUUSD_M5_real.csv"
    if not os.path.exists(data_file):
        print(f"❌ Training data not found: {data_file}")
        return
    
    # Load training data
    print(f"📊 Loading training data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df):,} rows")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize bot
    print("\n🤖 Initializing XAUUSD Trading Bot...")
    bot = SimpleForexBot(
        symbol='XAUUSD',
        model_type='PPO',
        training_timesteps=50000,  # Reduced for faster training
        lookback_window=100
    )
    
    # Train the model
    print("\n🚀 Starting training process...")
    print("   This may take several minutes...")
    
    try:
        # Train with the real data
        bot.train_with_data(df)
        
        print("\n✅ Training completed successfully!")
        
        # Test the trained model
        print("\n🧪 Testing trained model...")
        results = bot.test_model(df.tail(1000))  # Test on last 1000 rows
        
        print(f"\n📈 Test Results:")
        print(f"   Total Return: {results.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"   Win Rate: {results.get('win_rate', 0):.2%}")
        
        # Save model
        model_path = f"models/xauusd_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        bot.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()