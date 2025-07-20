#!/usr/bin/env python3
"""
Simple script to restart optimized training
"""
import sys
import os
import json
from datetime import datetime

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("🚀 STARTING OPTIMIZED TRAINING")
    print("="*50)
    
    # Backup previous results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"training_logs/backup_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    src_file = "training_logs/async_logs/xauusd_async_training.json"
    if os.path.exists(src_file):
        import shutil
        dst_file = f"{backup_dir}/xauusd_async_training_backup.json"
        shutil.copy2(src_file, dst_file)
        print(f"✅ Backed up previous results to: {dst_file}")
    
    # Clear previous log
    os.makedirs("training_logs/async_logs", exist_ok=True)
    with open("training_logs/async_logs/xauusd_async_training.json", 'w') as f:
        json.dump([], f)
    print("✅ Cleared previous training log")
    
    print("\n🎯 OPTIMIZED CONFIGURATION APPLIED:")
    print("   ✅ Enhanced profit-factor focused reward function")
    print("   ✅ PPO-prioritized algorithm selection")  
    print("   ✅ Optimized hyperparameter ranges")
    print("   ✅ Increased training timesteps")
    print("   ✅ Better risk management")
    
    print("\n💡 IMPROVEMENTS MADE:")
    print("   🎯 Reward System:")
    print("      - Primary focus on profit factor > 1.0")
    print("      - Enhanced rewards for profit factor > 1.5, 2.0")
    print("      - Stricter penalties for poor risk management")
    print("      - Multi-factor final episode rewards")
    print("   📊 Hyperparameters:")
    print("      - Focused learning rates: 0.0005-0.001 (proven range)")
    print("      - PPO algorithm prioritization (80% selection rate)")
    print("      - Smaller n_steps values (2048, 4096) work better")
    print("      - Lower transaction costs (0.0001-0.0002)")
    print("      - Extended training to 500K-1M timesteps")
    
    print(f"\n📁 Previous results backed up to: {backup_dir}")
    print("🔄 Ready to start new training with optimized settings")
    
    # Now import and start the actual training
    print("\n🚀 Importing training modules...")
    from train_all_models import AdaptiveTrainer
    import asyncio
    import pandas as pd
    
    async def start_training():
        try:
            # Load data
            data_file = "xauusd_professional_history.json"
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    raw_data = json.load(f)
                df = pd.DataFrame(raw_data)
                print(f"✅ Loaded {len(df)} data points from professional history")
            else:
                # Create test data if professional data not available
                print("⚠️ Using synthetic data for testing...")
                dates = pd.date_range(start='2024-01-01', periods=8000, freq='5min')
                import numpy as np
                np.random.seed(42)
                
                base_price = 2000.0
                price_changes = np.random.normal(0, 2, len(dates))
                prices = base_price + np.cumsum(price_changes)
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices + np.random.normal(0, 0.5, len(dates)),
                    'high': prices + np.abs(np.random.normal(2, 1, len(dates))),
                    'low': prices - np.abs(np.random.normal(2, 1, len(dates))),
                    'close': prices,
                    'volume': np.random.randint(100, 1000, len(dates))
                })
                
                # Fix OHLC relationships
                for i in range(len(df)):
                    df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'high'])
                    df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'low'])
                
                print(f"✅ Created synthetic data: {len(df)} points")
            
            # Initialize trainer with optimized settings
            print("\n🎯 Initializing AdaptiveTrainer...")
            trainer = AdaptiveTrainer('XAUUSD')
            
            print("🔥 Starting optimized async training...")
            print("   Target: Profit Factor > 1.5")
            print("   Target: Win Rate > 60%")
            print("   Target: Max Drawdown < 10%")
            
            # Start training with reasonable parameters
            best_result = await trainer.adaptive_train_async(
                df,
                max_attempts=30,        # Start with reasonable number
                target_tier='silver',   # Achievable target
                batch_size=3,           # Small batch for stability
                patience=10             # Allow time for learning
            )
            
            if best_result:
                print(f"\n🎉 TRAINING SUCCESSFUL!")
                print(f"   🏆 Best Score: {best_result.get('score', 0):.2f}")
                print(f"   💰 Profit Factor: {best_result.get('metrics', {}).get('profit_factor', 0):.3f}")
                print(f"   🎯 Win Rate: {best_result.get('metrics', {}).get('win_rate', 0):.1%}")
                print(f"   📈 Total Return: {best_result.get('metrics', {}).get('total_return', 0):.1%}")
            else:
                print("\n⚠️ No excellent model found yet")
                print("💡 Continue training or adjust parameters as needed")
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the training
    asyncio.run(start_training())
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Script completed")
