#!/usr/bin/env python3
"""
🚀 Restart Training with Optimized Settings
Based on analysis of previous results, restart with improved configuration
"""

import os
import shutil
import json
from datetime import datetime
import asyncio
from train_all_models import AdaptiveTrainer
import pandas as pd

def backup_previous_results():
    """Backup previous training results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"training_logs/backup_{timestamp}"
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup current async training log
    src_file = "training_logs/async_logs/xauusd_async_training.json"
    if os.path.exists(src_file):
        dst_file = f"{backup_dir}/xauusd_async_training_backup.json"
        shutil.copy2(src_file, dst_file)
        print(f"✅ Backed up previous results to: {dst_file}")
    
    return backup_dir

def analyze_best_models():
    """Analyze best performing models from previous runs"""
    log_file = "training_logs/async_logs/xauusd_async_training.json"
    
    if not os.path.exists(log_file):
        print("⚠️ No previous training log found")
        return None
    
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        best_models = []
        for entry in data:
            if isinstance(entry, dict) and 'results' in entry:
                for result in entry['results']:
                    if isinstance(result, dict) and 'score' in result and result.get('score', 0) > 30:
                        best_models.append({
                            'score': result['score'],
                            'hyperparameters': result['hyperparameters'],
                            'metrics': result['metrics']
                        })
        
        if best_models:
            # Sort by score
            best_models.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"📊 Found {len(best_models)} promising models:")
            for i, model in enumerate(best_models[:3], 1):
                print(f"   {i}. Score: {model['score']:.2f}")
                print(f"      Algorithm: {model['hyperparameters']['algorithm']}")
                print(f"      Learning Rate: {model['hyperparameters']['learning_rate']}")
                print(f"      Profit Factor: {model['metrics'].get('profit_factor', 0):.3f}")
                print(f"      Win Rate: {model['metrics'].get('win_rate', 0):.1%}")
                print()
            
            return best_models[0]  # Return the best model
        else:
            print("📊 No high-scoring models found in previous runs")
            return None
            
    except Exception as e:
        print(f"❌ Error analyzing previous results: {e}")
        return None

def setup_optimized_training_environment():
    """Setup environment with optimized settings"""
    print("🔧 Setting up optimized training environment...")
    
    # Create fresh log file
    os.makedirs("training_logs/async_logs", exist_ok=True)
    
    # Initialize with empty array for new training session
    with open("training_logs/async_logs/xauusd_async_training.json", 'w') as f:
        json.dump([], f)
    
    print("✅ Fresh training environment prepared")

async def start_optimized_training():
    """Start optimized async training"""
    print("🚀 STARTING OPTIMIZED ASYNC TRAINING")
    print("="*60)
    
    # Backup previous results
    backup_dir = backup_previous_results()
    
    # Analyze best models from previous runs
    best_model = analyze_best_models()
    
    if best_model:
        print("🎯 Using insights from best performing model:")
        print(f"   Best Score Achieved: {best_model['score']:.2f}")
        print(f"   Best Algorithm: {best_model['hyperparameters']['algorithm']}")
        print(f"   Best Learning Rate: {best_model['hyperparameters']['learning_rate']}")
    
    # Setup fresh training environment
    setup_optimized_training_environment()
    
    print("\n🎯 OPTIMIZED TRAINING CONFIGURATION:")
    print("   ✅ Enhanced reward function (profit factor focused)")
    print("   ✅ PPO-prioritized algorithm selection")
    print("   ✅ Optimized hyperparameter ranges")
    print("   ✅ Increased timesteps for better learning")
    print("   ✅ Improved risk management rewards")
    print("   ✅ Multi-factor performance evaluation")
    
    # Load training data
    print("\n📈 Loading XAUUSD training data...")
    try:
        # Use the same data loading approach as the original system
        data_file = "xauusd_professional_history.json"
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                raw_data = json.load(f)
            
            # Convert to DataFrame (assuming the data structure from original)
            df = pd.DataFrame(raw_data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print(f"   ✅ Loaded {len(df):,} data points")
            print(f"   📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print("   ⚠️ Professional data not found, using synthetic data...")
            # Create synthetic data for testing
            from test_async_training import create_test_data
            df = create_test_data('XAUUSD', 8000)
        
        # Initialize optimized trainer
        trainer = AdaptiveTrainer('XAUUSD')
        
        print(f"\n🔥 Starting optimized training with:")
        print(f"   🎯 Target: Profit Factor > 1.5")
        print(f"   🎯 Target: Win Rate > 60%")
        print(f"   🎯 Target: Max Drawdown < 10%")
        print(f"   🎯 Target: Sharpe Ratio > 1.0")
        
        # Start optimized async training
        best_result = await trainer.adaptive_train_async(
            df,
            max_attempts=50,      # Reasonable number for quality training
            target_tier='gold',   # Aim higher with improved system
            batch_size=4,         # Optimal batch size for GPU utilization
            patience=15           # Allow more time for convergence
        )
        
        if best_result:
            print(f"\n🎉 TRAINING COMPLETED!")
            print(f"   🏆 Best Score: {best_result.get('score', 0):.2f}")
            print(f"   💰 Profit Factor: {best_result.get('metrics', {}).get('profit_factor', 0):.3f}")
            print(f"   🎯 Win Rate: {best_result.get('metrics', {}).get('win_rate', 0):.1%}")
            print(f"   📈 Total Return: {best_result.get('metrics', {}).get('total_return', 0):.1%}")
        else:
            print("⚠️ Training completed but no excellent model found yet")
            print("💡 Consider running additional training cycles or adjusting parameters")
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("💡 Check data files and system configuration")
        
    print(f"\n📁 Previous results backed up to: {backup_dir}")
    print("🔚 Training session completed")

if __name__ == "__main__":
    print("🚀 FINRL OPTIMIZED TRAINING RESTART")
    print("📊 Enhanced with profit-factor focused rewards")
    print("🎯 PPO-prioritized algorithm selection")
    print("⚡ Improved hyperparameter optimization")
    print()
    
    # Run the optimized training
    asyncio.run(start_optimized_training())
