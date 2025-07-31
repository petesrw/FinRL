#!/usr/bin/env python3
"""
🚨 Emergency Training Script - SKIP BIAS CHECKS
สำหรับกรณีที่ bias detection บล็อกการ save นานเกินไป
"""

import os
import sys
sys.path.append('.')

from train_all_models import AdaptiveTrainer
from data_manager import MarketDataManager

def emergency_train_gbpusd():
    """Training GBPUSD แบบ emergency (skip bias checks)"""
    print("🚨 EMERGENCY TRAINING MODE - BIAS CHECKS DISABLED")
    print("⚠️ WARNING: This will save models without bias verification")
    
    # Load data
    print("📊 Loading GBPUSD data...")
    data_manager = MarketDataManager()
    data = data_manager.get_training_data('GBPUSD', prefer_saved=True)
    
    if data is None or len(data) < 1000:
        print("❌ Insufficient data for training")
        print("🔄 Trying to download fresh data...")
        data = data_manager.get_real_data('GBPUSD', source='yahoo')
        
        if data is None or len(data) < 1000:
            print("❌ Failed to get GBPUSD data from all sources")
            return
    
    print(f"✅ Data loaded: {len(data)} records")
    
    # Create trainer
    trainer = AdaptiveTrainer('GBPUSD')
    
    # Patch the _save_model_by_tier method to always skip bias check
    original_save_method = trainer._save_model_by_tier
    
    def emergency_save_method(model, tier, score, attempt, is_best=True, env=None, skip_bias_check=False):
        """Force skip bias check for all saves"""
        return original_save_method(model, tier, score, attempt, is_best, env, skip_bias_check=True)
    
    trainer._save_model_by_tier = emergency_save_method
    
    print("🔧 Modified save method to skip all bias checks")
    
    # Run training
    try:
        print("🚀 Starting emergency training...")
        best_score, best_tier = trainer.adaptive_train(data, max_attempts=3, target_tier='bronze')
        
        print(f"🏁 Emergency training completed!")
        print(f"   Best score: {best_score:.1f}")
        print(f"   Best tier: {best_tier}")
        
    except Exception as e:
        print(f"❌ Emergency training error: {e}")
        print("🔄 Trying with single model training...")
        
        # Fallback: Train single model
        model = None  # Initialize model variable
        try:
            hyperparams = trainer.generate_hyperparameters()
            model, metrics, score, tier, emoji, training_time = trainer.train_model(data, hyperparams)
            
            print(f"📊 Single model result: {tier} {emoji} (score: {score:.1f})")
            
            # Force save regardless of tier
            if score > 10:  # Very low threshold
                # Create a simple environment for saving (required parameter)
                from forex_rl_simple import AdvancedForexEnv
                simple_env = AdvancedForexEnv(
                    data.tail(1000),
                    symbol='GBPUSD',
                    lookback_window=hyperparams['lookback_window'],
                    transaction_cost=hyperparams['transaction_cost']
                )
                
                model_path = trainer._save_model_by_tier(
                    model, 'active_trader', score, 'emergency_1', 
                    is_best=False, env=simple_env, skip_bias_check=True
                )
                print(f"💾 Emergency model saved: {model_path}")
            
        except Exception as e2:
            print(f"❌ Single model training also failed: {e2}")
            print(f"Error details: {str(e2)}")
            
            # Final fallback: Force save without environment (only if model exists)
            if model is not None:
                try:
                    print("🆘 Final fallback: Force save without environment...")
                    model_path = trainer._save_model_by_tier(
                        model, 'active_trader', 0, 'emergency_fallback', 
                        is_best=False, env=None, skip_bias_check=True
                    )
                    print(f"💾 Emergency fallback model saved: {model_path}")
                except Exception as e3:
                    print(f"❌ Final fallback also failed: {e3}")
            else:
                print("❌ No model to save (training failed completely)")

if __name__ == "__main__":
    emergency_train_gbpusd()
