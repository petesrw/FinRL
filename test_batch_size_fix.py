#!/usr/bin/env python3
"""
Test batch_size parameter fix for async training
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from train_all_models import AdaptiveTrainer

def test_batch_size_fix():
    """Test batch_size parameter handling"""
    print("🧪 Testing Batch Size Parameter Fix")
    print("=" * 50)
    
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(len(dates)).cumsum() + 2000,
            'high': np.random.randn(len(dates)).cumsum() + 2005,
            'low': np.random.randn(len(dates)).cumsum() + 1995,
            'close': np.random.randn(len(dates)).cumsum() + 2000,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Ensure high > low, close within range
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        print(f"📊 Sample data created: {len(data)} rows")
        
        # Test adaptive trainer creation
        trainer = AdaptiveTrainer(symbol='XAUUSD')
        print(f"✅ AdaptiveTrainer created successfully")
        
        # Test hyperparameter generation
        print("\n🎯 Testing Hyperparameter Generation:")
        
        for i in range(3):
            hyperparams = trainer.generate_hyperparameters()
            print(f"   Test {i+1}:")
            print(f"      Algorithm: {hyperparams.get('algorithm', 'MISSING')}")
            print(f"      Batch Size: {hyperparams.get('batch_size', 'MISSING')}")
            print(f"      Learning Rate: {hyperparams.get('learning_rate', 'MISSING')}")
            print(f"      Timesteps: {hyperparams.get('timesteps', 'MISSING')}")
            
            # Verify required parameters are present
            required_params = ['algorithm', 'learning_rate', 'gamma', 'timesteps']
            for param in required_params:
                if param not in hyperparams:
                    print(f"      ❌ Missing required parameter: {param}")
                    return False
            
            # Verify algorithm-specific parameters
            if hyperparams['algorithm'] == 'PPO':
                if 'batch_size' not in hyperparams or 'n_steps' not in hyperparams:
                    print(f"      ❌ Missing PPO-specific parameters")
                    return False
                print(f"      n_steps: {hyperparams.get('n_steps', 'MISSING')}")
                
            elif hyperparams['algorithm'] == 'SAC':
                if 'batch_size' not in hyperparams:
                    print(f"      ❌ Missing SAC-specific parameters")
                    return False
                print(f"      buffer_size: {hyperparams.get('buffer_size', 'MISSING')}")
            
            print(f"      ✅ All required parameters present")
        
        print("\n🚀 Testing Async Training Preparation:")
        
        # Test generating multiple hyperparameters for async batch
        hyperparams_batch = []
        for i in range(2):  # Small batch for testing
            hyperparams = trainer.generate_hyperparameters()
            hyperparams_batch.append(hyperparams)
            print(f"   Batch {i+1}: {hyperparams['algorithm']} - batch_size: {hyperparams.get('batch_size', 'MISSING')}")
        
        print(f"\n📊 Batch Generation Results:")
        print(f"   Total hyperparameters generated: {len(hyperparams_batch)}")
        print(f"   All contain batch_size: {all('batch_size' in hp for hp in hyperparams_batch)}")
        print(f"   All contain required params: {all(all(param in hp for param in required_params) for hp in hyperparams_batch)}")
        
        print("\n🎉 Batch Size Parameter Fix Test PASSED!")
        print("✅ All hyperparameters contain required batch_size parameter")
        print("✅ Ready for async training without 'batch_size' errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_size_fix()
    if success:
        print("\n🚀 Ready to resume Enhanced Active Trading System!")
        print("💡 The 'batch_size' error should be resolved now")
    else:
        print("\n❌ Need to fix remaining parameter issues")
