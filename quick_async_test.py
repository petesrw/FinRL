#!/usr/bin/env python3
"""
Quick training test with batch_size fix
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import asyncio
from train_all_models import AdaptiveTrainer

async def quick_async_test():
    """Quick test for async training with fixed batch_size"""
    print("🚀 Quick Async Training Test with Batch Size Fix")
    print("=" * 60)
    
    try:
        # Create minimal test data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')  # Smaller dataset
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(len(dates)).cumsum() + 2000,
            'high': np.random.randn(len(dates)).cumsum() + 2005,
            'low': np.random.randn(len(dates)).cumsum() + 1995,
            'close': np.random.randn(len(dates)).cumsum() + 2000,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Fix price relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        print(f"📊 Test data: {len(data)} rows")
        
        # Create trainer
        trainer = AdaptiveTrainer(symbol='TEST')
        print(f"🎯 Trainer created with {trainer.max_concurrent_models} concurrent models")
        
        # Test async training with very small batch (1 model)
        print(f"\n🔄 Starting micro async training test...")
        
        # Generate test hyperparameters
        hyperparams_batch = []
        for i in range(1):  # Just 1 model for quick test
            hyperparams = trainer.generate_hyperparameters()
            # Reduce timesteps for faster test
            hyperparams['timesteps'] = 10000  # Much smaller for testing
            hyperparams_batch.append(hyperparams)
            print(f"   Model {i+1}: {hyperparams['algorithm']} - batch_size: {hyperparams.get('batch_size', 'ERROR')}")
        
        # Run micro async training
        try:
            successful_results, failed_results = await trainer.train_models_async_batch(data, hyperparams_batch)
            
            print(f"\n📊 Micro Test Results:")
            print(f"   ✅ Successful: {len(successful_results)}")
            print(f"   ❌ Failed: {len(failed_results)}")
            
            # Display results
            for i, result in enumerate(successful_results):
                print(f"   🏆 Model {result['model_id']}: Score {result['score']:.1f}, Trades: {result['metrics']['total_trades']}")
            
            for i, result in enumerate(failed_results):
                error_msg = result.get('error', 'Unknown error')[:50]
                print(f"   💥 Failed Model: {error_msg}...")
            
            if len(successful_results) > 0:
                print(f"\n🎉 BATCH SIZE FIX SUCCESSFUL!")
                print(f"✅ At least one model trained without batch_size errors")
                return True
            else:
                print(f"\n⚠️ All models failed - need to investigate further")
                return False
                
        except Exception as training_error:
            print(f"❌ Async training failed: {training_error}")
            return False
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_async_test())
    if success:
        print("\n🚀 Enhanced Active Trading System is ready!")
        print("💡 Batch size parameter issue resolved")
    else:
        print("\n❌ Still need to fix remaining issues")
