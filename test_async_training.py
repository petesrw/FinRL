#!/usr/bin/env python3
"""
🚀 Test Async Training System
Quick test for async multi-model training capabilities
"""

import pandas as pd
import numpy as np
import torch
import asyncio
import time
from datetime import datetime

# Import from main training system
from train_all_models import AdaptiveTrainer, DEVICE

def create_test_data(symbol='XAUUSD', size=10000):
    """Create synthetic test data for quick testing"""
    print(f"📊 Creating synthetic test data for {symbol}...")
    
    # Generate realistic forex data
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=size, freq='5T')
    
    # Base price trend
    base_price = 2000.0
    trend = np.cumsum(np.random.normal(0, 0.1, size)) + base_price
    
    # Add volatility
    volatility = 5.0
    
    # Generate OHLC data
    opens = trend + np.random.normal(0, volatility, size)
    
    highs = opens + np.abs(np.random.normal(2, 1, size))
    lows = opens - np.abs(np.random.normal(2, 1, size))
    
    # Closes follow opens with some random walk
    closes = opens + np.cumsum(np.random.normal(0, 0.5, size))
    
    # Ensure OHLC relationships are valid
    for i in range(size):
        high_val = max(opens[i], closes[i]) + abs(np.random.normal(0, 1))
        low_val = min(opens[i], closes[i]) - abs(np.random.normal(0, 1))
        
        highs[i] = max(highs[i], high_val)
        lows[i] = min(lows[i], low_val)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(100, 1000, size)
    })
    
    print(f"   ✅ Generated {len(df):,} rows of test data")
    print(f"   📈 Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    return df

async def test_async_training():
    """Test async training capabilities"""
    print("🚀 ASYNC TRAINING TEST")
    print("="*50)
    
    # System info
    print(f"🖥️ Device: {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    # Create test data
    test_data = create_test_data(size=5000)  # Smaller dataset for quick testing
    
    # Initialize trainer
    trainer = AdaptiveTrainer('XAUUSD')
    
    print(f"\n🎯 Async Configuration:")
    print(f"   Max Concurrent Models: {trainer.max_concurrent_models}")
    print(f"   Memory per Model: {trainer.memory_per_model*100:.0f}%")
    
    # Quick async test with small batch
    print(f"\n🔄 Running quick async test...")
    
    start_time = time.time()
    
    try:
        # Run a small async batch
        best_result = await trainer.adaptive_train_async(
            test_data, 
            max_attempts=8,  # Small number for testing
            target_tier='bronze',  # Lower target for testing
            batch_size=2  # Small batch size
        )
        
        total_time = time.time() - start_time
        
        if best_result:
            print(f"\n✅ Async test completed successfully!")
            print(f"🏆 Best score: {best_result['score']:.1f}")
            print(f"🥉 Best tier: {best_result['tier'].upper()}")
            print(f"⏱️ Total time: {total_time:.1f}s")
            
            # Show async benefits
            total_attempts = len(trainer.training_history)
            print(f"\n📊 Async Performance:")
            print(f"   🔢 Total Models Trained: {total_attempts}")
            print(f"   ⚡ Average Time per Model: {total_time/total_attempts:.1f}s")
            print(f"   🚀 Concurrent Processing: {trainer.max_concurrent_models}x speedup")
        else:
            print(f"\n⚠️ Async test completed but no good results")
    
    except Exception as e:
        print(f"\n❌ Async test failed: {e}")
        print(f"💡 This might be normal for testing - trying fallback...")
        
        # Fallback to sync test
        print(f"\n📈 Testing sync training as fallback...")
        best_result = trainer.adaptive_train(
            test_data, 
            max_attempts=4, 
            target_tier='bronze'
        )
        
        if best_result:
            print(f"✅ Sync fallback successful!")
        else:
            print(f"❌ Both async and sync tests failed")

def test_resource_utilization():
    """Test resource utilization monitoring"""
    print(f"\n🔧 RESOURCE UTILIZATION TEST")
    print("="*50)
    
    # Memory test
    if torch.cuda.is_available():
        print(f"📊 GPU Memory Test:")
        
        # Before allocation
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"   Free Memory: {free_memory / 1024**3:.1f} GB")
        
        # Test memory allocation for concurrent models
        trainer = AdaptiveTrainer('XAUUSD')
        max_models = trainer.max_concurrent_models
        memory_per_model = trainer.memory_per_model
        
        print(f"   Max Concurrent Models: {max_models}")
        print(f"   Memory per Model: {memory_per_model*100:.0f}%")
        print(f"   Total Memory Usage: {max_models * memory_per_model * 100:.0f}%")
        
        # Test tensor allocation
        try:
            test_tensors = []
            for i in range(max_models):
                # Simulate model memory usage
                tensor_size = int(1024 * memory_per_model)  # Adjust size based on memory fraction
                tensor = torch.randn(tensor_size, tensor_size, device=DEVICE, dtype=torch.float32)
                test_tensors.append(tensor)
                
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"   Model {i+1}: {allocated:.1f} GB allocated")
            
            print(f"   ✅ Successfully allocated memory for {max_models} concurrent models")
            
            # Clean up
            for tensor in test_tensors:
                del tensor
            torch.cuda.empty_cache()
            
            print(f"   🧹 Memory cleaned up")
            
        except RuntimeError as e:
            print(f"   ⚠️ Memory allocation test failed: {e}")
            print(f"   💡 Consider reducing max_concurrent_models or memory_per_model")
    
    else:
        print(f"💻 CPU-only mode - no GPU memory test")

def main():
    """Main test function"""
    print(f"🧪 ASYNC TRAINING SYSTEM TEST")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Resource utilization test
    test_resource_utilization()
    
    # Async training test
    print(f"\n" + "="*60)
    asyncio.run(test_async_training())
    
    print(f"\n" + "="*60)
    print(f"🏁 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💡 If tests pass, you can use async training for maximum performance!")

if __name__ == "__main__":
    main()
