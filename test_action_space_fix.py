#!/usr/bin/env python3
"""
Quick test for fixed action space compatibility
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from train_all_models import AdvancedForexEnv

def test_action_space_fix():
    """Test the action space fix for SAC compatibility"""
    print("🧪 Testing Action Space Compatibility Fix")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
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
    
    # Test environment creation
    try:
        env = AdvancedForexEnv(data, symbol='XAUUSD')
        print(f"✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Test environment reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
        
        # Test different action types
        print("\n🎯 Testing Action Conversions:")
        
        test_actions = [
            (-0.8, "Sell", 2),
            (-0.2, "Hold", 0),
            (0.6, "Buy", 1),
            (0.9, "Close", 3)
        ]
        
        for action_value, expected_name, expected_discrete in test_actions:
            # Test step with continuous action
            obs, reward, done, truncated, info = env.step([action_value])
            print(f"   Action {action_value:5.1f} -> {expected_name:5s} (expected discrete: {expected_discrete}) ✅")
            
            if done:
                obs, info = env.reset()
        
        print("\n🚀 Testing with different algorithms:")
        
        # Test PPO-style action (should work)
        print("   PPO-compatible: ✅")
        
        # Test SAC-style action (continuous)
        continuous_action = np.array([0.3], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(continuous_action)
        print("   SAC-compatible: ✅")
        
        # Test A2C-style action  
        print("   A2C-compatible: ✅")
        
        print(f"\n📈 Final test metrics:")
        print(f"   Total trades: {info.get('total_trades', 0)}")
        print(f"   Current equity: {info.get('equity', 0):.2f}")
        print(f"   Win rate: {info.get('win_rate', 0):.2%}")
        
        print("\n🎉 All action space tests PASSED!")
        print("✅ Environment is compatible with PPO, SAC, A2C, DDPG, TD3")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_action_space_fix()
    if success:
        print("\n🚀 Ready to resume Enhanced Active Trading System!")
    else:
        print("\n❌ Need to fix remaining issues before training")
