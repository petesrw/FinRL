#!/usr/bin/env python3
"""
Simple test to check if environment can be created without errors
"""
import sys
import os
sys.path.append('.')

try:
    print("🔍 Testing environment import...")
    from train_all_models import AdvancedForexEnv
    print("✅ Import successful")
    
    print("🔍 Testing minimal data creation...")
    import pandas as pd
    import numpy as np
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        'open': [100.0] * 100,
        'high': [101.0] * 100,
        'low': [99.0] * 100,
        'close': [100.0] * 100,
        'volume': [1000] * 100
    })
    print("✅ Test data created")
    
    print("🔍 Testing environment creation...")
    env = AdvancedForexEnv(test_data)
    print("✅ Environment created")
    
    print("🔍 Testing attributes...")
    print(f"   position_closes: {hasattr(env, 'position_closes')} = {getattr(env, 'position_closes', 'NOT_FOUND')}")
    print(f"   persistent_total_trades: {hasattr(env, 'persistent_total_trades')} = {getattr(env, 'persistent_total_trades', 'NOT_FOUND')}")
    print(f"   persistent_profitable_trades: {hasattr(env, 'persistent_profitable_trades')} = {getattr(env, 'persistent_profitable_trades', 'NOT_FOUND')}")
    
    print("🔍 Testing reset...")
    obs, info = env.reset()
    print("✅ Reset successful")
    
    print("🔍 Testing step...")
    obs, reward, done, truncated, info = env.step([0.0])
    print("✅ Step successful")
    
    print("🎉 ALL TESTS PASSED!")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
