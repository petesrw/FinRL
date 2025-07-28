#!/usr/bin/env python3
"""
Test action mapping
"""
import pandas as pd
import numpy as np

# Create minimal test data
test_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=60, freq='h'),
    'open': [1.1000] * 60,
    'high': [1.1005] * 60,
    'low': [1.0995] * 60, 
    'close': [1.1000] * 60,
    'volume': [5000] * 60
})

from train_all_models import AdvancedForexEnv

env = AdvancedForexEnv(test_data, lookback_window=10)
env.reset()

print('🧪 ACTION MAPPING TEST:')
print('='*40)

# Test different action values
test_actions = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for action_val in test_actions:
    print(f'\nTesting action: {action_val}')
    try:
        obs, reward, done, truncated, info = env.step([action_val])
        print(f'  Result: Position={env.position}, Trades={env.total_trades}')
    except Exception as e:
        print(f'  Error: {e}')
    
    # Reset for next test
    env.reset()

print('\n🎯 ACTION MAPPING TEST COMPLETE!')
