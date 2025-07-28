#!/usr/bin/env python3
"""
Test the position_closes attribute fix
"""
import sys
import pandas as pd
import numpy as np

# Create test data
test_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 101,
    'low': np.random.randn(100).cumsum() + 99,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
})

print('✅ Test data created')

# Import the environment
from train_all_models import AdvancedForexEnv

env = AdvancedForexEnv(test_data)
print('✅ Environment created successfully')

# Test attributes
print(f'   position_closes initialized: {hasattr(env, "position_closes")}')
print(f'   position_closes value: {env.position_closes}')
print(f'   total_trades: {env.total_trades}')
print(f'   profitable_trades: {env.profitable_trades}')

# Test reset
env.reset()
print('✅ Reset successful')
print(f'   position_closes after reset: {env.position_closes}')
print(f'   total_trades after reset: {env.total_trades}')
print(f'   profitable_trades after reset: {env.profitable_trades}')

# Test step
obs, reward, done, truncated, info = env.step([0.5])
print('✅ Step successful')

print('🎉 ALL TESTS PASSED! Ready for training.')
