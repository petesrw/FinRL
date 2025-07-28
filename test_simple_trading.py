#!/usr/bin/env python3
"""
Simple trading test with appropriate data size
"""
import pandas as pd
import numpy as np
import torch

# Configure PyTorch threading
torch.set_num_threads(1)
print('✅ PyTorch threading configured')

# Create sufficient test data (100 rows)
np.random.seed(42)  # For reproducible results
prices = np.cumsum(np.random.randn(100) * 0.01) + 1.1000  # EURUSD-like price movement

test_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    'open': prices + np.random.randn(100) * 0.0001,
    'high': prices + np.abs(np.random.randn(100)) * 0.0005,
    'low': prices - np.abs(np.random.randn(100)) * 0.0005, 
    'close': prices,
    'volume': np.random.randint(1000, 10000, 100)
})

print(f'✅ Test data created ({len(test_data)} rows)')
print(f'   Price range: {test_data["close"].min():.4f} - {test_data["close"].max():.4f}')

# Import the environment
from train_all_models import AdvancedForexEnv

# Create environment with smaller lookback
env = AdvancedForexEnv(test_data, lookback_window=10, stop_loss_pct=0.01, take_profit_pct=0.02)
print('✅ Environment created with 10 lookback, 1% SL / 2% TP')

# Reset environment
obs = env.reset()
print('✅ Environment reset')
print(f'   Current step: {env.current_step}')
print(f'   Max steps: {env.max_steps}')
print(f'   Data length: {len(env.data)}')

print('\n🧪 SIMPLE TRADING TEST:')
print('='*50)

# Test sequence: Buy -> Hold -> Check -> Close
print('1️⃣ Opening BUY position...')
obs, reward, done, truncated, info = env.step([0.5])  # Buy signal (0.3 <= 0.5 < 0.7)
print(f'   Position: {env.position}')
print(f'   Position Size: {env.position_size:.2f}')
print(f'   Entry Price: {env.entry_price:.4f}')
print(f'   Current Step: {env.current_step}')

if env.position != 0:
    print(f'✅ Position opened successfully!')
    
    # Take a few steps to see price movement
    for i in range(5):
        obs, reward, done, truncated, info = env.step([0.0])  # Hold
        current_price = env.data.iloc[env.current_step-1]['close']
        print(f'   Step {env.current_step}: Price = {current_price:.4f}, Position = {env.position}')
        if done:
            break
    
    print(f'\n📊 FINAL STATE:')
    print(f'   Total Trades: {env.total_trades}')
    print(f'   Position Opens: {env.position_opens}')
    print(f'   Position Closes: {env.position_closes}')
    print(f'   Final Balance: {env.balance:.5f}')
else:
    print('❌ Position failed to open!')

print('\n🎯 SIMPLE TEST COMPLETE!')
