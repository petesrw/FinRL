#!/usr/bin/env python3
"""
Simple Manual SL/TP Test - Force specific scenarios
"""
import sys
import pandas as pd
import numpy as np

# Create simple test data with known price movements
test_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
    'open': [1.1000, 1.1000, 1.0950, 1.0900, 1.0850, 1.0950, 1.1050, 1.1100, 1.1150, 1.1200],
    'high': [1.1010, 1.1010, 1.0960, 1.0910, 1.0860, 1.0960, 1.1060, 1.1110, 1.1160, 1.1210],
    'low':  [1.0990, 1.0990, 1.0940, 1.0890, 1.0840, 1.0940, 1.1040, 1.1090, 1.1140, 1.1190],
    'close': [1.1000, 1.0950, 1.0900, 1.0850, 1.0800, 1.0950, 1.1050, 1.1100, 1.1150, 1.1200],  # -5% then +5%
    'volume': [10000] * 10
})

print('✅ Manual test data created with specific price movements')
print('📊 Price sequence:')
for i, price in enumerate(test_data['close']):
    print(f"   Step {i}: {price:.4f}")

# Import and create environment
from train_all_models import AdvancedForexEnv

# Tight SL/TP for testing: 1% SL, 2% TP
env = AdvancedForexEnv(test_data, stop_loss_pct=0.01, take_profit_pct=0.02)
print('✅ Environment created with 1% SL / 2% TP')

obs = env.reset()
print('✅ Environment reset')

print('\n🧪 MANUAL TRADING TEST:')
print('='*50)

print(f"Step {env.current_step}: Price = {test_data.iloc[env.current_step]['close']:.4f}")

# Force open a BUY position
print('\n1️⃣ OPENING BUY POSITION:')
obs, reward, done, truncated, info = env.step([0.8])  # Strong buy
print(f"   Position: {env.position}")
print(f"   Entry Price: {env.entry_price:.4f}")
print(f"   Position Size: {env.position_size:.2f}")
print(f"   Current Price: {test_data.iloc[env.current_step]['close']:.4f}")

# Step through and check SL/TP
print('\n2️⃣ STEPPING THROUGH PRICE MOVEMENTS:')
for step in range(8):  # Step through remaining data
    if env.current_step >= len(test_data) - 1:
        break
        
    # Get current state before step
    current_price = test_data.iloc[env.current_step]['close']
    if env.position != 0:
        current_return = (current_price - env.entry_price) / env.entry_price * env.position
        print(f"\nBEFORE Step {step}:")
        print(f"   Price: {current_price:.4f}, Entry: {env.entry_price:.4f}")
        print(f"   Return: {current_return:.4f} ({'Profit' if current_return > 0 else 'Loss'})")
        print(f"   SL Check: {current_return:.4f} <= {-env.stop_loss_pct:.4f} ? {current_return <= -env.stop_loss_pct}")
        print(f"   TP Check: {current_return:.4f} >= {env.take_profit_pct:.4f} ? {current_return >= env.take_profit_pct}")
    
    # Take action (hold)
    obs, reward, done, truncated, info = env.step([0.0])
    
    # Check result after step
    new_price = test_data.iloc[env.current_step]['close'] if env.current_step < len(test_data) else current_price
    print(f"AFTER Step {step}:")
    print(f"   New Price: {new_price:.4f}")
    print(f"   Position: {env.position}")
    print(f"   Total Trades: {env.total_trades}")
    
    if env.total_trades > 0:
        print(f"   🎯 TRADE COMPLETED!")
        if env.trades:
            last_trade = env.trades[-1]
            print(f"      Close Reason: {last_trade['close_reason']}")
            print(f"      Profit: {last_trade['profit']:.5f}")
            print(f"      Profit %: {last_trade['profit_pct']:.4f}")
        break

print('\n📊 FINAL RESULTS:')
print(f"Total Trades: {env.total_trades}")
print(f"Position Opens: {env.position_opens}")
print(f"Position Closes: {env.position_closes}")
print(f"Balance: {env.balance:.5f}")

if env.trades:
    for i, trade in enumerate(env.trades):
        print(f"Trade {i+1}: {trade['close_reason']}, P&L: {trade['profit']:.5f}")

print('\n🎯 MANUAL TEST COMPLETE!')
