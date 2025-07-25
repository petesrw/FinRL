#!/usr/bin/env python3
"""
Quick test to verify position closing logic
"""
import pandas as pd
import numpy as np
import random
from train_all_models import AdvancedForexEnv

# Create dummy data
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='5min'),
    'open': 2000 + np.random.randn(1000) * 0.5,
    'high': 2000 + np.random.randn(1000) * 0.5 + 0.5,
    'low': 2000 + np.random.randn(1000) * 0.5 - 0.5,
    'close': 2000 + np.random.randn(1000) * 0.5,
    'volume': np.random.randint(100, 1000, 1000)
})

print("🧪 Testing Position Closing Logic...")

# Create environment
env = AdvancedForexEnv(data, initial_balance=10000)
obs, info = env.reset()

print(f"Initial state: Position={env.position}, Balance={env.balance:.2f}")

# Test 1: Open a Buy position
print("\n📈 Test 1: Opening Buy position...")
action = np.array([0.3])  # Should be Buy action (0.1 to 0.5 range)
obs, reward, done, truncated, info = env.step(action)
print(f"After Buy: Position={env.position}, Size={env.position_size:.2f}, Entry={env.entry_price:.5f}")
print(f"Total trades: {env.total_trades}, Position opens: {env.position_opens}")

# Test 2: Try to close manually
print("\n🔒 Test 2: Manual close attempt...")
action = np.array([0.8])  # Should be Close action (0.5 to 1.0 range)
obs, reward, done, truncated, info = env.step(action)
print(f"After Close attempt: Position={env.position}, Total trades: {env.total_trades}")
print(f"Position opens: {env.position_opens}, Position closes: {env.position_closes}")

# Test 3: If still have position, force multiple close attempts
if env.position != 0:
    print("\n🔄 Test 3: Multiple close attempts...")
    for i in range(5):
        action = np.array([0.9])  # Strong close signal
        obs, reward, done, truncated, info = env.step(action)
        print(f"Close attempt {i+1}: Position={env.position}, Total trades: {env.total_trades}")
        if env.position == 0:
            print("✅ Position successfully closed!")
            break
    else:
        print("❌ Position still not closed after 5 attempts")

print(f"\n📊 Final Results:")
print(f"Total trades: {env.total_trades}")
print(f"Position opens: {env.position_opens}")
print(f"Position closes: {env.position_closes}")
print(f"Current position: {env.position}")
print(f"Balance: {env.balance:.2f}")
