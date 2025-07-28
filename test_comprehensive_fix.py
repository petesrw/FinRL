#!/usr/bin/env python3
"""
Comprehensive test to verify all fixes work correctly
This will test the critical issues found in the training logs
"""
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

print("🔍 COMPREHENSIVE FIX TESTING")
print("=" * 50)

# Create realistic FOREX test data
np.random.seed(42)  # For reproducible results
test_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h'),
    'open': 1.1000 + np.random.randn(1000).cumsum() * 0.0001,
    'high': 1.1000 + np.random.randn(1000).cumsum() * 0.0001 + 0.0005,
    'low': 1.1000 + np.random.randn(1000).cumsum() * 0.0001 - 0.0005,
    'close': 1.1000 + np.random.randn(1000).cumsum() * 0.0001,
    'volume': np.random.randint(1000, 10000, 1000)
})

print('✅ Realistic FOREX test data created (EURUSD-like)')
print(f"   Price range: {test_data['close'].min():.4f} - {test_data['close'].max():.4f}")

# Import the environment
try:
    from train_all_models import AdvancedForexEnv
    print('✅ Environment imported successfully')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)

# Create environment
env = AdvancedForexEnv(test_data)
print('✅ Environment created successfully')

# Check SL/TP percentages
print(f"\n📊 Risk Management Settings:")
print(f"   Stop Loss: {env.stop_loss_pct:.1%}")
print(f"   Take Profit: {env.take_profit_pct:.1%}")

# Test 1: Initial state
print(f"\n🧪 TEST 1: Initial State")
print(f"   position_closes: {env.position_closes}")
print(f"   total_trades: {env.total_trades}")
print(f"   profitable_trades: {env.profitable_trades}")
print(f"   balance: {env.balance}")
print(f"   equity: {env.equity}")

# Test 2: Reset behavior
env.reset()
print(f"\n🧪 TEST 2: Reset Behavior")
print(f"   position_closes after reset: {env.position_closes}")
print(f"   total_trades after reset: {env.total_trades}")
print(f"   balance after reset: {env.balance}")

# Test 3: Simulate trading sequence
print(f"\n🧪 TEST 3: Trading Sequence Simulation")
print("Testing Buy -> Wait -> Force Close sequence...")

# Force a Buy action
obs, reward, done, truncated, info = env.step([0.8])  # Strong buy signal
print(f"   After Buy: position={env.position}, balance={env.balance:.2f}")

# Wait a few steps
for i in range(10):
    obs, reward, done, truncated, info = env.step([0.0])  # Hold

# Force close
obs, reward, done, truncated, info = env.step([0.9])  # Strong close signal
print(f"   After Close: position={env.position}, balance={env.balance:.2f}")
print(f"   Total trades: {env.total_trades}")
print(f"   Position closes: {env.position_closes}")

# Test 4: SL/TP trigger testing
print(f"\n🧪 TEST 4: SL/TP Logic Testing")

# Simulate many trades to see if SL/TP triggers
trade_count = 0
sl_count = 0
tp_count = 0

for step in range(100):
    # Randomly take buy/sell actions
    if env.position == 0:
        action = [0.8] if np.random.random() > 0.5 else [-0.8]  # Buy or Sell
    else:
        action = [0.0]  # Hold when we have position
    
    obs, reward, done, truncated, info = env.step(action)
    
    # Check if we completed a trade
    if len(env.trades) > trade_count:
        last_trade = env.trades[-1]
        trade_count = len(env.trades)
        close_reason = last_trade.get('close_reason', 'Unknown')
        profit = last_trade.get('profit', 0)
        
        print(f"   Trade #{trade_count}: {close_reason}, Profit: {profit:.6f}")
        
        if close_reason == "Stop Loss":
            sl_count += 1
        elif close_reason == "Take Profit":
            tp_count += 1
    
    if done:
        break

print(f"\n📈 FINAL RESULTS:")
print(f"   Total Trades: {len(env.trades)}")
print(f"   Stop Loss hits: {sl_count}")
print(f"   Take Profit hits: {tp_count}")
print(f"   Final Balance: {env.balance:.6f}")
print(f"   Final Equity: {env.equity:.6f}")
print(f"   Balance Change: {env.balance - env.initial_balance:.6f}")

# Test performance metrics
metrics = env._get_performance_metrics()
print(f"\n📊 PERFORMANCE METRICS:")
print(f"   Win Rate: {metrics['win_rate']:.1%}")
print(f"   SL Hit Rate: {metrics['sl_hit_rate']:.1%}")
print(f"   TP Hit Rate: {metrics['tp_hit_rate']:.1%}")
print(f"   Total Trades: {metrics['total_trades']}")
print(f"   Profitable Trades: {metrics['profitable_trades']}")

# Verify no persistent counter issues
if hasattr(env, 'persistent_total_trades'):
    print(f"   ⚠️  WARNING: persistent_total_trades still exists: {env.persistent_total_trades}")
else:
    print(f"   ✅ persistent_total_trades removed successfully")

# Create a detailed trade log
if env.trades:
    print(f"\n📋 TRADE LOG (Last 5 trades):")
    for i, trade in enumerate(env.trades[-5:]):
        profit_pct = trade['profit_pct'] * 100
        print(f"   Trade {len(env.trades)-4+i}: {trade['close_reason']} | "
              f"Profit: {trade['profit']:.6f} ({profit_pct:.2f}%) | "
              f"Entry: {trade['entry_price']:.4f} | Exit: {trade['exit_price']:.4f}")

print(f"\n🎯 DIAGNOSIS:")
if metrics['tp_hit_rate'] == 0 and sl_count == 0 and tp_count == 0:
    print("   ❌ CRITICAL: SL/TP logic still broken - no automatic triggers detected")
elif metrics['tp_hit_rate'] == 0 and tp_count > 0:
    print("   ❌ CRITICAL: TP hit rate calculation broken")
elif abs(env.balance - env.initial_balance) < 0.01:
    print("   ❌ CRITICAL: Balance barely changes despite multiple trades")
else:
    print("   ✅ All major issues appear to be resolved!")

print("\n" + "=" * 50)
print("🏁 COMPREHENSIVE TEST COMPLETED")
