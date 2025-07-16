#!/usr/bin/env python3
"""
Simple XAUUSD Training Script
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def simple_train():
    """
    Simple training function
    """
    print("🔄 Simple XAUUSD Training...")
    print("="*40)
    
    # Check if data exists
    data_file = "train_data/XAUUSD/XAUUSD_M5_real.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Load data
    print(f"📊 Loading data...")
    df = pd.read_csv(data_file)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Basic data validation
    print(f"\n📈 Data Summary:")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"   Average price: {df['close'].mean():.2f}")
    
    # Calculate basic indicators
    print(f"\n🔧 Calculating indicators...")
    
    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_rsi(df['close'])
    
    # Remove NaN values
    df = df.dropna()
    print(f"   After indicator calculation: {len(df):,} rows")
    
    # Simple trading simulation
    print(f"\n🧪 Running simple trading simulation...")
    
    balance = 10000
    position = 0
    trades = []
    
    for i in range(100, len(df)):
        current_price = df.iloc[i]['close']
        sma_20 = df.iloc[i]['sma_20']
        sma_50 = df.iloc[i]['sma_50']
        rsi = df.iloc[i]['rsi']
        
        # Simple strategy: Buy when SMA20 > SMA50 and RSI < 70
        if position == 0:  # No position
            if sma_20 > sma_50 and rsi < 70:
                position = 1
                entry_price = current_price
                trades.append({
                    'type': 'BUY',
                    'price': entry_price,
                    'timestamp': df.iloc[i]['timestamp']
                })
        
        elif position == 1:  # Long position
            if sma_20 < sma_50 or rsi > 80:
                position = 0
                exit_price = current_price
                profit = exit_price - entry_price
                balance += profit * 100  # Assume 100 units
                trades.append({
                    'type': 'SELL',
                    'price': exit_price,
                    'profit': profit,
                    'timestamp': df.iloc[i]['timestamp']
                })
    
    # Results
    print(f"\n📊 Simulation Results:")
    print(f"   Initial balance: $10,000")
    print(f"   Final balance: ${balance:,.2f}")
    print(f"   Total return: {((balance - 10000) / 10000) * 100:.2f}%")
    print(f"   Total trades: {len([t for t in trades if t['type'] == 'SELL'])}")
    
    if len(trades) > 0:
        profitable_trades = len([t for t in trades if t['type'] == 'SELL' and t.get('profit', 0) > 0])
        total_trades = len([t for t in trades if t['type'] == 'SELL'])
        if total_trades > 0:
            win_rate = (profitable_trades / total_trades) * 100
            print(f"   Win rate: {win_rate:.1f}%")
    
    print(f"\n✅ Simple training simulation completed!")
    
    # Save results
    results_file = f"xauusd_simple_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"XAUUSD Simple Training Results\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Data rows: {len(df):,}\n")
        f.write(f"Final balance: ${balance:,.2f}\n")
        f.write(f"Total return: {((balance - 10000) / 10000) * 100:.2f}%\n")
        f.write(f"Total trades: {len([t for t in trades if t['type'] == 'SELL'])}\n")
    
    print(f"📄 Results saved to: {results_file}")

if __name__ == "__main__":
    simple_train()