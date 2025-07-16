#!/usr/bin/env python3
"""
FinRL Quick Start - Minimal Working Example
This is the simplest possible example to get you started with FinRL
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def quick_demo():
    """A minimal demo showing FinRL data processing"""
    print("🚀 FinRL Quick Start Demo")
    print("=" * 40)
    
    try:
        # Test 1: Import FinRL components
        print("📦 Testing FinRL imports...")
        from finrl.meta.data_processor import DataProcessor
        print("✅ DataProcessor imported successfully")
        
        # Test 2: Download some data
        print("\n📊 Downloading sample data...")
        dp = DataProcessor(data_source="yahoofinance")
        
        # Download just one stock for a short period
        data = dp.download_data(
            ticker_list=["AAPL"], 
            start_date="2023-01-01", 
            end_date="2023-12-31", 
            time_interval="1D"
        )
        
        print(f"✅ Downloaded {len(data)} days of AAPL data")
        
        # Test 3: Clean the data
        print("\n🧹 Cleaning data...")
        clean_data = dp.clean_data(data)
        print(f"✅ Cleaned data shape: {clean_data.shape}")
        
        # Test 4: Add simple technical indicators (without TA-lib)
        print("\n📈 Adding technical indicators...")
        try:
            # These indicators don't require TA-lib
            indicators = ["close_30_sma", "close_60_sma"]
            final_data = dp.add_technical_indicator(clean_data, indicators)
            print(f"✅ Added technical indicators")
            print(f"📊 Final data shape: {final_data.shape}")
            print(f"📋 Columns: {list(final_data.columns)}")
        except Exception as e:
            print(f"⚠️ Technical indicators failed: {e}")
            final_data = clean_data
        
        # Test 5: Show sample data
        print("\n📋 Sample of the data:")
        print(final_data.head())
        
        # Test 6: Basic statistics
        print(f"\n📊 Price Statistics for AAPL:")
        print(f"💰 Average Close Price: ${final_data['close'].mean():.2f}")
        print(f"📈 Highest Price: ${final_data['high'].max():.2f}")
        print(f"📉 Lowest Price: ${final_data['low'].min():.2f}")
        print(f"📊 Total Volume: {final_data['volume'].sum():,}")
        
        print("\n🎉 Quick demo completed successfully!")
        print("💡 You're ready to explore more FinRL features!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have internet connection for data download")
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n" + "=" * 50)
    print("🎯 What to do next:")
    print("1. Run: python simple_trading_example.py")
    print("2. Read: GETTING_STARTED_GUIDE.md")
    print("3. Try: python finrl/main.py --mode=train")
    print("4. Explore: examples/ folder")
    print("=" * 50)

if __name__ == "__main__":
    success = quick_demo()
    if success:
        show_next_steps()