#!/usr/bin/env python3
"""
🔍 Test MT5 Market Data
ทดสอบการดึงข้อมูลตลาดจาก MT5
"""

import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_market_data():
    """ทดสอบการดึงข้อมูลตลาด"""
    print("📊 Testing MT5 Market Data Access...")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    # Login
    login = int(os.getenv('MT5_LOGIN'))
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    
    if not mt5.login(login, password=password, server=server):
        print(f"❌ MT5 login failed: {mt5.last_error()}")
        return False
    
    print("✅ Connected to MT5")
    
    # Test different symbols
    test_symbols = ["EURUSDm", "EURUSD", "EURUSDecn", "EURUSD."]
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing symbol: {symbol}")
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            print(f"   ✅ Symbol found: {symbol}")
            print(f"   💰 Spread: {symbol_info.spread}")
            print(f"   📊 Digits: {symbol_info.digits}")
            print(f"   📈 Visible: {symbol_info.visible}")
            
            # Try to get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"   💵 Current price: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
                
                # Try to get historical data
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
                if rates is not None:
                    print(f"   📈 Historical data: Got {len(rates)} bars")
                    latest = rates[-1]
                    print(f"   📊 Latest bar: O={latest['open']:.5f}, C={latest['close']:.5f}")
                else:
                    print(f"   ❌ No historical data available")
            else:
                print(f"   ❌ No current price available")
        else:
            print(f"   ❌ Symbol not found: {symbol}")
    
    # Test with M5 timeframe (used in trading)
    print(f"\n📊 Testing M5 timeframe data for EURUSDm...")
    rates_m5 = mt5.copy_rates_from_pos("EURUSDm", mt5.TIMEFRAME_M5, 0, 100)
    if rates_m5 is not None:
        print(f"   ✅ M5 data: Got {len(rates_m5)} bars")
        latest = rates_m5[-1]
        print(f"   📊 Latest M5 bar: {latest['close']:.5f}")
    else:
        print(f"   ❌ No M5 data available")
    
    # List all available symbols
    print(f"\n📝 Listing available symbols...")
    symbols = mt5.symbols_get()
    if symbols:
        eur_symbols = [s.name for s in symbols if "EUR" in s.name and "USD" in s.name]
        print(f"   📈 EUR/USD variants found: {len(eur_symbols)}")
        for sym in eur_symbols[:5]:  # Show first 5
            print(f"      - {sym}")
        if len(eur_symbols) > 5:
            print(f"      ... and {len(eur_symbols) - 5} more")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_market_data()
