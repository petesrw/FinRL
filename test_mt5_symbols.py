#!/usr/bin/env python3
"""Test MT5 symbol detection and market data"""

import MetaTrader5 as mt5
import sys

def test_mt5_symbols():
    """Test MT5 symbol detection"""
    print("🧪 Testing MT5 Symbol Detection")
    print("=" * 40)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    print("✅ MT5 initialized")
    
    # Test symbol variants
    test_symbol = "EURUSD"
    symbol_variants = [
        test_symbol,
        f"{test_symbol}.m",
        f"{test_symbol}m", 
        f"{test_symbol}.c",
        f"{test_symbol}c",
        f"{test_symbol}.",
        test_symbol.replace("USD", "usd"),
        test_symbol.lower()
    ]
    
    print(f"\n🔍 Testing symbol variants for {test_symbol}:")
    working_symbols = []
    
    for variant in symbol_variants:
        symbol_info = mt5.symbol_info(variant)
        if symbol_info is not None:
            print(f"✅ {variant} - Available")
            working_symbols.append(variant)
            
            # Try to get market data
            rates = mt5.copy_rates_from_pos(variant, mt5.TIMEFRAME_M5, 0, 10)
            if rates is not None and len(rates) > 0:
                print(f"   📊 Market data: {len(rates)} records")
            else:
                print(f"   ❌ No market data available")
        else:
            print(f"❌ {variant} - Not found")
    
    if working_symbols:
        print(f"\n✅ Found {len(working_symbols)} working symbol format(s)")
        best_symbol = working_symbols[0]
        
        # Test getting current tick
        tick = mt5.symbol_info_tick(best_symbol)
        if tick:
            print(f"💰 Current {best_symbol}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
        else:
            print(f"⚠️ Cannot get current tick for {best_symbol}")
    else:
        print(f"\n❌ No working symbol formats found for {test_symbol}")
        
        # List available symbols for debugging
        print("\n🔍 Available symbols containing 'EUR' or 'USD':")
        symbols = mt5.symbols_get()
        if symbols:
            forex_symbols = [s.name for s in symbols 
                           if 'EUR' in s.name or 'USD' in s.name][:20]
            for symbol in forex_symbols:
                print(f"   📈 {symbol}")
        else:
            print("   ❌ No symbols available")
    
    # Check market status
    print(f"\n📊 Market Status:")
    if working_symbols:
        symbol_info = mt5.symbol_info(working_symbols[0])
        if symbol_info:
            print(f"   Trading Mode: {symbol_info.trade_mode}")
            print(f"   Session: {symbol_info.session_deals} deals")
            print(f"   Spread: {symbol_info.spread} points")
    
    mt5.shutdown()
    print("\n🔌 MT5 shutdown")

if __name__ == "__main__":
    test_mt5_symbols()
