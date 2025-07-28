#!/usr/bin/env python3
"""Test MT5 Trading Bot Market Data"""

import sys
sys.path.append('.')

try:
    from mt5_trading_bot import TradingBot
    import MetaTrader5 as mt5
    
    print("🧪 Testing MT5 Trading Bot Market Data")
    print("=" * 45)
    
    # Test with EURUSD (should auto-detect EURUSDm)
    symbol = "EURUSD"
    print(f"🎯 Testing symbol: {symbol}")
    
    # Create trading bot
    bot = TradingBot(symbol, risk_percent=1.0)
    
    # Connect to MT5
    print("\n🔌 Connecting to MT5...")
    if bot.connect_mt5():
        print("✅ Connected to MT5")
        
        # Test getting market data
        print(f"\n📊 Testing market data for {symbol}...")
        
        # Test the get_rates method directly
        df = bot.mt5.get_rates(symbol, mt5.TIMEFRAME_M5, 50)
        
        if df is not None:
            print(f"✅ Market data retrieved successfully!")
            print(f"   Records: {len(df)}")
            print(f"   Latest close: {df['close'].iloc[-1]:.5f}")
            print(f"   Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            
            # Test single iteration
            print(f"\n🔄 Testing single trading iteration...")
            result = bot.run_single_iteration()
            
            if result is not None:
                print("✅ Single iteration completed successfully!")
            else:
                print("⚠️ Single iteration returned None (might be normal)")
                
        else:
            print("❌ Failed to get market data")
            
        # Disconnect
        print("\n🔌 Disconnecting from MT5...")
        mt5.shutdown()
        print("✅ Disconnected")
        
    else:
        print("❌ Failed to connect to MT5")
        print("💡 Make sure MT5 is running and credentials are correct")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
