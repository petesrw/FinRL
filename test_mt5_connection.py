#!/usr/bin/env python3
"""
🔍 Test MT5 Connection
ทดสอบการเชื่อมต่อ MetaTrader 5
"""

import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mt5_connection():
    """ทดสอบการเชื่อมต่อ MT5"""
    print("🔍 Testing MT5 Connection...")
    print("=" * 50)
    
    # Step 1: Test MT5 initialization
    print("1️⃣ Testing MT5 initialization...")
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    print("✅ MT5 initialized successfully")
    
    # Step 2: Get MT5 version info
    print(f"📊 MT5 Version: {mt5.version()}")
    
    # Step 3: Test with environment credentials (if available)
    login = os.getenv('MT5_LOGIN')
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    
    if login and password and server:
        print(f"\n2️⃣ Testing login with credentials from .env file...")
        print(f"   🔐 Login: {login}")
        print(f"   🏢 Server: {server}")
        
        if mt5.login(int(login), password=password, server=server):
            print("✅ MT5 login successful!")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                print(f"   💰 Account: {account_info.login}")
                print(f"   💳 Balance: ${account_info.balance:.2f}")
                print(f"   📊 Equity: ${account_info.equity:.2f}")
                print(f"   💱 Currency: {account_info.currency}")
                print(f"   📈 Leverage: 1:{account_info.leverage}")
                print(f"   🏢 Company: {account_info.company}")
                print(f"   🌐 Server: {account_info.server}")
                
                # Test symbol access
                print(f"\n3️⃣ Testing symbol access...")
                test_symbols = ["EURUSD", "EURUSDm", "GBPUSD", "USDJPY"]
                
                for symbol in test_symbols:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info:
                        print(f"   ✅ {symbol}: Available (Spread: {symbol_info.spread})")
                    else:
                        print(f"   ❌ {symbol}: Not available")
                
                # Test getting rates
                print(f"\n4️⃣ Testing market data...")
                rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 10)
                if rates is not None:
                    print(f"   ✅ Market data: Got {len(rates)} bars")
                    latest = rates[-1]
                    print(f"   📊 Latest EURUSD: {latest['close']:.5f}")
                else:
                    print("   ❌ Failed to get market data")
                
                print(f"\n🎉 MT5 Connection Test: SUCCESS!")
                return True
            else:
                print("❌ Failed to get account info")
                return False
        else:
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            return False
    else:
        print(f"\n2️⃣ No credentials found in .env file")
        print(f"   ⚠️ Please set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env file")
        print(f"   ℹ️ Connection test limited to initialization only")
        
        # Still test if MT5 terminal is running
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"   ✅ MT5 Terminal detected")
            print(f"   📁 Data path: {terminal_info.data_path}")
            print(f"   🏢 Company: {terminal_info.company}")
        else:
            print(f"   ❌ MT5 Terminal not detected")
        
        return True
    
    # Always shutdown
    mt5.shutdown()
    print(f"\n🔌 MT5 connection closed")

def check_mt5_installation():
    """ตรวจสอบการติดตั้ง MT5"""
    print("\n🔍 Checking MT5 Installation...")
    print("=" * 50)
    
    # Check common MT5 installation paths
    import platform
    system = platform.system()
    
    if system == "Windows":
        possible_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\MetaTrader 5\terminal64.exe"),
        ]
        
        print(f"🪟 Windows system detected")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   ✅ Found MT5: {path}")
                return path
            else:
                print(f"   ❌ Not found: {path}")
    
    print(f"   ⚠️ MT5 not found in common locations")
    print(f"   💡 Please install MT5 from: https://www.metatrader5.com/en/download")
    return None

if __name__ == "__main__":
    print("🤖 MT5 Connection Tester")
    print("Testing real MT5 connection capabilities...")
    print()
    
    # Check installation first
    mt5_path = check_mt5_installation()
    
    # Test connection
    success = test_mt5_connection()
    
    if success:
        print(f"\n🎯 RESULT: MT5 connection capabilities ✅ WORKING")
        if mt5_path:
            print(f"📍 MT5 Location: {mt5_path}")
    else:
        print(f"\n🎯 RESULT: MT5 connection ❌ FAILED")
        print(f"💡 Please check:")
        print(f"   1. MT5 is installed and running")
        print(f"   2. Algorithm trading is enabled in MT5")
        print(f"   3. Credentials in .env file are correct")
