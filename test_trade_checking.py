#!/usr/bin/env python3
"""
Test script for checking existing MT5 trades and position handling
"""

import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_mt5_connection():
    """Test basic MT5 connection"""
    try:
        import MetaTrader5 as mt5  # type: ignore
        
        print("🔌 Testing MT5 connection...")
        
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Connected to MT5:")
            print(f"   💰 Account: {account_info.login}")
            print(f"   💳 Balance: ${account_info.balance:.2f}")
            print(f"   📊 Equity: ${account_info.equity:.2f}")
            print(f"   🏢 Server: {account_info.server}")
        
        return True
    except ImportError:
        print("❌ MetaTrader5 module not installed")
        return False
    except Exception as e:
        print(f"❌ MT5 connection error: {e}")
        return False

def test_existing_positions():
    """Test checking existing positions"""
    try:
        import MetaTrader5 as mt5  # type: ignore
        
        print("\n🔍 Checking existing positions...")
        
        # Check positions for common forex pairs
        symbols_to_check = ["EURUSD", "EURUSDm", "GBPUSD", "GBPUSDm", "USDJPY", "USDJPYm"]
        
        total_positions = 0
        for symbol in symbols_to_check:
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                count = len(positions)
                total_positions += count
                print(f"📊 {symbol}: {count} position(s)")
                
                for i, pos in enumerate(positions, 1):
                    position_type = "BUY" if pos.type == 0 else "SELL"
                    print(f"   Position #{i}: {position_type}")
                    print(f"      📍 Entry: {pos.price_open:.5f}")
                    print(f"      💰 P&L: ${pos.profit:.2f}")
                    print(f"      📏 Volume: {pos.volume}")
                    print(f"      🕐 Time: {pos.time}")
        
        if total_positions == 0:
            print("📊 No existing positions found")
        else:
            print(f"\n📊 Total positions found: {total_positions}")
        
        return total_positions
    except Exception as e:
        print(f"❌ Error checking positions: {e}")
        return 0

def test_recent_deals():
    """Test checking recent deals/trades"""
    try:
        import MetaTrader5 as mt5  # type: ignore
        from datetime import datetime, timedelta
        
        print("\n📈 Checking recent deals (last 24 hours)...")
        
        # Get deals from last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        deals = mt5.history_deals_get(start_time, end_time)
        if deals:
            print(f"📊 Found {len(deals)} deals in last 24 hours")
            
            # Filter trading deals
            trade_deals = [deal for deal in deals if deal.type in [0, 1]]  # Buy/Sell deals
            
            if trade_deals:
                print(f"📊 Trading deals: {len(trade_deals)}")
                
                for deal in trade_deals[-5:]:  # Show last 5 deals
                    deal_type = "BUY" if deal.type == 0 else "SELL"
                    print(f"   {deal_type} {deal.symbol}: ${deal.profit:.2f} @ {deal.price:.5f}")
            else:
                print("📊 No trading deals found")
        else:
            print("📊 No deals found in last 24 hours")
    
    except Exception as e:
        print(f"❌ Error checking deals: {e}")

def main():
    """Main test function"""
    print("🧪 MT5 Trade Checking Test")
    print("=" * 50)
    
    # Test 1: Basic connection
    if not test_mt5_connection():
        print("❌ Cannot proceed without MT5 connection")
        return
    
    # Test 2: Check existing positions
    position_count = test_existing_positions()
    
    # Test 3: Check recent deals
    test_recent_deals()
    
    print("\n" + "=" * 50)
    print("✅ Test completed")
    
    if position_count > 0:
        print(f"⚠️ WARNING: {position_count} existing position(s) detected!")
        print("   The trading system should handle these automatically.")
    else:
        print("ℹ️ No existing positions - system will start fresh")

if __name__ == "__main__":
    main()
