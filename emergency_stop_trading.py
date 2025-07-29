#!/usr/bin/env python3
"""
🚨 Emergency Stop Trading Script
ใช้เมื่อตรวจพบ model failure และต้องหยุดการเทรดทันที
"""

import MetaTrader5 as mt5
import time
from datetime import datetime

def emergency_stop_all_trades():
    """หยุดการเทรดและปิด position ทั้งหมดทันที"""
    print("🚨 EMERGENCY STOP INITIATED")
    print("=" * 40)
    
    # เชื่อมต่อ MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    print("✅ Connected to MT5 for emergency stop")
    
    # ดู position ทั้งหมด
    positions = mt5.positions_get()
    if not positions:
        print("📊 No open positions found")
    else:
        print(f"📊 Found {len(positions)} open positions")
        
        # ปิด position ทั้งหมด
        for pos in positions:
            print(f"\n🔄 Closing position {pos.ticket}")
            print(f"   Symbol: {pos.symbol}")
            print(f"   Type: {'BUY' if pos.type == 0 else 'SELL'}")
            print(f"   Volume: {pos.volume}")
            print(f"   P&L: ${pos.profit:.2f}")
            
            # สร้าง order ปิด position
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(pos.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(pos.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "magic": 123456,
                "comment": "EMERGENCY CLOSE - Model Failure",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"   ✅ Position closed successfully")
            else:
                print(f"   ❌ Failed to close: {result.retcode} - {result.comment}")
    
    # ดูผลสรุป
    final_positions = mt5.positions_get()
    if not final_positions:
        print("\n✅ ALL POSITIONS CLOSED SUCCESSFULLY")
    else:
        print(f"\n⚠️ {len(final_positions)} positions still open")
    
    # แสดงข้อมูลบัญชี
    account_info = mt5.account_info()
    if account_info:
        print(f"\n💰 Account Summary:")
        print(f"   Balance: ${account_info.balance:.2f}")
        print(f"   Equity: ${account_info.equity:.2f}")
        print(f"   Margin: ${account_info.margin:.2f}")
        print(f"   Free Margin: ${account_info.margin_free:.2f}")
    
    mt5.shutdown()
    print("\n🔌 Disconnected from MT5")
    
    return True

def create_stop_trading_flag():
    """สร้างไฟล์เพื่อสั่งให้ระบบอื่นหยุดเทรด"""
    with open("STOP_TRADING.flag", "w") as f:
        f.write(f"EMERGENCY_STOP\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Reason: Model failure detected\n")
        f.write(f"Action: All trading systems must stop immediately\n")
    
    print("🚩 Created STOP_TRADING.flag file")

if __name__ == "__main__":
    print("🚨 EMERGENCY TRADING STOP")
    print("=" * 30)
    print("This will:")
    print("1. Close all open positions")
    print("2. Create stop flag for other systems")
    print("3. Prevent further trading")
    print("")
    
    confirm = input("Continue? (yes/y/1/GO): ").lower().strip()
    if confirm in ['yes', 'y', '1', 'go']:
        print("🚀 Starting emergency stop procedure...")
        emergency_stop_all_trades()
        create_stop_trading_flag()
        print("\n🛑 EMERGENCY STOP COMPLETED")
        print("📋 Next steps:")
        print("   1. Stop all trading bots")
        print("   2. Start fresh model retraining")
        print("   3. Use retrain_fresh_model.py")
        print("   4. DO NOT use existing model for retraining")
    else:
        print("❌ Emergency stop cancelled")
