#!/usr/bin/env python3
"""
🧪 Test Real Trading System
ทดสอบว่าระบบเทรดจริงกับ MT5 หรือแค่เทรน model
"""

import os
from datetime import datetime

def test_live_trading_integration():
    """ทดสอบการ integrate ระหว่าง forex_launcher และ MT5 trading"""
    
    print("🧪 TESTING REAL TRADING INTEGRATION")
    print("=" * 60)
    
    # 1. ตรวจสอบว่ามี model ใหม่หรือไม่
    print("\n1. 📊 Checking available models...")
    
    from forex_launcher import ForexLauncher
    launcher = ForexLauncher()
    
    for symbol_key, (symbol, desc) in launcher.symbols.items():
        model_file, tier, score = launcher.find_best_model(symbol)
        if model_file:
            actual_score = score-1000 if score > 1000 else score-100 if score > 100 else score-10 if score > 10 else score-1 if score > 1 else score
            print(f"   {symbol}: {tier} (Score: {actual_score:.1f}) - {model_file}")
        else:
            print(f"   {symbol}: ❌ No model found")
    
    # 2. ตรวจสอบ .env configuration
    print("\n2. ⚙️ Checking .env configuration...")
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            env_content = f.read()
        
        demo_mode = "DEMO_MODE=true" in env_content
        mode_text = "DEMO MODE" if demo_mode else "LIVE MODE"
        print(f"   Trading Mode: {mode_text}")
        
        if "DEFAULT_SYMBOL=" in env_content:
            for line in env_content.split('\n'):
                if line.startswith('DEFAULT_SYMBOL='):
                    symbol = line.split('=')[1]
                    print(f"   Default Symbol: {symbol}")
                    break
    else:
        print("   ❌ .env file not found!")
    
    # 3. ทดสอบ MT5 connection
    print("\n3. 🔌 Testing MT5 connection...")
    try:
        import MetaTrader5 as mt5
        
        if mt5.initialize():
            print("   ✅ MT5 initialize successful")
            
            # Try to get some market data
            eurusd_rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 10)
            if eurusd_rates is not None:
                print(f"   ✅ Market data available (EURUSD: {len(eurusd_rates)} bars)")
                current_price = mt5.symbol_info_tick("EURUSD")
                if current_price:
                    print(f"   💰 Current EURUSD: {current_price.bid:.5f}")
            else:
                print("   ⚠️ Market data not available (may need broker connection)")
            
            mt5.shutdown()
        else:
            print("   ❌ MT5 initialize failed")
            error = mt5.last_error()
            print(f"   Error: {error}")
    except ImportError:
        print("   ❌ MetaTrader5 module not installed")
    except Exception as e:
        print(f"   ❌ MT5 test error: {e}")
    
    # 4. ทดสอบ trading bot integration
    print("\n4. 🤖 Testing trading bot integration...")
    try:
        from mt5_trading_bot import TradingBot, ModelLoader
        
        # Test model loading
        model_loader = ModelLoader()
        models = model_loader.list_available_models()
        
        if models:
            best_model = models[0]
            print(f"   ✅ Best model: {best_model['tier']} (Score: {best_model['score']:.1f})")
            print(f"   📁 Path: {best_model['model_path']}")
            
            # Test trading bot creation (without MT5 connection)
            trading_bot = TradingBot("EURUSD", model_path=best_model['model_path'])
            print("   ✅ Trading bot created successfully")
            
        else:
            print("   ❌ No models found")
            
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
    except Exception as e:
        print(f"   ❌ Trading bot test error: {e}")
    
    # 5. สรุปผล
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print("   🎯 forex_launcher.py: เลือก symbol และ mode")
    print("   ⚙️ forex_system_with_config.py: main() function สำหรับรัน")
    print("   🤖 mt5_trading_bot.py: ตัวจริงที่เทรดกับ MT5")
    print("   📊 models/: มี model DIAMOND tier พร้อมใช้งาน")
    
    print("\n💡 FLOW การทำงาน:")
    print("   1. forex_launcher.py → เลือก Start Live Trading")
    print("   2. อัพเดท .env → กำหนด DEMO_MODE และ DEFAULT_SYMBOL") 
    print("   3. เรียก forex_system_with_config.main()")
    print("   4. main() → สร้าง ConfigurableForexBot")
    print("   5. ConfigurableForexBot → เรียก _live_trading_step()")
    print("   6. _live_trading_step() → ใช้ mt5_trading_bot.TradingBot")
    print("   7. TradingBot → เชื่อมต่อ MT5 และเทรดจริง")
    
    print("\n🚨 สิ่งที่ต้องระวัง:")
    print("   - ถ้า DEMO_MODE=true → จำลองการเทรด")
    print("   - ถ้า DEMO_MODE=false → เทรดด้วยเงินจริง!")
    print("   - ต้องมี MT5 credentials ใน .env")
    print("   - ต้องเปิด MT5 และเปิด Algorithm Trading")


if __name__ == "__main__":
    test_live_trading_integration()
