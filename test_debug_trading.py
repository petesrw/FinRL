#!/usr/bin/env python3
"""
Test debug trading execution
"""

from forex_system_with_config import ConfigurableForexBot
import time

def test_debug_trading():
    print('🔍 Testing enhanced logging...')
    
    try:
        bot = ConfigurableForexBot(symbol='EURUSD')
        print('✅ Bot initialized')
        
        print('🚀 Starting single trading step...')
        bot._live_trading_step()
        print('✅ Test completed.')
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_trading()
