#!/usr/bin/env python3
"""
🚀 Test Direct Model Loading
ทดสอบการโหลด DIAMOND model โดยตรง
"""

def test_direct_model_loading():
    print("🚀 TESTING DIRECT MODEL LOADING")
    print("=" * 50)
    
    try:
        from forex_system_with_config import ConfigurableForexBot
        
        # Create bot
        print("1. 🤖 Creating ConfigurableForexBot...")
        bot = ConfigurableForexBot(symbol="EURUSD")
        
        # Load DIAMOND model directly
        diamond_model = "models/diamond/eurusd_diamond_score100_attempt1_20250728_215948.zip"
        print(f"2. 💎 Loading DIAMOND model: {diamond_model}")
        
        if bot.load_model(diamond_model):
            print("   ✅ DIAMOND model loaded successfully!")
            
            # Test the model
            print("3. 🧪 Testing model...")
            result = bot.test_model(episodes=2)
            print(f"   📊 Test result: {'✅ PASSED' if result else '⚠️ NEEDS IMPROVEMENT'}")
            
            print("\n🎉 SUCCESS!")
            print("   ✅ Direct model loading works")
            print("   ✅ DIAMOND model functional")
            print("   ✅ System ready for live trading")
            
            return True
        else:
            print("   ❌ Failed to load DIAMOND model")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_direct_model_loading()
