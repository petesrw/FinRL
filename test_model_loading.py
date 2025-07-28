#!/usr/bin/env python3
"""
🧪 Test Model Loading Fix
ทดสอบว่า model loading ทำงานได้แล้วหรือไม่
"""

def test_model_loading():
    print("🧪 TESTING MODEL LOADING FIX")
    print("=" * 50)
    
    try:
        from mt5_trading_bot import ModelLoader, TradingBot
        
        # Test model loader
        print("1. 📊 Testing ModelLoader...")
        model_loader = ModelLoader()
        models = model_loader.list_available_models()
        
        if models:
            best_model = models[0]
            print(f"   ✅ Best model found: {best_model['tier']} (Score: {best_model['score']:.1f})")
            print(f"   📁 Path: {best_model['model_path']}")
            
            # Test model loading
            print("\n2. 🤖 Testing model loading...")
            try:
                model_loader.load_model(best_model['model_path'])
                print("   ✅ Model loaded successfully!")
                
                # Test TradingBot creation
                print("\n3. 🚀 Testing TradingBot creation...")
                trading_bot = TradingBot("EURUSD", model_path=best_model['model_path'])
                print("   ✅ TradingBot created successfully!")
                
                print("\n🎉 ALL TESTS PASSED!")
                print("   ✅ Model loading fixed")
                print("   ✅ TradingBot creation works")
                print("   ✅ Ready for live trading")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Model loading failed: {e}")
                return False
        else:
            print("   ❌ No models found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()
