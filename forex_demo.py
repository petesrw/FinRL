#!/usr/bin/env python3
"""
Forex Trading System Demo
Quick test of the forex auto trading system
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required libraries are available"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        print("✅ NumPy")
    except ImportError:
        print("❌ NumPy - pip install numpy")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas")
    except ImportError:
        print("❌ Pandas - pip install pandas")
        return False
    
    try:
        import gymnasium as gym
        print("✅ Gymnasium")
    except ImportError:
        print("❌ Gymnasium - pip install gymnasium")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✅ Stable Baselines3")
    except ImportError:
        print("❌ Stable Baselines3 - pip install stable-baselines3[extra]")
        return False
    
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5")
    except ImportError:
        print("❌ MetaTrader5 - pip install MetaTrader5")
        print("   Also install MT5 application from https://www.metatrader5.com/")
        return False
    
    # TA-lib is optional but recommended
    try:
        import talib
        print("✅ TA-lib")
    except ImportError:
        print("⚠️ TA-lib (optional) - pip install TA-lib")
        print("   System will work without it but with fewer indicators")
    
    return True

def test_mt5_connection():
    """Test MT5 connection"""
    print("\n🔗 Testing MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        
        # Try to initialize MT5
        if mt5.initialize():
            print("✅ MT5 initialized successfully")
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"   Terminal: {terminal_info.name}")
                print(f"   Version: {terminal_info.build}")
                print(f"   Path: {terminal_info.path}")
            
            # Test symbol availability
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            print(f"\n📊 Testing symbol availability:")
            
            for symbol in symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    print(f"   ✅ {symbol}: Spread={symbol_info.spread} points")
                else:
                    print(f"   ❌ {symbol}: Not available")
            
            mt5.shutdown()
            return True
        else:
            print("❌ MT5 initialization failed")
            print("   Make sure MT5 is installed and running")
            print("   Enable 'Allow automated trading' in MT5 settings")
            return False
            
    except Exception as e:
        print(f"❌ MT5 connection error: {e}")
        return False

def demo_environment():
    """Demo the trading environment"""
    print("\n🏋️ Testing Trading Environment...")
    
    try:
        from forex_trading_system import ForexTradingEnvironment
        
        # Create environment
        env = ForexTradingEnvironment(
            symbol="EURUSD",
            lookback_window=50,  # Smaller for demo
            initial_balance=10000.0
        )
        
        print("✅ Environment created")
        
        # Reset environment
        obs, info = env.reset()
        print(f"✅ Environment reset - Observation shape: {obs.shape}")
        
        # Test a few steps
        print("\n📈 Testing trading steps:")
        for i in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            
            action_names = ["Hold", "Buy", "Sell", "Close"]
            print(f"   Step {i+1}: Action={action_names[action]}, Reward={reward:.4f}")
            
            if done or truncated:
                break
        
        print(f"✅ Environment test completed")
        print(f"   Balance: ${env.current_balance:.2f}")
        print(f"   Total trades: {env.total_trades}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def demo_training():
    """Demo model training (quick version)"""
    print("\n🤖 Testing Model Training...")
    
    try:
        from forex_trading_system import ForexTradingBot
        
        # Create bot with minimal settings for demo
        bot = ForexTradingBot(
            symbol="EURUSD",
            model_type="PPO",
            risk_per_trade=0.02,
            target_win_rate=0.65
        )
        
        print("✅ Trading bot created")
        
        # Quick training (very short for demo)
        print("🏋️ Starting quick training (this may take 2-5 minutes)...")
        bot.train_model(total_timesteps=5000)  # Very short for demo
        
        print("✅ Training completed")
        
        # Quick test
        print("🧪 Testing trained model...")
        bot.test_model(episodes=3)  # Just 3 episodes for demo
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def main():
    """Main demo function"""
    print("🚀 Forex Auto Trading System Demo")
    print("=" * 50)
    
    # Test 1: Check imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages:")
        print("pip install -r forex_requirements.txt")
        return
    
    # Test 2: MT5 connection
    mt5_ok = test_mt5_connection()
    if not mt5_ok:
        print("\n⚠️ MT5 connection failed, but we can continue with demo data")
    
    # Test 3: Trading environment
    if not demo_environment():
        print("\n❌ Environment test failed")
        return
    
    # Test 4: Model training (optional - takes time)
    print(f"\n🤔 Do you want to test model training? (takes 2-5 minutes)")
    response = input("Enter 'y' for yes, any other key to skip: ").lower().strip()
    
    if response == 'y':
        if not demo_training():
            print("\n❌ Training test failed")
            return
    else:
        print("⏭️ Skipping training test")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\n📋 Summary:")
    print("✅ All imports working")
    if mt5_ok:
        print("✅ MT5 connection working")
    else:
        print("⚠️ MT5 connection needs setup")
    print("✅ Trading environment working")
    print("✅ System ready for use")
    
    print("\n🎯 Next steps:")
    print("1. Setup MT5 with demo account")
    print("2. Run full training: python forex_trading_system.py")
    print("3. Read FOREX_TRADING_GUIDE.md for complete setup")
    print("4. Start with demo trading before going live")
    
    print("\n⚠️ Remember:")
    print("- Always start with demo account")
    print("- Never risk money you can't afford to lose")
    print("- Monitor the system constantly")
    print("- Forex trading involves significant risk")

if __name__ == "__main__":
    main()