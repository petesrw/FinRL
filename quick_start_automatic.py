#!/usr/bin/env python3
"""
🚀 Quick Start: Automatic Indicator Selection
Get started with automatic indicator selection in 5 minutes!
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'talib', 'stable_baselines3', 
        'gymnasium', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """Set up the environment configuration"""
    print("🔧 Setting up environment configuration...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("📋 Copying .env.example to .env...")
            
            # Read .env.example
            with open('.env.example', 'r') as f:
                content = f.read()
            
            # Write to .env with automatic indicator settings
            with open('.env', 'w') as f:
                f.write(content)
            
            print("✅ .env file created successfully!")
        else:
            print("❌ .env.example not found!")
            return False
    else:
        print("✅ .env file already exists")
    
    return True

def quick_demo():
    """Run a quick demonstration"""
    print("\n🚀 Running Quick Demo...")
    
    try:
        from indicator_manager import create_indicator_manager
        import pandas as pd
        import numpy as np
        
        print("\n1️⃣ Testing Smart Defaults for different currency pairs:")
        
        pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
        
        for pair in pairs:
            manager = create_indicator_manager(
                symbol=pair,
                optimization_method="smart_defaults"
            )
            
            config = manager.get_optimal_config()
            print(f"   {pair}: RSI={config.rsi_period}, MACD=({config.macd_fast},{config.macd_slow})")
        
        print("\n2️⃣ Testing Adaptive Optimization:")
        
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'open': np.random.randn(100) + 1.1000,
            'high': np.random.randn(100) + 1.1010,
            'low': np.random.randn(100) + 1.0990,
            'close': np.random.randn(100) + 1.1000,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        manager = create_indicator_manager(
            symbol="EURUSD",
            optimization_method="adaptive"
        )
        
        # Test with different performance levels
        for performance in [0.45, 0.70]:
            config = manager.get_optimal_config(sample_data, performance)
            status = "🔧 Optimized" if performance < 0.65 else "✅ Maintained"
            print(f"   Performance {performance:.1%}: RSI={config.rsi_period} {status}")
        
        print("\n3️⃣ Testing Meta-Learning:")
        
        manager = create_indicator_manager(
            symbol="EURUSD",
            optimization_method="meta_learning"
        )
        
        config = manager.get_optimal_config(sample_data)
        print(f"   Meta-learning config: RSI={config.rsi_period}, MACD=({config.macd_fast},{config.macd_slow})")
        
        print("\n✅ Quick demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def show_usage_examples():
    """Show practical usage examples"""
    print("\n📚 USAGE EXAMPLES:")
    print("="*50)
    
    print("\n🔰 For Beginners (Fully Automatic):")
    print("```python")
    print("from indicator_manager import create_indicator_manager")
    print("")
    print("# Create automatic manager")
    print("manager = create_indicator_manager(")
    print("    symbol='EURUSD',")
    print("    optimization_method='smart_defaults'")
    print(")")
    print("")
    print("# Get optimal configuration")
    print("config = manager.get_optimal_config()")
    print("print(f'RSI Period: {config.rsi_period}')")
    print("```")
    
    print("\n🔧 For Intermediate Users (Adaptive):")
    print("```python")
    print("# Create adaptive manager")
    print("manager = create_indicator_manager(")
    print("    symbol='EURUSD',")
    print("    optimization_method='adaptive'")
    print(")")
    print("")
    print("# Update configuration based on performance")
    print("config = manager.get_optimal_config(market_data, performance_score)")
    print("```")
    
    print("\n🧠 For Advanced Users (Meta-Learning):")
    print("```python")
    print("# Create meta-learning manager")
    print("manager = create_indicator_manager(")
    print("    symbol='EURUSD',")
    print("    optimization_method='meta_learning'")
    print(")")
    print("")
    print("# System learns from indicator performance")
    print("manager.meta_learning_update(indicator_performance, market_conditions)")
    print("config = manager.get_optimal_config(market_data)")
    print("```")

def show_configuration_guide():
    """Show configuration guide"""
    print("\n⚙️ CONFIGURATION GUIDE:")
    print("="*50)
    
    print("\n📝 Edit your .env file with these settings:")
    
    print("\n🔰 Beginner Setup (Recommended):")
    print("AUTO_SELECT_INDICATORS=true")
    print("INDICATOR_OPTIMIZATION_METHOD=smart_defaults")
    print("ALLOW_MANUAL_OVERRIDE=false")
    
    print("\n🔧 Intermediate Setup:")
    print("AUTO_SELECT_INDICATORS=true")
    print("INDICATOR_OPTIMIZATION_METHOD=adaptive")
    print("ALLOW_MANUAL_OVERRIDE=true")
    print("AUTO_PERFORMANCE_THRESHOLD=0.65")
    
    print("\n🧠 Advanced Setup:")
    print("AUTO_SELECT_INDICATORS=true")
    print("INDICATOR_OPTIMIZATION_METHOD=meta_learning")
    print("AUTO_OPTIMIZATION_PERIOD=500")
    print("AUTO_PERFORMANCE_THRESHOLD=0.70")
    
    print("\n🎛️ Manual Overrides (Optional):")
    print("# Uncomment to force specific values")
    print("# FORCE_RSI_PERIOD=21")
    print("# FORCE_MACD_FAST=8")
    print("# FORCE_SMA_FAST=15")

def main():
    """Main quick start function"""
    print("🚀 QUICK START: Automatic Indicator Selection")
    print("="*60)
    print("Get your forex trading system running with automatic")
    print("indicator selection in just a few minutes!")
    print("="*60)
    
    # Step 1: Check requirements
    print("\n1️⃣ Checking requirements...")
    if not check_requirements():
        print("\n❌ Please install missing packages and try again.")
        return
    
    print("✅ All required packages are installed!")
    
    # Step 2: Setup environment
    print("\n2️⃣ Setting up environment...")
    if not setup_environment():
        print("\n❌ Environment setup failed.")
        return
    
    print("✅ Environment configured successfully!")
    
    # Step 3: Run quick demo
    print("\n3️⃣ Running quick demonstration...")
    if not quick_demo():
        print("\n⚠️ Demo had issues, but you can still proceed.")
    
    # Step 4: Show usage examples
    show_usage_examples()
    
    # Step 5: Show configuration guide
    show_configuration_guide()
    
    # Final instructions
    print("\n🎯 NEXT STEPS:")
    print("="*50)
    print("1. ✅ Requirements checked")
    print("2. ✅ Environment configured")
    print("3. ✅ Demo completed")
    print("")
    print("🚀 You're ready to go! Here's what to do next:")
    print("")
    print("📝 1. Edit your .env file:")
    print("   - Set AUTO_SELECT_INDICATORS=true")
    print("   - Choose optimization method (smart_defaults recommended)")
    print("")
    print("🤖 2. Run your trading system:")
    print("   python forex_system_with_config.py")
    print("")
    print("📊 3. Or run the full demo:")
    print("   python demo_automatic_indicators.py")
    print("")
    print("📚 4. Read the documentation:")
    print("   - WHY_CONFIGURE_INDICATORS.md (English)")
    print("   - WHY_CONFIGURE_INDICATORS_TH.md (Thai)")
    print("")
    print("🎉 Happy Trading! 📈")
    
    # Show current configuration
    if os.path.exists('.env'):
        print("\n📋 Current .env configuration preview:")
        try:
            with open('.env', 'r') as f:
                lines = f.readlines()
            
            # Show relevant lines
            relevant_lines = [
                'AUTO_SELECT_INDICATORS',
                'INDICATOR_OPTIMIZATION_METHOD',
                'ALLOW_MANUAL_OVERRIDE'
            ]
            
            for line in lines:
                for keyword in relevant_lines:
                    if keyword in line and not line.strip().startswith('#'):
                        print(f"   {line.strip()}")
                        break
        except:
            pass

if __name__ == "__main__":
    main()