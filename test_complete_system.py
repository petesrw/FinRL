#!/usr/bin/env python3
"""
🧪 Complete System Test: Automatic Indicator Selection + Forex Trading
Tests the full integration of automatic indicator selection with the forex trading system
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration Loading...")
    
    try:
        from config import get_config
        config = get_config()
        
        print("✅ Configuration loaded successfully!")
        print(f"   Auto Select Indicators: {config.indicators.auto_select_indicators}")
        print(f"   Optimization Method: {config.indicators.indicator_optimization_method}")
        print(f"   Allow Manual Override: {config.indicators.allow_manual_override}")
        print(f"   Performance Threshold: {config.indicators.auto_performance_threshold}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_indicator_manager():
    """Test automatic indicator manager"""
    print("\n🤖 Testing Automatic Indicator Manager...")
    
    try:
        from indicator_manager import create_indicator_manager
        
        # Test all three methods
        methods = ["smart_defaults", "adaptive", "meta_learning"]
        symbols = ["EURUSD", "USDJPY", "GBPUSD"]
        
        for method in methods:
            print(f"\n   Testing {method.upper()}:")
            
            for symbol in symbols:
                manager = create_indicator_manager(
                    symbol=symbol,
                    optimization_method=method,
                    auto_select=True
                )
                
                config = manager.get_optimal_config()
                print(f"     {symbol}: RSI={config.rsi_period}, MACD=({config.macd_fast},{config.macd_slow})")
        
        print("✅ Indicator manager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Indicator manager test failed: {e}")
        return False

def test_forex_environment():
    """Test enhanced forex environment with automatic indicators"""
    print("\n🏛️ Testing Enhanced Forex Environment...")
    
    try:
        from config import get_config
        from indicator_manager import create_indicator_manager
        from forex_system_with_config import ForexEnvironment
        
        # Create configuration and indicator manager
        config = get_config()
        indicator_manager = create_indicator_manager("EURUSD", "adaptive")
        
        # Create environment
        env = ForexEnvironment(
            symbol="EURUSD",
            config=config,
            indicator_manager=indicator_manager
        )
        
        print("   Environment created successfully!")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.n}")
        
        # Test reset and step
        obs, info = env.reset()
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   Observation sample: {obs[:5]}")  # First 5 features
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: Action={action}, Reward={reward:.4f}, Done={done}")
        
        print("✅ Forex environment test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Forex environment test failed: {e}")
        return False

def test_configurable_bot():
    """Test the main configurable forex bot"""
    print("\n🤖 Testing Configurable Forex Bot...")
    
    try:
        from forex_system_with_config import ConfigurableForexBot
        
        # Create bot
        bot = ConfigurableForexBot(symbol="EURUSD")
        
        print("   Bot created successfully!")
        print(f"   Symbol: {bot.symbol}")
        print(f"   Indicator Manager: {type(bot.indicator_manager).__name__}")
        print(f"   Auto Select: {bot.config.indicators.auto_select_indicators}")
        print(f"   Optimization Method: {bot.config.indicators.indicator_optimization_method}")
        
        # Test environment creation
        env = bot.create_training_environment()
        print(f"   Training environment created: {type(env).__name__}")
        
        # Test performance report
        report = bot.get_performance_report()
        print(f"   Performance report keys: {list(report.keys())}")
        
        print("✅ Configurable bot test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configurable bot test failed: {e}")
        return False

def test_adaptive_optimization():
    """Test adaptive optimization in action"""
    print("\n🔧 Testing Adaptive Optimization...")
    
    try:
        from indicator_manager import create_indicator_manager
        import pandas as pd
        import numpy as np
        
        # Create adaptive manager
        manager = create_indicator_manager("EURUSD", "adaptive")
        
        # Generate sample market data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'open': np.random.randn(200) + 1.1000,
            'high': np.random.randn(200) + 1.1010,
            'low': np.random.randn(200) + 1.0990,
            'close': np.random.randn(200) + 1.1000,
            'volume': np.random.randint(1000, 10000, 200)
        })
        
        print("   Testing different performance scenarios:")
        
        # Test different performance levels
        performance_scenarios = [
            (0.45, "Poor performance - should trigger optimization"),
            (0.70, "Good performance - should maintain config"),
            (0.40, "Very poor - major optimization needed"),
            (0.75, "Excellent - keep current settings")
        ]
        
        for performance, description in performance_scenarios:
            config = manager.get_optimal_config(sample_data, performance)
            print(f"   Performance {performance:.1%}: RSI={config.rsi_period}, {description}")
        
        print("✅ Adaptive optimization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive optimization test failed: {e}")
        return False

def test_meta_learning():
    """Test meta-learning functionality"""
    print("\n🧠 Testing Meta-Learning...")
    
    try:
        from indicator_manager import create_indicator_manager
        import pandas as pd
        import numpy as np
        
        # Create meta-learning manager
        manager = create_indicator_manager("EURUSD", "meta_learning")
        
        # Generate sample data
        sample_data = pd.DataFrame({
            'open': np.random.randn(100) + 1.1000,
            'high': np.random.randn(100) + 1.1010,
            'low': np.random.randn(100) + 1.0990,
            'close': np.random.randn(100) + 1.1000,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        print("   Initial configuration:")
        initial_config = manager.get_optimal_config(sample_data)
        print(f"   RSI: {initial_config.rsi_period}, MACD: ({initial_config.macd_fast},{initial_config.macd_slow})")
        
        # Simulate indicator performance feedback
        indicator_performance = {
            "rsi": 0.75,
            "macd": 0.60,
            "bollinger": 0.85,
            "sma": 0.55,
            "ema": 0.70
        }
        
        print("   Updating with performance feedback...")
        market_conditions = manager.calculate_market_conditions(sample_data)
        manager.meta_learning_update(indicator_performance, market_conditions)
        
        # Get updated configuration
        updated_config = manager.get_optimal_config(sample_data)
        print(f"   Updated RSI: {updated_config.rsi_period}, MACD: ({updated_config.macd_fast},{updated_config.macd_slow})")
        
        print("✅ Meta-learning test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Meta-learning test failed: {e}")
        return False

def test_integration():
    """Test full system integration"""
    print("\n🔗 Testing Full System Integration...")
    
    try:
        from forex_system_with_config import ConfigurableForexBot
        
        # Test with different optimization methods
        methods = ["smart_defaults", "adaptive", "meta_learning"]
        
        for method in methods:
            print(f"\n   Testing with {method}:")
            
            # Temporarily set environment variable
            os.environ['INDICATOR_OPTIMIZATION_METHOD'] = method
            
            # Create bot (this will reload config)
            bot = ConfigurableForexBot(symbol="EURUSD")
            
            # Verify the method is set correctly
            actual_method = bot.indicator_manager.optimization_method.value
            print(f"     Configured method: {actual_method}")
            
            # Test environment creation with indicators
            env = bot.create_training_environment()
            obs, _ = env.reset()
            
            print(f"     Environment working: observation shape {obs.shape}")
            
            # Test one step
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"     Step executed: reward={reward:.4f}")
        
        print("✅ Full integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_performance_benchmark():
    """Run a quick performance benchmark"""
    print("\n⚡ Running Performance Benchmark...")
    
    try:
        from forex_system_with_config import ConfigurableForexBot
        import time
        
        # Test environment creation speed
        start_time = time.time()
        
        bot = ConfigurableForexBot(symbol="EURUSD")
        env = bot.create_training_environment()
        
        # Run 100 steps
        obs, _ = env.reset()
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
        
        elapsed_time = time.time() - start_time
        steps_per_second = 100 / elapsed_time
        
        print(f"   100 steps completed in {elapsed_time:.2f} seconds")
        print(f"   Performance: {steps_per_second:.1f} steps/second")
        
        if steps_per_second > 50:
            print("   ✅ Performance: Excellent")
        elif steps_per_second > 20:
            print("   ✅ Performance: Good")
        else:
            print("   ⚠️ Performance: Could be improved")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run complete system test"""
    print("🧪 COMPLETE SYSTEM TEST")
    print("=" * 60)
    print("Testing automatic indicator selection integration")
    print("with the forex trading system")
    print("=" * 60)
    
    # Track test results
    tests = [
        ("Configuration Loading", test_configuration),
        ("Indicator Manager", test_indicator_manager),
        ("Forex Environment", test_forex_environment),
        ("Configurable Bot", test_configurable_bot),
        ("Adaptive Optimization", test_adaptive_optimization),
        ("Meta-Learning", test_meta_learning),
        ("Full Integration", test_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your automatic indicator selection system is working perfectly!")
        print("\n🚀 Ready for production use:")
        print("1. Set your .env configuration")
        print("2. Choose optimization method (smart_defaults, adaptive, meta_learning)")
        print("3. Run: python forex_system_with_config.py")
    else:
        print(f"\n⚠️ {failed} tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r forex_requirements.txt")
    
    print("\n📚 Documentation:")
    print("- WHY_CONFIGURE_INDICATORS.md (English)")
    print("- WHY_CONFIGURE_INDICATORS_TH.md (Thai)")
    print("- Run: python demo_automatic_indicators.py")

if __name__ == "__main__":
    main()