#!/usr/bin/env python3
"""
🚀 Automatic Indicator Selection Demo
Demonstrates all three optimization methods: smart_defaults, adaptive, meta_learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import time

# Import our automatic indicator system
from indicator_manager import create_indicator_manager, OptimizationMethod
from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_forex_data(symbol: str = "EURUSD", days: int = 365) -> pd.DataFrame:
    """Generate realistic forex sample data for testing"""
    
    # Base price for different currency pairs
    base_prices = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.3000,
        "USDJPY": 110.00,
        "AUDUSD": 0.7500,
        "USDCHF": 0.9200,
        "USDCAD": 1.2500,
        "NZDUSD": 0.7000,
        "EURGBP": 0.8500,
        "EURJPY": 121.00,
        "GBPJPY": 143.00
    }
    
    base_price = base_prices.get(symbol, 1.1000)
    
    # Generate time series
    periods = days * 96  # 15-minute bars
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=periods, freq='15T')
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # Different volatility for different pairs
    volatility = {
        "EURUSD": 0.0008, "GBPUSD": 0.0012, "USDJPY": 0.008,
        "AUDUSD": 0.0010, "USDCHF": 0.0007, "USDCAD": 0.0009,
        "NZDUSD": 0.0011, "EURGBP": 0.0006, "EURJPY": 0.010,
        "GBPJPY": 0.012
    }.get(symbol, 0.0008)
    
    # Generate returns with some trend and mean reversion
    returns = []
    trend = 0.0
    
    for i in range(periods):
        # Add some trend persistence
        if i > 0:
            trend = 0.95 * trend + 0.05 * np.random.normal(0, volatility)
        else:
            trend = np.random.normal(0, volatility)
        
        # Add noise
        noise = np.random.normal(0, volatility)
        
        # Combine trend and noise
        daily_return = trend + noise
        returns.append(daily_return)
    
    # Convert to prices
    returns = np.array(returns)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility_factor = np.random.uniform(0.5, 1.5)
        spread = volatility * volatility_factor * price
        
        high = price + np.random.uniform(0, spread)
        low = price - np.random.uniform(0, spread)
        
        # Ensure OHLC consistency
        if i == 0:
            open_price = price
        else:
            open_price = data[-1]['close']
        
        # Adjust high and low to include open and close
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        data.append({
            'time': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)

def demo_smart_defaults():
    """🎯 Demo: Smart Defaults Method"""
    print("\n" + "="*60)
    print("🎯 DEMO: Smart Defaults Method")
    print("="*60)
    
    # Test different currency pairs
    test_pairs = ["EURUSD", "USDJPY", "AUDUSD", "GBPUSD", "USDTRY"]
    
    for symbol in test_pairs:
        print(f"\n📊 Testing {symbol}:")
        
        # Create manager with smart defaults
        manager = create_indicator_manager(
            symbol=symbol,
            optimization_method="smart_defaults",
            auto_select=True
        )
        
        # Get configuration
        config = manager.get_optimal_config()
        
        print(f"  RSI Period: {config.rsi_period}")
        print(f"  MACD: ({config.macd_fast}, {config.macd_slow})")
        print(f"  SMA: ({config.sma_fast}, {config.sma_slow})")
        print(f"  Bollinger: {config.bb_period} periods, {config.bb_std} std")
        print(f"  ATR Period: {config.atr_period}")
        
        # Show reasoning
        if "JPY" in symbol:
            print("  💡 Yen pair detected - using shorter periods for faster signals")
        elif symbol in ["AUDUSD", "USDCAD", "NZDUSD"]:
            print("  💡 Commodity currency detected - using medium periods")
        elif "TRY" in symbol or "ZAR" in symbol:
            print("  💡 Exotic pair detected - using shorter periods for volatility")
        else:
            print("  💡 Major pair detected - using standard periods")

def demo_adaptive_optimization():
    """🔧 Demo: Adaptive Optimization Method"""
    print("\n" + "="*60)
    print("🔧 DEMO: Adaptive Optimization Method")
    print("="*60)
    
    symbol = "EURUSD"
    print(f"📊 Testing adaptive optimization for {symbol}")
    
    # Generate sample data
    print("📈 Generating sample market data...")
    market_data = generate_sample_forex_data(symbol, days=30)
    
    # Create adaptive manager
    manager = create_indicator_manager(
        symbol=symbol,
        optimization_method="adaptive",
        auto_select=True
    )
    
    print(f"\n🎯 Initial Configuration:")
    initial_config = manager.get_optimal_config()
    print(f"  RSI: {initial_config.rsi_period}")
    print(f"  MACD: ({initial_config.macd_fast}, {initial_config.macd_slow})")
    print(f"  SMA: ({initial_config.sma_fast}, {initial_config.sma_slow})")
    
    # Simulate trading performance over time
    print(f"\n🔄 Simulating adaptive optimization...")
    
    performance_scenarios = [
        (0.45, "Low performance - triggering optimization"),
        (0.70, "Good performance - keeping configuration"),
        (0.40, "Poor performance - major optimization needed"),
        (0.68, "Target performance reached")
    ]
    
    for i, (performance, description) in enumerate(performance_scenarios):
        print(f"\n📊 Scenario {i+1}: {description}")
        print(f"   Performance Score: {performance:.1%}")
        
        # Get optimized configuration
        optimized_config = manager.get_optimal_config(market_data, performance)
        
        print(f"   Optimized RSI: {optimized_config.rsi_period}")
        print(f"   Optimized MACD: ({optimized_config.macd_fast}, {optimized_config.macd_slow})")
        print(f"   Optimized SMA: ({optimized_config.sma_fast}, {optimized_config.sma_slow})")
        
        if performance < manager.performance_threshold:
            print("   🔧 Configuration adjusted due to poor performance")
        else:
            print("   ✅ Configuration maintained - performance acceptable")

def demo_meta_learning():
    """🧠 Demo: Meta-Learning Method"""
    print("\n" + "="*60)
    print("🧠 DEMO: Meta-Learning Method")
    print("="*60)
    
    symbol = "EURUSD"
    print(f"🤖 Testing meta-learning optimization for {symbol}")
    
    # Generate sample data
    market_data = generate_sample_forex_data(symbol, days=60)
    
    # Create meta-learning manager
    manager = create_indicator_manager(
        symbol=symbol,
        optimization_method="meta_learning",
        auto_select=True
    )
    
    print(f"\n🎯 Initial Meta-Learning Configuration:")
    initial_config = manager.get_optimal_config(market_data)
    print(f"  RSI: {initial_config.rsi_period}")
    print(f"  MACD: ({initial_config.macd_fast}, {initial_config.macd_slow})")
    print(f"  Bollinger: {initial_config.bb_period}")
    
    # Simulate learning process
    print(f"\n🧠 Simulating meta-learning process...")
    
    # Simulate indicator performance feedback
    indicator_performance = {
        "rsi": 0.72,      # Good performance
        "macd": 0.58,     # Below average
        "bollinger": 0.81, # Excellent performance
        "sma": 0.65,      # Average performance
        "ema": 0.69,      # Good performance
        "stochastic": 0.45, # Poor performance
        "williams_r": 0.52, # Below average
        "cci": 0.78       # Very good performance
    }
    
    print(f"📊 Indicator Performance Feedback:")
    for indicator, performance in indicator_performance.items():
        status = "🟢 Excellent" if performance > 0.75 else "🟡 Good" if performance > 0.65 else "🟠 Average" if performance > 0.55 else "🔴 Poor"
        print(f"  {indicator.upper()}: {performance:.1%} {status}")
    
    # Update meta-learning weights
    market_conditions = manager.calculate_market_conditions(market_data)
    manager.meta_learning_update(indicator_performance, market_conditions)
    
    print(f"\n🔄 Updated Selection Weights:")
    for indicator, weight in list(manager.selection_weights.items())[:8]:
        print(f"  {indicator.upper()}: {weight:.3f}")
    
    # Get updated configuration
    updated_config = manager.get_optimal_config(market_data)
    print(f"\n🎯 Updated Meta-Learning Configuration:")
    print(f"  RSI: {updated_config.rsi_period}")
    print(f"  MACD: ({updated_config.macd_fast}, {updated_config.macd_slow})")
    print(f"  Bollinger: {updated_config.bb_period}")
    
    print(f"\n💡 Meta-learning insights:")
    print(f"  - Bollinger Bands performing best (weight increased)")
    print(f"  - CCI showing strong performance (weight increased)")
    print(f"  - Stochastic underperforming (weight decreased)")
    print(f"  - System learning which indicators work best for {symbol}")

def demo_comparison():
    """⚖️ Demo: Compare All Three Methods"""
    print("\n" + "="*60)
    print("⚖️ DEMO: Method Comparison")
    print("="*60)
    
    symbol = "EURUSD"
    market_data = generate_sample_forex_data(symbol, days=30)
    
    methods = ["smart_defaults", "adaptive", "meta_learning"]
    configs = {}
    
    print(f"📊 Comparing all methods for {symbol}:\n")
    
    for method in methods:
        manager = create_indicator_manager(
            symbol=symbol,
            optimization_method=method,
            auto_select=True
        )
        
        if method == "adaptive":
            config = manager.get_optimal_config(market_data, 0.55)  # Moderate performance
        else:
            config = manager.get_optimal_config(market_data)
        
        configs[method] = config
        
        print(f"🎯 {method.upper().replace('_', ' ')}:")
        print(f"  RSI Period: {config.rsi_period}")
        print(f"  MACD: ({config.macd_fast}, {config.macd_slow})")
        print(f"  SMA: ({config.sma_fast}, {config.sma_slow})")
        print(f"  Bollinger: {config.bb_period}")
        print(f"  ATR: {config.atr_period}")
        print()
    
    # Show recommendations
    print("💡 RECOMMENDATIONS:")
    print("  🔰 Beginners: Use 'smart_defaults' - proven configurations")
    print("  🔧 Intermediate: Use 'adaptive' - automatic optimization")
    print("  🧠 Advanced: Use 'meta_learning' - AI-driven selection")
    print("  🎯 Best Practice: Start with 'adaptive' for best balance")

def demo_real_world_usage():
    """🌍 Demo: Real-World Usage Examples"""
    print("\n" + "="*60)
    print("🌍 DEMO: Real-World Usage Examples")
    print("="*60)
    
    print("📋 Configuration Examples for Different User Types:\n")
    
    # Beginner configuration
    print("🔰 BEGINNER TRADER:")
    print("   .env configuration:")
    print("   AUTO_SELECT_INDICATORS=true")
    print("   INDICATOR_OPTIMIZATION_METHOD=smart_defaults")
    print("   ALLOW_MANUAL_OVERRIDE=false")
    print("   💡 System handles everything automatically\n")
    
    # Intermediate configuration
    print("🔧 INTERMEDIATE TRADER:")
    print("   .env configuration:")
    print("   AUTO_SELECT_INDICATORS=true")
    print("   INDICATOR_OPTIMIZATION_METHOD=adaptive")
    print("   ALLOW_MANUAL_OVERRIDE=true")
    print("   AUTO_PERFORMANCE_THRESHOLD=0.65")
    print("   # FORCE_RSI_PERIOD=21  # Optional manual override")
    print("   💡 Automatic optimization with manual control\n")
    
    # Advanced configuration
    print("🧠 ADVANCED TRADER:")
    print("   .env configuration:")
    print("   AUTO_SELECT_INDICATORS=true")
    print("   INDICATOR_OPTIMIZATION_METHOD=meta_learning")
    print("   AUTO_OPTIMIZATION_PERIOD=500")
    print("   AUTO_PERFORMANCE_THRESHOLD=0.70")
    print("   ENABLE_INDICATOR_DISCOVERY=true")
    print("   💡 AI learns optimal indicator combinations\n")
    
    # Multi-symbol configuration
    print("🌐 MULTI-SYMBOL TRADER:")
    print("   Each symbol gets its own optimized configuration:")
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    for symbol in symbols:
        manager = create_indicator_manager(symbol, "smart_defaults")
        config = manager.get_optimal_config()
        print(f"   {symbol}: RSI={config.rsi_period}, MACD=({config.macd_fast},{config.macd_slow})")
    
    print("   💡 Different pairs, different optimal settings")

def demo_performance_monitoring():
    """📈 Demo: Performance Monitoring and Optimization"""
    print("\n" + "="*60)
    print("📈 DEMO: Performance Monitoring")
    print("="*60)
    
    symbol = "EURUSD"
    manager = create_indicator_manager(symbol, "adaptive")
    
    # Simulate performance tracking
    print("📊 Simulating performance tracking over time:\n")
    
    # Generate sample performance data
    np.random.seed(42)
    performance_data = []
    
    for week in range(1, 13):  # 12 weeks
        # Simulate varying performance
        base_performance = 0.60 + 0.15 * np.sin(week * 0.5)  # Cyclical performance
        noise = np.random.normal(0, 0.05)
        performance = np.clip(base_performance + noise, 0.3, 0.9)
        
        performance_data.append(performance)
        
        # Check if reoptimization is needed
        if performance < manager.performance_threshold:
            status = "🔧 REOPTIMIZING"
            action = "Adjusting indicator parameters"
        else:
            status = "✅ MAINTAINING"
            action = "Configuration performing well"
        
        print(f"Week {week:2d}: {performance:.1%} - {status} - {action}")
    
    # Show summary
    avg_performance = np.mean(performance_data)
    print(f"\n📊 Summary:")
    print(f"   Average Performance: {avg_performance:.1%}")
    print(f"   Target Performance: {manager.performance_threshold:.1%}")
    print(f"   Reoptimizations: {sum(1 for p in performance_data if p < manager.performance_threshold)}")
    print(f"   Status: {'🎯 Target Met' if avg_performance >= manager.performance_threshold else '🔧 Needs Optimization'}")

def main():
    """🚀 Main demo function"""
    print("🚀 AUTOMATIC INDICATOR SELECTION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases all three optimization methods:")
    print("1. 🎯 Smart Defaults - Currency-specific proven configurations")
    print("2. 🔧 Adaptive - Performance-based automatic optimization")
    print("3. 🧠 Meta-Learning - AI learns optimal indicator combinations")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_smart_defaults()
        demo_adaptive_optimization()
        demo_meta_learning()
        demo_comparison()
        demo_real_world_usage()
        demo_performance_monitoring()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("💡 Next Steps:")
        print("1. Copy .env.example to .env")
        print("2. Set AUTO_SELECT_INDICATORS=true")
        print("3. Choose your optimization method:")
        print("   - smart_defaults (recommended for beginners)")
        print("   - adaptive (recommended for most users)")
        print("   - meta_learning (for advanced users)")
        print("4. Run your forex trading system!")
        print("\n🚀 Happy Trading! 📈")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()