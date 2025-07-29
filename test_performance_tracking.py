#!/usr/bin/env python3
"""
Test script to verify correct performance tracking
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_system_with_config import ConfigurableForexBot

def test_performance_tracking():
    """Test the new performance tracking logic"""
    print("🧪 Testing Performance Tracking Logic")
    print("=" * 50)
    
    # Create a bot instance
    bot = ConfigurableForexBot("EURUSD")
    
    print(f"📊 Initial Performance Stats:")
    print(f"   📂 Total Opened Trades: {bot.performance_stats['total_trades']}")
    print(f"   ✅ Winning Trades: {bot.performance_stats['winning_trades']}")
    print(f"   ❌ Losing Trades: {bot.performance_stats['losing_trades']}")
    print(f"   🏆 Win Rate: {bot.performance_stats['win_rate']:.1%}")
    print(f"   💰 Total P&L: ${bot.performance_stats['total_profit']:.2f}")
    
    print(f"\n🎯 Simulating Trade Opening...")
    # Simulate opening a trade
    trade_result = {
        'action': 1,  # BUY
        'current_price': 1.15371,
        'confidence': 0.65,
        'result': True
    }
    
    bot._update_trade_performance(trade_result)
    
    print(f"\n📊 After Trade Opening:")
    print(f"   📂 Total Opened Trades: {bot.performance_stats['total_trades']}")
    print(f"   ✅ Winning Trades: {bot.performance_stats['winning_trades']}")
    print(f"   ❌ Losing Trades: {bot.performance_stats['losing_trades']}")
    total_closed = bot.performance_stats['winning_trades'] + bot.performance_stats['losing_trades']
    print(f"   🔢 Total Closed Trades: {total_closed}")
    print(f"   🏆 Win Rate: {bot.performance_stats['win_rate']:.1%}")
    print(f"   💰 Total P&L: ${bot.performance_stats['total_profit']:.2f}")
    
    print(f"\n✅ Simulating Trade Closure (PROFIT)...")
    # Simulate closing the trade with profit
    bot._update_closed_trade_performance(25.50)
    
    print(f"\n📊 After Trade Closure (Profit):")
    print(f"   📂 Total Opened Trades: {bot.performance_stats['total_trades']}")
    print(f"   ✅ Winning Trades: {bot.performance_stats['winning_trades']}")
    print(f"   ❌ Losing Trades: {bot.performance_stats['losing_trades']}")
    total_closed = bot.performance_stats['winning_trades'] + bot.performance_stats['losing_trades']
    print(f"   🔢 Total Closed Trades: {total_closed}")
    print(f"   🏆 Win Rate: {bot.performance_stats['win_rate']:.1%}")
    print(f"   💰 Total P&L: ${bot.performance_stats['total_profit']:.2f}")
    
    print(f"\n🎯 Simulating Another Trade Opening...")
    # Simulate opening another trade
    trade_result2 = {
        'action': 2,  # SELL
        'current_price': 1.15250,
        'confidence': 0.55,
        'result': True
    }
    
    bot._update_trade_performance(trade_result2)
    
    print(f"\n📊 After Second Trade Opening:")
    print(f"   📂 Total Opened Trades: {bot.performance_stats['total_trades']}")
    print(f"   ✅ Winning Trades: {bot.performance_stats['winning_trades']}")
    print(f"   ❌ Losing Trades: {bot.performance_stats['losing_trades']}")
    total_closed = bot.performance_stats['winning_trades'] + bot.performance_stats['losing_trades']
    print(f"   🔢 Total Closed Trades: {total_closed}")
    print(f"   🏆 Win Rate: {bot.performance_stats['win_rate']:.1%}")
    print(f"   💰 Total P&L: ${bot.performance_stats['total_profit']:.2f}")
    
    print(f"\n❌ Simulating Trade Closure (LOSS)...")
    # Simulate closing the second trade with loss
    bot._update_closed_trade_performance(-15.75)
    
    print(f"\n📊 Final Performance Stats:")
    print(f"   📂 Total Opened Trades: {bot.performance_stats['total_trades']}")
    print(f"   ✅ Winning Trades: {bot.performance_stats['winning_trades']}")
    print(f"   ❌ Losing Trades: {bot.performance_stats['losing_trades']}")
    total_closed = bot.performance_stats['winning_trades'] + bot.performance_stats['losing_trades']
    print(f"   🔢 Total Closed Trades: {total_closed}")
    print(f"   🏆 Win Rate: {bot.performance_stats['win_rate']:.1%}")
    print(f"   💰 Total P&L: ${bot.performance_stats['total_profit']:.2f}")
    print(f"   🔥 Consecutive Losses: {bot.performance_stats['consecutive_losses']}")
    
    print(f"\n✅ Performance tracking test completed!")
    print(f"📋 Summary:")
    print(f"   - Win Rate is now calculated based on CLOSED trades only")
    print(f"   - Opening trades increments 'total_trades' but doesn't affect win rate")
    print(f"   - Win rate = winning_trades / (winning_trades + losing_trades)")
    print(f"   - Current: {bot.performance_stats['winning_trades']} wins / {total_closed} closed = {bot.performance_stats['win_rate']:.1%}")

if __name__ == "__main__":
    test_performance_tracking()
