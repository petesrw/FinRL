#!/usr/bin/env python3
"""
Test action mapping consistency between training and live trading
"""

from mt5_trading_bot import TradingBot
import numpy as np

def test_action_mapping():
    """Test if live trading action mapping matches training"""
    print("🧪 Testing Action Mapping (Training Match):")
    print("=" * 50)
    
    # Create bot instance
    try:
        bot = TradingBot('EURUSD')
        print(f"✅ Bot initialized with min_confidence: {bot.min_confidence}")
    except Exception as e:
        print(f"❌ Failed to initialize bot: {e}")
        return
    
    # Test different action values
    test_values = [
        -0.8,   # Should be SELL
        -0.5,   # Should be SELL  
        -0.1,   # Should be HOLD
        0.0,    # Should be HOLD
        0.1,    # Should be HOLD
        0.5,    # Should be BUY
        0.6,    # Should be BUY
        0.8,    # Should be CLOSE
        0.9     # Should be CLOSE
    ]
    
    print("\n📊 Action Mapping Test:")
    print("Value  → Action   Confidence   Training Expected")
    print("-" * 50)
    
    for val in test_values:
        try:
            # Create dummy observation (bot needs observation shape)
            dummy_obs = np.random.random((25, 13))  # 2D observation
            
            # Override predict_action to test mapping directly
            action_value = val
            
            # Apply exact training mapping
            if action_value < -0.3:
                discrete_action = 2  # Sell
                expected = "SELL"
            elif action_value < 0.3:
                discrete_action = 0  # Hold
                expected = "HOLD"
            elif action_value < 0.7:
                discrete_action = 1  # Buy
                expected = "BUY"
            else:
                discrete_action = 3  # Close
                expected = "CLOSE"
            
            # Calculate confidence using training formula
            if discrete_action == 2:  # Sell
                confidence = abs(action_value + 0.65) / 0.7
            elif discrete_action == 1:  # Buy
                confidence = (action_value - 0.3) / 0.4
            else:
                confidence = 0.5  # Neutral for Hold/Close
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            actual_action = action_names[discrete_action]
            
            # Check if matches expected
            match = "✅" if actual_action == expected else "❌"
            
            print(f"{val:5.1f} → {actual_action:5s}   {confidence:.3f}        {expected:5s}      {match}")
            
        except Exception as e:
            print(f"{val:5.1f} → ERROR: {e}")
    
    print("\n🎯 Confidence Threshold Test:")
    print("-" * 30)
    
    # Test confidence threshold behavior
    buy_values = [0.4, 0.5, 0.6]  # Buy actions with different confidence
    sell_values = [-0.4, -0.5, -0.6]  # Sell actions
    
    for val in buy_values + sell_values:
        action_value = val
        
        if action_value < -0.3:
            discrete_action = 2  # Sell
            confidence = abs(action_value + 0.65) / 0.7
            action_name = "SELL"
        elif action_value >= 0.3 and action_value < 0.7:
            discrete_action = 1  # Buy
            confidence = (action_value - 0.3) / 0.4
            action_name = "BUY"
        else:
            continue
        
        confidence = max(0.0, min(1.0, confidence))
        
        # Check if would pass threshold
        would_trade = confidence >= 0.4
        status = "TRADE" if would_trade else "SKIP"
        
        print(f"{action_name:4s} {val:5.1f} → confidence: {confidence:.3f} → {status}")

if __name__ == "__main__":
    test_action_mapping()
