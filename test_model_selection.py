#!/usr/bin/env python3
"""
Test script to check model selection functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_launcher import ForexLauncher

def test_model_discovery():
    """Test if we can find all EURUSD models"""
    launcher = ForexLauncher()
    
    print("🔍 Testing model discovery for EURUSD...")
    print("=" * 50)
    
    models = launcher.find_all_models("EURUSD")
    
    if not models:
        print("❌ No models found!")
        return False
    
    print(f"✅ Found {len(models)} models:")
    print()
    print(f"{'#':<3} {'Tier':<12} {'Score':<8} {'Attempt':<8} {'Date':<12} {'Filename'}")
    print("-" * 80)
    
    for i, model in enumerate(models, 1):
        actual_score = (model['final_score'] - 1000 if model['final_score'] > 1000 
                      else model['final_score'] - 100 if model['final_score'] > 100 
                      else model['final_score'] - 10 if model['final_score'] > 10 
                      else model['final_score'] - 1 if model['final_score'] > 1 
                      else model['final_score'])
        
        # Format timestamp for display
        timestamp = model['timestamp']
        if timestamp != "unknown" and len(timestamp) == 15:  # Format: YYYYMMDD_HHMMSS
            display_date = f"{timestamp[:8]}"  # Just the date part
        else:
            display_date = "unknown"
        
        print(f"{i:<3} {model['tier']:<12} {actual_score:<8.1f} {model['attempt']:<8} {display_date:<12} {model['filename'][:35]}")
    
    print()
    print("🏆 Best model:")
    best = models[0]
    print(f"   Path: {best['path']}")
    print(f"   Tier: {best['tier']}")
    print(f"   Score: {best['base_score']:.1f}")
    
    return True

if __name__ == "__main__":
    test_model_discovery()
