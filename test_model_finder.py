#!/usr/bin/env python3
"""
Test script for model finder functionality
"""

import os
import sys
import glob

def find_best_model(symbol):
    """Find the best available model for a symbol"""
    print(f"🔍 Searching for {symbol} models...")
    
    # Look for models in order of preference: diamond > gold > silver > bronze > simple
    search_paths = [
        f"models/diamond/{symbol.lower()}_diamond_*.zip",
        f"models/gold/{symbol.lower()}_gold_*.zip", 
        f"models/silver/{symbol.lower()}_silver_*.zip",
        f"models/bronze/{symbol.lower()}_bronze_*.zip",
        f"models/diamond/simple_forex_model_{symbol}_PPO.zip",
        f"simple_forex_model_{symbol}_PPO.zip"
    ]
    
    best_model = None
    best_score = 0
    best_tier = ""
    
    for pattern in search_paths:
        print(f"   Checking: {pattern}")
        files = glob.glob(pattern)
        print(f"   Found {len(files)} files: {files}")
        
        for file_path in files:
            try:
                # Try to get score from filename
                if "_score" in file_path:
                    score_part = file_path.split("_score")[1].split("_")[0]
                    score = float(score_part)
                else:
                    score = 50  # Default score for simple models
                
                # Get tier from path
                if "diamond" in file_path:
                    tier = "💎 DIAMOND"
                    score += 1000  # Bonus for diamond tier
                elif "gold" in file_path:
                    tier = "🥇 GOLD"
                    score += 100
                elif "silver" in file_path:
                    tier = "🥈 SILVER"
                    score += 10
                elif "bronze" in file_path:
                    tier = "🥉 BRONZE"
                    score += 1
                else:
                    tier = "📦 SIMPLE"
                
                print(f"   -> {file_path}: {tier}, Score: {score}")
                
                if score > best_score:
                    best_model = file_path
                    best_score = score
                    best_tier = tier
                    print(f"   ✅ New best: {best_model} ({best_tier}, {best_score})")
                    
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")
                continue
    
    return best_model, best_tier, best_score

def test_all_symbols():
    """Test model finder for all symbols"""
    symbols = ['EURUSD', 'XAUUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF']
    
    print("🧪 TESTING MODEL FINDER")
    print("=" * 50)
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}:")
        model_file, tier, score = find_best_model(symbol)
        
        if model_file:
            actual_score = score-1000 if score > 1000 else score-100 if score > 100 else score-10 if score > 10 else score-1 if score > 1 else score
            print(f"✅ Found: {model_file}")
            print(f"   🏆 Tier: {tier}")
            print(f"   📊 Score: {actual_score:.1f}")
            print(f"   📁 Exists: {os.path.exists(model_file)}")
        else:
            print(f"❌ No model found for {symbol}")
    
    print("\n" + "=" * 50)
    print("🎯 MODEL FINDER TEST COMPLETE")

if __name__ == "__main__":
    test_all_symbols()
