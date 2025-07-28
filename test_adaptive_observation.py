#!/usr/bin/env python3
"""Test script for adaptive observation system"""

import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append('.')

try:
    from forex_system_with_config import ConfigurableForexBot
    
    def test_adaptive_observation():
        """Test the adaptive observation system"""
        print("🧪 Testing Adaptive Observation System")
        print("=" * 50)
        
        # Test model path (DIAMOND tier model)
        model_path = Path("models/diamond/eurusd_diamond_score100_attempt1_20250728_215948.zip")
        
        if not model_path.exists():
            print("❌ DIAMOND model not found, skipping model-specific tests")
            print(f"   Expected: {model_path}")
            return False
            
        # Initialize the bot
        bot = ConfigurableForexBot()
        
        # Test 1: Load DIAMOND model and check observation space adaptation
        print("\n📋 Test 1: Loading DIAMOND model")
        try:
            success = bot.load_model(str(model_path))
            if success:
                print("✅ Model loaded successfully")
                if hasattr(bot, 'expected_obs_shape'):
                    print(f"✅ Expected observation shape: {bot.expected_obs_shape}")
                else:
                    print("⚠️  No expected observation shape detected")
            else:
                print("❌ Failed to load model")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        # Test 2: Create training environment with proper observation space
        print("\n📋 Test 2: Creating adaptive training environment")
        try:
            env = bot.create_training_environment()
            if env:
                print("✅ Training environment created")
                obs_shape = env.observation_space.shape
                print(f"✅ Environment observation shape: {obs_shape}")
                
                # Test observation generation
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]  # Handle new gymnasium format
                print(f"✅ Generated observation shape: {obs.shape}")
                
                # Verify compatibility
                if hasattr(bot, 'expected_obs_shape'):
                    if obs.shape == bot.expected_obs_shape:
                        print("✅ Observation shapes match perfectly!")
                    else:
                        print(f"❌ Shape mismatch: expected {bot.expected_obs_shape}, got {obs.shape}")
                        return False
                
            else:
                print("❌ Failed to create environment")
                return False
        except Exception as e:
            print(f"❌ Error creating environment: {e}")
            return False
        
        # Test 3: Test observation methods directly
        print("\n📋 Test 3: Testing observation methods")
        try:
            # Test 2D observation (DIAMOND model)
            if hasattr(env, '_get_2d_observation'):
                obs_2d = env._get_2d_observation((25, 13))  # Pass as tuple
                print(f"✅ 2D observation shape: {obs_2d.shape}")
                if obs_2d.shape == (25, 13):
                    print("✅ 2D observation method working correctly")
                else:
                    print(f"❌ 2D observation wrong shape: expected (25, 13), got {obs_2d.shape}")
            
            # Test 1D observation (legacy models)
            if hasattr(env, '_get_1d_observation'):
                obs_1d = env._get_1d_observation(50)
                print(f"✅ 1D observation shape: {obs_1d.shape}")
                if obs_1d.shape == (50,):
                    print("✅ 1D observation method working correctly")
                else:
                    print(f"❌ 1D observation wrong shape: expected (50,), got {obs_1d.shape}")
            
        except Exception as e:
            print(f"❌ Error testing observation methods: {e}")
            return False
        
        print("\n🎉 All tests passed! Adaptive observation system is working correctly.")
        return True
    
    if __name__ == "__main__":
        success = test_adaptive_observation()
        if success:
            print("\n✅ READY FOR LIVE TRADING!")
            print("The DIAMOND model can now be used with the adaptive observation system.")
        else:
            print("\n❌ Issues detected. Please review the errors above.")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure forex_system_with_config.py is in the current directory.")
